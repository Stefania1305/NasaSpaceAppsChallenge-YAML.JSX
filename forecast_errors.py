"""How wrong the weather forecasts were: a pool of real forecast errors, used to train on forecast-like weather.

The models learned from the weather that actually happened (ERA5), but in real use they only get forecasts.
This script takes fire-days from March 2024 on (when archived forecasts exist), fetches the forecasts that were
issued for days d+1..d+3 (Open-Meteo previous-runs API, ECMWF IFS 0.25), turns them into the same daily weather
features, and stores forecast minus observed for every fire-day and day ahead: one error vector per row.

Training then adds a randomly drawn error vector (same day ahead, same region group when there are enough) to the
observed weather, so the models see weather as uncertain as a real forecast. Drawing whole vectors keeps the
errors of related variables together (a forecast too warm is usually also too dry).

Usage:  python forecast_errors.py [--cells 200]
Output: data/prediction/forecast_residuals.csv.gz, forecast_residuals_summary.json
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import requests

from firms import DATA_DIR
from prediction_data import H
from weather_worker import RAIN_MM, THROTTLE, USER_AGENT, cell_latlon

OUT = DATA_DIR / "prediction"
VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "precipitation", "vapour_pressure_deficit"]
NOISY = ["tmax", "tmean", "rh_min", "rh_mean", "vpd_max", "wind_mean", "wind_max", "gust_max", "u_mean", "v_mean", "rain"]
GROUPS = {"Americas": [0, 1, 2, 11, 12, 13], "Africa": [14, 15], "Asia and Australia": [3, 5, 6, 7, 8, 9, 10], "Europe and other": [4, 16]}
WINDOW_DAYS = 10
PER_CELL = 12


def group_of(region):
    for g, ids in GROUPS.items():
        if region in ids:
            return g
    return "Europe and other"


def fetch(cell, lat, lon, start, end):
    names = [f"{v}_previous_day{k}" for k in range(1, H + 1) for v in VARS]
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    THROTTLE.wait(math.ceil(len(names) / 10) * math.ceil(span / 14))
    for attempt in range(5):
        try:
            r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", headers={"User-Agent": USER_AGENT}, timeout=120,
                             params={"latitude": lat, "longitude": lon, "hourly": ",".join(names), "models": "ecmwf_ifs025",
                                     "start_date": start, "end_date": end, "timezone": "GMT"})
            if r.status_code == 200:
                h = pd.DataFrame(r.json()["hourly"])
                h["weather_cell"] = cell
                return h
            if r.status_code not in (429, 500, 502, 503, 504):
                print(f"  {cell}: {r.status_code} {r.text[:120]}", flush=True)
                return None
        except requests.exceptions.RequestException:
            pass
        time.sleep(10 * (attempt + 1))
    return None


def daily(hours: pd.DataFrame, k: int) -> pd.DataFrame:
    p = f"_previous_day{k}"
    h = hours[["weather_cell", "time"] + [v + p for v in VARS]].rename(columns={v + p: v for v in VARS})
    for v in VARS:
        h[v] = pd.to_numeric(h[v], errors="coerce")
    h["day"] = h.time.str[:10]
    rad = np.radians(h.wind_direction_10m)
    h["u"], h["v"] = -h.wind_speed_10m * np.sin(rad), -h.wind_speed_10m * np.cos(rad)
    h["wet"] = h.precipitation >= RAIN_MM
    g = h.groupby(["weather_cell", "day"])
    return pd.DataFrame({"tmax": g.temperature_2m.max(), "tmean": g.temperature_2m.mean(), "rh_min": g.relative_humidity_2m.min(),
                         "rh_mean": g.relative_humidity_2m.mean(), "vpd_max": g.vapour_pressure_deficit.max(),
                         "wind_mean": g.wind_speed_10m.mean(), "wind_max": g.wind_speed_10m.max(), "gust_max": g.wind_gusts_10m.max(),
                         "u_mean": g.u.mean(), "v_mean": g.v.mean(), "rain": g.precipitation.sum(min_count=1)}).reset_index()


def forecast_like(df: pd.DataFrame, rows=None, seed: int = 0) -> pd.DataFrame:
    """Copy of the fire-day table whose weather for days +1..+3 looks like a forecast: soil moisture from day d
    (as the engine has it) and, for the chosen rows, one real forecast error vector added (same day ahead and
    region group when there are enough errors)."""
    R = pd.read_csv(OUT / "forecast_residuals.csv.gz")
    rng = np.random.default_rng(seed)
    d = df.copy()
    for k in range(1, H + 1):
        for c in ("sm0", "sm1", "sm2"):
            if f"w{k}_{c}" in d and f"w0_{c}" in d:
                d[f"w{k}_{c}"] = d[f"w0_{c}"]
    rows = np.ones(len(d), bool) if rows is None else np.asarray(rows)
    grp = d.region.map(group_of).to_numpy()
    for k in range(1, H + 1):
        draw = np.zeros((len(d), len(NOISY)))
        for g in GROUPS:
            m = rows & (grp == g)
            pool = R[(R.k == k) & (R.group == g)][NOISY].fillna(0).to_numpy()
            if len(pool) < 50:
                pool = R[R.k == k][NOISY].fillna(0).to_numpy()
            draw[m] = pool[rng.integers(len(pool), size=int(m.sum()))]
        for j, c in enumerate(NOISY):
            col = f"w{k}_{c}"
            if col not in d:
                continue
            v = d[col].to_numpy("float64") + draw[:, j]
            if c.startswith("rh"):
                v = np.clip(v, 1, 100)
            elif c in ("rain", "wind_mean", "wind_max", "gust_max", "vpd_max"):
                v = np.clip(v, 0, None)
            d[col] = np.where(np.isnan(d[col].to_numpy("float64")), np.nan, v)
        if f"w{k}_wind_to_nb" in d and "nb_bearing" in d:
            b = np.radians(d.nb_bearing.to_numpy())
            d[f"w{k}_wind_to_nb"] = d[f"w{k}_u_mean"].to_numpy() * np.sin(b) + d[f"w{k}_v_mean"].to_numpy() * np.cos(b)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    cols = ["run", "fire_id", "day", "region", "weather_cell"] + [f"w{k}_{c}" for k in range(1, H + 1) for c in NOISY]
    df = pd.read_csv(OUT / "fire_days.csv.gz", usecols=cols, low_memory=False)
    pool = df[(df.day >= "2024-03-05") & df.w1_tmax.notna() & df.weather_cell.notna()].copy()
    pool["group"] = pool.region.map(group_of)
    # cells spread over the region groups (in proportion, at least 15 each), one random 10-day window per cell
    rng = np.random.default_rng(0)
    cell_group = pool.groupby("weather_cell").group.first()
    share = cell_group.value_counts(normalize=True)
    want = {g: max(15, int(round(args.cells * s))) for g, s in share.items()}
    chosen = []
    for g, n in want.items():
        cs = cell_group[cell_group == g].index.to_numpy()
        chosen += list(rng.choice(cs, size=min(n, len(cs)), replace=False))
    rows = []
    for c in chosen:
        s = pool[pool.weather_cell == c].sort_values("day")
        d0 = pd.Timestamp(rng.choice(s.day.to_numpy()))
        s = s[(pd.to_datetime(s.day) >= d0 - pd.Timedelta(days=WINDOW_DAYS // 2)) & (pd.to_datetime(s.day) <= d0 + pd.Timedelta(days=WINDOW_DAYS // 2))]
        rows.append(s.sample(min(len(s), PER_CELL), random_state=0))
    sample = pd.concat(rows, ignore_index=True)
    print(f"{len(sample)} fire-days in {len(chosen)} weather cells: " + ", ".join(f"{g} {n}" for g, n in sample.group.value_counts().items()), flush=True)
    ll = cell_latlon(pd.Series(chosen)).set_index("weather_cell")
    hours = []
    for i, c in enumerate(chosen):
        s = sample[sample.weather_cell == c]
        start = (pd.Timestamp(s.day.min()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(s.day.max()) + pd.Timedelta(days=H)).strftime("%Y-%m-%d")
        h = fetch(c, ll.lat[c], ll.lon[c], start, end)
        if h is not None:
            hours.append(h)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chosen)} cells fetched ({time.time() - t0:.0f} s)", flush=True)
    Hh = pd.concat(hours, ignore_index=True)
    res = []
    for k in range(1, H + 1):
        d = daily(Hh, k)
        dk = (pd.to_datetime(sample.day) + pd.Timedelta(days=k)).dt.strftime("%Y-%m-%d")
        m = pd.DataFrame({"weather_cell": sample.weather_cell, "day": dk}).merge(d, on=["weather_cell", "day"], how="left")
        r = pd.DataFrame({"k": k, "group": sample.group, "run": sample.run, "fire_id": sample.fire_id, "from_day": sample.day})
        for c in NOISY:
            r[c] = m[c].to_numpy() - sample[f"w{k}_{c}"].to_numpy()
        res.append(r[m.tmax.notna().to_numpy() & sample[f"w{k}_tmax"].notna().to_numpy()])
    R = pd.concat(res, ignore_index=True)
    R.to_csv(OUT / "forecast_residuals.csv.gz", index=False)
    summ = {"error_vectors": len(R), "fire_days": len(sample), "cells": len(chosen), "seconds": round(time.time() - t0),
            "by_group": {g: int(n) for g, n in R.group.value_counts().items()},
            "mean_abs_error": {f"day+{k}": {c: round(float(R[R.k == k][c].abs().mean()), 3) for c in NOISY} for k in range(1, H + 1)},
            "bias": {f"day+{k}": {c: round(float(R[R.k == k][c].mean()), 3) for c in NOISY} for k in range(1, H + 1)}}
    (OUT / "forecast_residuals_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ["mean_abs_error"], indent=1))
    print(f"{len(R)} error vectors -> {OUT / 'forecast_residuals.csv.gz'}")


if __name__ == "__main__":
    main()
