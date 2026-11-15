"""Prediction engine, step B: predict how each active fire evolves over the next 3 days.

Input: one pipeline run (its knowledge-graph files, fire part) and the weather forecast for the next 3 days.
For every fire seen on the run's last day it predicts, for day +1, +2, +3:
  expansion   new burned area and total footprint (km2), movement of the fire centre (km east / north),
              spread direction (degrees, 0 = north) and spread rate (km/day), centre and radius of the fire
  fire power  mean and peak FRP of a VIIRS pass (MW)
  merging     chance it merges with a separate neighbouring fire
  extinction  chance it is out
with low / likely / high values (10th / 50th / 90th percentile) for every number.

Weather forecast for days d+1..d+3 (d = the run's last day), per fire's 0.25 degree cell:
  live      the run ends today or yesterday: Open-Meteo forecast API (ECMWF IFS 0.25), next 4 days
  archive   an older run (from March 2024): the forecasts that were issued 1, 2 and 3 days before each day
            (Open-Meteo previous-runs API), so a past run is predicted with the forecast available back then;
            day d itself and soil moisture come from ERA5 (observed)
  observed  runs before March 2024: ERA5 weather that actually happened (labelled as such)
  none      --no-weather, or no weather reachable: the no-weather models are used

Usage:  python prediction_engine.py --run data/runs/2026-11-01_1d_world [--no-weather] [--limit 500]
Output: <run>/prediction/predictions.csv, predictions.json, summary.json
"""

import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from firms import DATA_DIR
from prediction_data import H, KM_LAT, KM_LON, add_place_and_weather, daily_weather, feature_columns, fire_days, load_freq
from prediction_train import CONT
from weather_worker import RAIN_MM, THROTTLE, USER_AGENT, cell_latlon, weather_key

MODELS = DATA_DIR / "prediction" / "models"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
          "precipitation", "vapour_pressure_deficit", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]
FORECAST_FROM = "2024-03-01"
LEAD_DAYS = 14
BATCH = 50


def get(url, params):
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=180)
            if r.status_code == 200:
                d = r.json()
                return d if isinstance(d, list) else [d]
            if r.status_code not in (429, 500, 502, 503, 504):
                raise SystemExit(f"Open-Meteo {r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(10 * (attempt + 1))
    raise SystemExit("Open-Meteo did not answer")


def hourly_frames(url, cells, params, names, rename=None):
    out, elev = [], {}
    for k in range(0, len(cells), BATCH):
        b = cells.iloc[k:k + BATCH]
        THROTTLE.wait(len(b) * math.ceil(len(names) / 10))
        res = get(url, {"latitude": ",".join(f"{x:.2f}" for x in b.lat), "longitude": ",".join(f"{x:.2f}" for x in b.lon),
                        "hourly": ",".join(names), "timezone": "GMT", **params})
        for cell, d in zip(b.weather_cell, res):
            h = pd.DataFrame(d["hourly"])
            h.insert(0, "weather_cell", cell)
            out.append(h)
            elev[cell] = d.get("elevation")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if rename and len(df):
        df = df.rename(columns=rename)
    for c in df.columns.drop(["weather_cell", "time"], errors="ignore"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, elev


def since_rain(h: pd.DataFrame) -> pd.DataFrame:
    """hours_since_rain per cell, the same way the weather worker derives it (rain hour = at least RAIN_MM)."""
    h = h.sort_values(["weather_cell", "time"], ignore_index=True)
    wet = h.precipitation >= RAIN_MM
    block = wet.groupby(h.weather_cell).cumsum()
    h["hours_since_rain"] = h.groupby([h.weather_cell, block]).cumcount().astype(float)
    h.loc[~wet.groupby(h.weather_cell).cummax(), "hours_since_rain"] = np.nan
    return h


def forecast_weather(cells: pd.DataFrame, d: str, mode: str):
    """Hourly weather for days d..d+3 at the cells, as the forecast would have given it on day d
    (plus LEAD_DAYS before d, observed, so hours since rain is known)."""
    start = (pd.Timestamp(d) - timedelta(days=LEAD_DAYS)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(d) + timedelta(days=H)).strftime("%Y-%m-%d")
    if mode == "live":
        h, elev = hourly_frames(FORECAST_URL, cells, {"models": "ecmwf_ifs025", "past_days": LEAD_DAYS, "forecast_days": H + 1}, HOURLY)
        return (since_rain(h) if len(h) else h), elev
    stop = end if mode == "observed" else d
    obs, elev = hourly_frames(ARCHIVE_URL, cells, {"models": "era5", "start_date": start, "end_date": stop}, HOURLY)
    if mode == "observed" or not len(obs):
        return (since_rain(obs) if len(obs) else obs), elev
    keep_soil = obs[obs.time.str[:10] == d].groupby("weather_cell")[HOURLY[-3:]].mean()
    parts = [obs]
    for k in range(1, H + 1):
        dk = (pd.Timestamp(d) + timedelta(days=k)).strftime("%Y-%m-%d")
        names = [f"{v}_previous_day{k}" for v in HOURLY[:-3]]
        f, _ = hourly_frames(PREVIOUS_RUNS_URL, cells, {"models": "ecmwf_ifs025", "start_date": dk, "end_date": dk}, names,
                             {f"{v}_previous_day{k}": v for v in HOURLY[:-3]})
        if len(f):
            parts.append(f.join(keep_soil, on="weather_cell"))   # soil moisture is not in the forecast archive: day d's value
    return since_rain(pd.concat(parts, ignore_index=True)), elev


def load_models(variant: str) -> dict:
    d = MODELS / variant
    models = {}
    for p in d.glob("*.json"):
        b = xgb.Booster()
        b.load_model(str(p))
        models[p.stem] = b
    return models


def predict(run_dir: Path, use_weather: bool, limit: int) -> dict:
    t0 = time.time()
    fd = fire_days(run_dir, with_targets=False)
    if fd.empty:
        raise SystemExit("No fires in this run.")
    last = fd.day.max()
    active = fd[fd.day == last].copy()
    if limit:
        active = active.sort_values("fine_fp", ascending=False).head(limit)
    # weather forecast for the active fires' cells
    mode = "none"
    wd = cells = None
    if use_weather:
        mode = "live" if last >= (date.today() - timedelta(days=1)).isoformat() else "archive" if last >= FORECAST_FROM else "observed"
        ck = sorted(set(weather_key(active.lat0, active.lon0)))
        cells = cell_latlon(pd.Series(ck))
        hours, elev = forecast_weather(cells, last, mode)
        if len(hours):
            cells["elevation_m"] = cells.weather_cell.map(elev)
            wd = daily_weather(hours)
        else:
            mode = "none"
    X = add_place_and_weather(active, run_dir, wd, cells, load_freq())
    variant = "weather" if mode != "none" else "no_weather"
    models = load_models(variant)
    if not models:
        raise SystemExit(f"No trained models in {MODELS / variant}. Run prediction_train.py first.")
    feats = feature_columns(X, weather=(variant == "weather"))
    M = xgb.DMatrix(X[feats].astype("float32"))
    out = []
    for k in range(1, H + 1):
        r = X[["run", "fire_id", "day", "lat0", "lon0", "clat", "clon", "fine_fp", "frp"]].copy()
        r["day_ahead"] = k
        r["date"] = (pd.to_datetime(r.day) + timedelta(days=k)).dt.strftime("%Y-%m-%d")
        for target, (fwd, inv) in CONT.items():
            m = models.get(f"{target}_{k}")
            if m is None:
                continue
            q = inv(np.sort(m.predict(M), axis=1).astype("float64"))
            for j, name in enumerate(("low", "likely", "high")):
                r[f"{target}_{name}"] = q[:, j]
        for target in ("out", "merge"):
            m = models.get(f"{target}_{k}")
            r[f"p_{target}"] = m.predict(M).astype("float64") if m is not None else np.nan
        out.append(r)
    P = pd.concat(out, ignore_index=True).sort_values(["fire_id", "day_ahead"], ignore_index=True)
    # consistency: growth >= 0 and never shrinking over the days; chances never falling over the days
    for name in ("low", "likely", "high"):
        P[f"growth_{name}"] = P[f"growth_{name}"].clip(lower=0)
        P[f"growth_{name}"] = P.groupby("fire_id")[f"growth_{name}"].cummax()
        P[f"footprint_{name}_km2"] = P.fine_fp + P[f"growth_{name}"]
    for c in ("p_out", "p_merge"):
        P[c] = P.groupby("fire_id")[c].cummax().clip(0, 1)
    # direction, spread rate, predicted centre and radius from the likely movement and footprint
    P["direction_deg"] = (np.degrees(np.arctan2(P.dx_likely, P.dy_likely)) % 360).round(1)
    P["move_km"] = np.hypot(P.dx_likely, P.dy_likely).round(3)
    P["spread_rate_km_day"] = (P.move_km / P.day_ahead).round(3)
    base_lat, base_lon = P.clat.fillna(P.lat0), P.clon.fillna(P.lon0)
    P["centre_lat"] = (base_lat + P.dy_likely / KM_LAT).round(5)
    P["centre_lon"] = (base_lon + P.dx_likely / (KM_LON * np.cos(np.radians(base_lat)))).round(5)
    P["radius_km"] = np.sqrt(P.footprint_likely_km2 / math.pi).round(3)
    P = P.rename(columns={"day": "from_day", "fine_fp": "footprint_now_km2", "frp": "frp_now_mw"})
    keep = ["run", "fire_id", "from_day", "day_ahead", "date", "lat0", "lon0", "footprint_now_km2", "frp_now_mw",
            "footprint_low_km2", "footprint_likely_km2", "footprint_high_km2", "growth_low", "growth_likely", "growth_high",
            "direction_deg", "spread_rate_km_day", "move_km", "dx_low", "dx_likely", "dx_high", "dy_low", "dy_likely", "dy_high",
            "centre_lat", "centre_lon", "radius_km", "frp_low", "frp_likely", "frp_high",
            "frp_peak_low", "frp_peak_likely", "frp_peak_high", "p_out", "p_merge"]
    P = P[[c for c in keep if c in P]].round(4)
    out_dir = run_dir / "prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "predictions.json").write_text(P.to_json(orient="records"), encoding="utf-8")
    summary = {"run": run_dir.name, "from_day": last, "active_fires": int(len(active)), "rows": len(P),
               "weather": mode, "models": variant, "seconds": round(time.time() - t0, 1),
               "note": {"live": "forecast issued now", "archive": "forecasts issued 1-3 days before each day (as available then)",
                        "observed": "ERA5 weather that actually happened (no archived forecast before March 2024)",
                        "none": "no weather: no-weather models"}[mode]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"predictions": P, "summary": summary, "out": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the N largest active fires")
    args = ap.parse_args()
    res = predict(Path(args.run), not args.no_weather, args.limit)
    s, P = res["summary"], res["predictions"]
    print(json.dumps(s, indent=1))
    top = P[P.day_ahead == H].sort_values("footprint_likely_km2", ascending=False).head(8)
    print(top[["fire_id", "lat0", "lon0", "footprint_now_km2", "footprint_likely_km2", "footprint_high_km2", "direction_deg",
               "spread_rate_km_day", "frp_likely", "p_out", "p_merge"]].to_string(index=False))
    print(f"\nSaved to {res['out']}")


if __name__ == "__main__":
    main()
