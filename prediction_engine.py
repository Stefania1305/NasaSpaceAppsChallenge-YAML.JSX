"""Prediction engine, step B: predict how each active fire evolves over the next 3 days.

Input: one pipeline run (its knowledge-graph files, fire part) and, optionally, the weather forecast for the
fires' area, provided by the caller (the frontend). The engine does not download any weather itself.
For every fire seen on the run's last day it predicts, for day +1, +2, +3:
  expansion   new burned area and total footprint (km2), movement of the fire centre (km east / north),
              spread direction (degrees, 0 = north) and spread rate (km/day), centre and radius of the fire
  fire power  mean and peak FRP of a VIIRS pass (MW)
  merging     chance it merges with a separate neighbouring fire
  extinction  chance it is out
  spread map  chance that each 1 km cell around the fire burns
with low / likely / high values (10th / 50th / 90th percentile) for every number.

Models (newest available is used):
  v3  fire chains (fires that vanish for up to 3 days and come back are one fire), extinction from a calibrated
      survival model, growth from a two-part model (chance it grows + how much), plus the sequence network:
      averaged with it for growth, fire power, movement and merging. Adds p_grow (chance it grows >= 0.1 km2)
      and growth_expected (km2) to the output.
  v2  XGBoost blended with the sequence network; extinction from XGBoost alone.

Weather forecast input (--weather file.json / .csv, or a DataFrame / list of dicts passed to predict()): hourly rows
  latitude, longitude                 point of the forecast (degrees); matched to fires within 0.5 degree
  time                                UTC hour, ISO 8601 (e.g. 2026-11-14T13:00)
  temperature_2m                      degrees C
  relative_humidity_2m                %
  wind_speed_10m, wind_gusts_10m      km/h
  wind_direction_10m                  degrees the wind comes FROM (meteorological convention)
  precipitation                       mm in the hour
  vapour_pressure_deficit             kPa
  soil_moisture_0_to_7cm, soil_moisture_7_to_28cm, soil_moisture_28_to_100cm   m3/m3 (optional)
  elevation                           m (optional)
It should cover day d (the run's last day) through d+3; a few days before d let the engine count the hours since
the last rain. Missing variables or hours are allowed (the models handle gaps). Without a forecast the no-weather
models are used; fires with no forecast point within 0.5 degree keep empty weather features.

Usage:  python prediction_engine.py --run data/runs/2026-11-01_1d_world [--weather forecast.json] [--limit 500]
Output: <run>/prediction/predictions.csv, predictions.json, spread.csv, summary.json
"""

import argparse
import json
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from firms import DATA_DIR
from prediction_data import H, KM_LAT, KM_LON, add_history, add_place_and_weather, daily_weather, feature_columns, fire_days, load_freq
from prediction_train import CONT
from weather_worker import RAIN_MM, weather_key

MODELS = DATA_DIR / "prediction" / "models"                # models trained locally (prediction_train.py)
SHIPPED = Path(__file__).resolve().parent / "models" / "v2"  # models shipped with the repository (Git LFS), used when none are trained locally
SEQ_LOCAL = DATA_DIR / "prediction" / "models_seq_full"      # sequence network trained locally on all years
REPO = Path(__file__).resolve().parent
# newest first: v4 (v3 + terrain and land cover), then v3; local models trained on all years, then shipped ones
V3_CANDIDATES = [(DATA_DIR / "prediction" / "models_v4_full", DATA_DIR / "prediction" / "models_seq_v4_full"),
                 (REPO / "models" / "v4", REPO / "models" / "v4" / "seq"),
                 (DATA_DIR / "prediction" / "models_v3_full", DATA_DIR / "prediction" / "models_seq_v3_full"),
                 (REPO / "models" / "v3", REPO / "models" / "v3" / "seq")]
BLEND_SKIP = {"out"}                                         # outputs taken from XGBoost alone


def v3_dir(variant):
    """Folder of the newest v3-style models (v4, then v3; local first, then shipped), or None."""
    for d, _ in V3_CANDIDATES:
        if (d / variant / "card.json").exists():
            return d
    return None


def v3_seq_dirs(variant):
    for d, sq in V3_CANDIDATES:
        if (d / variant / "card.json").exists():
            return (sq,)
    return ()


def load_seq(v3=False, variant="weather"):
    """The sequence network (numpy), local if trained, else shipped; None if neither exists."""
    from seq_numpy import NumpySeq
    for d in (v3_seq_dirs(variant) if v3 else (SEQ_LOCAL, SHIPPED / "seq")):
        if (d / "weights.npz").exists() and (d / "prep.json").exists():
            head = (d / "weights.npz").read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                raise SystemExit(f"{d / 'weights.npz'} is a Git LFS pointer: run 'git lfs pull' to download the models")
            return NumpySeq.load(d)
    return None
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
          "precipitation", "vapour_pressure_deficit", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]
REQUIRED = ["latitude", "longitude", "time"]


def since_rain(h: pd.DataFrame) -> pd.DataFrame:
    """hours_since_rain per cell, the same way the weather worker derives it (rain hour = at least RAIN_MM)."""
    h = h.sort_values(["weather_cell", "time"], ignore_index=True)
    wet = h.precipitation >= RAIN_MM
    block = wet.groupby(h.weather_cell).cumsum()
    h["hours_since_rain"] = h.groupby([h.weather_cell, block]).cumcount().astype(float)
    h.loc[~wet.groupby(h.weather_cell).cummax(), "hours_since_rain"] = np.nan
    return h


def read_forecast(weather) -> pd.DataFrame:
    """The caller's hourly forecast (path to .json / .csv, list of dicts or DataFrame) as a clean DataFrame."""
    if weather is None:
        return pd.DataFrame()
    if isinstance(weather, (str, Path)):
        p = Path(weather)
        if p.suffix.lower() == ".csv":
            w = pd.read_csv(p)
        else:
            d = json.loads(p.read_text(encoding="utf-8"))
            w = pd.DataFrame(d.get("hourly", d) if isinstance(d, dict) else d)
    elif isinstance(weather, pd.DataFrame):
        w = weather.copy()
    else:
        w = pd.DataFrame(weather)
    w = w.rename(columns={"lat": "latitude", "lon": "longitude", "lng": "longitude"})
    missing = [c for c in REQUIRED if c not in w]
    if missing:
        raise SystemExit(f"Weather forecast is missing columns: {', '.join(missing)}")
    for c in HOURLY + ["latitude", "longitude", "elevation"]:
        if c in w:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        elif c in HOURLY:
            w[c] = np.nan
    w["time"] = pd.to_datetime(w.time, utc=True).dt.strftime("%Y-%m-%dT%H:%M")
    return w.dropna(subset=["latitude", "longitude"])


def forecast_cells(w: pd.DataFrame, run_dir: Path):
    """Hourly forecast per 0.25 degree weather cell (points in the same cell are averaged), with hours since rain,
    and the cell table (centre, elevation from the forecast or from the run's weather cells)."""
    w = w.copy()
    w["weather_cell"] = weather_key(w.latitude, w.longitude).to_numpy()
    hours = w.groupby(["weather_cell", "time"], as_index=False)[HOURLY].mean()
    cells = w.groupby("weather_cell", as_index=False).agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
    if "elevation" in w and w.elevation.notna().any():
        cells = cells.merge(w.groupby("weather_cell").elevation.mean().rename("elevation_m"), on="weather_cell", how="left")
    else:
        kgc = run_dir / "kg" / "nodes_WeatherCell.csv.gz"
        if kgc.exists():
            c = pd.read_csv(kgc)
            c.columns = [x.split(":")[0] for x in c.columns]
            cells = cells.merge(c[["weather_cell", "elevation_m"]], on="weather_cell", how="left")
    return since_rain(hours), cells


SPREAD = DATA_DIR / "prediction" / "models" / "spread.json"
if not SPREAD.exists():
    SPREAD = SHIPPED / "spread.json"
SPREAD_MIN_P = 0.02


def write_spread(run_dir: Path, P: pd.DataFrame, last: str, out_dir: Path) -> int:
    """Burn chance per 1 km cell around each fire for days +1..+3 (prediction_spread.py), from today's pixels."""
    if not SPREAD.exists():
        return 0
    from prediction_spread import spread_cells
    coef = json.loads(SPREAD.read_text(encoding="utf-8"))["coef"]
    kg = run_dir / "kg"
    det = pd.read_csv(kg / "nodes_Detection.csv.gz", low_memory=False,
                      usecols=["id:ID(Detection)", "latitude:double", "longitude:double", "acq_date:string", "frp_mw:double",
                               "removed_by_interpreter:string"])
    det.columns = ["id", "lat", "lon", "day", "frp", "removed"]
    det = det[det.removed.isna() & (det.day == last)]
    inc = pd.read_csv(kg / "rels_IN_CLUSTER.csv.gz")
    inc.columns = ["det", "cl", "type"]
    inc["fire"] = inc.cl.str.rsplit("|", n=1).str[-1].str.split(".").str[0].astype(int)
    det = det.merge(inc[["det", "fire"]].drop_duplicates("det"), left_on="id", right_on="det")
    by_fire = {f: g for f, g in det.groupby("fire")}
    rows = []
    for r in P.itertuples():
        pix = by_fire.get(r.fire_id)
        if pix is None or pix.empty:
            continue
        w = pix.frp.fillna(0).clip(lower=0) + 0.1
        lat0, lon0 = float((pix.lat * w).sum() / w.sum()), float((pix.lon * w).sum() / w.sum())
        rk = {"fp": r.footprint_likely_km2, "hi": max(r.footprint_high_km2, r.footprint_likely_km2), "dx": r.dx_likely,
              "dy": r.dy_likely, "dir": r.direction_deg, "move": r.move_km, "out": r.p_out, "frp": r.frp_likely}
        c = spread_cells(lat0, lon0, pix.lat.to_numpy(), pix.lon.to_numpy(), pix.frp.to_numpy(), r.footprint_now_km2,
                         r.frp_now_mw if not pd.isna(r.frp_now_mw) else 0, rk, coef[str(r.day_ahead)])
        c = c[c.p >= SPREAD_MIN_P]
        rows.append(c.assign(fire_id=r.fire_id, day_ahead=r.day_ahead, date=r.date))
    if not rows:
        return 0
    S = pd.concat(rows, ignore_index=True)[["fire_id", "day_ahead", "date", "lat", "lon", "p"]].round({"lat": 5, "lon": 5, "p": 4})
    S.to_csv(out_dir / "spread.csv", index=False)
    return len(S)


def load_models(variant: str) -> dict:
    """The trained models of one variant: local ones if present, else the v2 models shipped in models/v2."""
    models = {}
    for p in (MODELS / variant).glob("*.json"):
        b = xgb.Booster()
        b.load_model(str(p))
        models[p.stem] = b
    if not models:
        import gzip
        for p in (SHIPPED / variant).glob("*.ubj.gz"):
            data = p.read_bytes()
            if data.startswith(b"version https://git-lfs"):
                raise SystemExit(f"{p} is a Git LFS pointer: run 'git lfs pull' to download the models")
            raw = gzip.decompress(data)
            b = xgb.Booster()
            b.load_model(bytearray(raw))
            models[p.name[:-len(".ubj.gz")]] = b
    return models


def predict(run_dir: Path, weather=None, limit: int = 0, weather_label: str = "") -> dict:
    """weather: the caller's hourly forecast (path, list of dicts or DataFrame), or None for the no-weather models."""
    t0 = time.time()
    fd = fire_days(run_dir, with_targets=False)
    if fd.empty:
        raise SystemExit("No fires in this run.")
    variant0 = "weather" if weather is not None else "no_weather"
    use_v3 = v3_dir(variant0) is not None
    if use_v3:
        from prediction_labels import relabel
        fd = relabel(fd, with_targets=False)          # link fires that vanished and came back, as in training
    fd = add_history(fd)
    last = fd.day.max()
    active = fd[fd.day == last].copy()
    act_pos = np.flatnonzero((fd.day == last).to_numpy())
    if limit:
        keep = np.argsort(-active.fine_fp.to_numpy(), kind="stable")[:limit]
        active, act_pos = active.iloc[keep], act_pos[keep]
    # weather forecast provided by the caller
    mode, wd, cells, coverage = "none", None, None, {}
    w = read_forecast(weather)
    if len(w):
        hours, cells = forecast_cells(w, run_dir)
        wd = daily_weather(hours)
        mode = "provided"
        days = [(pd.Timestamp(last) + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(0, H + 1)]
        coverage = {f"day+{k}": bool((wd.day == d).any()) for k, d in enumerate(days)}
    X = add_place_and_weather(active, run_dir, wd, cells, load_freq())
    if use_v3 and (v3_dir(variant0) / variant0 / "card.json").exists() and \
            json.loads((v3_dir(variant0) / variant0 / "card.json").read_text(encoding="utf-8")).get("static_features"):
        from prediction_data import add_static
        X = add_static(X, sample_missing=True)         # reads new places from the cloud files, then caches them
    if mode == "provided":
        # as in training for real use: days +1..+3 take day d's soil moisture
        for k in range(1, H + 1):
            for c in ("sm0", "sm1", "sm2"):
                X[f"w{k}_{c}"] = X[f"w0_{c}"]
        coverage["fires_with_forecast"] = int(X.w1_tmax.notna().sum())
    variant = "weather" if mode != "none" else "no_weather"
    use_v3 = v3_dir(variant) is not None
    if use_v3:
        import prediction_v3
        Pd = prediction_v3.predict(v3_dir(variant), X, variant)
        model_name = ("v4: v3 + terrain and land cover" if "v4" in str(v3_dir(variant)) else "v3") + " (survival + two-part growth)"
    else:
        models = load_models(variant)
        if not models:
            raise SystemExit(f"No trained models in {MODELS / variant}. Run prediction_train.py first.")
        # each model knows the features it was trained on; older models simply ignore newer columns
        feats = next(iter(models.values())).feature_names or feature_columns(X, weather=(variant == "weather"))
        M = xgb.DMatrix(X.reindex(columns=feats).astype("float32"))
        Pd = {}
        for k in range(1, H + 1):
            for target, (fwd, inv) in CONT.items():
                if f"{target}_{k}" in models:
                    Pd[f"{target}_{k}"] = inv(np.sort(models[f"{target}_{k}"].predict(M), axis=1).astype("float64"))
            for target in ("out", "merge"):
                m = models.get(f"{target}_{k}")
                Pd[f"{target}_{k}"] = m.predict(M).astype("float64") if m is not None else np.full(len(X), np.nan)
        model_name = "v2"
    seq = load_seq(v3=use_v3, variant=variant)
    S = None
    if seq is not None:
        # the network reads each fire's previous fire-days, so its history comes from the whole run
        cols = list(dict.fromkeys(seq.prep.feats + ["run", "fire_id", "day", "chain"]))
        hcols = list(dict.fromkeys(["run", "fire_id", "day", "chain"] + __import__("seq_numpy").DYN))
        hist = seq.prep.history(fd.reindex(columns=hcols))[act_pos]
        S = seq.predict_df(X.reindex(columns=cols), weather=(variant == "weather"), history=hist)
        for key in list(Pd):
            if key in S and key.split("_")[0] not in BLEND_SKIP:
                v = (Pd[key] + S[key]) / 2
                Pd[key] = np.sort(v, axis=1) if v.ndim == 2 else v
        model_name += " + sequence network"
    out = []
    for k in range(1, H + 1):
        r = X[["run", "fire_id", "day", "lat0", "lon0", "clat", "clon", "fine_fp", "frp"]].copy()
        r["day_ahead"] = k
        r["date"] = (pd.to_datetime(r.day) + timedelta(days=k)).dt.strftime("%Y-%m-%d")
        for target in CONT:
            if f"{target}_{k}" in Pd:
                for j, name in enumerate(("low", "likely", "high")):
                    r[f"{target}_{name}"] = Pd[f"{target}_{k}"][:, j]
        r["p_out"], r["p_merge"] = Pd[f"out_{k}"], Pd[f"merge_{k}"]
        if f"grow_{k}" in Pd:
            r["p_grow"] = Pd[f"grow_{k}"]
            r["growth_expected"] = Pd[f"growth_mean_{k}"]
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
            "frp_peak_low", "frp_peak_likely", "frp_peak_high", "p_out", "p_merge", "p_grow", "growth_expected"]
    P = P[[c for c in keep if c in P]].round(4)
    out_dir = run_dir / "prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(out_dir / "predictions.csv", index=False)
    n_cells = write_spread(run_dir, P, last, out_dir)
    (out_dir / "predictions.json").write_text(P.to_json(orient="records"), encoding="utf-8")
    summary = {"run": run_dir.name, "from_day": last, "active_fires": int(len(active)), "rows": len(P), "spread_cells": n_cells,
               "weather": mode, "weather_coverage": coverage, "models": f"{variant}: {model_name}", "seconds": round(time.time() - t0, 1),
               "note": weather_label or {"provided": "weather forecast provided by the caller", "none": "no forecast provided: no-weather models"}[mode]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"predictions": P, "summary": summary, "out": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--weather", help="hourly weather forecast (.json or .csv), see the format above; omit for no-weather models")
    ap.add_argument("--limit", type=int, default=0, help="only the N largest active fires")
    ap.add_argument("--weather-label", default="", help="where the forecast came from (shown in summary.json and on the map)")
    args = ap.parse_args()
    res = predict(Path(args.run), args.weather, args.limit, args.weather_label)
    s, P = res["summary"], res["predictions"]
    print(json.dumps(s, indent=1))
    top = P[P.day_ahead == H].sort_values("footprint_likely_km2", ascending=False).head(8)
    print(top[["fire_id", "lat0", "lon0", "footprint_now_km2", "footprint_likely_km2", "footprint_high_km2", "direction_deg",
               "spread_rate_km_day", "frp_likely", "p_out", "p_merge"]].to_string(index=False))
    print(f"\nSaved to {res['out']}")


if __name__ == "__main__":
    main()
