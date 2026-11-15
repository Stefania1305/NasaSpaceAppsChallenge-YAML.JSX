"""Backtest of the app's fire prediction against what really happened.

For each test zone and each "prediction day" d, the app's backend (server on localhost:8000) is used exactly as
the frontend uses it: the pipeline runs on the 5 days ending on d (correlation -> interpreter -> localisation +
fire power -> epicenter -> fire expansion -> knowledge graph -> prediction, with the Open-Meteo weather forecast),
then the predictions for d+1..d+3 are compared with the real harmonised FIRMS detections of those days
(/api/harmonise), which the model never saw.

Scores (k = days ahead)
  extinction    the model's chance a fire is out by d+k vs whether real detections show it again (from d+k to the
                last day checked): Brier (lower is better), AUC (0.5 = guessing), accuracy at 50%; baselines:
                "every fire keeps burning" and the base rate
  spread map    each 1 km cell's burn chance vs whether a real detection fell in that cell on d+k: AUC, Brier,
                calibration (mean predicted vs observed share, in bins)
  movement      for fires seen again: distance from the predicted centre to the real centre (FRP-weighted) of the
                day's detections, vs the persistence baseline (the fire stays where it is)
  fire power    real mean VIIRS-pass FRP inside the predicted low..high range (the model aims at ~80%), and the
                error of the likely value
  coverage      real detections on d+k far from every predicted fire (new ignitions, which the model does not
                forecast)

Usage:  python backtest/backtest.py            (writes backtest/results.json and backtest/report.md)
"""

import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

API = "http://localhost:8000"
OUT = Path(__file__).resolve().parent
KM_LAT, KM_LON = 110.574, 111.320
STAGES = ["correlation", "interpreter", "localisation", "fire_power", "epicenter", "expansion", "kg", "prediction"]

ZONES = {"Brazil (Mato Grosso / Bolivia border)": "-62,-16,-56,-10",
         "Central Africa (CAR / South Sudan savanna)": "22,7,25,9.5"}
TESTS = [  # (name, last observed day d, days of truth available)
    ("today", date(2026, 11, 14), 1),         # predict 15 Nov (today) from 10-14 Nov
    ("3-day", date(2026, 11, 11), 3),         # predict 12-14 Nov from 7-11 Nov
]


def get(path, **q):
    url = f"{API}{path}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=1800) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(f"{path} {q.get('stage', '')}: {d['error']}")
    return d


def km_dist(lat1, lon1, lat2, lon2):
    """Equirectangular distance in km (fine at these scales); broadcasts over numpy arrays."""
    kx = KM_LON * np.cos(np.radians((np.asarray(lat1) + np.asarray(lat2)) / 2))
    return np.hypot((np.asarray(lon2) - np.asarray(lon1)) * kx, (np.asarray(lat2) - np.asarray(lat1)) * KM_LAT)


def auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    pos, neg = p[y], p[~y]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allp = np.concatenate([pos, neg])                      # average ranks for ties
    for v in np.unique(allp):
        m = allp == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def run_pipeline(area, d):
    start = d - timedelta(days=4)
    q = dict(date=start.isoformat(), days=5, area=area, predict_all=1)   # the model's output even for past days
    times = {}
    for st in STAGES:
        r = get("/api/pipeline/stage", stage=st, **q)
        times[st] = r.get("seconds")
        if st == "correlation" and r.get("empty"):
            return None, times
    return get("/api/pipeline/results", **q), times


def truth(area, d, ndays):
    """Real harmonised detections of d+1..d+ndays as an array table."""
    h = get("/api/harmonise", date=(d + timedelta(days=1)).isoformat(), days=ndays, area=area)
    c = h["columns"]
    ix = {k: c.index(k) for k in ("latitude", "longitude", "acq_date", "frp_mw", "instrument", "acq_time", "satellite")}
    rows = [r for r in h["rows"] if r[ix["latitude"]] not in ("", None)]
    t = {
        "lat": np.array([float(r[ix["latitude"]]) for r in rows]),
        "lon": np.array([float(r[ix["longitude"]]) for r in rows]),
        "day": np.array([r[ix["acq_date"]] for r in rows]),
        "frp": np.array([float(r[ix["frp_mw"]]) if r[ix["frp_mw"]] not in ("", None) else np.nan for r in rows]),
        "viirs": np.array([str(r[ix["instrument"]]).upper().startswith("VIIRS") for r in rows]),
        "pass": np.array([f"{r[ix['satellite']]}|{r[ix['acq_date']]}|{str(r[ix['acq_time']])[:2]}" for r in rows]),
    }
    return t, h.get("range")


def score(res, t, d, ndays):
    preds, spread = res["predictions"], res["spread"]
    days = [(d + timedelta(days=k)).isoformat() for k in range(1, ndays + 1)]
    out = {"active_fires": len({p["fire_id"] for p in preds}), "truth_detections": {dy: int((t["day"] == dy).sum()) for dy in days},
           "model": (res.get("prediction") or {}).get("models"), "weather": (res.get("prediction") or {}).get("weather"), "per_day": {}}
    by_fire = {}
    for p in preds:
        by_fire.setdefault(p["fire_id"], {})[p["day_ahead"]] = p
    # is each fire seen on each truth day? matched detections within its reach
    seen, matched = {}, {}
    used = {dy: np.zeros(int((t["day"] == dy).sum()), bool) for dy in days}
    for fid, rows in by_fire.items():
        for k, dy in enumerate(days, 1):
            r = rows.get(k)
            if r is None:
                continue
            m = t["day"] == dy
            reach = max(2.0, math.sqrt(max(r["footprint_high_km2"] or 0, 0) / math.pi) + 1.5)
            d1 = km_dist(r["centre_lat"], r["centre_lon"], t["lat"][m], t["lon"][m])
            d0 = km_dist(r["lat0"], r["lon0"], t["lat"][m], t["lon"][m])
            hit = (d1 <= reach) | (d0 <= reach)
            seen[(fid, k)] = bool(hit.any())
            matched[(fid, k)] = np.flatnonzero(m)[hit]
            used[dy] |= hit
    for k, dy in enumerate(days, 1):
        rows = [r[k] for r in by_fire.values() if k in r]
        if not rows:
            continue
        # extinction: out by d+k = never seen again from d+k to the last checked day
        y = [all(not seen.get((r["fire_id"], j), False) for j in range(k, ndays + 1)) for r in rows]
        p = [r["p_out"] for r in rows]
        base = float(np.mean(y))
        ext = {"fires": len(rows), "really_out": int(sum(y)), "brier": brier(y, p), "auc": auc(y, p),
               "accuracy_at_50": float(np.mean([(pi >= .5) == yi for pi, yi in zip(p, y)])),
               "baseline_all_burn_accuracy": float(np.mean([not v for v in y])),
               "baseline_base_rate_brier": brier(y, [base] * len(y)),
               "mean_predicted": float(np.mean(p))}
        # spread map: 1 km cells; burned if a real detection lies within the cell (0.5 km of its centre on each axis)
        cells = [c for c in spread if c["day_ahead"] == k]
        sp = None
        if cells:
            m = t["day"] == dy
            tl, tn = t["lat"][m], t["lon"][m]
            yb = []
            for c in cells:
                dy_km = np.abs(tl - c["lat"]) * KM_LAT
                dx_km = np.abs(tn - c["lon"]) * KM_LON * math.cos(math.radians(c["lat"]))
                yb.append(bool(((dy_km <= .5) & (dx_km <= .5)).any()))
            pc = np.array([c["p"] for c in cells]); yb = np.array(yb)
            bins = [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.01)]
            sp = {"cells": len(cells), "burned": int(yb.sum()), "auc": auc(yb, pc), "brier": brier(yb, pc),
                  "calibration": [{"bin": f"{a:.2f}-{b:.2f}", "cells": int(((pc >= a) & (pc < b)).sum()),
                                   "mean_predicted": float(pc[(pc >= a) & (pc < b)].mean()) if ((pc >= a) & (pc < b)).any() else None,
                                   "observed_share": float(yb[(pc >= a) & (pc < b)].mean()) if ((pc >= a) & (pc < b)).any() else None}
                                  for a, b in bins]}
        # movement and fire power, for fires seen again that day
        mv, mv0, frp_in, frp_err = [], [], [], []
        for r in rows:
            ix = matched.get((r["fire_id"], k))
            if ix is None or not len(ix):
                continue
            w = np.nan_to_num(t["frp"][ix], nan=0) + .1
            alat, alon = float((t["lat"][ix] * w).sum() / w.sum()), float((t["lon"][ix] * w).sum() / w.sum())
            mv.append(float(km_dist(r["centre_lat"], r["centre_lon"], alat, alon)))
            mv0.append(float(km_dist(r["lat0"], r["lon0"], alat, alon)))
            vi = ix[t["viirs"][ix]]
            if len(vi) and r.get("frp_likely") is not None:
                passes = {}
                for i in vi:
                    passes[t["pass"][i]] = passes.get(t["pass"][i], 0) + (0 if np.isnan(t["frp"][i]) else t["frp"][i])
                real = float(np.mean(list(passes.values())))
                frp_in.append(r["frp_low"] <= real <= r["frp_high"])
                frp_err.append(abs(real - r["frp_likely"]))
        move = {"fires_seen": len(mv), "median_error_km": float(np.median(mv)) if mv else None,
                "median_error_persistence_km": float(np.median(mv0)) if mv0 else None}
        frp = {"fires_with_viirs": len(frp_in), "inside_low_high": float(np.mean(frp_in)) if frp_in else None,
               "median_abs_error_mw": float(np.median(frp_err)) if frp_err else None}
        new = int((~used[dy]).sum())
        out["per_day"][f"+{k} ({dy})"] = {"extinction": ext, "spread_map": sp, "movement": move, "fire_power": frp,
                                          "detections_away_from_predicted_fires": new,
                                          "detections_total": int((t["day"] == dy).sum())}
    return out


def main():
    results = {}
    for zname, area in ZONES.items():
        for tname, d, ndays in TESTS:
            key = f"{zname} · {tname}"
            print(f"\n== {key}: pipeline on {(d - timedelta(days=4))}..{d}, truth {d + timedelta(days=1)}..{d + timedelta(days=ndays)}", flush=True)
            res, times = run_pipeline(area, d)
            if res is None:
                print("  no fires in the window"); continue
            print("  pipeline stages (s):", times, flush=True)
            t, rng = truth(area, d, ndays)
            print(f"  truth: {len(t['lat'])} real detections", flush=True)
            s = score(res, t, d, ndays)
            s.update(zone=area, last_observed_day=d.isoformat(), stage_seconds=times, truth_range=rng)
            results[key] = s
            for dk, v in s["per_day"].items():
                e, sp, mv, fp = v["extinction"], v["spread_map"], v["movement"], v["fire_power"]
                print(f"  {dk}: out {e['really_out']}/{e['fires']}  Brier {e['brier']:.3f} (base {e['baseline_base_rate_brier']:.3f})"
                      f"  AUC {e['auc'] if e['auc'] is None else round(e['auc'], 3)}  acc {e['accuracy_at_50']:.2f} (all-burn {e['baseline_all_burn_accuracy']:.2f})"
                      + (f" | spread AUC {sp['auc'] if sp['auc'] is None else round(sp['auc'], 3)} cells {sp['cells']} burned {sp['burned']}" if sp else "")
                      + (f" | move {mv['median_error_km']:.2f} km vs stay {mv['median_error_persistence_km']:.2f} km (n={mv['fires_seen']})" if mv['median_error_km'] is not None else "")
                      + (f" | FRP in range {fp['inside_low_high']:.0%} (n={fp['fires_with_viirs']})" if fp['inside_low_high'] is not None else ""),
                      flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
