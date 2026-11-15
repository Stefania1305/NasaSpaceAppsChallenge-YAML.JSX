"""Spread map of the prediction engine: chance that each 1 km cell around a fire burns on day +1, +2, +3.

For a fire on day d, today's satellite pixels are moved with the predicted centre shift and spread out to the
predicted footprint (the same picture the map page draws). Each 1 km cell within reach then gets features:
  distance to the nearest moved pixel, distance to the predicted centre in units of the likely and high radius,
  whether it lies ahead of the fire (cosine to the spread direction), the chance the fire is out, its size
and a logistic model per day ahead turns them into a burn chance. It is fitted on 2024 fire-days (cells that really
had a detection on that day = burned) and checked on 2025-2026, so a cell marked 30 % burns about 30 % of the time.

Usage:  python prediction_spread.py fit          (fit and check, writes data/prediction/models/spread.json)
In code: spread_cells(pixels, centre, row_k, coef) -> cells with burn chance (used by prediction_engine.py)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from firms import DATA_DIR
from prediction_data import H, KM_LAT, KM_LON, feature_columns

OUT = DATA_DIR / "prediction" / "models" / "spread.json"
CELL_KM = 1.0
REACH_KM = 3.0
FEATS = ["d_pix", "d_lik", "d_high", "ahead", "p_out", "log_fp", "log_frp", "inside_lik"]


def km(lat, lon, lat0, lon0):
    return (np.asarray(lon) - lon0) * KM_LON * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * KM_LAT


def cell_table(px, py, frp, fp_now, frp_now, r):
    """Candidate cells (km grid around the fire's current centre at 0,0) and their features for one day ahead.
    px, py: today's pixels (km from the current centre); r: dict with fp, hi, dx, dy, dir, move, out, frp."""
    s = min(5.0, max(1.0, math.sqrt(r["fp"] / max(fp_now, 0.05))))
    mx, my = px * s + r["dx"], py * s + r["dy"]
    rl, rh = math.sqrt(r["fp"] / math.pi), math.sqrt(r["hi"] / math.pi)
    span = max(rh + REACH_KM, np.abs(np.r_[mx, my]).max() + REACH_KM if len(mx) else 0)
    cx0, cy0 = r["dx"], r["dy"]
    xs = np.arange(math.floor((min(cx0, mx.min() if len(mx) else 0) - span) / CELL_KM), math.ceil((max(cx0, mx.max() if len(mx) else 0) + span) / CELL_KM) + 1)
    ys = np.arange(math.floor((min(cy0, my.min() if len(my) else 0) - span) / CELL_KM), math.ceil((max(cy0, my.max() if len(my) else 0) + span) / CELL_KM) + 1)
    gx, gy = np.meshgrid((xs + 0.5) * CELL_KM, (ys + 0.5) * CELL_KM)
    gx, gy = gx.ravel(), gy.ravel()
    d_pix = cKDTree(np.c_[mx, my]).query(np.c_[gx, gy], k=1)[0] if len(mx) else np.hypot(gx - cx0, gy - cy0)
    dc = np.hypot(gx - cx0, gy - cy0)
    keep = (d_pix <= REACH_KM + 1) | (dc <= rh + 1)
    gx, gy, d_pix, dc = gx[keep], gy[keep], d_pix[keep], dc[keep]
    a = math.radians(r["dir"])
    ahead = np.where(r["move"] > 0.05, ((gx - 0) * math.sin(a) + (gy - 0) * math.cos(a)) / np.maximum(np.hypot(gx, gy), 0.5), 0)
    f = pd.DataFrame({"x": gx, "y": gy, "d_pix": d_pix, "d_lik": dc / max(rl, 0.3), "d_high": dc / max(rh, 0.3),
                      "ahead": ahead, "p_out": r["out"], "log_fp": math.log1p(r["fp"]), "log_frp": math.log1p(max(r["frp"], 0)),
                      "inside_lik": (dc <= rl).astype(float)})
    return f


def design(f):
    X = f[FEATS].to_numpy("float64").copy()
    X[:, 0] = np.log1p(X[:, 0])
    X[:, 1] = np.log1p(X[:, 1])
    X[:, 2] = np.log1p(X[:, 2])
    return np.c_[X, X[:, 0] * X[:, 4], X[:, 0] * X[:, 3]]


def spread_cells(lat0, lon0, pix_lat, pix_lon, pix_frp, fp_now, frp_now, r, coef_k):
    """Cells with burn chance for one fire and day ahead, as lat/lon of the cell centres."""
    px, py = km(pix_lat, pix_lon, lat0, lon0)
    f = cell_table(px, py, pix_frp, fp_now, frp_now, r)
    z = design(f) @ np.array(coef_k["w"]) + coef_k["b"]
    f["p"] = 1 / (1 + np.exp(-z))
    f["lat"] = lat0 + f.y / KM_LAT
    f["lon"] = lon0 + f.x / (KM_LON * math.cos(math.radians(lat0)))
    return f[["lat", "lon", "p"]]


# ---------------------------------------------------------------------------
def sample_rows(split, n, seed):
    df = pd.read_csv(DATA_DIR / "prediction" / "fire_days.csv.gz", low_memory=False)
    s = df[(df.split == split) & (df.fine_fp >= 0.3) & df.run.str.startswith("fires_batch")]
    return df, s.sample(min(n, len(s)), random_state=seed)


def model_predictions(df_rows, models_dir):
    feats = None
    P = {}
    for k in range(1, H + 1):
        for t in ("growth", "dx", "dy", "frp", "out"):
            b = xgb.Booster(); b.load_model(str(models_dir / f"{t}_{k}.json")); b.set_param({"device": "cuda"})
            feats = b.feature_names
            M = xgb.DMatrix(df_rows.reindex(columns=feats).astype("float32"))
            p = b.predict(M)
            if p.ndim == 2:
                p = np.sort(p, axis=1)
                p = np.expm1(p) if t in ("growth", "frp") else np.sinh(p)
                P[f"{t}_{k}"] = p
            else:
                P[f"{t}_{k}"] = p
    return P


def build(split, n, seed, models_dir):
    df, s = sample_rows(split, n, seed)
    s = s.reset_index(drop=True)
    P = model_predictions(s, models_dir)
    tabs = []
    for run, g in s.groupby("run"):
        det = pd.read_csv(DATA_DIR / "runs" / run / "kg" / "nodes_Detection.csv.gz", low_memory=False,
                          usecols=["id:ID(Detection)", "latitude:double", "longitude:double", "acq_date:string", "frp_mw:double",
                                   "removed_by_interpreter:string"])
        det.columns = ["id", "lat", "lon", "day", "frp", "removed"]
        det = det[det.removed.isna()]
        inc = pd.read_csv(DATA_DIR / "runs" / run / "kg" / "rels_IN_CLUSTER.csv.gz")
        inc.columns = ["det", "cl", "type"]
        inc["fire"] = inc.cl.str.rsplit("|", n=1).str[-1].str.split(".").str[0].astype(int)
        det = det.merge(inc[["det", "fire"]].drop_duplicates("det"), left_on="id", right_on="det", how="left")
        by_day = {d: x for d, x in det.groupby("day")}
        for i, row in g.iterrows():
            lat0, lon0 = (row.clat if not pd.isna(row.clat) else row.lat0), (row.clon if not pd.isna(row.clon) else row.lon0)
            today = by_day.get(row.day)
            if today is None:
                continue
            mine = today[today.fire == row.fire_id]
            if mine.empty:
                continue
            px, py = km(mine.lat, mine.lon, lat0, lon0)
            for k in range(1, H + 1):
                dk = (pd.Timestamp(row.day) + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
                r = {"fp": row.fine_fp + max(0.0, P[f"growth_{k}"][i, 1]), "hi": row.fine_fp + max(0.0, P[f"growth_{k}"][i, 2]),
                     "dx": P[f"dx_{k}"][i, 1], "dy": P[f"dy_{k}"][i, 1], "out": float(P[f"out_{k}"][i]),
                     "frp": P[f"frp_{k}"][i, 1]}
                r["dir"] = math.degrees(math.atan2(r["dx"], r["dy"])) % 360
                r["move"] = math.hypot(r["dx"], r["dy"])
                r["hi"] = max(r["hi"], r["fp"])
                f = cell_table(px, py, mine.frp.to_numpy(), row.fine_fp, row.frp if not pd.isna(row.frp) else 0, r)
                fut = by_day.get(dk)
                burned = np.zeros(len(f), bool)
                if fut is not None and len(fut):
                    fx, fy = km(fut.lat, fut.lon, lat0, lon0)
                    near = (np.abs(fx) < 200) & (np.abs(fy) < 200)
                    if near.any():
                        cells = set(zip(np.floor(fx[near] / CELL_KM).astype(int), np.floor(fy[near] / CELL_KM).astype(int)))
                        burned = np.array([(int(math.floor(x / CELL_KM)), int(math.floor(y / CELL_KM))) in cells for x, y in zip(f.x, f.y)])
                f["burned"] = burned
                f["k"] = k
                tabs.append(f)
    return pd.concat(tabs, ignore_index=True)


def fit():
    t0 = time.time()
    md = DATA_DIR / "prediction" / "models_split" / "weather"        # trained on <= 2023: 2024 and 2025-26 are unseen
    tr = build("check", 2500, 0, md)
    te = build("test", 2500, 1, md)
    print(f"cells: fit {len(tr):,} (2024), check {len(te):,} (2025-2026); {time.time() - t0:.0f} s", flush=True)
    coef, card = {}, {}
    for k in range(1, H + 1):
        a, b = tr[tr.k == k], te[te.k == k]
        m = LogisticRegression(max_iter=2000, C=1.0).fit(design(a), a.burned)
        p = m.predict_proba(design(b))[:, 1]
        coef[str(k)] = {"w": m.coef_[0].tolist(), "b": float(m.intercept_[0])}
        bins = np.clip((p * 10).astype(int), 0, 9)
        rel = [{"predicted_%": round(100 * float(p[bins == i].mean()), 1), "burned_%": round(100 * float(b.burned.to_numpy()[bins == i].mean()), 1),
                "cells": int((bins == i).sum())} for i in range(10) if (bins == i).sum() >= 50]
        card[f"day+{k}"] = {"cells_check": len(b), "burned_share_%": round(100 * float(b.burned.mean()), 2),
                            "auc": round(float(roc_auc_score(b.burned, p)), 3),
                            "brier": round(float(brier_score_loss(b.burned, p)), 4),
                            "brier_base_rate": round(float(brier_score_loss(b.burned, np.full(len(b), a.burned.mean()))), 4),
                            "reliability": rel}
        print(f"  day +{k}: AUC {card[f'day+{k}']['auc']}, Brier {card[f'day+{k}']['brier']} (base {card[f'day+{k}']['brier_base_rate']})", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cell_km": CELL_KM, "features": FEATS, "coef": coef, "check": card}, indent=1), encoding="utf-8")
    print(f"saved {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
