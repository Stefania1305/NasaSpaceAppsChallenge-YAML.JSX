"""Full metrics of every prediction model on the same unseen test years (2025-2026).

Models compared (all trained on 2016-2023, so 2025-2026 is unseen):
  fast         the first XGBoost models (87 features, observed weather)            data/prediction/models_split
  full_xgb     XGBoost v2 (118 features, forecast-like weather)                     data/prediction/models_v2
  sequence     the sequence network (GRU, numpy)                                    data/prediction/models_seq
  blend        production setup: XGBoost v2 and the network averaged, extinction from XGBoost
  persistence  "the fire keeps doing what it did today"

Metrics per day ahead (with weather; the no-weather variants are scored too):
  extinction, merge   accuracy, balanced accuracy, precision, recall, F1, ROC AUC, PR AUC (average precision),
                      Brier score, calibration error (mean |predicted - observed| over 10 bins, weighted)
                      threshold: 0.5 for extinction; for merges the one with the best F1 on 2024
  growth, frp,        MAE and median absolute error of the likely value, low..high coverage (aim 80 %),
  frp_peak            pinball (quantile) loss on the log scale; growth class accuracy and macro F1;
                      fire power within a factor of 2
  direction           share within 45 / 90 degrees, median angle error, median centre error (km)
  overall             mean of growth class accuracy, fire power within x2, direction within 45, extinction
                      accuracy and merge balanced accuracy (the headline figure used elsewhere)

v3 (fire chains, survival extinction, two-part growth) is scored when PRED_TABLE=fire_days_v3.csv.gz: then every
model is scored on the corrected labels (fires that vanish and come back are not counted as extinguished):
  v2_blend     the v2 production setup                       models_v2 + models_seq
  v3_xgb       v3 tree models                                models_v3
  v3_sequence  the sequence network trained on chains        models_seq_v3
  v3_blend     v3 production setup: averaged, extinction from the survival model

Usage:  python prediction_metrics.py            (PRED_TABLE=fire_days_v3.csv.gz for the v3 comparison)
Output: data/prediction/accuracy/metrics_all.json (metrics_v3.json with the v3 table)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from firms import DATA_DIR
from prediction_data import H, add_history
from prediction_train import CONT, persistence

P_DIR = DATA_DIR / "prediction"
OUT = P_DIR / "accuracy"
CUTS = (0.1, 2.0)


def xgb_models(d, variant):
    m = {}
    for p in (d / variant).glob("*.json"):
        b = xgb.Booster(); b.load_model(str(p)); b.set_param({"device": "cuda"}); m[p.stem] = b
    return m


def xgb_predict(models, X):
    feats = next(iter(models.values())).feature_names
    M = xgb.DMatrix(X.reindex(columns=feats).astype("float32"))
    P = {}
    for k in range(1, H + 1):
        for t, (_, inv) in CONT.items():
            if f"{t}_{k}" in models:
                P[f"{t}_{k}"] = inv(np.sort(models[f"{t}_{k}"].predict(M), axis=1).astype("float64"))
        for t in ("out", "merge"):
            if f"{t}_{k}" in models:
                P[f"{t}_{k}"] = models[f"{t}_{k}"].predict(M).astype("float64")
    return P


def blend(a, b, skip=("out",)):
    out = {}
    for k in a:
        if k.split("_")[0] in skip or k not in b:
            out[k] = a[k]
        else:
            v = (a[k] + b[k]) / 2
            out[k] = np.sort(v, axis=1) if v.ndim == 2 else v
    return out


def calib(y, p):
    bins = np.clip((p * 10).astype(int), 0, 9)
    tot = sum(abs(p[bins == i].mean() - y[bins == i].mean()) * (bins == i).sum() for i in range(10) if (bins == i).sum())
    return float(tot / len(p) * 100)


def binary(y, p, thr):
    keep = ~np.isnan(y.astype("float64"))          # censored extinction labels are unknown, left out
    y, p = y[keep].astype(int), p[keep]
    h = (p >= thr).astype(int)
    return {"accuracy_%": round(100 * float((h == y).mean()), 1),
            "balanced_accuracy_%": round(100 * float((np.mean(h[y == 1]) + np.mean(1 - h[y == 0])) / 2), 1),
            "precision_%": round(100 * float(precision_score(y, h, zero_division=0)), 1),
            "recall_%": round(100 * float(recall_score(y, h, zero_division=0)), 1),
            "f1_%": round(100 * float(f1_score(y, h, zero_division=0)), 1),
            "roc_auc": round(float(roc_auc_score(y, p)), 3), "pr_auc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4), "calibration_error_pts": round(calib(y, p), 2),
            "threshold": round(float(thr), 3), "rate_%": round(100 * float(y.mean()), 2)}


def best_f1_threshold(y, p):
    keep = ~np.isnan(y.astype("float64"))
    y, p = y[keep], p[keep]
    ts = np.linspace(0.02, 0.8, 79)
    return float(max(ts, key=lambda t: f1_score(y.astype(int), (p >= t).astype(int), zero_division=0)))


def pinball(y, q, fwd):
    yt = fwd(y)
    qt = fwd(q)
    loss = 0
    for j, a in enumerate((0.1, 0.5, 0.9)):
        e = yt - qt[:, j]
        loss += np.mean(np.maximum(a * e, (a - 1) * e))
    return float(loss / 3)


def continuous(df, P, k, kind):
    s = {}
    for t in ("growth", "frp", "frp_peak"):
        y = df[f"{t}_{k}"].to_numpy("float64")
        ok = ~np.isnan(y) & ((y > 0) if t != "growth" else True)
        if kind == "persistence":
            lik = persistence(df, t, k)
            ok &= ~np.isnan(lik)
            r = {"mae": round(float(np.mean(np.abs(lik[ok] - y[ok]))), 3), "median_ae": round(float(np.median(np.abs(lik[ok] - y[ok]))), 3)}
        else:
            q = P[f"{t}_{k}"]
            lik = q[:, 1]
            fwd = CONT[t][0]
            r = {"mae": round(float(np.mean(np.abs(lik[ok] - y[ok]))), 3), "median_ae": round(float(np.median(np.abs(lik[ok] - y[ok]))), 3),
                 "coverage_low_high_%": round(100 * float(np.mean((y[ok] >= q[ok, 0]) & (y[ok] <= q[ok, 2]))), 1),
                 "pinball_log": round(pinball(np.clip(y[ok], 0, None), np.clip(q[ok], 0, None), fwd), 4)}
        if t == "growth":
            yc, pc = np.digitize(y[ok], CUTS), np.digitize(lik[ok], CUTS)
            r["class_accuracy_%"] = round(100 * float((yc == pc).mean()), 1)
            r["class_macro_f1_%"] = round(100 * float(f1_score(yc, pc, average="macro")), 1)
        if t == "frp":
            ratio = lik[ok] / y[ok]
            r["within_x2_%"] = round(100 * float(np.mean((ratio >= 0.5) & (ratio <= 2))), 1)
        s[t] = r
    rx, ry = df[f"dx_{k}"].to_numpy(), df[f"dy_{k}"].to_numpy()
    if kind == "persistence":
        px, py = persistence(df, "dx", k), persistence(df, "dy", k)
    else:
        px, py = P[f"dx_{k}"][:, 1], P[f"dy_{k}"][:, 1]
    mv = np.hypot(rx, ry) >= 1.0
    if kind == "persistence":
        mv &= np.hypot(px, py) > 0
    e = np.abs((np.degrees(np.arctan2(px[mv], py[mv])) - np.degrees(np.arctan2(rx[mv], ry[mv])) + 180) % 360 - 180)
    s["direction"] = {"within_45_%": round(100 * float(np.mean(e <= 45)), 1), "within_90_%": round(100 * float(np.mean(e <= 90)), 1),
                      "median_angle_error_deg": round(float(np.median(e)), 1),
                      "median_centre_error_km": round(float(np.median(np.hypot(px[mv] - rx[mv], py[mv] - ry[mv]))), 3),
                      "fires_moved_1km": int(mv.sum())}
    return s


def score(df, P, k, merge_thr, kind="model"):
    r = continuous(df, P, k, kind)
    if kind == "persistence":
        r["out"] = binary(df[f"out_{k}"].to_numpy(), np.zeros(len(df)), 0.5)          # "still burning"
        r["merge"] = binary(df[f"merge_{k}"].to_numpy(), np.zeros(len(df)), 0.5)      # "never merges"
        for x in ("roc_auc", "pr_auc", "brier", "calibration_error_pts"):
            r["out"][x] = r["merge"][x] = None
    else:
        r["out"] = binary(df[f"out_{k}"].to_numpy(), P[f"out_{k}"], 0.5)
        r["merge"] = binary(df[f"merge_{k}"].to_numpy(), P[f"merge_{k}"], merge_thr)
    r["overall_%"] = round(float(np.mean([r["growth"]["class_accuracy_%"], r["frp"]["within_x2_%"], r["direction"]["within_45_%"],
                                          r["out"]["accuracy_%"], r["merge"]["balanced_accuracy_%"]])), 1)
    return r


def main():
    table = os.environ.get("PRED_TABLE", "fire_days.csv.gz")
    if table != "fire_days.csv.gz":
        return main_v3(table)
    return main_v2()


def main_v3(table):
    import prediction_v3 as V
    t0 = time.time()
    df = add_history(pd.read_csv(P_DIR / table, low_memory=False))
    soil = df.copy()
    for k in range(1, H + 1):
        for c in ("sm0", "sm1", "sm2"):
            soil[f"w{k}_{c}"] = soil[f"w0_{c}"]
    te, ck = df.split == "test", df.split == "check"
    from seq_numpy import NumpySeq
    new_dir, new_seq = os.environ.get("PRED_NEW", "models_v3"), os.environ.get("PRED_NEW_SEQ", "models_seq_v3")
    seq2, seq3 = NumpySeq.load(P_DIR / "models_seq"), NumpySeq.load(P_DIR / new_seq)
    res = {"table": table, "test_rows": int(te.sum()), "test_fires": int(df[te].groupby(["run", "fire_id"]).ngroups), "models": {}}
    for variant in ("weather", "no_weather"):
        w = variant == "weather"
        x2 = xgb_models(P_DIR / "models_v2", variant)
        m3 = V.load(P_DIR / new_dir, variant)
        fns = {"v2_blend": lambda D: blend(xgb_predict(x2, D), seq2.predict_df(D, weather=w)),
               "v3_xgb": lambda D: V.predict(P_DIR / new_dir, D, variant, models=m3),
               "v3_sequence": lambda D: seq3.predict_df(D, weather=w)}
        preds = {n: (f(soil[te]), f(soil[ck])) for n, f in fns.items()}
        preds["v3_blend"] = (blend(preds["v3_xgb"][0], preds["v3_sequence"][0]), blend(preds["v3_xgb"][1], preds["v3_sequence"][1]))
        for name, (Pt, Pc) in preds.items():
            out = {}
            for k in range(1, H + 1):
                thr = best_f1_threshold(df[ck][f"merge_{k}"].to_numpy(), Pc[f"merge_{k}"])
                out[str(k)] = score(df[te], Pt, k, thr)
                if f"grow_{k}" in Pt:
                    y = (df[te][f"growth_{k}"].to_numpy() >= 0.1).astype(float)
                    out[str(k)]["grow"] = binary(y, Pt[f"grow_{k}"], 0.5)
            res["models"].setdefault(name, {})[variant] = out
        print(f"  {variant}: scored in {time.time() - t0:.0f} s", flush=True)
    res["models"]["persistence"] = {"weather": {str(k): score(df[te], None, k, 0.5, "persistence") for k in range(1, H + 1)}}
    OUT.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("PRED_OUT_NAME", "metrics_v3.json")
    (OUT / name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"saved {OUT / name} in {time.time() - t0:.0f} s")


def main_v2():
    t0 = time.time()
    df = add_history(pd.read_csv(P_DIR / "fire_days.csv.gz", low_memory=False))
    soil = df.copy()                                   # v2 models expect day d's soil moisture on days +1..+3
    for k in range(1, H + 1):
        for c in ("sm0", "sm1", "sm2"):
            soil[f"w{k}_{c}"] = soil[f"w0_{c}"]
    te, ck = df.split == "test", df.split == "check"
    from seq_numpy import NumpySeq
    seq = NumpySeq.load(P_DIR / "models_seq")
    res = {"test_rows": int(te.sum()), "test_fires": int(df[te].groupby(["run", "fire_id"]).ngroups), "models": {}}
    for variant in ("weather", "no_weather"):
        w = variant == "weather"
        fast, full = xgb_models(P_DIR / "models_split", variant), xgb_models(P_DIR / "models_v2", variant)
        preds = {}
        for name, fn in (("fast", lambda X, Xs: xgb_predict(fast, X)),
                         ("full_xgb", lambda X, Xs: xgb_predict(full, Xs)),
                         ("sequence", lambda X, Xs: seq.predict_df(Xs, weather=w))):
            preds[name] = (fn(df[te], soil[te]), fn(df[ck], soil[ck]))
        preds["blend"] = (blend(preds["full_xgb"][0], preds["sequence"][0]), blend(preds["full_xgb"][1], preds["sequence"][1]))
        for name, (Pt, Pc) in preds.items():
            out = {}
            for k in range(1, H + 1):
                thr = best_f1_threshold(df[ck][f"merge_{k}"].to_numpy(), Pc[f"merge_{k}"])
                out[str(k)] = score(df[te], Pt, k, thr)
            res["models"].setdefault(name, {})[variant] = out
        print(f"  {variant}: scored in {time.time() - t0:.0f} s", flush=True)
    res["models"]["persistence"] = {"weather": {str(k): score(df[te], None, k, 0.5, "persistence") for k in range(1, H + 1)}}
    spread = json.loads((P_DIR / "models" / "spread.json").read_text(encoding="utf-8"))["check"]
    res["spread_map"] = {k: {x: v[x] for x in ("auc", "brier", "brier_base_rate", "burned_share_%", "cells_check")} for k, v in spread.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_all.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"saved {OUT / 'metrics_all.json'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
