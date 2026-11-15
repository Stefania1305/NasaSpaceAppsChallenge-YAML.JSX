"""Accuracy tests of the prediction engine (quality and quantity), with accuracy percentages.

  heldout    accuracy on the 2025-2026 fire-days the models never saw, per output and day ahead, against
             persistence and the no-weather models; split by region, fire size, season and weather coverage;
             calibration of the low / likely / high ranges and of the chances
  behaviour  does the model react like fire does: wind turned 90 degrees, stronger wind, drier air, rain;
             plus rules (low <= likely <= high, chances in 0..1, same input -> same output, missing weather works)
  cv         robustness: retrain without one year (2022, 2023, 2024) or one world region (Americas, Africa,
             Asia and Australia) and test on what was left out (quicker check models: learning rate 0.1, days +1 and +3)
  forecast   real use: ~600 fire-days from March 2024 on, predicted again with the weather forecasts that were
             issued back then (Open-Meteo previous-runs API) instead of the weather that actually happened

Accuracy % per output (share of predictions that are right):
  growth class    likely growth in the right class: none (< 0.1 km2), small (0.1-2 km2), large (>= 2 km2)
  growth within   real growth within max(0.5 km2, 30 %) of the likely value
  range           real value inside the low..high range (aim ~80 %)
  FRP x2          likely fire power within a factor of 2 of the measured one
  direction 45    spread direction within 45 degrees of the real one (fires that moved >= 1 km)
  out             right about whether the fire is out by that day (threshold 0.5)
  merge           balanced accuracy (mean of hit rate on merges and on non-merges; merges are rare, so plain
                  accuracy would be 94 % for "never merges"), threshold tuned on 2024
  overall         mean of growth class, FRP x2, direction 45, out and merge

Usage:  python test_prediction_accuracy.py heldout|behaviour|cv|forecast|all
Output: data/prediction/accuracy/<test>.json and report.md
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from firms import DATA_DIR
from prediction_data import H, WX, feature_columns
from prediction_train import CONT, persistence

OUT = DATA_DIR / "prediction" / "accuracy"
import os
MODELS = DATA_DIR / "prediction" / os.environ.get("PRED_MODELS", "models")       # model folder to test
SOIL_FROM_D = os.environ.get("PRED_SOIL_FROM_D") == "1"
TABLE = os.environ.get("PRED_TABLE", "fire_days.csv.gz")                         # fire_days_v3.csv.gz: fire chains                           # days +1..+3 soil moisture = day d (as in real use)
GROWTH_CUTS = (0.1, 2.0)
REGION_GROUPS = {"Americas": [0, 1, 2, 11, 12, 13], "Africa": [14, 15], "Asia and Australia": [3, 5, 6, 7, 8, 9, 10]}
SIZE = [(0, 1, "under 1 km2"), (1, 10, "1-10 km2"), (10, 100, "10-100 km2"), (100, 1e9, "over 100 km2")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    from prediction_data import add_history
    df = add_history(pd.read_csv(DATA_DIR / "prediction" / TABLE, low_memory=False))
    if SOIL_FROM_D:
        for k in range(1, H + 1):
            for c in ("sm0", "sm1", "sm2"):
                df[f"w{k}_{c}"] = df[f"w0_{c}"]
    return df


class SeqWrap:
    """The sequence model (numpy) behind the same interface as a set of XGBoost models."""

    def __init__(self, folder, weather):
        from seq_numpy import NumpySeq
        self.m, self.weather = NumpySeq.load(folder), weather

    def predict(self, X):
        return self.m.predict_df(X, weather=self.weather)


class V3Wrap:
    """v3 models (survival + two-part growth) behind the usual interface."""

    def __init__(self, folder, variant):
        import prediction_v3 as V
        self.V, self.folder, self.variant = V, folder, variant
        self.models = V.load(folder, variant)

    def predict(self, X):
        return self.V.predict(self.folder, X, self.variant, models=self.models)


class BlendWrap:
    """Average of XGBoost and the sequence model: quantiles averaged then sorted, chances averaged."""

    def __init__(self, xgb_models, seq):
        self.x, self.s = xgb_models, seq

    def predict(self, X):
        a, b = predict(self.x, X, None), self.s.predict(X)
        out = {}
        skip_out = os.environ.get("PRED_BLEND_OUT", "xgb") == "xgb"     # extinction from the tree model alone
        for k in a:
            if k in b and not (skip_out and k.startswith("out")):
                v = (a[k] + b[k]) / 2
                out[k] = np.sort(v, axis=1) if v.ndim == 2 else v
            else:
                out[k] = a[k]
        return out


def boosters(variant):
    blend = os.environ.get("PRED_BLEND")                    # e.g. "models_v2,models_seq"
    if blend:
        xd, sd = blend.split(",")
        xdir = DATA_DIR / "prediction" / xd
        xm = V3Wrap(xdir, variant) if (xdir / variant / "survival.json").exists() else boosters_in(xdir, variant)
        return BlendWrap(xm, SeqWrap(DATA_DIR / "prediction" / sd, variant == "weather"))
    if (MODELS / "weights.npz").exists():
        return SeqWrap(MODELS, variant == "weather")
    if (MODELS / variant / "survival.json").exists():
        return V3Wrap(MODELS, variant)
    return boosters_in(MODELS, variant)


def boosters_in(models_dir, variant):
    out = {}
    for p in (models_dir / variant).glob("*.json"):
        b = xgb.Booster()
        b.load_model(str(p))
        b.set_param({"device": "cuda"})
        out[p.stem] = b
    return out


def predict(models, X, feats):
    if hasattr(models, "predict"):
        return models.predict(X)
    feats = next(iter(models.values())).feature_names or feats
    M = xgb.DMatrix(X.reindex(columns=feats).astype("float32"))
    P = {}
    for k in range(1, H + 1):
        for t, (fwd, inv) in CONT.items():
            m = models.get(f"{t}_{k}")
            if m is not None:
                P[f"{t}_{k}"] = inv(np.sort(m.predict(M), axis=1).astype("float64"))
        for t in ("out", "merge"):
            m = models.get(f"{t}_{k}")
            if m is not None:
                P[f"{t}_{k}"] = m.predict(M).astype("float64")
    return P


def gclass(v):
    return np.digitize(v, GROWTH_CUTS)


def angle_err(px, py, rx, ry):
    a = np.degrees(np.arctan2(px, py)) - np.degrees(np.arctan2(rx, ry))
    return np.abs((a + 180) % 360 - 180)


def scores(df, P, k, merge_thr=0.5, kind="model"):
    """Accuracy % of one day ahead. P[...] arrays aligned with df rows (quantiles [n,3] for continuous)."""
    s = {}
    g = df[f"growth_{k}"].to_numpy()
    if kind == "persistence":
        gl = persistence(df, "growth", k); glo = ghi = None
        frp = persistence(df, "frp", k)
        dx, dy = persistence(df, "dx", k), persistence(df, "dy", k)
        pout = np.zeros(len(df)); pm = np.zeros(len(df))
    else:
        q = P[f"growth_{k}"]; gl, glo, ghi = q[:, 1], q[:, 0], q[:, 2]
        frp = P[f"frp_{k}"][:, 1]
        dx, dy = P[f"dx_{k}"][:, 1], P[f"dy_{k}"][:, 1]
        pout, pm = P[f"out_{k}"], P[f"merge_{k}"]
    s["growth_class"] = float(np.mean(gclass(gl) == gclass(g)))
    s["growth_within"] = float(np.mean(np.abs(gl - g) <= np.maximum(0.5, 0.3 * np.abs(g))))
    if glo is not None:
        s["growth_range"] = float(np.mean((g >= glo) & (g <= ghi)))
    fr = df[f"frp_{k}"].to_numpy(); okf = ~np.isnan(fr) & (fr > 0) & ~np.isnan(frp) & (frp > 0)
    if okf.any():
        s["frp_x2"] = float(np.mean((frp[okf] / fr[okf] >= 0.5) & (frp[okf] / fr[okf] <= 2)))
        if kind != "persistence":
            q = P[f"frp_{k}"]
            s["frp_range"] = float(np.mean((fr[okf] >= q[okf, 0]) & (fr[okf] <= q[okf, 2])))
    rx, ry = df[f"dx_{k}"].to_numpy(), df[f"dy_{k}"].to_numpy()
    mv = np.hypot(rx, ry) >= 1.0
    if kind == "persistence":
        mv &= np.hypot(dx, dy) > 0
    if mv.any():
        e = angle_err(dx[mv], dy[mv], rx[mv], ry[mv])
        s["direction_45"] = float(np.mean(e <= 45)); s["direction_90"] = float(np.mean(e <= 90))
        s["direction_median_err_deg"] = float(np.median(e))
        s["centre_err_median_km"] = float(np.median(np.hypot(dx[mv] - rx[mv], dy[mv] - ry[mv])))
    out = df[f"out_{k}"].to_numpy("float64")
    ko = ~np.isnan(out)
    s["out"] = float(np.mean((pout[ko] >= 0.5) == (out[ko] == 1)))
    if kind != "persistence" and len(np.unique(out[ko])) > 1:
        s["out_auc"] = float(roc_auc_score(out[ko], pout[ko]))
    m = df[f"merge_{k}"].to_numpy() == 1
    hit = (pm >= merge_thr)
    if m.any() and (~m).any():
        s["merge"] = float((np.mean(hit[m]) + np.mean(~hit[~m])) / 2)
        s["merge_recall"] = float(np.mean(hit[m]))
        s["merge_precision"] = float(np.mean(m[hit])) if hit.any() else 0.0
        s["merge_plain_accuracy"] = float(np.mean(hit == m))
        if kind != "persistence":
            s["merge_auc"] = float(roc_auc_score(m, pm))
    parts = [s.get(x) for x in ("growth_class", "frp_x2", "direction_45", "out", "merge")]
    s["overall"] = float(np.mean([p for p in parts if p is not None]))
    s["rows"] = int(len(df))
    return {k2: (round(v * 100, 1) if isinstance(v, float) and not k2.endswith(("_deg", "_km")) and "auc" not in k2 else
                 round(v, 3) if isinstance(v, float) else v) for k2, v in s.items()}


def merge_thresholds(models, df, feats):
    """Merge threshold per day ahead that maximises balanced accuracy on the 2024 check year."""
    c = df[df.split == "check"]
    P = predict(models, c, feats)
    thr = {}
    for k in range(1, H + 1):
        m = c[f"merge_{k}"].to_numpy() == 1
        best = max(np.linspace(0.02, 0.6, 59), key=lambda t: (np.mean(P[f"merge_{k}"][m] >= t) + np.mean(P[f"merge_{k}"][~m] < t)) / 2)
        thr[k] = float(best)
    return thr


def season(df):
    m = df.month.to_numpy()
    south = df.lat0.to_numpy() < 0
    m = np.where(south, (m + 5) % 12 + 1, m)       # southern hemisphere: shift by 6 months so seasons line up
    return np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])], ["winter", "spring", "summer"], "autumn")


# ---------------------------------------------------------------------------
def heldout(df):
    te = df[df.split == "test"].reset_index(drop=True)
    res = {"rows_test": len(te), "fires_test": int(te.groupby(["run", "fire_id"]).ngroups), "days": {}}
    for variant in ("weather", "no_weather"):
        models = boosters(variant)
        feats = feature_columns(df, weather=(variant == "weather"))
        thr = merge_thresholds(models, df, feats)
        P = predict(models, te, feats)
        for k in range(1, H + 1):
            res["days"].setdefault(str(k), {})[variant] = scores(te, P, k, thr[k])
        res.setdefault("merge_threshold", {})[variant] = thr
        if variant == "weather":
            Pw = P
            thr_w = thr
    for k in range(1, H + 1):
        res["days"][str(k)]["persistence"] = scores(te, None, k, kind="persistence")
        res["days"][str(k)]["never_merges_balanced"] = 50.0
    # breakdowns (weather models, mean over the 3 days)
    def group_scores(mask):
        sub = te[mask].reset_index(drop=True)
        if len(sub) < 100:
            return None
        Ps = {c: v[mask.to_numpy() if hasattr(mask, "to_numpy") else mask] for c, v in Pw.items()}
        r = [scores(sub, Ps, k, thr_w[k]) for k in range(1, H + 1)]
        keys = ["overall", "growth_class", "growth_range", "frp_x2", "direction_45", "out", "merge"]
        return {"rows": len(sub), **{x: round(float(np.mean([d[x] for d in r if x in d])), 1) for x in keys if any(x in d for d in r)}}
    from prediction_data import REGIONS
    br = {"region": {}, "size": {}, "season": {}, "weather_data": {}}
    for i, name in enumerate(REGIONS):
        g = group_scores(te.region == i)
        if g: br["region"][name] = g
    for lo, hi, name in SIZE:
        g = group_scores((te.fine_fp >= lo) & (te.fine_fp < hi))
        if g: br["size"][name] = g
    se = pd.Series(season(te))
    for name in ("spring", "summer", "autumn", "winter"):
        g = group_scores(se == name)
        if g: br["season"][name] = g
    hw = te.w1_tmax.notna()
    br["weather_data"]["with weather"] = group_scores(hw)
    br["weather_data"]["without weather"] = group_scores(~hw)
    res["breakdown"] = br
    # calibration
    cal = {"quantiles": {}, "chances": {}}
    for k in range(1, H + 1):
        for t in ("growth", "frp", "dx", "dy"):
            y = te[f"{t}_{k}"].to_numpy(); ok = ~np.isnan(y)
            q = Pw[f"{t}_{k}"]
            cal["quantiles"][f"{t}_{k}"] = {"below_low_%": round(100 * float(np.mean(y[ok] < q[ok, 0])), 1),
                                           "above_high_%": round(100 * float(np.mean(y[ok] > q[ok, 2])), 1),
                                           "below_likely_%": round(100 * float(np.mean(y[ok] < q[ok, 1])), 1),
                                           "aim": "10 / 10 / 50"}
        for t in ("out", "merge"):
            p, y = Pw[f"{t}_{k}"], te[f"{t}_{k}"].to_numpy("float64")
            keep = ~np.isnan(y)
            p, y = p[keep], y[keep]
            bins = np.clip((p * 10).astype(int), 0, 9)
            tab = [{"predicted_%": round(100 * float(p[bins == b].mean()), 1), "observed_%": round(100 * float(y[bins == b].mean()), 1),
                    "rows": int((bins == b).sum())} for b in range(10) if (bins == b).sum() >= 30]
            ece = sum(abs(r["predicted_%"] - r["observed_%"]) * r["rows"] for r in tab) / max(1, sum(r["rows"] for r in tab))
            cal["chances"][f"{t}_{k}"] = {"calibration_error_%": round(ece, 2), "bins": tab}
    res["calibration"] = cal
    return res


# ---------------------------------------------------------------------------
def behaviour(df):
    te = df[(df.split == "test") & df.w1_tmax.notna()].sample(3000, random_state=1).reset_index(drop=True)
    models = boosters("weather")
    feats = feature_columns(df, weather=True)
    base = predict(models, te, feats)
    res = {"rows": len(te), "checks": {}}

    def change(fn):
        X = te.copy()
        for k in range(1, H + 1):
            fn(X, k)
        return predict(models, X, feats)

    def turn(X, k):           # wind turned 90 degrees clockwise (to the right) on days +1..+3
        u, v = X[f"w{k}_u_mean"].copy(), X[f"w{k}_v_mean"].copy()
        X[f"w{k}_u_mean"], X[f"w{k}_v_mean"] = v, -u
    P = change(turn)
    rows = []
    for k in range(1, H + 1):
        d0 = np.degrees(np.arctan2(base[f"dx_{k}"][:, 1], base[f"dy_{k}"][:, 1]))
        d1 = np.degrees(np.arctan2(P[f"dx_{k}"][:, 1], P[f"dy_{k}"][:, 1]))
        sh = (d1 - d0 + 180) % 360 - 180
        moved = np.hypot(P[f"dx_{k}"][:, 1], P[f"dy_{k}"][:, 1]) > 0.2
        rows.append({"day": k, "turned_clockwise_%": round(100 * float(np.mean(sh[moved] > 5)), 1),
                     "turned_anticlockwise_%": round(100 * float(np.mean(sh[moved] < -5)), 1),
                     "median_turn_deg": round(float(np.median(sh[moved])), 1)})
    res["checks"]["wind turned 90 deg clockwise -> spread turns clockwise"] = {
        "per_day": rows, "pass": all(r["turned_clockwise_%"] > r["turned_anticlockwise_%"] for r in rows)}

    def effect(name, fn, target, expect):
        P = change(fn)
        out = []
        for k in range(1, H + 1):
            a = P[f"{target}_{k}"][:, 1] if P[f"{target}_{k}"].ndim == 2 else P[f"{target}_{k}"]
            b = base[f"{target}_{k}"][:, 1] if base[f"{target}_{k}"].ndim == 2 else base[f"{target}_{k}"]
            up, down = float(np.mean(a > b * 1.001 + 1e-9)), float(np.mean(a < b * 0.999 - 1e-9))
            out.append({"day": k, "went_up_%": round(100 * up, 1), "went_down_%": round(100 * down, 1),
                        "mean_before": round(float(np.mean(b)), 4), "mean_after": round(float(np.mean(a)), 4)})
        ok = all((r["mean_after"] > r["mean_before"]) if expect == "up" else (r["mean_after"] < r["mean_before"]) for r in out)
        res["checks"][name] = {"per_day": out, "expected": expect, "pass": ok}

    def windier(X, k):
        for c in ("wind_mean", "wind_max", "gust_max", "u_mean", "v_mean"):
            X[f"w{k}_{c}"] = X[f"w{k}_{c}"] * 1.5
    def drier(X, k):
        X[f"w{k}_rh_min"] = (X[f"w{k}_rh_min"] - 20).clip(lower=3); X[f"w{k}_rh_mean"] = (X[f"w{k}_rh_mean"] - 20).clip(lower=5)
        X[f"w{k}_vpd_max"] = X[f"w{k}_vpd_max"] * 1.5
    def rain(X, k):
        X[f"w{k}_rain"] = 15.0; X[f"w{k}_since_rain"] = 0.0
        X[f"w{k}_rh_min"] = X[f"w{k}_rh_min"].clip(lower=70); X[f"w{k}_rh_mean"] = X[f"w{k}_rh_mean"].clip(lower=85)
        X[f"w{k}_vpd_max"] = X[f"w{k}_vpd_max"].clip(upper=0.5)
    effect("wind x1.5 -> more growth", windier, "growth", "up")
    effect("air 20 points drier -> more growth", drier, "growth", "up")
    effect("air 20 points drier -> more fire power", drier, "frp", "up")
    effect("15 mm rain a day -> higher chance out", rain, "out", "up")
    effect("15 mm rain a day -> less growth", rain, "growth", "down")
    # rules
    rules = {}
    rules["low <= likely <= high (after sorting)"] = all(np.all(np.diff(v, axis=1) >= 0) for c, v in base.items() if v.ndim == 2)
    rules["growth never negative"] = all(np.all(base[f"growth_{k}"] >= -1e-6) for k in range(1, H + 1))
    rules["chances in 0..1"] = all(np.all((base[f"{t}_{k}"] >= 0) & (base[f"{t}_{k}"] <= 1)) for t in ("out", "merge") for k in range(1, H + 1))
    again = predict(models, te, feats)
    rules["same input -> same output"] = all(np.array_equal(base[c], again[c]) for c in base)
    X = te.copy()
    X[[f"w{k}_{c}" for k in range(H + 1) for c in WX]] = np.nan
    try:
        P = predict(models, X, feats)
        rules["all weather missing still predicts"] = all(np.isfinite(v).all() for v in P.values())
    except Exception as e:                      # noqa: BLE001
        rules["all weather missing still predicts"] = False
    res["rules"] = rules
    res["passed"] = int(sum(c["pass"] for c in res["checks"].values()) + sum(rules.values()))
    res["total"] = len(res["checks"]) + len(rules)
    return res


# ---------------------------------------------------------------------------
CV_PARAMS = {"tree_method": "hist", "device": "cuda", "eta": 0.1, "max_depth": 8, "min_child_weight": 5,
             "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0}


def cv_fold(df, name, train, test, feats):
    tr, te = df[train], df[test].reset_index(drop=True)
    fires = tr[["run", "fire_id"]].drop_duplicates()
    val_f = fires.sample(frac=0.1, random_state=0)
    isval = tr.set_index(["run", "fire_id"]).index.isin(val_f.set_index(["run", "fire_id"]).index)
    a, b = tr[~isval], tr[isval]
    Xa, Xb, Xt = (x[feats].astype("float32") for x in (a, b, te))
    da = xgb.QuantileDMatrix(Xa, label=np.zeros(len(a)))
    db = xgb.QuantileDMatrix(Xb, label=np.zeros(len(b)), ref=da)
    dt = xgb.DMatrix(Xt)
    P = {}
    for k in (1, 3):
        for t, (fwd, inv) in CONT.items():
            col = f"{t}_{k}"
            for d, src in ((da, a), (db, b)):
                y = src[col].to_numpy(); ok = ~np.isnan(y)
                d.set_label(np.where(ok, fwd(np.nan_to_num(y)), 0).astype("float32")); d.set_weight(ok.astype("float32"))
            bst = xgb.train({**CV_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.5, 0.9]}, da, 1500,
                            evals=[(db, "v")], early_stopping_rounds=40, verbose_eval=False)
            P[col] = inv(np.sort(bst.predict(dt, iteration_range=(0, bst.best_iteration + 1)), axis=1).astype("float64"))
        for t in ("out", "merge"):
            col = f"{t}_{k}"
            for d, src in ((da, a), (db, b)):
                y = src[col].to_numpy("float64"); ok = ~np.isnan(y)          # censored extinction labels: weight 0
                d.set_label(np.nan_to_num(y).astype("float32")); d.set_weight(ok.astype("float32"))
            bst = xgb.train({**CV_PARAMS, "objective": "binary:logistic", "eval_metric": "logloss"}, da, 1500,
                            evals=[(db, "v")], early_stopping_rounds=40, verbose_eval=False)
            P[col] = bst.predict(dt, iteration_range=(0, bst.best_iteration + 1)).astype("float64")
    # merge threshold: the one tuned on 2024 for the production models
    res = {"rows_train": len(a), "rows_test": len(te)}
    for k in (1, 3):
        res[str(k)] = scores(te, P, k, MERGE_THR.get(k, 0.5))
    return res


MERGE_THR = {}


def cv(df):
    feats = feature_columns(df, weather=True)
    MERGE_THR.update(merge_thresholds(boosters("weather"), df, feats))
    res = {}
    only = os.environ.get("PRED_CV_FOLDS", "")                 # e.g. "year 2024,region Africa" to run only those folds
    want = lambda name: not only or name in only.split(",")
    for y in [y for y in (2022, 2023, 2024) if want(f"year {y}")]:
        t0 = time.time()
        res[f"year {y}"] = cv_fold(df, f"year {y}", df.year != y, df.year == y, feats)
        log(f"  cv year {y}: overall d+1 {res[f'year {y}']['1']['overall']} %, d+3 {res[f'year {y}']['3']['overall']} % ({time.time() - t0:.0f} s)")
    for name, ids in [(n, i) for n, i in REGION_GROUPS.items() if want(f"region {n}")]:
        t0 = time.time()
        inr = df.region.isin(ids)
        res[f"region {name}"] = cv_fold(df, name, ~inr & (df.split != "test"), inr & (df.split == "test"), feats)
        log(f"  cv region {name}: overall d+1 {res[f'region {name}']['1']['overall']} %, d+3 {res[f'region {name}']['3']['overall']} % ({time.time() - t0:.0f} s)")
    return res


# ---------------------------------------------------------------------------
def forecast(df):
    from weather_worker import RAIN_MM, THROTTLE, USER_AGENT, cell_latlon
    pool = df[(df.day >= "2024-03-05") & df.w1_tmax.notna() & df.weather_cell.notna() & (df.split != "train")]
    # fire-days that share weather cells: take the busiest cells until ~600 fire-days
    counts = pool.groupby("weather_cell").size().sort_values(ascending=False)
    cells, n = [], 0
    for c, v in counts.items():
        take = min(v, 12)
        cells.append(c); n += take
        if n >= 600:
            break
    sample = pool[pool.weather_cell.isin(cells)].sample(frac=1, random_state=0).groupby("weather_cell").head(12).reset_index(drop=True)
    log(f"  forecast test: {len(sample)} fire-days in {len(cells)} weather cells")
    vars_ = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
             "precipitation", "vapour_pressure_deficit"]
    hours = []
    ll = cell_latlon(pd.Series(cells)).set_index("weather_cell")
    for c in cells:
        s = sample[sample.weather_cell == c]
        start = (pd.Timestamp(s.day.min()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(s.day.max()) + pd.Timedelta(days=H)).strftime("%Y-%m-%d")
        names = [f"{v}_previous_day{k}" for k in range(1, H + 1) for v in vars_]
        span = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        THROTTLE.wait(math.ceil(len(names) / 10) * math.ceil(span / 14))
        for attempt in range(5):
            try:
                r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", headers={"User-Agent": USER_AGENT}, timeout=120,
                                 params={"latitude": ll.lat[c], "longitude": ll.lon[c], "hourly": ",".join(names), "models": "ecmwf_ifs025",
                                         "start_date": start, "end_date": end, "timezone": "GMT"})
                if r.status_code == 200:
                    h = pd.DataFrame(r.json()["hourly"]); h["weather_cell"] = c; hours.append(h)
                    break
                if r.status_code not in (429, 500, 502, 503, 504):
                    log(f"  {c}: {r.status_code} {r.text[:120]}"); break
            except requests.exceptions.RequestException:
                pass
            time.sleep(10 * (attempt + 1))
    Hh = pd.concat(hours, ignore_index=True)
    Hh["day"] = Hh.time.str[:10]
    # daily forecast features for day d+k from the forecast issued k days before (i.e. on day d)
    X = sample.copy()
    replaced = 0
    for k in range(1, H + 1):
        p = f"_previous_day{k}"
        h = Hh[["weather_cell", "day", "time"] + [v + p for v in vars_]].rename(columns={v + p: v for v in vars_})
        for v in vars_:
            h[v] = pd.to_numeric(h[v], errors="coerce")
        rad = np.radians(h.wind_direction_10m)
        h["u"], h["v"] = -h.wind_speed_10m * np.sin(rad), -h.wind_speed_10m * np.cos(rad)
        g = h.groupby(["weather_cell", "day"])
        d = pd.DataFrame({"tmax": g.temperature_2m.max(), "tmean": g.temperature_2m.mean(), "rh_min": g.relative_humidity_2m.min(),
                          "rh_mean": g.relative_humidity_2m.mean(), "vpd_max": g.vapour_pressure_deficit.max(),
                          "wind_mean": g.wind_speed_10m.mean(), "wind_max": g.wind_speed_10m.max(), "gust_max": g.wind_gusts_10m.max(),
                          "u_mean": g.u.mean(), "v_mean": g.v.mean(), "rain": g.precipitation.sum(min_count=1),
                          "rain_hours": (g.precipitation.apply(lambda s: (s >= RAIN_MM).sum()))}).reset_index()
        X["_dk"] = (pd.to_datetime(X.day) + pd.Timedelta(days=k)).dt.strftime("%Y-%m-%d")
        m = X[["weather_cell", "_dk"]].merge(d, left_on=["weather_cell", "_dk"], right_on=["weather_cell", "day"], how="left")
        ok = m.tmax.notna().to_numpy()
        for c in ("tmax", "tmean", "rh_min", "rh_mean", "vpd_max", "wind_mean", "wind_max", "gust_max", "u_mean", "v_mean", "rain"):
            X.loc[ok, f"w{k}_{c}"] = m.loc[ok, c].to_numpy()
        # soil moisture: forecast archive has none, keep day d's; hours since rain: approximate from forecast rain
        for c in ("sm0", "sm1", "sm2"):
            X[f"w{k}_{c}"] = X[f"w0_{c}"]
        X.loc[ok, f"w{k}_since_rain"] = np.where(m.loc[ok, "rain_hours"].to_numpy() > 0, 12.0, X.loc[ok, "w0_since_rain"].to_numpy() + 24 * k)
        replaced += int(ok.sum())
    X = X.drop(columns=["_dk"])
    if "nb_bearing" in X:
        b = np.radians(X.nb_bearing.to_numpy())
        for k in range(1, H + 1):
            X[f"w{k}_wind_to_nb"] = X[f"w{k}_u_mean"].to_numpy() * np.sin(b) + X[f"w{k}_v_mean"].to_numpy() * np.cos(b)
    full = np.ones(len(X), bool)
    for k in range(1, H + 1):
        full &= X[f"w{k}_tmax"].ne(sample[f"w{k}_tmax"]).to_numpy() | sample[f"w{k}_tmax"].isna().to_numpy()
    mw, mn = boosters("weather"), boosters("no_weather")
    fw, fn = feature_columns(df, True), feature_columns(df, False)
    thr = merge_thresholds(mw, df, fw)
    thr_n = merge_thresholds(mn, df, fn)
    Po, Pf, Pn = predict(mw, sample, fw), predict(mw, X, fw), predict(mn, sample, fn)
    res = {"fire_days": len(sample), "weather_cells": len(cells), "forecast_rows_replaced": replaced, "days": {}}
    for k in range(1, H + 1):
        res["days"][str(k)] = {"observed weather (training setup)": scores(sample, Po, k, thr[k]),
                               "issued forecast (real use)": scores(sample, Pf, k, thr[k]),
                               "no weather": scores(sample, Pn, k, thr_n[k]),
                               "persistence": scores(sample, None, k, kind="persistence")}
    return res


# ---------------------------------------------------------------------------
def save(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "blend" if os.environ.get("PRED_BLEND") else os.environ.get("PRED_MODELS", "models")
    name = name if tag == "models" else f"{name}_{tag}"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    df = load()
    log(f"{len(df):,} fire-days loaded")
    for name, fn in (("heldout", heldout), ("behaviour", behaviour), ("cv", cv), ("forecast", forecast)):
        if which in (name, "all"):
            t1 = time.time()
            log(f"{name} ...")
            res = fn(df)
            save(name, res)
            log(f"{name} done in {time.time() - t1:.0f} s -> {OUT / (name + '.json')}")
    log(f"all done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
