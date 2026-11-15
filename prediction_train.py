"""Prediction engine, step A5-A6: train and evaluate the models on the fire-day table.

For every output and every day ahead (k = 1, 2, 3) it trains gradient-boosting models (XGBoost on the NVIDIA GPU
through CUDA; one model predicts low, likely and high together):
  growth_k        footprint growth, km2                 low / likely / high (10th / 50th / 90th percentile)
  frp_k           mean VIIRS pass FRP on day d+k, MW    low / likely / high
  frp_peak_k      peak VIIRS pass FRP on day d+k, MW    low / likely / high
  dx_k, dy_k      fire-centre movement east / north km  low / likely / high  -> direction and spread rate
  out_k           chance the fire is out by day d+k
  merge_k         chance it merges with a separate neighbouring fire by day d+k
twice: with the weather features and without them. Split by time: train on 2016-2023, stop training
on 2024 ("check"), report on 2025-2026 ("test", never seen in training).

Scores on the test years, against the persistence baseline ("it keeps doing what it did today"):
  continuous   mean absolute error of the likely value, share of real values inside low..high (aim ~80%)
  direction    angle between predicted and real movement, for fires that moved at least 1 km
  chances      AUC (0.5 = guessing, 1 = perfect) and Brier score (lower is better)

Usage:  python prediction_train.py [--device cuda|cpu]
Output: data/prediction/models/<variant>/<target>.json, data/prediction/model_card.json
"""

import argparse
import json
import math
import time

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from firms import DATA_DIR
from prediction_data import H, OUT, feature_columns

QUANTILES = (0.1, 0.5, 0.9)
CONT = {  # target -> (transform, inverse)
    "growth": (np.log1p, np.expm1), "frp": (np.log1p, np.expm1), "frp_peak": (np.log1p, np.expm1),
    "dx": (np.arcsinh, np.sinh), "dy": (np.arcsinh, np.sinh)}
BINARY = ["out", "merge"]
PARAMS = dict(n_estimators=1500, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", early_stopping_rounds=60)


def fit(kind, Xtr, ytr, Xva, yva, device):
    if kind == "quantile":
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES), device=device, **PARAMS)
    else:
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", device=device, **PARAMS)
    m.fit(Xtr.astype("float32"), ytr, eval_set=[(Xva.astype("float32"), yva)], verbose=False)
    return m


def persistence(df, target, k):
    if target == "growth":
        return (df.new_area.fillna(0).clip(lower=0) * k).to_numpy()
    if target in ("frp", "frp_peak"):
        return df[target].to_numpy()
    if target == "dx":
        return (df.dx_l1.fillna(0) * k).to_numpy()
    if target == "dy":
        return (df.dy_l1.fillna(0) * k).to_numpy()


def train_variant(df, variant, device, models_dir, full=False):
    feats = feature_columns(df, weather=(variant == "weather"))
    if full:
        # all years: 5 % of the fires (random, whole fires) only decide when to stop; scores below are on them
        fires = df[["run", "fire_id"]].drop_duplicates().sample(frac=0.05, random_state=0)
        hold = df.set_index(["run", "fire_id"]).index.isin(fires.set_index(["run", "fire_id"]).index)
        tr, va, te = df[~hold], df[hold], df[hold]
    else:
        tr, va, te = (df[df.split == s] for s in ("train", "check", "test"))
    card, preds = {"features": feats, "targets": {}}, {}
    for k in range(1, H + 1):
        for target, (fwd, inv) in CONT.items():
            col = f"{target}_{k}"
            ok = lambda d: d[d[col].notna()]
            a, b, c = ok(tr), ok(va), ok(te)
            if len(a) < 200 or len(b) < 50 or len(c) < 50:
                continue
            m = fit("quantile", a[feats], fwd(a[col]), b[feats], fwd(b[col]), device)
            m.save_model(str(models_dir / f"{col}.json"))
            qs = inv(np.sort(m.predict(c[feats].astype("float32")), axis=1))      # sorted: low <= likely <= high
            lo, mid, hi = qs[:, 0], qs[:, 1], qs[:, 2]
            y, base = c[col].to_numpy(), persistence(c, target, k)
            okb = ~np.isnan(base)
            card["targets"][col] = {
                "rows_train": len(a), "rows_test": len(c),
                "mae_model": round(float(np.mean(np.abs(mid - y))), 3),
                "mae_persistence": round(float(np.mean(np.abs(base[okb] - y[okb]))), 3) if okb.any() else None,
                "inside_low_high": round(float(np.mean((y >= lo) & (y <= hi))), 3)}
            preds[col] = mid
            preds[f"_{col}_index"] = c.index
        # direction from the movement models
        if f"dx_{k}" in preds and f"dy_{k}" in preds:
            c = te.loc[preds[f"_dx_{k}_index"]]
            dxp, dyp = preds[f"dx_{k}"], preds[f"dy_{k}"]
            dyp = pd.Series(dyp, index=preds[f"_dy_{k}_index"]).reindex(c.index).to_numpy()
            moved = np.hypot(c[f"dx_{k}"], c[f"dy_{k}"]) >= 1.0
            ang = lambda x, y: np.degrees(np.arctan2(x, y))
            err = lambda p: np.abs((p - ang(c[f"dx_{k}"], c[f"dy_{k}"]) + 180) % 360 - 180)[moved]
            pers = np.hypot(c.dx_l1.fillna(0), c.dy_l1.fillna(0)) > 0
            card["targets"][f"direction_{k}"] = {
                "fires_moved_1km": int(moved.sum()),
                "median_angle_error_model_deg": round(float(np.nanmedian(err(ang(dxp, dyp)))), 1),
                "median_angle_error_persistence_deg": round(float(np.nanmedian(err(ang(c.dx_l1.fillna(0), c.dy_l1.fillna(0)))[pers[moved]])), 1)
                if (pers & moved).any() else None}
        for target in BINARY:
            col = f"{target}_{k}"
            a, b, c = tr, va, te
            if a[col].nunique() < 2 or b[col].nunique() < 2 or c[col].nunique() < 2:
                continue
            m = fit("binary", a[feats], a[col], b[feats], b[col], device)
            m.save_model(str(models_dir / f"{col}.json"))
            p = m.predict_proba(c[feats].astype("float32"))[:, 1]
            base = np.full(len(c), a[col].mean())
            card["targets"][col] = {
                "rows_train": len(a), "rows_test": len(c), "rate_test": round(float(c[col].mean()), 4),
                "auc": round(float(roc_auc_score(c[col], p)), 3),
                "brier_model": round(float(brier_score_loss(c[col], p)), 4),
                "brier_base_rate": round(float(brier_score_loss(c[col], base)), 4)}
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--full", action="store_true", help="train on all years (2016-2026); no unseen test years left")
    ap.add_argument("--noise", action="store_true", help="train on forecast-like weather (real forecast errors added)")
    ap.add_argument("--out", default="models", help="model folder under data/prediction")
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(OUT / "fire_days.csv.gz", low_memory=False)
    from prediction_data import add_history
    df = add_history(df)
    if args.noise:
        from forecast_errors import forecast_like
        # training rows get forecast-like weather; check and test rows keep soil moisture from day d only
        df = forecast_like(df, rows=(df.split == "train").to_numpy() | args.full, seed=0)
    print(f"{len(df):,} fire-days: " + ", ".join(f"{k} {v:,}" for k, v in df.split.value_counts().items()), flush=True)
    card = {"rows": len(df), "split": {k: int(v) for k, v in df.split.value_counts().items()}, "horizon_days": H,
            "quantiles": QUANTILES, "variants": {}}
    for variant in ("weather", "no_weather"):
        d = OUT / args.out / variant
        d.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        card["variants"][variant] = train_variant(df, variant, args.device, d, full=args.full)
        card["variants"][variant]["seconds"] = round(time.time() - t1, 1)
        print(f"  {variant}: {len(card['variants'][variant]['targets'])} targets in {time.time() - t1:.0f} s", flush=True)
    card["seconds"] = round(time.time() - t0, 1)
    card["device"] = args.device
    card["full"] = args.full
    (OUT / ("model_card.json" if args.out == "models" else f"model_card_{args.out}.json")).write_text(json.dumps(card, indent=1), encoding="utf-8")
    w, n = card["variants"]["weather"]["targets"], card["variants"]["no_weather"]["targets"]
    print("\nTest years 2025-2026 (likely value; persistence = keeps doing what it did today):")
    print(f"  {'output':<14} {'with weather':>13} {'no weather':>11} {'persistence':>12} {'inside low..high':>17}")
    for col, v in w.items():
        if "mae_model" in v:
            print(f"  {col:<14} {v['mae_model']:>13} {n.get(col, {}).get('mae_model', '-'):>11} {str(v['mae_persistence']):>12} {v['inside_low_high']:>17}")
    for col, v in w.items():
        if "auc" in v:
            print(f"  {col:<14} AUC {v['auc']} (no weather {n.get(col, {}).get('auc', '-')}), Brier {v['brier_model']} vs base rate {v['brier_base_rate']}, rate {v['rate_test']}")
        if "median_angle_error_model_deg" in v:
            print(f"  {col:<14} median angle error {v['median_angle_error_model_deg']} deg (no weather {n.get(col, {}).get('median_angle_error_model_deg', '-')}, persistence {v['median_angle_error_persistence_deg']}) over {v['fires_moved_1km']} fires")
    print(f"\nTrained in {card['seconds']} s on {args.device}; saved to {OUT / 'models'}")


if __name__ == "__main__":
    main()
