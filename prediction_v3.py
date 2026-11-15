"""Prediction models v3: trained on fire chains, with a survival model for extinction and a two-part growth model.

Trained on fire_days_v3.csv.gz (prediction_data.py --chains), where fires that vanish for up to 3 days and come
back are one continuing fire. For every day ahead k = 1, 2, 3 and each variant (with / without weather):

  extinction   survival model (XGBoost survival:aft, accelerated failure time, normal distribution): the time until
               the fire is out, learned from fires whose end was seen (interval: between day T-1 and T) and from
               fires still burning when the data ended (censored: at least T days). Chance out by day d+k = the
               model's probability that the time is at most k. A plain yes/no classifier per day is trained too
               for comparison (out_k).
  growth       two parts: grow_k, the chance the fire grows by at least 0.1 km2 that day, and amount_k, how much
               (10th / 50th / 90th percentile) when it does. Combined into the usual low / likely / high growth
               (the mixture's percentiles), plus the expected growth.
  fire power, movement, merging   as in v2: frp_k, frp_peak_k, dx_k, dy_k quantile models, merge_k classifier.

All on forecast-like weather (real forecast errors added to the training rows, day d's soil moisture on days
+1..+3), at learning rate 0.05 with early stopping on 2024 (or on 5 % of the fires with --full).

Usage:  python prediction_v3.py [--full] [--variant weather|no_weather] [--out models_v3]
In code: predict(models_dir, X, variant) -> dict in the usual format (+ grow_k, growth_mean_k)
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from firms import DATA_DIR
from prediction_data import H, add_history, feature_columns

P_DIR = DATA_DIR / "prediction"
GROW_KM2 = 0.1
QUANT = (0.1, 0.5, 0.9)
Q_TARGETS = {"frp": (np.log1p, np.expm1), "frp_peak": (np.log1p, np.expm1), "dx": (np.arcsinh, np.sinh), "dy": (np.arcsinh, np.sinh),
             "amount": (np.log1p, np.expm1)}
PARAMS = dict(tree_method="hist", eta=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
AFT_SCALES = (0.6, 0.9, 1.3)


def load_table(table="fire_days_v3.csv.gz"):
    df = pd.read_csv(P_DIR / table, low_memory=False)
    df = add_history(df)
    for k in range(1, H + 1):
        df[f"grow_{k}"] = np.where(df[f"growth_{k}"].isna(), np.nan, (df[f"growth_{k}"] >= GROW_KM2).astype(float))
        df[f"amount_{k}"] = np.where(df[f"growth_{k}"] >= GROW_KM2, df[f"growth_{k}"], np.nan)
    return df


def split(df, full):
    if full:
        fires = df[["run", "chain"]].drop_duplicates().sample(frac=0.05, random_state=0)
        hold = df.set_index(["run", "chain"]).index.isin(fires.set_index(["run", "chain"]).index)
        return df[~hold], df[hold], df[hold]
    return df[df.split == "train"], df[df.split == "check"], df[df.split == "test"]


def dm(X, feats, label=None, weight=None):
    d = xgb.DMatrix(X[feats].astype("float32"), label=label, weight=weight, feature_names=feats)
    return d


def train_quantile(a, b, feats, col, fwd, device, log):
    ya, yb = a[col].to_numpy("float64"), b[col].to_numpy("float64")
    oka, okb = ~np.isnan(ya), ~np.isnan(yb)
    if oka.sum() < 200 or okb.sum() < 50:
        return None
    da, db = dm(a[oka], feats, fwd(ya[oka])), dm(b[okb], feats, fwd(yb[okb]))
    m = xgb.train({**PARAMS, "device": device, "objective": "reg:quantileerror", "quantile_alpha": list(QUANT)}, da, 4000,
                  evals=[(db, "check")], early_stopping_rounds=80, verbose_eval=False)
    log(f"    {col}: {m.best_iteration + 1} trees")
    return m


def train_binary(a, b, feats, col, device, log):
    ya, yb = a[col].to_numpy("float64"), b[col].to_numpy("float64")
    oka, okb = ~np.isnan(ya), ~np.isnan(yb)
    if len(np.unique(ya[oka])) < 2:
        return None
    da, db = dm(a[oka], feats, ya[oka]), dm(b[okb], feats, yb[okb])
    m = xgb.train({**PARAMS, "device": device, "objective": "binary:logistic", "eval_metric": "logloss"}, da, 4000,
                  evals=[(db, "check")], early_stopping_rounds=80, verbose_eval=False)
    p = m.predict(db, iteration_range=(0, m.best_iteration + 1))
    log(f"    {col}: {m.best_iteration + 1} trees, check AUC {roc_auc_score(yb[okb], p):.3f}")
    return m


def surv_bounds(d):
    t = d.surv_t.to_numpy("float64")
    obs = d.surv_observed.to_numpy() == 1
    lo = np.where(obs, np.maximum(t - 1, 0.0), t)          # seen to end: out between day T-1 and T after d
    hi = np.where(obs, t, np.inf)                           # still burning when the data ended: at least T days
    return lo, hi


def surv_dm(d, feats):
    x = dm(d, feats)
    lo, hi = surv_bounds(d)
    x.set_float_info("label_lower_bound", lo)
    x.set_float_info("label_upper_bound", hi)
    return x


def surv_prob(pred_time, scale, k):
    """P(time to extinction <= k days) from the AFT model's predicted time (log-normal)."""
    return norm.cdf((math.log(k) - np.log(np.maximum(pred_time, 1e-6))) / scale)


def train_survival(a, b, feats, device, log):
    best = None
    ok = lambda d: d[d.surv_t.notna()]
    a, b = ok(a), ok(b)
    da, db = surv_dm(a, feats), surv_dm(b, feats)
    for sc in AFT_SCALES:
        m = xgb.train({**PARAMS, "device": device, "objective": "survival:aft", "eval_metric": "aft-nloglik",
                       "aft_loss_distribution": "normal", "aft_loss_distribution_scale": sc}, da, 4000,
                      evals=[(db, "check")], early_stopping_rounds=80, verbose_eval=False)
        pt = m.predict(db, iteration_range=(0, m.best_iteration + 1))
        # choose the scale by how well P(out by day 1..3) ranks and fits the check year's real outcomes
        score = 0
        for k in range(1, H + 1):
            y = b[f"out_{k}"].to_numpy("float64")
            okk = ~np.isnan(y)
            p = surv_prob(pt[okk], sc, k)
            score += np.mean((p - y[okk]) ** 2)
        log(f"    survival scale {sc}: {m.best_iteration + 1} trees, check Brier (days 1-3) {score / H:.4f}")
        if best is None or score < best[0]:
            best = (score, m, sc, pt)
    # calibration per day ahead: survival chance -> observed share out, fitted on the stopping data
    calib = {}
    for k in range(1, H + 1):
        y = b[f"out_{k}"].to_numpy("float64")
        okk = ~np.isnan(y)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(surv_prob(best[3][okk], best[2], k), y[okk])
        calib[str(k)] = {"x": iso.X_thresholds_.round(5).tolist(), "y": iso.y_thresholds_.round(5).tolist()}
    return best[1], best[2], calib


def train(df, variant, full, device, out_dir, log=print):
    feats = feature_columns(df, weather=(variant == "weather"))
    a, b, _ = split(df, full)
    out_dir.mkdir(parents=True, exist_ok=True)
    from prediction_data import STATIC_COLS
    card = {"variant": variant, "features": len(feats), "full": full, "rows_train": len(a), "models": {},
            "static_features": any(c in feats for c in STATIC_COLS)}
    t0 = time.time()
    m, sc, calib = train_survival(a, b, feats, device, log)
    m.save_model(str(out_dir / "survival.json"))
    card["survival_scale"] = sc
    card["survival_calibration"] = calib
    for k in range(1, H + 1):
        for col in (f"grow_{k}", f"out_{k}", f"merge_{k}"):
            m = train_binary(a, b, feats, col, device, log)
            if m is not None:
                m.save_model(str(out_dir / f"{col}.json"))
        for t, (fwd, _) in Q_TARGETS.items():
            m = train_quantile(a, b, feats, f"{t}_{k}", fwd, device, log)
            if m is not None:
                m.save_model(str(out_dir / f"{t}_{k}.json"))
    card["seconds"] = round(time.time() - t0, 1)
    (out_dir / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    return card


# ---------------------------------------------------------------------------
def load(models_dir, variant):
    d = Path(models_dir) / variant
    ms = {}
    for p in d.glob("*.json"):
        if p.name == "card.json":
            continue
        b = xgb.Booster()
        b.load_model(str(p))
        ms[p.stem] = b
    if not ms:                                   # models shipped in the repository: compressed binary (Git LFS)
        import gzip
        for p in d.glob("*.ubj.gz"):
            data = p.read_bytes()
            if data.startswith(b"version https://git-lfs"):
                raise SystemExit(f"{p} is a Git LFS pointer: run 'git lfs pull' to download the models")
            b = xgb.Booster()
            b.load_model(bytearray(gzip.decompress(data)))
            ms[p.name[:-len(".ubj.gz")]] = b
    card = json.loads((d / "card.json").read_text(encoding="utf-8"))
    return ms, card


def _interp_quantile(levels, values, r):
    """Conditional quantile at level r from the 10/50/90 % values (linear in level, flat outside)."""
    r = np.clip(r, levels[0], levels[-1])
    out = np.empty_like(r)
    lo = r <= levels[1]
    w = (r - levels[0]) / (levels[1] - levels[0])
    out[lo] = (values[:, 0] + w * (values[:, 1] - values[:, 0]))[lo]
    w2 = (r - levels[1]) / (levels[2] - levels[1])
    out[~lo] = (values[:, 1] + w2 * (values[:, 2] - values[:, 1]))[~lo]
    return out


def predict(models_dir, X, variant="weather", models=None):
    """Usual output dict: growth_k (low, likely, high), frp_k, frp_peak_k, dx_k, dy_k, out_k, merge_k, plus grow_k
    (chance it grows >= 0.1 km2) and growth_mean_k (expected growth). Extinction from the survival model."""
    ms, card = models or load(models_dir, variant)
    feats = next(iter(ms.values())).feature_names
    M = xgb.DMatrix(X.reindex(columns=feats).astype("float32"), feature_names=feats)
    P = {}
    pt = ms["survival"].predict(M)
    for k in range(1, H + 1):
        raw = surv_prob(pt, card["survival_scale"], k)
        cal = card.get("survival_calibration", {}).get(str(k))
        P[f"out_{k}"] = np.interp(raw, cal["x"], cal["y"]) if cal else raw
        P[f"out_clf_{k}"] = ms[f"out_{k}"].predict(M) if f"out_{k}" in ms else np.full(len(X), np.nan)
        P[f"merge_{k}"] = ms[f"merge_{k}"].predict(M)
        for t in ("frp", "frp_peak", "dx", "dy"):
            P[f"{t}_{k}"] = Q_TARGETS[t][1](np.sort(ms[f"{t}_{k}"].predict(M), axis=1).astype("float64"))
        p = ms[f"grow_{k}"].predict(M).astype("float64")
        lt = np.sort(ms[f"amount_{k}"].predict(M), axis=1).astype("float64")       # log1p scale, conditional on growing
        amt = np.expm1(lt).clip(min=GROW_KM2)
        q = np.zeros((len(X), 3))
        for j, lev in enumerate(QUANT):
            grows = lev > 1 - p
            r = np.where(grows, (lev - (1 - p)) / np.maximum(p, 1e-9), 0.1)
            val = np.expm1(_interp_quantile(QUANT, lt, r)).clip(min=GROW_KM2)
            q[:, j] = np.where(grows, val, 0.0)
        P[f"growth_{k}"] = np.sort(q, axis=1)
        sig = np.maximum((lt[:, 2] - lt[:, 0]) / (2 * 1.2816), 1e-3)            # log-normal fit to the 10-90 % range
        gm = p * (np.exp(lt[:, 1] + sig ** 2 / 2) - 1)
        cal = card.get("growth_mean_calibration", {}).get(str(k))
        if cal:                                   # fitted on unseen data: model expected growth -> real average growth
            gm = np.expm1(np.interp(np.log1p(gm), cal["x"], cal["y"]))
            gm = gm * np.interp(gm, cal["ratio_at"], cal["ratio"])
        P[f"growth_mean_{k}"] = gm
        P[f"grow_{k}"] = p
    for k in range(2, H + 1):                     # chances of being out never fall over the days
        P[f"out_{k}"] = np.maximum(P[f"out_{k}"], P[f"out_{k - 1}"])
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--variant", default="weather", choices=["weather", "no_weather"])
    ap.add_argument("--out", default="models_v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table", default="fire_days_v3.csv.gz", help="fire_days_v4.csv.gz adds terrain and land cover")
    args = ap.parse_args()
    t0 = time.time()
    df = load_table(args.table)
    from forecast_errors import forecast_like
    df = forecast_like(df, rows=(df.split == "train").to_numpy() | args.full, seed=0)
    print(f"{len(df):,} fire-days, variant {args.variant}, {'all years' if args.full else 'train 2016-2023'}", flush=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    card = train(df, args.variant, args.full, args.device, P_DIR / args.out / args.variant, log)
    print(json.dumps({k: v for k, v in card.items() if k != "models"}), flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
