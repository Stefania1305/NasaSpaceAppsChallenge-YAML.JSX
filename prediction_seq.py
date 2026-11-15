"""Sequence model of the prediction engine (full version), PyTorch on the GPU.

One network predicts every output for days +1, +2, +3 at once:
  growth, frp, frp_peak, dx, dy    low / likely / high (10th / 50th / 90th percentile, pinball loss)
  out, merge                       probability
from
  history     the fire's previous (up to) 4 fire-days: size, growth, power, passes, clusters, separate parts,
              movement, merges, days back (a GRU reads them in order)
  now         every feature of the fire-day table (fire state, place, season, neighbours, weather d..d+3)
Trained for real use:
  weather dropout   the weather is hidden for 40 % of the rows (and missing for rows without it), so one model
                    serves both "with weather" and "no weather"
  forecast noise    a real forecast error vector (forecast_residuals.csv.gz: same day ahead, same region group)
                    is added to the weather of days +1..+3 for every row, every time it is seen, so the model
                    learns to trust the weather as much as a forecast deserves
  soil moisture     days +1..+3 use day d's value, as the engine does (the forecast archive has none)

Usage:  python prediction_seq.py [--epochs 14] [--full]   (--full: train on all years for production)
Output: data/prediction/models/seq/model.pt, prep.json, card.json
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from firms import DATA_DIR
from forecast_errors import GROUPS, NOISY, group_of
from prediction_data import H, WX, feature_columns

OUT = DATA_DIR / "prediction" / "models" / "seq"
QUANT = torch.tensor([0.1, 0.5, 0.9])
CONT = ["growth", "frp", "frp_peak", "dx", "dy"]
BIN = ["out", "merge"]
HIST = 4
DYN = ["fine_fp", "new_area", "frp", "frp_peak", "pixels", "passes", "clusters_alive", "front", "sep_parts", "sep_gap_km",
       "dx_l1", "dy_l1", "merge"]
LOG = {"fine_fp", "all_fp", "frp", "frp_l1", "frp_peak", "pixels", "passes", "clusters_alive", "clusters_seen", "front",
       "nb_dist_km", "nb_gap_km", "nb_fp", "nb_frp", "nb_n10", "nb_n25", "sep_parts", "sep_gap_km", "risk_days", "age_days",
       "merges_so_far"}
ASINH = {"new_area", "new_area_l1", "new_area_l2", "dx_l1", "dy_l1", "dx_l2", "dy_l2", "nb_new_area"}
WEATHER_DROP = 0.4


def col_kind(c):
    if c in LOG or c.endswith(("_rain", "_since_rain")):
        return "log"
    if c in ASINH:
        return "asinh"
    return "lin"


def fwd_target(t, y):
    return np.log1p(np.clip(y, 0, None)) if t in ("growth", "frp", "frp_peak") else np.arcsinh(y)


def inv_target(t, y):
    return np.expm1(y).clip(min=0) if t in ("growth", "frp", "frp_peak") else np.sinh(y)


def transform(c, v):
    k = col_kind(c)
    if k == "log":
        return np.log1p(np.clip(v, 0, None))
    if k == "asinh":
        return np.arcsinh(v)
    return v


class Prep:
    """Feature preparation shared by training and the engine."""

    def __init__(self, feats, stats=None):
        self.feats = feats
        self.wcols = [c for c in feats if c[:1] == "w" and c[1:2].isdigit()]
        self.stats = stats or {}

    def fit(self, df):
        for c in self.feats + DYN:
            v = transform(c, df[c].to_numpy("float64"))
            m, s = np.nanmean(v), np.nanstd(v)
            self.stats[c] = [float(0 if np.isnan(m) else m), float(1 if not s or np.isnan(s) else s)]
        return self

    def matrix(self, df, soil_from_d=True):
        d = df
        if soil_from_d:
            d = df.copy()
            for k in range(1, H + 1):
                for c in ("sm0", "sm1", "sm2"):
                    if f"w{k}_{c}" in d and f"w0_{c}" in d:
                        d[f"w{k}_{c}"] = d[f"w0_{c}"]
        cols = []
        for c in self.feats:
            m, s = self.stats[c]
            cols.append((transform(c, d[c].to_numpy("float64")) - m) / s)
        X = np.stack(cols, axis=1).astype("float32")
        miss = np.isnan(X)
        X[miss] = 0
        return np.concatenate([X, miss.astype("float32")], axis=1), d

    def history(self, df):
        """Previous up to HIST fire-days of the same fire: [n, HIST, len(DYN)+2] (features, days back, present)."""
        key = df.run.astype(str) + "|" + df.fire_id.astype(str)
        order = np.lexsort((df.day.to_numpy(), key.to_numpy()))
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        s = df.iloc[order]
        k = key.iloc[order].to_numpy()
        day = pd.to_datetime(s.day).to_numpy()
        dyn = np.stack([(transform(c, s[c].to_numpy("float64")) - self.stats[c][0]) / self.stats[c][1] for c in DYN], axis=1)
        dyn = np.nan_to_num(dyn).astype("float32")
        out = np.zeros((len(s), HIST, len(DYN) + 2), "float32")
        for j in range(1, HIST + 1):
            same = np.zeros(len(s), bool)
            same[j:] = k[j:] == k[:-j]
            back = np.zeros(len(s), "float32")
            back[j:] = (day[j:] - day[:-j]).astype("timedelta64[D]").astype("float32")
            ok = same & (back <= 7)
            src = np.roll(dyn, j, axis=0)
            out[ok, HIST - j, :len(DYN)] = src[ok]
            out[ok, HIST - j, len(DYN)] = back[ok] / 7
            out[ok, HIST - j, len(DYN) + 1] = 1
        return out[inv]

    def to_json(self):
        return {"feats": self.feats, "stats": self.stats}

    @classmethod
    def from_json(cls, d):
        return cls(d["feats"], d["stats"])


class Net(nn.Module):
    def __init__(self, n_in, n_dyn, hidden=384):
        super().__init__()
        self.gru = nn.GRU(n_dyn, 64, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(n_in + 64, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
                                 nn.Linear(hidden, 256), nn.GELU())
        self.q = nn.Linear(256, len(CONT) * H * 3)
        self.b = nn.Linear(256, len(BIN) * H)

    def forward(self, x, h):
        _, hn = self.gru(h)
        z = self.mlp(torch.cat([x, hn[-1]], dim=1))
        q = self.q(z).view(-1, len(CONT) * H, 3)
        q = torch.cat([q[..., :1], q[..., :1] + torch.cumsum(nn.functional.softplus(q[..., 1:]), dim=-1)], dim=-1)  # low<=likely<=high
        return q, self.b(z)


def targets(df):
    yc = np.stack([fwd_target(t, df[f"{t}_{k}"].to_numpy("float64")) for t in CONT for k in range(1, H + 1)], axis=1)
    yb = np.stack([df[f"{t}_{k}"].to_numpy("float64") for t in BIN for k in range(1, H + 1)], axis=1)
    return yc.astype("float32"), yb.astype("float32")


class Noise:
    """Adds real forecast errors to the weather of days +1..+3 of a batch (in the original units)."""

    def __init__(self, prep, device):
        R = pd.read_csv(DATA_DIR / "prediction" / "forecast_residuals.csv.gz")
        self.idx = []           # (column position, k, residual column)
        for k in range(1, H + 1):
            for c in NOISY:
                col = f"w{k}_{c}"
                if col in prep.feats:
                    self.idx.append((prep.feats.index(col), k, c))
        self.prep = prep
        self.pools = {}
        for k in range(1, H + 1):
            for gname in list(GROUPS) + ["all"]:
                r = R[(R.k == k) & ((R.group == gname) | (gname == "all"))][NOISY].fillna(0).to_numpy("float32")
                if len(r) >= 50:
                    self.pools[(k, gname)] = torch.tensor(r, device=device)
        self.cpos = {c: i for i, c in enumerate(NOISY)}
        self.device = device

    def apply(self, xb, raw, grp):
        """xb: prepared batch (modified copy), raw: [b, len(idx)] original values, grp: list of group ids per row."""
        xb = xb.clone()
        n = xb.shape[0]
        for k in range(1, H + 1):
            draws = torch.zeros((n, len(NOISY)), device=self.device)
            for gi, gname in enumerate(GROUPS):
                m = grp == gi
                if not m.any():
                    continue
                pool = self.pools.get((k, gname), self.pools[(k, "all")])
                draws[m] = pool[torch.randint(len(pool), (int(m.sum()),), device=self.device)]
            for j, (pos, kk, c) in enumerate(self.idx):
                if kk != k:
                    continue
                v = raw[:, j] + draws[:, self.cpos[c]]
                if c.startswith("rh"):
                    v = v.clamp(1, 100)
                elif c in ("rain", "wind_mean", "wind_max", "gust_max", "vpd_max"):
                    v = v.clamp(min=0)
                if col_kind(f"w{k}_{c}") == "log":
                    v = torch.log1p(v)
                m_, s_ = self.prep.stats[f"w{k}_{c}"]
                ok = ~torch.isnan(raw[:, j])
                xb[ok, pos] = ((v - m_) / s_)[ok]
        return xb


def pinball(q, y, w):
    ok = ~torch.isnan(y)
    yy = torch.nan_to_num(y)[..., None]
    e = yy - q
    loss = torch.maximum(QUANT.to(q.device) * e, (QUANT.to(q.device) - 1) * e).mean(-1)
    return (loss * ok * w).sum() / ok.sum().clamp(min=1)


def run_epochs(net, tensors, epochs, noise, wmask_cols, lr=2e-3, bs=2048, val=None, log=print):
    X, Hh, YC, YB, RAW, G = tensors
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    wcont = 1 / torch.nan_to_num(YC, nan=0).std(0).clamp(min=0.05)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best = (float("inf"), None, 0)
    nf = X.shape[1] // 2
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=X.device)
        tot = 0.0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            xb = noise.apply(X[b], RAW[b], G[b]) if noise else X[b]
            drop = torch.rand(len(b), device=X.device) < WEATHER_DROP
            if drop.any():
                sel = drop[:, None] & wmask_cols[None, :nf].bool()          # hide all weather of these rows
                xb[:, :nf] = xb[:, :nf].masked_fill(sel, 0)
                xb[:, nf:] = torch.where(sel, torch.ones_like(xb[:, nf:]), xb[:, nf:])
            q, logit = net(xb, Hh[b])
            yb = YB[b]
            okb = ~torch.isnan(yb)
            loss = pinball(q, YC[b], wcont) + 0.5 * (bce(logit, torch.nan_to_num(yb)) * okb).sum() / okb.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss) * len(b)
        msg = f"  epoch {ep + 1}/{epochs}: train loss {tot / len(X):.4f}"
        if val is not None:
            vl = evaluate_loss(net, val, wcont)
            msg += f", check loss {vl:.4f}"
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep + 1)
        log(msg)
    if val is not None and best[1] is not None:
        net.load_state_dict(best[1])
    return best[2] if val is not None else epochs


@torch.no_grad()
def evaluate_loss(net, val, wcont, bs=8192):
    net.eval()
    X, Hh, YC, YB = val
    bce = nn.BCEWithLogitsLoss(reduction="none")
    tot = n = 0.0
    for i in range(0, len(X), bs):
        q, lg = net(X[i:i + bs], Hh[i:i + bs])
        yb = YB[i:i + bs]; ok = ~torch.isnan(yb)
        l = pinball(q, YC[i:i + bs], wcont) + 0.5 * (bce(lg, torch.nan_to_num(yb)) * ok).sum() / ok.sum().clamp(min=1)
        tot += float(l) * len(q); n += len(q)
    return tot / n


class SeqModel:
    """Trained network + preparation; predicts in the same format as the XGBoost models."""

    def __init__(self, net, prep, device):
        self.net, self.prep, self.device = net.to(device).eval(), prep, device

    @classmethod
    def load(cls, d=OUT, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        prep = Prep.from_json(json.loads((d / "prep.json").read_text(encoding="utf-8")))
        net = Net(len(prep.feats) * 2, len(DYN) + 2)
        net.load_state_dict(torch.load(d / "model.pt", map_location=device))
        return cls(net, prep, device)

    @torch.no_grad()
    def predict_df(self, df, weather=True, history=None):
        X, _ = self.prep.matrix(df.reindex(columns=list(dict.fromkeys(self.prep.feats + ["run", "fire_id", "day"]))))
        if not weather:
            nf = len(self.prep.feats)
            for i, c in enumerate(self.prep.feats):
                if c in self.prep.wcols:
                    X[:, i] = 0; X[:, nf + i] = 1
        Hh = history if history is not None else self.prep.history(df.reindex(columns=list(dict.fromkeys(DYN + ["run", "fire_id", "day"]))))
        qs, ps = [], []
        for i in range(0, len(X), 8192):
            q, lg = self.net(torch.tensor(X[i:i + 8192], device=self.device), torch.tensor(Hh[i:i + 8192], device=self.device))
            qs.append(q.cpu().numpy()); ps.append(torch.sigmoid(lg).cpu().numpy())
        q, p = np.concatenate(qs), np.concatenate(ps)
        P = {}
        for ti, t in enumerate(CONT):
            for k in range(1, H + 1):
                P[f"{t}_{k}"] = inv_target(t, q[:, ti * H + (k - 1), :].astype("float64"))
        for ti, t in enumerate(BIN):
            for k in range(1, H + 1):
                P[f"{t}_{k}"] = p[:, ti * H + (k - 1)].astype("float64")
        return P


def tensors_for(prep, df, device, noise_cols):
    X, d = prep.matrix(df)
    Hh = prep.history(df)
    YC, YB = targets(df)
    RAW = np.stack([d[f"w{k}_{c}"].to_numpy("float32") for (_, k, c) in noise_cols], axis=1) if noise_cols else np.zeros((len(df), 0), "float32")
    G = df.region.map(lambda r: list(GROUPS).index(group_of(r))).to_numpy()
    T = lambda a: torch.tensor(a, device=device)
    return T(X), T(Hh), T(YC), T(YB), T(RAW), T(G)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--full", action="store_true", help="train on all years (production); epochs from the checked run")
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(DATA_DIR / "prediction" / "fire_days.csv.gz", low_memory=False)
    feats = feature_columns(df, weather=True)
    tr = df if args.full else df[df.split == "train"]
    prep = Prep(feats).fit(tr)
    noise = None if args.no_noise else Noise(prep, dev)
    ncols = noise.idx if noise else []
    print(f"{len(df):,} fire-days, {len(feats)} features, device {dev}, forecast noise {'on' if noise else 'off'}", flush=True)
    T = tensors_for(prep, tr.reset_index(drop=True), dev, ncols)
    wmask = torch.tensor([1.0 if c in prep.wcols else 0.0 for c in feats] * 2, device=dev)
    net = Net(len(feats) * 2, len(DYN) + 2).to(dev)
    val = None
    if not args.full:
        va = df[df.split == "check"].reset_index(drop=True)
        V = tensors_for(prep, va, dev, ncols)
        val = (V[0], V[1], V[2], V[3])
    best_ep = run_epochs(net, T, args.epochs, noise, wmask, val=val, log=lambda m: print(m, flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT / "model.pt")
    (OUT / "prep.json").write_text(json.dumps(prep.to_json()), encoding="utf-8")
    card = {"rows_train": len(tr), "features": len(feats), "epochs": args.epochs, "best_epoch": best_ep, "full": args.full,
            "forecast_noise": noise is not None, "weather_dropout": WEATHER_DROP, "seconds": round(time.time() - t0, 1)}
    (OUT / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
