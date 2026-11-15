"""The sequence model of the prediction engine, run with numpy only (no PyTorch needed).

prediction_seq.py trains the network with PyTorch on the GPU and saves its weights as plain arrays (weights.npz) next
to the feature preparation (prep.json). This module holds that preparation and the network's arithmetic, a GRU over
the fire's previous fire-days followed by dense layers, so the engine and the final app can predict on any machine.
Its results match the PyTorch network (checked by prediction_seq.py when it saves a model).

Usage in code:
    from seq_numpy import NumpySeq
    m = NumpySeq.load(folder)                 # folder with weights.npz and prep.json
    P = m.predict_df(fire_days, weather=True) # same output format as the XGBoost models
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from prediction_data import H

CONT = ["growth", "frp", "frp_peak", "dx", "dy"]
BIN = ["out", "merge"]
HIST = 4
DYN = ["fine_fp", "new_area", "frp", "frp_peak", "pixels", "passes", "clusters_alive", "front", "sep_parts", "sep_gap_km",
       "dx_l1", "dy_l1", "merge"]
LOG = {"fine_fp", "all_fp", "frp", "frp_l1", "frp_peak", "pixels", "passes", "clusters_alive", "clusters_seen", "front",
       "nb_dist_km", "nb_gap_km", "nb_fp", "nb_frp", "nb_n10", "nb_n25", "sep_parts", "sep_gap_km", "risk_days", "age_days",
       "merges_so_far"}
ASINH = {"new_area", "new_area_l1", "new_area_l2", "dx_l1", "dy_l1", "dx_l2", "dy_l2", "nb_new_area"}


def col_kind(c):
    if c in LOG or c.endswith(("_rain", "_since_rain")):
        return "log"
    if c in ASINH:
        return "asinh"
    return "lin"


def transform(c, v):
    k = col_kind(c)
    if k == "log":
        return np.log1p(np.clip(v, 0, None))
    if k == "asinh":
        return np.arcsinh(v)
    return v


def fwd_target(t, y):
    return np.log1p(np.clip(y, 0, None)) if t in ("growth", "frp", "frp_peak") else np.arcsinh(y)


def inv_target(t, y):
    return np.expm1(y).clip(min=0) if t in ("growth", "frp", "frp_peak") else np.sinh(y)


class Prep:
    """Feature preparation shared by training and prediction."""

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
        """[n, 2 x features]: standardised values (missing = 0) and missing flags. Days +1..+3 take day d's soil
        moisture, as in real use."""
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
        """Previous up to HIST fire-days of the same fire: [n, HIST, len(DYN) + 2] (features, days back, present)."""
        key = df.run.astype(str) + "|" + df.fire_id.astype(str)
        order = np.lexsort((df.day.to_numpy(), key.to_numpy()))
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
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


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _gelu(x):
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softplus(x):
    return np.logaddexp(0, x)


class NumpySeq:
    """The trained network evaluated with numpy."""

    def __init__(self, w: dict, prep: Prep):
        self.w = {k: v.astype("float64") for k, v in w.items()}
        self.prep = prep

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        prep = Prep.from_json(json.loads((folder / "prep.json").read_text(encoding="utf-8")))
        with np.load(folder / "weights.npz") as z:
            w = {k: z[k] for k in z.files}
        return cls(w, prep)

    def forward(self, X, Hh):
        w = self.w
        n, hid = len(X), w["gru.weight_hh_l0"].shape[1]
        h = np.zeros((n, hid))
        Wi, Wh, bi, bh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
        for t in range(Hh.shape[1]):                         # GRU, gates in PyTorch order: reset, update, new
            gi = Hh[:, t, :] @ Wi.T + bi
            gh = h @ Wh.T + bh
            r = _sig(gi[:, :hid] + gh[:, :hid])
            z = _sig(gi[:, hid:2 * hid] + gh[:, hid:2 * hid])
            nn_ = np.tanh(gi[:, 2 * hid:] + r * gh[:, 2 * hid:])
            h = (1 - z) * nn_ + z * h
        a = np.concatenate([X, h], axis=1)
        a = _ln(_gelu(a @ w["mlp.0.weight"].T + w["mlp.0.bias"]), w["mlp.2.weight"], w["mlp.2.bias"])
        a = _ln(_gelu(a @ w["mlp.4.weight"].T + w["mlp.4.bias"]), w["mlp.6.weight"], w["mlp.6.bias"])
        a = _gelu(a @ w["mlp.8.weight"].T + w["mlp.8.bias"])
        q = (a @ w["q.weight"].T + w["q.bias"]).reshape(n, len(CONT) * H, 3)
        q = np.concatenate([q[..., :1], q[..., :1] + np.cumsum(_softplus(q[..., 1:]), axis=-1)], axis=-1)
        p = _sig(a @ w["b.weight"].T + w["b.bias"])
        return q, p

    def predict_df(self, df, weather=True, batch=20000, history=None):
        """history: precomputed [n, HIST, ...] (e.g. from the whole run when df holds only the latest fire-days)."""
        cols = list(dict.fromkeys(self.prep.feats + DYN + ["run", "fire_id", "day"]))
        d = df.reindex(columns=cols)
        X, _ = self.prep.matrix(d)
        if not weather:
            nf = len(self.prep.feats)
            for i, c in enumerate(self.prep.feats):
                if c in self.prep.wcols:
                    X[:, i] = 0
                    X[:, nf + i] = 1
        Hh = self.prep.history(d) if history is None else history
        qs, ps = [], []
        for i in range(0, len(X), batch):
            q, p = self.forward(X[i:i + batch].astype("float64"), Hh[i:i + batch].astype("float64"))
            qs.append(q); ps.append(p)
        q, p = np.concatenate(qs), np.concatenate(ps)
        P = {}
        for ti, t in enumerate(CONT):
            for k in range(1, H + 1):
                P[f"{t}_{k}"] = inv_target(t, q[:, ti * H + (k - 1), :])
        for ti, t in enumerate(BIN):
            for k in range(1, H + 1):
                P[f"{t}_{k}"] = p[:, ti * H + (k - 1)]
        return P
