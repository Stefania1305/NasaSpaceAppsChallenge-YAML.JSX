"""Short-range fire activity forecast from FIRMS detections.

Model
-----
The zone is cut into a lat/lon grid; for every cell and day we know whether it
had a vegetation-fire detection (volcanoes, gas flares and offshore sources
are dropped using the `type` column). For a target day t and horizon h, a cell
is described using only data up to the base day b = t - h:

    persistence  recency-weighted "was this cell burning" on b, b-1, b-2
    history      share of all days up to b on which the cell burned
    neighbours   share of the 8 surrounding cells burning on b (and b-1, half weight)
    regional     share of the surrounding 5x5 block burning on b and b-1
    intensity    log of fire radiative power (FRP) in the 3x3 block on b
    downwind     neighbours burning on the side the fire front is moving away
                 from (dominant spread direction seen in the data)

One ridge-regularised logistic regression is fitted per horizon (1, 2, 3 days)
on the loaded days themselves, so the model adapts to the region and season.
Candidate cells are those within 2 cells of recent fire; ignitions far away
from any fire are not predictable from this data and are reported separately.

Backtest: with 3+ days loaded, the 1-day model is refitted without the last
day and asked to predict it. It is scored by ranking (AUC, and precision /
recall of the top-K cells, K = expected number of burning cells) and compared
with the naive "tomorrow burns where today burned" baseline.

Pure standard library (no numpy) to match the rest of the backend.
"""

import math
import random
from collections import defaultdict
from datetime import date, timedelta

FEATURES = ["persistence", "history", "neighbours", "regional", "intensity", "downwind"]
PRIOR = [-3.5, 2.0, 1.0, 1.5, 2.0, 0.3, 0.5]   # intercept + FEATURES, used when data is too thin
MAX_TRAIN = 40000
MIN_POSITIVES = 15
NB1 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
NB2 = [(di, dj) for di in range(-2, 3) for dj in range(-2, 3)]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MAX_CELLS_OUT = 5000


def choose_cell(area: tuple) -> float:
    w, s, e, n = area
    deg2 = abs(e - w) * abs(n - s)
    return 0.05 if deg2 <= 25 else 0.1 if deg2 <= 400 else 0.25


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Grid:
    """Observed fire activity per (cell, day)."""

    def __init__(self, columns, rows, start: date, ndays: int, cell: float):
        self.cell, self.ndays, self.start = cell, ndays, start
        col = {c: i for i, c in enumerate(columns)}
        li, lo, di = col["latitude"], col["longitude"], col["acq_date"]
        fi = col.get("frp_mw", col.get("frp"))
        ti = col.get("type")
        self.burn = [set() for _ in range(ndays)]
        self.frp = [defaultdict(float) for _ in range(ndays)]
        self.used = self.skipped = 0
        for r in rows:
            if ti is not None and str(r[ti]).strip() not in ("", "0"):
                self.skipped += 1          # volcano, industrial source or offshore: not a wildfire
                continue
            lat, lon = _f(r[li]), _f(r[lo])
            try:
                t = (date.fromisoformat(r[di]) - start).days
            except ValueError:
                continue
            if lat is None or lon is None or not 0 <= t < ndays:
                continue
            k = (math.floor(lon / cell), math.floor(lat / cell))
            self.burn[t].add(k)
            self.frp[t][k] += (_f(r[fi]) or 0.0) if fi is not None else 0.0
            self.used += 1
        # Running count of burning days per cell, for the `history` feature.
        self.cum = []
        running = defaultdict(int)
        for day in self.burn:
            for k in day:
                running[k] += 1
            self.cum.append(dict(running))

    def on(self, t, k):
        return 1.0 if 0 <= t < self.ndays and k in self.burn[t] else 0.0

    def spread_vector(self, upto: int):
        """Mean unit step (x=east, y=north) from a burning cell to a newly burning
        neighbour, over transitions into days 1..upto-1. Returns (vx, vy, events)."""
        sx = sy = 0.0
        events = 0
        for t in range(1, upto):
            prev = self.burn[t - 1]
            for (i, j) in self.burn[t] - prev:
                src = [(di, dj) for di, dj in NB1 if (i - di, j - dj) in prev]
                if not src:
                    continue                # isolated new ignition, not spread
                for di, dj in src:
                    norm = math.hypot(di, dj)
                    sx += di / norm / len(src)
                    sy += dj / norm / len(src)
                events += 1
        return (sx / events, sy / events, events) if events else (0.0, 0.0, 0)

    def features(self, b, k, spread):
        """Features of cell k using data up to and including base day b."""
        i, j = k
        on = self.on
        pers = on(b, k) + 0.5 * on(b - 1, k) + 0.25 * on(b - 2, k)
        hist = self.cum[b].get(k, 0) / (b + 1)
        nb = (sum(on(b, (i + di, j + dj)) for di, dj in NB1)
              + 0.5 * sum(on(b - 1, (i + di, j + dj)) for di, dj in NB1)) / 8
        reg = (sum(on(b, (i + di, j + dj)) for di, dj in NB2)
               + sum(on(b - 1, (i + di, j + dj)) for di, dj in NB2)) / 50
        power = sum(self.frp[b].get((i + di, j + dj), 0.0) for di, dj in NB1 + [(0, 0)])
        vx, vy = spread
        down = 0.0
        if vx or vy:
            for di, dj in NB1:
                if (i + di, j + dj) in self.burn[b]:
                    # a burning neighbour at offset (di,dj) pushes fire towards us if spread points from it to us
                    down += max(0.0, (-di * vx - dj * vy) / math.hypot(di, dj))
        return [pers, hist, nb, reg, math.log1p(power) / 5, down / 3]

    def candidates(self, b):
        """Cells within 2 cells of fire on base day b or the day before."""
        keys = set()
        for t in (b, b - 1):
            if 0 <= t < self.ndays:
                for (i, j) in self.burn[t]:
                    keys.update((i + di, j + dj) for di, dj in NB2)
        return keys


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1 / (1 + math.exp(-z))


def _solve(a, b):
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                if f:
                    m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_logistic(X, y, offset=0.0, ridge=1.0, iters=15):
    """Ridge logistic regression by Newton/IRLS; `offset` undoes negative subsampling."""
    d = len(X[0]) + 1
    w = [0.0] * d
    for _ in range(iters):
        H = [[0.0] * d for _ in range(d)]
        g = [0.0] * d
        for x, t in zip(X, y):
            xi = (1.0, *x)
            p = _sigmoid(sum(a * b for a, b in zip(w, xi)))
            r = p * (1 - p)
            for a in range(d):
                g[a] += (p - t) * xi[a]
                ra = r * xi[a]
                Ha = H[a]
                for b in range(a, d):
                    Ha[b] += ra * xi[b]
        for a in range(d):
            for b in range(a):
                H[a][b] = H[b][a]
            if a:                                   # no penalty on the intercept
                H[a][a] += ridge
                g[a] += ridge * w[a]
        step = _solve(H, g)
        w = [a - b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-4:
            break
    w[0] += offset
    return w


def score(w, x):
    return _sigmoid(w[0] + sum(a * b for a, b in zip(w[1:], x)))


def train(grid, horizon, targets, spread, rng):
    """Fit the horizon-h model on target days `targets`. Returns (weights, info)."""
    pos, neg = [], []
    for t in targets:
        b = t - horizon
        if b < 0:
            continue
        burning = grid.burn[t]
        for k in grid.candidates(b):
            (pos if k in burning else neg).append(grid.features(b, k, spread))
    if len(pos) < MIN_POSITIVES or not neg:
        return PRIOR[:], {"fitted": False, "positives": len(pos), "negatives": len(neg)}
    keep = max(1, MAX_TRAIN - len(pos))
    rate = min(1.0, keep / len(neg))
    if rate < 1.0:
        neg = rng.sample(neg, keep)
    w = fit_logistic(pos + neg, [1] * len(pos) + [0] * len(neg), offset=math.log(rate))
    return w, {"fitted": True, "positives": len(pos), "negatives": len(neg)}


def _auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = sorted(pos), sorted(neg)
    j, total = 0, 0.0
    for p in pos:
        while j < len(neg) and neg[j] < p:
            j += 1
        k = j
        while k < len(neg) and neg[k] == p:
            k += 1
        total += j + 0.5 * (k - j)
    return round(total / (len(pos) * len(neg)), 3)


def backtest(grid, rng):
    """Refit the 1-day model without the last day, predict it, compare with reality."""
    last = grid.ndays - 1
    b = last - 1
    spread = grid.spread_vector(last)[:2]
    w, info = train(grid, 1, range(1, last), spread, rng)
    burning = grid.burn[last]
    cands = grid.candidates(b)
    probs = {k: score(w, grid.features(b, k, spread)) for k in cands}
    expected = sum(probs.values())
    K = max(1, round(expected))
    top = set(sorted(probs, key=probs.get, reverse=True)[:K])
    hits = len(top & burning)
    # Naive baseline: tomorrow burns exactly where today burned.
    base = grid.burn[b]
    base_hits = len(base & burning)
    return {
        "day": (grid.start + timedelta(days=last)).isoformat(),
        "actual_cells": len(burning),
        "actual_cells_near_recent_fire": len(burning & cands),
        "expected_cells": round(expected, 1),
        "top_k": K,
        "precision": round(hits / K, 3),
        "recall": round(hits / len(burning), 3) if burning else None,
        "auc": _auc([p for k, p in probs.items() if k in burning], [p for k, p in probs.items() if k not in burning]),
        "baseline_precision": round(base_hits / len(base), 3) if base else None,
        "baseline_recall": round(base_hits / len(burning), 3) if burning else None,
        "model_fitted": info["fitted"],
    }


def predict(columns, rows, start: str, ndays: int, area: tuple, horizon: int = 3, seed: int = 7) -> dict:
    start_d = date.fromisoformat(start)
    cell = choose_cell(area)
    grid = Grid(columns, rows, start_d, ndays, cell)
    last = ndays - 1

    vx, vy, events = grid.spread_vector(ndays)
    strength = math.hypot(vx, vy)
    spread = (vx, vy) if events >= 10 and strength >= 0.1 else (0.0, 0.0)

    cands = grid.candidates(last)
    feats = {k: grid.features(last, k, spread) for k in cands}
    per_cell = {k: [0.0] * horizon for k in cands}
    forecast, models = [], []
    for h in range(1, horizon + 1):
        # Train on every target day that has a base day h days earlier; if none exist
        # (fewer days loaded than the horizon), reuse the longest horizon we could fit.
        w, info = train(grid, h, range(h, ndays), spread, random.Random(seed + h))
        if not info["fitted"] and models and models[-1]["fitted"]:
            w, info = models[-1]["_w"], {**models[-1]["_info"], "borrowed_from_horizon": models[-1]["horizon"]}
        models.append({"horizon": h, "fitted": info["fitted"], "_w": w, "_info": info,
                       "weights": dict(zip(["intercept"] + FEATURES, (round(x, 3) for x in w))),
                       **{k: v for k, v in info.items() if k != "fitted"}})
        probs = [score(w, feats[k]) for k in cands]
        for k, p in zip(cands, probs):
            per_cell[k][h - 1] = p
        forecast.append({
            "date": (start_d + timedelta(days=last + h)).isoformat(),
            "expected_cells": round(sum(probs), 1),
            "likely_cells": sum(1 for p in probs if p >= 0.5),
        })
    for m in models:
        m.pop("_w"), m.pop("_info")

    burning_now = grid.burn[last]
    cells = []
    for (i, j), ps in per_cell.items():
        if max(ps) < 0.1:
            continue
        cells.append({
            "w": round(i * cell, 4), "s": round(j * cell, 4),
            "e": round((i + 1) * cell, 4), "n": round((j + 1) * cell, 4),
            "p": [round(p, 3) for p in ps], "burning_now": (i, j) in burning_now,
        })
    cells.sort(key=lambda c: -c["p"][0])
    truncated = len(cells) > MAX_CELLS_OUT
    cells = cells[:MAX_CELLS_OUT]

    # Where new fire is most likely: highest-risk cells that are not burning yet.
    hotspots = [{"lat": round((c["s"] + c["n"]) / 2, 3), "lon": round((c["w"] + c["e"]) / 2, 3), "p": c["p"][0]}
                for c in cells if not c["burning_now"]][:5]

    direction = None
    if spread != (0.0, 0.0):
        bearing = (math.degrees(math.atan2(vx, vy)) + 360) % 360
        direction = {"bearing_deg": round(bearing), "compass": COMPASS[round(bearing / 45) % 8],
                     "strength": round(strength, 2), "events": events}

    # Ignitions far from any recent fire can't be placed on the map, but their typical
    # daily count can be estimated from the loaded days.
    far = [len(grid.burn[t] - grid.candidates(t - 1)) for t in range(1, ndays)]
    elsewhere = round(sum(far) / len(far), 1) if far else None
    for f in forecast:
        f["expected_new_elsewhere"] = elsewhere

    observed = [len(d) for d in grid.burn]
    trend = round((observed[-1] - observed[-2]) / observed[-2], 3) if ndays >= 2 and observed[-2] else None

    return {
        "range": [start, (start_d + timedelta(days=last)).isoformat()],
        "forecast": forecast,
        "cells": cells,
        "cells_truncated": truncated,
        "cell_deg": cell,
        "cell_km": round(cell * 111.32, 1),
        "observed_cells_per_day": observed,
        "trend_last_day": trend,
        "spread_direction": direction,
        "hotspots": hotspots,
        "models": models,
        "features": FEATURES,
        "backtest": backtest(grid, random.Random(seed)) if ndays >= 3 else None,
        "detections_used": grid.used,
        "detections_skipped_non_vegetation": grid.skipped,
    }
