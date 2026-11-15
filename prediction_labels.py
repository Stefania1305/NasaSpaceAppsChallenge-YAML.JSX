"""Fire chains: correct the training labels for fires that disappear from the satellite data and come back.

A fire can vanish for a day or two (clouds, smoke, a missed overpass, too small to be seen on one pass) and then be
detected again. The tracking then starts a new fire, so the first fire looks extinguished and the second looks
new. In the test years 43 % of the "extinctions" were followed by a fire within 5 km in the next 3 days.

This module links such fires into chains and recomputes everything that depends on a fire's identity along the
chain:
  link     fire B continues fire A when B is first seen 1-3 days after A's last sighting, and B's first centre is
           within 5 km plus A's radius (radius of a circle with A's footprint) of A's last centre. Each fire gets
           at most one predecessor and one successor (nearest first).
  history  footprint continues across the chain (B's footprint is added to A's final one), new area, previous-day
           power and movement, age and merges so far follow the chain, over calendar days.
  targets  for each fire-day d and day ahead k: growth, fire power, movement and merges on day d+k along the chain;
           out_k = the chain was last seen before d+k AND no fire continued it within 3 days, so it is really out;
           when the chain's last sighting is within 3 days of the end of the data, it is unknown (censored, NaN).
  survival time to extinction for the survival model: t_event = days from d to the day after the chain's last
           sighting; censored when the end of the data comes before it could be confirmed.

Built into the training table by:  python prediction_data.py --chains --out fire_days_v3.csv.gz
In code: relabel(every_sighted_fire_day_of_one_run, with_targets)   (also used by the prediction engine)
"""

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from firms import DATA_DIR
from prediction_data import H, KM_LAT, KM_LON

GAP_DAYS = 3          # a fire may vanish for up to this many days and still be the same fire
LINK_KM = 5.0         # plus the earlier fire's radius
OUT = DATA_DIR / "prediction"


def _days(s):
    return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64)


def link(t: pd.DataFrame) -> pd.Series:
    """Chain id for every row of one run (rows: seen fire-days with fire_id, day, clat/clon/lat0/lon0, fine_fp, last_day)."""
    lat = t.clat.fillna(t.lat0).to_numpy()
    lon = t.clon.fillna(t.lon0).to_numpy()
    d = _days(t.day)
    f = pd.DataFrame({"fire": t.fire_id.to_numpy(), "d": d, "lat": lat, "lon": lon, "fp": t.fine_fp.fillna(0).to_numpy()})
    g = f.sort_values("d").groupby("fire")
    first, last = g.first(), g.last()
    last["last_seen"] = _days(t.groupby("fire_id").last_day.first().reindex(last.index))
    last["last_seen"] = np.maximum(last.last_seen.fillna(last.d), last.d)
    fires = first.index.to_numpy()
    succ, pred = {}, {}
    if len(fires) > 1:
        lat0 = math.radians(float(np.nanmean(first.lat)))
        xy_first = np.c_[first.lon * KM_LON * math.cos(lat0), first.lat * KM_LAT]
        tree = cKDTree(xy_first)
        radius = np.sqrt(last.fp.to_numpy() / math.pi)
        xy_last = np.c_[last.lon * KM_LON * math.cos(lat0), last.lat * KM_LAT]
        cand = []
        for i, a in enumerate(fires):
            reach = LINK_KM + radius[i]
            for j in tree.query_ball_point(xy_last[i], r=reach):
                b = fires[j]
                gap = first.d.iat[j] - last.last_seen.iat[i]
                if b != a and 1 <= gap <= GAP_DAYS:
                    cand.append((float(np.hypot(*(xy_first[j] - xy_last[i]))), a, b))
        for dist, a, b in sorted(cand):                      # nearest links first, one in and one out per fire
            if a not in succ and b not in pred:
                succ[a], pred[b] = b, a
    chain = {}
    for a in fires:
        if a in pred:
            continue
        cid, x = a, a
        while True:
            chain[x] = cid
            if x not in succ:
                break
            x = succ[x]
    return t.fire_id.map(chain)


def relabel(t: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    """One run's fire-day table with chain ids, chain-wise history features and (optionally) targets."""
    t = t.copy()
    t["chain"] = link(t).astype(np.int64)
    run_last = int(_days(pd.Series([t.run_last.iloc[0]]))[0]) if "run_last" in t else int(_days(t.day).max())
    t["_d"] = _days(t.day)
    # footprint continues along the chain: each later fire adds to the earlier fires' final footprint
    order = t.groupby(["chain", "fire_id"])._d.min().reset_index().sort_values(["chain", "_d"])
    final_fp = t.groupby("fire_id").fine_fp.max()
    order["fp_final"] = order.fire_id.map(final_fp).fillna(0)
    order["offset"] = order.groupby("chain").fp_final.cumsum() - order.fp_final
    t["fp_chain"] = t.fine_fp.fillna(0) + t.fire_id.map(order.set_index("fire_id").offset).fillna(0)
    last_seen = t.groupby("fire_id").last_day.first().pipe(_days) if "last_day" in t else None
    t["_fire_last"] = t.fire_id.map(pd.Series(np.asarray(last_seen), index=t.groupby("fire_id").last_day.first().index)) \
        if last_seen is not None else t.groupby("fire_id")._d.transform("max")
    t["_fire_last"] = np.maximum(t._fire_last.fillna(t._d), t.groupby("fire_id")._d.transform("max"))
    chain_last = t.groupby("chain")._fire_last.transform("max")
    # calendar of each chain: every day from its first to its last sighting
    span = t.groupby("chain").agg(a=("_d", "min"), b=("_fire_last", "max"))
    span["b"] = np.maximum(span.b, t.groupby("chain")._d.max())
    n = (span.b - span.a + 1).astype(int).to_numpy()
    cal = pd.DataFrame({"chain": np.repeat(span.index.to_numpy(), n),
                        "_d": np.repeat(span.a.to_numpy(), n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))})
    day_rows = t.sort_values("fp_chain").groupby(["chain", "_d"]).agg(
        fp=("fp_chain", "max"), frp=("frp", "mean"), frp_peak=("frp_peak", "max"), clat=("clat", "mean"), clon=("clon", "mean"),
        merge=("merge", "max"), seen=("fire_id", "size")).reset_index()
    cal = cal.merge(day_rows, on=["chain", "_d"], how="left")
    cal["seen"] = cal.seen.fillna(0) > 0
    gc = cal.groupby("chain")
    cal["fp"] = gc.fp.ffill().fillna(0)
    cal["fp"] = cal.groupby("chain").fp.cummax()
    cal["merge"] = cal["merge"].fillna(0)
    cal["new_area"] = cal.groupby("chain").fp.diff().fillna(cal.fp)
    gc = cal.groupby("chain")
    lat0 = t.groupby("chain").lat0.first()
    coslat = np.cos(np.radians(cal.chain.map(lat0)))
    feats = {}
    for lag in (1, 2):
        feats[f"new_area_l{lag}"] = gc.new_area.shift(lag)
        feats[f"frp_l{lag}"] = gc.frp.shift(lag)
        feats[f"dx_l{lag}"] = (cal.clon - gc.clon.shift(lag)) * KM_LON * coslat
        feats[f"dy_l{lag}"] = (cal.clat - gc.clat.shift(lag)) * KM_LAT
    feats["age_days"] = gc.cumcount()
    feats["merges_so_far"] = gc["merge"].cumsum() - cal["merge"]
    feats["new_area"] = cal.new_area
    feats["fp_chain"] = cal.fp
    if with_targets:
        clast = cal.chain.map(t.groupby("chain")._fire_last.max())
        for k in range(1, H + 1):
            fut_fp = gc.fp.shift(-k)
            # past the chain's last calendar day the fire is out: footprint stays at its final value
            fut_fp = fut_fp.fillna(cal.chain.map(cal.groupby("chain").fp.last()))
            feats[f"growth_{k}"] = (fut_fp - cal.fp).clip(lower=0)
            feats[f"frp_{k}"] = gc.frp.shift(-k)
            feats[f"frp_peak_{k}"] = gc.frp_peak.shift(-k)
            feats[f"dx_{k}"] = (gc.clon.shift(-k) - cal.clon) * KM_LON * coslat
            feats[f"dy_{k}"] = (gc.clat.shift(-k) - cal.clat) * KM_LAT
            feats[f"merge_{k}"] = sum(gc["merge"].shift(-j).fillna(0) for j in range(1, k + 1)).clip(upper=1)
            out = (clast < cal._d + k).astype(float)
            confirmable = clast + GAP_DAYS <= run_last
            feats[f"out_{k}"] = np.where(out.to_numpy() == 1, np.where(confirmable, 1.0, np.nan), 0.0)
        confirm = (clast + GAP_DAYS <= run_last).to_numpy()
        feats["surv_t"] = (clast + 1 - cal._d).astype(float)          # days until the first day without fire
        feats["surv_observed"] = confirm.astype(float)
    F = pd.DataFrame(feats)
    F["chain"], F["_d"] = cal.chain.to_numpy(), cal._d.to_numpy()
    drop = [c for c in F.columns if c not in ("chain", "_d") and c in t.columns]
    t = t.drop(columns=drop).merge(F, on=["chain", "_d"], how="left")
    t["fine_fp"] = t.pop("fp_chain")
    return t.drop(columns=["_d", "_fire_last"])
