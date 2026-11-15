"""Epicenter worker: stage 4 of the prediction pipeline.

Combines the localisation (3a) and fire-power (3b) outputs to estimate, for
every fire, where and when it started. Runs after both workers and before the
weather join.

Where
- Earliest detections: the fire's first satellite pass plus any other pass
  within 30 min of it (a coarse MODIS pass and a fine VIIRS pass can be close).
- Centre: their weighted centre; weight = FRP / pixel area, so hot pixels and
  fine (VIIRS) pixels count more.
- Step back: by the time a fire is first seen it has already spread, and the
  origin lies towards the back of the burning area. When localisation knows
  the spread direction, the centre is moved halfway from the centre to the back
  edge of the earliest pixels (measured along the spread direction).
- Uncertainty radius: distance from the epicenter to the farthest outer edge of
  the earliest pixels. Assuming no spotting, the origin lies inside the area
  that was already burning at the first pass.
- Several origins: groups of earliest pixels more than 5 km apart (two fires
  that merged later) each get their own epicenter. Closer gaps are usually
  smoke, cloud or pixels too faint to detect inside one fire. Fires already
  burning when the data starts get one epicenter (their origin is unknown).

When
- Latest possible start: the first detection.
- Earliest possible start: the last time a satellite passed over the epicenter
  (inside its swath, 50 km margin) without seeing a fire, from the swath model
  fitted per satellite to this run's detections (Aqua and the three VIIRS
  satellites; Terra and Landsat are not modelled). A pass only counts when
  that satellite has at least one detection anywhere within 3 min of it, which
  proves its data for that stretch of orbit exists (near-real-time files have
  gaps, and a missing file must not read as "looked and saw nothing").
  Passes without that proof are skipped. Clouds or a fire too small
  to detect would also explain a miss, so this is the last clear chance to
  have seen it, not proof that it was not burning.
- origin: "observed start" when such a pass exists in the data; "before data
  start" when the fire was already burning at the first pass the data allows
  (its real start is unknown and must not be learned as a start time).

Merging neighbouring fires
- Fires whose pixels lie next to each other (centre distance <= the two pixel
  sizes added, i.e. edges at most about one pixel apart) at any time in the
  run are merged into one fire, also through chains of neighbours. Each
  original fire keeps its own epicenter, so a merged fire has several.
- The merged fire's totals are recomputed: burning area and FRP are summed per
  satellite pass (then the peak pass taken), energy and biomass added up,
  extent and hull recomputed from all pixels, spread averaged over the
  original fires weighted by detections.

Start conditions
- FRP of the first pass and its ratio to the fire's peak (same sensor family,
  from the fire-power worker): a small ratio means it was caught early.
- Day / night, local solar time, type and confidence of the earliest detections.

Usage:  python epicenter_worker.py --date 2026-10-13
Output: <run>/epicenter/epicenters.csv.gz (one row per epicenter, merged fire ids),
        fires.csv.gz (one row per merged fire), fire_map.csv.gz (original -> merged), summary.json
"""

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from correlation import components
from firms import DATA_DIR
from swath import SwathModel

EARLY_WINDOW_MIN = 30
WORKERS = os.cpu_count() or 4
MULTI_ORIGIN_KM = 5.0        # earliest pixel groups further apart than this are separate origins
TOUCH_FACTOR = 1.5
SWATH_MARGIN_KM = 50
DATA_PROOF_MIN = 3            # a pass counts only if that satellite has a detection within +-3 min
MAX_LOOKBACK_PASSES = 40
SWATH_SATELLITES = {"Aqua": "MODIS", "Suomi NPP": "VIIRS", "NOAA-20": "VIIRS", "NOAA-21": "VIIRS"}
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320


def epoch_min(dates: pd.Series, minutes: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates) - pd.Timestamp("1970-01-01")).dt.days * 1440 + minutes


def iso_many(m: pd.Series) -> list:
    """Vectorised iso(): epoch minutes -> 'YYYY-MM-DDTHH:MMZ' ('' where missing)."""
    v = pd.to_numeric(pd.Series(m), errors="coerce")
    out = (pd.Timestamp("1970-01-01") + pd.to_timedelta(v, unit="min")).dt.strftime("%Y-%m-%dT%H:%MZ")
    return out.where(v.notna(), "").tolist()


def iso(m) -> str:
    return "" if m is None or (isinstance(m, float) and math.isnan(m)) else \
        (pd.Timestamp("1970-01-01") + pd.Timedelta(minutes=float(m))).strftime("%Y-%m-%dT%H:%MZ")


# ---------------------------------------------------------------------------
# Swath models: when could each satellite have seen the spot?
# ---------------------------------------------------------------------------

def swath_rows(run_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Rows to fit the swath model and prove satellite data on: the run's own rows, or, for small-area runs,
    worldwide detections of the same days from <run>/swath_context.csv.gz (a single fire's area has too few
    detections per satellite and day to reconstruct the orbits)."""
    ctx = run_dir / "swath_context.csv.gz"
    if not ctx.exists():
        return rows
    c = pd.read_csv(ctx, dtype={"acq_date": str})
    c["t_abs"] = (pd.to_datetime(c.acq_date) - pd.Timestamp("1970-01-01")).dt.days * 1440 + c.acq_minutes
    return c


def fit_swaths(rows: pd.DataFrame) -> dict:
    models = {}
    for sat, inst in SWATH_SATELLITES.items():
        g = rows[rows.satellite == sat]
        if g.empty:
            continue
        m = SwathModel(inst)
        m.fit(zip(g.acq_date, g.acq_minutes, g.latitude, g.longitude, g.daynight, g.pixel_area_km2.fillna(1.0)))
        if m.fits:
            models[sat] = m
    return models


def has_data(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """True where the sorted detection times contain one within DATA_PROOF_MIN of t."""
    k = np.searchsorted(times, t)
    lo = np.abs(t - times[np.clip(k - 1, 0, len(times) - 1)])
    hi = np.abs(times[np.clip(k, 0, len(times) - 1)] - t)
    return np.minimum(lo, hi) <= DATA_PROOF_MIN


def last_look(models: dict, sat_times: dict, lat, lon, before_abs, days: list) -> tuple:
    """Latest satellite pass over each point before before_abs (epoch minutes) whose data provably
    exists; returns (time, satellite, passes skipped for missing data)."""
    best = np.full(len(lat), np.nan)
    who = np.full(len(lat), "", dtype=object)
    skipped = np.zeros(len(lat), int)
    for sat, m in models.items():
        limit = np.asarray(before_abs, float).copy()
        found = np.full(len(lat), np.nan)
        todo = np.ones(len(lat), bool)
        for _ in range(MAX_LOOKBACK_PASSES):
            if not todo.any():
                break
            cand = np.full(len(lat), np.nan)
            for day in days:
                day0 = (pd.Timestamp(day) - pd.Timestamp("1970-01-01")).days * 1440
                t = m.last_look_before(day, lat[todo], lon[todo], limit[todo] - day0, SWATH_MARGIN_KM) + day0
                c = cand[todo]
                cand[todo] = np.where(~np.isnan(t) & ~(c >= t), t, c)
            none = todo & np.isnan(cand)
            todo &= ~none                                   # no earlier pass in the data
            ok = todo & has_data(sat_times[sat], np.nan_to_num(cand))
            found[ok] = cand[ok]
            todo &= ~ok
            skipped += todo
            limit[todo] = cand[todo] - 1                    # try the pass before
        better = ~np.isnan(found) & ~(best >= found)
        best = np.where(better, found, best)
        who = np.where(better, sat, who)
    return best, who, skipped


def swath_self_check(models: dict, rows: pd.DataFrame, sample: int = 3000) -> dict:
    """Each satellite's own detections should sit on a modelled pass: median time error."""
    out = {}
    for sat, m in models.items():
        g = rows[rows.satellite == sat]
        g = g.sample(min(sample, len(g)), random_state=0)
        errs = []
        for day, d in g.groupby("acq_date"):
            t = m.last_look_before(day, d.latitude, d.longitude, d.acq_minutes + 10, margin_km=-200)
            errs.append(np.abs(d.acq_minutes.to_numpy() - t))
        e = np.concatenate(errs) if errs else np.array([])
        e = e[~np.isnan(e)]
        out[sat] = {"checked": int(len(g)), "found": int(len(e)),
                    "median_error_min": round(float(np.median(e)), 2) if len(e) else None,
                    "within_10_min": round(float((e <= 10).mean()), 4) if len(e) else None}
    return out


# ---------------------------------------------------------------------------
# One fire
# ---------------------------------------------------------------------------

def local_xy(lat, lon, lat0, lon0):
    dlon = (np.asarray(lon) - lon0 + 180) % 360 - 180
    return dlon * KM_PER_DEG_LON * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * KM_PER_DEG_LAT


def to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / KM_PER_DEG_LAT
    lon = lon0 + x / (KM_PER_DEG_LON * math.cos(math.radians(lat0)))
    return lat, (lon + 180) % 360 - 180


def epicenters_for_fire(fire: pd.Series, early: pd.DataFrame, split: bool = True) -> list:
    lat0, lon0 = float(early.latitude.mean()), float(early.longitude.mean())
    x, y = local_xy(early.latitude.to_numpy(), early.longitude.to_numpy(), lat0, lon0)
    half = early.pixel_km.to_numpy() / 2
    w = (early.frp_mw.fillna(0).clip(lower=0).to_numpy() + 0.1) / early.area.to_numpy()

    # groups of earliest pixels more than MULTI_ORIGIN_KM apart = separate origins
    n = len(early)
    if n > 1 and split:
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        touch = TOUCH_FACTOR * (early.pixel_km.to_numpy()[:, None] + early.pixel_km.to_numpy()[None, :]) / 2
        i, j = np.nonzero(np.triu(d <= np.maximum(touch, MULTI_ORIGIN_KM), 1))
        group = components(n, i, j)
    else:
        group = np.zeros(n, int)

    sdir = fire.get("spread_direction_deg")
    have_dir = sdir is not None and not (isinstance(sdir, float) and math.isnan(sdir))
    ux, uy = (math.sin(math.radians(sdir)), math.cos(math.radians(sdir))) if have_dir else (0.0, 0.0)

    out = []
    for gno in range(group.max() + 1):
        m = group == gno
        gx, gy, gw, gh = x[m], y[m], w[m], half[m]
        cx, cy = float(np.average(gx, weights=gw)), float(np.average(gy, weights=gw))
        step = 0.0
        if have_dir and m.sum() >= 2 and split:      # unknown origin: report the centre of what was first seen
            proj = (gx - cx) * ux + (gy - cy) * uy
            back = float((proj - gh).min())          # back edge along the spread direction (negative)
            step = back / 2
        ex, ey = cx + step * ux, cy + step * uy
        radius = float(np.max(np.hypot(gx - ex, gy - ey) + gh * math.sqrt(2)))
        lat, lon = to_latlon(ex, ey, lat0, lon0)
        out.append({"lat": round(lat, 5), "lon": round(lon, 5), "uncertainty_km": round(radius, 3),
                    "backstep_km": round(-step, 3), "first_pixels": int(m.sum()), "rows": early.index[m]})
    return out


# ---------------------------------------------------------------------------
# Merging neighbouring fires
# ---------------------------------------------------------------------------

NEIGHBOUR_SCALE = 4 / 3   # correlation's 1.5 x mean pixel size becomes 2 x: edges at most ~one pixel apart


def neighbour_map(d: pd.DataFrame) -> pd.Series:
    """original fire_id -> merged fire_id (smallest id in its group of neighbouring fires)."""
    from itertools import combinations
    from correlation import _pairs, _xyz
    d = d.reset_index(drop=True)
    xyz, size, t = _xyz(d), d.pixel_km.to_numpy() * NEIGHBOUR_SCALE, d.t_abs.to_numpy(float)
    groups = {inst: np.flatnonzero(d.instrument.to_numpy() == inst) for inst in d.instrument.unique()}
    parts = [_pairs(xyz, size, t, groups[a], groups[b], False, np.inf) for a, b in combinations(sorted(groups), 2)]
    parts += [_pairs(xyz, size, t, groups[a], groups[a], True, np.inf) for a in groups]
    pairs = pd.concat(parts, ignore_index=True)
    codes, fires = pd.factorize(d.fire_id)
    a, b = codes[pairs.i.to_numpy(int)], codes[pairs.j.to_numpy(int)]
    keep = a != b
    comp = components(len(fires), a[keep], b[keep])
    merged = pd.Series(np.asarray(fires)).groupby(comp).transform("min")
    return pd.Series(merged.to_numpy(), index=np.asarray(fires), name="fire_id")


def circular_mean(deg: pd.Series, w: pd.Series) -> float:
    ok = deg.notna()
    if not ok.any():
        return np.nan
    r = np.radians(deg[ok])
    return float(np.degrees(np.arctan2((np.sin(r) * w[ok]).sum(), (np.cos(r) * w[ok]).sum())) % 360)


def merged_fire_table(fmap: pd.Series, lf: pd.DataFrame, lobs: pd.DataFrame, fpf: pd.DataFrame,
                      fpo: pd.DataFrame, ep: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    from localisation_worker import hull_area
    lf = lf.assign(merged=lf.fire_id.map(fmap))
    rank = {"tracked": 3, "no comparable passes": 2, "single pass": 1, "single detection": 0}
    area = (lobs.assign(merged=lobs.fire_id.map(fmap)).groupby(["merged", "pass_id"]).burning_area_km2.sum()
            .groupby("merged").max())
    passes = lobs.assign(merged=lobs.fire_id.map(fmap)).groupby("merged").pass_id.nunique()
    fp = fpo[fpo.partial_view != "yes"].assign(merged=fpo.fire_id.map(fmap))
    per_pass = fp.groupby(["merged", "pass_id"]).frp_sum_mw.sum().rename("frp").reset_index()   # numeric: fast path
    per_pass["time"] = per_pass.set_index(["merged", "pass_id"]).index.map(
        fp.drop_duplicates(["merged", "pass_id"]).set_index(["merged", "pass_id"]).time)
    peak = per_pass.loc[per_pass.groupby("merged").frp.idxmax()].set_index("merged")
    energy = (fpf.reset_index().assign(merged=lambda x: x.fire_id.map(fmap))
              .groupby("merged")[["fre_mj", "biomass_burned_t"]].sum(min_count=1))
    eps = pd.DataFrame({"epicenters": ep.groupby("fire_id").size(),
                        "observed_starts": ep.origin.eq("observed start").groupby(ep.fire_id).sum()})
    # everything as grouped operations; only merged fires made of several fires need a new hull
    g = lf.groupby("merged")
    w = lf.detections.astype(float)
    sizes = g.size()
    out = pd.DataFrame({
        "source_fires": sizes,
        "source_fire_ids": sizes.index.astype(str),          # single fires: their own id (multi fires below)
        "detections": g.detections.sum().astype(int),
        # times as datetimes for the fast grouped min/max, then back to text
        "first_seen": pd.to_datetime(lf.first_seen, format="%Y-%m-%dT%H:%MZ").groupby(lf.merged).min().dt.strftime("%Y-%m-%dT%H:%MZ"),
        "last_seen": pd.to_datetime(lf.last_seen, format="%Y-%m-%dT%H:%MZ").groupby(lf.merged).max().dt.strftime("%Y-%m-%dT%H:%MZ"),
        "latitude": ((lf.latitude * w).groupby(lf.merged).sum() / w.groupby(lf.merged).sum()).round(5),
        "longitude": ((lf.longitude * w).groupby(lf.merged).sum() / w.groupby(lf.merged).sum()).round(5),
        "west": g.west.min(), "south": g.south.min(), "east": g.east.max(), "north": g.north.max(),
        "hull_area_km2": g.hull_area_km2.first().round(3),
    })
    multi = sizes[sizes > 1].index
    if len(multi):
        ml = lf[lf.merged.isin(multi)].sort_values("fire_id")
        out.loc[multi, "source_fire_ids"] = ml.groupby("merged").fire_id.agg(lambda x: "+".join(map(str, x)))
        pix = d.assign(merged=d.fire_id.map(fmap))
        pix = pix[pix.merged.isin(multi)]
        for mid, p in pix.groupby("merged"):
            lat0, lon0 = float(p.latitude.mean()), float(p.longitude.mean())
            x, y = local_xy(p.latitude.to_numpy(), p.longitude.to_numpy(), lat0, lon0)
            hs = (p.scan_km.fillna(p.pixel_km) / 2).to_numpy()
            ht = (p.track_km.fillna(p.pixel_km) / 2).to_numpy()
            out.at[mid, "hull_area_km2"] = round(hull_area(np.concatenate([x - hs, x + hs, x - hs, x + hs]),
                                                           np.concatenate([y - ht, y - ht, y + ht, y + ht])), 3)
    out["epicenters"] = eps.epicenters.reindex(out.index).fillna(0).astype(int)
    out["observed_starts"] = eps.observed_starts.reindex(out.index).fillna(0).astype(int)
    out["observations"] = passes.reindex(out.index).fillna(0).astype(int)
    out["max_burning_area_km2"] = area.reindex(out.index).round(3)
    out["peak_frp_mw"] = peak.frp.reindex(out.index).round(2)
    out["peak_time"] = peak.time.reindex(out.index).fillna("")
    out["fre_mj"] = energy.fre_mj.reindex(out.index).round(1)
    out["biomass_burned_t"] = energy.biomass_burned_t.reindex(out.index).round(2)
    tr = lf[(lf.status == "tracked") & lf.spread_direction_deg.notna()]
    rad, tw = np.radians(tr.spread_direction_deg), tr.detections.astype(float)
    sx, cx = (np.sin(rad) * tw).groupby(tr.merged).sum(), (np.cos(rad) * tw).groupby(tr.merged).sum()
    out["spread_direction_deg"] = (np.degrees(np.arctan2(sx, cx)) % 360).round(1).reindex(out.index)
    rr = lf[(lf.status == "tracked") & lf.spread_rate_km_h.notna()]
    rw = rr.detections.astype(float)
    out["spread_rate_km_h"] = ((rr.spread_rate_km_h * rw).groupby(rr.merged).sum() / rw.groupby(rr.merged).sum()).round(3).reindex(out.index)
    inv = {v: k for k, v in rank.items()}
    out["status"] = lf.status.map(rank).fillna(-1).groupby(lf.merged).max().map(inv).reindex(out.index)
    out = out.rename_axis("fire_id").reset_index()
    out = out[["fire_id", "source_fires", "source_fire_ids", "epicenters", "observed_starts", "detections", "observations",
               "first_seen", "last_seen", "latitude", "longitude", "west", "south", "east", "north", "hull_area_km2",
               "max_burning_area_km2", "peak_frp_mw", "peak_time", "fre_mj", "biomass_burned_t", "spread_direction_deg",
               "spread_rate_km_h", "status"]]
    return out.sort_values("detections", ascending=False, ignore_index=True)


def epicenter_records(early, fires, fpf, first_by_family, passes_by_family, window, fire_first) -> list:
    """Epicenters for a chunk of fires (runs in a worker process)."""
    records = []
    for fid, e in early.groupby("fire_id"):
        f = fires.loc[fid]
        observed = not np.isnan(window.look[fid])
        for k, ep in enumerate(epicenters_for_fire(f, e, split=observed)):
            r = e.loc[ep["rows"]]
            first = r.sort_values("t_abs").iloc[0]
            peak_frp = fpf.peak_frp_mw.get(fid, np.nan)
            fam = fpf.peak_family.get(fid, None)
            # only meaningful when the peak's sensor family saw the fire more than once
            start_frp = first_by_family.get((fid, fam), np.nan) if fam and passes_by_family.get((fid, fam), 0) >= 2 else np.nan
            records.append({
                "fire_id": fid, "epicenter_no": k + 1, "epicenters_in_fire": 0,
                "latitude": ep["lat"], "longitude": ep["lon"], "uncertainty_km": ep["uncertainty_km"],
                "backstep_km": ep["backstep_km"], "spread_direction_deg": f.get("spread_direction_deg", np.nan),
                "first_pixels": ep["first_pixels"], "latest_start_abs": float(fire_first[fid]),
                "earliest_start_abs": window.look[fid], "earliest_from": window.who[fid],
                "passes_skipped_no_data": int(window.skipped[fid]),
                "first_satellite": first.satellite, "first_instrument": first.instrument,
                "start_daynight": r.daynight.mode().iloc[0],
                "start_local_solar_hour": round(float(((r.acq_minutes / 60 + r.longitude / 15) % 24).iloc[0]), 2),
                "start_type": r.type.astype(str).mode().iloc[0],
                "start_confidence": r.confidence.mode().iloc[0],
                "start_frp_mw": round(float(r.frp_mw.sum()), 2) if r.frp_mw.notna().any() else np.nan,
                "start_to_peak_ratio": round(float(start_frp / peak_frp), 3) if peak_frp and peak_frp > 0 and not math.isnan(start_frp) else np.nan,
                "fire_status": f.status,
            })
    return records


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class Timer:
    """Seconds spent in each part of a run (reported in summary.json)."""
    def __init__(self):
        self.t, self.parts = time.time(), {}

    def mark(self, name: str):
        now = time.time()
        self.parts[name] = round(now - self.t, 2)
        self.t = now


def run(run_dir: Path) -> dict:
    t0 = time.time()
    T = Timer()
    need = [run_dir / p for p in ("correlation/rows.csv.gz", "localisation/fires.csv.gz",
                                  "localisation/detections.csv.gz", "fire_power/observations.csv.gz",
                                  "fire_power/fires.csv.gz", "localisation/observations.csv.gz")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run layers 1-3 first): " + ", ".join(missing))

    rows = pd.read_csv(need[0], dtype={"acq_time": str, "type": str}, keep_default_na=False, na_values=[""])
    rows["row_id"] = np.arange(len(rows))
    rows["t_abs"] = epoch_min(rows.acq_date, rows.acq_minutes)
    default = rows.instrument.map({"MODIS": 1.0, "VIIRS": 0.375, "OLI": 0.03}).fillna(1.0)
    rows["pixel_km"] = np.fmax(rows.scan_km.fillna(default), rows.track_km.fillna(default))
    rows["area"] = rows.pixel_area_km2.fillna(rows.scan_km.fillna(default) * rows.track_km.fillna(default))
    lf = pd.read_csv(need[1], low_memory=False)
    det = pd.read_csv(need[2])
    fpo = pd.read_csv(need[3])
    fpf = pd.read_csv(need[4]).set_index("fire_id")
    lobs = pd.read_csv(need[5], low_memory=False)
    T.mark("load")

    d = det.merge(rows, on="row_id", how="left")
    first_t = d.groupby("fire_id").t_abs.transform("min")
    early = d[d.t_abs <= first_t + EARLY_WINDOW_MIN].set_index("row_id")
    fires = lf.set_index("fire_id")

    # first pass of the peak's sensor family, for the start-to-peak ratio
    fpo["t"] = pd.to_datetime(fpo.time, format="%Y-%m-%dT%H:%MZ")
    fpo = fpo[fpo.partial_view != "yes"].sort_values("t")
    first_by_family = fpo.groupby(["fire_id", "family"]).frp_sum_mw.first()
    passes_by_family = fpo.groupby(["fire_id", "family"]).size()

    # when did a satellite last look at the spot? once per fire, before its first detection
    T.mark("prepare")
    sw = swath_rows(run_dir, rows)
    models = fit_swaths(sw)
    T.mark("fit swath models")
    days = sorted(sw.acq_date.unique())
    sat_times = {s: np.sort(sw.loc[sw.satellite == s, "t_abs"].to_numpy(float)) for s in models}
    g = early.groupby("fire_id")
    fire_first = g.t_abs.min()
    look, who, skipped = last_look(models, sat_times, g.latitude.mean().to_numpy(), g.longitude.mean().to_numpy(),
                                   fire_first.to_numpy() - 5, days)
    window = pd.DataFrame({"look": look, "who": who, "skipped": skipped}, index=fire_first.index)
    T.mark("last look before each fire")

    fpf_cols = fpf[["peak_frp_mw", "peak_family"]]
    fbf, pbf = first_by_family.to_dict(), passes_by_family.to_dict()
    n_early = early.groupby("fire_id").size()

    # fires with one first pixel (most of them): the epicenter is that pixel, all in one vectorised step
    one = early[early.fire_id.isin(n_early[n_early == 1].index)].reset_index().set_index("fire_id")
    fid = one.index.to_series()
    fam = fid.map(fpf_cols.peak_family)
    peak = fid.map(fpf_cols.peak_frp_mw)
    key = list(zip(fid, fam))
    start_frp = pd.Series([fbf.get(k, np.nan) if isinstance(k[1], str) and pbf.get(k, 0) >= 2 else np.nan for k in key], index=fid.index)
    singles = pd.DataFrame({
        "fire_id": fid.to_numpy(), "epicenter_no": 1, "epicenters_in_fire": 0,
        "latitude": one.latitude.round(5).to_numpy(), "longitude": one.longitude.round(5).to_numpy(),
        "uncertainty_km": (one.pixel_km / 2 * math.sqrt(2)).round(3).to_numpy(), "backstep_km": 0.0,
        "spread_direction_deg": fid.map(fires.spread_direction_deg).to_numpy(), "first_pixels": 1,
        "latest_start_abs": fid.map(fire_first).astype(float).to_numpy(),
        "earliest_start_abs": fid.map(window.look).to_numpy(), "earliest_from": fid.map(window.who).to_numpy(),
        "passes_skipped_no_data": fid.map(window.skipped).astype(int).to_numpy(),
        "first_satellite": one.satellite.to_numpy(), "first_instrument": one.instrument.to_numpy(),
        "start_daynight": one.daynight.to_numpy(),
        "start_local_solar_hour": ((one.acq_minutes / 60 + one.longitude / 15) % 24).round(2).to_numpy(),
        "start_type": one.type.astype(str).to_numpy(), "start_confidence": one.confidence.to_numpy(),
        "start_frp_mw": one.frp_mw.round(2).to_numpy(),
        "start_to_peak_ratio": (start_frp / peak.where(peak > 0)).round(3).to_numpy(),
        "fire_status": fid.map(fires.status).to_numpy(),
    })

    # fires with several first pixels: geometry per fire, in parallel worker processes
    multi = early[early.fire_id.isin(n_early[n_early > 1].index)]
    mids = n_early[n_early > 1].sort_values(ascending=False).index
    chunk_of = pd.Series(np.arange(len(mids)) % (WORKERS * 4), index=mids)
    jobs = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for c, e_chunk in multi.groupby(multi.fire_id.map(chunk_of)):
            ids = e_chunk.fire_id.unique()
            sid = set(ids.tolist())
            jobs.append(pool.submit(epicenter_records, e_chunk, fires.loc[fires.index.intersection(ids)],
                                    fpf_cols.loc[fpf_cols.index.intersection(ids)],
                                    {k: v for k, v in fbf.items() if k[0] in sid},
                                    {k: v for k, v in pbf.items() if k[0] in sid},
                                    window.loc[ids], fire_first.loc[ids]))
        records = [r for j in jobs for r in j.result()]
    T.mark("epicenters (vectorised + parallel)")
    ep = pd.concat([singles, pd.DataFrame(records)], ignore_index=True).sort_values(["fire_id", "epicenter_no"], ignore_index=True)
    ep["epicenters_in_fire"] = ep.groupby("fire_id").epicenter_no.transform("max")
    look = ep.earliest_start_abs.to_numpy()
    data_start = float(epoch_min(pd.Series([days[0]]), pd.Series([0])).iloc[0])
    ep["origin"] = np.where(np.isnan(look), "before data start", "observed start")
    ep["start_window_h"] = ((ep.latest_start_abs - ep.earliest_start_abs) / 60).round(2)
    ep["earliest_start"] = iso_many(ep.earliest_start_abs)
    ep["latest_start"] = iso_many(ep.latest_start_abs)
    ep = ep.drop(columns=["earliest_start_abs", "latest_start_abs"])
    cols = ["fire_id", "epicenter_no", "epicenters_in_fire", "latitude", "longitude", "uncertainty_km", "origin",
            "earliest_start", "latest_start", "start_window_h", "earliest_from", "first_satellite",
            "first_instrument", "first_pixels", "backstep_km", "spread_direction_deg", "passes_skipped_no_data", "start_frp_mw",
            "start_to_peak_ratio", "start_daynight", "start_local_solar_hour", "start_type", "start_confidence",
            "fire_status"]
    ep = ep[cols]

    # merge fires whose pixels neighbour each other; each keeps its epicenter(s)
    T.mark("tidy epicenters")
    fmap = neighbour_map(d)
    T.mark("neighbour search")
    ep.insert(1, "source_fire_id", ep.fire_id)
    ep["fire_id"] = ep.source_fire_id.map(fmap)
    ep = ep.sort_values(["fire_id", "latest_start", "source_fire_id", "epicenter_no"], ignore_index=True)
    ep["epicenter_no"] = ep.groupby("fire_id").cumcount() + 1
    ep["epicenters_in_fire"] = ep.groupby("fire_id").epicenter_no.transform("max")
    merged = merged_fire_table(fmap, lf, lobs, fpf, fpo, ep, d)
    T.mark("merged fire table")

    out = run_dir / "epicenter"
    out.mkdir(parents=True, exist_ok=True)
    ep.to_csv(out / "epicenters.csv.gz", index=False)
    merged.to_csv(out / "fires.csv.gz", index=False)
    fmap.rename_axis("source_fire_id").reset_index().to_csv(out / "fire_map.csv.gz", index=False)
    obs = ep[ep.origin == "observed start"]
    summary = {
        "fires_before_merge": int(fmap.size), "fires": int(ep.fire_id.nunique()), "epicenters": len(ep),
        "merged_fires": int((merged.source_fires > 1).sum()),
        "fires_absorbed_by_merging": int(fmap.size - fmap.nunique()),
        "largest_merge": int(merged.source_fires.max()),
        "fires_with_several_origins": int((ep.groupby("fire_id").size() > 1).sum()),
        "origin": {k: int(v) for k, v in ep.origin.value_counts().items()},
        "observed_start": {
            "median_window_h": round(float(obs.start_window_h.median()), 2) if len(obs) else None,
            "earliest_from": {k: int(v) for k, v in obs.earliest_from.value_counts().items()}},
        "passes_skipped_for_missing_data": int(ep.passes_skipped_no_data.sum()),
        "median_uncertainty_km": round(float(ep.uncertainty_km.median()), 3),
        "stepped_back": int((ep.backstep_km > 0).sum()),
        "caught_early_ratio_below_0.25": int((ep.start_to_peak_ratio < 0.25).sum()),
        "swath_models": {s: sorted(m.fits) for s, m in models.items()},
        "swath_self_check": (swath_self_check(models, sw), T.mark("write + swath self-check"))[0],
        "data_start": iso(data_start),
        "settings": {"early_window_min": EARLY_WINDOW_MIN, "multi_origin_km": MULTI_ORIGIN_KM, "touch_factor": TOUCH_FACTOR,
                     "neighbour_rule": "centre distance <= sum of the two pixel sizes",
                     "swath_margin_km": SWATH_MARGIN_KM},
        "seconds": round(time.time() - t0, 1),
        "timings": T.parts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"epicenters": ep, "fires": merged, "summary": summary, "out": out}


def print_summary(res: dict) -> None:
    s, ep = res["summary"], res["epicenters"]
    print(f"\n{s['fires_before_merge']:,} fires -> {s['fires']:,} after merging neighbours "
          f"({s['merged_fires']:,} merged fires absorbed {s['fires_absorbed_by_merging']:,}; largest joins {s['largest_merge']})")
    print(f"{s['epicenters']:,} epicenters in {s['seconds']} s; {s['fires_with_several_origins']:,} fires have several epicenters")
    f = res["fires"]
    cols_f = ["fire_id", "source_fires", "epicenters", "observed_starts", "detections", "latitude", "longitude",
              "hull_area_km2", "peak_frp_mw", "spread_direction_deg", "status"]
    print("\nMerged fires with the most epicenters:")
    print(f.sort_values(["epicenters", "detections"], ascending=False).head(8)[cols_f].to_string(index=False))
    print("Origin: " + ", ".join(f"{k} {v:,}" for k, v in s["origin"].items()) + f"  (data start {s['data_start']})")
    o = s["observed_start"]
    if o["median_window_h"] is not None:
        print(f"Observed starts: median start window {o['median_window_h']} h; earliest bound from "
              + ", ".join(f"{k} {v:,}" for k, v in o["earliest_from"].items()))
    print(f"Passes skipped because the satellite's data for them is missing: {s['passes_skipped_for_missing_data']:,}")
    print(f"Median uncertainty radius {s['median_uncertainty_km']} km; {s['stepped_back']:,} epicenters stepped back "
          f"along the spread direction; {s['caught_early_ratio_below_0.25']:,} caught early (start < 25% of peak)")
    print("Swath model self-check (own detections on a modelled pass):")
    for sat, c in s["swath_self_check"].items():
        print(f"  {sat:<10} {c['found']:,}/{c['checked']:,} found, median error {c['median_error_min']} min, "
              f"{c['within_10_min']:.1%} within 10 min")
    cols = ["fire_id", "epicenter_no", "latitude", "longitude", "uncertainty_km", "origin", "earliest_start",
            "latest_start", "start_window_h", "earliest_from", "start_to_peak_ratio"]
    big = ep.merge(ep.groupby("fire_id").size().rename("n"), on="fire_id")
    print("\nObserved starts with the narrowest windows (among fires with 3+ first pixels):")
    print(ep[(ep.origin == "observed start") & (ep.first_pixels >= 3)].nsmallest(8, "start_window_h")[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    args = ap.parse_args()
    if args.run:
        run_dir = Path(args.run)
    elif args.date:
        run_dir = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    else:
        ap.error("give --run or --date")
    res = run(run_dir)
    print_summary(res)
    print(f"\nSaved to {res['out']}")


if __name__ == "__main__":
    main()
