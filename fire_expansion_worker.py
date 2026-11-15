"""Fire expansion worker: stage 4b of the prediction pipeline
(epicenter -> fire expansion -> knowledge graph -> weather join).

For every merged fire it replays the satellite passes in acq_time order, from
the first sighting (the burst) until the fire is extinguished, and follows how
its pixel clusters form, grow, join and split.

At each pass over a fire
- Clusters: that pass's pixels that neighbour each other (centre distance
  <= the two pixel sizes added, the same rule the epicenter worker uses to
  merge fires).
- Each cluster is matched to the clusters already being followed (tracks)
  whose latest footprint it neighbours:
    matches none            -> new track: "ignition" (the fire's first pixels)
                               or "new burst" (a new hot spot later on)
    matches one track       -> that track continues
    matches several tracks  -> MERGE: the tracks end and one new track starts
                               whose history is all of theirs (parents,
                               source fires, epicenters, first-seen time)
    one track -> several    -> SPLIT: the track ends, each part starts a new
                               track with that history
    several <-> several     -> "merge and split"
  Only fine passes (VIIRS, Landsat) decide merges and splits. A coarse MODIS
  pixel (1-5 km) often covers two separate VIIRS clusters; letting it merge
  them would produce merges that the next VIIRS pass undoes. MODIS passes only
  continue the tracks they touch, or start new ones.
- Flicker: VIIRS pixels grow from 0.375 km to ~0.8 km towards the swath edge,
  so two clusters can be neighbours on one satellite's pass and not on the
  next. A merge whose result splits again, or a split whose parts merge again,
  within 2 h is kept but marked undone_within_2h, and only lasting merges and
  splits are counted per fire.
- A track that is not seen on a pass keeps going (cloud, swath edge, partial
  view); only the end of the data decides whether it went out.

Extinguished
- For each track still going at the end, the next satellite pass after its
  last sighting that looked at the spot (swath model, with proof that the
  satellite's data for that stretch exists, as in the epicenter worker) and
  did not see it. The track went out between its last sighting and that pass.
  No such pass in the data: "burning at data end". A track only ever seen by
  VIIRS needs a VIIRS pass for this: MODIS cannot see small fires, so its
  "saw nothing" proves nothing for them.
- A fire is extinguished when all its last tracks are; its window runs from
  the last sighting to the latest of those passes.

Per pass the timeline records clusters seen and alive, pixels, burning area,
FRP, the cumulative footprint (union of all pixel squares so far, on a 100 m
raster; all sensors, and VIIRS/Landsat only since MODIS pixels inflate it),
the new footprint added, and how far the farthest pixel lies from the nearest
epicenter (the front distance).

Usage:  python fire_expansion_worker.py --date 2026-10-13
Output: <run>/expansion/fires.csv.gz, timeline.csv.gz, clusters.csv.gz, events.csv.gz,
        detections.csv.gz, summary.json
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
from epicenter_worker import Timer, fit_swaths, has_data, iso, iso_many, local_xy, swath_rows
from firms import DATA_DIR

RASTER_KM = 0.1
WORKERS = os.cpu_count() or 4
FLICKER_MIN = 120
FINE = {"VIIRS", "OLI"}
DEFAULT_PIXEL_KM = {"MODIS": 1.0, "VIIRS": 0.375, "OLI": 0.03}
MAX_LOOKAHEAD_PASSES = 40
SWATH_MARGIN_KM = 50


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def neighbours(ax, ay, asz, bx, by, bsz) -> np.ndarray:
    """Boolean matrix: pixel i of A neighbours pixel j of B (centre distance <= size_i + size_j)."""
    return np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :]) <= asz[:, None] + bsz[None, :]


def raster_cells(x, y, hs, ht) -> np.ndarray:
    """Keys of the 100 m raster cells covered by pixel squares (centre x, y, half sizes hs, ht)."""
    keys = []
    for xi, yi, a, b in zip(x, y, hs, ht):
        ix = np.arange(math.floor((xi - a) / RASTER_KM), math.floor((xi + a) / RASTER_KM) + 1)
        iy = np.arange(math.floor((yi - b) / RASTER_KM), math.floor((yi + b) / RASTER_KM) + 1)
        gx, gy = np.meshgrid(ix, iy)
        keys.append((gx.ravel().astype(np.int64) << 32) + (gy.ravel().astype(np.int64) & 0xFFFFFFFF))
    return np.unique(np.concatenate(keys)) if keys else np.array([], np.int64)


def near_pairs(ax, ay, asz, bx, by, bsz, limit=None):
    """(i, j, distance) for pixels i of A and j of B whose centres are within asz_i + bsz_j (or `limit` km),
    found through a grid so the cost grows with the number of pixels, not their square."""
    if not len(ax) or not len(bx):
        return np.array([], int), np.array([], int), np.array([])
    G = float(limit if limit is not None else 2 * max(asz.max(), bsz.max()))
    a = pd.DataFrame({"gx": np.floor(ax / G).astype(np.int64), "gy": np.floor(ay / G).astype(np.int64), "i": np.arange(len(ax))})
    b = pd.DataFrame({"gx": np.floor(bx / G).astype(np.int64), "gy": np.floor(by / G).astype(np.int64), "j": np.arange(len(bx))})
    parts = [a.assign(gx=a.gx + dx, gy=a.gy + dy).merge(b, on=["gx", "gy"])[["i", "j"]] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    m = pd.concat(parts, ignore_index=True)
    i, j = m.i.to_numpy(), m.j.to_numpy()
    d = np.hypot(ax[i] - bx[j], ay[i] - by[j])
    keep = d <= (limit if limit is not None else asz[i] + bsz[j])
    return i[keep], j[keep], d[keep]


def groups_of(labels: np.ndarray, k: int) -> list:
    """Index arrays of each label 0..k-1 (one sort instead of one scan per label)."""
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(k + 1))
    return [order[bounds[c]:bounds[c + 1]] for c in range(k)]


def nearest_distance(px, py, ex, ey) -> np.ndarray:
    """Distance from each pixel to the nearest epicenter; a 7.5 km grid first, all epicenters only for the rest."""
    if len(ex) <= 64:
        return np.min(np.hypot(px[:, None] - ex[None, :], py[:, None] - ey[None, :]), axis=1)
    out = np.full(len(px), np.inf)
    i, _, d = near_pairs(px, py, np.zeros(len(px)), ex, ey, np.zeros(len(ex)), limit=7.5)
    np.minimum.at(out, i, d)
    rest = np.flatnonzero(np.isinf(out))
    for k in range(0, len(rest), 2000):
        r = rest[k:k + 2000]
        out[r] = np.min(np.hypot(px[r, None] - ex[None, :], py[r, None] - ey[None, :]), axis=1)
    return out


# ---------------------------------------------------------------------------
# One fire
# ---------------------------------------------------------------------------

FOLLOW_COLUMNS = ["row_id", "source_fire_id", "pass_id", "t_abs", "instrument", "satellite", "latitude", "longitude",
                  "size", "scan", "track", "area", "frp_mw"]


def follow_fire(fire_id, g: pd.DataFrame, epis: pd.DataFrame):
    g = g[[c for c in FOLLOW_COLUMNS if c in g]]          # per-cluster row selections copy every column: keep only these
    lat0, lon0 = float(g.latitude.mean()), float(g.longitude.mean())
    x, y = local_xy(g.latitude.to_numpy(), g.longitude.to_numpy(), lat0, lon0)
    g = g.assign(x=x, y=y)
    ex, ey = local_xy(epis.latitude.to_numpy(), epis.longitude.to_numpy(), lat0, lon0)

    passes = g.groupby("pass_id", sort=False).t_abs.mean().sort_values()
    tracks, alive, events, steps, assign = {}, [], [], [], {}
    cum_all, cum_fine = set(), set()
    first_source_track = {}

    def new_track(rows, t, how, parents=(), fam=None):
        tid = f"{fire_id}.{len(tracks) + 1}"
        par = [tracks[p] for p in parents]
        srcs = set(rows.source_fire_id) | set().union(*[p["sources"] for p in par]) if par else set(rows.source_fire_id)
        tracks[tid] = {"cluster_id": tid, "fire_id": fire_id, "born": t, "born_how": how,
                       "parents": "+".join(parents), "first_seen_history": min([t] + [p["first_seen_history"] for p in par]),
                       "sources": srcs, "last_seen": t, "passes_seen": 1, "pixels": len(rows), "max_pixels": len(rows),
                       "foot": rows[["x", "y", "size"]].to_numpy(), "foot_family": fam,
                       "families": set(rows.instrument) | set().union(*[p["families"] for p in par]) if par else set(rows.instrument),
                       "end": None, "end_how": "", "children": ""}
        for s in set(rows.source_fire_id):
            first_source_track.setdefault(s, tid)
        alive.append(tid)
        for r in rows.row_id:
            assign[r] = tid
        return tid

    def end_track(tid, t, how, children):
        tr = tracks[tid]
        tr["end"], tr["end_how"], tr["children"] = t, how, "+".join(children)
        alive.remove(tid)

    for step, (pid, t) in enumerate(passes.items(), start=1):
        P = g[g.pass_id == pid]
        fam = P.instrument.iloc[0]
        px, py, ps = P.x.to_numpy(), P.y.to_numpy(), P["size"].to_numpy()
        n = len(P)
        if n > 1:
            i, j, _ = near_pairs(px, py, ps, px, py, ps)
            keep = i < j
            cl = components(n, i[keep], j[keep])
        else:
            cl = np.zeros(1, int)
        k_cl = cl.max() + 1
        members = groups_of(cl, k_cl)
        rows_of = lambda c: P.iloc[members[c]]
        # which alive tracks does each cluster touch? (all footprints at once, through a grid)
        touch = np.zeros((k_cl, len(alive)), bool)
        if alive:
            feet = [tracks[tid]["foot"] for tid in alive]
            F = np.concatenate(feet)
            owner = np.repeat(np.arange(len(alive)), [len(f) for f in feet])
            i, j, _ = near_pairs(px, py, ps, F[:, 0], F[:, 1], F[:, 2])
            touch[cl[i], owner[j]] = True
        step_events = []
        before = list(alive)
        if fam in FINE:
            # bipartite components: clusters 0..k_cl-1, tracks k_cl..
            ci, ti = np.nonzero(touch)
            comp = components(k_cl + len(before), ci, ti + k_cl)
            for grp in groups_of(comp, comp.max() + 1):
                C = [int(x) for x in grp if x < k_cl]
                T = [before[int(x) - k_cl] for x in grp if x >= k_cl]
                if not C:
                    continue
                if not T:
                    for c in C:
                        how = "ignition" if not tracks else "new burst"
                        tid = new_track(rows_of(c), t, how, fam=fam)
                        step_events.append((how, [tid], rows_of(c)))
                elif len(T) == 1 and len(C) == 1:
                    tr, rows = tracks[T[0]], rows_of(C[0])
                    tr.update(last_seen=t, passes_seen=tr["passes_seen"] + 1, pixels=tr["pixels"] + len(rows),
                              max_pixels=max(tr["max_pixels"], len(rows)), foot=rows[["x", "y", "size"]].to_numpy(),
                              foot_family=fam)
                    tr["sources"] |= set(rows.source_fire_id)
                    tr["families"].add(fam)
                    for src in set(rows.source_fire_id):         # a source fire first seen inside an existing cluster
                        first_source_track.setdefault(src, T[0])
                    for r in rows.row_id:
                        assign[r] = T[0]
                else:
                    how = "merge" if len(C) == 1 else "split" if len(T) == 1 else "merge and split"
                    kids = [new_track(rows_of(c), t, how, parents=T, fam=fam) for c in C]
                    for tid in T:
                        end_track(tid, t, how, kids)
                    step_events.append((how, T + ["->"] + kids, P.iloc[np.concatenate([members[c] for c in C])]))
        else:   # coarse MODIS pass: continue touched tracks, start tracks for untouched clusters, never merge/split
            for c in range(k_cl):
                rows = rows_of(c)
                hit = [before[a] for a in np.flatnonzero(touch[c])]
                if not hit:
                    how = "ignition" if not tracks else "new burst"
                    tid = new_track(rows, t, how, fam=fam)
                    step_events.append((how, [tid], rows))
                    continue
                for tid in hit:
                    tr = tracks[tid]
                    tr.update(last_seen=t, passes_seen=tr["passes_seen"] + 1, pixels=tr["pixels"] + len(rows),
                              max_pixels=max(tr["max_pixels"], len(rows)))
                    tr["families"].add(fam)
                    if tr["foot_family"] not in FINE:
                        tr["foot"], tr["foot_family"] = rows[["x", "y", "size"]].to_numpy(), fam
                # pixels belong to the nearest touched track
                cx, cy = rows.x.mean(), rows.y.mean()
                best = min(hit, key=lambda tid: np.hypot(tracks[tid]["foot"][:, 0] - cx, tracks[tid]["foot"][:, 1] - cy).min())
                for r in rows.row_id:
                    assign[r] = best
                tracks[best]["sources"] |= set(rows.source_fire_id)
                for src in set(rows.source_fire_id):
                    first_source_track.setdefault(src, best)

        cells = raster_cells(px, py, P.scan.to_numpy() / 2, P.track.to_numpy() / 2)
        new_all = len(set(cells.tolist()) - cum_all)
        cum_all.update(cells.tolist())
        if fam in FINE:
            cum_fine.update(cells.tolist())
        front = float(nearest_distance(px, py, ex, ey).max()) if len(ex) else math.nan
        for how, ids, rows in step_events:
            events.append({"fire_id": fire_id, "t_abs": t, "step": step, "pass_id": pid, "event": how,
                           "clusters": "+".join(ids).replace("+->+", " -> "), "pixels": len(rows),
                           "latitude": round(float(rows.latitude.mean()), 5), "longitude": round(float(rows.longitude.mean()), 5)})
        steps.append({"fire_id": fire_id, "step": step, "t_abs": t, "pass_id": pid, "satellite": P.satellite.iloc[0],
                      "family": fam, "clusters_seen": int(k_cl), "clusters_alive": len(alive), "pixels": n,
                      "burning_area_km2": round(float(P.area.sum()), 4),
                      "frp_sum_mw": round(float(P.frp_mw.sum()), 2) if P.frp_mw.notna().any() else math.nan,
                      "cumulative_footprint_km2": round(len(cum_all) * RASTER_KM ** 2, 3),
                      "cumulative_fine_km2": round(len(cum_fine) * RASTER_KM ** 2, 3),
                      "new_footprint_km2": round(new_all * RASTER_KM ** 2, 3),
                      "front_distance_km": round(front, 3),
                      "events": "; ".join(e[0] for e in step_events)})

    # where to look for extinction: each track still going, at its last footprint
    leaves = []
    for tid in alive:
        tr = tracks[tid]
        f = tr["foot"]
        lat = lat0 + float(f[:, 1].mean()) / 110.574
        lon = lon0 + float(f[:, 0].mean()) / (111.320 * math.cos(math.radians(lat0)))
        leaves.append((tid, lat, ((lon + 180) % 360) - 180, tr["last_seen"], "MODIS" in tr["families"]))
    return tracks, steps, events, assign, leaves, first_source_track


def flicker(evs: pd.DataFrame, tracks: dict) -> pd.Series:
    """True for a merge whose result splits again, or a split whose parts merge again, within FLICKER_MIN."""
    out = pd.Series(False, index=evs.index)
    cand = evs[evs.event.isin(["merge", "split", "merge and split"]) & evs.clusters.astype(str).str.contains(" -> ")]
    for k, e in cand.iterrows():
        kids = e.clusters.split(" -> ")[1].split("+")
        if e.event == "merge":                      # the merged track splits soon after
            tr = tracks[kids[0]]
            out[k] = tr["end_how"] in ("split", "merge and split") and tr["end"] is not None and tr["end"] - e.t_abs <= FLICKER_MIN
        else:                                       # two of the parts merge again soon after
            ends = [(tracks[c]["end_how"], tracks[c]["end"], tracks[c]["children"]) for c in kids]
            merged = [(t, ch) for how, t, ch in ends if how in ("merge", "merge and split") and t is not None
                      and t - e.t_abs <= FLICKER_MIN]
            out[k] = len(merged) >= 2 and len({ch for _, ch in merged}) < len(merged)
    if out.any():                                   # the event that undid it is marked too
        by_fire = {f: idx for f, idx in evs.groupby("fire_id").groups.items()}
        for k in out[out].index.tolist():
            kids = set(evs.at[k, "clusters"].split(" -> ")[1].split("+"))
            for m in by_fire[evs.at[k, "fire_id"]]:
                c = evs.at[m, "clusters"]
                if m > k and " -> " in c and set(c.split(" -> ")[0].split("+")) & kids:
                    out[m] = True
                    break
    return out


def single_pixel_fires(d: pd.DataFrame, ep: pd.DataFrame) -> tuple:
    """Fast path, same records as follow_fire, for fires that are one pixel on one pass (most fires):
    one ignition, one timeline step, a footprint of one pixel, built for all of them at once."""
    d = d.reset_index(drop=True)
    e = ep[ep.fire_id.isin(set(d.fire_id))][["fire_id", "latitude", "longitude"]]
    m = d[["fire_id", "latitude", "longitude"]].merge(e, on="fire_id", suffixes=("", "_e"))
    km = np.hypot(((m.longitude_e - m.longitude + 180) % 360 - 180) * 111.320 * np.cos(np.radians(m.latitude)),
                  (m.latitude_e - m.latitude) * 110.574)
    front = km.groupby(m.fire_id).min()
    a, b = d.scan.to_numpy() / 2, d.track.to_numpy() / 2       # pixel square centred on itself (x = y = 0)
    cells = ((np.floor(a / RASTER_KM) - np.floor(-a / RASTER_KM) + 1) * (np.floor(b / RASTER_KM) - np.floor(-b / RASTER_KM) + 1))
    foot_km2 = np.round(cells * RASTER_KM ** 2, 3)
    tracks, steps, events, assign, leaves, firsts = {}, [], [], {}, [], {}
    for k, r in enumerate(d.itertuples(index=False)):
        fid, t, tid = int(r.fire_id), float(r.t_abs), f"{int(r.fire_id)}.1"
        tracks[tid] = {"cluster_id": tid, "fire_id": fid, "born": t, "born_how": "ignition", "parents": "",
                       "first_seen_history": t, "sources": {r.source_fire_id}, "last_seen": t, "passes_seen": 1, "pixels": 1,
                       "max_pixels": 1, "foot": np.array([[0.0, 0.0, r.size]]), "foot_family": r.instrument,
                       "families": {r.instrument}, "end": None, "end_how": "", "children": ""}
        fine = r.instrument in FINE
        steps.append({"fire_id": fid, "step": 1, "t_abs": t, "pass_id": r.pass_id, "satellite": r.satellite,
                      "family": r.instrument, "clusters_seen": 1, "clusters_alive": 1, "pixels": 1,
                      "burning_area_km2": round(float(r.area), 4),
                      "frp_sum_mw": round(float(r.frp_mw), 2) if r.frp_mw == r.frp_mw else math.nan,
                      "cumulative_footprint_km2": foot_km2[k], "cumulative_fine_km2": foot_km2[k] if fine else 0.0,
                      "new_footprint_km2": foot_km2[k], "front_distance_km": round(float(front.get(fid, math.nan)), 3),
                      "events": "ignition"})
        events.append({"fire_id": fid, "t_abs": t, "step": 1, "pass_id": r.pass_id, "event": "ignition", "clusters": tid,
                       "pixels": 1, "latitude": round(float(r.latitude), 5), "longitude": round(float(r.longitude), 5)})
        assign[r.row_id] = tid
        leaves.append((tid, float(r.latitude), ((float(r.longitude) + 180) % 360) - 180, t, r.instrument == "MODIS"))
        firsts[r.source_fire_id] = tid
    return tracks, steps, events, assign, leaves, firsts


def follow_chunk(d: pd.DataFrame, ep: pd.DataFrame) -> tuple:
    """follow_fire for a chunk of fires (runs in a worker process)."""
    tracks, steps, events, assign, leaves, firsts = {}, [], [], {}, [], {}
    ep_by = dict(tuple(ep.groupby("fire_id")))
    empty = ep.iloc[:0]
    for fid, g in d.groupby("fire_id", sort=False):
        tr, st, ev, asg, lv, fst = follow_fire(int(fid), g, ep_by.get(fid, empty))
        tracks.update(tr)
        steps += st
        events += ev
        assign.update(asg)
        leaves += lv
        firsts.update(fst)
    return tracks, steps, events, assign, leaves, firsts


# ---------------------------------------------------------------------------
# Extinction: next proven look after the last sighting
# ---------------------------------------------------------------------------

def next_look(models, sat_times, lat, lon, after_abs, days):
    best = np.full(len(lat), np.nan)
    who = np.full(len(lat), "", dtype=object)
    for sat, m in models.items():
        limit = np.asarray(after_abs, float).copy()
        found = np.full(len(lat), np.nan)
        todo = np.ones(len(lat), bool)
        for _ in range(MAX_LOOKAHEAD_PASSES):
            if not todo.any():
                break
            cand = np.full(len(lat), np.nan)
            for day in days:
                day0 = (pd.Timestamp(day) - pd.Timestamp("1970-01-01")).days * 1440
                t = m.next_look_after(day, lat[todo], lon[todo], limit[todo] - day0, SWATH_MARGIN_KM) + day0
                c = cand[todo]
                cand[todo] = np.where(~np.isnan(t) & ~(c <= t), t, c)
            todo &= ~np.isnan(cand)
            ok = todo & has_data(sat_times[sat], np.nan_to_num(cand))
            found[ok] = cand[ok]
            todo &= ~ok
            limit[todo] = cand[todo] + 1
        better = ~np.isnan(found) & ~(best <= found)
        best = np.where(better, found, best)
        who = np.where(better, sat, who)
    return best, who


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(run_dir: Path) -> dict:
    t0 = time.time()
    need = {k: run_dir / v for k, v in {"rows": "correlation/rows.csv.gz", "ldet": "localisation/detections.csv.gz",
                                         "fmap": "epicenter/fire_map.csv.gz", "ep": "epicenter/epicenters.csv.gz",
                                         "mfires": "epicenter/fires.csv.gz"}.items()}
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run layers 1-4 first): " + ", ".join(missing))
    rows = pd.read_csv(need["rows"], dtype={"acq_time": str, "pass_id": str}, keep_default_na=False, na_values=[""],
                       low_memory=False)
    rows["row_id"] = np.arange(len(rows))
    rows["t_abs"] = (pd.to_datetime(rows.acq_date) - pd.Timestamp("1970-01-01")).dt.days * 1440 + rows.acq_minutes
    default = rows.instrument.map(DEFAULT_PIXEL_KM).fillna(1.0)
    rows["scan"], rows["track"] = rows.scan_km.fillna(default), rows.track_km.fillna(default)
    rows["size"] = np.fmax(rows.scan, rows.track)
    rows["area"] = rows.pixel_area_km2.fillna(rows.scan * rows.track)
    ldet = pd.read_csv(need["ldet"], dtype={"pass_id": str})
    fmap = pd.read_csv(need["fmap"])
    ep = pd.read_csv(need["ep"])
    d = ldet.rename(columns={"fire_id": "source_fire_id"}).merge(fmap, on="source_fire_id") \
            .merge(rows.drop(columns=["pass_id"]), on="row_id")
    T = Timer()

    # fires are independent: follow them in parallel, in chunks of similar size (largest fires spread out)
    sizes = d.fire_id.value_counts()
    tracks, steps, events, assign, leaves, firsts = single_pixel_fires(d[d.fire_id.isin(sizes[sizes == 1].index)], ep)
    d = d[d.fire_id.isin(sizes[sizes > 1].index)]
    sizes = sizes[sizes > 1]
    chunk_of = pd.Series(np.arange(len(sizes)) % (WORKERS * 4), index=sizes.index)
    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = []
        ep_chunk = ep.fire_id.map(chunk_of)
        ep_parts = dict(tuple(ep.groupby(ep_chunk)))
        for c, part in d.groupby(d.fire_id.map(chunk_of)):
            jobs.append(pool.submit(follow_chunk, part, ep_parts.get(c, ep.iloc[:0])))
        for j in jobs:
            tr, st, ev, asg, lv, fst = j.result()
            tracks.update(tr)
            steps += st
            events += ev
            assign.update(asg)
            leaves += lv
            firsts.update(fst)
    steps.sort(key=lambda r: (r["fire_id"], r["step"]))
    T.mark("follow fires (parallel)")

    # extinction looks for every track still going at the end
    sw = swath_rows(run_dir, rows)
    models = fit_swaths(sw)
    days = sorted(sw.acq_date.unique())
    sat_times = {s: np.sort(sw.loc[sw.satellite == s, "t_abs"].to_numpy(float)) for s in models}
    lv = pd.DataFrame(leaves, columns=["cluster_id", "lat", "lon", "last_seen", "seen_by_modis"])
    lv["look"], lv["look_by"] = np.nan, ""
    viirs_only = {s: m for s, m in models.items() if m.instrument == "VIIRS"}
    for modis_seen, group in lv.groupby("seen_by_modis"):
        use = models if modis_seen else viirs_only           # MODIS cannot confirm a fire only VIIRS could see
        look, who = next_look(use, sat_times, group.lat.to_numpy(), group.lon.to_numpy(), group.last_seen.to_numpy() + 5, days)
        lv.loc[group.index, "look"], lv.loc[group.index, "look_by"] = look, who
    for r in lv.itertuples():
        tr = tracks[r.cluster_id]
        if np.isnan(r.look):
            tr["end_how"] = "burning at data end"
        else:
            tr["end"], tr["end_how"], tr["out_seen_by"] = r.look, "extinguished", r.look_by
            events.append({"fire_id": tr["fire_id"], "t_abs": r.look, "step": None, "pass_id": "", "event": "extinguished",
                           "clusters": r.cluster_id, "pixels": 0, "latitude": round(r.lat, 5), "longitude": round(r.lon, 5)})

    T.mark("extinction looks")
    cl = pd.DataFrame(tracks.values())
    cl["sources"] = cl.sources.map(lambda s: "+".join(map(str, sorted(s))))
    cl["out_seen_by"] = cl.get("out_seen_by", pd.Series("", index=cl.index)).fillna("")
    cl["families"] = cl.families.map(lambda s: "+".join(sorted(s)))
    cl = cl.drop(columns=["foot", "foot_family"])
    tl = pd.DataFrame(steps)
    evs = pd.DataFrame(events).sort_values(["fire_id", "t_abs"], kind="stable", ignore_index=True)
    evs["undone_within_2h"] = flicker(evs, tracks)
    T.mark("tables + flicker")

    # per fire summary (grouped operations; no per-fire scans of whole tables)
    leaf = cl[cl.end_how.isin(["extinguished", "burning at data end"])]
    tg = tl.sort_values(["fire_id", "step"]).groupby("fire_id", sort=False)
    fs = pd.DataFrame({"burst": tg.t_abs.min(), "last_seen": tg.t_abs.max(), "passes": tg.size(),
                       "max_clusters_alive": tg.clusters_alive.max().astype(int),
                       "final_footprint_km2": tg.cumulative_footprint_km2.last(), "first_footprint_km2": tg.cumulative_footprint_km2.first(),
                       "final_fine_km2": tg.cumulative_fine_km2.last(), "max_front_distance_km": tg.front_distance_km.max()})
    life_h = (fs.last_seen - fs.burst) / 60
    fs.insert(2, "seen_for_h", life_h.round(2))
    lg = leaf.assign(out=leaf.end_how.eq("extinguished")).groupby("fire_id")
    all_out = lg.out.all().reindex(fs.index).fillna(False).astype(bool)
    fs.insert(3, "status", np.where(all_out, "extinguished", "burning at data end"))
    fs.insert(4, "extinguished_by", lg.end.max().reindex(fs.index).where(all_out))
    cg = cl.groupby("fire_id")
    fs["clusters_total"] = cg.size().reindex(fs.index).fillna(0).astype(int)
    fs["bursts"] = cl[cl.born_how.isin(["ignition", "new burst"])].groupby("fire_id").size().reindex(fs.index).fillna(0).astype(int)
    lasting = ~evs.undone_within_2h.astype(bool)
    count = lambda m: evs[m].groupby("fire_id").size().reindex(fs.index).fillna(0).astype(int)
    fs["merges"] = count(evs.event.isin(["merge", "merge and split"]) & lasting)
    fs["splits"] = count(evs.event.isin(["split", "merge and split"]) & lasting)
    fs["flicker_events"] = count(evs.undone_within_2h.astype(bool))
    fs["footprint_growth_km2_h"] = ((fs.final_footprint_km2 - fs.first_footprint_km2) / life_h.where(life_h > 0)).round(4)
    fs = fs.drop(columns=["first_footprint_km2"]).reset_index()
    fs = fs[["fire_id", "burst", "last_seen", "seen_for_h", "status", "extinguished_by", "passes", "clusters_total", "bursts",
             "merges", "splits", "flicker_events", "max_clusters_alive", "final_footprint_km2", "final_fine_km2",
             "max_front_distance_km", "footprint_growth_km2_h"]]
    for df, cols in ((fs, ["burst", "last_seen", "extinguished_by"]), (tl, ["t_abs"]), (evs, ["t_abs"]),
                     (cl, ["born", "first_seen_history", "last_seen", "end"])):
        for c in cols:
            df[c if c != "t_abs" else "time"] = iso_many(df[c])
    tl, evs = tl.drop(columns=["t_abs"]), evs.drop(columns=["t_abs"])
    det = pd.DataFrame({"row_id": list(assign), "cluster_id": list(assign.values())})
    det["fire_id"] = det.cluster_id.str.split(".").str[0].astype(int)
    first = pd.DataFrame({"source_fire_id": list(firsts), "first_cluster_id": list(firsts.values())})

    T.mark("fire summaries")
    out = run_dir / "expansion"
    out.mkdir(parents=True, exist_ok=True)
    fs.sort_values("clusters_total", ascending=False).to_csv(out / "fires.csv.gz", index=False)
    tl.to_csv(out / "timeline.csv.gz", index=False)
    cl.to_csv(out / "clusters.csv.gz", index=False)
    evs.to_csv(out / "events.csv.gz", index=False)
    det.to_csv(out / "detections.csv.gz", index=False)
    first.to_csv(out / "source_first_clusters.csv.gz", index=False)
    summary = {
        "fires": len(fs), "clusters": len(cl), "timeline_steps": len(tl), "detections": len(det),
        "status": {k: int(v) for k, v in fs.status.value_counts().items()},
        "events": {k: int(v) for k, v in evs.event.value_counts().items()},
        "undone_within_2h": int(evs.undone_within_2h.sum()),
        "fires_with_merges": int((fs.merges > 0).sum()), "fires_with_splits": int((fs.splits > 0).sum()),
        "median_seen_for_h": round(float(fs.seen_for_h.median()), 2),
        "settings": {"raster_km": RASTER_KM, "merge_split_decided_by": sorted(FINE), "swath_margin_km": SWATH_MARGIN_KM},
        "seconds": round(time.time() - t0, 1),
    }
    T.mark("write")
    summary["timings"] = T.parts
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"fires": fs, "timeline": tl, "clusters": cl, "events": evs, "summary": summary, "out": out}


def print_summary(res: dict) -> None:
    s, f = res["summary"], res["fires"]
    print(f"\n{s['fires']:,} fires followed through {s['timeline_steps']:,} passes in {s['seconds']} s; "
          f"{s['clusters']:,} clusters tracked")
    print("Status: " + ", ".join(f"{k} {v:,}" for k, v in s["status"].items()))
    print("Events: " + ", ".join(f"{k} {v:,}" for k, v in s["events"].items()))
    print(f"Merges/splits undone within 2 h (flicker): {s['undone_within_2h']:,}")
    print(f"Fires with lasting merges {s['fires_with_merges']:,}, with splits {s['fires_with_splits']:,}; "
          f"median time seen {s['median_seen_for_h']} h")
    cols = ["fire_id", "passes", "clusters_total", "bursts", "merges", "splits", "max_clusters_alive",
            "final_fine_km2", "max_front_distance_km", "status", "burst", "extinguished_by"]
    print("\nFires with the most clusters:")
    print(f.sort_values(["clusters_total", "passes"], ascending=False).head(8)[cols].to_string(index=False))


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
