"""Dataset of fire merges: every observed merge in every run, plus the predicted merge outlook of runs with predictions.

1. Observed merges  ->  data/prediction/merges.csv.gz  (+ merges_summary.json)
   One row per merge event of the expansion worker (two or more fire clusters joining in one satellite pass):
     when / where       time, lat, lon, run, fire_id, pass, event ("merge" or "merge and split")
     what joined        parent and child clusters, number of parents, separate_fires (the parents came from
                        different source fires, so two separate fires joined, not a fire re-joining itself),
                        undone_within_2h (the join fell apart again within 2 h)
     the two largest parents (a = larger, b = smaller): detections, FRP sum (MW), burned pixel area (km2),
                        age at the merge (h), how each was born (ignition / split / ...)
     geometry           distance between the parents' centres and the gap between their nearest pixels (km),
                        bearing from a to b
     weather            temperature, humidity, wind speed / direction / gusts, VPD, hours since rain at the merge
                        hour (the run's weather feed), and wind_alignment_deg: angle between where the wind blows
                        to and the direction from a to b (0 = the wind pushes a straight towards b)
2. Merge outlook    ->  <run>/prediction/merge_outlook.csv  for every run with predictions
   For each predicted fire and day ahead: its nearest active neighbour, centre distance and the gap left
   between their predicted likely footprints, with the predicted chance of merging.

Usage:  python merges_dataset.py [--runs fires_batch_039 ...]
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from firms import DATA_DIR
from prediction_data import KM_LAT, KM_LON
from weather_worker import weather_key

OUT = DATA_DIR / "prediction"
WX = {"temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct", "wind_speed_10m": "wind_kmh",
      "wind_direction_10m": "wind_from_deg", "wind_gusts_10m": "gust_kmh", "vapour_pressure_deficit": "vpd_kpa",
      "hours_since_rain": "hours_since_rain"}


def read(kg: Path, name: str, **kw):
    p = kg / f"{name}.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p, low_memory=False, **kw)
    d.columns = [c.split(":")[0] for c in d.columns]
    return d


def xy(lat, lon, lat0):
    return np.c_[np.asarray(lon) * KM_LON * math.cos(math.radians(lat0)), np.asarray(lat) * KM_LAT]


def bearing(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * KM_LON * np.cos(np.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * KM_LAT
    return np.degrees(np.arctan2(dx, dy)) % 360


def observed(run_dir: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    kg = run_dir / "kg"
    ev = read(kg, "nodes_ExpansionEvent", dtype={"clusters:string": str})
    if ev is None or ev.empty:
        return pd.DataFrame()
    ev = ev[ev.event.isin(["merge", "merge and split"])].copy()
    if ev.empty:
        return pd.DataFrame()
    cl = read(kg, "nodes_Cluster", dtype={"cluster_id:string": str, "sources:string": str, "parents:string": str, "children:string": str})
    cl = cl.set_index("cluster_id")
    det = read(kg, "nodes_Detection", usecols=["id:ID(Detection)", "latitude:double", "longitude:double", "time:datetime",
                                               "frp_mw:double", "pixel_area_km2:double"])
    inc = pd.read_csv(kg / "rels_IN_CLUSTER.csv.gz")
    inc.columns = ["det", "cl", "type"]
    inc["cluster_id"] = inc.cl.str.rsplit("|", n=1).str[-1]
    det = det.merge(inc[["det", "cluster_id"]], left_on="id", right_on="det")
    ev["parents"] = ev.clusters.astype(str).str.split(" -> ").str[0]
    ev["children"] = ev.clusters.astype(str).str.split(" -> ").str[1]
    need = set(p for s in ev.parents for p in s.split("+"))
    det = det[det.cluster_id.isin(need)]
    by = {k: g for k, g in det.groupby("cluster_id")}
    stats = det.groupby("cluster_id").agg(n=("id", "size"), frp=("frp_mw", "sum"), area=("pixel_area_km2", "sum"),
                                          lat=("latitude", "mean"), lon=("longitude", "mean"))

    # weather at the merge hour
    wh = read(kg, "nodes_WeatherHour", usecols=lambda c: c.split(":")[0] in {"weather_cell", "time", *WX})
    wcell = read(kg, "nodes_WeatherCell")
    ev["time"] = pd.to_datetime(ev.time, utc=True)
    rows = []
    for e in ev.itertuples():
        ps = [p for p in e.parents.split("+") if p in stats.index]
        if len(ps) < 2:
            continue
        ps = sorted(ps, key=lambda p: -stats.n[p])
        a, b = ps[0], ps[1]
        sa, sb = stats.loc[a], stats.loc[b]
        sets = [set(str(cl.sources.get(p, "")).split("+")) for p in e.parents.split("+")]
        separate = any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        da, db = by[a], by[b]
        A, B = xy(da.latitude, da.longitude, e.latitude), xy(db.latitude, db.longitude, e.latitude)
        gap = float(cKDTree(A).query(B, k=1)[0].min())
        age = lambda p: (e.time - pd.to_datetime(cl.born.get(p), utc=True)).total_seconds() / 3600 if p in cl.index else np.nan
        cd = float(np.hypot(*(xy([sa.lat], [sa.lon], e.latitude) - xy([sb.lat], [sb.lon], e.latitude))[0]))
        child = e.children.split("+")[0] if isinstance(e.children, str) else None
        rows.append({
            "run": run_dir.name, "event_id": e.id, "time": e.time.strftime("%Y-%m-%dT%H:%MZ"), "date": e.time.strftime("%Y-%m-%d"),
            "event": e.event, "lat": e.latitude, "lon": e.longitude, "fire_id": int(str(a).split(".")[0]), "pass": e.pass_id,
            "parents": e.parents, "children": e.children, "n_parents": len(e.parents.split("+")),
            "separate_fires": separate, "undone_within_2h": str(e.undone_within_2h) == "True",
            "parent_a": a, "parent_b": b, "a_detections": int(sa.n), "b_detections": int(sb.n),
            "a_frp_mw": round(sa.frp, 1), "b_frp_mw": round(sb.frp, 1), "a_area_km2": round(sa.area, 3), "b_area_km2": round(sb.area, 3),
            "a_age_h": round(age(a), 2), "b_age_h": round(age(b), 2),
            "a_born_how": cl.born_how.get(a), "b_born_how": cl.born_how.get(b),
            "centroid_distance_km": round(cd, 3), "gap_km": round(gap, 3),
            "bearing_a_to_b_deg": round(float(bearing(sa.lat, sa.lon, sb.lat, sb.lon)), 1),
            "pixels_after": e.pixels, "child_max_pixels": cl.max_pixels.get(child) if child else np.nan})
    m = pd.DataFrame(rows)
    if m.empty:
        return m
    # weather join: the merge's 0.25 degree cell (or nearest fetched within 0.5 degree), same UTC hour
    for c in WX.values():
        m[c] = np.nan
    m["weather_source"] = None
    if wh is not None and wcell is not None and len(wh):
        wh["hour"] = pd.to_datetime(wh.time, utc=True).dt.floor("h")
        wh = wh.drop_duplicates(["weather_cell", "hour"]).set_index(["weather_cell", "hour"])
        own = weather_key(m.lat, m.lon).to_numpy()
        have = set(wcell.weather_cell)
        cl_lat, cl_lon = wcell.lat.to_numpy(), wcell.lon.to_numpy()
        cells = []
        for i, k in enumerate(own):
            if k in have:
                cells.append(k)
                continue
            d = np.hypot(cl_lat - m.lat.iat[i], (cl_lon - m.lon.iat[i]) * math.cos(math.radians(m.lat.iat[i])))
            j = int(d.argmin())
            cells.append(wcell.weather_cell.iat[j] if d[j] <= 0.5 else None)
        keys = list(zip(cells, pd.to_datetime(m.time, utc=True).dt.floor("h")))
        hit = [k in wh.index for k in keys]
        sub = wh.reindex([k for k, h in zip(keys, hit) if h])
        for src, dst in WX.items():
            if src in sub:
                m.loc[np.array(hit), dst] = sub[src].to_numpy()
        m.loc[np.array(hit), "weather_source"] = "ERA5 (run weather feed)"
    blow_to = (m.wind_from_deg + 180) % 360
    m["wind_alignment_deg"] = (np.abs((m.bearing_a_to_b_deg - blow_to + 180) % 360 - 180)).round(1)
    return m


def outlook(run_dir: Path):
    p = run_dir / "prediction" / "predictions.csv"
    if not p.exists():
        return None
    P = pd.read_csv(p)
    out = []
    for k, g in P.groupby("day_ahead"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        lat0 = float(g.centre_lat.mean())
        Z = xy(g.centre_lat, g.centre_lon, lat0)
        d, j = cKDTree(Z).query(Z, k=2)
        nb = g.iloc[j[:, 1]].reset_index(drop=True)
        r = pd.DataFrame({"run": run_dir.name, "fire_id": g.fire_id, "day_ahead": k, "date": g.date,
                          "lat": g.centre_lat, "lon": g.centre_lon, "footprint_likely_km2": g.footprint_likely_km2,
                          "p_merge": g.p_merge, "partner_fire_id": nb.fire_id, "partner_lat": nb.centre_lat, "partner_lon": nb.centre_lon,
                          "partner_footprint_likely_km2": nb.footprint_likely_km2, "partner_p_merge": nb.p_merge,
                          "centre_distance_km": d[:, 1].round(3)})
        ra = np.sqrt(g.footprint_likely_km2 / math.pi)
        rb = np.sqrt(nb.footprint_likely_km2 / math.pi)
        r["gap_between_likely_footprints_km"] = (r.centre_distance_km - ra - rb).clip(lower=0).round(3)
        out.append(r)
    if not out:
        return None
    o = pd.concat(out, ignore_index=True).sort_values(["fire_id", "day_ahead"])
    o.to_csv(run_dir / "prediction" / "merge_outlook.csv", index=False)
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()
    t0 = time.time()
    runs = [r for r in sorted((DATA_DIR / "runs").glob("*")) if (r / "kg" / "nodes_ExpansionEvent.csv.gz").exists()]
    if args.runs:
        runs = [r for r in runs if r.name in args.runs]
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r, m in zip(runs, pool.map(observed, map(str, runs))):
            print(f"  {r.name}: {len(m):,} merges", flush=True)
            parts.append(m)
    M = pd.concat([p for p in parts if len(p)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    M.to_csv(OUT / "merges.csv.gz", index=False)
    sep = M[M.separate_fires & ~M.undone_within_2h]
    summary = {
        "merges": len(M), "runs": int(M.run.nunique()),
        "separate_fires_lasting": len(sep), "re_joins_of_same_fire": int((~M.separate_fires).sum()),
        "undone_within_2h": int(M.undone_within_2h.sum()),
        "with_weather": round(float(M.wind_kmh.notna().mean()), 3),
        "separate_lasting": {
            "median_gap_km": round(float(sep.gap_km.median()), 3),
            "median_centroid_distance_km": round(float(sep.centroid_distance_km.median()), 3),
            "median_age_smaller_h": round(float(sep.b_age_h.median()), 1),
            "median_wind_kmh": round(float(sep.wind_kmh.median()), 1) if sep.wind_kmh.notna().any() else None,
            "median_rh_pct": round(float(sep.rh_pct.median()), 1) if sep.rh_pct.notna().any() else None,
            "share_wind_within_45deg_of_a_to_b": round(float((sep.wind_alignment_deg <= 45).mean()), 3)
            if sep.wind_alignment_deg.notna().any() else None,
            "share_wind_within_45deg_expected_if_random": 0.25},
        "seconds": round(time.time() - t0, 1)}
    (OUT / "merges_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in runs:
        o = outlook(r)
        if o is not None:
            print(f"  outlook {r.name}: {len(o)} rows -> {r / 'prediction' / 'merge_outlook.csv'}")


if __name__ == "__main__":
    main()
