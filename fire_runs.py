"""Select the major and medium fires of the 10-year archive scan and run the full pipeline on them.

Works only from the files archive_scan.py downloaded (data/archive/): no FIRMS
requests. Output: one pipeline run per batch of fires under data/runs/, each
with its own knowledge-graph files (ids prefixed by the run name).

Selection (data/archive/selected_fires.csv)
- Major fires: connected burning 0.1 degree cell-days with FRP >= 500 MW each
  (the scan's big-fire events), seen on at least 2 days. Picked per REGION and
  spread over the years, so the set is not just the boreal and Australian
  megafires that dominate a plain global top list: each region gets a share in
  proportion to its candidates, with a floor, and within a region the largest
  fires are taken round-robin by year.
- Medium fires: connected cell-days with FRP >= 100 MW each, 2-20 cells, at
  least 2 days, never reaching 500 MW in one cell-day, not overlapping a major
  fire; a random sample per region and year (fixed seed). They keep the
  expansion model from learning that every fire grows huge.
Regions are latitude/longitude boxes (a stand-in for fire type: boreal, savanna,
tropical forest, Mediterranean, ...).

Runs
- Fires are sorted by start date and processed in batches of BATCH fires. A
  batch's input is every downloaded MODIS / VIIRS detection inside each fire's
  box (+0.1 degree) from the day before it starts to the day after it ends,
  harmonised exactly like the app does (type kept from the archive files).
- swath_context.csv.gz: worldwide Aqua / Suomi NPP / NOAA-20 detections of the
  same days (a sample per satellite and day plus one per minute of data), so
  epicenter and expansion can reconstruct satellite passes and prove data
  exists; one fire's box alone has too few detections for that.
- Stages: correlation -> interpreter -> localisation + fire power -> epicenter
  -> fire expansion -> KG feed 1. Several batches run at once.

Usage:  python fire_runs.py --major 1000 --medium 1000 [--batch 50] [--parallel 3]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import archive_scan as A
from firms import DATA_DIR

RUNS = DATA_DIR / "runs"
PAD_DEG = 0.1
CONTEXT_PER_SAT_DAY = 3000
SWATH_SATS = {"Aqua": "Aqua", "A": "Aqua", "N": "Suomi NPP", "N20": "NOAA-20", "1": "NOAA-20"}   # as named in FIRMS files
REGIONS = [  # (name, south, north, west, east), first match wins
    ("boreal North America", 50, 75, -170, -50), ("western North America", 15, 50, -130, -100),
    ("eastern North America", 15, 50, -100, -50), ("Siberia and boreal Asia", 50, 80, 45, 180),
    ("Europe and Mediterranean", 30, 72, -25, 45), ("Middle East and Central Asia", 25, 50, 45, 90),
    ("South Asia", 5, 35, 60, 90), ("East Asia", 20, 50, 90, 150), ("Southeast Asia and Indonesia", -11, 20, 90, 160),
    ("northern Australia", -25, -10, 110, 155), ("southern Australia", -45, -25, 110, 155),
    ("Central America and Caribbean", 5, 25, -120, -55), ("Amazon and northern South America", -15, 12, -82, -34),
    ("southern South America", -56, -15, -82, -34), ("North Africa savanna", 0, 25, -20, 52),
    ("southern Africa", -36, 0, 8, 52), ("other", -90, 90, -180, 180)]


def region_of(lat: float, lon: float) -> str:
    return next(n for n, s, no, w, e in REGIONS if s <= lat < no and w <= lon < e)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def events(files: list, frp_min: float) -> pd.DataFrame:
    old = A.BIG_FRP_MW
    A.BIG_FRP_MW = frp_min
    try:
        return A.big_fires(files, 10 ** 9)
    finally:
        A.BIG_FRP_MW = old


def stratified(ev: pd.DataFrame, n: int, floor: int, rank_col: str | None, seed: int) -> pd.DataFrame:
    ev = ev.copy()
    ev["region"] = [region_of((s + no) / 2, (w + e) / 2) for s, no, w, e in zip(ev.south, ev.north, ev.west, ev.east)]
    ev["year"] = ev.start.str[:4]
    counts = ev.region.value_counts()
    quota = (counts / counts.sum() * n).clip(lower=np.minimum(counts, floor)).round().astype(int)
    while quota.sum() > n:                      # trim the largest regions back to the total
        quota[quota.idxmax()] -= 1
    rng = np.random.default_rng(seed)
    picked = []
    for region, q in quota.items():
        g = ev[ev.region == region]
        g = g.sort_values(rank_col, ascending=False) if rank_col else g.sample(frac=1, random_state=int(rng.integers(1e9)))
        g = g.assign(k=g.groupby("year").cumcount()).sort_values(["k", rank_col or "k"], ascending=[True, False])
        picked.append(g.head(int(min(q, len(g)))))   # round-robin over years, largest first within each round
    return pd.concat(picked, ignore_index=True).drop(columns=["k"])


def select(n_major: int, n_medium: int) -> pd.DataFrame:
    files = sorted((A.ARCHIVE / "cellday").glob("*.csv.gz"))
    if not files:
        raise SystemExit("No archive scan output in data/archive/cellday. Run archive_scan.py first.")
    t0 = time.time()
    big = events(files, 500)
    big = big[big.days >= 2]
    major = stratified(big, n_major, floor=30, rank_col="frp_total_mw", seed=1).assign(kind="major")
    med = events(files, 100)
    med = med[(med.days >= 2) & med.cells.between(2, 20) & (med.frp_peak_cell_mw < 500)]
    # not overlapping a major fire in space and time
    over = np.zeros(len(med), bool)
    for m in major.itertuples():
        over |= ((med.south < m.north) & (med.north > m.south) & (med.west < m.east) & (med.east > m.west) &
                 (med.start <= m.end) & (med.end >= m.start)).to_numpy()
    medium = stratified(med[~over], n_medium, floor=20, rank_col=None, seed=2).assign(kind="medium")
    sel = pd.concat([major, medium], ignore_index=True).sort_values("start", ignore_index=True)
    sel.insert(0, "fire_key", [f"{k[0].upper()}{i:04d}" for i, k in enumerate(sel.kind)])
    sel.to_csv(A.ARCHIVE / "selected_fires.csv", index=False)
    print(f"selected {len(major)} major (of {len(big):,} candidates) and {len(medium)} medium (of {int((~over).sum()):,}) "
          f"fires in {time.time() - t0:.0f} s; regions: {major.region.nunique()} / {medium.region.nunique()}", flush=True)
    return sel


# ---------------------------------------------------------------------------
# Building a batch's input from the downloaded files
# ---------------------------------------------------------------------------

def windows_for(d0: date, d1: date, plan: dict) -> list:
    return [w for w in plan if w <= d1 and w + timedelta(days=A.WINDOW - 1) >= d0]


def raw_frame(sensor: str, w: date) -> pd.DataFrame | None:
    p = A.raw_path(sensor, w)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df if len(df) else None


def build_batch(fires: pd.DataFrame, plan: dict, run_dir: Path) -> int:
    from server import HARMONISED_COLUMNS, TYPE_REF, add_type_columns, harmonise_rows
    days, rows_out, ctx = set(), [], []
    spans = []
    for f in fires.itertuples():
        d0, d1 = date.fromisoformat(f.start) - timedelta(days=1), date.fromisoformat(f.end) + timedelta(days=1)
        spans.append((f, d0, d1))
        days |= {(d0 + timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)}
    need = sorted({(s, w) for f, d0, d1 in spans for w in windows_for(d0, d1, plan) for _, s, *_ in plan[w]})
    harmonised = []
    for sensor, w in need:
        df = raw_frame(sensor, w)
        if df is None:
            continue
        df = df[df.acq_date.isin(days)]
        lat, lon = df.latitude.astype(float), df.longitude.astype(float)
        inside = np.zeros(len(df), bool)
        for f, d0, d1 in spans:
            inside |= ((lat >= f.south - PAD_DEG) & (lat <= f.north + PAD_DEG) & (lon >= f.west - PAD_DEG) &
                       (lon <= f.east + PAD_DEG) & (df.acq_date >= d0.isoformat()) & (df.acq_date <= d1.isoformat())).to_numpy()
        sub = df[inside]
        if len(sub):
            harmonised += harmonise_rows(sensor, {"columns": list(sub.columns), "rows": sub.values.tolist()}, keep_scan_track=True)
        sat = df.satellite.map(SWATH_SATS)
        c = df[sat.notna()].assign(satellite=sat[sat.notna()])
        if len(c):
            t = c.acq_time.str.zfill(4)
            c = c.assign(acq_minutes=t.str[:2].astype(int) * 60 + t.str[2:].astype(int))
            keep = c.sample(frac=1, random_state=0).groupby(["satellite", "acq_date"]).head(CONTEXT_PER_SAT_DAY)
            minute = c.drop_duplicates(["satellite", "acq_date", "acq_minutes"])
            c = pd.concat([keep, minute]).drop_duplicates()
            ctx.append(pd.DataFrame({"satellite": c.satellite, "instrument": np.where(c.satellite == "Aqua", "MODIS", "VIIRS"),
                                     "acq_date": c.acq_date, "acq_minutes": c.acq_minutes,
                                     "latitude": c.latitude.astype(float), "longitude": c.longitude.astype(float),
                                     "daynight": c.daynight,
                                     "pixel_area_km2": c.scan.astype(float) * c.track.astype(float)}))
    # same detection can fall in two fires' boxes: keep it once
    seen, uniq = set(), []
    for r in harmonised:
        k = (r[0], r[3], r[4], r[5], r[6])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    base = HARMONISED_COLUMNS[:-1] + ["scan_km", "track_km", "type"]
    cols, rows = add_type_columns(TYPE_REF, base, uniq)
    from correlation import run_correlation, save
    res = run_correlation(cols, rows)
    res["report"]["run"] = {"date": min(days), "days": len(days), "area": "selected fires",
                            "fires": fires.fire_key.tolist(), "sources": [], "range": [min(days), max(days)]}
    save(res, run_dir / "correlation")
    pd.concat(ctx, ignore_index=True).to_csv(run_dir / "swath_context.csv.gz", index=False)
    fires.to_csv(run_dir / "fires_in_batch.csv", index=False)
    return len(uniq)


STAGES = [("interpreter.py", ["--packages", "4"]), (("localisation_worker.py", "fire_power_worker.py"), ["--processes", "3"]),
          ("epicenter_worker.py", []), ("fire_expansion_worker.py", []), ("kg_builder.py", [])]


def run_batch(k: int, fires: pd.DataFrame, plan: dict) -> dict:
    run_dir = RUNS / f"fires_batch_{k:03d}"
    if (run_dir / "kg" / "summary.json").exists():
        return {"batch": k, "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = build_batch(fires, plan, run_dir)
    times = {"build + correlation": round(time.time() - t0, 1)}
    for scripts, extra in STAGES:
        t1 = time.time()
        scripts = scripts if isinstance(scripts, tuple) else (scripts,)
        procs = [subprocess.Popen([sys.executable, "-X", "utf8", s, "--run", str(run_dir), *extra],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for s in scripts]
        outs = [p.communicate() for p in procs]
        if any(p.returncode for p in procs):
            err = "\n".join(e[-1500:] for _, e in outs)
            (run_dir / "error.txt").write_text(err, encoding="utf-8")
            return {"batch": k, "failed": scripts, "error": err[-400:]}
        times[" + ".join(scripts)] = round(time.time() - t1, 1)
    out = {"batch": k, "fires": len(fires), "detections": n, "seconds": round(time.time() - t0, 1), "stages": times}
    (run_dir / "batch.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--major", type=int, default=1000)
    ap.add_argument("--medium", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--start", default="2016-11-14")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--reselect", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N batches (for a test)")
    args = ap.parse_args()
    from firms import fetch_availability
    plan = A.plan_tasks(date.fromisoformat(args.start), date.fromisoformat(args.end), fetch_availability())
    selp = A.ARCHIVE / "selected_fires.csv"
    sel = pd.read_csv(selp) if selp.exists() and not args.reselect else select(args.major, args.medium)
    batches = [sel.iloc[i:i + args.batch] for i in range(0, len(sel), args.batch)]
    if args.limit:
        batches = batches[:args.limit]
    t0 = time.time()
    print(f"{len(sel)} fires in {len(batches)} batches, {args.parallel} at a time", flush=True)
    results = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for r in pool.map(lambda kb: run_batch(kb[0], kb[1], plan), enumerate(batches)):
            results.append(r)
            state = "skipped (done)" if r.get("skipped") else f"FAILED in {r['failed']}" if r.get("failed") else \
                f"{r['detections']:,} detections in {r['seconds']} s"
            print(f"  batch {r['batch']:03d}: {state}  [{len(results)}/{len(batches)}, {(time.time() - t0) / 60:.1f} min]", flush=True)
    (A.ARCHIVE / "fire_runs.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failed = [r for r in results if r.get("failed")]
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {len(results) - len(failed)} batches ok, {len(failed)} failed")


if __name__ == "__main__":
    main()
