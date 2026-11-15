"""10-year archive scan, MODIS + VIIRS: fire frequency map and the biggest fires.

Downloads every MODIS and VIIRS fire detection worldwide for the period (FIRMS
archive "SP" files, then near-real-time "NRT" for the months the archive does
not cover yet), in 5-day world chunks, 3 downloads at a time. Each 5-day window
is aggregated as soon as its files are in, in time order, while later windows
keep downloading. Raw files are kept (data/archive/raw/) so the fire pipeline
can reuse them for the biggest fires without downloading again; an interrupted
scan resumes where it stopped.

Wildfires only: archive files carry the real `type` (0 = vegetation fire); for
near-real-time months, 0.1 degree cells that the archive shows are mostly static
sources (type 1-3: volcanoes, gas flares, industry, offshore) are masked out.

Sensors change over the decade (one VIIRS satellite until 2018, three from 2023),
so the map counts FIRE DAYS per cell (a cell burned on a day if any satellite saw
it), not detections, and keeps MODIS and VIIRS fire days apart. The heads-up uses
the share of years with fire in each month, which hardly depends on the number
of satellites; the trend uses MODIS only (the same instrument all 10 years).

Outputs (data/archive/)
  cellday/<window>.csv.gz   0.1 degree cell x day: MODIS and VIIRS detections, FRP
  frequency.csv.gz          cell x calendar month: years with fire, share of years,
                            mean fire days, FRP, MODIS/VIIRS years, trend, heads-up level
  big_fires.csv.gz          the largest fire events (connected burning cells and days)
  summary.json

Usage:  python archive_scan.py --start 2016-11-14 --end 2026-11-13 [--threads 8] [--top 1000]
Keys: FIRMS_MAP_KEYS in .env (comma separated, issued by FIRMS for this project); each download uses the
next key with room left (checked with the free mapkey_status call).
"""

import argparse
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from correlation import components
import os

from firms import DATA_DIR, FIRMS_API, MAP_KEY, fetch_availability, firms_get

ARCHIVE = DATA_DIR / "archive"
WINDOW = 5
CELL = 0.1
# MODIS throughout; one VIIRS satellite at a time: Suomi NPP until NOAA-20 starts (April 2018), then NOAA-20.
# The three VIIRS satellites pass within ~20 min of each other, so one is enough for big fires (87% of FRP alone).
SATELLITES = [("MODIS", "MODIS_SP", "MODIS_NRT"), ("VIIRS", "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
              ("VIIRS", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT")]
HANDOVER = {"VIIRS_SNPP_SP": "VIIRS_NOAA20_SP", "VIIRS_SNPP_NRT": "VIIRS_NOAA20_SP"}   # use only before this one starts
BIG_FRP_MW = 500          # a cell-day joins a big-fire event when its FRP sum is at least this (100 chained whole savanna regions)
STATIC_SHARE = 0.5        # cells where >= half the archive detections are static sources are masked
QUOTA_RESERVE = 800       # FIRMS counts transactions per 10 min; one world 5-day request costs several hundred
_quota_lock = threading.Lock()
HEADS_UP = [(0.7, "high"), (0.4, "medium"), (0.15, "low"), (0.0, "minimal")]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def plan_tasks(start: date, end: date, avail: dict) -> dict:
    """window start -> list of (family, sensor, kind, keep_from, keep_to)."""
    windows = {}
    d = start
    while d <= end:
        windows[d] = []
        d += timedelta(days=WINDOW)
    for family, sp, nrt in SATELLITES:
        a_sp, a_nrt = avail.get(sp), avail.get(nrt)
        if sp in HANDOVER and avail.get(HANDOVER[sp]):        # stop where the next satellite takes over
            cut = (date.fromisoformat(avail[HANDOVER[sp]]["min"]) - timedelta(days=1)).isoformat()
            a_sp = {**a_sp, "max": min(a_sp["max"], cut)} if a_sp else None
            a_nrt = None
        sp_from = date.fromisoformat(a_sp["min"]) if a_sp else None
        sp_to = date.fromisoformat(a_sp["max"]) if a_sp else None
        for w in windows:
            w_end = min(w + timedelta(days=WINDOW - 1), end)
            if sp_from and w_end >= sp_from and w <= sp_to:
                windows[w].append((family, sp, "SP", max(w, sp_from), min(w_end, sp_to)))
            nrt_from = max(sp_to + timedelta(days=1) if sp_to else w, date.fromisoformat(a_nrt["min"])) if a_nrt else None
            nrt_to = date.fromisoformat(a_nrt["max"]) if a_nrt else None
            if a_nrt and w_end >= nrt_from and w <= nrt_to:
                windows[w].append((family, nrt, "NRT", max(w, nrt_from), min(w_end, nrt_to)))
    return windows


def raw_path(sensor: str, w: date, base: Path | None = None) -> Path:
    return (base or ARCHIVE) / "raw" / sensor / f"{w.isoformat()}.csv.gz"


KEYS = [k for k in os.environ.get("FIRMS_MAP_KEYS", "").split(",") if k] or [MAP_KEY]   # keys FIRMS issued for this project


def quota_status(key: str = MAP_KEY) -> tuple[int, int]:
    """(transactions used in the current 10-minute window, limit) for one key. Checking costs no transaction."""
    with urllib.request.urlopen(f"{FIRMS_API}/mapkey_status/?MAP_KEY={key}", timeout=60) as r:
        st = json.loads(r.read())
    return int(st["current_transactions"]), int(st["transaction_limit"])


_next_key = [0]


def take_key() -> str:
    """The next key (round robin) that has room for one large request; waits only if every key is near its limit."""
    while True:
        with _quota_lock:
            start = _next_key[0]
            for n in range(len(KEYS)):
                k = KEYS[(start + n) % len(KEYS)]
                try:
                    used, limit = quota_status(k)
                except Exception:
                    continue
                if used + QUOTA_RESERVE <= limit:
                    _next_key[0] = (start + n + 1) % len(KEYS)
                    return k
        time.sleep(30)


def download(sensor: str, w: date) -> tuple[Path, float, int]:
    path = raw_path(sensor, w)
    if path.exists():
        return path, 0.0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    attempt = 0
    while True:
        key = take_key()
        try:
            text = firms_get(f"area/csv/{key}/{sensor}/world/{WINDOW}/{w.isoformat()}")
            if not text.lstrip().lower().startswith("latitude"):
                if "no data" in text.lower() or not text.strip():
                    text = ""
                else:
                    raise ValueError(text.strip()[:200])
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as f:   # level 1: 3-5x faster, slightly larger
                f.write(text)
            tmp.replace(path)                      # only complete files count, so a resume never reads half a file
            return path, time.time() - t0, len(text)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and b"transaction limit" in (exc.read() or b"").lower():
                time.sleep(60)                     # over the quota: wait for the window to free up, then retry
                continue
            attempt += 1
            if attempt >= 5:
                raise
            time.sleep(15 * attempt)
        except Exception:
            attempt += 1
            if attempt >= 5:
                raise
            time.sleep(15 * attempt)


# ---------------------------------------------------------------------------
# Aggregate one window
# ---------------------------------------------------------------------------

def read_raw(path: Path) -> pd.DataFrame | None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        if not f.read(16).strip():
            return None
    return pd.read_csv(path, usecols=lambda c: c in {"latitude", "longitude", "acq_date", "frp", "type"},
                       dtype={"acq_date": str})


def aggregate_window(w: date, tasks: list, static_cells: set, base: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cell-day table for the window, per-cell archive type counts)."""
    parts, types = [], []
    for family, sensor, kind, keep_from, keep_to in tasks:
        df = read_raw(raw_path(sensor, w, base))
        if df is None or df.empty:
            continue
        df = df[(df.acq_date >= keep_from.isoformat()) & (df.acq_date <= keep_to.isoformat())]
        df["i"] = np.floor(df.latitude / CELL).astype(np.int32)
        df["j"] = np.floor(df.longitude / CELL).astype(np.int32)
        if kind == "SP" and "type" in df:
            df["static"] = df["type"] != 0
            types.append(df.groupby(["i", "j"]).static.agg(["sum", "size"]).reset_index())
            df = df[~df.static]
        else:   # NRT: drop cells the archive shows are mostly static sources
            key = df.i.astype(np.int64) * 100000 + df.j
            df = df[~key.isin(static_cells)]
        df["family"] = family
        parts.append(df[["i", "j", "acq_date", "family", "frp"]])
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    cd = d.pivot_table(index=["i", "j", "acq_date"], columns="family", values="frp",
                       aggfunc=["size", "sum", "max"], fill_value=0)
    cd.columns = [f"{a}_{b}" for a, b in cd.columns]
    cd = cd.reset_index().rename(columns={"size_MODIS": "modis_det", "size_VIIRS": "viirs_det",
                                          "sum_MODIS": "modis_frp", "sum_VIIRS": "viirs_frp",
                                          "max_MODIS": "modis_frp_max", "max_VIIRS": "viirs_frp_max"})
    for c in ("modis_det", "viirs_det", "modis_frp", "viirs_frp", "modis_frp_max", "viirs_frp_max"):
        if c not in cd:
            cd[c] = 0
    t = pd.concat(types).groupby(["i", "j"]).sum().reset_index() if types else pd.DataFrame()
    return cd, t


def aggregate_and_save(w: date, tasks: list, static_cells: set, base: Path):
    """Runs in a worker process: aggregate one window, save its cell-day file, return the small parts."""
    cd, types = aggregate_window(w, tasks, static_cells, base)
    if len(cd):
        cd.to_csv(base / "cellday" / f"{w.isoformat()}.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
        return month_partials(cd), types
    return None, types


# ---------------------------------------------------------------------------
# Frequency map and big fires
# ---------------------------------------------------------------------------

def month_partials(cd: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(cd.acq_date)
    g = cd.assign(year=dt.dt.year, month=dt.dt.month, any_day=1, modis_day=(cd.modis_det > 0).astype(int),
                  viirs_day=(cd.viirs_det > 0).astype(int), frp=cd.modis_frp + cd.viirs_frp,
                  frp_max=np.maximum(cd.modis_frp_max, cd.viirs_frp_max))
    return g.groupby(["i", "j", "year", "month"]).agg(fire_days=("any_day", "sum"), modis_days=("modis_day", "sum"),
                                                     viirs_days=("viirs_day", "sum"), modis_det=("modis_det", "sum"),
                                                     viirs_det=("viirs_det", "sum"), frp=("frp", "sum"),
                                                     frp_max=("frp_max", "max")).reset_index()


def combine(parts: list) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["i", "j", "year", "month"]).agg(fire_days=("fire_days", "sum"), modis_days=("modis_days", "sum"),
                                                      viirs_days=("viirs_days", "sum"), modis_det=("modis_det", "sum"),
                                                      viirs_det=("viirs_det", "sum"), frp=("frp", "sum"),
                                                      frp_max=("frp_max", "max")).reset_index()


def frequency_map(cm: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    months = pd.period_range(start, end, freq="M")
    cover = pd.Series(months.month).value_counts()                 # how many times each calendar month occurs
    years = sorted({p.year for p in months})
    mid = years[len(years) // 2]
    g = cm.groupby(["i", "j", "month"])
    f = pd.DataFrame({
        "years_with_fire": g.year.nunique(), "years_with_modis": g.modis_days.apply(lambda s: int((s > 0).sum())),
        "years_with_viirs": g.viirs_days.apply(lambda s: int((s > 0).sum())),
        "fire_days_total": g.fire_days.sum(), "frp_total_mw": g.frp.sum().round(1), "frp_max_mw": g.frp_max.max().round(1),
    }).reset_index()
    early = cm[cm.year < mid].groupby(["i", "j", "month"]).modis_days.sum()
    late = cm[cm.year >= mid].groupby(["i", "j", "month"]).modis_days.sum()
    f = f.merge(early.rename("modis_days_early"), on=["i", "j", "month"], how="left") \
         .merge(late.rename("modis_days_late"), on=["i", "j", "month"], how="left").fillna({"modis_days_early": 0, "modis_days_late": 0})
    f["years_covered"] = f.month.map(cover)
    f["share_of_years"] = (f.years_with_fire / f.years_covered).round(3)
    f["mean_fire_days"] = (f.fire_days_total / f.years_covered).round(2)
    f["trend_modis"] = np.select([f.modis_days_late > 1.5 * f.modis_days_early + 1, f.modis_days_early > 1.5 * f.modis_days_late + 1],
                                 ["more frequent", "less frequent"], "stable")
    f["heads_up"] = pd.cut(f.share_of_years, [-0.01, 0.15, 0.4, 0.7, 1.01], labels=["minimal", "low", "medium", "high"]).astype(str)
    f["lat_min"], f["lon_min"] = (f.i * CELL).round(1), (f.j * CELL).round(1)
    return f


def big_fires(files: list, top: int) -> pd.DataFrame:
    """Connected burning cell-days (8 neighbours, up to 1 day apart) with FRP >= BIG_FRP_MW each."""
    parts = []
    for p in files:
        cd = pd.read_csv(p, dtype={"acq_date": str})
        cd["frp"] = cd.modis_frp + cd.viirs_frp
        parts.append(cd[cd.frp >= BIG_FRP_MW][["i", "j", "acq_date", "frp", "modis_det", "viirs_det"]])
    hot = pd.concat(parts, ignore_index=True)
    hot["d"] = (pd.to_datetime(hot.acq_date) - pd.Timestamp("2000-01-01")).dt.days.astype(np.int32)
    hot = hot.reset_index(drop=True)
    hot["n"] = np.arange(len(hot))
    keyed = hot[["i", "j", "d", "n"]]
    pairs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dd in (0, 1):
                m = keyed.assign(i=keyed.i + di, j=keyed.j + dj, d=keyed.d + dd).merge(keyed, on=["i", "j", "d"], suffixes=("", "_b"))
                m = m[m.n != m.n_b]
                pairs.append(m[["n", "n_b"]])
    pr = pd.concat(pairs, ignore_index=True)
    hot["event"] = components(len(hot), pr.n.to_numpy(), pr.n_b.to_numpy())
    g = hot.groupby("event")
    ev = pd.DataFrame({"start": g.acq_date.min(), "end": g.acq_date.max(), "days": g.d.nunique(),
                       "cell_days": g.size(), "cells": g.apply(lambda x: len(set(zip(x.i, x.j)))),
                       "frp_total_mw": g.frp.sum().round(1), "frp_peak_cell_mw": g.frp.max().round(1),
                       "detections": (g.modis_det.sum() + g.viirs_det.sum()),
                       "south": (g.i.min() * CELL).round(1), "north": ((g.i.max() + 1) * CELL).round(1),
                       "west": (g.j.min() * CELL).round(1), "east": ((g.j.max() + 1) * CELL).round(1)}).reset_index()
    ev["approx_area_km2"] = (ev.cells * (11.1 ** 2) * np.cos(np.radians((ev.south + ev.north) / 2))).round(0)
    ev = ev.sort_values("frp_total_mw", ascending=False, ignore_index=True)
    ev["rank"] = np.arange(1, len(ev) + 1)
    return ev.head(top)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads")
    ap.add_argument("--agg-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2), help="aggregation processes")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--out", help="output folder (default data/archive)")
    args = ap.parse_args()
    global ARCHIVE
    if args.out:
        ARCHIVE = Path(args.out)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    t0 = time.time()
    windows = plan_tasks(start, end, fetch_availability())
    n_tasks = sum(len(v) for v in windows.values())
    (ARCHIVE / "cellday").mkdir(parents=True, exist_ok=True)
    print(f"{len(windows)} windows, {n_tasks} downloads ({args.threads} at a time, {args.agg_workers} aggregation processes)", flush=True)

    # downloads in threads; windows aggregated in worker processes as their files arrive.
    # Archive windows go straight to the workers; near-real-time windows wait for the static-source mask,
    # which is built from all archive windows' type counts.
    dl = ThreadPoolExecutor(args.threads)
    agg = ProcessPoolExecutor(args.agg_workers)
    futures = {w: [dl.submit(download, s, w) for _, s, *_ in tasks] for w, tasks in windows.items()}
    sp_windows = [w for w, t in windows.items() if not any(kind == "NRT" for _, _, kind, *_ in t)]
    nrt_windows = [w for w in windows if w not in sp_windows]
    static_counts, partials, merged, mb = [], [], [], 0.0
    jobs, done = [], 0

    def collect(job):
        nonlocal done
        part, types = job.result()
        if part is not None:
            partials.append(part)
        if len(types):
            static_counts.append(types)
        done += 1
        if done % 20 == 0 or done == len(windows):
            el = time.time() - t0
            print(f"  {done}/{len(windows)} windows aggregated, {mb:,.0f} MB downloaded this run, "
                  f"{el / 60:.1f} min elapsed, ~{el / done * (len(windows) - done) / 60:.0f} min left", flush=True)

    for w in sp_windows:
        for f in futures[w]:
            mb += f.result()[2] / 1e6
        jobs.append(agg.submit(aggregate_and_save, w, windows[w], set(), ARCHIVE))
        while jobs and jobs[0].done():
            collect(jobs.pop(0))
        if len(partials) >= 40:
            merged.append(combine(partials)); partials.clear()
    for j in jobs:
        collect(j)
    jobs = []
    static_cells = set()
    if static_counts:
        sc = pd.concat(static_counts).groupby(["i", "j"]).sum()
        sc = sc[sc["sum"] >= STATIC_SHARE * sc["size"]]
        static_cells = set((sc.index.get_level_values(0).astype(np.int64) * 100000 + sc.index.get_level_values(1)).tolist())
    for w in nrt_windows:
        for f in futures[w]:
            mb += f.result()[2] / 1e6
        jobs.append(agg.submit(aggregate_and_save, w, windows[w], static_cells, ARCHIVE))
    for j in jobs:
        collect(j)
    dl.shutdown()
    agg.shutdown()

    cm = combine(merged + partials)
    freq = frequency_map(cm, start, end)
    freq.to_csv(ARCHIVE / "frequency.csv.gz", index=False)
    files = sorted((ARCHIVE / "cellday").glob("*.csv.gz"))
    ev = big_fires(files, args.top)
    ev.to_csv(ARCHIVE / "big_fires.csv.gz", index=False)
    summary = {"start": args.start, "end": args.end, "windows": len(windows), "downloads": n_tasks,
               "cells_with_fire": int(freq[["i", "j"]].drop_duplicates().shape[0]),
               "cell_months": len(freq), "heads_up": {k: int(v) for k, v in freq.heads_up.value_counts().items()},
               "static_cells_masked_in_nrt": len(static_cells), "big_fires_kept": len(ev),
               "biggest_fire": ev.iloc[0].to_dict() if len(ev) else None,
               "settings": {"cell_deg": CELL, "window_days": WINDOW, "big_frp_mw": BIG_FRP_MW, "static_share": STATIC_SHARE},
               "minutes": round((time.time() - t0) / 60, 1)}
    (ARCHIVE / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"\nDone in {summary['minutes']} min: {summary['cells_with_fire']:,} cells with fire, "
          f"heads-up {summary['heads_up']}, {len(ev)} big fires kept")


if __name__ == "__main__":
    main()
