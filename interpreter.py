"""Interpreter: stage 2 of the prediction pipeline.

Reads the correlation layer's output for one run and prepares the work for the
stage-3 workers (localisation and fire power). It decides three things:

1. Which fields each worker gets.
   - Every worker has a fixed minimum set it always receives, whatever the
     correlations say (time is uncorrelated with position, yet spread cannot
     be measured without it).
   - On top of that it gets every field related to its goal at least
     moderately (|value| >= 0.4) in this dataset:
       fire power   -> fields related to frp_mw (per sensor; pooled numbers
                       mix sensors that measure differently)
       localisation -> fields that agree across linked detections, i.e. that
                       help tell whether two detections are the same fire
   - Never sent: duplicates of another field (type_label, confidence_raw,
     type_basis) and fields that match a kept field almost exactly
     (|value| >= 0.98 in every sensor scope).
2. Which rows each worker gets.
   - Non-wildfire detections (volcano, static land source, offshore) are
     removed when the type came from the file or was inferred with
     type_confidence >= 0.5. Less certain ones are kept and flagged.
   - Fire power needs FRP and brightness, so rows without them (Landsat) go
     to localisation only.
3. How the work is split.
   - Detections joined by any link (same fire complex) form one unit and are
     never split across work packages.
   - Units are ordered by 10-degree tile and cut into packages of similar
     size, so each package covers a compact area.

Usage:  python interpreter.py --run data/runs/2026-10-13_1d_world [--packages 8]
        (or --date/--days/--area to locate the run)
Output: <run>/interpreter/plan.json and <run>/interpreter/<worker>/package_NNN.csv.gz
        (localisation packages also get links_NNN.csv.gz)
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from correlation import CATEGORICAL, NUMERIC, components
from firms import DATA_DIR

SELECT_MIN = 0.4          # moderate or stronger
DUPLICATE_MIN = 0.98      # a field this close to a kept field adds nothing
TILE_DEG = 10
NON_WILDFIRE = {"1", "2", "3"}
TYPE_SURE = 0.5
ALWAYS_EXCLUDED = {
    "type_label": "duplicate of type",
    "type_basis": "text explanation, duplicate of type_existed / type_confidence",
    "confidence_raw": "duplicate of confidence (MODIS 0-100 kept as confidence_pct)",
}
GROUP_COLUMNS = ["cell_group", "pass_id", "patch_id", "link_count", "unit_id"]

WORKERS = {
    "fire_power": {
        "goal": "estimate fire power per fire and how it changes over time",
        "targets": ["frp_mw"],
        "how": "correlation",
        "required": ["frp_mw", "bright_main_k", "bright_secondary_k", "pixel_area_km2", "instrument", "satellite",
                     "acq_date", "acq_time", "acq_minutes", "latitude", "longitude", "daynight", "confidence",
                     "type", "type_confidence", "type_existed"],
        "rows": "frp_mw and bright_main_k present",
    },
    "localisation": {
        "goal": "group detections into fires and follow them over time",
        "targets": ["same_pass", "repeat_pass", "other_satellite"],
        "how": "link_agreement",
        "required": ["latitude", "longitude", "acq_date", "acq_time", "acq_minutes", "scan_km", "track_km",
                     "pixel_area_km2", "daynight", "instrument", "satellite", "confidence", "frp_mw", "type",
                     "type_confidence"],
        "rows": "all wildfire rows",
    },
}


# ---------------------------------------------------------------------------
# 1. Field selection
# ---------------------------------------------------------------------------

def _scopes(report: dict) -> list:
    """Per-sensor scopes when there are any, else the pooled one."""
    per = [s for s in report["scopes"] if s.startswith("instrument=")]
    return per or ["all"]


def _relation_index(report: dict) -> dict:
    idx = {}
    for r in report["relations"]:
        idx.setdefault(r["scope"], {})[frozenset((r["a"], r["b"]))] = r
    return idx


def select_fields(name: str, spec: dict, report: dict, available: set, constant: set = frozenset()) -> dict:
    rel = _relation_index(report)
    scopes = _scopes(report)
    chosen, reasons, dropped = [], {}, {}

    for f in spec["required"]:
        if f in available:
            chosen.append(f)
            reasons[f] = "required for this worker"

    candidates = [f for f in NUMERIC + CATEGORICAL if f in available and f not in chosen]
    for f in candidates:
        if f in ALWAYS_EXCLUDED:
            dropped[f] = ALWAYS_EXCLUDED[f]
            continue
        if f in constant:
            dropped[f] = "same value on every row in this dataset"
            continue
        best = None
        if spec["how"] == "correlation":
            for scope in scopes:
                for t in spec["targets"]:
                    r = rel.get(scope, {}).get(frozenset((f, t)))
                    if r and not r["by_construction"] and (best is None or abs(r["value"]) > abs(best["value"])):
                        best = {**r, "target": t}
            if best and abs(best["value"]) >= SELECT_MIN:
                chosen.append(f)
                reasons[f] = f"{best['measure']} {best['value']:+.3f} with {best['target']} ({best['scope']}, n={best['n']:,})"
            else:
                dropped[f] = (f"weak: best {best['measure']} {best['value']:+.3f} with {best['target']} ({best['scope']})"
                              if best else "no measurable relation to the goal")
        else:
            agree = [r for r in report["link_relations"] if r["field"] == f and r["kind"] in spec["targets"]
                     and r["measure"] in ("spearman", "cramers_v") and r["pair"] == "all" and not r.get("by_construction")]
            best = max(agree, key=lambda r: r["value"], default=None)  # only positive values mean agreement
            if best and best["value"] >= SELECT_MIN:
                chosen.append(f)
                reasons[f] = f"agrees across {best['kind']} links: {best['measure']} {best['value']:+.3f} (n={best['n']:,})"
            else:
                dropped[f] = (f"weak link agreement: best {best['value']:+.3f} ({best['kind']})" if best
                              else "not measured across links")

    for f in ALWAYS_EXCLUDED:
        if f in available and f not in dropped:
            dropped[f] = ALWAYS_EXCLUDED[f]

    # remove near-exact duplicates of a field already kept (never a required one)
    required = set(spec["required"])
    for f in [c for c in chosen if c not in required]:
        for keep in chosen:
            if keep == f or keep not in chosen:
                continue
            vals = [rel.get(s, {}).get(frozenset((f, keep))) for s in scopes + ["all"]]
            vals = [v for v in vals if v]
            if vals and all(abs(v["value"]) >= DUPLICATE_MIN for v in vals):
                chosen.remove(f)
                dropped[f] = f"near-duplicate of {keep} (|value| >= {DUPLICATE_MIN} in every scope)"
                reasons.pop(f, None)
                break
    return {"fields": chosen, "reasons": reasons, "dropped": dropped}


# ---------------------------------------------------------------------------
# 2. Rows and 3. packages
# ---------------------------------------------------------------------------

def wildfire_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(keep, flagged): drop sure non-wildfire types, flag unsure ones."""
    t = df["type"].astype(str)
    conf = pd.to_numeric(df["type_confidence"], errors="coerce").fillna(0)
    sure = (df["type_existed"] == "yes") | (conf >= TYPE_SURE)
    non = t.isin(NON_WILDFIRE)
    return ~(non & sure), non & ~sure


def make_packages(df: pd.DataFrame, rows: np.ndarray, n_packages: int) -> list:
    """Split row indices into packages of whole units, ordered by tile."""
    sub = df.loc[rows, ["unit_id", "latitude", "longitude"]]
    units = sub.groupby("unit_id").agg(n=("latitude", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"))
    units["tile"] = (np.floor(units.lat / TILE_DEG) * 1000 + np.floor(units.lon / TILE_DEG)).astype(int)
    units = units.sort_values(["tile", "lat", "lon"])
    target = max(1, int(np.ceil(len(sub) / max(1, n_packages))))
    pack, size, out = [], 0, []
    for uid, n in zip(units.index, units.n):
        pack.append(uid)
        size += n
        if size >= target:
            out.append(pack)
            pack, size = [], 0
    if pack:
        out.append(pack)
    by_unit = sub.groupby("unit_id").groups
    return [np.sort(np.concatenate([np.asarray(by_unit[u]) for u in p])) for p in out]


def interpret(run_dir: Path, n_packages: int) -> dict:
    t0 = time.time()
    corr = run_dir / "correlation"
    report = json.loads((corr / "correlations.json").read_text(encoding="utf-8"))
    df = pd.read_csv(corr / "rows.csv.gz", dtype={"acq_time": str, "type": str, "confidence_raw": str},
                     keep_default_na=False, na_values=[""])
    links = pd.read_csv(corr / "links.csv.gz")
    df["unit_id"] = components(len(df), links.i.to_numpy(), links.j.to_numpy())
    available = set(df.columns)
    constant = {c for c in df.columns if df[c].nunique(dropna=True) <= 1}

    keep, flagged = wildfire_mask(df)
    df["type_unsure_non_wildfire"] = np.where(flagged, "yes", "no")
    out_dir = run_dir / "interpreter"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    plan = {"run": report.get("run", {}), "settings": {"select_min": SELECT_MIN, "duplicate_min": DUPLICATE_MIN,
            "tile_deg": TILE_DEG, "type_sure": TYPE_SURE, "packages_requested": n_packages},
            "rows_in": len(df), "units": int(df["unit_id"].nunique()),
            "row_filter": {"removed_non_wildfire": int((~keep).sum()), "flagged_unsure_non_wildfire": int(flagged.sum()),
                           "removed_by_type": {k: int(v) for k, v in df.loc[~keep, "type"].value_counts().items()}},
            "workers": {}}

    for name, spec in WORKERS.items():
        sel = select_fields(name, spec, report, available, constant)
        mask = keep.copy()
        if name == "fire_power":
            mask &= df["frp_mw"].notna() & df["bright_main_k"].notna()
        rows = np.flatnonzero(mask.to_numpy())
        packages = make_packages(df, rows, n_packages)
        cols = sel["fields"] + [c for c in GROUP_COLUMNS + ["type_unsure_non_wildfire"] if c not in sel["fields"]]
        wdir = out_dir / name
        wdir.mkdir(parents=True, exist_ok=True)
        info = []
        for k, idx in enumerate(packages):
            part = df.loc[idx, cols].assign(row_id=idx)
            part.to_csv(wdir / f"package_{k:03d}.csv.gz", index=False)
            entry = {"package": k, "rows": len(idx), "units": int(part["unit_id"].nunique()),
                     "bbox": [round(part.longitude.min(), 3), round(part.latitude.min(), 3),
                              round(part.longitude.max(), 3), round(part.latitude.max(), 3)]}
            if name == "localisation":
                s = set(idx.tolist())
                lk = links[links.i.isin(s) & links.j.isin(s)]
                lk.to_csv(wdir / f"links_{k:03d}.csv.gz", index=False)
                entry["links"] = len(lk)
            info.append(entry)
        plan["workers"][name] = {"goal": spec["goal"], "row_rule": spec["rows"], "rows": int(len(rows)),
                                 "fields": sel["fields"], "group_columns": [c for c in cols if c not in sel["fields"]],
                                 "why": sel["reasons"], "not_sent": sel["dropped"], "packages": info}

    # every link must stay inside one localisation package (units are never split)
    loc_links = sum(p.get("links", 0) for p in plan["workers"]["localisation"]["packages"])
    kept = set(np.flatnonzero(keep.to_numpy()).tolist())
    plan["checks"] = {"links_between_kept_rows": int((links.i.isin(kept) & links.j.isin(kept)).sum()),
                      "links_inside_packages": int(loc_links)}
    plan["seconds"] = round(time.time() - t0, 1)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def print_summary(plan: dict) -> None:
    f = plan["row_filter"]
    print(f"\n{plan['rows_in']:,} rows, {plan['units']:,} units (fire complexes) in {plan['seconds']} s")
    print(f"Removed {f['removed_non_wildfire']:,} non-wildfire rows {f['removed_by_type']}; "
          f"flagged {f['flagged_unsure_non_wildfire']:,} unsure ones")
    for name, w in plan["workers"].items():
        print(f"\n== {name}: {w['goal']}")
        print(f"   {w['rows']:,} rows in {len(w['packages'])} packages "
              f"({min(p['rows'] for p in w['packages']):,}-{max(p['rows'] for p in w['packages']):,} rows each)")
        print(f"   {len(w['fields'])} fields sent:")
        for fld in w["fields"]:
            print(f"     {fld:<20} {w['why'].get(fld, '')}")
        print("   not sent:")
        for fld, why in w["not_sent"].items():
            print(f"     {fld:<20} {why}")
    c = plan["checks"]
    print(f"\nLinks kept inside localisation packages: {c['links_inside_packages']:,} of {c['links_between_kept_rows']:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", help="run folder made by correlation.py")
    ap.add_argument("--date")
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    ap.add_argument("--packages", type=int, default=os.cpu_count() or 4, help="work packages per worker")
    args = ap.parse_args()
    if args.run:
        run = Path(args.run)
    elif args.date:
        run = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    else:
        ap.error("give --run or --date")
    if not (run / "correlation" / "correlations.json").exists():
        raise SystemExit(f"No correlation output in {run}. Run correlation.py first.")
    plan = interpret(run, args.packages)
    print_summary(plan)
    print(f"\nSaved to {run / 'interpreter'}")


if __name__ == "__main__":
    main()
