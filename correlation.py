"""Correlation layer: stage 1 of the prediction pipeline.

Runs once per input dataset (the harmonised NASA FIRMS data) and produces what
the interpreter needs to decide which data goes to which worker:

1. Field relationships, measured on this dataset, per instrument scope
   (all rows, MODIS only, VIIRS only, ...), because the sensors measure
   differently and pooled numbers can mislead:
   - numeric vs numeric:          Spearman rank correlation (-1..1)
   - categorical vs numeric:      correlation ratio eta on ranks (0..1)
   - categorical vs categorical:  Cramer's V, bias-corrected (0..1)
   Pairs that are related by construction (pixel_area from scan and track,
   ...) are flagged so they are not mistaken for findings.
2. Exact relationships, checked row by row (pixel_area = scan x track, ...).
3. Row links: which detections belong together.
   - cell_group: same grid cell, date and day/night
   - pass_id:    same satellite orbit segment
   - links:      pairs of detections whose pixel footprints touch, labelled
                 same_pass / repeat_pass (same satellite, later pass) /
                 other_satellite, with distance and time gap
   - patch_id:   connected groups of touching pixels within one pass
   - link relations: how strongly a field agrees across linked detections
                 (e.g. MODIS FRP vs VIIRS FRP at the same spot)

Usage:  python correlation.py --date 2026-10-13 --days 1 --area world
Output: data/runs/<run>/correlation/ (rows.csv.gz, links.csv.gz, correlations.json)
"""

import argparse
import json
import math
import re
import time
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from comparison import overpass_slot
from firms import DATA_DIR
from type_inference import TYPE_LABELS

MIN_N = 30                 # fewest rows a relationship is reported on
TOUCH_FACTOR = 1.5         # footprints touch if distance <= 1.5 x mean pixel size (covers diagonals)
PASS_GAP_MIN = 20          # a gap longer than this starts a new orbit segment
MAX_LINK_GAP_H = 24        # links between detections at most this far apart in time
DEFAULT_PIXEL_KM = {"MODIS": 1.0, "VIIRS": 0.375, "OLI": 0.03}
CONF_SCORE = {"low": 1, "nominal": 2, "high": 3}
EARTH_KM = 6371.0

NUMERIC = ["latitude", "longitude", "acq_minutes", "local_solar_hour", "confidence_score", "confidence_pct",
           "bright_main_k", "bright_secondary_k", "bright_diff_k", "frp_mw", "frp_physics_proxy",
           "pixel_area_km2", "scan_km", "track_km", "type_confidence"]
CATEGORICAL = ["source", "instrument", "satellite", "acq_date", "daynight", "overpass_slot",
               "confidence", "type", "type_existed"]
DERIVED = {  # field -> fields it is computed from (relationships among them are by construction)
    "pixel_area_km2": {"scan_km", "track_km"},
    "bright_diff_k": {"bright_main_k", "bright_secondary_k"},
    "frp_physics_proxy": {"bright_main_k", "bright_secondary_k", "pixel_area_km2", "scan_km", "track_km", "bright_diff_k"},
    "local_solar_hour": {"acq_minutes", "longitude"},
    "overpass_slot": {"acq_minutes", "longitude", "local_solar_hour"},
    "confidence": {"confidence_pct", "confidence_score"},
    "confidence_score": {"confidence_pct"},
    "instrument": {"source", "satellite"},
    "satellite": {"source"},
    "type_existed": {"source", "type_confidence"},
}
DESCRIPTIONS = {
    "acq_minutes": "UTC time of day in minutes",
    "local_solar_hour": "local solar time (UTC + longitude / 15)",
    "confidence_score": "confidence as 1 low / 2 nominal / 3 high",
    "confidence_pct": "MODIS confidence 0-100 (empty for other sensors)",
    "bright_diff_k": "fire channel minus background channel (K)",
    "frp_physics_proxy": "pixel area x (T_main^8 - T_secondary^8) x 1e-20, the shape of NASA's FRP formula",
}


def strength(v: float) -> str:
    a = abs(v)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def by_construction(a: str, b: str) -> bool:
    return b in DERIVED.get(a, ()) or a in DERIVED.get(b, ())


# ---------------------------------------------------------------------------
# Preparing the table
# ---------------------------------------------------------------------------

def prepare(columns: list, rows: list) -> pd.DataFrame:
    """Harmonised rows -> DataFrame with typed numeric fields and the derived fields."""
    df = pd.DataFrame(rows, columns=columns)
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    for c in ["latitude", "longitude", "bright_main_k", "bright_secondary_k", "frp_mw", "pixel_area_km2",
              "scan_km", "track_km", "type_confidence"]:
        df[c] = num(c)
    t = df["acq_time"].astype(str).str.zfill(4)
    df["acq_minutes"] = pd.to_numeric(t.str[:2], errors="coerce") * 60 + pd.to_numeric(t.str[2:], errors="coerce")
    df["local_solar_hour"] = (df["acq_minutes"] / 60 + df["longitude"] / 15) % 24
    df["confidence_score"] = df["confidence"].map(CONF_SCORE)
    df["confidence_pct"] = np.where(df["instrument"] == "MODIS", pd.to_numeric(df["confidence_raw"], errors="coerce"), np.nan)
    df["bright_diff_k"] = df["bright_main_k"] - df["bright_secondary_k"]
    df["frp_physics_proxy"] = df["pixel_area_km2"] * (df["bright_main_k"] ** 8 - df["bright_secondary_k"] ** 8) * 1e-20
    start = pd.to_datetime(df["acq_date"], errors="coerce").min()
    df["t_min"] = (pd.to_datetime(df["acq_date"], errors="coerce") - start).dt.days * 1440 + df["acq_minutes"]
    return df


# ---------------------------------------------------------------------------
# 1. Field relationships
# ---------------------------------------------------------------------------

def _eta(cat: pd.Series, ranks: pd.Series) -> tuple[float, int]:
    ok = cat.notna() & (cat != "") & ranks.notna()
    c, y = cat[ok], ranks[ok]
    if len(y) < MIN_N or c.nunique() < 2:
        return math.nan, len(y)
    total = ((y - y.mean()) ** 2).sum()
    if total == 0:
        return math.nan, len(y)
    g = y.groupby(c)
    between = (g.count() * (g.mean() - y.mean()) ** 2).sum()
    return math.sqrt(between / total), len(y)


def _cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna() & (a != "") & (b != "")
    a, b = a[ok], b[ok]
    n = len(a)
    if n < MIN_N or a.nunique() < 2 or b.nunique() < 2:
        return math.nan, n
    obs = pd.crosstab(a, b).to_numpy(dtype=float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / n
    chi2 = ((obs - exp) ** 2 / exp).sum()
    r, k = obs.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return (math.sqrt(phi2 / d) if d > 0 else math.nan), n


def field_relations(df: pd.DataFrame, scope: str) -> list:
    numeric = [c for c in NUMERIC if df[c].notna().sum() >= MIN_N and df[c].nunique() > 1]
    categorical = [c for c in CATEGORICAL if c in df and df[c].replace("", np.nan).nunique() > 1]
    out = []

    def add(a, b, measure, v, n):
        if not math.isnan(v):
            out.append({"scope": scope, "a": a, "b": b, "measure": measure, "value": round(float(v), 4),
                        "n": int(n), "strength": strength(v), "by_construction": by_construction(a, b)})

    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns give NaN, which is skipped
        sp = df[numeric].rank().corr(min_periods=MIN_N)  # Spearman = Pearson on ranks (pairwise rows)
    counts = df[numeric].notna().astype(int)
    pair_n = counts.T @ counts
    for a, b in combinations(numeric, 2):
        add(a, b, "spearman", sp.loc[a, b], pair_n.loc[a, b])

    ranks = {c: df[c].rank() for c in numeric}
    for c in categorical:
        for x in numeric:
            add(c, x, "eta", *_eta(df[c], ranks[x]))
    for a, b in combinations(categorical, 2):
        add(a, b, "cramers_v", *_cramers_v(df[a], df[b]))
    return out


# ---------------------------------------------------------------------------
# 2. Exact relationships
# ---------------------------------------------------------------------------

def _confidence_rule(inst: str, raw: str) -> str:
    raw = str(raw).strip()
    if inst == "MODIS":
        try:
            v = float(raw)
        except ValueError:
            return ""
        return "low" if v < 30 else "high" if v >= 80 else "nominal"
    return {"l": "low", "n": "nominal", "m": "nominal", "h": "high"}.get(raw.lower(), "")


def exact_relations(df: pd.DataFrame) -> list:
    checks = []

    def check(name, mask, match):
        n = int(mask.sum())
        if n:
            m = int((match & mask).sum())
            checks.append({"relation": name, "rows_checked": n, "rows_matching": m, "share": round(m / n, 6)})

    has = df["scan_km"].notna() & df["track_km"].notna() & df["pixel_area_km2"].notna()
    check("pixel_area_km2 = scan_km x track_km", has,
          (df["pixel_area_km2"] - (df["scan_km"] * df["track_km"]).round(4)).abs() < 5e-5)
    conf = [_confidence_rule(i, r) for i, r in zip(df["instrument"], df["confidence_raw"])]
    check("confidence = rule(confidence_raw, instrument)", df["confidence_raw"].astype(str) != "", df["confidence"] == pd.Series(conf, index=df.index))
    slot = [overpass_slot(t, str(lon)) for t, lon in zip(df["acq_time"], df["longitude"])]
    check("overpass_slot = slot(acq_time + longitude / 15)", df["longitude"].notna(), df["overpass_slot"] == pd.Series(slot, index=df.index))
    if "type_label" in df:
        labels = df["type"].map(lambda t: TYPE_LABELS.get(int(t)) if str(t).isdigit() else None)
        # labels for inferred types are prefixed with "presumed"; compare on the base label
        base = df["type_label"].str.replace(r"^presumed ", "", regex=True)
        check("type_label = label(type)", df["type"].astype(str) != "", base == labels.str.replace(r"^presumed ", "", regex=True))
    return checks


# ---------------------------------------------------------------------------
# 3. Row links
# ---------------------------------------------------------------------------

def add_groups(df: pd.DataFrame, cell_deg: float) -> None:
    i = np.floor(df["latitude"] / cell_deg).astype("Int64").astype(str)
    j = np.floor(df["longitude"] / cell_deg).astype("Int64").astype(str)
    df["cell_group"] = i + ":" + j + "|" + df["acq_date"] + "|" + df["daynight"]
    df["pass_id"] = ""
    for sat, g in df.sort_values("t_min").groupby("satellite", sort=False):
        new = g["t_min"].diff().fillna(PASS_GAP_MIN + 1) > PASS_GAP_MIN
        df.loc[g.index, "pass_id"] = f"{sat}#" + new.cumsum().astype(str)


def _pixel_km(df: pd.DataFrame) -> np.ndarray:
    s = np.fmax(df["scan_km"], df["track_km"])
    s = s.fillna(np.sqrt(df["pixel_area_km2"]))
    return s.fillna(df["instrument"].map(DEFAULT_PIXEL_KM)).fillna(1.0).to_numpy()


def _xyz(df: pd.DataFrame) -> np.ndarray:
    lat, lon = np.radians(df["latitude"].to_numpy()), np.radians(df["longitude"].to_numpy())
    return EARTH_KM * np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _pairs(xyz, size, t, ia, ib, same: bool, max_gap: float) -> pd.DataFrame:
    """Pairs (i in ia, j in ib) whose footprints touch, via a 3-D grid on the sphere."""
    if not len(ia) or not len(ib):
        return pd.DataFrame(columns=["i", "j", "dist_km", "gap_min"])
    d = TOUCH_FACTOR * (size[ia].max() + size[ib].max()) / 2
    key = lambda idx: pd.DataFrame(np.floor(xyz[idx] / d).astype(np.int64), columns=["bx", "by", "bz"]).assign(i=idx)
    a, b = key(ia), key(ib).rename(columns={"i": "j"})
    found = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                m = a.assign(bx=a.bx + dx, by=a.by + dy, bz=a.bz + dz).merge(b, on=["bx", "by", "bz"])[["i", "j"]]
                if same:
                    m = m[m.i < m.j]
                if len(m):
                    i, j = m.i.to_numpy(), m.j.to_numpy()
                    dist = np.linalg.norm(xyz[i] - xyz[j], axis=1)
                    gap = np.abs(t[i] - t[j])
                    keep = (dist <= TOUCH_FACTOR * (size[i] + size[j]) / 2) & (gap <= max_gap)
                    found.append(pd.DataFrame({"i": i[keep], "j": j[keep], "dist_km": dist[keep], "gap_min": gap[keep]}))
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["i", "j", "dist_km", "gap_min"])


def find_links(df: pd.DataFrame, max_gap_h: float = MAX_LINK_GAP_H) -> pd.DataFrame:
    ok = df["latitude"].notna() & df["longitude"].notna() & df["t_min"].notna()
    xyz, size, t = _xyz(df.fillna({"latitude": 0, "longitude": 0})), _pixel_km(df), df["t_min"].fillna(-1e9).to_numpy()
    groups = {inst: np.flatnonzero(ok & (df["instrument"] == inst)) for inst in df["instrument"].unique()}
    parts = []
    for a, b in combinations(sorted(groups), 2):
        parts.append(_pairs(xyz, size, t, groups[a], groups[b], False, max_gap_h * 60))
    for a in groups:
        parts.append(_pairs(xyz, size, t, groups[a], groups[a], True, max_gap_h * 60))
    links = pd.concat(parts, ignore_index=True).astype({"i": np.int64, "j": np.int64})
    lo, hi = np.minimum(links.i, links.j), np.maximum(links.i, links.j)  # always i < j
    links["i"], links["j"] = lo, hi
    sat, pid = df["satellite"].to_numpy(), df["pass_id"].to_numpy()
    same_sat = sat[links.i] == sat[links.j]
    links["kind"] = np.where(~same_sat, "other_satellite",
                             np.where(pid[links.i] == pid[links.j], "same_pass", "repeat_pass"))
    links["dist_km"] = links["dist_km"].round(3)
    links["gap_min"] = links["gap_min"].round(1)
    return links.sort_values(["i", "j"], ignore_index=True)


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component label (0..k-1) for n nodes joined by edges i-j (min-label propagation)."""
    lab = np.arange(n)
    while len(i):
        m = np.minimum(lab[i], lab[j])
        before = lab.copy()
        np.minimum.at(lab, i, m)
        np.minimum.at(lab, j, m)
        lab = lab[lab]
        if np.array_equal(lab, before):
            break
    return np.unique(lab, return_inverse=True)[1]


def connected_patches(n: int, links: pd.DataFrame) -> np.ndarray:
    """Connected groups of touching pixels within one pass."""
    sp = links[links.kind == "same_pass"]
    return components(n, sp.i.to_numpy(), sp.j.to_numpy())


LINK_CONSTRUCTION = {  # fields that agree across a link kind because of how that kind is defined
    "same_pass": {"acq_minutes", "local_solar_hour", "acq_date", "daynight", "overpass_slot", "satellite",
                  "instrument", "source"},
    "repeat_pass": {"satellite", "instrument", "source"},
    "other_satellite": {"satellite", "instrument", "source"},  # the two ends are different satellites by definition
}


def link_relations(df: pd.DataFrame, links: pd.DataFrame) -> list:
    """How well each field agrees across the two ends of each kind of link.

    Numeric fields: Spearman between the two ends. Categorical: Cramer's V.
    Latitude and longitude are left out (links are defined by distance)."""
    out = []
    inst = df["instrument"].to_numpy()
    numeric = [f for f in NUMERIC if f not in ("latitude", "longitude")]
    categorical = [f for f in CATEGORICAL if f in df]
    for kind, g in links.groupby("kind"):
        pairs = {"all": g}
        if kind == "other_satellite":
            for a, b in combinations(sorted(set(inst)), 2):
                m = ((inst[g.i] == a) & (inst[g.j] == b)) | ((inst[g.i] == b) & (inst[g.j] == a))
                if m.sum() >= MIN_N:
                    pairs[f"{a}-{b}"] = g[m]
        for label, p in pairs.items():
            def add(f, measure, v, n):
                if not math.isnan(v):
                    out.append({"kind": kind, "pair": label, "field": f, "measure": measure, "value": round(float(v), 4),
                                "n": int(n), "strength": strength(v), "by_construction": f in LINK_CONSTRUCTION[kind]})
            for f in numeric:
                x, y = df[f].to_numpy()[p.i], df[f].to_numpy()[p.j]
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() >= MIN_N and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                    add(f, "spearman", pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()), ok.sum())
            for f in categorical:
                col = df[f].astype(str).to_numpy()
                add(f, "cramers_v", *_cramers_v(pd.Series(col[p.i]), pd.Series(col[p.j])))
            if len(p) >= MIN_N:
                same = df["type"].to_numpy()[p.i] == df["type"].to_numpy()[p.j]
                out.append({"kind": kind, "pair": label, "field": "type", "measure": "share_equal",
                            "value": round(float(same.mean()), 4), "n": int(len(p)), "strength": "", "by_construction": False})
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_correlation(columns: list, rows: list, cell_deg: float = 0.05, max_gap_h: float = MAX_LINK_GAP_H) -> dict:
    t0 = time.time()
    df = prepare(columns, rows)
    scopes = {"all": df, **{f"instrument={i}": g for i, g in df.groupby("instrument") if len(g) >= MIN_N}}
    relations = [r for name, g in scopes.items() for r in field_relations(g, name)]
    exact = exact_relations(df)
    add_groups(df, cell_deg)
    links = find_links(df, max_gap_h)
    df["patch_id"] = connected_patches(len(df), links)
    degree = np.bincount(np.concatenate([links.i, links.j]), minlength=len(df)) if len(links) else np.zeros(len(df), int)
    df["link_count"] = degree
    report = {
        "settings": {"cell_deg": cell_deg, "touch_factor": TOUCH_FACTOR, "pass_gap_min": PASS_GAP_MIN,
                     "max_link_gap_h": max_gap_h, "min_n": MIN_N},
        "rows": len(df),
        "scopes": {name: len(g) for name, g in scopes.items()},
        "fields": {"numeric": NUMERIC, "categorical": CATEGORICAL, "derived_from": {k: sorted(v) for k, v in DERIVED.items()},
                   "descriptions": DESCRIPTIONS},
        "relations": relations,
        "exact": exact,
        "groups": {"cell_groups": int(df["cell_group"].nunique()), "passes": int(df["pass_id"].nunique()),
                   "patches": int(df["patch_id"].nunique()),
                   "rows_with_links": int((degree > 0).sum())},
        "links": {k: int(v) for k, v in links["kind"].value_counts().items()},
        "link_relations": link_relations(df, links),
        "seconds": round(time.time() - t0, 1),
    }
    return {"frame": df, "links": links, "report": report}


def ranking(report: dict, field: str, scope: str = "all", include_construction: bool = False) -> list:
    """Every field related to `field` in one scope, strongest first (for the interpreter)."""
    out = []
    for r in report["relations"]:
        if r["scope"] != scope or (r["by_construction"] and not include_construction):
            continue
        if field in (r["a"], r["b"]):
            out.append({**r, "other": r["b"] if r["a"] == field else r["a"]})
    return sorted(out, key=lambda r: -abs(r["value"]))


def save(result: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["frame"].drop(columns=["t_min"]).to_csv(out_dir / "rows.csv.gz", index=False)
    result["links"].to_csv(out_dir / "links.csv.gz", index=False)
    (out_dir / "correlations.json").write_text(json.dumps(result["report"], indent=1), encoding="utf-8")


def print_summary(rep: dict) -> None:
    print(f"\n{rep['rows']:,} rows in {rep['seconds']} s   scopes: " + ", ".join(f"{k} {v:,}" for k, v in rep["scopes"].items()))
    print("\nExact relationships (row by row):")
    for e in rep["exact"]:
        print(f"  {e['share']:.2%}  {e['relation']}  ({e['rows_matching']:,} / {e['rows_checked']:,})")
    for scope in rep["scopes"]:
        top = [r for r in rep["relations"] if r["scope"] == scope and not r["by_construction"] and r["strength"] in ("strong", "moderate")]
        top.sort(key=lambda r: -abs(r["value"]))
        print(f"\nStrongest relationships, {scope} ({len(top)} strong or moderate, by-construction pairs left out):")
        for r in top[:15]:
            print(f"  {r['value']:+.3f} {r['measure']:<9} {r['a']} ~ {r['b']}  ({r['strength']}, n={r['n']:,})")
    g = rep["groups"]
    print(f"\nGroups: {g['cell_groups']:,} cell groups, {g['passes']:,} orbit segments, {g['patches']:,} patches; "
          f"{g['rows_with_links']:,} rows linked to at least one other")
    print("Links: " + ", ".join(f"{k} {v:,}" for k, v in rep["links"].items()))
    print("Agreement across links, moderate or stronger (Spearman / Cramer's V / share of equal type):")
    for r in rep["link_relations"]:
        if r["by_construction"] or (r["measure"] != "share_equal" and abs(r["value"]) < 0.4):
            continue
        print(f"  {r['kind']:<15} {r['pair']:<12} {r['field']:<19} {r['value']:+.3f} ({r['measure']}, n={r['n']:,})")


def main() -> None:
    from server import fetch_harmonised, validate  # server owns harmonisation and input checks

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    ap.add_argument("--cell", type=float, default=0.05, help="grid size in degrees for cell_group")
    ap.add_argument("--max-gap-h", type=float, default=MAX_LINK_GAP_H, help="longest time gap for a link, hours")
    args = ap.parse_args()
    day, days, area = validate({"date": args.date, "days": args.days, "area": args.area})
    print(f"Fetching harmonised data for {day}, {days} day(s), area {area} ...")
    h = fetch_harmonised(day, days, area, keep_scan_track=True)
    result = run_correlation(h["columns"], h["rows"], args.cell, args.max_gap_h)
    result["report"]["run"] = {"date": day, "days": days, "area": area, "sources": h["sources"], "range": h["range"]}
    slug = f"{day}_{days}d_" + re.sub(r"[^\w.-]+", "_", area)
    out = DATA_DIR / "runs" / slug / "correlation"
    save(result, out)
    print_summary(result["report"])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
