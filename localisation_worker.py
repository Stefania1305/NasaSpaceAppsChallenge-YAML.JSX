"""Localisation worker: stage 3a of the prediction pipeline.

Takes the interpreter's localisation packages and, for every fire, works out
where it is, how big it is and how it moved:

- Fire: the detections the correlation layer linked together (touching pixel
  footprints, within 24 h, any satellite). The interpreter never splits one
  across packages, so each package is processed on its own, in parallel.
- Observation: one satellite pass over a fire. Records its time, FRP-weighted
  centre, burning area (sum of pixel areas), pixel count and FRP.
- Extent: bounding box and the convex hull of all pixel footprints (km2).
- Spread, from consecutive passes of the same sensor family (VIIRS satellites
  together, MODIS together; their pixel sizes differ ~10x, so mixing them would
  look like growth or shrinkage):
    new front pixels   later pixels more than one pixel from every earlier one
    spread direction   bearing from the earlier centre to the new front pixels
                       (0 = north, 90 = east), averaged over pass pairs
    spread rate        median distance the new front pixels lie beyond the
                       earlier fire, per hour
    centre speed       least-squares drift of the FRP-weighted centre
    growth             change in burning area per hour
  Fires seen on only one pass get no spread values.

Usage:  python localisation_worker.py --date 2026-10-13 [--processes N]
        (or --run <run folder>)
Output: <run>/localisation/fires.csv.gz, observations.csv.gz, detections.csv.gz,
        summary.json
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

from firms import DATA_DIR

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320
NEW_FRONT_FACTOR = 1.0     # a later pixel is new front if > 1 pixel size from every earlier pixel
MIN_PAIR_GAP_MIN = 20      # passes closer than this are not compared
MAX_PAIR_GAP_H = 24
DEFAULT_PIXEL_KM = {"MODIS": 1.0, "VIIRS": 0.375, "OLI": 0.03}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def to_km(lat, lon, lat0, lon0):
    """Local tangent-plane coordinates in km around (lat0, lon0)."""
    dlon = (np.asarray(lon) - lon0 + 180) % 360 - 180
    return dlon * KM_PER_DEG_LON * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * KM_PER_DEG_LAT


def bearing(dx, dy) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360


def hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area of the convex hull of points (monotone chain + shoelace)."""
    pts = sorted(set(zip(np.round(x, 4), np.round(y, 4))))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return abs(sum(h[k][0] * h[k - 1][1] - h[k - 1][0] * h[k][1] for k in range(len(h)))) / 2


def nearest(bx, by, ax, ay):
    """Index and distance of the nearest A point for every B point (chunked)."""
    idx, dist = np.empty(len(bx), int), np.empty(len(bx))
    step = max(1, 2_000_000 // max(1, len(ax)))
    for s in range(0, len(bx), step):
        d = np.hypot(bx[s:s + step, None] - ax[None, :], by[s:s + step, None] - ay[None, :])
        idx[s:s + step] = d.argmin(1)
        dist[s:s + step] = d[np.arange(d.shape[0]), idx[s:s + step]]
    return idx, dist


# ---------------------------------------------------------------------------
# One fire
# ---------------------------------------------------------------------------

def family(instrument: str) -> str:
    return instrument if instrument in ("MODIS", "VIIRS") else "OLI"


def analyse_fire(fire_id, g: pd.DataFrame):
    g = g.sort_values("t_min")
    w = g["weight"].to_numpy()
    lat0 = float(np.average(g["latitude"], weights=w))
    lon0 = float(np.degrees(np.arctan2(np.average(np.sin(np.radians(g["longitude"])), weights=w),
                                       np.average(np.cos(np.radians(g["longitude"])), weights=w))))
    x, y = to_km(g["latitude"].to_numpy(), g["longitude"].to_numpy(), lat0, lon0)
    g = g.assign(x=x, y=y)

    # observations = satellite passes over this fire
    obs = []
    for (pid, fam), o in g.groupby(["pass_id", "family"], sort=False):
        ow = o["weight"].to_numpy()
        obs.append({"fire_id": fire_id, "pass_id": pid, "family": fam, "satellite": o["satellite"].iloc[0],
                    "t_min": float(o["t_min"].mean()), "daynight": o["daynight"].mode().iloc[0],
                    "x": float(np.average(o["x"], weights=ow)), "y": float(np.average(o["y"], weights=ow)),
                    "pixels": len(o), "burning_area_km2": float(o["area"].sum()),
                    "frp_sum_mw": float(o["frp_mw"].sum()) if o["frp_mw"].notna().any() else math.nan,
                    "bright_max_k": float(o["bright_main_k"].max()) if "bright_main_k" in o and o["bright_main_k"].notna().any() else math.nan,
                    "rows": o.index.to_numpy()})
    obs.sort(key=lambda r: r["t_min"])

    # extent: hull of pixel footprints (square of scan x track around each centre)
    hs, ht = g["scan"].to_numpy() / 2, g["track"].to_numpy() / 2
    cx = np.concatenate([x - hs, x + hs, x - hs, x + hs])
    cy = np.concatenate([y - ht, y - ht, y + ht, y + ht])

    # spread from consecutive passes of one sensor family (prefer VIIRS: finer pixels, more passes)
    new_front = pd.Series(False, index=g.index)
    by_fam = {}
    for o in obs:
        by_fam.setdefault(o["family"], []).append(o)
    basis = next((f for f in ("VIIRS", "MODIS", "OLI") if len(by_fam.get(f, [])) >= 2), None)
    vecs, rates, pairs = [], [], 0
    growth = speed = cbear = math.nan
    if basis:
        seq = by_fam[basis]
        for a, b in zip(seq, seq[1:]):
            gap_h = (b["t_min"] - a["t_min"]) / 60
            if gap_h * 60 < MIN_PAIR_GAP_MIN or gap_h > MAX_PAIR_GAP_H:
                continue
            A, B = g.loc[a["rows"]], g.loc[b["rows"]]
            _, d = nearest(B["x"].to_numpy(), B["y"].to_numpy(), A["x"].to_numpy(), A["y"].to_numpy())
            size = B["pixel_km"].to_numpy()
            new = d > NEW_FRONT_FACTOR * size
            pairs += 1
            if new.any():
                nb = B[new]
                new_front.loc[nb.index] = True
                nw = nb["weight"].to_numpy()
                dx = float(np.average(nb["x"], weights=nw)) - a["x"]
                dy = float(np.average(nb["y"], weights=nw)) - a["y"]
                vecs.append((dx, dy, int(new.sum())))
                rates.append((float(np.median(d[new] - size[new] / 2)) / gap_h, int(new.sum())))
        if len(seq) >= 2:
            t = np.array([o["t_min"] for o in seq]) / 60
            if np.ptp(t) > 0:
                vx = np.polyfit(t, [o["x"] for o in seq], 1)[0]
                vy = np.polyfit(t, [o["y"] for o in seq], 1)[0]
                speed, cbear = float(math.hypot(vx, vy)), bearing(vx, vy)
                growth = float(np.polyfit(t, [o["burning_area_km2"] for o in seq], 1)[0])

    if vecs:  # direction: new-front vectors as unit vectors, weighted by number of new pixels
        ux = sum(n * dx / max(math.hypot(dx, dy), 1e-9) for dx, dy, n in vecs)
        uy = sum(n * dy / max(math.hypot(dx, dy), 1e-9) for dx, dy, n in vecs)
        sdir = bearing(ux, uy)
        # 1 = every pass pair spread the same way; only meaningful with two or more pairs
        consistency = math.hypot(ux, uy) / sum(n for *_, n in vecs) if len(vecs) >= 2 else math.nan
        srate = sum(r * n for r, n in rates) / sum(n for _, n in rates)
    else:
        sdir = consistency = srate = math.nan

    first = obs[0]
    flat, flon = lat0 + first["y"] / KM_PER_DEG_LAT, lon0 + first["x"] / (KM_PER_DEG_LON * math.cos(math.radians(lat0)))
    types = g["type"].astype(str).value_counts()
    fire = {
        "fire_id": fire_id, "detections": len(g), "observations": len(obs),
        "instruments": "+".join(sorted(g["instrument"].unique())), "satellites": "+".join(sorted(g["satellite"].unique())),
        "first_seen_min": first["t_min"], "last_seen_min": obs[-1]["t_min"],
        "duration_h": round((obs[-1]["t_min"] - first["t_min"]) / 60, 2),
        "latitude": round(lat0, 5), "longitude": round(((lon0 + 180) % 360) - 180, 5),
        "first_latitude": round(flat, 5), "first_longitude": round(((flon + 180) % 360) - 180, 5),
        "west": round(g["longitude"].min(), 5), "south": round(g["latitude"].min(), 5),
        "east": round(g["longitude"].max(), 5), "north": round(g["latitude"].max(), 5),
        "hull_area_km2": round(hull_area(cx, cy), 3),
        "max_burning_area_km2": round(max(o["burning_area_km2"] for o in obs), 3),
        "day_and_night": "yes" if g["daynight"].nunique() > 1 else "no",
        "type": types.index[0], "type_confidence": round(float(g["type_confidence"].mean()), 3),
        "spread_basis": basis or "", "spread_pairs": pairs,
        "spread_direction_deg": round(sdir, 1), "spread_consistency": round(consistency, 3),
        "spread_rate_km_h": round(srate, 3), "centre_speed_km_h": round(speed, 3), "centre_bearing_deg": round(cbear, 1),
        "growth_km2_h": round(growth, 4), "new_front_pixels": int(new_front.sum()),
        "status": "single detection" if len(g) == 1 else "single pass" if len(obs) == 1 else
                  "tracked" if pairs else "no comparable passes",
    }
    lat_of = lambda o: round(lat0 + o["y"] / KM_PER_DEG_LAT, 5)
    lon_of = lambda o: round(((lon0 + o["x"] / (KM_PER_DEG_LON * math.cos(math.radians(lat0))) + 180) % 360) - 180, 5)
    obs_rows = [{"fire_id": fire_id, "pass_id": o["pass_id"], "family": o["family"], "satellite": o["satellite"],
                 "t_min": o["t_min"], "daynight": o["daynight"], "latitude": lat_of(o), "longitude": lon_of(o),
                 "pixels": o["pixels"], "burning_area_km2": round(o["burning_area_km2"], 4),
                 "frp_sum_mw": round(o["frp_sum_mw"], 2), "bright_max_k": o["bright_max_k"]} for o in obs]
    det = pd.DataFrame({"row_id": g["row_id"].to_numpy(), "fire_id": fire_id, "pass_id": g["pass_id"].to_numpy(),
                        "new_front": np.where(new_front.to_numpy(), "yes", "no")})
    return fire, obs_rows, det


# ---------------------------------------------------------------------------
# One package (runs in its own process)
# ---------------------------------------------------------------------------

def load_package(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"acq_time": str, "type": str, "pass_id": str}, keep_default_na=False, na_values=[""])
    df["t_min"] = (pd.to_datetime(df["acq_date"]) - pd.Timestamp("1970-01-01")).dt.days * 1440 + df["acq_minutes"]
    df["family"] = df["instrument"].map(family)
    default = df["instrument"].map(DEFAULT_PIXEL_KM).fillna(1.0)
    df["scan"] = df["scan_km"].fillna(default) if "scan_km" in df else default
    df["track"] = df["track_km"].fillna(default) if "track_km" in df else default
    df["pixel_km"] = np.fmax(df["scan"], df["track"])
    df["area"] = df["pixel_area_km2"].fillna(df["scan"] * df["track"])
    frp = df["frp_mw"] if "frp_mw" in df else pd.Series(np.nan, index=df.index)
    df["weight"] = frp.clip(lower=0).fillna(0) + 0.1   # FRP-weighted, every pixel counts a little
    return df


def single_detection_fires(g1: pd.DataFrame) -> tuple:
    """Fast path, same records as analyse_fire, for fires that are a single detection (most fires)."""
    fires, obs = [], []
    has_bt = "bright_main_k" in g1
    for r in g1.itertuples(index=False):
        fid, lat, lon = int(r.unit_id), float(r.latitude), float(r.longitude)
        lon0 = math.degrees(math.atan2(math.sin(math.radians(lon)), math.cos(math.radians(lon))))
        hs, ht = r.scan / 2, r.track / 2
        hull = hull_area(np.array([-hs, hs, -hs, hs]), np.array([-ht, -ht, ht, ht]))
        frp = float(r.frp_mw) if r.frp_mw == r.frp_mw else math.nan
        fires.append({
            "fire_id": fid, "detections": 1, "observations": 1, "instruments": r.instrument, "satellites": r.satellite,
            "first_seen_min": float(r.t_min), "last_seen_min": float(r.t_min), "duration_h": 0.0,
            "latitude": round(lat, 5), "longitude": round(((lon0 + 180) % 360) - 180, 5),
            "first_latitude": round(lat, 5), "first_longitude": round(((lon0 + 180) % 360) - 180, 5),
            "west": round(lon, 5), "south": round(lat, 5), "east": round(lon, 5), "north": round(lat, 5),
            "hull_area_km2": round(hull, 3), "max_burning_area_km2": round(float(r.area), 3), "day_and_night": "no",
            "type": str(r.type), "type_confidence": round(float(r.type_confidence), 3), "spread_basis": "", "spread_pairs": 0,
            "spread_direction_deg": math.nan, "spread_consistency": math.nan, "spread_rate_km_h": math.nan,
            "centre_speed_km_h": math.nan, "centre_bearing_deg": math.nan, "growth_km2_h": math.nan,
            "new_front_pixels": 0, "status": "single detection"})
        obs.append({"fire_id": fid, "pass_id": r.pass_id, "family": r.family, "satellite": r.satellite, "t_min": float(r.t_min),
                    "daynight": r.daynight, "latitude": round(lat, 5), "longitude": round(((lon0 + 180) % 360) - 180, 5),
                    "pixels": 1, "burning_area_km2": round(float(r.area), 4), "frp_sum_mw": round(frp, 2),
                    "bright_max_k": float(r.bright_main_k) if has_bt and r.bright_main_k == r.bright_main_k else math.nan})
    det = pd.DataFrame({"row_id": g1.row_id.to_numpy(), "fire_id": g1.unit_id.astype(int).to_numpy(),
                        "pass_id": g1.pass_id.to_numpy(), "new_front": "no"})
    return fires, obs, det


def process_package(path: str) -> tuple:
    t0 = time.time()
    df = load_package(Path(path))
    n = df.unit_id.map(df.unit_id.value_counts())
    fires, obs, single_det = single_detection_fires(df[n == 1])
    dets = [single_det]
    df = df[n > 1]
    for uid, g in df.groupby("unit_id", sort=False):
        f, o, d = analyse_fire(int(uid), g)
        fires.append(f)
        obs.extend(o)
        dets.append(d)
    return (pd.DataFrame(fires), pd.DataFrame(obs), pd.concat(dets, ignore_index=True), Path(path).name,
            round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def minutes_to_iso(m: pd.Series) -> pd.Series:
    return (pd.Timestamp("1970-01-01") + pd.to_timedelta(m, unit="min")).dt.strftime("%Y-%m-%dT%H:%MZ")


def run(run_dir: Path, processes: int) -> dict:
    t0 = time.time()
    pkgs = sorted((run_dir / "interpreter" / "localisation").glob("package_*.csv.gz"))
    if not pkgs:
        raise SystemExit(f"No localisation packages in {run_dir}. Run interpreter.py first.")
    with ProcessPoolExecutor(max_workers=processes) as pool:
        results = list(pool.map(process_package, map(str, pkgs)))
    fires = pd.concat([r[0] for r in results], ignore_index=True)
    obs = pd.concat([r[1] for r in results], ignore_index=True)
    dets = pd.concat([r[2] for r in results], ignore_index=True)
    for col in ("first_seen_min", "last_seen_min"):
        fires[col.replace("_min", "")] = minutes_to_iso(fires[col])
    fires = fires.drop(columns=["first_seen_min", "last_seen_min"])
    obs["time"] = minutes_to_iso(obs["t_min"])
    obs = obs.drop(columns=["t_min"])
    fires = fires.sort_values(["detections"], ascending=False, ignore_index=True)

    out = run_dir / "localisation"
    out.mkdir(parents=True, exist_ok=True)
    fires.to_csv(out / "fires.csv.gz", index=False)
    obs.to_csv(out / "observations.csv.gz", index=False)
    dets.to_csv(out / "detections.csv.gz", index=False)
    tracked = fires[fires.status == "tracked"]
    summary = {
        "packages": [{"package": r[3], "fires": len(r[0]), "seconds": r[4]} for r in results],
        "fires": len(fires), "detections": int(fires.detections.sum()), "observations": len(obs),
        "status": {k: int(v) for k, v in fires.status.value_counts().items()},
        "tracked": {"fires": len(tracked),
                    "median_spread_rate_km_h": round(float(tracked.spread_rate_km_h.median()), 3) if len(tracked) else None,
                    "median_growth_km2_h": round(float(tracked.growth_km2_h.median()), 4) if len(tracked) else None},
        "largest_hull_km2": round(float(fires.hull_area_km2.max()), 1),
        "settings": {"new_front_factor": NEW_FRONT_FACTOR, "min_pair_gap_min": MIN_PAIR_GAP_MIN,
                     "max_pair_gap_h": MAX_PAIR_GAP_H},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"fires": fires, "observations": obs, "detections": dets, "summary": summary, "out": out}


def print_summary(res: dict) -> None:
    s, f = res["summary"], res["fires"]
    print(f"\n{s['fires']:,} fires from {s['detections']:,} detections ({s['observations']:,} satellite observations) "
          f"in {s['seconds']} s across {len(s['packages'])} packages")
    print("Status: " + ", ".join(f"{k} {v:,}" for k, v in s["status"].items()))
    t = f[f.status == "tracked"]
    if len(t):
        c = t.spread_consistency.dropna()
        print(f"Tracked fires: {len(t):,}; median spread rate {t.spread_rate_km_h.median():.2f} km/h, "
              f"median growth {t.growth_km2_h.median():+.3f} km2/h; direction consistency (fires with 2+ pass pairs, "
              f"n={len(c):,}) median {c.median():.2f}")
    cols = ["fire_id", "detections", "observations", "satellites", "latitude", "longitude", "hull_area_km2",
            "duration_h", "spread_direction_deg", "spread_rate_km_h", "growth_km2_h", "status"]
    print("\nLargest fires:")
    print(f.head(10)[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    if args.run:
        run_dir = Path(args.run)
    elif args.date:
        run_dir = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    else:
        ap.error("give --run or --date")
    res = run(run_dir, args.processes)
    print_summary(res)
    print(f"\nSaved to {res['out']}")


if __name__ == "__main__":
    main()
