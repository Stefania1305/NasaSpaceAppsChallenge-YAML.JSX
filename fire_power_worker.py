"""Fire power worker: stage 3b of the prediction pipeline.

Takes the interpreter's fire-power packages (rows with FRP and brightness;
Landsat and non-wildfire rows already removed) and works out how much power
each fire releases and how that changes over time. Fire ids are the same fire
complexes the localisation worker uses, so the two outputs join on fire_id;
this worker does not have to wait for localisation.

Per detection
- frp_density_mw_km2: FRP / pixel area, a measure of intensity that does not
  grow with pixel size (MODIS pixels are ~10x larger than VIIRS ones)
- frp_anomaly_z: how far FRP is from what the brightness says, per sensor.
  NASA's FRP follows pixel area x (T_main^8 - T_background^8); the worker fits
  log FRP against log of that proxy over ALL packages (one fit per sensor and
  day/night) and reports the residual in standard deviations. Large values
  mark detections whose FRP and brightness disagree (cloud edges, sub-pixel
  hot spots, saturation).

Per satellite pass over a fire (observation)
- FRP sum, max and mean intensity, pixels, local solar time

Per fire
- peak pass FRP and its time, intensity class of the peak
  (< 10 MW small, 10-100 moderate, 100-1000 large, >= 1000 extreme)
- FRP trend from consecutive passes of one sensor family (VIIRS preferred:
  finer pixels and more passes; MODIS misses small fires, so mixing the two
  would look like a change): % change per hour of the FRP sum, from a
  least-squares fit of log FRP over time, only when the passes span at least
  60 min; rising / falling / stable. Also the plain ratio last / first pass.
  Partial views are left out first: a pass that saw 5x fewer pixels than a
  neighbouring pass of the same family less than 60 min away caught only part
  of the fire (edge of the swath, cloud), since no fire grows that fast.
  trend_quality is "good" when every pass used has at least 3 pixels, else
  "few pixels" (FRP of a 1-2 pixel pass is noisy, so the trend is too).
- fire radiative energy FRE (MJ) = FRP integrated over time between the first
  and last pass of that family, and biomass burned = FRE x 0.368 kg/MJ
  (Wooster et al. 2005)
- day / night FRP ratio when the fire was seen both by day and by night

Usage:  python fire_power_worker.py --date 2026-10-13 [--processes N]
Output: <run>/fire_power/fires.csv.gz, observations.csv.gz, detections.csv.gz, summary.json
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

BIOMASS_KG_PER_MJ = 0.368
TREND_STABLE_PCT_H = 5.0      # |trend| below this is "stable"
MIN_TREND_SPAN_MIN = 60      # shorter spans blow up into absurd %/h rates
PARTIAL_RATIO = 5            # a pass with 5x fewer pixels than a neighbouring pass...
PARTIAL_GAP_MIN = 60         # ...less than 60 min away saw only part of the fire
GOOD_TREND_MIN_PIXELS = 3    # trends from passes with fewer pixels are marked "few pixels"
CLASSES = [(10, "small"), (100, "moderate"), (1000, "large"), (math.inf, "extreme")]


def load_package(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"acq_time": str, "type": str, "pass_id": str}, keep_default_na=False, na_values=[""])
    df["t_min"] = (pd.to_datetime(df["acq_date"]) - pd.Timestamp("1970-01-01")).dt.days * 1440 + df["acq_minutes"]
    df["family"] = df["instrument"]
    df["frp_density_mw_km2"] = df["frp_mw"] / df["pixel_area_km2"]
    if "frp_physics_proxy" not in df:
        df["frp_physics_proxy"] = df["pixel_area_km2"] * (df["bright_main_k"] ** 8 - df["bright_secondary_k"] ** 8) * 1e-20
    if "local_solar_hour" not in df:
        df["local_solar_hour"] = (df["acq_minutes"] / 60 + df["longitude"] / 15) % 24
    return df


# ---------------------------------------------------------------------------
# Physics fit (one pass over all packages, before the workers start)
# ---------------------------------------------------------------------------

def fit_physics(paths: list) -> dict:
    """log10 FRP = a + b log10(proxy), per instrument and day/night, over all packages."""
    parts = [pd.read_csv(p, usecols=lambda c: c in {"instrument", "daynight", "frp_mw", "pixel_area_km2",
                                                    "bright_main_k", "bright_secondary_k", "frp_physics_proxy"})
             for p in paths]
    df = pd.concat(parts, ignore_index=True)
    if "frp_physics_proxy" not in df:
        df["frp_physics_proxy"] = df["pixel_area_km2"] * (df["bright_main_k"] ** 8 - df["bright_secondary_k"] ** 8) * 1e-20
    fits = {}
    for (inst, dn), g in df.groupby(["instrument", "daynight"]):
        ok = (g["frp_mw"] > 0) & (g["frp_physics_proxy"] > 0)
        if ok.sum() < 30:
            continue
        x, y = np.log10(g.loc[ok, "frp_physics_proxy"]), np.log10(g.loc[ok, "frp_mw"])
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        fits[f"{inst}|{dn}"] = {"a": float(a), "b": float(b), "sd": float(resid.std()), "n": int(ok.sum()),
                                "r2": float(1 - resid.var() / y.var())}
    return fits


def anomaly_z(df: pd.DataFrame, fits: dict) -> pd.Series:
    z = pd.Series(np.nan, index=df.index)
    for key, f in fits.items():
        inst, dn = key.split("|")
        m = (df["instrument"] == inst) & (df["daynight"] == dn) & (df["frp_mw"] > 0) & (df["frp_physics_proxy"] > 0)
        pred = f["a"] + f["b"] * np.log10(df.loc[m, "frp_physics_proxy"])
        z[m] = (np.log10(df.loc[m, "frp_mw"]) - pred) / f["sd"]
    return z.round(2)


# ---------------------------------------------------------------------------
# One fire
# ---------------------------------------------------------------------------

def intensity_class(frp: float) -> str:
    return next(name for limit, name in CLASSES if frp < limit)


def analyse_fire(fire_id, g: pd.DataFrame):
    obs = []
    for (pid, fam), o in g.groupby(["pass_id", "family"], sort=False):
        frp = o["frp_mw"]
        obs.append({"fire_id": fire_id, "pass_id": pid, "family": fam, "satellite": o["satellite"].iloc[0],
                    "t_min": float(o["t_min"].mean()), "daynight": o["daynight"].mode().iloc[0],
                    "local_solar_hour": round(float(o["local_solar_hour"].mean()), 2),
                    "pixels": len(o), "frp_sum_mw": round(float(frp.sum()), 2), "frp_max_mw": round(float(frp.max()), 2),
                    "frp_density_mw_km2": round(float(frp.sum() / o["pixel_area_km2"].sum()), 3),
                    "bright_max_k": round(float(o["bright_main_k"].max()), 2),
                    "high_confidence_share": round(float((o["confidence"] == "high").mean()), 3),
                    "partial_view": "no"})
    obs.sort(key=lambda r: r["t_min"])
    peak = max(obs, key=lambda r: r["frp_sum_mw"])

    by_fam = {}
    for o in obs:
        by_fam.setdefault(o["family"], []).append(o)
    basis = next((f for f in ("VIIRS", "MODIS") if len(by_fam.get(f, [])) >= 2), None)
    trend = fre = ratio = math.nan
    quality = ""
    partial = 0
    if basis:
        seq = [o for o in by_fam[basis] if o["frp_sum_mw"] > 0]
        drop = set()
        for k in range(len(seq) - 1):
            a, b = seq[k], seq[k + 1]
            if b["t_min"] - a["t_min"] < PARTIAL_GAP_MIN:
                if a["pixels"] * PARTIAL_RATIO < b["pixels"]:
                    drop.add(k)
                elif b["pixels"] * PARTIAL_RATIO < a["pixels"]:
                    drop.add(k + 1)
        for k in drop:
            seq[k]["partial_view"] = "yes"
        partial = len(drop)
        seq = [o for k, o in enumerate(seq) if k not in drop]
        t = np.array([o["t_min"] for o in seq])
        if len(seq) >= 2:
            ratio = seq[-1]["frp_sum_mw"] / seq[0]["frp_sum_mw"]
            frp = np.array([o["frp_sum_mw"] for o in seq])
            fre = float(np.sum((frp[1:] + frp[:-1]) / 2 * np.diff(t) * 60))   # MW x s = MJ
        if len(seq) >= 2 and np.ptp(t) >= MIN_TREND_SPAN_MIN:
            slope = np.polyfit(t / 60, np.log([o["frp_sum_mw"] for o in seq]), 1)[0]
            trend = (math.exp(slope) - 1) * 100
            quality = "good" if min(o["pixels"] for o in seq) >= GOOD_TREND_MIN_PIXELS else "few pixels"
    label = ("" if math.isnan(trend) else "stable" if abs(trend) < TREND_STABLE_PCT_H
             else "rising" if trend > 0 else "falling")

    day = [o["frp_sum_mw"] for o in obs if o["daynight"] == "D"]
    night = [o["frp_sum_mw"] for o in obs if o["daynight"] == "N"]
    fire = {
        "fire_id": fire_id, "detections": len(g), "observations": len(obs),
        "instruments": "+".join(sorted(g["instrument"].unique())),
        "peak_frp_mw": peak["frp_sum_mw"], "peak_time_min": peak["t_min"], "peak_family": peak["family"],
        "peak_class": intensity_class(peak["frp_sum_mw"]),
        "max_pixel_frp_mw": round(float(g["frp_mw"].max()), 2),
        "mean_frp_density_mw_km2": round(float(g["frp_mw"].sum() / g["pixel_area_km2"].sum()), 3),
        "trend_basis": basis or "", "trend_passes": len(by_fam.get(basis, [])) - partial if basis else 0,
        "partial_view_passes": partial,
        "frp_trend_pct_h": round(trend, 2), "frp_trend": label, "trend_quality": quality, "frp_last_first_ratio": round(ratio, 3),
        "fre_mj": round(fre, 1), "biomass_burned_t": round(fre * BIOMASS_KG_PER_MJ / 1000, 2),
        "day_night_frp_ratio": round(float(np.mean(day) / np.mean(night)), 3) if day and night and np.mean(night) > 0 else math.nan,
        "max_bright_k": round(float(g["bright_main_k"].max()), 2),
        "mean_bright_diff_k": round(float((g["bright_main_k"] - g["bright_secondary_k"]).mean()), 2),
        "frp_anomalies": int((g["frp_anomaly_z"].abs() >= 3).sum()),
        "high_confidence_share": round(float((g["confidence"] == "high").mean()), 3),
    }
    return fire, obs


def process_package(args) -> tuple:
    path, fits = args
    t0 = time.time()
    df = load_package(path)
    df["frp_anomaly_z"] = anomaly_z(df, fits)
    fires, obs = [], []
    for uid, g in df.groupby("unit_id", sort=False):
        f, o = analyse_fire(int(uid), g)
        fires.append(f)
        obs.extend(o)
    det = pd.DataFrame({"row_id": df["row_id"], "fire_id": df["unit_id"],
                        "frp_density_mw_km2": df["frp_density_mw_km2"].round(3), "frp_anomaly_z": df["frp_anomaly_z"]})
    return pd.DataFrame(fires), pd.DataFrame(obs), det, Path(path).name, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def minutes_to_iso(m: pd.Series) -> pd.Series:
    return (pd.Timestamp("1970-01-01") + pd.to_timedelta(m, unit="min")).dt.strftime("%Y-%m-%dT%H:%MZ")


def run(run_dir: Path, processes: int) -> dict:
    t0 = time.time()
    pkgs = sorted((run_dir / "interpreter" / "fire_power").glob("package_*.csv.gz"))
    if not pkgs:
        raise SystemExit(f"No fire-power packages in {run_dir}. Run interpreter.py first.")
    fits = fit_physics(pkgs)
    with ProcessPoolExecutor(max_workers=processes) as pool:
        results = list(pool.map(process_package, [(str(p), fits) for p in pkgs]))
    fires = pd.concat([r[0] for r in results], ignore_index=True)
    obs = pd.concat([r[1] for r in results], ignore_index=True)
    dets = pd.concat([r[2] for r in results], ignore_index=True)
    fires["peak_time"] = minutes_to_iso(fires["peak_time_min"])
    fires = fires.drop(columns=["peak_time_min"]).sort_values("peak_frp_mw", ascending=False, ignore_index=True)
    obs["time"] = minutes_to_iso(obs["t_min"])
    obs = obs.drop(columns=["t_min"])

    out = run_dir / "fire_power"
    out.mkdir(parents=True, exist_ok=True)
    fires.to_csv(out / "fires.csv.gz", index=False)
    obs.to_csv(out / "observations.csv.gz", index=False)
    dets.to_csv(out / "detections.csv.gz", index=False)

    # consistency with the localisation worker, when it has run
    loc = run_dir / "localisation" / "fires.csv.gz"
    joined = None
    if loc.exists():
        lf = pd.read_csv(loc, usecols=["fire_id"])
        joined = {"fire_power_fires_found_in_localisation": int(fires.fire_id.isin(lf.fire_id).sum()),
                  "fire_power_fires": len(fires)}
    t = fires[fires.frp_trend != ""]
    summary = {
        "packages": [{"package": r[3], "fires": len(r[0]), "seconds": r[4]} for r in results],
        "fires": len(fires), "detections": len(dets), "observations": len(obs),
        "physics_fit": fits,
        "peak_class": {k: int(v) for k, v in fires.peak_class.value_counts().items()},
        "trend": {k: int(v) for k, v in t.frp_trend.value_counts().items()},
        "trend_quality": {k: int(v) for k, v in t.trend_quality.value_counts().items()},
        "trend_good_median_pct_h": round(float(t[t.trend_quality == "good"].frp_trend_pct_h.median()), 2) if len(t) else None,
        "partial_view_passes": int(fires.partial_view_passes.sum()),
        "total_fre_tj": round(float(fires.fre_mj.sum()) / 1e6, 2),
        "total_biomass_burned_kt": round(float(fires.biomass_burned_t.sum()) / 1e3, 2),
        "frp_anomalies_abs_z_ge_3": int((dets.frp_anomaly_z.abs() >= 3).sum()),
        "join_with_localisation": joined,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"fires": fires, "observations": obs, "detections": dets, "summary": summary, "out": out}


def print_summary(res: dict) -> None:
    s, f = res["summary"], res["fires"]
    print(f"\n{s['fires']:,} fires from {s['detections']:,} detections ({s['observations']:,} observations) "
          f"in {s['seconds']} s across {len(s['packages'])} packages")
    print("Physics fit, log FRP vs log (area x (T_main^8 - T_secondary^8)):")
    for k, v in s["physics_fit"].items():
        print(f"  {k:<8} slope {v['b']:.3f}  R2 {v['r2']:.3f}  residual sd {v['sd']:.3f} (log10)  n={v['n']:,}")
    print(f"FRP anomalies (|z| >= 3): {s['frp_anomalies_abs_z_ge_3']:,}")
    print("Peak intensity: " + ", ".join(f"{k} {v:,}" for k, v in s["peak_class"].items()))
    print("FRP trend (fires with 2+ passes of one sensor family): " + ", ".join(f"{k} {v:,}" for k, v in s["trend"].items()))
    print("Trend quality: " + ", ".join(f"{k} {v:,}" for k, v in s["trend_quality"].items()) +
          # None when no fire has a good trend (e.g. a small zone): print a dash instead of failing
          "; median trend of good ones " + (f"{s['trend_good_median_pct_h']:+.1f} %/h"
                                            if s.get("trend_good_median_pct_h") is not None else "—"))
    print(f"Partial views left out of trends: {s['partial_view_passes']:,} passes")
    print(f"Total fire radiative energy {s['total_fre_tj']:,} TJ, biomass burned {s['total_biomass_burned_kt']:,} kt "
          "(between first and last pass only)")
    if s["join_with_localisation"]:
        j = s["join_with_localisation"]
        print(f"Join with localisation: {j['fire_power_fires_found_in_localisation']:,} of {j['fire_power_fires']:,} fires match")
    cols = ["fire_id", "detections", "observations", "peak_frp_mw", "peak_class", "peak_time", "frp_trend_pct_h",
            "frp_trend", "fre_mj", "biomass_burned_t", "day_night_frp_ratio"]
    print("\nMost powerful fires:")
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
