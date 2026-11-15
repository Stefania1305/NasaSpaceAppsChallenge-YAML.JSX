"""Animated GIF of how a fire spreads: the last observed days, then the predicted next 3 days.

Observed frames draw every satellite detection (its real pixel size): today's detections bright,
earlier ones as the burned scar. Predicted frames keep the scar and draw, for every active fire in view,
the likely footprint (filled), the high footprint (dashed ring, 90th percentile) and the spread direction,
with the chance it merges or goes out.

Usage:  python prediction_animation.py --run data/runs/fires_batch_039 [--fire 1501] [--half-km 15] [--past 4]
Needs:  <run>/prediction/predictions.csv (from prediction_engine.py)
Output: <run>/prediction/spread_<fire>.gif  (and the frames as PNG next to it)
"""

import argparse
import io
import math
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from prediction_data import KM_LAT, KM_LON, fire_days

BG, PANEL, GRID, INK, MUTED = "#12161b", "#1a2028", "#2a323c", "#e8ecf0", "#8d99a8"
SCAR, ACTIVE, PRED, HIGH = "#6b2a1f", "#ffb02e", "#ff6a3d", "#ffc59e"


def day_label(d: str) -> str:
    return datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--fire", type=int, help="fire to centre on (default: the largest predicted)")
    ap.add_argument("--half-km", type=float, default=15.0, help="half width of the view, km")
    ap.add_argument("--past", type=int, default=4, help="observed frames, today included")
    ap.add_argument("--ms", type=int, default=1100, help="milliseconds per frame")
    args = ap.parse_args()
    run = Path(args.run)
    P = pd.read_csv(run / "prediction" / "predictions.csv")
    today = P.from_day.iloc[0]
    fid = args.fire or int(P[P.day_ahead == 3].sort_values("footprint_likely_km2", ascending=False).fire_id.iloc[0])
    f0 = P[P.fire_id == fid].iloc[0]
    lat0, lon0, R = f0.lat0, f0.lon0, args.half_km
    kx = KM_LON * math.cos(math.radians(lat0))
    to_xy = lambda lat, lon: ((np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * KM_LAT)

    # observed detections in view
    det = pd.read_csv(run / "kg" / "nodes_Detection.csv.gz", usecols=["latitude:double", "longitude:double", "acq_date:string",
                      "frp_mw:double", "scan_km:double", "track_km:double", "removed_by_interpreter:string"])
    det.columns = ["lat", "lon", "day", "frp", "scan", "track", "removed"]
    det = det[det.removed.isna()]
    det["x"], det["y"] = to_xy(det.lat, det.lon)
    det = det[(det.x.abs() <= R * 1.05) & (det.y.abs() <= R * 1.05)]
    days_obs = [(pd.Timestamp(today) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.past - 1, -1, -1)]
    det = det[det.day <= today]

    # fires in view and their footprint history
    P["x"], P["y"] = to_xy(P.lat0, P.lon0)
    view = P[(P.x.abs() <= R) & (P.y.abs() <= R)]
    ids = sorted(view.fire_id.unique())
    fd = fire_days(run, with_targets=False)
    hist = fd[fd.fire_id.isin(ids)].pivot_table(index="day", columns="fire_id", values="fine_fp", aggfunc="max")
    hist = hist.reindex(sorted(set(hist.index) | set(days_obs))).ffill().fillna(0)
    area_obs = [float(hist.loc[d].sum()) if d in hist.index else 0.0 for d in days_obs]
    area_pred = [view[view.day_ahead == k][["footprint_low_km2", "footprint_likely_km2", "footprint_high_km2"]].sum().to_numpy()
                 for k in (1, 2, 3)]

    frames = [("obs", d, None) for d in days_obs] + [("pred", P[P.day_ahead == k].date.iloc[0], k) for k in (1, 2, 3)]
    out_dir = run / "prediction"
    images = []
    for i, (kind, d, k) in enumerate(frames):
        fig = plt.figure(figsize=(9.6, 6.0), dpi=100, facecolor=BG)
        ax = fig.add_axes([0.04, 0.08, 0.56, 0.78], facecolor=PANEL)
        ch = fig.add_axes([0.67, 0.42, 0.30, 0.44], facecolor=BG)
        # header
        fig.text(0.04, 0.93, f"Fire {fid} and neighbours", color=INK, fontsize=15, fontweight="bold")
        tag = "OBSERVED" if kind == "obs" else f"PREDICTED  ·  DAY +{k}"
        fig.text(0.04, 0.885, f"{tag}  ·  {day_label(d)}", color=ACTIVE if kind == "obs" else PRED, fontsize=11, fontweight="bold")
        fig.text(0.60, 0.93, f"{lat0:.3f}°, {lon0:.3f}°", color=MUTED, fontsize=10, ha="right")
        # map
        last_obs = d if kind == "obs" else today
        scar = det[det.day < last_obs] if kind == "obs" else det
        act = det[det.day == d] if kind == "obs" else det.iloc[0:0]
        for r in scar.itertuples():
            ax.add_patch(Rectangle((r.x - r.scan / 2, r.y - r.track / 2), r.scan, r.track, color=SCAR, lw=0, alpha=0.9))
        for r in act.itertuples():
            a = min(1.0, 0.55 + math.log10(1 + max(r.frp, 0)) / 4)
            ax.add_patch(Rectangle((r.x - r.scan / 2, r.y - r.track / 2), r.scan, r.track, color=ACTIVE, lw=0, alpha=a))
        if kind == "pred":
            vk = view[view.day_ahead == k]
            for r in vk.itertuples():
                cx, cy = to_xy(r.centre_lat, r.centre_lon)
                rl, rh = math.sqrt(r.footprint_likely_km2 / math.pi), math.sqrt(r.footprint_high_km2 / math.pi)
                ax.add_patch(Circle((cx, cy), rh, fill=False, ec=HIGH, lw=1.1, ls=(0, (3, 2)), alpha=0.9))
                ax.add_patch(Circle((cx, cy), rl, color=PRED, alpha=0.55, lw=0))
                if r.move_km > 0.05:
                    a = math.radians(r.direction_deg)
                    L = rh + 1.0 + min(r.spread_rate_km_day, 3) * 1.2
                    ax.add_patch(FancyArrowPatch((cx + math.sin(a) * rl, cy + math.cos(a) * rl), (cx + math.sin(a) * L, cy + math.cos(a) * L),
                                                 arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.3))
            for r in vk.sort_values("footprint_likely_km2", ascending=False).head(4).itertuples():
                cx, cy = to_xy(r.centre_lat, r.centre_lon)
                rh = math.sqrt(r.footprint_high_km2 / math.pi)
                ax.text(cx + rh + 0.4, cy - rh - 0.2, f"{r.footprint_likely_km2:.1f} km²\nmerge {r.p_merge:.0%} · out {r.p_out:.0%}",
                        color=INK, fontsize=7.5, va="top", linespacing=1.25)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xticks(np.arange(-R, R + 0.1, 5)); ax.set_yticks(np.arange(-R, R + 0.1, 5))
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("km east", color=MUTED, fontsize=8); ax.set_ylabel("km north", color=MUTED, fontsize=8)
        ax.text(R - 0.5, R - 0.6, "N ↑", color=MUTED, fontsize=9, ha="right", va="top")
        # area chart
        xs_o, xs_p = list(range(len(days_obs))), list(range(len(days_obs) - 1, len(frames)))
        lo = [area_obs[-1]] + [a[0] for a in area_pred]
        mid = [area_obs[-1]] + [a[1] for a in area_pred]
        hi = [area_obs[-1]] + [a[2] for a in area_pred]
        ch.fill_between(xs_p, lo, hi, color=PRED, alpha=0.18, lw=0)
        ch.plot(xs_o, area_obs, color=ACTIVE, lw=2, marker="o", ms=4)
        ch.plot(xs_p, mid, color=PRED, lw=2, ls="--", marker="o", ms=4)
        cur = (area_obs + mid[1:])[i]
        ch.scatter([i], [cur], s=70, color=ACTIVE if kind == "obs" else PRED, zorder=5, edgecolor=INK, lw=1.2)
        ch.axvline(len(days_obs) - 1, color=GRID, lw=1)
        ch.set_xticks(range(len(frames)))
        ch.set_xticklabels([datetime.strptime(fr[1], "%Y-%m-%d").strftime("%d.%m") for fr in frames], fontsize=7.5, color=MUTED, rotation=0)
        ch.tick_params(colors=MUTED, labelsize=7.5)
        for s in ch.spines.values():
            s.set_color(GRID)
        ch.set_ylim(0, max(hi + area_obs) * 1.12 or 1)
        ch.set_title("Burned footprint of fires in view, km²", color=INK, fontsize=9, loc="left")
        ch.text(0.02, 0.92, "observed", color=ACTIVE, fontsize=8, transform=ch.transAxes)
        ch.text(0.98, 0.92, "predicted (likely, low–high)", color=PRED, fontsize=8, transform=ch.transAxes, ha="right")
        # legend / notes
        y = 0.32
        for color, text, style in [(ACTIVE, "active fire pixels that day (satellite)", "sq"), (SCAR, f"burned earlier (since {day_label(det.day.min())})", "sq"),
                                   (PRED, "predicted likely footprint", "c"), (HIGH, "predicted high footprint (90%)", "ring"),
                                   (INK, "predicted spread direction", "arrow")]:
            fig.text(0.67, y, {"sq": "■", "c": "●", "ring": "◌", "arrow": "➜"}[style], color=color, fontsize=12, va="center")
            fig.text(0.695, y, text, color=MUTED, fontsize=8.5, va="center")
            y -= 0.045
        fig.text(0.67, 0.06, f"{i + 1}/{len(frames)}  ·  run {run.name}", color=MUTED, fontsize=7.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        img = Image.open(buf).convert("RGB")
        img.save(out_dir / f"spread_{fid}_frame{i + 1}.png")
        images.append(img)
    gif = out_dir / f"spread_{fid}.gif"
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    durations = [args.ms] * (len(pal) - 1) + [args.ms * 2]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{len(frames)} frames, {len(ids)} fires in view -> {gif} ({gif.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
