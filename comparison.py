"""Make MODIS and VIIRS fire data directly comparable.

Each known sensor difference is resolved as follows:

1. Pixel size (MODIS ~1 km, up to ~4.8 x 2 km at swath edge; VIIRS 375 m)
   -> both are aggregated onto one common lat/lon grid (default 0.05 deg, larger
      than the biggest MODIS pixel) and compared cell by cell.
2. FRP split (one MODIS row = whole 1 km area; VIIRS spreads the same fire over
   several sub-pixel rows) -> FRP is SUMMED per cell, so each fire's total energy
   is counted once for either sensor.
3. Raw counts (VIIRS 4-6x more pixels) -> fires are counted as occupied grid
   cells, not pixels.
4. Detection limit (VIIRS sees much smaller fires) -> one common minimum fire
   size, calibrated from the data (see detection_limit()), is applied to both:
   cells whose FRP is below it are left out for BOTH sensors.
5. Confidence (MODIS 0-100, VIIRS l/n/h) -> both on low / nominal / high
   (MODIS <30 low, 30-79 nominal, >=80 high, per the FIRMS documentation).
6. Saturation (MODIS ~500 K, VIIRS I4 clips at 367 K) -> brightness is capped at
   367 K for both sensors and flagged with bright_saturated.
7. Overpass times (Terra ~10:30/22:30 has no VIIRS equivalent) -> only Aqua is
   compared with a single VIIRS satellite on a similar afternoon/night schedule,
   matched day with day and night with night. The remaining time between the two
   passes is reported per cell (overpass_gap_min) and can be capped.
8. MODIS swath gaps near the equator (VIIRS is gapless) -> each satellite's
   ground tracks are reconstructed from its own detections (swath.py). A fire
   seen by only one sensor at a place the other did not observe is left out.
"""

import math
from collections import Counter, defaultdict

from swath import SwathModel

COMMON_BT_MAX_K = 367.0          # VIIRS I4 saturation; MODIS values are capped to it
FALLBACK_MIN_FRP_MW = 10.0
COMPARE_SATELLITE_MODIS = "Aqua"
CONF_RANK = {"": 0, "low": 1, "nominal": 2, "high": 3}


# ---------------------------------------------------------------------------
# Pixel-level helpers (also used in the harmonised table)
# ---------------------------------------------------------------------------

def overpass_slot(acq_time: str, lon: str) -> str:
    """Approximate local solar time of the overpass -> morning/afternoon/evening/night."""
    try:
        t = acq_time.zfill(4)
        utc_h = int(t[:2]) + int(t[2:]) / 60
        local = (utc_h + float(lon) / 15) % 24
    except (ValueError, TypeError):
        return ""
    if 6 <= local < 12:
        return "morning"
    if 12 <= local < 18:
        return "afternoon"
    if 18 <= local < 24:
        return "evening"
    return "night"


def clip_brightness(value: str) -> tuple[str, str]:
    """Cap brightness at the common 367 K ceiling. Returns (value, saturated yes/no)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "", ""
    if v >= COMMON_BT_MAX_K:
        return f"{COMMON_BT_MAX_K:.2f}", "yes"
    return value, "no"


def _minutes(acq_time: str) -> int:
    t = acq_time.zfill(4)
    return int(t[:2]) * 60 + int(t[2:])


# ---------------------------------------------------------------------------
# Grid comparison
# ---------------------------------------------------------------------------

def _new_cell():
    return {"pixels": 0, "frp": 0.0, "frp_max": 0.0, "bt": 0.0, "sat": False,
            "conf": "", "types": Counter(), "area": 0.0, "tsum": 0}


def detection_limit(cells) -> tuple[float, str]:
    """Common minimum fire size, calibrated from cells that BOTH satellites observed.

    For VIIRS fires of increasing FRP, count how often Aqua MODIS also detected
    the cell. Large fires are missed only because the passes are hours apart
    (the fire changed); small fires are missed for that reason AND because MODIS
    is less sensitive. So the hit rate rises with fire size to a plateau set by
    timing. The limit is the FRP where MODIS reaches half of that plateau,
    which separates sensitivity from timing."""
    pairs = sorted((c["v_frp"], c["m_hit"]) for c in cells if c["v_frp"] > 0 and c["both_observed"])
    if len(pairs) < 500:
        return FALLBACK_MIN_FRP_MW, f"fixed fallback ({len(pairs)} co-observed VIIRS cells; 500 needed to calibrate)"
    size = max(50, len(pairs) // 25)                      # ~25 equal-count FRP bins
    bins = [pairs[k:k + size] for k in range(0, len(pairs) - size + 1, size)]
    rates = [sum(h for _, h in b) / len(b) for b in bins]
    plateau = sum(rates[-2:]) / 2
    for b, rate in zip(bins, rates):
        if rate >= plateau / 2:
            return round(b[0][0], 1), (f"fire size where Aqua MODIS reaches half its large-fire hit rate "
                                       f"({plateau:.0%}, limited by pass timing); calibrated on "
                                       f"{len(pairs):,} cells both satellites observed")
    return FALLBACK_MIN_FRP_MW, "fixed fallback (no clear sensitivity curve)"


COMPARE_COLUMNS = [
    "cell_lat", "cell_lon", "acq_date", "daynight", "detected_by", "comparable", "reason",
    "overpass_gap_min", "modis_pixels", "viirs_pixels", "modis_frp_mw", "viirs_frp_mw",
    "modis_frp_max_mw", "viirs_frp_max_mw", "modis_bt_max_k", "viirs_bt_max_k",
    "modis_confidence", "viirs_confidence", "modis_pixel_area_km2", "viirs_pixel_area_km2", "type",
]


def compare(columns: list, rows: list, cell_deg: float, viirs_label: str,
            max_gap_min: float | None = None) -> dict:
    """Aggregate harmonised MODIS + VIIRS rows onto a common grid and compare."""
    c = {name: i for i, name in enumerate(columns)}
    removed = Counter()
    raw = {"MODIS": {"pixels": 0, "frp": 0.0}, "VIIRS": {"pixels": 0, "frp": 0.0}}
    grid = defaultdict(lambda: {"MODIS": _new_cell(), "VIIRS": _new_cell()})
    swaths = {"MODIS": SwathModel("MODIS"), "VIIRS": SwathModel("VIIRS")}
    fit_input = {"MODIS": [], "VIIRS": []}

    for r in rows:
        inst = r[c["instrument"]]
        if inst not in ("MODIS", "VIIRS"):
            continue
        if inst == "MODIS" and r[c["satellite"]] != COMPARE_SATELLITE_MODIS:
            removed["terra"] += 1          # no VIIRS overpass at Terra's times
            continue
        try:
            lat, lon = float(r[c["latitude"]]), float(r[c["longitude"]])
            frp = float(r[c["frp_mw"]] or 0)
            minutes = _minutes(r[c["acq_time"]])
            area = float(r[c["pixel_area_km2"]] or 0)
        except ValueError:
            continue
        day, dn = r[c["acq_date"]], r[c["daynight"]]
        fit_input[inst].append((day, minutes, lat, lon, dn, area))
        raw[inst]["pixels"] += 1
        raw[inst]["frp"] += frp
        cell = grid[(math.floor(lat / cell_deg), math.floor(lon / cell_deg), day, dn)][inst]
        cell["pixels"] += 1
        cell["frp"] += frp
        cell["frp_max"] = max(cell["frp_max"], frp)
        cell["tsum"] += minutes
        cell["area"] += area
        try:
            cell["bt"] = max(cell["bt"], float(r[c["bright_main_k"]]))
        except ValueError:
            pass
        if r[c["bright_saturated"]] == "yes":
            cell["sat"] = True
            if inst == "MODIS":
                removed["modis_clipped"] += 1
        if CONF_RANK.get(r[c["confidence"]], 0) > CONF_RANK[cell["conf"]]:
            cell["conf"] = r[c["confidence"]]
        cell["types"][r[c["type"]]] += 1

    for inst, model in swaths.items():
        model.fit(fit_input[inst])

    # Pass 1: coverage and timing for every cell
    info = []
    for (i, j, day, dn), pair in grid.items():
        m, v = pair["MODIS"], pair["VIIRS"]
        lat_c, lon_c = (i + 0.5) * cell_deg, (j + 0.5) * cell_deg
        times, observed = {}, {}
        for inst, cell in (("MODIS", m), ("VIIRS", v)):
            p = swaths[inst].nearest_pass(day, lat_c, lon_c, dn)
            if cell["pixels"]:
                observed[inst] = True
                times[inst] = cell["tsum"] / cell["pixels"]
            else:
                observed[inst] = None if p is None else p[0] <= swaths[inst].half
                times[inst] = None if p is None else p[1]
        gap = abs(times["MODIS"] - times["VIIRS"]) if None not in times.values() else None
        info.append({"key": (i, j, day, dn), "m": m, "v": v, "observed": observed, "gap": gap,
                     "v_frp": v["frp"], "m_hit": m["pixels"] > 0,
                     "both_observed": observed["MODIS"] is True and observed["VIIRS"] is True})

    min_frp, threshold_note = detection_limit(info)

    # Pass 2: decide comparability
    out_rows = []
    agree = Counter()
    comp = {"MODIS": {"cells": 0, "frp": 0.0}, "VIIRS": {"cells": 0, "frp": 0.0}}
    all_cells = Counter()
    for x in info:
        (i, j, day, dn), m, v = x["key"], x["m"], x["v"]
        has_m, has_v = m["pixels"] > 0, v["pixels"] > 0
        all_cells["MODIS"] += has_m
        all_cells["VIIRS"] += has_v
        detected = "both" if has_m and has_v else "MODIS only" if has_m else "VIIRS only"
        fire_size = max(m["frp"], v["frp"])
        comparable, reason = "yes", ""
        if detected == "VIIRS only" and x["observed"]["MODIS"] is False:
            comparable, reason = "no", "outside Aqua MODIS swath (not observed)"
            removed["modis_gap"] += 1
        elif detected == "MODIS only" and x["observed"]["VIIRS"] is False:
            comparable, reason = "no", "outside VIIRS swath (not observed)"
            removed["viirs_gap"] += 1
        elif fire_size < min_frp:
            comparable, reason = "no", f"below common detection limit ({min_frp} MW)"
            removed["below_limit_" + ("both" if detected == "both" else detected.split()[0])] += 1
        elif max_gap_min is not None and x["gap"] is not None and x["gap"] > max_gap_min:
            comparable, reason = "no", f"passes more than {max_gap_min:g} min apart"
            removed["time_gap"] += 1
        if comparable == "yes":
            agree[detected] += 1
            for inst, cell in (("MODIS", m), ("VIIRS", v)):
                if cell["pixels"]:
                    comp[inst]["cells"] += 1
                    comp[inst]["frp"] += cell["frp"]
        types = m["types"] + v["types"]
        fmt = lambda val, has: f"{val:.2f}" if has else ""
        out_rows.append([
            f"{(i + 0.5) * cell_deg:.4f}", f"{(j + 0.5) * cell_deg:.4f}", day, dn, detected, comparable, reason,
            "" if x["gap"] is None else f"{x['gap']:.0f}",
            str(m["pixels"]), str(v["pixels"]),
            fmt(m["frp"], has_m), fmt(v["frp"], has_v),
            fmt(m["frp_max"], has_m), fmt(v["frp_max"], has_v),
            (fmt(m["bt"], has_m) + (" (sat)" if m["sat"] else "")) if has_m else "",
            (fmt(v["bt"], has_v) + (" (sat)" if v["sat"] else "")) if has_v else "",
            m["conf"], v["conf"],
            f"{m['area']:.3f}" if has_m else "", f"{v['area']:.3f}" if has_v else "",
            types.most_common(1)[0][0] if types else "",
        ])
    out_rows.sort(key=lambda r: (r[2], r[3], -float(r[10] or 0) - float(r[11] or 0)))

    def ratio(a, b):
        return f"{b / a:.2f}" if a else ""

    gaps = sorted(x["gap"] for x in info if x["gap"] is not None)
    fits = {inst: {d: {"detections_used": f[2]} for d, f in s.fits.items()} for inst, s in swaths.items()}
    summary = [
        {"metric": "Raw pixels (detections)", "modis": raw["MODIS"]["pixels"], "viirs": raw["VIIRS"]["pixels"],
         "ratio": ratio(raw["MODIS"]["pixels"], raw["VIIRS"]["pixels"])},
        {"metric": "Raw total FRP (MW)", "modis": round(raw["MODIS"]["frp"]), "viirs": round(raw["VIIRS"]["frp"]),
         "ratio": ratio(raw["MODIS"]["frp"], raw["VIIRS"]["frp"])},
        {"metric": f"Fire cells ({cell_deg}° grid)", "modis": all_cells["MODIS"], "viirs": all_cells["VIIRS"],
         "ratio": ratio(all_cells["MODIS"], all_cells["VIIRS"])},
        {"metric": "Comparable fire cells", "modis": comp["MODIS"]["cells"], "viirs": comp["VIIRS"]["cells"],
         "ratio": ratio(comp["MODIS"]["cells"], comp["VIIRS"]["cells"])},
        {"metric": "Comparable total FRP (MW)", "modis": round(comp["MODIS"]["frp"]), "viirs": round(comp["VIIRS"]["frp"]),
         "ratio": ratio(comp["MODIS"]["frp"], comp["VIIRS"]["frp"])},
    ]
    return {
        "columns": COMPARE_COLUMNS,
        "rows": out_rows,
        "summary": summary,
        "agreement": {"both": agree["both"], "MODIS only": agree["MODIS only"], "VIIRS only": agree["VIIRS only"]},
        "settings": {
            "cell_deg": cell_deg, "min_frp_mw": min_frp, "min_frp_basis": threshold_note,
            "bt_ceiling_k": COMMON_BT_MAX_K, "modis_satellite": COMPARE_SATELLITE_MODIS,
            "viirs": viirs_label, "max_gap_min": max_gap_min,
            "median_overpass_gap_min": round(gaps[len(gaps) // 2]) if gaps else None,
            "swath_fits": fits,
        },
        "removed": {
            "Terra MODIS pixels (no matching VIIRS overpass)": removed["terra"],
            "Aqua MODIS pixels capped at 367 K": removed["modis_clipped"],
            "VIIRS-only cells outside the Aqua swath": removed["modis_gap"],
            "MODIS-only cells outside the VIIRS swath": removed["viirs_gap"],
            "Cells below the common detection limit (VIIRS only / MODIS only / both)":
                f"{removed['below_limit_VIIRS']:,} / {removed['below_limit_MODIS']:,} / {removed['below_limit_both']:,}",
            "Cells with passes too far apart in time": removed["time_gap"],
        },
    }
