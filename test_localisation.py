"""Offline checks for localisation_worker.py with synthetic fires of known movement (no network).

Run: python test_localisation.py
"""
import math
import tempfile
from pathlib import Path

import pandas as pd

import localisation_worker as L

STEP = 0.4 / L.KM_PER_DEG_LON          # 0.4 km in longitude at the equator
STEP_LAT = 0.4 / L.KM_PER_DEG_LAT


def pixel(row_id, unit, lat, lon, t, sat, pass_id, frp=5.0):
    return {"latitude": lat, "longitude": lon, "acq_date": "2026-10-13", "acq_time": f"{t // 60:02d}{t % 60:02d}",
            "acq_minutes": t, "scan_km": 0.4, "track_km": 0.4, "pixel_area_km2": 0.16, "daynight": "D",
            "instrument": "VIIRS", "satellite": sat, "confidence": "nominal", "frp_mw": frp, "type": "0",
            "type_confidence": 0.9, "bright_main_k": 330.0, "pass_id": pass_id, "unit_id": unit, "row_id": row_id}


def test_known_spread():
    rows, rid = [], 0
    for gx in range(3):                      # pass 1, 13:00: 3 x 3 block at x = 0..0.8 km
        for gy in range(3):
            rows.append(pixel(rid, 1, gy * STEP_LAT, gx * STEP, 780, "NOAA-20", "NOAA-20#1")); rid += 1
    for gx in list(range(3)) + [5, 6, 7]:   # pass 2, 14:40: same block plus a block 2.0..2.8 km east
        for gy in range(3):
            rows.append(pixel(rid, 1, gy * STEP_LAT, gx * STEP, 880, "NOAA-21", "NOAA-21#1")); rid += 1
    rows.append(pixel(rid, 2, 30.0, 30.0, 800, "NOAA-20", "NOAA-20#1"))   # an isolated single detection
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "package_000.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False)
        fires, obs, dets, *_ = L.process_package(str(path))
    f = fires.set_index("fire_id")
    a = f.loc[1]
    assert a.status == "tracked" and a.spread_basis == "VIIRS" and a.observations == 2, a
    assert abs(a.spread_direction_deg - 90) < 1, a.spread_direction_deg          # moved east
    # new pixels lie 1.2 / 1.6 / 2.0 km from the old edge, minus half a pixel: median 1.4 km over 100 min
    assert abs(a.spread_rate_km_h - 1.4 / (100 / 60)) < 0.02, a.spread_rate_km_h
    assert a.growth_km2_h > 0 and a.new_front_pixels == 9, (a.growth_km2_h, a.new_front_pixels)
    assert 60 < a.centre_bearing_deg < 120, a.centre_bearing_deg
    assert math.isnan(a.spread_consistency)                                       # only one pass pair
    assert f.loc[2].status == "single detection"
    assert sorted(dets.row_id) == list(range(len(rows))), "every row assigned to exactly one fire"
    assert len(obs[obs.fire_id == 1]) == 2


if __name__ == "__main__":
    test_known_spread()
    print("PASS test_known_spread")
