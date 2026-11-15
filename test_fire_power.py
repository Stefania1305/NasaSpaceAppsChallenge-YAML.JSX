"""Offline checks for fire_power_worker.py with synthetic fires of known power (no network).

Run: python test_fire_power.py
"""
import math
import tempfile
from pathlib import Path

import pandas as pd

import fire_power_worker as F


def pixel(row_id, unit, t, frp, inst="VIIRS", sat="NOAA-20", pass_id="p", dn="D", area=0.16, bt=340.0, bt2=295.0):
    return {"latitude": 10.0, "longitude": 20.0, "acq_date": "2026-10-13", "acq_time": f"{t // 60:02d}{t % 60:02d}",
            "acq_minutes": t, "pixel_area_km2": area, "daynight": dn, "instrument": inst, "satellite": sat,
            "confidence": "high", "frp_mw": frp, "bright_main_k": bt, "bright_secondary_k": bt2, "type": "0",
            "pass_id": pass_id, "unit_id": unit, "row_id": row_id}


def test_known_power():
    rows = []
    # fire 1: three VIIRS passes 60 min apart, two pixels each, FRP sum 100 -> 200 -> 400 MW (doubles every hour)
    for k, (t, total) in enumerate([(600, 100), (660, 200), (720, 400)]):
        rows += [pixel(len(rows), 1, t, total / 2, pass_id=f"v{k}"), pixel(len(rows) + 1, 1, t, total / 2, pass_id=f"v{k}")]
    # a night pass over fire 1 (MODIS, so it does not enter the VIIRS trend)
    rows.append(pixel(len(rows), 1, 1300, 50, inst="MODIS", sat="Aqua", pass_id="m0", dn="N", area=1.0))
    # fire 2: a single MODIS pixel
    rows.append(pixel(len(rows), 2, 800, 8, inst="MODIS", sat="Terra", pass_id="m1", area=2.0))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "package_000.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False)
        fits = F.fit_physics([path])                      # too few rows to fit: anomalies stay empty
        fires, obs, det, *_ = F.process_package((str(path), fits))
    f = fires.set_index("fire_id")
    a = f.loc[1]
    assert a.trend_basis == "VIIRS" and a.trend_passes == 3, a
    assert abs(a.frp_trend_pct_h - 100) < 0.01 and a.frp_trend == "rising", a.frp_trend_pct_h   # doubles per hour
    assert abs(a.frp_last_first_ratio - 4) < 1e-9
    assert a.trend_quality == "few pixels"            # passes of 2 pixels each
    assert abs(a.fre_mj - (150 * 3600 + 300 * 3600)) < 1, a.fre_mj         # trapezoid over two hours
    assert abs(a.biomass_burned_t - a.fre_mj * 0.368 / 1000) < 0.01
    assert a.peak_frp_mw == 400 and a.peak_class == "large"
    assert abs(a.day_night_frp_ratio - (700 / 3) / 50) < 1e-3, a.day_night_frp_ratio
    assert abs(a.mean_frp_density_mw_km2 - 750 / (6 * 0.16 + 1.0)) < 1e-3
    b = f.loc[2]
    assert b.peak_class == "small" and pd.isna(b.frp_trend_pct_h) and (pd.isna(b.frp_trend) or b.frp_trend == "")
    assert math.isnan(b.fre_mj)
    assert sorted(det.row_id) == list(range(len(rows)))
    assert abs(det.set_index("row_id").loc[len(rows) - 1, "frp_density_mw_km2"] - 4.0) < 1e-9


def test_short_span_gives_no_trend():
    rows = [pixel(0, 1, 600, 10, pass_id="a"), pixel(1, 1, 620, 30, pass_id="b")]   # 20 min apart
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "package_000.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False)
        fires, *_ = F.process_package((str(path), {}))
    a = fires.iloc[0]
    assert math.isnan(a.frp_trend_pct_h) and abs(a.frp_last_first_ratio - 3) < 1e-9
    assert abs(a.fre_mj - 20 * 20 * 60) < 1e-6, a.fre_mj   # energy still integrated over the 20 min


def test_partial_view_is_left_out():
    rows = [pixel(0, 1, 600, 12, pass_id="a")]                                   # 1 pixel: edge of swath
    rows += [pixel(len(rows) + k, 1, 620, 10, pass_id="b") for k in range(8)]    # 20 min later: 8 pixels
    rows += [pixel(len(rows) + k, 1, 700, 20, pass_id="c") for k in range(8)]    # 100 min after the first
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "package_000.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False)
        fires, obs, *_ = F.process_package((str(path), {}))
    a = fires.iloc[0]
    assert a.partial_view_passes == 1 and a.trend_passes == 2, a
    assert abs(a.frp_last_first_ratio - 2) < 1e-9                               # 160 / 80, not 160 / 12
    assert obs.set_index("pass_id").loc["a", "partial_view"] == "yes"


if __name__ == "__main__":
    test_partial_view_is_left_out()
    print("PASS test_partial_view_is_left_out")
    test_known_power()
    print("PASS test_known_power")
    test_short_span_gives_no_trend()
    print("PASS test_short_span_gives_no_trend")
