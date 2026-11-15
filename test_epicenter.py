"""Offline checks for epicenter_worker.py geometry and data-proof logic (no network).

Run: python test_epicenter.py
"""
import math

import numpy as np
import pandas as pd

import epicenter_worker as E

KM_LON = E.KM_PER_DEG_LON          # at the equator
KM_LAT = E.KM_PER_DEG_LAT


def early(points, size=0.4, frp=5.0):
    """points: list of (x_km, y_km) around (0, 0)."""
    return pd.DataFrame({"latitude": [y / KM_LAT for _, y in points], "longitude": [x / KM_LON for x, _ in points],
                         "pixel_km": size, "area": size * size, "frp_mw": frp}, index=range(len(points)))


def test_single_pixel():
    ep = E.epicenters_for_fire(pd.Series({"spread_direction_deg": 90.0}), early([(0, 0)]))
    assert len(ep) == 1 and ep[0]["backstep_km"] == 0           # one pixel: nothing to step back along
    assert abs(ep[0]["uncertainty_km"] - 0.2 * math.sqrt(2)) < 1e-3   # half pixel diagonal


def test_step_back_against_spread():
    # a 5-pixel line from x = 0 to 1.6 km, spreading east (90 deg): origin lies to the west of the centre
    pts = [(k * 0.4, 0) for k in range(5)]
    ep = E.epicenters_for_fire(pd.Series({"spread_direction_deg": 90.0}), early(pts))[0]
    centre_x = 0.8
    x = ep["lon"] * KM_LON
    # back edge = -0.8 - 0.2 = -1.0 km from the centre, so the step is 0.5 km west
    assert abs(ep["backstep_km"] - 0.5) < 1e-3 and abs(x - (centre_x - 0.5)) < 0.01, (ep, x)
    no_dir = E.epicenters_for_fire(pd.Series({"spread_direction_deg": float("nan")}), early(pts))[0]
    assert no_dir["backstep_km"] == 0 and abs(no_dir["lon"] * KM_LON - centre_x) < 0.01


def test_separate_origins():
    pts = [(0, 0), (0.4, 0), (10, 10), (10.4, 10)]                  # two groups 14 km apart
    ep = E.epicenters_for_fire(pd.Series({"spread_direction_deg": float("nan")}), early(pts))
    assert len(ep) == 2 and sorted(e["first_pixels"] for e in ep) == [2, 2]


def test_small_gaps_are_one_origin():
    pts = [(0, 0), (0.4, 0), (2.5, 0), (2.9, 0)]                    # 2 km gap: smoke or faint pixels, one fire
    ep = E.epicenters_for_fire(pd.Series({"spread_direction_deg": float("nan")}), early(pts))
    assert len(ep) == 1 and ep[0]["first_pixels"] == 4
    far = [(0, 0), (10, 10)]                                        # unknown origin: never split
    assert len(E.epicenters_for_fire(pd.Series({"spread_direction_deg": float("nan")}), early(far), split=False)) == 1


def test_weighting_prefers_hot_fine_pixels():
    df = early([(0, 0), (0.4, 0)])
    df.loc[1, "frp_mw"] = 95.0                                      # second pixel ~19x the weight
    ep = E.epicenters_for_fire(pd.Series({"spread_direction_deg": float("nan")}), df)[0]
    assert ep["lon"] * KM_LON > 0.35


def test_neighbouring_fires_merge():
    # VIIRS pixels 0.4 km wide; fire 2 is one pixel-gap away from fire 1 (centres 0.8 km apart),
    # fire 3 touches fire 2, fire 4 is three pixel-gaps from fire 3 (centres 1.6 km apart)
    xs = {1: [0.0, 0.4], 2: [1.2], 3: [1.6], 4: [3.2]}
    rows = [{"fire_id": f, "latitude": 0.0, "longitude": x / KM_LON, "pixel_km": 0.4, "instrument": "VIIRS",
             "t_abs": 1000.0 + 60 * f} for f, pts in xs.items() for x in pts]
    m = E.neighbour_map(pd.DataFrame(rows))
    assert m[1] == m[2] == m[3] == 1, m.to_dict()        # chain 1-2-3 becomes fire 1
    assert m[4] == 4, m.to_dict()                         # too far: stays its own fire


def test_data_proof():
    times = np.array([100.0, 200.0, 300.0])
    got = E.has_data(times, np.array([102.0, 150.0, 297.5, 400.0]))
    assert got.tolist() == [True, False, True, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
