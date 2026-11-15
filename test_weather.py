"""Offline checks for the weather worker's grid and quota logic (no network).

Run: python test_weather.py
"""
import pandas as pd

import weather_worker as W
from weather_worker import weather_key


def test_grid_points():
    k = weather_key(pd.Series([-8.37, 48.725, 0.12, 10.0, -10.0]), pd.Series([-44.18, 134.4, 179.9, -179.9, 180.0]))
    assert k.tolist() == ["-33:-177", "195:538", "0:-720", "40:-720", "-40:-720"], k.tolist()
    assert k[0] == "-33:-177" and k[1] == "195:538"              # -8.25, -44.25 and 48.75, 134.5
    ll = W.cell_latlon(pd.Series(["-33:-177", "195:538"]))
    assert ll.lat.tolist() == [-8.25, 48.75] and ll.lon.tolist() == [-44.25, 134.5]


def test_dateline_is_one_point():
    k = weather_key(pd.Series([10.0, 10.0]), pd.Series([179.99, -179.99]))
    assert k[0] == k[1], k.tolist()                               # 180 E and 180 W share a grid point


def test_call_estimate():
    # 8 observed and 7 forecast variables: one call each per location, per 14 days
    assert W.estimate_calls(100, 2) == 200
    assert W.estimate_calls(100, 15) == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
