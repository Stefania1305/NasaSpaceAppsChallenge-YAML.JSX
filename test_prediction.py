"""Offline checks for the prediction engine's weather and feature logic (no network, no trained models).

Run: python test_prediction.py
"""
import numpy as np
import pandas as pd

import prediction_engine as E
from prediction_data import H, WX, add_place_and_weather, daily_weather, feature_columns
from prediction_train import CONT


def hours(cell, start, n, rain_at=()):
    t = pd.date_range(start, periods=n, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M")
    h = pd.DataFrame({"weather_cell": cell, "time": t, "temperature_2m": 30.0, "relative_humidity_2m": 20.0,
                      "wind_speed_10m": 10.0, "wind_direction_10m": 270.0, "wind_gusts_10m": 25.0, "precipitation": 0.0,
                      "vapour_pressure_deficit": 3.0, "soil_moisture_0_to_7cm": 0.1, "soil_moisture_7_to_28cm": 0.15,
                      "soil_moisture_28_to_100cm": 0.2})
    h.loc[list(rain_at), "precipitation"] = 1.0
    return h


def test_since_rain():
    h = pd.concat([hours("1:1", "2025-07-01", 48, rain_at=[5]), hours("2:2", "2025-07-01", 48)])
    s = E.since_rain(h)
    a = s[s.weather_cell == "1:1"].hours_since_rain.to_numpy()
    assert np.isnan(a[:5]).all() and a[5] == 0 and a[6] == 1 and a[47] == 42, a[:8]
    assert s[s.weather_cell == "2:2"].hours_since_rain.isna().all()        # no rain seen: unknown, not zero


def test_daily_weather_wind_components():
    d = daily_weather(E.since_rain(hours("1:1", "2025-07-01", 24)))
    assert len(d) == 1
    r = d.iloc[0]
    assert abs(r.u_mean - 10.0) < 1e-6 and abs(r.v_mean) < 1e-6           # wind from the west blows east
    assert r.tmax == 30 and r.rain == 0 and abs(r.sm2 - 0.2) < 1e-9


def test_weather_features_line_up_by_day():
    h = pd.concat([hours("1:1", "2025-07-01", 24 * 5)])
    h.loc[h.time.str.startswith("2025-07-03"), "temperature_2m"] = 40.0     # day d+2
    wd = daily_weather(E.since_rain(h))
    cells = pd.DataFrame({"weather_cell": ["1:1"], "lat": [0.25], "lon": [0.25], "elevation_m": [100.0]})
    t = pd.DataFrame({"run": ["r"], "fire_id": ["f"], "day": ["2025-07-01"], "lat0": [0.26], "lon0": [0.24]})
    X = add_place_and_weather(t, None, wd, cells, None)
    assert X.w2_tmax.iloc[0] == 40 and X.w1_tmax.iloc[0] == 30 and X.w0_tmax.iloc[0] == 30
    assert X.elevation_m.iloc[0] == 100
    feats = feature_columns(X, weather=True)
    assert all(f"w{k}_{c}" in feats for k in range(H + 1) for c in WX)
    assert not any(f.startswith("w") and f[1].isdigit() for f in feature_columns(X, weather=False))


def test_transforms_invert():
    x = np.array([-50.0, -1.0, 0.0, 0.5, 3.0, 400.0])
    for name, (fwd, inv) in CONT.items():
        xx = x if name in ("dx", "dy") else np.abs(x)
        assert np.allclose(inv(fwd(xx)), xx), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
