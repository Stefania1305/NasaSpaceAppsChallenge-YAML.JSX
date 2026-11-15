"""Offline checks for fire_expansion_worker.follow_fire with scripted fires (no network).

Run: python test_expansion.py
"""
import pandas as pd

import fire_expansion_worker as X

KM_LON, KM_LAT = 111.320, 110.574            # at the equator


def px(row_id, x_km, y_km, t, pass_id, source, inst="VIIRS", size=0.4):
    return {"row_id": row_id, "latitude": y_km / KM_LAT, "longitude": x_km / KM_LON, "t_abs": t, "pass_id": pass_id,
            "instrument": inst, "satellite": "NOAA-20" if inst == "VIIRS" else "Aqua", "source_fire_id": source,
            "size": size, "scan": size, "track": size, "area": size * size, "frp_mw": 5.0}


def run(pixels, epis=((0, 0),)):
    g = pd.DataFrame(pixels)
    e = pd.DataFrame({"latitude": [y / KM_LAT for _, y in epis], "longitude": [x / KM_LON for x, _ in epis]})
    return X.follow_fire(1, g, e)


def test_two_bursts_merge_and_keep_history():
    p = [px(0, 0, 0, 600, "a", 10), px(1, 3, 0, 600, "a", 20)]                     # two separate hot spots
    p += [px(2, 0, 0, 700, "b", 10)] + [px(3 + k, 0.4 * (k + 1), 0, 700, "b", 10) for k in range(7)]   # bridge
    tracks, steps, events, assign, leaves, firsts = run(p)
    kinds = [e["event"] for e in events]
    assert kinds == ["ignition", "new burst", "merge"], kinds
    child = tracks[events[-1]["clusters"].split(" -> ")[1]]
    assert child["parents"] == "1.1+1.2" and child["sources"] == {10, 20}, child
    assert child["first_seen_history"] == 600 and child["born"] == 700
    assert steps[0]["clusters_seen"] == 2 and steps[1]["clusters_alive"] == 1
    assert firsts == {10: "1.1", 20: "1.2"}


def test_modis_does_not_merge():
    p = [px(0, 0, 0, 600, "a", 10), px(1, 2.0, 0, 600, "a", 20)]
    p += [px(2, 1.0, 0, 700, "m", 10, inst="MODIS", size=1.5)]                     # one coarse pixel over both
    tracks, steps, events, *_ = run(p)
    assert [e["event"] for e in events] == ["ignition", "new burst"], events
    assert tracks["1.1"]["last_seen"] == 700 and tracks["1.2"]["last_seen"] == 700  # both continue
    assert tracks["1.1"]["families"] == {"VIIRS", "MODIS"}


def test_split():
    p = [px(k, 0.4 * k, 0, 600, "a", 10) for k in range(6)]                         # one line of 6 pixels
    p += [px(6, 0, 0, 700, "b", 10), px(7, 2.0, 0, 700, "b", 10)]                   # only both ends remain
    tracks, steps, events, *_ = run(p)
    assert [e["event"] for e in events] == ["ignition", "split"], events
    assert tracks["1.1"]["end_how"] == "split" and tracks["1.1"]["children"] == "1.2+1.3"


def test_cumulative_footprint():
    p = [px(0, 0, 0, 600, "a", 10), px(1, 0, 0, 700, "b", 10)]                     # same square seen twice
    _, steps, *_ = run(p)
    assert abs(steps[0]["cumulative_footprint_km2"] - 0.25) < 0.06                  # 0.4 km square on a 100 m raster
    assert steps[1]["new_footprint_km2"] == 0 and steps[1]["cumulative_footprint_km2"] == steps[0]["cumulative_footprint_km2"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
