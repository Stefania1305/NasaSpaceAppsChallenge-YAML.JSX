"""Offline checks for correlation.py on small synthetic data (no network).

Run: python test_correlation.py
"""
import numpy as np
import pandas as pd

import correlation as C

COLS = ["source", "instrument", "satellite", "latitude", "longitude", "acq_date", "acq_time", "overpass_slot",
        "daynight", "confidence", "confidence_raw", "bright_main_k", "bright_secondary_k", "frp_mw",
        "pixel_area_km2", "scan_km", "track_km", "type", "type_label", "type_existed", "type_confidence", "type_basis"]


def row(lat, lon, t="1340", sat="NOAA-20", inst="VIIRS", frp="5.0", bt="330", bt2="290", scan="0.4", track="0.4",
        conf="n", day="2026-10-13", typ="0"):
    from comparison import overpass_slot
    from server import harmonise_confidence
    area = f"{float(scan) * float(track):.4f}"
    return ["VIIRS_NOAA20_NRT" if inst == "VIIRS" else "MODIS_NRT", inst, sat, str(lat), str(lon), day, t,
            overpass_slot(t, str(lon)), "D", harmonise_confidence(inst, conf), conf, bt, bt2, frp, area, scan, track,
            typ, C.TYPE_LABELS[int(typ)], "no", "0.900", "test"]


def test_measures():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    df = pd.DataFrame({"x": x, "y": x ** 3, "z": rng.normal(size=500)})
    r = df.rank().corr()
    assert abs(r.loc["x", "y"] - 1) < 1e-9, "Spearman of a monotonic function must be 1"
    assert abs(r.loc["x", "z"]) < 0.15
    cat = pd.Series(np.where(x > 0, "a", "b"))
    eta, n = C._eta(cat, pd.Series(x).rank())
    assert eta > 0.85 and n == 500, eta
    v, _ = C._cramers_v(cat, cat.copy())
    assert v > 0.99, v
    v, _ = C._cramers_v(cat, pd.Series(rng.choice(["p", "q"], 500)))
    assert v < 0.1, v
    assert C.strength(0.75) == "strong" and C.strength(-0.5) == "moderate" and C.strength(0.05) == "negligible"


def test_links_and_patches():
    # three touching VIIRS pixels in one pass, one far away, one same spot 100 min later, one MODIS overlap
    rows = [row(10.0, 20.0), row(10.0, 20.0036), row(10.0036, 20.0036),   # ~0.4 km apart: touching
            row(11.0, 21.0),                                               # far: no links
            row(10.0, 20.0, t="1520"),                                     # same satellite, later orbit
            row(10.001, 20.001, t="1545", sat="Aqua", inst="MODIS", conf="85", scan="1.0", track="1.0")]
    res = C.run_correlation(COLS, rows * 1)
    links, df = res["links"], res["frame"]
    kinds = {(int(a), int(b)): k for a, b, k in zip(links.i, links.j, links.kind)}
    assert kinds[(0, 1)] == "same_pass" and kinds[(1, 2)] == "same_pass", kinds
    assert not any(3 in pair for pair in kinds), "far pixel must not be linked"
    assert kinds[(0, 4)] == "repeat_pass", kinds
    assert kinds[(0, 5)] == "other_satellite", kinds
    p = df["patch_id"].to_numpy()
    assert p[0] == p[1] == p[2] and len({p[0], p[3], p[4], p[5]}) == 4, p
    assert df.loc[3, "link_count"] == 0 and df.loc[0, "link_count"] >= 4


def test_exact_and_construction():
    rows = [row(10 + i * 0.1, 20, frp=str(1 + i), bt=str(320 + i)) for i in range(40)]
    res = C.run_correlation(COLS, rows)
    ex = {e["relation"]: e["share"] for e in res["report"]["exact"]}
    assert ex["pixel_area_km2 = scan_km x track_km"] == 1.0
    assert ex["confidence = rule(confidence_raw, instrument)"] == 1.0
    assert ex["overpass_slot = slot(acq_time + longitude / 15)"] == 1.0
    rel = {(r["a"], r["b"]): r for r in res["report"]["relations"] if r["scope"] == "all"}
    assert rel[("bright_main_k", "frp_mw")]["value"] == 1.0
    assert rel[("bright_main_k", "bright_diff_k")]["by_construction"] is True
    top = C.ranking(res["report"], "frp_mw")
    assert top[0]["other"] in ("bright_main_k", "latitude", "bright_diff_k", "frp_physics_proxy"), top[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
