"""Run the harmonisation test plan (test.html) automatically against the running app.

    python server.py          # in one terminal
    python run_test_plan.py   # in another

Each test mirrors one entry of the manual plan. UI-only steps (sorting a column,
copying rows) are checked through the same data the tables display.
"""

import json
import math
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

from firms import fetch_fires

APP = "http://localhost:8000"
AREA_A = "-44.6,-8.7,-43.9,-8.1"
DAY = "2026-10-13"
SAT_NAMES = {"N": "Suomi NPP", "N20": "NOAA-20", "N21": "NOAA-21", "L8": "Landsat 8", "L9": "Landsat 9"}
_cache = {}


def api(path, **params):
    """GET an app endpoint. Returns (status, json). Cached per URL."""
    url = f"{APP}/api/{path}?{urllib.parse.urlencode(params)}"
    if url not in _cache:
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                _cache[url] = (r.status, json.loads(r.read()))
        except urllib.error.HTTPError as e:
            _cache[url] = (e.code, json.loads(e.read()))
    return _cache[url]


def ok(path, **params):
    status, d = api(path, **params)
    assert status == 200, f"{path} returned HTTP {status}: {d.get('error')}"
    return d


def col(d, name):
    return d["columns"].index(name)


def rows_as_dicts(d):
    return [dict(zip(d["columns"], r)) for r in d["rows"]]


def world_harmonised():
    return ok("harmonise", date=DAY, days=1, area="world")


def world_compare(**extra):
    return ok("compare", date=DAY, days=1, area="world", viirs="NOAA20", cell="0.05", **extra)


def summary(d, metric_prefix):
    return next(s for s in d["summary"] if s["metric"].startswith(metric_prefix))


TESTS = []


def test(tid, title):
    def deco(fn):
        TESTS.append((tid, title, fn))
        return fn
    return deco


# ----------------------------------------------------------------- A: pixel level
@test("A1", "Raw scan and track are gone, pixel area is present")
def _():
    raw_cols = {"scan", "track", "scan_km", "track_km"}
    for d in (ok("harmonise", date=DAY, days=1, area=AREA_A), ok("fires", sensor="MODIS_NRT", date=DAY, days=1, area=AREA_A)):
        assert not raw_cols & set(d["columns"]), f"raw columns present: {raw_cols & set(d['columns'])}"
        assert "pixel_area_km2" in d["columns"]


@test("A2", "Pixel area = scan x track")
def _():
    raw = fetch_fires("MODIS_NRT", DAY, 1, AREA_A)
    rc = {c: i for i, c in enumerate(raw["columns"])}
    expected = {(r[rc["latitude"]], r[rc["longitude"]], r[rc["acq_time"]].zfill(4)):
                f"{float(r[rc['scan']]) * float(r[rc['track']]):.4f}" for r in raw["rows"]}
    h = ok("harmonise", date=DAY, days=1, area=AREA_A)
    checked = 0
    for r in rows_as_dicts(h):
        key = (r["latitude"], r["longitude"], r["acq_time"])
        if r["source"] == "MODIS_NRT" and key in expected:
            assert r["pixel_area_km2"] == expected[key], f"{key}: {r['pixel_area_km2']} != {expected[key]}"
            checked += 1
    assert checked == len(expected) > 0, f"matched {checked} of {len(expected)} raw rows"
    return f"{checked} rows match"


@test("A3", "MODIS confidence boundaries 29/30/79/80")
def _():
    want = {"29": "low", "30": "nominal", "79": "nominal", "80": "high"}
    seen = set()
    for r in rows_as_dicts(world_harmonised()):
        if r["instrument"] != "MODIS":
            continue
        v = float(r["confidence_raw"])
        exp = "low" if v < 30 else "high" if v >= 80 else "nominal"
        assert r["confidence"] == exp, f"MODIS conf {v} -> {r['confidence']}, expected {exp}"
        if r["confidence_raw"] in want:
            seen.add(r["confidence_raw"])
    assert seen == set(want), f"boundary values present in data: {sorted(seen)}"


@test("A4", "VIIRS and Landsat confidence letters")
def _():
    m = {"l": "low", "n": "nominal", "m": "nominal", "h": "high"}
    letters = Counter()
    for r in rows_as_dicts(world_harmonised()):
        if r["instrument"] in ("VIIRS", "OLI"):
            assert r["confidence"] == m.get(r["confidence_raw"].lower()), f"{r['confidence_raw']} -> {r['confidence']}"
            letters[(r["instrument"], r["confidence_raw"])] += 1
    return ", ".join(f"{i} {k}: {v:,}" for (i, k), v in sorted(letters.items()))


@test("A5", "Brightness kept at full precision (no cap, no saturation flag)")
def _():
    d = world_harmonised()
    assert "bright_saturated" not in d["columns"], "bright_saturated column still present"
    rows = [r for r in rows_as_dicts(d) if r["bright_main_k"]]
    top = {i: max(float(r["bright_main_k"]) for r in rows if r["instrument"] == i) for i in ("MODIS", "VIIRS")}
    assert top["MODIS"] > 367.0, f"MODIS max {top['MODIS']} - still capped?"
    # every raw brightness value comes through unchanged, including those above 367 K
    raw = fetch_fires("MODIS_NRT", DAY, 1, AREA_A)
    rc = {c: i for i, c in enumerate(raw["columns"])}
    h = {(r["latitude"], r["longitude"], r["acq_time"]): r for r in rows_as_dicts(ok("harmonise", date=DAY, days=1, area=AREA_A)) if r["instrument"] == "MODIS"}
    hot = 0
    for r in raw["rows"]:
        k = (r[rc["latitude"]], r[rc["longitude"]], r[rc["acq_time"]].zfill(4))
        assert h[k]["bright_main_k"] == r[rc["brightness"]], f"{k}: {h[k]['bright_main_k']} != {r[rc['brightness']]}"
        hot += float(r[rc["brightness"]]) >= 367
    return f"max MODIS {top['MODIS']:.2f} K, VIIRS {top['VIIRS']:.2f} K; {len(raw['rows'])} raw values identical ({hot} above 367 K)"


@test("A6", "Landsat rows have no brightness, FRP or area")
def _():
    ls = [r for r in rows_as_dicts(world_harmonised()) if r["source"] == "LANDSAT_NRT"]
    assert ls, "no Landsat rows"
    for r in ls:
        assert r["instrument"] == "OLI"
        assert not (r["bright_main_k"] or r["bright_secondary_k"] or r["frp_mw"] or r["pixel_area_km2"])
    return f"{len(ls):,} Landsat rows"


@test("A7", "Satellite names")
def _():
    names = Counter(r["satellite"] for r in rows_as_dicts(world_harmonised()))
    allowed = set(SAT_NAMES.values()) | {"Terra", "Aqua"}
    assert set(names) <= allowed, f"unexpected names: {set(names) - allowed}"
    return ", ".join(sorted(names))


@test("A8", "Acquisition time always has 4 digits")
def _():
    bad = [r["acq_time"] for r in rows_as_dicts(world_harmonised()) if len(r["acq_time"]) != 4 or not r["acq_time"].isdigit()]
    assert not bad, f"{len(bad)} bad times, e.g. {bad[:3]}"


@test("A9", "Overpass slot from local solar time")
def _():
    n = 0
    for r in rows_as_dicts(world_harmonised()):
        h = int(r["acq_time"][:2]) + int(r["acq_time"][2:]) / 60
        local = (h + float(r["longitude"]) / 15) % 24
        exp = "night" if local < 6 else "morning" if local < 12 else "afternoon" if local < 18 else "evening"
        assert r["overpass_slot"] == exp, f"{r['acq_time']} {r['longitude']}: {r['overpass_slot']} != {exp}"
        n += 1
    return f"{n:,} rows recomputed"


@test("A10", "NRT rows overlapping SP data are dropped")
def _():
    d = ok("harmonise", date="2026-07-30", days=3, area="20,40,30,50")
    notes = [s for s in d["sources"] if s["status"] == "note" and s["sensor"] == "VIIRS_NOAA21_NRT"]
    assert notes and "overlapping VIIRS_NOAA21_SP" in notes[0]["note"], "no overlap note"
    early = [r for r in rows_as_dicts(d) if r["source"] == "VIIRS_NOAA21_NRT" and r["acq_date"] <= "2026-07-31"]
    assert not early, f"{len(early)} NRT rows still overlap"
    return notes[0]["note"]


# ----------------------------------------------------------------- B: type
@test("B1", "SP files keep their own type")
def _():
    d = ok("fires", sensor="MODIS_SP", date="2026-08-30", days=1, area="world")
    rs = rows_as_dicts(d)
    assert all(r["type_existed"] == "yes" for r in rs)
    counts = Counter(r["type"] for r in rs)
    assert counts == Counter({"0": 14178, "1": 68, "2": 666, "3": 129}), f"type counts {dict(counts)}"
    return f"{len(rs):,} rows, {dict(sorted(counts.items()))}"


@test("B2", "NRT rows get an inferred type")
def _():
    rs = rows_as_dicts(ok("fires", sensor="MODIS_NRT", date=DAY, days=1, area="world"))
    assert len(rs) == 11168, f"{len(rs)} rows"
    for r in rs:
        assert r["type_existed"] == "no" and r["type_basis"]
        assert 0 <= float(r["type_confidence"]) <= 1
    return f"0 / {len(rs):,} from file / inferred"


@test("B3", "A known volcano is recognised (Semeru)")
def _():
    rs = rows_as_dicts(ok("fires", sensor="MODIS_NRT", date=DAY, days=1, area="world"))
    r = next((r for r in rs if r["latitude"] == "-8.11577" and r["longitude"] == "112.93331"), None)
    assert r, "row not found"
    assert r["type"] == "1" and r["type_label"] == "active volcano", f"type {r['type']} {r['type_label']}"


@test("B4", "Repeating a query does not double-count the reference")
def _():
    before = ok("sensors")["type_reference"]["labelled_detections"]
    _cache.clear()
    ok("fires", sensor="MODIS_SP", date="2026-08-30", days=1, area="world")
    _cache.pop(next((k for k in _cache if "/api/sensors" in k), ""), None)
    after = ok("sensors")["type_reference"]["labelled_detections"]
    assert before == after, f"{before:,} -> {after:,}"
    return f"{after:,} labelled detections before and after"


@test("B5", "Mixed files: kept and inferred side by side")
def _():
    rs = rows_as_dicts(ok("harmonise", date="2026-07-30", days=3, area="20,40,30,50"))
    for r in rs:
        exp = "yes" if r["source"].endswith("_SP") else "no"
        assert r["type_existed"] == exp, f"{r['source']} type_existed={r['type_existed']}"
    c = Counter(r["type_existed"] for r in rs)
    assert c["yes"] and c["no"]
    return f"yes {c['yes']:,} / no {c['no']:,}"


# ----------------------------------------------------------------- C: comparison
def compare_a():
    return ok("compare", date=DAY, days=1, area=AREA_A, viirs="NOAA20", cell="0.05")


@test("C1", "Known answer for test area A")
def _():
    d = compare_a()
    s = {x["metric"]: (x["modis"], x["viirs"]) for x in d["summary"]}
    exp = {"Raw pixels (detections)": (49, 156), "Raw total FRP (MW)": (6286, 12258),
           "Fire cells (0.05° grid)": (15, 10), "Comparable fire cells": (15, 10)}
    for k, v in exp.items():
        assert s[k] == v, f"{k}: {s[k]} != {v}"
    assert d["agreement"] == {"both": 8, "MODIS only": 7, "VIIRS only": 2}, d["agreement"]
    assert d["settings"]["min_frp_mw"] == 10.0
    assert not any("capped" in k for k in d["removed"]), d["removed"]
    top = dict(zip(d["columns"], next(r for r in d["rows"] if r[0] == "-8.3750" and r[1] == "-44.1750" and r[3] == "D")))
    assert (top["modis_pixels"], top["modis_frp_mw"], top["viirs_pixels"], top["viirs_frp_mw"], top["overpass_gap_min"]) == \
        ("5", "3329.12", "21", "3486.85", "103"), top


@test("C2", "Cell sums match an independent recomputation")
def _():
    h = ok("harmonise", date=DAY, days=1, area=AREA_A)
    cell, grid = 0.05, defaultdict(lambda: {"MODIS": [0, 0.0], "VIIRS": [0, 0.0]})
    for r in rows_as_dicts(h):
        if r["instrument"] == "MODIS" and r["satellite"] != "Aqua":
            continue
        if r["instrument"] == "VIIRS" and r["satellite"] != "NOAA-20":
            continue
        if r["instrument"] not in ("MODIS", "VIIRS"):
            continue
        k = (f"{(math.floor(float(r['latitude']) / cell) + 0.5) * cell:.4f}",
             f"{(math.floor(float(r['longitude']) / cell) + 0.5) * cell:.4f}", r["acq_date"], r["daynight"])
        grid[k][r["instrument"]][0] += 1
        grid[k][r["instrument"]][1] += float(r["frp_mw"] or 0)
    app = {(r["cell_lat"], r["cell_lon"], r["acq_date"], r["daynight"]): r for r in rows_as_dicts(compare_a())}
    assert set(app) == set(grid), f"cells differ: app {len(app)}, recomputed {len(grid)}"
    for k, g in grid.items():
        a = app[k]
        m, v = g["MODIS"], g["VIIRS"]
        assert int(a["modis_pixels"]) == m[0] and int(a["viirs_pixels"]) == v[0], k
        assert (a["modis_frp_mw"] or "") == (f"{m[1]:.2f}" if m[0] else ""), k
        assert (a["viirs_frp_mw"] or "") == (f"{v[1]:.2f}" if v[0] else ""), k
        det = "both" if m[0] and v[0] else "MODIS only" if m[0] else "VIIRS only"
        assert a["detected_by"] == det, k
    return f"{len(grid)} cells identical"


@test("C3", "Terra is excluded")
def _():
    d = world_compare()
    terra = d["removed"]["Terra MODIS pixels (no matching VIIRS overpass)"]
    aqua = summary(d, "Raw pixels")["modis"]
    assert (terra, aqua) == (4161, 7007), f"Terra {terra}, Aqua {aqua}"


@test("C4", "Day and night are never mixed in one cell")
def _():
    rs = rows_as_dicts(world_compare())
    keys = Counter((r["cell_lat"], r["cell_lon"], r["acq_date"], r["daynight"]) for r in rs)
    assert all(r["daynight"] in ("D", "N") for r in rs)
    dup = [k for k, n in keys.items() if n > 1]
    assert not dup, f"duplicate cells: {dup[:3]}"
    both_dn = len({k[:3] for k in keys if k[3] == "D"} & {k[:3] for k in keys if k[3] == "N"})
    return f"{len(rs):,} cells, {both_dn:,} squares appear once for day and once for night"


@test("C5", "Detection limit applied to both sensors")
def _():
    d = world_compare()
    lim = d["settings"]["min_frp_mw"]
    f = lambda x: float(x or 0)
    for r in rows_as_dicts(d):
        if r["reason"].startswith("below common detection limit"):
            assert f(r["modis_frp_mw"]) < lim and f(r["viirs_frp_mw"]) < lim, r
        if r["comparable"] == "yes":
            assert max(f(r["modis_frp_mw"]), f(r["viirs_frp_mw"])) >= lim, r
    return f"limit {lim} MW"


@test("C6", "Swath exclusions are plausible")
def _():
    out = [r for r in rows_as_dicts(world_compare()) if r["reason"].startswith("outside Aqua")]
    assert out, "no swath exclusions"
    for r in out:
        assert r["modis_pixels"] == "0" and int(r["viirs_pixels"]) > 0, r
    low = sum(abs(float(r["cell_lat"])) <= 35 for r in out) / len(out)
    assert low >= 0.9, f"only {low:.0%} within ±35°"
    return f"{len(out):,} cells, {low:.1%} within ±35° latitude"


@test("C7", "Max pass gap filter")
def _():
    d = world_compare(maxgap=120)
    over = [r for r in rows_as_dicts(d) if r["comparable"] == "yes" and r["overpass_gap_min"] and float(r["overpass_gap_min"]) > 120]
    assert not over, f"{len(over)} comparable cells over 120 min"
    removed = d["removed"]["Cells with passes too far apart in time"]
    assert removed > 0
    return f"{removed:,} cells removed"


@test("C8", "Cell brightness is uncapped")
def _():
    top = {"modis_bt_max_k": 0.0, "viirs_bt_max_k": 0.0}
    for r in rows_as_dicts(world_compare()):
        for k in top:
            if r[k]:
                assert "(sat)" not in r[k], r[k]
                top[k] = max(top[k], float(r[k]))
    assert top["modis_bt_max_k"] > 367.0, f"MODIS cell max {top['modis_bt_max_k']} - still capped?"
    return f"highest cell brightness: MODIS {top['modis_bt_max_k']:.2f} K, VIIRS {top['viirs_bt_max_k']:.2f} K"


@test("C9", "Summary adds up")
def _():
    for d in (world_compare(), compare_a()):
        a, s = d["agreement"], summary(d, "Comparable fire cells")
        assert s["modis"] == a["both"] + a["MODIS only"], (s, a)
        assert s["viirs"] == a["both"] + a["VIIRS only"], (s, a)
        rs = rows_as_dicts(d)
        assert sum(r["comparable"] == "yes" for r in rs) == a["both"] + a["MODIS only"] + a["VIIRS only"]


@test("C10", "Harmonising narrows the gap")
def _():
    d = world_compare()
    raw, cells, frp = (float(summary(d, m)["ratio"]) for m in ("Raw pixels", "Comparable fire cells", "Comparable total FRP"))
    assert 7.5 <= raw <= 9.5, f"raw ratio {raw}"
    assert 1.5 <= cells <= 2.3 and 1.5 <= frp <= 2.3, f"comparable ratios {cells}, {frp}"
    assert cells < raw / 2
    return f"raw {raw}x -> cells {cells}x, FRP {frp}x"


@test("C11", "Raw counts match the single-sensor data")
def _():
    n = len(ok("fires", sensor="VIIRS_NOAA20_NRT", date=DAY, days=1, area="world")["rows"])
    v = summary(world_compare(), "Raw pixels")["viirs"]
    assert n == v == 58943, f"single fetch {n}, compare {v}"


# ----------------------------------------------------------------- D: input checks
def expect_error(path, fragment, **params):
    status, d = api(path, **params)
    assert status == 400 and fragment in d.get("error", ""), f"HTTP {status}: {d}"


@test("D1", "Day range is limited to 1-5")
def _():
    expect_error("fires", "Day range must be 1-5", sensor="MODIS_NRT", date=DAY, days=9)


@test("D2", "Invalid grid size is refused")
def _():
    expect_error("compare", "Grid must be one of", date=DAY, days=1, cell="0.3")


@test("D3", "Non-numeric max pass gap is refused")
def _():
    expect_error("compare", "Max overpass gap must be a number", date=DAY, days=1, maxgap="abc")


@test("D4", "Malformed area is refused")
def _():
    expect_error("fires", "Area must be", sensor="MODIS_NRT", date=DAY, days=1, area="abc")


@test("D5", "Dates outside a sensor's range")
def _():
    d = ok("fires", sensor="MODIS_SP", date=DAY, days=1, area="world")
    assert len(d["rows"]) == 0, f"{len(d['rows'])} rows"
    h = ok("harmonise", date=DAY, days=1, area=AREA_A)
    skipped = {s["sensor"] for s in h["sources"] if s["status"] == "skipped"}
    assert {"MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"} <= skipped, skipped
    assert all("available" in s["note"] for s in h["sources"] if s["status"] == "skipped")


@test("D6", "An area with no fires")
def _():
    ocean = "-150,-40,-149,-39"
    for path, extra in (("fires", {"sensor": "MODIS_NRT"}), ("harmonise", {}), ("compare", {})):
        d = ok(path, date=DAY, days=1, area=ocean, **extra)
        assert d["rows"] == [], f"{path}: {len(d['rows'])} rows"


def main():
    try:
        ok("sensors")
    except Exception as exc:
        sys.exit(f"Cannot reach the app at {APP} ({exc}). Start it with: python server.py")
    passed = 0
    results = []
    for tid, title, fn in TESTS:
        try:
            info = fn()
            passed += 1
            results.append((tid, "PASS", title, info or ""))
        except AssertionError as exc:
            results.append((tid, "FAIL", title, str(exc)[:300]))
        except Exception as exc:
            results.append((tid, "ERROR", title, f"{type(exc).__name__}: {exc}"[:300]))
        tid_, st, t, info = results[-1]
        print(f"{tid_:<4} {st:<5} {t}" + (f"  [{info}]" if info else ""), flush=True)
    pct = 100 * passed / len(TESTS)
    print(f"\n{passed} / {len(TESTS)} passed = {pct:.1f}%")
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    main()
