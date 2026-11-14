"""Small local server for browsing NASA FIRMS fire data.

Reads FIRMS_MAP_KEY from .env, serves index.html, and proxies requests to the
FIRMS API so the key never reaches the browser. Every table gets a `type`
column: kept from the file when it exists (SP data), otherwise inferred from
the reference built by build_type_reference.py (see type_inference.py).

Run:  python server.py   then open http://localhost:8000
"""

import json
import mimetypes
import re
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from comparison import clip_brightness, compare, overpass_slot
from firms import BASE_DIR, MAP_KEY, PORT, SENSORS, fetch_availability, fetch_fires

VIIRS_CHOICES = {"NOAA20": "NOAA-20", "NOAA21": "NOAA-21", "SNPP": "Suomi NPP"}
GRID_CHOICES = {0.01, 0.05, 0.1, 0.25}
from type_inference import TypeReference, add_type_columns
from prediction import predict
import threading
import time
from collections import OrderedDict

TYPE_REF = TypeReference.load()
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
# Windows' registry can map .js to text/plain, which browsers refuse for module scripts.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

# Small in-memory cache so /api/predict can reuse the data the user just loaded
# instead of fetching it from FIRMS again (FIRMS calls can take a minute).
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
CACHE_SIZE = 12
CACHE_TTL_S = 1800   # near-real-time data keeps arriving, so don't hold results too long


def cached(key: tuple, compute):
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
            _CACHE.move_to_end(key)
            return hit[1]
    value = compute()
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), value)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return value


def fires_payload(sensor: str, day: str, days: int, area: str) -> dict:
    data = fetch_fires(sensor, day, days, area)
    learn_types(sensor, data)
    data = replace_scan_track(sensor, data)
    columns, rows = add_type_columns(TYPE_REF, data["columns"], data["rows"])
    return {"columns": columns, "rows": rows, "type_reference": TYPE_REF.status()}


def learn_types(sensor: str, data: dict) -> None:
    """Feed SP rows (which carry `type`) back into the reference so it keeps improving."""
    cols = data["columns"]
    if "type" not in cols:
        return
    li, lo, ti, di = (cols.index(c) for c in ("latitude", "longitude", "type", "acq_date"))
    if TYPE_REF.learn(sensor, ((r[li], r[lo], r[ti], r[di]) for r in data["rows"])):
        TYPE_REF.save()


# ---------------------------------------------------------------------------
# Harmonisation: fetch every sensor and map it onto one common schema.
# ---------------------------------------------------------------------------

HARMONISED_COLUMNS = [
    "source", "instrument", "satellite", "latitude", "longitude",
    "acq_date", "acq_time", "overpass_slot", "daynight", "confidence", "confidence_raw",
    "bright_main_k", "bright_saturated", "bright_secondary_k", "frp_mw", "pixel_area_km2",
    "type",  # replaced by the full set of type columns in add_type_columns()
]


def pixel_area(scan: str, track: str) -> str:
    """Pixel_Area = scan * track, in km². Empty if either value is missing."""
    try:
        return f"{float(scan) * float(track):.4f}"
    except (TypeError, ValueError):
        return ""


def replace_scan_track(sensor: str, data: dict) -> dict:
    """Swap the raw scan/track columns for a single pixel_area_km2 column.
    Landsat's scan/track are not pixel sizes in km, so its area is left empty."""
    cols = data["columns"]
    if "scan" not in cols or "track" not in cols:
        return data
    si, ti = cols.index("scan"), cols.index("track")
    landsat = sensor.startswith("LANDSAT")
    new_cols = [("pixel_area_km2" if k == si else c) for k, c in enumerate(cols) if k != ti]
    new_rows = []
    for r in data["rows"]:
        area = "" if landsat else pixel_area(r[si], r[ti])
        new_rows.append([(area if k == si else v) for k, v in enumerate(r) if k != ti])
    return {**data, "columns": new_cols, "rows": new_rows}

SATELLITE_NAMES = {"N": "Suomi NPP", "N20": "NOAA-20", "N21": "NOAA-21",
                   "L8": "Landsat 8", "L9": "Landsat 9"}


def harmonise_confidence(instrument: str, raw: str) -> str:
    """Map each sensor's confidence onto low / nominal / high."""
    raw = raw.strip()
    if instrument == "MODIS":
        try:
            v = float(raw)
        except ValueError:
            return ""
        return "low" if v < 30 else "high" if v >= 80 else "nominal"
    return {"l": "low", "n": "nominal", "m": "nominal", "h": "high"}.get(raw.lower(), "")


def harmonise_rows(sensor: str, data: dict) -> list:
    out = []
    for values in data["rows"]:
        r = dict(zip(data["columns"], values))
        if sensor.startswith("MODIS"):
            instrument, b_main, b_sec = "MODIS", r.get("brightness", ""), r.get("bright_t31", "")
        elif sensor.startswith("VIIRS"):
            instrument, b_main, b_sec = "VIIRS", r.get("bright_ti4", ""), r.get("bright_ti5", "")
        else:  # Landsat: no brightness/FRP; scan/track are not pixel sizes in km, so no area
            instrument, b_main, b_sec = "OLI", "", ""
        landsat = instrument == "OLI"
        sat = r.get("satellite", "")
        b_main, saturated = clip_brightness(b_main)  # common 367 K ceiling (VIIRS I4 limit)
        out.append([
            sensor, instrument, SATELLITE_NAMES.get(sat, sat),
            r.get("latitude", ""), r.get("longitude", ""),
            r.get("acq_date", ""), r.get("acq_time", "").zfill(4),
            overpass_slot(r.get("acq_time", ""), r.get("longitude", "")),
            r.get("daynight", ""),
            harmonise_confidence(instrument, r.get("confidence", "")), r.get("confidence", ""),
            b_main, saturated, b_sec, r.get("frp", ""),
            "" if landsat else pixel_area(r.get("scan"), r.get("track")),
            r.get("type", ""),
        ])
    return out


def fetch_harmonised(day: str, days: int, area: str, sensors: list = SENSORS) -> dict:
    start = date.fromisoformat(day)
    end = start + timedelta(days=days - 1)  # FIRMS ranges run forward from `date`
    try:
        avail = fetch_availability()
    except Exception:
        avail = {}

    sources, to_fetch = [], []
    for sensor in sensors:
        a = avail.get(sensor)
        if a and (end < date.fromisoformat(a["min"]) or start > date.fromisoformat(a["max"])):
            sources.append({"sensor": sensor, "status": "skipped", "count": 0,
                            "note": f"no data for this range (available {a['min']} to {a['max']})"})
        else:
            to_fetch.append(sensor)

    results = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_fires, s, day, days, area): s for s in to_fetch}
        for fut, sensor in futures.items():
            try:
                data = fut.result()
                learn_types(sensor, data)
                results[sensor] = harmonise_rows(sensor, data)
            except Exception as exc:
                sources.append({"sensor": sensor, "status": "error", "count": 0, "note": str(exc)[:200]})

    # Where a satellite has both NRT and SP data for the same day, keep only the
    # SP (final, reprocessed) detections so the same fire is not counted twice.
    date_idx = HARMONISED_COLUMNS.index("acq_date")
    for sensor, rows in list(results.items()):
        if not sensor.endswith("_NRT"):
            continue
        sp = sensor.replace("_NRT", "_SP")
        if sp in results and sp in avail:
            sp_max = avail[sp]["max"]
            before = len(rows)
            results[sensor] = [r for r in rows if r[date_idx] > sp_max]
            if before != len(results[sensor]):
                sources.append({"sensor": sensor, "status": "note", "count": 0,
                                "note": f"dropped {before - len(results[sensor])} rows overlapping {sp}"})

    rows = []
    for sensor in SENSORS:
        if sensor in results:
            rows.extend(results[sensor])
            sources.append({"sensor": sensor, "status": "ok", "count": len(results[sensor]), "note": ""})
    rows.sort(key=lambda r: (r[date_idx], r[date_idx + 1]))
    order = {s: i for i, s in enumerate(SENSORS)}
    sources.sort(key=lambda s: (order.get(s["sensor"], 99), s["status"] != "ok"))
    columns, rows = add_type_columns(TYPE_REF, HARMONISED_COLUMNS, rows)
    return {"columns": columns, "rows": rows, "sources": sources,
            "range": [start.isoformat(), end.isoformat()], "type_reference": TYPE_REF.status()}


def validate(q: dict):
    """Return (day, days, area) or raise ValueError with a user-facing message."""
    day = q.get("date", "")
    area = q.get("area", "world") or "world"
    try:
        days = int(q.get("days", "1"))
    except ValueError:
        days = 0
    if not 1 <= days <= 5:
        raise ValueError("Day range must be 1-5.")
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError("Date must be YYYY-MM-DD.")
    if area != "world" and not re.fullmatch(r"-?[\d.]+(,-?[\d.]+){3}", area):
        raise ValueError("Area must be 'world' or west,south,east,north.")
    return day, days, area


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, no_cache: bool = False) -> None:
        """Serve a static file from the React build. Hashed assets can be cached; index.html never is,
        so a rebuild is picked up on the next reload."""
        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if no_cache else "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}

        # The React app (frontend/, built with `npm run build`) lives at /app. The original single-file
        # version stays available at /app-old, and /app falls back to it if the build is missing.
        if url.path in ("/app", "/app/", "/app/index.html") and (FRONTEND_DIST / "index.html").exists():
            return self.send_file(FRONTEND_DIST / "index.html", no_cache=True)
        if url.path.startswith("/app/"):
            rel = url.path[len("/app/"):]
            target = (FRONTEND_DIST / rel).resolve()
            if FRONTEND_DIST.resolve() in target.parents and target.is_file():
                return self.send_file(target)
            return self.send_json(404, {"error": "Not found"})

        if url.path in ("/", "/index.html", "/test", "/test.html", "/app", "/app.html", "/app-old"):
            page = ("test.html" if url.path.startswith("/test")
                    else "app.html" if url.path.startswith("/app") else "index.html")
            body = (BASE_DIR / page).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if url.path == "/api/sensors":
            try:
                avail = fetch_availability()
            except Exception:
                avail = {}
            self.send_json(200, {"sensors": SENSORS, "availability": avail,
                                 "type_reference": TYPE_REF.status()})
            return

        if url.path == "/api/harmonise":
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            try:
                self.send_json(200, cached(("harmonise", day, days, area),
                                           lambda: fetch_harmonised(day, days, area)))
            except Exception as exc:
                self.send_json(502, {"error": f"Harmonisation failed: {exc}"})
            return

        if url.path == "/api/predict":
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            source = q.get("source", "harmonise")
            sensor = q.get("sensor", "")
            if source not in ("harmonise", "fetch"):
                return self.send_json(400, {"error": "source must be 'harmonise' or 'fetch'."})
            if source == "fetch" and sensor not in SENSORS:
                return self.send_json(400, {"error": "Unknown sensor."})
            bbox = (-180.0, -90.0, 180.0, 90.0) if area == "world" else tuple(float(x) for x in area.split(","))
            try:
                if source == "harmonise":
                    data = cached(("harmonise", day, days, area), lambda: fetch_harmonised(day, days, area))
                else:
                    data = cached(("fires", sensor, day, days, area), lambda: fires_payload(sensor, day, days, area))
            except Exception as exc:
                return self.send_json(502, {"error": f"Could not get the input data: {exc}"})
            try:
                result = predict(data["columns"], data["rows"], day, days, bbox)
                result["source"] = "harmonised sensors" if source == "harmonise" else sensor
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(500, {"error": f"Prediction failed: {exc}"})
            return

        if url.path == "/api/compare":
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            viirs = q.get("viirs", "NOAA20")
            if viirs not in VIIRS_CHOICES:
                return self.send_json(400, {"error": "Unknown VIIRS satellite."})
            try:
                cell = float(q.get("cell", "0.05"))
            except ValueError:
                cell = 0
            if cell not in GRID_CHOICES:
                return self.send_json(400, {"error": f"Grid must be one of {sorted(GRID_CHOICES)}."})
            gap_raw = q.get("maxgap", "")
            try:
                max_gap = float(gap_raw) if gap_raw else None
            except ValueError:
                return self.send_json(400, {"error": "Max overpass gap must be a number of minutes."})
            sensors = ["MODIS_NRT", "MODIS_SP", f"VIIRS_{viirs}_NRT", f"VIIRS_{viirs}_SP"]
            try:
                h = fetch_harmonised(day, days, area, sensors)
                result = compare(h["columns"], h["rows"], cell, VIIRS_CHOICES[viirs], max_gap)
                result.update(sources=h["sources"], range=h["range"])
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(502, {"error": f"Comparison failed: {exc}"})
            return

        if url.path == "/api/fires":
            sensor = q.get("sensor", "")
            if sensor not in SENSORS:
                return self.send_json(400, {"error": "Unknown sensor."})
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            try:
                self.send_json(200, cached(("fires", sensor, day, days, area),
                                           lambda: fires_payload(sensor, day, days, area)))
            except (urllib.error.URLError, ValueError) as exc:
                self.send_json(502, {"error": f"FIRMS request failed: {exc}"})
            return

        self.send_json(404, {"error": "Not found"})

    def log_message(self, fmt, *args):  # quieter console
        print("%s - %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    if not MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY is missing. Add it to .env")
    st = TYPE_REF.status()
    if st["labelled_detections"]:
        print(f"Type reference: {st['labelled_detections']:,} labelled detections "
              f"from {st['sensor_days']} sensor-days")
    else:
        print("No type reference yet - run: python build_type_reference.py")
    print(f"Serving on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
