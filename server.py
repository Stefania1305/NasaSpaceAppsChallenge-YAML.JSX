"""Small local server for browsing NASA FIRMS fire data.

Reads FIRMS_MAP_KEY from .env, serves index.html, and proxies requests to the
FIRMS API so the key never reaches the browser. Every table gets a `type`
column: kept from the file when it exists (SP data), otherwise inferred from
the reference built by build_type_reference.py (see type_inference.py).

Run:  python server.py   then open http://localhost:8000
"""

import json
import re
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import csv
import mimetypes
import gzip
import threading

from comparison import compare, overpass_slot
from firms import BASE_DIR, DATA_DIR, MAP_KEY, PORT, SENSORS, fetch_availability, fetch_fires
from weather import fetch_weather

VIIRS_CHOICES = {"NOAA20": "NOAA-20", "NOAA21": "NOAA-21", "SNPP": "Suomi NPP"}
GRID_CHOICES = {0.01, 0.05, 0.1, 0.25}
for _ext, _type in ((".geojson", "application/geo+json"), (".woff2", "font/woff2"), (".woff", "font/woff")):
    mimetypes.add_type(_type, _ext)
from type_inference import TypeReference, add_type_columns

TYPE_REF = TypeReference.load()


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
    "bright_main_k", "bright_secondary_k", "frp_mw", "pixel_area_km2",
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


def harmonise_rows(sensor: str, data: dict, keep_scan_track: bool = False) -> list:
    """Map one sensor's rows onto HARMONISED_COLUMNS. With keep_scan_track, scan_km and
    track_km are added before `type` (for the pipeline workers; not shown on the page)."""
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
        out.append([
            sensor, instrument, SATELLITE_NAMES.get(sat, sat),
            r.get("latitude", ""), r.get("longitude", ""),
            r.get("acq_date", ""), r.get("acq_time", "").zfill(4),
            overpass_slot(r.get("acq_time", ""), r.get("longitude", "")),
            r.get("daynight", ""),
            harmonise_confidence(instrument, r.get("confidence", "")), r.get("confidence", ""),
            b_main, b_sec, r.get("frp", ""),
            "" if landsat else pixel_area(r.get("scan"), r.get("track")),
            *(("" if landsat else r.get("scan", ""), "" if landsat else r.get("track", "")) if keep_scan_track else ()),
            r.get("type", ""),
        ])
    return out


def fetch_harmonised(day: str, days: int, area: str, sensors: list = SENSORS,
                     keep_scan_track: bool = False) -> dict:
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
                results[sensor] = harmonise_rows(sensor, data, keep_scan_track)
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
    base = HARMONISED_COLUMNS[:-1] + ["scan_km", "track_km", "type"] if keep_scan_track else HARMONISED_COLUMNS
    columns, rows = add_type_columns(TYPE_REF, base, rows)
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


# ---------------------------------------------------------------------------
# Pipeline data for the app: the fire detections as the pipeline's first worker
# (correlation.py) prepares them - harmonised across sensors, with its cell, pass
# and patch groupings. A run already on disk (data/runs/<run>/correlation/) is
# reused; otherwise the stage runs for that zone and saves its run like the CLI.
# ---------------------------------------------------------------------------

_PIPELINE_LOCK = threading.Lock()


def pipeline_detections(day: str, days: int, area: str, sensor: str = "") -> dict:
    from correlation import run_correlation, save  # pipeline stage 1

    slug = f"{day}_{days}d_" + re.sub(r"[^\w.-]+", "_", area) + (f"_{sensor}" if sensor else "")
    out = DATA_DIR / "runs" / slug / "correlation"
    rows_file, report_file = out / "rows.csv.gz", out / "correlations.json"
    reused = True
    with _PIPELINE_LOCK:
        if not rows_file.exists():
            reused = False
            h = fetch_harmonised(day, days, area, [sensor] if sensor else SENSORS, keep_scan_track=True)
            if not h["rows"]:  # nothing burning in the zone: no run to save
                return {"columns": h["columns"], "rows": [], "sources": h["sources"], "range": h["range"],
                        "run": slug, "reused": False}
            result = run_correlation(h["columns"], h["rows"])
            result["report"]["run"] = {"date": day, "days": days, "area": area,
                                       "sources": h["sources"], "range": h["range"]}
            save(result, out)
    with gzip.open(rows_file, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)
        rows = list(reader)
    run = json.loads(report_file.read_text(encoding="utf-8")).get("run", {})
    return {"columns": columns, "rows": rows, "sources": run.get("sources", []), "range": run.get("range"),
            "run": slug, "reused": reused}


# The pipeline's stages after correlation, in order. Each worker runs as its own process on the run folder,
# exactly as run_pipeline.py starts it; the app calls them one by one (localisation and fire power side by side).
PIPELINE_STAGES = {  # stage -> (script, output folder)
    "interpreter": ("interpreter.py", "interpreter"),
    "localisation": ("localisation_worker.py", "localisation"),
    "fire_power": ("fire_power_worker.py", "fire_power"),
    "epicenter": ("epicenter_worker.py", "epicenter"),
    "expansion": ("fire_expansion_worker.py", "expansion"),
    "kg": ("kg_builder.py", "kg"),                       # knowledge graph, fire part: what the model reads
    "prediction": ("prediction_engine.py", "prediction"),  # the trained model (models/v2), see predict_stage
}


def run_folder(day: str, days: int, area: str, sensor: str = ""):
    slug = f"{day}_{days}d_" + re.sub(r"[^\w.-]+", "_", area) + (f"_{sensor}" if sensor else "")
    return slug, DATA_DIR / "runs" / slug


def _csv_count(path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def prediction_weather(run) -> list:
    """Hourly forecast (our weather API, Open-Meteo) for the fires active on the run's last day, from 3 days before
    it to 3 days after, saved beside the run for prediction_engine.py. Returns its command-line arguments, or none
    (no-weather models) when the forecast can't be had."""
    import pandas as pd
    from weather import fetch_model_forecast, model_points

    try:
        fires = pd.read_csv(run / "epicenter" / "fires.csv.gz")
        last_day = fires.last_seen.str[:10]
        last = last_day.max()
        active = fires[last_day == last]
        pts = model_points(list(zip(active.latitude, active.longitude)))
        d = date.fromisoformat(last)
        rows = fetch_model_forecast(pts, d - timedelta(days=3), d + timedelta(days=3))
        if not rows:
            return []
        path = run / "weather_forecast.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return ["--weather", str(path), "--weather-label", "Open-Meteo hourly forecast (our weather API)"]
    except Exception as exc:  # no forecast: the engine falls back to its no-weather models
        print(f"Weather forecast for the prediction failed: {exc}")
        return []


def real_days(day: str, days: int) -> tuple[list[str], list[str]]:
    """The 3 days after the window, split into those with real data (complete days: before today, UTC, and
    published by FIRMS near-real-time) and those still to come (today and later), which are predicted."""
    from datetime import datetime, timezone
    end = date.fromisoformat(day) + timedelta(days=days - 1)
    after = [(end + timedelta(days=k)).isoformat() for k in (1, 2, 3)]
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        nrt_max = max(v["max"] for k, v in fetch_availability().items() if k.endswith("_NRT"))
    except Exception:
        nrt_max = today
    real = [d for d in after if d < today and d <= nrt_max]
    return real, [d for d in after if d not in real]


_STAGE_LOCKS: dict = {}
_STAGE_LOCKS_GUARD = threading.Lock()


def pipeline_stage(stage: str, day: str, days: int, area: str, sensor: str = "", predict_all: bool = False) -> dict:
    slug, _ = run_folder(day, days, area, sensor)
    with _STAGE_LOCKS_GUARD:
        lock = _STAGE_LOCKS.setdefault((slug, stage), threading.Lock())
    with lock:
        return _pipeline_stage(stage, day, days, area, sensor, predict_all)


def _pipeline_stage(stage: str, day: str, days: int, area: str, sensor: str = "", predict_all: bool = False) -> dict:
    """Run one stage on the zone's run folder (or reuse its saved output) and describe what it produced."""
    import shutil
    import subprocess
    import sys
    import time as _time

    slug, run = run_folder(day, days, area, sensor)
    t0 = _time.time()
    if stage == "correlation":
        before = (run / "correlation" / "rows.csv.gz").exists()
        d = pipeline_detections(day, days, area, sensor)  # runs correlation.py's stage if needed
        if not before:  # fresh input: the later stages must run again on it
            for _, folder in PIPELINE_STAGES.values():
                shutil.rmtree(run / folder, ignore_errors=True)
        ok_sources = sum(1 for s in d.get("sources", []) if s.get("status") == "ok")
        return {"stage": stage, "run": slug, "reused": before, "seconds": round(_time.time() - t0, 1),
                "empty": not d["rows"], "detail": f"{len(d['rows']):,} detections from {ok_sources} sensor feeds"}

    if stage not in PIPELINE_STAGES:
        raise ValueError("Unknown stage.")
    if stage == "prediction" and not predict_all:
        real, future = real_days(day, days)
        if not future:      # nothing to predict: the real data of all 3 days is shown instead
            return {"stage": stage, "run": slug, "reused": False, "skipped": True, "seconds": 0.0,
                    "detail": f"not needed: real data exists for {real[0]} to {real[-1]}, shown instead"}
    script, folder = PIPELINE_STAGES[stage]
    out = run / folder
    marker = out / {"interpreter": "plan.json", "kg": "nodes_Fire.csv.gz"}.get(stage, "summary.json")
    reused = marker.exists()
    args = ["--run", str(run)]
    if not reused and stage == "prediction":
        args += prediction_weather(run)          # our weather API's hourly forecast for the active fires
    if not reused:
        # Started like run_pipeline.py does. One retry: on Windows a worker's process pool can occasionally break
        # while its processes start up (BrokenProcessPool); a second start goes through.
        # Each worker would start one process per CPU thread, every one loading pandas: on a 24-thread machine that is
        # more memory than is usually free, and Windows then kills processes mid-run (BrokenProcessPool). The worker
        # scripts are run unchanged as __main__, through a small wrapper that caps the CPU count they see at 4.
        launcher = ("import os, runpy, sys; os.cpu_count = lambda: 4; sys.argv = sys.argv[1:]; "
                    "runpy.run_path(sys.argv[0], run_name='__main__')")
        for _attempt in range(2):
            proc = subprocess.run([sys.executable, "-X", "utf8", "-c", launcher, script, *args],
                                  cwd=BASE_DIR, capture_output=True, text=True, timeout=1800)
            if proc.returncode == 0 and marker.exists():
                break
        if proc.returncode != 0 or not marker.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"{script} failed: {tail[0]}")
    s = json.loads(marker.read_text(encoding="utf-8")) if marker.suffix == ".json" else {}
    if stage == "kg":
        detail = f"{_csv_count(out / 'nodes_Fire.csv.gz'):,} fires, {_csv_count(out / 'nodes_FireState.csv.gz'):,} fire states"
    elif stage == "prediction":
        detail = (f"{s.get('active_fires', 0):,} active fires × 3 days, {s.get('spread_cells', 0):,} spread cells; "
                  + ("with the weather forecast" if s.get("weather") == "provided" else "no weather (forecast unavailable)"))
    elif stage == "interpreter":
        w = s.get("workers", {})
        detail = (f"{s.get('rows_in', 0):,} rows → {s.get('units', 0):,} fire units; "
                  f"localisation {w.get('localisation', {}).get('rows', 0):,} rows, "
                  f"fire power {w.get('fire_power', {}).get('rows', 0):,} rows")
    elif stage == "localisation":
        detail = f"{_csv_count(out / 'fires.csv.gz'):,} fires located, {_csv_count(out / 'observations.csv.gz'):,} passes"
    elif stage == "fire_power":
        detail = f"{_csv_count(out / 'fires.csv.gz'):,} fires measured, {_csv_count(out / 'observations.csv.gz'):,} passes"
    elif stage == "epicenter":
        detail = (f"{s.get('epicenters', 0):,} epicenters, {s.get('fires', 0):,} fires"
                  + (f" ({s['merged_fires']} merged)" if s.get("merged_fires") else ""))
    else:
        ev = s.get("events", {})
        detail = f"{s.get('fires', 0):,} fires followed; " + ", ".join(f"{v} {k}" for k, v in ev.items())
    return {"stage": stage, "run": slug, "reused": reused, "seconds": round(_time.time() - t0, 1), "detail": detail}


def pipeline_results(day: str, days: int, area: str, sensor: str = "", predict_all: bool = False) -> dict:
    """What the workers produced for the zone: the correlated detections (for the map's fire dots), the fires
    after the epicenter merge with their expansion, the epicenters and the expansion events."""
    import pandas as pd

    slug, run = run_folder(day, days, area, sensor)
    if not (run / "expansion" / "summary.json").exists():
        raise ValueError("The pipeline has not finished for this zone.")

    def records(df):
        return json.loads(df.to_json(orient="records"))

    det = pipeline_detections(day, days, area, sensor)
    fires = pd.read_csv(run / "epicenter" / "fires.csv.gz")
    exp = pd.read_csv(run / "expansion" / "fires.csv.gz")
    exp = exp.rename(columns={"status": "expansion_status", "last_seen": "expansion_last_seen"})
    fires = fires.merge(exp, on="fire_id", how="left")
    power = pd.read_csv(run / "fire_power" / "fires.csv.gz")[["fire_id", "peak_class", "frp_trend"]]
    fires = fires.merge(power, on="fire_id", how="left")   # fire-power ids that survived the epicenter merge
    fires = fires.sort_values("peak_frp_mw", ascending=False, na_position="last")
    epis = pd.read_csv(run / "epicenter" / "epicenters.csv.gz")[
        ["fire_id", "epicenter_no", "latitude", "longitude", "uncertainty_km", "origin", "latest_start",
         "first_satellite", "start_frp_mw"]]
    events = pd.read_csv(run / "expansion" / "events.csv.gz")[["fire_id", "event", "latitude", "longitude", "time"]]
    # each fire's history pass by pass (fire expansion worker), for the fire card's day-by-day view
    hist = pd.read_csv(run / "expansion" / "timeline.csv.gz")[
        ["fire_id", "time", "pixels", "burning_area_km2", "frp_sum_mw", "cumulative_footprint_km2", "events"]]
    summaries = {name: json.loads((run / name / ("plan.json" if name == "interpreter" else "summary.json"))
                                  .read_text(encoding="utf-8"))
                 for name in ("interpreter", "epicenter", "expansion")}
    # The model's output (prediction stage): per fire and day ahead, and the burn chance of 1 km cells around them.
    pred_dir = run / "prediction"
    predictions, spread, pred_summary = [], [], None
    if (pred_dir / "summary.json").exists():
        pred_summary = json.loads((pred_dir / "summary.json").read_text(encoding="utf-8"))
        predictions = records(pd.read_csv(pred_dir / "predictions.csv"))
        if (pred_dir / "spread.csv").exists():
            spread = records(pd.read_csv(pred_dir / "spread.csv")[["fire_id", "day_ahead", "lat", "lon", "p"]])
    # Days after the window that already have real data: their real detections are shown, not the prediction.
    real, future = real_days(day, days) if not predict_all else ([], [])
    actual = {"columns": det["columns"], "rows": [], "days": real}
    if real:
        h = fetch_harmonised(real[0], len(real), area, [sensor] if sensor else SENSORS)
        actual = {"columns": h["columns"], "rows": h["rows"], "days": real}
        predictions = [r for r in predictions if r["date"] not in real]
        keep_k = {r["day_ahead"] for r in predictions}
        spread = [c for c in spread if c["day_ahead"] in keep_k]
    return {
        "predictions": predictions, "spread": spread, "prediction": pred_summary, "actual": actual,
        "run": slug, "detections": {"columns": det["columns"], "rows": det["rows"]},
        "sources": det["sources"], "range": det["range"],
        "fires": records(fires), "epicenters": records(epis), "events": records(events), "history": records(hist),
        "summary": {
            "detections": len(det["rows"]),
            "fire_units": summaries["interpreter"].get("units"),
            "fires": summaries["epicenter"].get("fires"),
            "epicenters": summaries["epicenter"].get("epicenters"),
            "merged_fires": summaries["epicenter"].get("merged_fires"),
            "events": summaries["expansion"].get("events", {}),
            "status": summaries["expansion"].get("status", {}),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}

        # The built React app (frontend/dist), served under /app/ (vite base). /app-old is the original page.
        if url.path == "/app":
            self.send_response(301)
            self.send_header("Location", "/app/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if url.path.startswith("/app/"):
            dist = (BASE_DIR / "frontend" / "dist").resolve()
            rel = unquote(url.path[len("/app/"):]) or "index.html"
            target = (dist / rel).resolve()
            if not dist.is_dir():
                self.send_json(404, {"error": "The app is not built yet: run `npm run build` in the frontend folder."})
                return
            if dist not in target.parents or not target.is_file():
                self.send_json(404, {"error": "not found"})
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            # hashed bundles never change; the page itself is always re-checked; the shapes may be cached for a while
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if rel.startswith("assets/")
                             else "no-cache" if rel == "index.html" else "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if url.path in ("/", "/index.html", "/test", "/test.html", "/app-old"):
            page = "test.html" if url.path.startswith("/test") else "index.html"
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
                self.send_json(200, fetch_harmonised(day, days, area))
            except Exception as exc:
                self.send_json(502, {"error": f"Harmonisation failed: {exc}"})
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

        # Fire detections from the pipeline (see pipeline_detections): ?date&days&area[&sensor=<one sensor>]
        if url.path == "/api/pipeline/harmonised":
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            sensor = q.get("sensor", "")
            if sensor and sensor not in SENSORS:
                return self.send_json(400, {"error": "Unknown sensor."})
            try:
                self.send_json(200, pipeline_detections(day, days, area, sensor))
            except Exception as exc:
                self.send_json(502, {"error": f"Pipeline failed: {exc}"})
            return

        # Before running the pipeline: which of the 3 days after the window already have real data (see real_days).
        # When all of them do, the app skips the pipeline and shows the real data instead.
        if url.path == "/api/pipeline/plan":
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            real, future = real_days(day, days)
            end = date.fromisoformat(day) + timedelta(days=days - 1)
            return self.send_json(200, {"window": [day, end.isoformat()], "real_days": real, "future_days": future,
                                        "run_pipeline": bool(future)})

        # One pipeline stage for the zone (see pipeline_stage): ?stage=correlation|interpreter|localisation|
        # fire_power|epicenter|expansion&date&days&area[&sensor]. Then ?.../results gathers what they produced.
        if url.path in ("/api/pipeline/stage", "/api/pipeline/results"):
            try:
                day, days, area = validate(q)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            sensor = q.get("sensor", "")
            if sensor and sensor not in SENSORS:
                return self.send_json(400, {"error": "Unknown sensor."})
            try:
                if url.path.endswith("/results"):
                    self.send_json(200, pipeline_results(day, days, area, sensor, q.get("predict_all") == "1"))
                else:
                    self.send_json(200, pipeline_stage(q.get("stage", ""), day, days, area, sensor, q.get("predict_all") == "1"))
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(502, {"error": f"Pipeline failed: {exc}"})
            return

        # Fire-weather for the zone on the forecast days (Open-Meteo, see weather.py):
        # ?area=w,s,e,n&start=YYYY-MM-DD&days=1..7
        if url.path == "/api/weather":
            try:
                area = q.get("area", "")
                if not re.fullmatch(r"-?[\d.]+(,-?[\d.]+){3}", area):
                    raise ValueError("Area must be west,south,east,north.")
                w, s_, e, n = (float(x) for x in area.split(","))
                start = date.fromisoformat(q.get("start", ""))
                ndays = int(q.get("days", "3"))
                if not (1 <= ndays <= 10):
                    raise ValueError("days must be 1 to 10.")
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            try:
                self.send_json(200, fetch_weather(w, s_, e, n, start, start + timedelta(days=ndays - 1)))
            except urllib.error.HTTPError as exc:
                self.send_json(502, {"error": f"The weather service refused the request ({exc.code})."})
            except Exception as exc:
                self.send_json(502, {"error": f"Could not get the weather: {exc}"})
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
                data = fetch_fires(sensor, day, days, area)
                learn_types(sensor, data)
                data = replace_scan_track(sensor, data)
                columns, rows = add_type_columns(TYPE_REF, data["columns"], data["rows"])
                self.send_json(200, {"columns": columns, "rows": rows,
                                     "type_reference": TYPE_REF.status()})
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
