"""Weather worker: stage 5 of the prediction pipeline
(... epicenter -> fire expansion -> weather -> knowledge graph -> prediction).

Inputs: only the two weather APIs and the processed outputs of the earlier
workers (correlation rows, localisation, fire power, epicenter, expansion).
It does not read or write the knowledge graph; kg_builder.py reads this
worker's output afterwards.

1. Weather (per 0.25 degree ERA5 / CEMS grid point and hour)
   Open-Meteo (no key; CC BY 4.0; free tier non-commercial)
   - observed: ERA5 (archive API)
   - forecast: ECMWF IFS 0.25 issued the day before (previous-runs API), kept
     apart as fc_*: prediction must train on what was known beforehand
   CEMS Fire Danger Indices (Copernicus Early Warning Data Store; EWDS_API_KEY
   in .env): daily FWI, FFMC, DMC, DC, ISI, BUI. Skipped without a key.
   Derived per hour: wind as east/north components, rain in the last 24 h,
   hours since rain, and a weather condition class (thresholds below).

2. Weather attached to the fire data
   - epicenters: summary over the start window (unknown starts: the 6 h
     before the first detection), and at the first detection
   - localisation observations (one source fire, one pass): weather at the
     pass time, interpolated between hours; how well the spread direction
     follows the wind (cos of the angle to the downwind direction)
   - expansion steps (one merged fire, one pass): weather at that pass
   - merged fires: summary over all their passes

3. Fire behaviour correlated with weather conditions (weather/correlations.json)
   - spread rate and growth vs temperature, humidity, VPD, wind, recent rain,
     fire danger (tracked fires, Spearman), observed and forecast
   - does fire spread with the wind? alignment by wind-speed class
   - FRP trend vs weather and vs local solar time (daily cycle)
   - FRP per pass (VIIRS) vs weather at the pass
   - footprint growth per hour vs weather (expansion, VIIRS passes)
   - extinction vs rain and humidity after the last sighting
   - ignition: each start window vs the same cell and hours one day earlier
     (paired: share of starts that were hotter / drier / windier than the day
     before), and vs all hours in the fire cells (AUC; includes the daily
     cycle, since start windows end at daytime passes)
   - everything above grouped by weather condition class

Usage:  python weather_worker.py --date 2026-10-13 --days 2 --area -48,-12,-42,-6
Output: <run>/weather/hours.csv.gz, fire_danger.csv.gz, epicenters.csv.gz, observations.csv.gz,
        steps.csv.gz, fires.csv.gz, correlations.json, summary.json
"""

import argparse
import json
import math
import os
import re
import time
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from firms import DATA_DIR

WEATHER_CELL_DEG = 0.25
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OBSERVED = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "precipitation", "vapour_pressure_deficit", "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]   # deeper layers: weeks-to-months of dryness (instead of CEMS)
FORECAST = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "precipitation", "vapour_pressure_deficit"]          # soil moisture has no forecast in this archive
BATCH = 50
CALLS_PER_MIN = 400
CALLS_PER_HOUR = 4500            # public limit 5,000 per hour
FORECAST_FROM = "2024-03-01"     # first date the day-before forecast archive has values
MAJOR_CELLS, MEDIUM_CELLS = 3, 1 # per-fire mode: weather cells per picked fire (most detections first)
CACHE = DATA_DIR / "weather_cache"
USER_AGENT = "fire-pipeline/0.1"
EWDS_URL = "https://ewds.climate.copernicus.eu/api"
CEMS_DATASET = "cems-fire-historical-v1"
CEMS_VARIABLES = {"fire_weather_index": "fwi", "fine_fuel_moisture_code": "ffmc", "duff_moisture_code": "dmc",
                  "drought_code": "dc", "initial_fire_spread_index": "isi", "build_up_index": "bui"}
CEMS_SHORT = {"fwinx": "fwi", "ffmcode": "ffmc", "dufmcode": "dmc", "drtcode": "dc", "infsinx": "isi", "buinx": "bui"}
RAIN_MM = 0.2                     # an hour with at least this much rain counts as rain
MAX_WINDOW_H = 24
UNKNOWN_START_CONTEXT_H = 6
CONDITIONS = """rain: precipitation >= 0.2 mm in the hour
hot, dry and windy: humidity < 30 %, wind >= 20 km/h, temperature >= 30 C
dry: humidity < 30 % (otherwise)
humid: humidity >= 60 %
moderate: everything else"""
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "vapour_pressure_deficit", "wind_speed_10m",
                "wind_gusts_10m", "rain_24h_mm", "hours_since_rain"]
MIN_N = 20


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def weather_key(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Nearest 0.25 degree ERA5 / CEMS grid point, as '<lat_index>:<lon_index>' (index x 0.25 = degrees)."""
    lon = (lon + 180) % 360 - 180
    i = np.round(lat / WEATHER_CELL_DEG).astype("Int64")
    j = np.round(lon / WEATHER_CELL_DEG).astype("Int64")
    j = j.where(j < 720, j - 1440)
    return i.astype(str) + ":" + j.astype(str)


def cell_latlon(cells: pd.Series) -> pd.DataFrame:
    ij = cells.str.split(":", expand=True).astype(int)
    return pd.DataFrame({"weather_cell": cells.to_numpy(), "lat": ij[0].to_numpy() * WEATHER_CELL_DEG,
                         "lon": ij[1].to_numpy() * WEATHER_CELL_DEG})


# ---------------------------------------------------------------------------
# Open-Meteo and CEMS
# ---------------------------------------------------------------------------

def estimate_calls(n_cells: int, n_days: int) -> int:
    per = lambda nvars: math.ceil(nvars / 10) * math.ceil(n_days / 14)
    return n_cells * (per(len(OBSERVED)) + per(len(FORECAST)))


class Throttle:
    """Stays under Open-Meteo's per-minute and per-hour limits (shared by every fetch in this process)."""
    def __init__(self, per_min: int, per_hour: int = CALLS_PER_HOUR):
        self.per_min, self.per_hour, self.sent = per_min, per_hour, []

    def wait(self, calls: int):
        while True:
            now = time.time()
            self.sent = [(t, c) for t, c in self.sent if now - t < 3600]
            minute = sum(c for t, c in self.sent if now - t < 60)
            hour = sum(c for _, c in self.sent)
            if minute + calls <= self.per_min and hour + calls <= self.per_hour:
                break
            time.sleep(2)
        self.sent.append((time.time(), calls))


THROTTLE = Throttle(CALLS_PER_MIN)


def cache_file(cdir: Path, cell: str, new: bool = False) -> Path:
    """Cache file of one weather cell. Named with '_' (a name like '6:-202' reads as drive '6:' on Windows);
    files cached earlier as '<lat>:<lon>.json' are still found."""
    safe = cdir / (cell.replace(":", "_") + ".json")
    if new or safe.exists():
        return safe
    old = cdir / f"{cell}.json"
    return old if len(cell.split(":")[0]) > 1 and old.exists() else safe


def fetch_open_meteo(kind: str, cells: pd.DataFrame, start: str, end: str, throttle: Throttle, log) -> pd.DataFrame:
    url, names, extra = ((ARCHIVE_URL, OBSERVED, {"models": "era5"}) if kind == "observed" else
                         (PREVIOUS_RUNS_URL, [f"{v}_previous_day1" for v in FORECAST], {"models": "ecmwf_ifs025"}))
    cdir = CACHE / "open-meteo" / kind / f"{start}_{end}"
    cdir.mkdir(parents=True, exist_ok=True)
    todo = cells[~cells.weather_cell.map(lambda c: cache_file(cdir, c).exists())]
    log(f"  {kind}: {len(cells) - len(todo):,} cells cached, {len(todo):,} to fetch")
    for k in range(0, len(todo), BATCH):
        b = todo.iloc[k:k + BATCH]
        params = {"latitude": ",".join(f"{x:.2f}" for x in b.lat), "longitude": ",".join(f"{x:.2f}" for x in b.lon),
                  "start_date": start, "end_date": end, "hourly": ",".join(names), "timezone": "GMT", **extra}
        throttle.wait(len(b) * math.ceil(len(names) / 10))
        r = None
        for attempt in range(6):
            try:
                r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=180)
            except requests.exceptions.RequestException:      # timeouts and dropped connections: wait and retry
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            break
        if r is None:
            raise SystemExit(f"Open-Meteo {kind} did not answer after 6 attempts")
        if r.status_code != 200:
            raise SystemExit(f"Open-Meteo {kind} failed ({r.status_code}): {r.text[:300]}")
        data = r.json()
        for cell, d in zip(b.weather_cell, data if isinstance(data, list) else [data]):
            cache_file(cdir, cell, new=True).write_text(json.dumps(d), encoding="utf-8")
        log(f"    {kind}: {min(k + BATCH, len(todo)):,}/{len(todo):,}")
    frames, elev = [], {}
    for cell in cells.weather_cell:
        d = json.loads(cache_file(cdir, cell).read_text(encoding="utf-8"))
        h = pd.DataFrame(d["hourly"])
        h.insert(0, "weather_cell", cell)
        frames.append(h)
        elev[cell] = d.get("elevation")
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns.drop(["weather_cell", "time"]):      # all-empty columns arrive as None objects
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if kind == "forecast":
        out = out.rename(columns={f"{v}_previous_day1": f"fc_{v}" for v in FORECAST})
    out.attrs["elevation"] = elev
    return out


def fetch_cems(cells: pd.DataFrame, days: list, key: str, log):
    if not key:
        return None, "skipped: no EWDS_API_KEY in .env"
    import cdsapi
    import xarray as xr
    client = cdsapi.Client(url=os.environ.get("EWDS_API_URL", EWDS_URL), key=key, quiet=True, progress=False)
    cdir = CACHE / "cems"
    cdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in days:
        path = cdir / f"fwi_{day}.nc"
        if not path.exists():
            y, m, d = day.split("-")
            last_err = None
            for dataset_type in ("consolidated_dataset", "intermediate_dataset"):
                req = {"product_type": "reanalysis", "variable": list(CEMS_VARIABLES), "dataset_type": dataset_type,
                       "system_version": ["4_1"], "year": [y], "month": [m], "day": [d], "grid": "0.25/0.25",
                       "data_format": "netcdf"}
                try:
                    log(f"  CEMS {day}: requesting ({dataset_type})")
                    client.retrieve(CEMS_DATASET, req, str(path))
                    break
                except Exception as exc:
                    last_err = exc
            if not path.exists():
                return None, f"failed for {day}: {str(last_err)[:300]}"
        ds = xr.open_dataset(path)
        la = "latitude" if "latitude" in ds.coords else "lat"
        lo = "longitude" if "longitude" in ds.coords else "lon"
        lons = cells.lon % 360 if float(ds[lo].max()) > 180 else cells.lon
        pts = ds.sel({la: xr.DataArray(cells.lat.to_numpy(), dims="p"), lo: xr.DataArray(lons.to_numpy(), dims="p")},
                     method="nearest")
        row = {"weather_cell": cells.weather_cell.to_numpy(), "date": day}
        for v in ds.data_vars:
            prop = CEMS_SHORT.get(v) or CEMS_VARIABLES.get(v)
            if prop:
                row[prop] = np.asarray(pts[v].squeeze()).reshape(-1)[:len(cells)]
        frames.append(pd.DataFrame(row))
        ds.close()
    return pd.concat(frames, ignore_index=True), "ok"


# ---------------------------------------------------------------------------
# Derived weather and lookups
# ---------------------------------------------------------------------------

def condition(t, rh, ws, rain) -> np.ndarray:
    t, rh, ws, rain = (np.asarray(x, float) for x in (t, rh, ws, rain))
    return np.select([rain >= RAIN_MM, (rh < 30) & (ws >= 20) & (t >= 30), rh < 30, rh >= 60],
                     ["rain", "hot, dry and windy", "dry", "humid"], "moderate")


def derive_hours(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["time"] = pd.to_datetime(h.time, utc=True)
    h = h.drop_duplicates(["weather_cell", "time"]).sort_values(["weather_cell", "time"], ignore_index=True)
    for p in ("", "fc_"):
        rad = np.radians(h[f"{p}wind_direction_10m"])          # direction the wind comes FROM
        h[f"{p}wind_u"] = -h[f"{p}wind_speed_10m"] * np.sin(rad)   # component blowing toward east
        h[f"{p}wind_v"] = -h[f"{p}wind_speed_10m"] * np.cos(rad)   # component blowing toward north
        # rain in the last 24 h by clock time (fires leave gaps between date ranges of the same cell)
        r = h.set_index("time").groupby("weather_cell")[f"{p}precipitation"].rolling("24h", min_periods=1).sum()
        h[f"{p}rain_24h_mm"] = r.reset_index(level=0, drop=True).to_numpy().round(2)
        wet = h[f"{p}precipitation"] >= RAIN_MM
        block = wet.groupby(h.weather_cell).cumsum()
        h[f"{p}hours_since_rain"] = h.groupby([h.weather_cell, block]).cumcount()
        seen = wet.groupby(h.weather_cell).cummax()
        h.loc[~seen, f"{p}hours_since_rain"] = np.nan                # no rain yet in the fetched period
        h[f"{p}condition"] = condition(h[f"{p}temperature_2m"], h[f"{p}relative_humidity_2m"],
                                       h[f"{p}wind_speed_10m"], h[f"{p}precipitation"])
    return h


NUM = [c for c in OBSERVED if c not in ("wind_direction_10m", "precipitation")] + ["wind_u", "wind_v"]
FC_NUM = [f"fc_{c}" for c in FORECAST if c not in ("wind_direction_10m", "precipitation")] + ["fc_wind_u", "fc_wind_v"]
STEP = ["precipitation", "rain_24h_mm", "hours_since_rain", "fc_precipitation", "fc_rain_24h_mm", "fc_hours_since_rain"]


def at_time(H: pd.DataFrame, cells: pd.Series, times: pd.Series) -> pd.DataFrame:
    """Weather at exact times: smooth values interpolated between the two hours, rain from the hour
    that contains the time (Open-Meteo rain at hour T fell during the hour before T)."""
    t = pd.to_datetime(times, utc=True).reset_index(drop=True)
    c = cells.reset_index(drop=True)
    h0 = t.dt.floor("h")
    h1 = h0 + pd.Timedelta(hours=1)
    w = ((t - h0).dt.total_seconds() / 3600).to_numpy()
    idx = H.set_index(["weather_cell", "time"])
    a = idx.reindex(pd.MultiIndex.from_arrays([c, h0])).reset_index(drop=True)
    b = idx.reindex(pd.MultiIndex.from_arrays([c, h1])).reset_index(drop=True)
    out = pd.DataFrame(index=range(len(t)))
    for col in NUM + FC_NUM:
        out[col] = (a[col].to_numpy() * (1 - w) + b[col].fillna(a[col]).to_numpy() * w).round(3)
    for col in STEP:
        out[col] = b[col].fillna(a[col]).to_numpy()
    for p in ("", "fc_"):
        out[f"{p}wind_direction_10m"] = (np.degrees(np.arctan2(-out[f"{p}wind_u"], -out[f"{p}wind_v"])) % 360).round(1)
        out[f"{p}condition"] = condition(out[f"{p}temperature_2m"], out[f"{p}relative_humidity_2m"],
                                         out[f"{p}wind_speed_10m"], out[f"{p}precipitation"])
    return out


def window_summary(H: pd.DataFrame, cells: pd.Series, start: pd.Series, end: pd.Series, prefix: str = "") -> pd.DataFrame:
    """Weather summarised over [start hour, end hour] per row (both inclusive)."""
    s, e = pd.to_datetime(start, utc=True).dt.floor("h"), pd.to_datetime(end, utc=True).dt.floor("h")
    n = ((e - s).dt.total_seconds() / 3600).clip(0, MAX_WINDOW_H - 1).astype(int) + 1
    rep = pd.DataFrame({"row": np.repeat(np.arange(len(s)), n), "weather_cell": np.repeat(cells.to_numpy(), n)})
    ends = np.repeat(e.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]"), n)
    rep["time"] = pd.to_datetime(ends - rep.groupby("row").cumcount().to_numpy() * np.timedelta64(1, "h")).tz_localize("UTC")
    j = rep.merge(H, on=["weather_cell", "time"], how="left")
    g = j.groupby("row")
    p = prefix
    out = pd.DataFrame({
        f"{p}hours": g.size(), f"{p}temp_mean_c": g[f"{p}temperature_2m"].mean(), f"{p}temp_max_c": g[f"{p}temperature_2m"].max(),
        f"{p}rh_min_pct": g[f"{p}relative_humidity_2m"].min(), f"{p}rh_mean_pct": g[f"{p}relative_humidity_2m"].mean(),
        f"{p}vpd_max_kpa": g[f"{p}vapour_pressure_deficit"].max(), f"{p}wind_mean_kmh": g[f"{p}wind_speed_10m"].mean(),
        f"{p}gust_max_kmh": g[f"{p}wind_gusts_10m"].max(), f"{p}rain_sum_mm": g[f"{p}precipitation"].sum(),
        f"{p}rain_24h_before_mm": g[f"{p}rain_24h_mm"].max(),
        f"{p}wind_u": g[f"{p}wind_u"].mean(), f"{p}wind_v": g[f"{p}wind_v"].mean(),
        f"{p}condition": g[f"{p}condition"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else ""),
    }).reindex(range(len(s)))
    out[f"{p}wind_dir_mean"] = (np.degrees(np.arctan2(-out[f"{p}wind_u"], -out[f"{p}wind_v"])) % 360).round(1)
    return out.round(3)


def alignment(spread_deg, wind_from_deg) -> np.ndarray:
    """cos of the angle between the spread direction and the downwind direction (1 = with the wind)."""
    downwind = (np.asarray(wind_from_deg, float) + 180) % 360
    return np.round(np.cos(np.radians(np.asarray(spread_deg, float) - downwind)), 3)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def spearman(x, y):
    x, y = pd.Series(x, dtype=float).reset_index(drop=True), pd.Series(y, dtype=float).reset_index(drop=True)
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_N or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return None, int(ok.sum())
    return round(float(x[ok].rank().corr(y[ok].rank())), 3), int(ok.sum())


def auc(a, b):
    """Chance that a value from a is larger than one from b (0.5 = no difference)."""
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    r = pd.concat([a, b]).rank()
    return round(float((r.iloc[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))), 3)


def strength(v):
    a = abs(v)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def relate(df: pd.DataFrame, targets: list, scope: str, weather: list) -> list:
    out = []
    for t, w in product(targets, weather):
        if t in df and w in df:
            v, n = spearman(df[w], df[t])
            if v is not None:
                out.append({"scope": scope, "fire": t, "weather": w, "spearman": v, "n": n, "strength": strength(v)})
    return out


def per_fire_weather(run_dir, rows, mfires, fmap, max_calls, log):
    """Archive batches: only each picked fire's own cells (top MAJOR_CELLS / MEDIUM_CELLS by detections of its main
    merged fire) over its own days (2 days before it starts to the day after it ends)."""
    picked = pd.read_csv(run_dir / "fires_in_batch.csv")
    ldet = pd.read_csv(run_dir / "localisation" / "detections.csv.gz", usecols=["row_id", "fire_id"])
    d = ldet.rename(columns={"fire_id": "source_fire_id"}).merge(fmap, on="source_fire_id").merge(
        rows[["row_id", "weather_cell"]], on="row_id")
    cell_counts = d.groupby(["fire_id", "weather_cell"]).size().rename("n").reset_index()
    mf = mfires.assign(first=mfires.first_seen.astype(str).str[:10], last=mfires.last_seen.astype(str).str[:10])
    tasks = []
    for f in picked.itertuples():
        m = mf[mf.latitude.between(f.south - 0.2, f.north + 0.2) & mf.longitude.between(f.west - 0.2, f.east + 0.2) &
               (mf["first"] <= f.end) & (mf["last"] >= f.start)]
        if m.empty:
            continue
        main = m.sort_values("detections", ascending=False).iloc[0].fire_id
        top = cell_counts[cell_counts.fire_id == main].sort_values("n", ascending=False).weather_cell
        cells = top.head(MAJOR_CELLS if f.kind == "major" else MEDIUM_CELLS).tolist()
        start = (pd.Timestamp(f.start) - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(f.end) + timedelta(days=1)).strftime("%Y-%m-%d")
        tasks.append((f.fire_key, cells, start, end))

    def cost(kind, cells, start, end):
        cdir = CACHE / "open-meteo" / kind / f"{start}_{end}"
        todo = [c for c in cells if not cache_file(cdir, c).exists()]
        return len(todo) * math.ceil(((pd.Timestamp(end) - pd.Timestamp(start)).days + 1) / 14)
    plan = []
    for key, cells, start, end in tasks:
        plan.append(("observed", cells, start, end))
        if end >= FORECAST_FROM:
            plan.append(("forecast", cells[:1], max(start, FORECAST_FROM), end))
    calls = sum(cost(*p) for p in plan)
    log(f"per-fire weather: {len(tasks)} picked fires, {sum(len(c) for _, c, _, _ in tasks)} cells, about {calls:,} new Open-Meteo calls")
    if calls > max_calls:
        raise SystemExit(f"Would need {calls:,} calls, more than --max-calls {max_calls:,} left for today.")
    obs, fc, all_cells = [], [], set()
    for kind, cells, start, end in plan:
        if not cells:
            continue
        df = fetch_open_meteo(kind, cell_latlon(pd.Series(cells)), start, end, THROTTLE, lambda m: None)
        (obs if kind == "observed" else fc).append(df)
        all_cells |= set(cells)
    o = pd.concat(obs, ignore_index=True) if obs else pd.DataFrame(columns=["weather_cell", "time"])
    f = pd.concat(fc, ignore_index=True) if fc else pd.DataFrame(columns=["weather_cell", "time"])
    o.attrs["elevation"] = {k: v for df in obs for k, v in df.attrs.get("elevation", {}).items()}
    days = sorted({t[:10] for t in o.time.astype(str)}) if len(o) else []
    return o, f, cell_latlon(pd.Series(sorted(all_cells))), calls, days, len(tasks)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(run_dir: Path, max_calls: int) -> dict:
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    need = {k: run_dir / v for k, v in {
        "rows": "correlation/rows.csv.gz", "lfires": "localisation/fires.csv.gz", "lobs": "localisation/observations.csv.gz",
        "ffires": "fire_power/fires.csv.gz", "fobs": "fire_power/observations.csv.gz",
        "ep": "epicenter/epicenters.csv.gz", "mfires": "epicenter/fires.csv.gz", "fmap": "epicenter/fire_map.csv.gz",
        "xtl": "expansion/timeline.csv.gz", "xcl": "expansion/clusters.csv.gz", "xdet": "expansion/detections.csv.gz",
    }.items()}
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run the earlier workers first): " + ", ".join(missing))
    rd = lambda k, **kw: pd.read_csv(need[k], low_memory=False, **kw)
    rows = rd("rows", dtype={"acq_time": str, "pass_id": str}, keep_default_na=False, na_values=[""])
    rows["row_id"] = np.arange(len(rows))
    rows["time"] = pd.to_datetime(rows.acq_date) + pd.to_timedelta(rows.acq_minutes, unit="min")
    lfires, lobs = rd("lfires"), rd("lobs", dtype={"pass_id": str})
    ffires, fobs = rd("ffires"), rd("fobs", dtype={"pass_id": str})
    ep, mfires, fmap = rd("ep"), rd("mfires", dtype={"source_fire_ids": str}), rd("fmap")
    xtl, xdet = rd("xtl", dtype={"pass_id": str}), rd("xdet", dtype={"cluster_id": str})
    xcl = rd("xcl", dtype={"cluster_id": str, "parents": str, "children": str})

    # -- which cells and days ------------------------------------------------------
    rows["weather_cell"] = weather_key(rows.latitude, rows.longitude)
    mode = "whole run"
    if (run_dir / "fires_in_batch.csv").exists():
        mode = "per fire"
    ep["weather_cell"] = weather_key(ep.latitude, ep.longitude)
    lobs["weather_cell"] = weather_key(lobs.latitude, lobs.longitude)
    first_day = min(rows.acq_date.min(), pd.to_datetime(ep.earliest_start, utc=True).min().strftime("%Y-%m-%d")
                    if ep.earliest_start.notna().any() else rows.acq_date.min())
    d0 = pd.Timestamp(first_day) - timedelta(days=1)              # one day before, for rain in the last 24 h
    d1 = pd.Timestamp(rows.acq_date.max())
    days = [(d0 + timedelta(days=k)).strftime("%Y-%m-%d") for k in range((d1 - d0).days + 1)]
    if mode == "per fire":
        o, f, cells, calls, days, n_picked = per_fire_weather(run_dir, rows, mfires, fmap, max_calls, log)
        if not len(f):
            f = pd.DataFrame({"weather_cell": o.weather_cell, "time": o.time, **{f"fc_{v}": np.nan for v in FORECAST}})
        H = derive_hours(o.merge(f, on=["weather_cell", "time"], how="outer"))
        cems, cems_status = None, "skipped (not used)"
        mode = f"per fire: {n_picked} picked fires, {len(cells):,} cells"
    else:
        # also the weather cell of every correlation-grid cell centre (so every grid cell has weather)
        cd = json.loads((run_dir / "correlation" / "correlations.json").read_text(encoding="utf-8"))["settings"]["cell_deg"]
        centre = lambda v: (np.floor(v / cd) + 0.5) * cd
        grid_cells = set(weather_key(centre(rows.latitude), centre(rows.longitude))) | set(weather_key(centre(ep.latitude), centre(ep.longitude)))
        cells = cell_latlon(pd.Series(sorted(set(rows.weather_cell) | set(ep.weather_cell) | set(lobs.weather_cell) | grid_cells)))
        calls = estimate_calls(len(cells), len(days))
        log(f"{len(cells):,} weather cells x {len(days)} days ({days[0]} to {days[-1]}): about {calls:,} Open-Meteo calls")
        if calls > max_calls:
            raise SystemExit(f"That is more than --max-calls {max_calls:,}. The public API allows about 10,000 calls a day: "
                             "run a smaller area, raise --max-calls if you know your quota, or use a self-hosted Open-Meteo.")
        o = fetch_open_meteo("observed", cells, days[0], days[-1], THROTTLE, log)
        f = fetch_open_meteo("forecast", cells, days[0], days[-1], THROTTLE, log)
        H = derive_hours(o.merge(f, on=["weather_cell", "time"], how="outer"))
        cems, cems_status = fetch_cems(cells, days, os.environ.get("EWDS_API_KEY", ""), log)
        log(f"  CEMS fire danger indices: {cems_status}")
    fd_cols = list(CEMS_VARIABLES.values()) if cems is not None else []

    # -- weather attached to the fire data -------------------------------------------
    # epicenters: start window (unknown start: 6 h before the first detection) and the first detection itself
    latest = pd.to_datetime(ep.latest_start, utc=True)
    earliest = pd.to_datetime(ep.earliest_start, utc=True).fillna(latest - pd.Timedelta(hours=UNKNOWN_START_CONTEXT_H))
    epw = pd.concat([ep[["fire_id", "epicenter_no", "source_fire_id", "weather_cell", "origin", "spread_direction_deg"]].reset_index(drop=True),
                     window_summary(H, ep.weather_cell, earliest, latest),
                     window_summary(H, ep.weather_cell, earliest, latest, "fc_"),
                     at_time(H, ep.weather_cell, latest).add_prefix("at_detection_")], axis=1)
    epw["window"] = np.where(ep.origin.eq("observed start").to_numpy(), "start window", "6 h before first detection")
    epw["window_start"] = earliest.dt.floor("h").dt.strftime("%Y-%m-%dT%H:00Z").to_numpy()
    epw["window_end"] = latest.dt.floor("h").dt.strftime("%Y-%m-%dT%H:00Z").to_numpy()
    if cems is not None:
        epw["date"] = latest.dt.strftime("%Y-%m-%d").to_numpy()
        epw = epw.merge(cems, on=["weather_cell", "date"], how="left")

    # localisation observations: weather at the pass, spread vs wind
    obw = pd.concat([lobs[["fire_id", "pass_id", "family", "time", "weather_cell", "burning_area_km2", "frp_sum_mw", "pixels"]]
                     .reset_index(drop=True), at_time(H, lobs.weather_cell, lobs.time)], axis=1)
    sd = obw.fire_id.map(lfires.set_index("fire_id").spread_direction_deg)
    obw["spread_direction_deg"] = sd
    obw["wind_alignment"] = alignment(sd, obw.wind_direction_10m)
    obw["fc_wind_alignment"] = alignment(sd, obw.fc_wind_direction_10m)

    # expansion steps: weather at the merged fire's pixels on that pass
    xd = xdet.merge(rows[["row_id", "latitude", "longitude", "pass_id"]], on="row_id")
    cen = xd.groupby(["fire_id", "pass_id"])[["latitude", "longitude"]].mean().reset_index()
    st = xtl.merge(cen, on=["fire_id", "pass_id"], how="left")
    st["weather_cell"] = weather_key(st.latitude, st.longitude)
    stw = pd.concat([st[["fire_id", "step", "time", "pass_id", "family", "pixels", "frp_sum_mw", "new_footprint_km2",
                         "cumulative_fine_km2", "front_distance_km", "events", "weather_cell"]].reset_index(drop=True),
                     at_time(H, st.weather_cell, st.time)], axis=1)
    stw["hours_since_prev"] = (pd.to_datetime(stw.time, utc=True) - pd.to_datetime(stw.time, utc=True)
                               .groupby(stw.fire_id).shift()).dt.total_seconds() / 3600
    stw["fine_growth_km2_h"] = (stw.cumulative_fine_km2 - stw.groupby("fire_id").cumulative_fine_km2.shift()) / stw.hours_since_prev

    # merged fires: weather over all their passes
    g = stw.groupby("fire_id")
    fw = pd.DataFrame({"temp_mean_c": g.temperature_2m.mean(), "rh_min_pct": g.relative_humidity_2m.min(),
                       "vpd_max_kpa": g.vapour_pressure_deficit.max(), "wind_mean_kmh": g.wind_speed_10m.mean(),
                       "gust_max_kmh": g.wind_gusts_10m.max(), "rain_any": g.precipitation.max() >= RAIN_MM,
                       "rain_24h_max_mm": g.rain_24h_mm.max(), "wind_u": g.wind_u.mean(), "wind_v": g.wind_v.mean(),
                       "condition": g.condition.agg(lambda x: x.mode().iloc[0])}).reset_index()
    fw["wind_dir_mean"] = (np.degrees(np.arctan2(-fw.wind_u, -fw.wind_v)) % 360).round(1)
    fw = fw.merge(mfires[["fire_id", "spread_direction_deg", "spread_rate_km_h", "peak_frp_mw", "detections"]], on="fire_id", how="left")
    fw["wind_alignment"] = alignment(fw.spread_direction_deg, fw.wind_dir_mean)
    fw = fw.round(3)

    # -- correlations ------------------------------------------------------------------
    W = WEATHER_VARS
    FC = [f"fc_{v}" for v in W if f"fc_{v}" in obw]
    corr = {"conditions": CONDITIONS, "min_n": MIN_N, "relations": [], "notes": []}

    # spread and growth (tracked source fires; weather averaged over their observations)
    tracked = lfires[lfires.status == "tracked"].set_index("fire_id")
    per_fire = obw[obw.fire_id.isin(tracked.index)].groupby("fire_id")[W + FC + ["wind_alignment"]].mean()
    sp = per_fire.join(tracked[["spread_rate_km_h", "growth_km2_h", "centre_speed_km_h"]])
    corr["relations"] += relate(sp, ["spread_rate_km_h", "growth_km2_h", "centre_speed_km_h"], "tracked fires (observed)", W)
    corr["relations"] += relate(sp, ["spread_rate_km_h", "growth_km2_h"], "tracked fires (day-before forecast)", FC)

    # does fire spread with the wind?
    al = obw.dropna(subset=["wind_alignment"]).drop_duplicates("fire_id")[["fire_id", "wind_alignment", "wind_speed_10m"]]
    bands = pd.cut(al.wind_speed_10m, [0, 10, 20, 1000], right=False, labels=["calm (< 10 km/h)", "moderate (10-20)", "strong (>= 20)"])
    corr["wind_alignment"] = {
        "meaning": "cos of the angle between spread direction and downwind direction: 1 with the wind, -1 against, 0 across; "
                   "random directions average 0 with 25% within 45 degrees",
        "all": {"fires": len(al), "mean": round(float(al.wind_alignment.mean()), 3) if len(al) else None,
                "within_45_deg": round(float((al.wind_alignment >= math.cos(math.radians(45))).mean()), 3) if len(al) else None},
        "by_wind": {str(b): {"fires": int(len(x)), "mean": round(float(x.wind_alignment.mean()), 3),
                             "within_45_deg": round(float((x.wind_alignment >= math.cos(math.radians(45))).mean()), 3)}
                    for b, x in al.groupby(bands, observed=True) if len(x)}}

    # FRP trend (good-quality trends) vs weather and the daily cycle
    ft = ffires[ffires.trend_quality == "good"].set_index("fire_id")
    fo = fobs.merge(lobs[["fire_id", "pass_id", "weather_cell"]], on=["fire_id", "pass_id"], how="left")
    fo = pd.concat([fo.reset_index(drop=True), at_time(H, fo.weather_cell, fo.time)], axis=1)
    agg = fo[fo.fire_id.isin(ft.index)].groupby("fire_id")
    tr = agg[W].mean().join(agg.local_solar_hour.mean()).join(
        (agg.relative_humidity_2m.last() - agg.relative_humidity_2m.first()).rename("rh_change_pct")).join(
        (agg.temperature_2m.last() - agg.temperature_2m.first()).rename("temp_change_c")).join(ft[["frp_trend_pct_h"]])
    corr["relations"] += relate(tr, ["frp_trend_pct_h"], "FRP trend, good quality (observed)",
                                W + ["rh_change_pct", "temp_change_c", "local_solar_hour"])

    # FRP per pass vs weather at the pass (VIIRS only, so pixel sizes are comparable)
    v = fo[(fo.family == "VIIRS") & (fo.frp_sum_mw > 0)].assign(log_frp=lambda x: np.log10(x.frp_sum_mw))
    corr["relations"] += relate(v, ["log_frp", "frp_density_mw_km2"], "FRP per VIIRS pass (observed)", W)

    # footprint growth per hour between consecutive passes (VIIRS passes, positive time gaps)
    gr = stw[(stw.family == "VIIRS") & (stw.hours_since_prev > 0)]
    corr["relations"] += relate(gr, ["fine_growth_km2_h", "new_footprint_km2"], "footprint growth per VIIRS pass (observed)", W)

    # extinction vs rain and humidity after the last sighting (window: last seen -> end or +6 h)
    lv = xcl[xcl.end_how.isin(["extinguished", "burning at data end"])].copy()
    lv_cell = lv.cluster_id.map(xdet.drop_duplicates("cluster_id", keep="last").set_index("cluster_id").row_id).map(rows.weather_cell)
    last = pd.to_datetime(lv.last_seen, utc=True)
    end = pd.to_datetime(lv.end, utc=True).fillna(last + pd.Timedelta(hours=6))
    ex = pd.concat([lv[["cluster_id", "end_how"]].reset_index(drop=True), window_summary(H, lv_cell, last, end)], axis=1)
    ex["out"] = ex.end_how.eq("extinguished")
    rainy = ex.rain_sum_mm >= RAIN_MM
    corr["extinction"] = {
        "window": "from the last sighting to the confirming pass (or 6 h for clusters still burning)",
        "clusters": int(len(ex)),
        "share_extinguished_with_rain": round(float(ex.out[rainy].mean()), 3) if rainy.sum() else None,
        "share_extinguished_without_rain": round(float(ex.out[~rainy].mean()), 3) if (~rainy).sum() else None,
        "clusters_with_rain": int(rainy.sum()),
        "auc_humidity_extinguished_vs_burning": auc(ex.rh_mean_pct[ex.out], ex.rh_mean_pct[~ex.out]),
    }

    # ignition: start windows (observed starts) vs all hours in the fire cells
    obs_start = epw[epw.origin == "observed start"]
    bg = H[H.weather_cell.isin(set(ep.weather_cell))]
    ost = ep[ep.origin == "observed start"]
    ctrl = window_summary(H, ost.weather_cell, pd.to_datetime(ost.earliest_start, utc=True) - pd.Timedelta(hours=24),
                          pd.to_datetime(ost.latest_start, utc=True) - pd.Timedelta(hours=24))
    sw = obs_start.reset_index(drop=True)

    def paired(a, b, lower=False):
        ok = a.notna() & b.notna()
        if ok.sum() < MIN_N:
            return None
        return round(float(((a < b) if lower else (a > b))[ok].mean()), 3)

    corr["ignition"] = {
        "meaning": "paired: share of start windows with a higher value than the same cell and hours one day earlier "
                   "(0.5 = no difference); AUC: chance a start window beats a random hour in the same cells, "
                   "which includes the daily cycle",
        "start_windows": int(len(obs_start)),
        "paired_vs_day_before": {"hotter (max temperature)": paired(sw.temp_max_c, ctrl.temp_max_c),
                                 "drier (min humidity lower)": paired(sw.rh_min_pct, ctrl.rh_min_pct, lower=True),
                                 "higher max VPD": paired(sw.vpd_max_kpa, ctrl.vpd_max_kpa),
                                 "windier (mean wind)": paired(sw.wind_mean_kmh, ctrl.wind_mean_kmh),
                                 "difference in max temperature, median C": round(float((sw.temp_max_c - ctrl.temp_max_c).median()), 2),
                                 "difference in min humidity, median %": round(float((sw.rh_min_pct - ctrl.rh_min_pct).median()), 2)},
        "auc": {"max temperature": auc(obs_start.temp_max_c, bg.temperature_2m),
                "min humidity (lower = drier)": auc(obs_start.rh_min_pct, bg.relative_humidity_2m),
                "max VPD": auc(obs_start.vpd_max_kpa, bg.vapour_pressure_deficit),
                "mean wind": auc(obs_start.wind_mean_kmh, bg.wind_speed_10m),
                "rain in the 24 h before": auc(obs_start.rain_24h_before_mm, bg.rain_24h_mm)},
        "median": {"start window max VPD kPa": round(float(obs_start.vpd_max_kpa.median()), 2) if len(obs_start) else None,
                   "all hours VPD kPa": round(float(bg.vapour_pressure_deficit.median()), 2)}}

    # by weather condition class
    share_bg = bg.condition.value_counts(normalize=True)
    starts = obs_start.condition.value_counts(normalize=True)
    cond = {}
    for c in ["hot, dry and windy", "dry", "moderate", "humid", "rain"]:
        so = obw[obw.condition == c]
        ft_c = sp.join(obw.groupby("fire_id").condition.agg(lambda x: x.mode().iloc[0]).rename("cond"))
        ft_c = ft_c[ft_c.cond == c]
        ex_c = ex[ex.condition == c]
        cond[c] = {"share_of_hours": round(float(share_bg.get(c, 0)), 3),
                   "share_of_fire_starts": round(float(starts.get(c, 0)), 3),
                   "observations": int(len(so)),
                   "median_frp_per_pass_mw": round(float(so.frp_sum_mw.median()), 2) if len(so) else None,
                   "tracked_fires": int(len(ft_c)),
                   "median_spread_rate_km_h": round(float(ft_c.spread_rate_km_h.median()), 3) if len(ft_c) else None,
                   "clusters_ending": int(len(ex_c)),
                   "share_extinguished": round(float(ex_c.out.mean()), 3) if len(ex_c) else None}
    corr["by_condition"] = cond
    if cems is None:
        corr["notes"].append("CEMS fire danger indices not included: " + cems_status)

    # -- write ------------------------------------------------------------------------
    out = run_dir / "weather"
    out.mkdir(parents=True, exist_ok=True)
    Hs = H.copy()
    Hs["time"] = Hs.time.dt.strftime("%Y-%m-%dT%H:00Z")
    Hs.round(3).to_csv(out / "hours.csv.gz", index=False)
    cells.assign(elevation_m=cells.weather_cell.map(o.attrs.get("elevation", {}))).to_csv(out / "cells.csv.gz", index=False)
    if cems is not None:
        cems.to_csv(out / "fire_danger.csv.gz", index=False)
    epw.to_csv(out / "epicenters.csv.gz", index=False)
    obw.to_csv(out / "observations.csv.gz", index=False)
    stw.round(4).to_csv(out / "steps.csv.gz", index=False)
    fw.to_csv(out / "fires.csv.gz", index=False)
    (out / "correlations.json").write_text(json.dumps(corr, indent=1), encoding="utf-8")
    summary = {"mode": mode, "cells": len(cells), "days": days, "hours": len(H), "cems": cems_status, "open_meteo_calls_estimated": calls,
               "epicenters": len(epw), "observations": len(obw), "steps": len(stw), "fires": len(fw),
               "sources": {"observed": "Open-Meteo archive API, ERA5 0.25 deg (CC BY 4.0)",
                           "forecast": "Open-Meteo previous-runs API, ECMWF IFS 0.25 deg, issued the day before (CC BY 4.0)",
                           "fire_danger": f"Copernicus EMS {CEMS_DATASET}: {cems_status}"},
               "attribution": "Weather data by Open-Meteo.com (CC BY 4.0); contains modified Copernicus EMS information",
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"summary": summary, "corr": corr, "out": out}


def print_summary(res: dict) -> None:
    s, c = res["summary"], res["corr"]
    print(f"\nWeather for {s['cells']:,} cells x {len(s['days'])} days in {s['seconds']} s; CEMS: {s['cems']}")
    print(f"Attached to {s['epicenters']:,} epicenters, {s['observations']:,} observations, {s['steps']:,} expansion steps, "
          f"{s['fires']:,} fires")
    rel = sorted([r for r in c["relations"] if r["strength"] != "negligible"], key=lambda r: -abs(r["spearman"]))
    print("\nFire behaviour vs weather (Spearman, weak or stronger):")
    for r in rel[:18]:
        print(f"  {r['spearman']:+.3f}  {r['fire']:<20} ~ {r['weather']:<26} {r['scope']} (n={r['n']:,})")
    wa = c["wind_alignment"]
    print(f"\nSpread with the wind: mean alignment {wa['all']['mean']}, {wa['all']['within_45_deg']} within 45 deg "
          f"({wa['all']['fires']:,} tracked fires; random = 0 and 0.25)")
    for b, x in wa["by_wind"].items():
        print(f"  {b:<20} {x['fires']:>5,} fires  mean {x['mean']:+.3f}  within 45 deg {x['within_45_deg']:.3f}")
    ig = c["ignition"]
    print(f"\nIgnition: {ig['start_windows']:,} start windows vs the same cell and hours one day earlier (0.5 = no difference)")
    for k, v in ig["paired_vs_day_before"].items():
        print(f"  {k:<44} {v}")
    print("  vs all hours in the fire cells (AUC; includes the daily cycle):")
    for k, v in ig["auc"].items():
        print(f"    {k:<30} {v}")
    e = c["extinction"]
    print(f"\nExtinction: {e['clusters']:,} last clusters; extinguished with rain {e['share_extinguished_with_rain']} "
          f"({e['clusters_with_rain']:,} with rain), without rain {e['share_extinguished_without_rain']}")
    print("\nBy weather condition:")
    print(f"  {'condition':<20} {'hours':>6} {'starts':>7} {'obs':>6} {'FRP/pass':>9} {'spread':>7} {'out':>6}")
    for k, x in c["by_condition"].items():
        fmt = lambda v, f: "-" if v is None else format(v, f)
        print(f"  {k:<20} {x['share_of_hours']:>6.2f} {x['share_of_fire_starts']:>7.2f} {x['observations']:>6,} "
              f"{fmt(x['median_frp_per_pass_mw'], '>9.1f')} {fmt(x['median_spread_rate_km_h'], '>7.2f')} {fmt(x['share_extinguished'], '>6.2f')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    ap.add_argument("--max-calls", type=int, default=9000, help="stop if Open-Meteo would need more calls than this")
    args = ap.parse_args()
    if args.run:
        run_dir = Path(args.run)
    elif args.date:
        run_dir = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    else:
        ap.error("give --run or --date")
    res = run(run_dir, args.max_calls)
    print_summary(res)
    print(f"\nSaved to {res['out']}")


if __name__ == "__main__":
    main()
