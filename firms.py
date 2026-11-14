"""Shared NASA FIRMS API client. Reads FIRMS_MAP_KEY from .env."""

import csv
import io
import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"

SENSORS = [
    "MODIS_NRT",
    "MODIS_SP",
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA21_SP",
    "LANDSAT_NRT",
]
SP_SENSORS = [s for s in SENSORS if s.endswith("_SP")]  # the files that carry `type`


def load_env(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments, optional quotes."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env(BASE_DIR / ".env")
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))


def firms_get(path: str) -> str:
    if not MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY is missing. Add it to .env")
    with urllib.request.urlopen(f"{FIRMS_API}/{path}", timeout=180) as resp:
        return resp.read().decode("utf-8")


def fetch_fires(sensor: str, day: str, days: int, area: str = "world") -> dict:
    """Return {"columns": [...], "rows": [[...], ...]} for one sensor."""
    text = firms_get(f"area/csv/{MAP_KEY}/{sensor}/{area}/{days}/{day}")
    # FIRMS returns plain-text errors (e.g. "Invalid MAP_KEY.") with HTTP 200.
    if not text.lstrip().lower().startswith(("latitude", "country_id")):
        raise ValueError(text.strip()[:300] or "Empty response from FIRMS")
    rows = list(csv.reader(io.StringIO(text)))
    return {"columns": rows[0], "rows": rows[1:]}


def fetch_availability() -> dict:
    text = firms_get(f"data_availability/csv/{MAP_KEY}/all")
    reader = csv.DictReader(io.StringIO(text))
    return {r["data_id"]: {"min": r["min_date"], "max": r["max_date"]} for r in reader}
