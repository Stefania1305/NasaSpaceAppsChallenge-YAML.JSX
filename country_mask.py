"""Point-in-country tests for limiting a prediction to one country's borders.

The country shapes are the ones the app ships for its "Pick a country" zone method
(frontend/public/countries.geojson: Natural Earth 1:50m admin-0 countries, each with a numeric `id`).

`CountryMask(id).contains(lon, lat)` uses the even-odd rule (holes and islands handled). The polygon edges are
grouped into latitude bands, so a test only looks at the few edges near the point's latitude: fast enough for
hundreds of thousands of fire detections.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

COUNTRIES_FILE = Path(__file__).resolve().parent / "frontend" / "public" / "countries.geojson"
BAND = 0.25          # degrees of latitude per band


def _countries() -> dict:
    # Keyed on the file's modification time: when countries.geojson is updated (e.g. a country's territories are
    # split differently) the server picks it up straight away, so its territory numbers always match the app's.
    return _load_countries(COUNTRIES_FILE.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def _load_countries(_mtime: int) -> dict:
    data = json.loads(COUNTRIES_FILE.read_text(encoding="utf-8"))
    return {f["properties"]["id"]: f for f in data["features"]}


class CountryMask:
    """`part`: one of the country's territories (its `parts` list, e.g. France's French Guiana); None = all of it."""
    def __init__(self, country_id: int, part: int | None = None):
        feature = _countries().get(country_id)
        if feature is None:
            raise ValueError("Unknown country.")
        props = feature["properties"]
        self.name = props["name"]
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        if part is not None:
            parts = props.get("parts") or []
            if not 0 <= part < len(parts):
                raise ValueError("Unknown territory.")
            polygons = [polygons[k] for k in parts[part]["polys"]]
            if parts[part]["name"] != self.name:
                self.name = f'{self.name}: {parts[part]["name"]}'
        self.bands: dict[int, list[tuple[float, float, float, float]]] = {}
        lons, lats = [], []
        for polygon in polygons:
            for ring in polygon:                          # outer ring and holes alike (even-odd rule)
                for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
                    if y1 == y2:
                        continue                          # horizontal edges never cross a horizontal ray
                    lo, hi = min(y1, y2), max(y1, y2)
                    for b in range(math.floor(lo / BAND), math.floor(hi / BAND) + 1):
                        self.bands.setdefault(b, []).append((x1, y1, x2, y2))
                    lons += (x1, x2); lats += (y1, y2)
        self.box = (min(lons), min(lats), max(lons), max(lats))

    def contains(self, lon: float, lat: float) -> bool:
        w, s, e, n = self.box
        if not (w <= lon <= e and s <= lat <= n):
            return False
        inside = False
        for x1, y1, x2, y2 in self.bands.get(math.floor(lat / BAND), ()):
            if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside


def country_mask(country_id: int, part: int | None = None) -> CountryMask:
    return _mask(country_id, part, COUNTRIES_FILE.stat().st_mtime_ns)


@lru_cache(maxsize=64)
def _mask(country_id: int, part: int | None, _mtime: int) -> CountryMask:
    return CountryMask(country_id, part)
