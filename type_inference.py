"""Infer the FIRMS `type` field for files that do not carry it.

Only the SP (standard processing) files have `type`:
    0 = presumed vegetation fire
    1 = active volcano
    2 = other static land source (gas flares, industry, ...)
    3 = offshore

FIRMS assigns these by *location* (volcano list, static-source mask, land/water
mask), so the generalisation is spatial: SP detections are binned onto a fine
lat/lon grid and each cell remembers how many detections of each type it has
seen. An unlabelled detection gets the majority type of its cell, falling back
to the surrounding cells, and finally to "vegetation fire" (the dominant class)
when nothing labelled has ever been seen nearby.
"""

import gzip
import json
import math
import threading

from firms import DATA_DIR

TYPE_LABELS = {
    0: "presumed vegetation fire",
    1: "active volcano",
    2: "other static land source",
    3: "offshore",
}
DEFAULT_CELL_DEG = 0.01          # ~1.1 km at the equator (best on held-out test)
NEIGHBOUR_RADII = (1, 2)         # fall back to 3x3, then 5x5 cells
REFERENCE_PATH = DATA_DIR / "type_reference.json.gz"

# Columns added to every table by add_type_columns().
TYPE_COLUMNS = ["type", "type_label", "type_existed", "type_confidence", "type_basis"]


class TypeReference:
    def __init__(self, cell_deg: float = DEFAULT_CELL_DEG):
        self.cell_deg = cell_deg
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.totals = [0, 0, 0, 0]
        self.learned: set[str] = set()   # "SENSOR:YYYY-MM-DD" already counted
        self.lock = threading.Lock()

    # -- building ---------------------------------------------------------
    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return math.floor(lat / self.cell_deg), math.floor(lon / self.cell_deg)

    def add(self, lat: float, lon: float, t: int) -> None:
        if t not in TYPE_LABELS:
            return
        counts = self.cells.setdefault(self._key(lat, lon), [0, 0, 0, 0])
        counts[t] += 1
        self.totals[t] += 1

    def learn(self, sensor: str, records) -> int:
        """Add (lat, lon, type, acq_date) records, skipping sensor-days already learned
        so that repeating the same query never double-counts. Returns rows added."""
        added, new_days = 0, set()
        with self.lock:
            for lat, lon, t, day in records:
                key = f"{sensor}:{day}"
                if key in self.learned:
                    continue
                try:
                    self.add(float(lat), float(lon), int(t))
                except (TypeError, ValueError):
                    continue
                new_days.add(key)
                added += 1
            self.learned |= new_days
        return added

    @property
    def size(self) -> int:
        return sum(self.totals)

    # -- inference --------------------------------------------------------
    def predict(self, lat: float, lon: float) -> tuple[int, float | None, str]:
        """Return (type, confidence 0-1, human-readable basis)."""
        i, j = self._key(lat, lon)
        counts = self.cells.get((i, j))
        if counts:
            return self._decide(counts, f"{self.cell_deg}° cell")
        for r in NEIGHBOUR_RADII:
            agg = [0, 0, 0, 0]
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    c = self.cells.get((i + di, j + dj))
                    if c:
                        agg = [a + b for a, b in zip(agg, c)]
            if any(agg):
                return self._decide(agg, f"{2 * r + 1}x{2 * r + 1} cells around")
        total = self.size
        prior = round(self.totals[0] / total, 3) if total else None
        return 0, prior, "no labelled detections nearby; default to dominant class"

    @staticmethod
    def _decide(counts, where: str):
        n = sum(counts)
        t = max(range(4), key=counts.__getitem__)
        return t, round(counts[t] / n, 3), f"{where} ({n} labelled, {counts[t]} were type {t})"

    # -- persistence ------------------------------------------------------
    def save(self, path=REFERENCE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            payload = {
                "cell_deg": self.cell_deg,
                "totals": self.totals,
                "learned": sorted(self.learned),
                "cells": {f"{i},{j}": c for (i, j), c in self.cells.items()},
            }
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        tmp.replace(path)

    @classmethod
    def load(cls, path=REFERENCE_PATH) -> "TypeReference":
        if not path.exists():
            return cls()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        ref = cls(payload["cell_deg"])
        ref.totals = payload["totals"]
        ref.learned = set(payload["learned"])
        for k, c in payload["cells"].items():
            i, j = k.split(",")
            ref.cells[(int(i), int(j))] = c
        return ref

    def status(self) -> dict:
        sensor_days = len(self.learned)
        return {
            "labelled_detections": self.size,
            "sensor_days": sensor_days,
            "cells": len(self.cells),
            "cell_deg": self.cell_deg,
            "type_counts": dict(zip(TYPE_LABELS.values(), self.totals)),
        }


def add_type_columns(ref: TypeReference, columns: list, rows: list) -> tuple[list, list]:
    """Return (columns, rows) with TYPE_COLUMNS filled in.

    If a row already has a `type` value it is kept (type_existed = "yes");
    otherwise it is inferred from `ref` (type_existed = "no")."""
    lat_i, lon_i = columns.index("latitude"), columns.index("longitude")
    type_i = columns.index("type") if "type" in columns else None
    base_cols = [c for c in columns if c != "type"]
    keep = [k for k, c in enumerate(columns) if c != "type"]

    out = []
    for r in rows:
        raw = r[type_i].strip() if type_i is not None and type_i < len(r) else ""
        if raw.isdigit() and int(raw) in TYPE_LABELS:
            t, existed, conf, basis = int(raw), "yes", 1.0, "original value from file"
        else:
            try:
                t, conf, basis = ref.predict(float(r[lat_i]), float(r[lon_i]))
            except ValueError:
                t, conf, basis = 0, None, "invalid coordinates; default"
            existed = "no"
        out.append([r[k] for k in keep] +
                   [str(t), TYPE_LABELS[t], existed, "" if conf is None else f"{conf:.3f}", basis])
    return base_cols + TYPE_COLUMNS, out
