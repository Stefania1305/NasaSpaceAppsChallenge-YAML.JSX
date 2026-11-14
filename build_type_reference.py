"""Build the reference used to infer `type` for NRT / Landsat detections.

Downloads world-wide SP data (the files that carry `type`) for a spread of
sample dates, bins it onto a lat/lon grid and saves data/type_reference.json.gz,
which server.py loads at start-up.

    python build_type_reference.py                 # 8 dates per SP sensor
    python build_type_reference.py --samples 12    # more history, better coverage
    python build_type_reference.py --evaluate      # also measure accuracy on a held-out date

Downloads are cached in data/type_samples/, so re-running is cheap.
"""

import argparse
import csv
import gzip
from collections import Counter
from datetime import date, timedelta

from firms import DATA_DIR, SP_SENSORS, fetch_availability, fetch_fires
from type_inference import DEFAULT_CELL_DEG, REFERENCE_PATH, TYPE_LABELS, TypeReference

SAMPLES_DIR = DATA_DIR / "type_samples"


def sample_dates(avail: dict, sensor: str, samples: int, spacing: int) -> list[str]:
    a = avail.get(sensor)
    if not a:
        return []
    newest, oldest = date.fromisoformat(a["max"]), date.fromisoformat(a["min"])
    days = [newest - timedelta(days=k * spacing) for k in range(samples)]
    return [d.isoformat() for d in days if d >= oldest]


def load_sample(sensor: str, day: str, refresh: bool) -> list[tuple]:
    """Return [(lat, lon, type, acq_date), ...] for one sensor-day, cached on disk."""
    path = SAMPLES_DIR / f"{sensor}_{day}.csv.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", newline="") as fh:
            return [tuple(r) for r in csv.reader(fh)]
    data = fetch_fires(sensor, day, 1, "world")
    cols = data["columns"]
    li, lo, ti, di = (cols.index(c) for c in ("latitude", "longitude", "type", "acq_date"))
    records = [(r[li], r[lo], r[ti], r[di]) for r in data["rows"]]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        csv.writer(fh).writerows(records)
    return records


def build(samples: dict, cell: float, exclude: set = frozenset()) -> TypeReference:
    ref = TypeReference(cell)
    for (sensor, day), records in samples.items():
        if (sensor, day) not in exclude:
            ref.learn(sensor, records)
    return ref


def evaluate(samples: dict, holdout: set, cells: list[float]) -> None:
    test = [(float(la), float(lo), int(t)) for k in holdout for la, lo, t, _ in samples[k]]
    print(f"\nHeld-out test set: {len(test):,} labelled detections from "
          f"{', '.join(f'{s} {d}' for s, d in sorted(holdout))}")
    base = Counter(t for *_, t in test)
    print(f"Baseline (always type 0): accuracy {base[0] / len(test):.1%}")
    for cell in cells:
        ref = build(samples, cell, exclude=holdout)
        confusion = Counter((t, ref.predict(la, lo)[0]) for la, lo, t in test)
        correct = sum(v for (a, p), v in confusion.items() if a == p)
        print(f"\nCell {cell}°  accuracy {correct / len(test):.2%}")
        print("  type  label                        actual  recall  precision")
        for t, label in TYPE_LABELS.items():
            actual = sum(v for (a, _), v in confusion.items() if a == t)
            predicted = sum(v for (_, p), v in confusion.items() if p == t)
            hit = confusion[(t, t)]
            rec = f"{hit / actual:.1%}" if actual else "-"
            prec = f"{hit / predicted:.1%}" if predicted else "-"
            print(f"  {t}     {label:<28} {actual:>6}  {rec:>6}  {prec:>9}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=8, help="sample dates per SP sensor (default 8)")
    p.add_argument("--spacing", type=int, default=23, help="days between sample dates (default 23)")
    p.add_argument("--cell", type=float, default=DEFAULT_CELL_DEG, help="grid cell size in degrees")
    p.add_argument("--evaluate", action="store_true", help="report accuracy on the newest date, held out")
    p.add_argument("--refresh", action="store_true", help="re-download cached samples")
    args = p.parse_args()

    avail = fetch_availability()
    samples = {}
    for sensor in SP_SENSORS:
        for day in sample_dates(avail, sensor, args.samples, args.spacing):
            samples[(sensor, day)] = load_sample(sensor, day, args.refresh)
            print(f"{sensor} {day}: {len(samples[(sensor, day)]):,} detections")

    if args.evaluate:
        newest = {s: max(d for ss, d in samples if ss == s) for s in {s for s, _ in samples}}
        evaluate(samples, {(s, d) for s, d in newest.items()},
                 sorted({0.01, DEFAULT_CELL_DEG, 0.05, args.cell}))

    ref = build(samples, args.cell)
    ref.save(REFERENCE_PATH)
    st = ref.status()
    print(f"\nSaved {REFERENCE_PATH.relative_to(DATA_DIR.parent)}: {st['labelled_detections']:,} labelled "
          f"detections from {st['sensor_days']} sensor-days in {st['cells']:,} cells")
    print("Type counts:", st["type_counts"])
    print("Restart server.py to load it.")


if __name__ == "__main__":
    main()
