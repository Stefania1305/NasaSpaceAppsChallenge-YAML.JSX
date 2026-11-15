"""Weather for every archive fire batch, then KG feed 2, within Open-Meteo's free daily quota.

For each data/runs/fires_batch_*: the weather worker in per-fire mode (each picked fire's own
cells and days; 3 cells for major fires, 1 for medium), then kg_weather_feed.py adds the weather
part to that batch's graph. All batches share one throttle (per minute and per hour) and one
daily budget of Open-Meteo calls; a batch that would go over the budget is left for later (rerun
the script tomorrow: everything downloaded is cached, finished batches are skipped).

Usage:  python weather_fires.py [--budget 9000]
"""

import argparse
import json
import time
from pathlib import Path

import kg_weather_feed
import weather_worker
from firms import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget", type=int, default=9000, help="Open-Meteo calls allowed in this run")
    args = ap.parse_args()
    runs = sorted((DATA_DIR / "runs").glob("fires_batch_*"))
    left, t0, done, skipped, failed = args.budget, time.time(), [], [], []
    for k, run in enumerate(runs, start=1):
        kg = run / "kg" / "schema.json"
        if kg.exists() and (json.loads(kg.read_text(encoding="utf-8"))["feeds"].get("weather")):
            skipped.append(run.name)
            continue
        try:
            res = weather_worker.run(run, left)
            used = int(res["summary"]["open_meteo_calls_estimated"])
            left -= used
            kg_weather_feed.build(run)
            done.append(run.name)
            print(f"[{k}/{len(runs)}] {run.name}: {res['summary']['mode']}, {used} calls, {left:,} left; "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        except SystemExit as e:
            failed.append((run.name, str(e)[:200]))
            print(f"[{k}/{len(runs)}] {run.name}: not done: {str(e)[:200]}", flush=True)
            if "more than --max-calls" in str(e) or "Would need" in str(e):
                continue
        except Exception as e:
            failed.append((run.name, repr(e)[:200]))
            print(f"[{k}/{len(runs)}] {run.name}: FAILED: {repr(e)[:200]}", flush=True)
    out = {"done": done, "already_had_weather": skipped, "not_done": failed, "calls_left": left,
           "minutes": round((time.time() - t0) / 60, 1)}
    (DATA_DIR / "archive" / "weather_fires.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nDone in {out['minutes']} min: {len(done)} batches got weather, {len(skipped)} already had it, "
          f"{len(failed)} not done; {left:,} calls left of {args.budget:,}")


if __name__ == "__main__":
    main()
