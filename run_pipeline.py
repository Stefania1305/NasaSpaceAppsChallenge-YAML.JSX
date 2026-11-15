"""Run the whole pipeline for one date range and time every stage.

    correlation -> interpreter -> localisation + fire power -> epicenter -> fire expansion
      -> KG feed 1 (fire) -> weather join -> KG feed 2 (weather)

Weather stages are skipped with --no-weather (a world day needs ~32,000 Open-Meteo
calls, more than the free daily quota). Timings go to <run>/timings.json.

Usage:  python run_pipeline.py --date 2026-11-01 [--days 1] [--area world] [--no-weather]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from firms import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--start-at", default="", help="skip the stages before this one, reusing their output (e.g. interpreter)")
    args = ap.parse_args()
    common = ["--date", args.date, "--days", args.days, f"--area={args.area}"]
    # a tuple of scripts runs them at the same time (they are independent)
    stages = [("correlation (incl. FIRMS download)", "correlation.py"), ("interpreter", "interpreter.py"),
              ("localisation + fire power (together)", ("localisation_worker.py", "fire_power_worker.py")),
              ("epicenter", "epicenter_worker.py"), ("fire expansion", "fire_expansion_worker.py"),
              ("KG feed 1 (fire)", "kg_builder.py")]
    if not args.no_weather:
        stages += [("weather join", "weather_worker.py"), ("KG feed 2 (weather)", "kg_weather_feed.py")]
    if args.start_at:
        names = [n for n, _ in stages]
        first = next((i for i, n in enumerate(names) if n.startswith(args.start_at)), None)
        if first is None:
            ap.error(f"--start-at must be the start of one of: {names}")
        stages = stages[first:]
    run_dir = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    log = {"date": args.date, "days": args.days, "area": args.area,
           "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stages": []}
    t_all = time.time()
    for name, script in stages:
        t0 = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} ...", flush=True)
        scripts = script if isinstance(script, tuple) else (script,)
        procs = [subprocess.Popen([sys.executable, "-X", "utf8", sc, *common], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for sc in scripts]
        outs = [p.communicate() for p in procs]
        ok = all(p.returncode == 0 for p in procs)
        secs = round(time.time() - t0, 1)
        log["stages"].append({"stage": name, "seconds": secs, "ok": ok})
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   done in {secs} s" + ("" if ok else " - FAILED"), flush=True)
        if not ok:
            for o, e in outs:
                print(o[-2000:], e[-2000:], sep="\n")
            break
    log["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["total_seconds"] = round(time.time() - t_all, 1)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "timings.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"\nTotal: {log['total_seconds']} s ({log['total_seconds'] / 60:.1f} min). Timings in {run_dir / 'timings.json'}")


if __name__ == "__main__":
    main()
