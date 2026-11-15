"""Knowledge graph, feed 2 (weather): runs after the weather join.

    correlation -> interpreter -> localisation + fire power -> epicenter -> fire expansion -> weather join
                                                                                |                 |
                                                                          KG feed 1: fire   KG feed 2: weather
                                                                                +--- one graph ---+

Adds the weather part to the one knowledge graph, in its own files, from the
weather join's output only. It never rewrites a fire file; it checks that every
fire node it links to exists (feed 1 must have run) and reuses feed 1's Field
nodes. Rerunning it replaces only its own files.

Nodes
  WeatherCell      0.25 degree ERA5 / CEMS grid point (with elevation)
  WeatherHour      one cell, one hour: observed values, day-before forecast (fc_*), condition
  WeatherDay       one cell, one day: CEMS fire danger indices (when the weather join had them)
  StartWeather     weather over one epicenter's start window
  FireWeather      weather over one fire's life
  WeatherFindings  the weather join's fire-weather findings for the run (JSON text)
  Field            new measures only (fire behaviour and weather variables)
  Worker           the weather join itself, with its run summary on (Run)-[:RAN]->(Worker)

Relationships
  (GridCell)-[:IN_WEATHER_CELL]->(WeatherCell)-[:HAS_HOUR]->(WeatherHour)
  (WeatherCell)-[:HAS_DAY]->(WeatherDay)
  (Epicenter)-[:STARTED_IN {role, hours_before_detection}]->(WeatherHour)   every hour of the start window
  (Epicenter)-[:HAD_START_WEATHER]->(StartWeather)
  (Observation)-[:OBSERVED_IN {weather at the pass, interpolated}]->(WeatherHour)
  (FireState)-[:OBSERVED_IN {weather at the pass, interpolated}]->(WeatherHour)
  (Fire)-[:HAD_WEATHER]->(FireWeather)
  (Field)-[:CORRELATES {scope: fire-weather ...}]->(Field)
  (Run)-[:HAS_WEATHER_FINDINGS]->(WeatherFindings)
  (Run)-[:RAN {summary_json, seconds}]->(Worker)
CORRELATES holds every fire-weather relationship the weather join measured (all strengths).

Usage:  python kg_weather_feed.py --date 2026-10-13 --days 2 --area -48,-12,-42,-6
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from firms import DATA_DIR
from kg_builder import Writer

MAX_WINDOW_H = 24
WEATHER_FIELDS = {"temperature_2m", "relative_humidity_2m", "vapour_pressure_deficit", "wind_speed_10m",
                  "wind_gusts_10m", "rain_24h_mm", "hours_since_rain", "rh_change_pct", "temp_change_c"}
PASS_VALUES = ["temperature_2m", "relative_humidity_2m", "vapour_pressure_deficit", "wind_speed_10m", "wind_direction_10m",
               "wind_gusts_10m", "precipitation", "rain_24h_mm", "hours_since_rain", "soil_moisture_0_to_7cm", "condition",
               "fc_temperature_2m", "fc_relative_humidity_2m", "fc_vapour_pressure_deficit", "fc_wind_speed_10m",
               "fc_wind_direction_10m", "fc_wind_gusts_10m", "fc_precipitation", "fc_condition"]


def hour_of(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t, utc=True).dt.floor("h").dt.strftime("%Y-%m-%dT%H")


def build(run_dir: Path) -> dict:
    t0 = time.time()
    kg, wdir = run_dir / "kg", run_dir / "weather"
    if not (kg / "schema.json").exists():
        raise SystemExit(f"No fire part in {kg}: run kg_builder.py (feed 1) first.")
    if not (wdir / "summary.json").exists():
        raise SystemExit(f"No weather join output in {wdir}: run weather_worker.py first.")
    schema = json.loads((kg / "schema.json").read_text(encoding="utf-8"))
    fire = schema["feeds"]["fire"]

    # replace only this feed's earlier files
    old = schema["feeds"].get("weather") or {}
    for f in list(old.get("node_files", {}).values()) + list(old.get("relationship_files", {}).values()):
        (kg / f).unlink(missing_ok=True)

    rd = lambda n, **kw: pd.read_csv(wdir / f"{n}.csv.gz", low_memory=False, **kw)
    cells, hours = rd("cells", dtype={"weather_cell": str}), rd("hours", dtype={"weather_cell": str})
    epw, obw = rd("epicenters", dtype={"weather_cell": str}), rd("observations", dtype={"weather_cell": str, "pass_id": str})
    stw, fw = rd("steps", dtype={"weather_cell": str, "pass_id": str, "events": str}), rd("fires")
    # only rows that actually have weather (archive batches fetch weather for the picked fires' cells only)
    epw = epw[epw.temp_mean_c.notna()].reset_index(drop=True)
    obw = obw[obw.temperature_2m.notna()].reset_index(drop=True)
    stw = stw[stw.temperature_2m.notna()].reset_index(drop=True)
    fw = fw[fw.temp_mean_c.notna()].reset_index(drop=True)
    fd = rd("fire_danger", dtype={"weather_cell": str}) if (wdir / "fire_danger.csv.gz").exists() else None
    corr = json.loads((wdir / "correlations.json").read_text(encoding="utf-8"))
    wsum = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))

    run_id = run_dir.name
    W = Writer(kg, run_id)
    for label in ("Run", "GridCell", "Epicenter", "Observation", "FireState", "Fire", "Field", "Worker"):
        files = [v for k, v in fire["node_files"].items() if k == label]
        W.load_ids(label, files)

    # -- weather grid -----------------------------------------------------------------
    W.nodes("WeatherCell", W.key("wx", cells.weather_cell), cells)
    hours["hour"] = hours.time.str[:13]
    hid = W.key("wx", hours.weather_cell, hours.hour)
    W.nodes("WeatherHour", hid, hours.drop(columns=["hour"]), {"time"})
    W.rels("HAS_HOUR", "WeatherCell", W.key("wx", hours.weather_cell), "WeatherHour", hid)
    gc = pd.read_csv(kg / fire["node_files"]["GridCell"], usecols=["id:ID(GridCell)", "weather_cell:string"])
    gc = gc[gc["weather_cell:string"].isin(set(cells.weather_cell))]
    W.rels("IN_WEATHER_CELL", "GridCell", gc["id:ID(GridCell)"], "WeatherCell", W.key("wx", gc["weather_cell:string"]))
    if fd is not None:
        did = W.key("wx", fd.weather_cell, fd.date)
        W.nodes("WeatherDay", did, fd)
        W.rels("HAS_DAY", "WeatherCell", W.key("wx", fd.weather_cell), "WeatherDay", did)
    have = set(hid)

    # -- epicenters: start-window hours and the window summary --------------------------
    ep_id = W.key(epw.fire_id, epw.epicenter_no)
    latest = pd.to_datetime(epw.window_end, utc=True)
    earliest = pd.to_datetime(epw.window_start, utc=True)
    span = ((latest - earliest).dt.total_seconds() / 3600).fillna(0).clip(0, MAX_WINDOW_H - 1).astype(int)
    r = epw.loc[epw.index.repeat(span + 1)].copy()
    off = r.groupby(level=0).cumcount()
    r["hour"] = (latest.loc[r.index] - pd.to_timedelta(off, unit="h")).dt.strftime("%Y-%m-%dT%H")
    r["role"] = np.where(off.to_numpy() == 0, "first detection hour", "inside start window")
    r["hours_before_detection"] = off.to_numpy()
    r["wh"] = W.key("wx", r.weather_cell, r.hour)
    r = r[r.wh.isin(have)]
    W.rels("STARTED_IN", "Epicenter", W.key(r.fire_id, r.epicenter_no), "WeatherHour", r.wh, r[["role", "hours_before_detection"]])
    sw_id = W.key("startwx", epw.fire_id, epw.epicenter_no)
    W.nodes("StartWeather", sw_id, epw.drop(columns=["fire_id", "epicenter_no", "source_fire_id", "spread_direction_deg"]),
            {"window_start", "window_end"})
    W.rels("HAD_START_WEATHER", "Epicenter", ep_id, "StartWeather", sw_id)

    # -- passes: weather at the pass on the link -------------------------------------------
    keep = [c for c in PASS_VALUES + ["wind_alignment", "fc_wind_alignment"] if c in obw]
    wh = W.key("wx", obw.weather_cell, hour_of(obw.time))
    ok = wh.isin(have).to_numpy()
    W.rels("OBSERVED_IN", "Observation", W.key(obw.fire_id, obw.pass_id)[ok], "WeatherHour", wh[ok],
           obw.loc[ok, keep].reset_index(drop=True), file_key="OBSERVED_IN_Observation")
    keep = [c for c in PASS_VALUES + ["fine_growth_km2_h"] if c in stw]
    wh = W.key("wx", stw.weather_cell, hour_of(stw.time))
    ok = wh.isin(have).to_numpy()
    W.rels("OBSERVED_IN", "FireState", W.key("state", stw.fire_id, stw.step)[ok], "WeatherHour", wh[ok],
           stw.loc[ok, keep].reset_index(drop=True), file_key="OBSERVED_IN_FireState")

    # -- fires --------------------------------------------------------------------------------
    fw_id = W.key("firewx", fw.fire_id)
    W.nodes("FireWeather", fw_id, fw.drop(columns=["fire_id", "spread_direction_deg", "spread_rate_km_h", "peak_frp_mw", "detections"]))
    W.rels("HAD_WEATHER", "Fire", W.key(fw.fire_id), "FireWeather", fw_id)

    # -- fire-weather correlations and findings -------------------------------------------------
    rel = pd.DataFrame(corr["relations"])
    if len(rel):
        names = sorted(set(rel.fire) | set(rel.weather))
        new = [n for n in names if W.key("field", n) not in W.ids["Field"]]
        if new:
            nf = pd.DataFrame({"name": new})
            nf["kind"] = nf.name.map(lambda n: "weather" if n.startswith("fc_") or n in WEATHER_FIELDS else "fire behaviour")
            nf["derived_from"], nf["description"] = "", "added by the weather feed"
            W.nodes("Field", W.key("field", nf.name), nf, file_key="Field_weather")
        W.rels("CORRELATES", "Field", W.key("field", rel.weather), "Field", W.key("field", rel.fire),
               pd.DataFrame({"measure": "spearman", "value": rel.spearman, "scope": "fire-weather: " + rel.scope,
                             "n": rel.n, "strength": rel.strength}), file_key="CORRELATES_weather")
    findings = pd.DataFrame({k + "_json": [json.dumps(corr[k])] for k in ("wind_alignment", "ignition", "extinction", "by_condition")})
    findings["conditions"] = corr["conditions"]
    findings["cems"], findings["attribution"] = wsum["cems"], wsum["attribution"]
    findings["observed_source"], findings["forecast_source"] = wsum["sources"]["observed"], wsum["sources"]["forecast"]
    wid = W.key("worker", "weather")
    if wid not in W.ids["Worker"]:
        W.nodes("Worker", pd.Series([wid]), pd.DataFrame({"name": ["weather"], "goal": [
            "fetch observed and forecast weather and fire danger, attach it to the fires and correlate fire behaviour with it"]}),
            file_key="Worker_weather")
    W.rels("RAN", "Run", pd.Series([run_id]), "Worker", pd.Series([wid]),
           pd.DataFrame({"summary_json": [json.dumps(wsum)], "seconds": [wsum.get("seconds")]}), file_key="RAN_weather")
    fid = pd.Series([W.key("wxfindings")])
    W.nodes("WeatherFindings", fid, findings)
    W.rels("HAS_WEATHER_FINDINGS", "Run", pd.Series([run_id]), "WeatherFindings", fid)

    # -- record the feed ------------------------------------------------------------------------
    schema["feeds"]["weather"] = {"script": "kg_weather_feed.py", "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "node_files": W.files["nodes"], "relationship_files": W.files["relationships"],
                                  "sources": wsum["sources"], "attribution": wsum["attribution"], "cems": wsum["cems"],
                                  "days": wsum["days"]}
    schema["node_files"] = {**fire["node_files"], **W.files["nodes"]}
    schema["relationship_files"] = {**fire["relationship_files"], **W.files["relationships"]}
    (kg / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    summary = json.loads((kg / "summary.json").read_text(encoding="utf-8"))
    counts = {k: v for k, v in summary["counts"].items() if k in schema["node_files"] or k in schema["relationship_files"]}
    counts.update(W.counts)
    summary.update({"counts": counts, "feeds": {"fire": True, "weather": True},
                    "nodes": int(sum(counts[k] for k in schema["node_files"])),
                    "relationships": int(sum(counts[k] for k in schema["relationship_files"])),
                    "size_mb": round(sum(p.stat().st_size for p in kg.glob("*.csv.gz")) / 1e6, 1),
                    "weather_feed_seconds": round(time.time() - t0, 1)})
    summary["neo4j_import"] = ("neo4j-admin database import full neo4j --overwrite-destination "
                               + " ".join(f"--nodes={kg.name}/{v}" for v in schema["node_files"].values()) + " "
                               + " ".join(f"--relationships={kg.name}/{v}" for v in schema["relationship_files"].values()))
    (kg / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"summary": summary, "added": W.counts, "files": W.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--days", default="1")
    ap.add_argument("--area", default="world")
    args = ap.parse_args()
    if args.run:
        run_dir = Path(args.run)
    elif args.date:
        run_dir = DATA_DIR / "runs" / (f"{args.date}_{args.days}d_" + re.sub(r"[^\w.-]+", "_", args.area))
    else:
        ap.error("give --run or --date")
    res = build(run_dir)
    s = res["summary"]
    print(f"\nFeed 2 (weather) added in {s['weather_feed_seconds']} s:")
    for k in list(res["files"]["nodes"]) + list(res["files"]["relationships"]):
        print(f"  {k:<26} {res['added'][k]:>9,}")
    print(f"Graph now: {s['nodes']:,} nodes, {s['relationships']:,} relationships, {s['size_mb']} MB "
          "(every link end checked against the node files)")


if __name__ == "__main__":
    main()
