"""Knowledge graph, feed 1 (fire): runs right after the fire expansion worker.

    correlation -> interpreter -> localisation + fire power -> epicenter -> fire expansion -> weather join
                                                                                |                 |
                                                                          KG feed 1: fire   KG feed 2: weather
                                                                                +--- one graph ---+

There is one knowledge graph, fed separately by two feeds. This feed writes
the fire part from the outputs of correlation, interpreter, localisation, fire
power, epicenter and fire expansion. It does not wait for weather and writes
no weather values. Feed 2 (kg_weather_feed.py) adds the weather part later,
in its own files, linked to the nodes written here.

Rerunning this feed rebuilds the fire part and removes feed 2's files (they
point at fire nodes that may have changed); rerun feed 2 afterwards.

Output is Neo4j bulk-import CSV (headers carry the types), one file per node
label and per relationship type, in <run>/kg/. Every id starts with the run
name, so several runs can be loaded into the same graph without clashing.

Nodes
  Run            the dataset this run covered
  Satellite      Terra, Aqua, Suomi NPP, NOAA-20, NOAA-21, Landsat 8/9
  Pass           one orbit segment of one satellite
  GridCell       0.05 degree cell (with the 0.25 degree weather cell it lies in)
  Detection      one harmonised row (all rows; non-wildfire ones flagged)
  SourceFire     a fire as localisation found it (with its fire-power values)
  Fire           the merged fire (neighbouring source fires joined)
  Epicenter      a starting point of a fire, with its start window
  Observation    one pass over one source fire
  Cluster        a pixel cluster followed through time by the expansion worker
  FireState      one fire at one satellite pass: its timeline step
  Field          a data field
  Worker         a pipeline worker (correlation, interpreter, localisation, fire power, epicenter, expansion)
  DataSource     a FIRMS source (MODIS_NRT, VIIRS_NOAA20_SP, ...) used by the run
  Package        one interpreter work package
  ExactRule      a formula checked row by row by the correlation layer
  LinkKind       same_pass / repeat_pass / other_satellite
  ExpansionEvent ignition, new burst, merge, split, extinction (fire expansion worker)

Relationships
  (Detection)-[:SEEN_ON]->(Pass)-[:BY]->(Satellite)
  (Detection)-[:IN_CELL]->(GridCell)            (Epicenter)-[:IN_CELL]->(GridCell)
  (Detection)-[:LINKED {kind, dist_km, gap_min}]->(Detection)   correlation links
  (Detection)-[:PART_OF]->(SourceFire)-[:MERGED_INTO]->(Fire)
  (Fire)-[:HAS_EPICENTER {epicenter_no}]->(Epicenter)-[:FROM_SOURCE]->(SourceFire)
  (Epicenter)-[:LAST_MISSED_BY {time}]->(Satellite)
  (Observation)-[:OF_FIRE]->(SourceFire)        (Observation)-[:DURING]->(Pass)
  (Fire)-[:IN_RUN]->(Run)
  (Fire)-[:HAS_CLUSTER]->(Cluster)-[:BECAME {how, time, undone_within_2h}]->(Cluster)   merges, splits
  (Detection)-[:IN_CLUSTER]->(Cluster)          (Epicenter)-[:STARTED_CLUSTER]->(Cluster)
  (Fire)-[:HAS_STATE]->(FireState)-[:NEXT]->(FireState)      (FireState)-[:DURING]->(Pass)
  (Field)-[:CORRELATES {measure, value, scope, n}]->(Field)   moderate or stronger, layer 1
  (Worker)-[:USES {reason}]->(Field)            routing decided by the interpreter
  (Worker)-[:NOT_SENT {reason}]->(Field)        fields the interpreter held back, and why
  (Run)-[:RAN {summary_json, seconds}]->(Worker)  every worker's run summary
  (Run)-[:USED_SOURCE {status, count, note}]->(DataSource)
  (Package)-[:FOR_WORKER]->(Worker)             (Detection)-[:IN_PACKAGE]->(Package)
  (ExactRule)-[:INVOLVES]->(Field)              with the share of rows that matched
  (Field)-[:AGREES_ACROSS {pair, measure, value, n}]->(LinkKind)   field agreement across linked detections
  (Fire)-[:HAD_EVENT]->(ExpansionEvent)-[:INVOLVES]->(Cluster)
CORRELATES now holds every field relationship the correlation layer measured
(all strengths and scopes, with a by_construction flag).

Usage:  python kg_builder.py --date 2026-10-13      (then: python kg_weather_feed.py --date ...)
Output: <run>/kg/nodes_<Label>.csv.gz, rels_<TYPE>.csv.gz, schema.json, summary.json
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from firms import DATA_DIR
from weather_worker import weather_key

MAX_WINDOW_H = 24


# ---------------------------------------------------------------------------
# Writing Neo4j import files
# ---------------------------------------------------------------------------

def neo_type(s: pd.Series, name: str, datetimes: set) -> str:
    if name in datetimes:
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "long"
    if pd.api.types.is_float_dtype(s):
        return "double"
    return "string"


def iso_dt(s: pd.Series) -> pd.Series:
    """'2026-10-13T12:25Z' -> '2026-10-13T12:25:00Z' (empty stays empty)."""
    s = s.fillna("").astype(str)
    return s.where(~s.str.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\dZ"), s.str[:-1] + ":00Z")


class Writer:
    def __init__(self, out: Path, run_id: str):
        self.out, self.run, self.counts, self.files = out, run_id, {}, {"nodes": {}, "relationships": {}}
        self.ids = {}
        out.mkdir(parents=True, exist_ok=True)

    def key(self, *parts):
        """Run-prefixed id from parts joined by '|'; vectorised when any part is a Series."""
        series = [p for p in parts if isinstance(p, pd.Series)]
        if not series:
            return self.run + "|" + "|".join(map(str, parts))
        out = pd.Series(self.run, index=series[0].index)
        for p in parts:
            out = out + "|" + (p.astype(str) if isinstance(p, pd.Series) else str(p))
        return out

    def load_ids(self, label: str, files: list) -> None:
        """Make nodes written earlier (by the other feed) known, so links to them can be checked."""
        ids = set()
        for f in files:
            ids |= set(pd.read_csv(self.out / f, usecols=[0]).iloc[:, 0])
        self.ids[label] = self.ids.get(label, set()) | ids

    def nodes(self, label: str, ids: pd.Series, props: pd.DataFrame, datetimes: set = frozenset(),
              file_key: str | None = None):
        df = props.copy()
        for c in datetimes & set(df.columns):
            df[c] = iso_dt(df[c])
        cols = {c: f"{c}:{neo_type(df[c], c, datetimes)}" for c in df.columns}
        df = df.rename(columns=cols)
        df.insert(0, f"id:ID({label})", ids.to_numpy())
        df[":LABEL"] = label
        if df.iloc[:, 0].duplicated().any() or df.iloc[:, 0].isin(self.ids.get(label, set())).any():
            raise ValueError(f"duplicate {label} ids")
        fk = file_key or label
        name = f"nodes_{fk}.csv.gz"
        df.to_csv(self.out / name, index=False, na_rep="")
        self.ids[label] = self.ids.get(label, set()) | set(df.iloc[:, 0])
        self.counts[fk] = len(df)
        self.files["nodes"][fk] = name

    def rels(self, rtype: str, start_label: str, start: pd.Series, end_label: str, end: pd.Series,
             props: pd.DataFrame | None = None, datetimes: set = frozenset(), file_key: str | None = None):
        df = pd.DataFrame({f":START_ID({start_label})": start.to_numpy(), f":END_ID({end_label})": end.to_numpy()})
        if props is not None:
            p = props.reset_index(drop=True).copy()
            for c in datetimes & set(p.columns):
                p[c] = iso_dt(p[c])
            for c in p.columns:
                df[f"{c}:{neo_type(p[c], c, datetimes)}"] = p[c].to_numpy()
        df[":TYPE"] = rtype
        bad_s = ~df.iloc[:, 0].isin(self.ids[start_label])
        bad_e = ~df.iloc[:, 1].isin(self.ids[end_label])
        if bad_s.any() or bad_e.any():
            raise ValueError(f"{rtype}: {int(bad_s.sum())} start and {int(bad_e.sum())} end ids have no node")
        fk = file_key or rtype                       # one type may link several label pairs
        name = f"rels_{fk}.csv.gz"
        df.to_csv(self.out / name, index=False, na_rep="")
        self.counts[fk] = len(df)
        self.files["relationships"][fk] = name


def cell_key(lat: pd.Series, lon: pd.Series, deg: float) -> pd.Series:
    return (np.floor(lat / deg).astype("Int64").astype(str) + ":" + np.floor(lon / deg).astype("Int64").astype(str))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(run_dir: Path) -> dict:
    t0 = time.time()
    need = {k: run_dir / v for k, v in {
        "rows": "correlation/rows.csv.gz", "links": "correlation/links.csv.gz", "corr": "correlation/correlations.json",
        "plan": "interpreter/plan.json",
        "lfires": "localisation/fires.csv.gz", "lobs": "localisation/observations.csv.gz", "ldet": "localisation/detections.csv.gz",
        "ffires": "fire_power/fires.csv.gz", "fobs": "fire_power/observations.csv.gz", "fdet": "fire_power/detections.csv.gz",
        "ep": "epicenter/epicenters.csv.gz", "mfires": "epicenter/fires.csv.gz", "fmap": "epicenter/fire_map.csv.gz",
        "xfires": "expansion/fires.csv.gz", "xtl": "expansion/timeline.csv.gz", "xcl": "expansion/clusters.csv.gz",
        "xev": "expansion/events.csv.gz", "xdet": "expansion/detections.csv.gz", "xfirst": "expansion/source_first_clusters.csv.gz",
    }.items()}
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run layers 1-4 and the expansion worker first): " + ", ".join(missing))
    summaries = {}
    for name, f in (("localisation", "localisation/summary.json"), ("fire_power", "fire_power/summary.json"),
                    ("epicenter", "epicenter/summary.json"), ("expansion", "expansion/summary.json")):
        if (run_dir / f).exists():
            summaries[name] = json.loads((run_dir / f).read_text(encoding="utf-8"))
    read = lambda k, **kw: pd.read_csv(need[k], low_memory=False, **kw)
    rows = read("rows", dtype={"acq_time": str, "type": str, "confidence_raw": str, "pass_id": str},
                keep_default_na=False, na_values=[""])
    links = read("links")
    corr = json.loads(need["corr"].read_text(encoding="utf-8"))
    plan = json.loads(need["plan"].read_text(encoding="utf-8"))
    lfires, lobs, ldet = read("lfires"), read("lobs", dtype={"pass_id": str}), read("ldet", dtype={"pass_id": str})
    ffires, fobs, fdet = read("ffires"), read("fobs", dtype={"pass_id": str}), read("fdet")
    ep, mfires, fmap = read("ep"), read("mfires", dtype={"source_fire_ids": str}), read("fmap")
    xfires, xtl = read("xfires"), read("xtl", dtype={"pass_id": str})
    xcl = read("xcl", dtype={"cluster_id": str, "parents": str, "children": str, "sources": str})   # "3.10" is not 3.1
    xev, xdet, xfirst = read("xev", dtype={"clusters": str}), read("xdet", dtype={"cluster_id": str}),         read("xfirst", dtype={"first_cluster_id": str})

    for df, col in ((lfires, "type"), (ep, "start_type")):      # type is text everywhere, as on Detection
        df[col] = df[col].astype("Int64").astype(str).replace("<NA>", "")
    run_id = run_dir.name
    kg = run_dir / "kg"
    if kg.exists():                                   # feed 1 rebuilds the fire part; feed 2's files go too
        for old in list(kg.glob("*.csv.gz")) + list(kg.glob("*.json")):
            old.unlink()
    W = Writer(kg, run_id)
    cell_deg = corr["settings"]["cell_deg"]
    rows["row_id"] = np.arange(len(rows))
    rows["time"] = rows.acq_date + "T" + rows.acq_time.str.zfill(4).str[:2] + ":" + rows.acq_time.str.zfill(4).str[2:] + "Z"

    # -- Run, Satellite, Pass, GridCell ------------------------------------
    info = corr.get("run", {})
    W.nodes("Run", pd.Series([run_id]), pd.DataFrame({
        "first_date": [info.get("range", [rows.acq_date.min()])[0]], "last_date": [info.get("range", [None, rows.acq_date.max()])[-1]],
        "area": [info.get("area", "")], "days": [int(info.get("days", rows.acq_date.nunique()))],
        "detections": [len(rows)], "built": [pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")]}), {"built"})

    sats = rows.groupby("satellite").instrument.first().reset_index()
    W.nodes("Satellite", W.key(sats.satellite), sats.rename(columns={"satellite": "name"}))

    passes = rows.groupby("pass_id").agg(satellite=("satellite", "first"), start=("time", "min"), end=("time", "max"),
                                         detections=("row_id", "size")).reset_index()
    W.nodes("Pass", W.key(passes.pass_id), passes, {"start", "end"})
    W.rels("BY", "Pass", W.key(passes.pass_id), "Satellite", W.key(passes.satellite))

    rows["cell"] = cell_key(rows.latitude, rows.longitude, cell_deg)
    ep["cell"] = cell_key(ep.latitude, ep.longitude, cell_deg)
    cells = pd.concat([rows[["cell", "latitude", "longitude"]], ep[["cell", "latitude", "longitude"]]]).drop_duplicates("cell")
    ij = cells.cell.str.split(":", expand=True).astype(int)
    cells = pd.DataFrame({"cell": cells.cell.to_numpy(), "cell_deg": cell_deg,
                          "centre_lat": ((ij[0] + 0.5) * cell_deg).round(4).to_numpy(),
                          "centre_lon": ((ij[1] + 0.5) * cell_deg).round(4).to_numpy()})
    cells["weather_cell"] = weather_key(cells.centre_lat, cells.centre_lon)
    W.nodes("GridCell", W.key(cells.cell), cells)

    # -- Detection ----------------------------------------------------------
    used = set(ldet.row_id)
    d = rows.merge(fdet[["row_id", "frp_density_mw_km2", "frp_anomaly_z"]], on="row_id", how="left") \
            .merge(ldet[["row_id", "new_front"]], on="row_id", how="left")
    d["wildfire"] = d.row_id.isin(used)
    d["new_front"] = d.new_front.eq("yes")
    d["removed_by_interpreter"] = np.where(d.wildfire, "", "non-wildfire type " + d.type.astype(str) + " (" + d.type_label.astype(str) + ")")
    keep = ["row_id", "source", "instrument", "satellite", "latitude", "longitude", "time", "acq_date", "daynight",
            "local_solar_hour", "overpass_slot", "confidence", "confidence_raw", "bright_main_k", "bright_secondary_k",
            "bright_diff_k", "frp_mw", "frp_physics_proxy", "frp_density_mw_km2", "frp_anomaly_z", "pixel_area_km2",
            "scan_km", "track_km", "type", "type_label", "type_existed", "type_confidence", "type_basis",
            "patch_id", "wildfire", "new_front", "removed_by_interpreter", "acq_minutes", "confidence_score",
            "confidence_pct", "cell_group", "link_count"]
    det_id = W.key(d.row_id)
    W.nodes("Detection", det_id, d[keep], {"time"})
    W.rels("SEEN_ON", "Detection", det_id, "Pass", W.key(d.pass_id))
    W.rels("IN_CELL", "Detection", det_id, "GridCell", W.key(d.cell))
    W.rels("LINKED", "Detection", W.key(links.i), "Detection", W.key(links.j), links[["kind", "dist_km", "gap_min"]])

    # -- SourceFire, Fire, Observation ---------------------------------------
    fp = ffires.drop(columns=["detections", "observations", "instruments"]).add_prefix("fp_").rename(columns={"fp_fire_id": "fire_id"})
    sf = lfires.merge(fp, on="fire_id", how="left")
    sf_id = W.key(sf.fire_id)
    W.nodes("SourceFire", sf_id, sf, {"first_seen", "last_seen", "fp_peak_time"})
    W.rels("PART_OF", "Detection", W.key(ldet.row_id), "SourceFire", W.key(ldet.fire_id))

    life = xfires.drop(columns=[]).add_prefix("exp_").rename(columns={"exp_fire_id": "fire_id"})
    mfires = mfires.merge(life, on="fire_id", how="left")
    W.nodes("Fire", W.key(mfires.fire_id), mfires,
            {"first_seen", "last_seen", "peak_time", "exp_burst", "exp_last_seen", "exp_extinguished_by"})
    W.rels("MERGED_INTO", "SourceFire", W.key(fmap.source_fire_id), "Fire", W.key(fmap.fire_id))
    W.rels("IN_RUN", "Fire", W.key(mfires.fire_id), "Run", pd.Series([run_id] * len(mfires)))

    fo = fobs.drop(columns=["family", "satellite", "time", "daynight"], errors="ignore")
    fo = fo.rename(columns={c: "fp_" + c for c in fo.columns if c not in ("fire_id", "pass_id")})
    obs = lobs.merge(fo, on=["fire_id", "pass_id"], how="left")
    obs["fp_partial_view"] = obs.fp_partial_view.eq("yes")
    obs_id = W.key(obs.fire_id, obs.pass_id)
    W.nodes("Observation", obs_id, obs, {"time"})
    W.rels("OF_FIRE", "Observation", obs_id, "SourceFire", W.key(obs.fire_id))
    W.rels("DURING", "Observation", obs_id, "Pass", W.key(obs.pass_id))

    # -- Epicenter -------------------------------------------------------------
    ep["weather_cell"] = weather_key(ep.latitude, ep.longitude)
    ep_id = W.key(ep.fire_id, ep.epicenter_no)
    W.nodes("Epicenter", ep_id, ep.drop(columns=["cell"]), {"earliest_start", "latest_start"})
    W.rels("HAS_EPICENTER", "Fire", W.key(ep.fire_id), "Epicenter", ep_id, ep[["epicenter_no"]])
    W.rels("FROM_SOURCE", "Epicenter", ep_id, "SourceFire", W.key(ep.source_fire_id))
    W.rels("IN_CELL", "Epicenter", ep_id, "GridCell", W.key(ep.cell), file_key="IN_CELL_Epicenter")
    missed = ep[ep.earliest_from.notna() & (ep.earliest_from != "")]
    W.rels("LAST_MISSED_BY", "Epicenter", W.key(missed.fire_id, missed.epicenter_no), "Satellite",
           W.key(missed.earliest_from), missed[["earliest_start"]].rename(columns={"earliest_start": "time"}), {"time"})

    # -- Cluster, FireState: how each fire formed and evolved (expansion worker) --
    xcl = xcl.fillna({"parents": "", "children": "", "out_seen_by": ""})
    W.nodes("Cluster", W.key("cl", xcl.cluster_id), xcl, {"born", "first_seen_history", "last_seen", "end"})
    W.rels("HAS_CLUSTER", "Fire", W.key(xcl.fire_id), "Cluster", W.key("cl", xcl.cluster_id))
    undone = {}
    for c, u in zip(xev.clusters, xev.undone_within_2h):
        if " -> " in str(c):
            for kid in c.split(" -> ")[1].split("+"):
                undone[kid] = bool(u)
    lin = xcl[xcl.parents != ""].assign(parent=lambda x: x.parents.str.split("+")).explode("parent")
    lin["undone_within_2h"] = lin.cluster_id.map(undone).fillna(False).astype(bool)
    W.rels("BECAME", "Cluster", W.key("cl", lin.parent), "Cluster", W.key("cl", lin.cluster_id),
           lin[["born_how", "born", "undone_within_2h"]].rename(columns={"born_how": "how", "born": "time"}), {"time"})
    W.rels("IN_CLUSTER", "Detection", W.key(xdet.row_id), "Cluster", W.key("cl", xdet.cluster_id))
    sc = ep[["fire_id", "epicenter_no", "source_fire_id"]].merge(xfirst, on="source_fire_id")
    W.rels("STARTED_CLUSTER", "Epicenter", W.key(sc.fire_id, sc.epicenter_no), "Cluster", W.key("cl", sc.first_cluster_id))
    xtl = xtl.fillna({"events": ""})
    st_id = W.key("state", xtl.fire_id, xtl.step)
    W.nodes("FireState", st_id, xtl, {"time"})
    W.rels("HAS_STATE", "Fire", W.key(xtl.fire_id), "FireState", st_id)
    nxt = xtl.assign(next_step=xtl.step + 1).merge(xtl[["fire_id", "step", "time"]], left_on=["fire_id", "next_step"],
                                                   right_on=["fire_id", "step"], suffixes=("", "_next"))
    gap = (pd.to_datetime(nxt.time_next, utc=True) - pd.to_datetime(nxt.time, utc=True)).dt.total_seconds() / 60
    W.rels("NEXT", "FireState", W.key("state", nxt.fire_id, nxt.step), "FireState", W.key("state", nxt.fire_id, nxt.next_step),
           pd.DataFrame({"gap_min": gap.round(1)}))
    W.rels("DURING", "FireState", st_id, "Pass", W.key(xtl.pass_id), file_key="DURING_FireState")
    xev = xev.reset_index(drop=True)
    ev_id = W.key("event", xev.index.to_series())
    W.nodes("ExpansionEvent", ev_id, xev.drop(columns=["fire_id"]).fillna({"pass_id": ""}), {"time"})
    W.rels("HAD_EVENT", "Fire", W.key(xev.fire_id), "ExpansionEvent", ev_id)
    inv = xev.assign(cl=xev.clusters.astype(str).str.replace(" -> ", "+").str.split("+")).explode("cl")
    W.rels("INVOLVES", "ExpansionEvent", W.key("event", inv.index.to_series()), "Cluster", W.key("cl", inv.cl),
           file_key="INVOLVES_ExpansionEvent")

    # -- Field, Worker: what layers 1 and 2 learned ------------------------------
    f = corr["fields"]
    names = sorted(set(f["numeric"]) | set(f["categorical"]) |
                   {r["a"] for r in corr["relations"]} | {r["b"] for r in corr["relations"]} |
                   {r["field"] for r in corr["link_relations"]} |
                   {fld for w in plan["workers"].values() for fld in list(w["fields"]) + list(w["not_sent"])})
    fields = pd.DataFrame({"name": names})
    fields["kind"] = fields.name.map(lambda n: "numeric" if n in f["numeric"] else "categorical" if n in f["categorical"] else "other")
    fields["derived_from"] = fields.name.map(lambda n: "+".join(f["derived_from"].get(n, [])))
    fields["description"] = fields.name.map(lambda n: f["descriptions"].get(n, ""))
    W.nodes("Field", W.key("field", fields.name), fields)
    rel = pd.DataFrame(corr["relations"])
    W.rels("CORRELATES", "Field", W.key("field", rel.a), "Field", W.key("field", rel.b),
           rel[["measure", "value", "scope", "n", "strength", "by_construction"]])

    ex = pd.DataFrame(corr["exact"])
    W.nodes("ExactRule", W.key("rule", ex.index.to_series()), ex)
    inv = ex.assign(field=ex.relation.map(lambda t: [n for n in names if re.search(rf"\b{re.escape(n)}\b", t)])).explode("field").dropna(subset=["field"])
    W.rels("INVOLVES", "ExactRule", W.key("rule", inv.index.to_series()), "Field", W.key("field", inv.field), file_key="INVOLVES_ExactRule")
    lr = pd.DataFrame(corr["link_relations"])
    kinds = pd.DataFrame({"kind": sorted(lr.kind.unique())})
    kinds["links"] = kinds.kind.map(links.kind.value_counts()).fillna(0).astype(int)
    W.nodes("LinkKind", W.key("linkkind", kinds.kind), kinds)
    W.rels("AGREES_ACROSS", "Field", W.key("field", lr.field), "LinkKind", W.key("linkkind", lr.kind),
           lr.drop(columns=["field", "kind"]).assign(by_construction=lr.get("by_construction", False)).fillna({"strength": ""}))

    # data sources the harmonisation used
    src = pd.DataFrame(corr.get("run", {}).get("sources", []))
    if len(src):
        ds = src.groupby("sensor").agg(status=("status", lambda x: "+".join(sorted(set(x)))), count=("count", "sum"),
                                       note=("note", lambda x: "; ".join(n for n in x if n))).reset_index()
        W.nodes("DataSource", W.key("source", ds.sensor), ds[["sensor"]].rename(columns={"sensor": "name"}))
        W.rels("USED_SOURCE", "Run", pd.Series([run_id] * len(ds)), "DataSource", W.key("source", ds.sensor),
               ds[["status", "count", "note"]])

    goals = {"correlation": "measure how fields relate and which detections belong together",
             "interpreter": "send each worker the fields and rows it needs, in work packages",
             "epicenter": "estimate where and when each fire started; merge neighbouring fires",
             "expansion": "follow how fire clusters form, merge, split and go out"}
    wk = pd.DataFrame([{"name": n, "goal": w["goal"], "rows": w["rows"], "packages": len(w["packages"])}
                       for n, w in plan["workers"].items()] +
                      [{"name": n, "goal": g, "rows": None, "packages": None} for n, g in goals.items()])
    W.nodes("Worker", W.key("worker", wk.name), wk)
    corr_summary = {k: corr[k] for k in ("settings", "rows", "scopes", "groups", "links", "seconds") if k in corr}
    plan_summary = {k: plan[k] for k in ("settings", "rows_in", "units", "row_filter", "checks", "seconds") if k in plan}
    ran = [("correlation", corr_summary), ("interpreter", plan_summary)] + list(summaries.items())
    W.rels("RAN", "Run", pd.Series([run_id] * len(ran)), "Worker", W.key("worker", pd.Series([n for n, _ in ran])),
           pd.DataFrame({"summary_json": [json.dumps(v) for _, v in ran], "seconds": [v.get("seconds") for _, v in ran]}))
    pk = pd.DataFrame([{"worker": n, **p} for n, w in plan["workers"].items() for p in w["packages"]])
    pk["bbox"] = pk.bbox.map(lambda b: ",".join(map(str, b)))
    pk_id = W.key("package", pk.worker, pk.package)
    W.nodes("Package", pk_id, pk)
    W.rels("FOR_WORKER", "Package", pk_id, "Worker", W.key("worker", pk.worker))
    members = []
    for n in plan["workers"]:
        for f_ in sorted((run_dir / "interpreter" / n).glob("package_*.csv.gz")):
            ids = pd.read_csv(f_, usecols=["row_id"]).row_id
            members.append(pd.DataFrame({"row_id": ids, "worker": n, "package": int(f_.stem.split("_")[1].split(".")[0])}))
    mem = pd.concat(members, ignore_index=True)
    W.rels("IN_PACKAGE", "Detection", W.key(mem.row_id), "Package", W.key("package", mem.worker, mem.package))
    ns = pd.DataFrame([{"worker": n, "field": fld, "reason": why} for n, w in plan["workers"].items() for fld, why in w["not_sent"].items()])
    W.rels("NOT_SENT", "Worker", W.key("worker", ns.worker), "Field", W.key("field", ns.field), ns[["reason"]])
    uses = pd.DataFrame([{"worker": n, "field": fld, "reason": w["why"].get(fld, "")}
                         for n, w in plan["workers"].items() for fld in w["fields"]])
    uses = uses[uses.field.isin(names)]
    W.rels("USES", "Worker", W.key("worker", uses.worker), "Field", W.key("field", uses.field), uses[["reason"]])

    # -- schema, import command, summary -----------------------------------------
    schema = {"graph": "one knowledge graph, fed by feed 1 (fire) and feed 2 (weather)", "run": run_id,
              "feeds": {"fire": {"script": "kg_builder.py", "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "node_files": W.files["nodes"], "relationship_files": W.files["relationships"]},
                        "weather": None},
              "node_files": W.files["nodes"], "relationship_files": W.files["relationships"]}
    (W.out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    cmd = ("neo4j-admin database import full neo4j --overwrite-destination "
           + " ".join(f"--nodes={W.out.name}/{v}" for v in W.files["nodes"].values()) + " "
           + " ".join(f"--relationships={W.out.name}/{v}" for v in W.files["relationships"].values()))
    summary = {"run": run_id, "counts": W.counts, "feeds": {"fire": True, "weather": False},
               "nodes": int(sum(W.counts[k] for k in W.files["nodes"])),
               "relationships": int(sum(W.counts[k] for k in W.files["relationships"])),
               "size_mb": round(sum(p.stat().st_size for p in W.out.glob("*.csv.gz")) / 1e6, 1),
               "neo4j_import": cmd, "seconds": round(time.time() - t0, 1)}
    (W.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def print_summary(s: dict, out: Path) -> None:
    print(f"\nFeed 1 (fire) for run {s['run']}: {s['nodes']:,} nodes, {s['relationships']:,} relationships, "
          f"{s['size_mb']} MB in {s['seconds']} s")
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    print("Nodes:")
    for k in schema["node_files"]:
        print(f"  {k:<14} {s['counts'][k]:>9,}")
    print("Relationships:")
    for k in schema["relationship_files"]:
        print(f"  {k:<18} {s['counts'][k]:>9,}")
    print("Every relationship end was checked against the node files.")
    print("Weather part: not added yet; run kg_weather_feed.py after the weather join.")


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
    s = build(run_dir)
    print_summary(s, run_dir / "kg")
    print(f"\nSaved to {run_dir / 'kg'}\nLoad into Neo4j (from {run_dir}):\n  {s['neo4j_import']}")


if __name__ == "__main__":
    main()
