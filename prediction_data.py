"""Prediction engine, step A1-A4: build the training table from the knowledge graph.

Reads only the graph files (<run>/kg/) of runs that have the weather part, so training uses
exactly what the engine will see at prediction time. One row = one fire on one day d on which
it was seen; features describe the fire up to the end of day d plus the weather of days d..d+3;
targets are what happened on days d+1, d+2, d+3.

Features (per fire-day)
  history   footprint so far (VIIRS-only cumulative, km2), new area on d, d-1, d-2; FRP (mean and peak
            of VIIRS passes) on d and d-1; pixels, passes, clusters alive, front distance; age;
            movement of the fire centre from d-1 and d-2 (east / north km); merges so far
  place     latitude, |latitude|, day of year (sin / cos), region, elevation, 10-year fire frequency
            of the 0.1 degree cell in that month (share of years with fire, mean fire days)
  weather   per day d..d+3 at the fire's weather cell: max / mean temperature, min / mean humidity,
            max VPD, mean / max wind, max gust, mean wind east / north, rain, soil moisture 0-7 /
            7-28 / 28-100 cm, hours since rain (ERA5 observed in training; the engine feeds the forecast)
Targets (k = 1, 2, 3)
  growth_k      footprint at the end of d+k minus at the end of d (km2)
  frp_k         mean FRP of the VIIRS passes on d+k (rows without a VIIRS pass that day are not used)
  frp_peak_k    peak VIIRS pass FRP on d+k
  dx_k, dy_k    movement of the FRP-weighted fire centre from d to d+k (km east / north)
  out_k         1 if the fire is never seen again from day d+k on
  merge_k       1 if, on days d+1..d+k, it merged with a separate neighbouring fire: a lasting cluster
                merge whose parents came from different source fires
Rows whose day d+3 lies after the end of the run are dropped (their future is unknown).

Usage:  python prediction_data.py            -> data/prediction/fire_days.csv.gz
"""

import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from firms import DATA_DIR
from fire_runs import region_of
from weather_worker import weather_key

OUT = DATA_DIR / "prediction"
H = 3
KM_LAT, KM_LON = 110.574, 111.320
REGIONS = ["boreal North America", "western North America", "eastern North America", "Siberia and boreal Asia",
           "Europe and Mediterranean", "Middle East and Central Asia", "South Asia", "East Asia",
           "Southeast Asia and Indonesia", "northern Australia", "southern Australia", "Central America and Caribbean",
           "Amazon and northern South America", "southern South America", "North Africa savanna", "southern Africa", "other"]
WX = ["tmax", "tmean", "rh_min", "rh_mean", "vpd_max", "wind_mean", "wind_max", "gust_max", "u_mean", "v_mean",
      "rain", "sm0", "sm1", "sm2", "since_rain"]


def strip(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: "id" if c.startswith("id:") else c.split(":")[0])


def read(kg: Path, name: str, cols=None) -> pd.DataFrame | None:
    p = kg / f"{name}.csv.gz"
    if not p.exists():
        return None
    usecols = None
    if cols:
        with __import__("gzip").open(p, "rt", encoding="utf-8") as fh:
            head = fh.readline().strip().split(",")
        usecols = [h for h in head if ("id" if h.startswith("id:") else h.split(":")[0]) in cols]
    # ids like "0.10" must stay text (as numbers they would read as 0.1 and stop matching)
    text = {h: str for h in (usecols or []) if h.split(":")[0] in TEXT_COLS}
    if usecols is None:
        with __import__("gzip").open(p, "rt", encoding="utf-8") as fh:
            text = {h: str for h in fh.readline().strip().split(",") if h.split(":")[0] in TEXT_COLS}
    return strip(pd.read_csv(p, usecols=usecols, dtype=text, low_memory=False))


TEXT_COLS = {"cluster_id", "sources", "clusters", "parents", "children"}


# ---------------------------------------------------------------------------
# Daily weather per weather cell (same function the engine uses on forecasts)
# ---------------------------------------------------------------------------

def daily_weather(hours: pd.DataFrame) -> pd.DataFrame:
    """hourly rows (weather_cell, time, temperature_2m, ...) -> one row per cell and UTC day."""
    h = hours.copy()
    h["day"] = pd.to_datetime(h.time, utc=True).dt.strftime("%Y-%m-%d")
    if "wind_u" not in h:
        rad = np.radians(h.wind_direction_10m)
        h["wind_u"], h["wind_v"] = -h.wind_speed_10m * np.sin(rad), -h.wind_speed_10m * np.cos(rad)
    g = h.groupby(["weather_cell", "day"])
    d = pd.DataFrame({
        "tmax": g.temperature_2m.max(), "tmean": g.temperature_2m.mean(),
        "rh_min": g.relative_humidity_2m.min(), "rh_mean": g.relative_humidity_2m.mean(),
        "vpd_max": g.vapour_pressure_deficit.max(), "wind_mean": g.wind_speed_10m.mean(), "wind_max": g.wind_speed_10m.max(),
        "gust_max": g.wind_gusts_10m.max(), "u_mean": g.wind_u.mean(), "v_mean": g.wind_v.mean(),
        "rain": g.precipitation.sum(min_count=1),
        "sm0": g.soil_moisture_0_to_7cm.mean() if "soil_moisture_0_to_7cm" in h else np.nan,
        "sm1": g.soil_moisture_7_to_28cm.mean() if "soil_moisture_7_to_28cm" in h else np.nan,
        "sm2": g.soil_moisture_28_to_100cm.mean() if "soil_moisture_28_to_100cm" in h else np.nan,
        "since_rain": g.hours_since_rain.last() if "hours_since_rain" in h else np.nan,
    }).reset_index()
    return d


def nearest_cell(cells: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, max_deg: float = 0.5) -> np.ndarray:
    """Weather cell for each point: its own cell if fetched, else the nearest fetched cell within max_deg."""
    if cells is None or cells.empty:
        return np.full(len(lat), None, dtype=object)
    own = weather_key(pd.Series(lat), pd.Series(lon)).to_numpy()
    have = set(cells.weather_cell)
    out = own.astype(object)
    miss = ~np.isin(own, list(have))
    if miss.any():
        cl, co = cells.lat.to_numpy(), cells.lon.to_numpy()
        for i in np.flatnonzero(miss):
            d = np.hypot(cl - lat[i], (co - lon[i]) * math.cos(math.radians(lat[i])))
            k = int(d.argmin())
            out[i] = cells.weather_cell.iloc[k] if d[k] <= max_deg else None
    return out


# ---------------------------------------------------------------------------
# Fire-days of one run
# ---------------------------------------------------------------------------

def fire_days(run_dir: Path, with_targets: bool = True, keep_all: bool = False) -> pd.DataFrame:
    """Daily state of every fire in a run. keep_all: every sighted fire-day, also the last H days of the run
    (needed to link fires into chains before the targets are worked out)."""
    kg = run_dir / "kg"
    fs = read(kg, "nodes_FireState", ["fire_id", "time", "family", "pixels", "frp_sum_mw", "cumulative_fine_km2",
                                      "cumulative_footprint_km2", "clusters_alive", "clusters_seen", "front_distance_km"])
    fires = read(kg, "nodes_Fire", ["fire_id", "latitude", "longitude", "first_seen", "last_seen"])
    if fs is None or fires is None or fs.empty:
        return pd.DataFrame()
    fs["day"] = pd.to_datetime(fs.time, utc=True).dt.strftime("%Y-%m-%d")
    run_last = fs.day.max()

    # daily fire state
    v = fs[fs.family == "VIIRS"]
    g = fs.groupby(["fire_id", "day"])
    day = pd.DataFrame({"fine_fp": g.cumulative_fine_km2.max(), "all_fp": g.cumulative_footprint_km2.max(),
                        "pixels": g.pixels.sum(), "passes": g.size(), "clusters_alive": g.clusters_alive.max(),
                        "clusters_seen": g.clusters_seen.max(), "front": g.front_distance_km.max()})
    gv = v.groupby(["fire_id", "day"]).frp_sum_mw
    day = day.join(pd.DataFrame({"frp": gv.mean(), "frp_peak": gv.max()}))
    day = day.reset_index()

    # FRP-weighted fire centre per day, from detections (cluster ids carry the fire id)
    det = read(kg, "nodes_Detection", ["id", "time", "latitude", "longitude", "frp_mw"])
    inc = pd.read_csv(kg / "rels_IN_CLUSTER.csv.gz")
    inc.columns = ["det", "cl", "type"]
    inc["fire_id"] = inc.cl.str.rsplit("|", n=1).str[-1].str.split(".").str[0].astype(int)
    inc["cluster_id"] = inc.cl.str.rsplit("|", n=1).str[-1]
    det = det.merge(inc[["det", "fire_id", "cluster_id"]], left_on="id", right_on="det")
    det["day"] = pd.to_datetime(det.time, utc=True).dt.strftime("%Y-%m-%d")
    det["w"] = det.frp_mw.fillna(0).clip(lower=0) + 0.1
    det["wlat"], det["wlon"] = det.latitude * det.w, det.longitude * det.w
    c = det.groupby(["fire_id", "day"])[["wlat", "wlon", "w"]].sum()
    cen = pd.DataFrame({"clat": c.wlat / c.w, "clon": c.wlon / c.w}).reset_index()
    day = day.merge(cen, on=["fire_id", "day"], how="left")
    day = day.merge(separate_parts(det, read(kg, "nodes_Cluster", ["cluster_id", "sources"])), on=["fire_id", "day"], how="left")

    # merges with separate neighbouring fires (lasting cluster merges whose parents had different source fires)
    merges = pd.DataFrame(columns=["fire_id", "day"])
    ev = read(kg, "nodes_ExpansionEvent", ["event", "clusters", "undone_within_2h", "time"])
    cl = read(kg, "nodes_Cluster", ["cluster_id", "sources"])
    if ev is not None and cl is not None and len(ev):
        src = dict(zip(cl.cluster_id.astype(str), cl.sources.astype(str)))
        rows = []
        for e in ev[ev.event.isin(["merge", "merge and split"]) & ~ev.undone_within_2h.astype(str).eq("True")].itertuples():
            parents = str(e.clusters).split(" -> ")[0].split("+")
            sets = [set(src.get(p, "").split("+")) for p in parents]
            if len(sets) >= 2 and any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))):
                rows.append((int(parents[0].split(".")[0]), str(e.time)[:10]))
        merges = pd.DataFrame(rows, columns=["fire_id", "day"]).drop_duplicates()

    # expand to every calendar day of each fire's life (gap days carry the footprint forward)
    fires["first_day"] = fires.first_seen.astype(str).str[:10]
    fires["last_day"] = fires.last_seen.astype(str).str[:10]
    f = fires.drop_duplicates("fire_id").set_index("fire_id")
    span = day.groupby("fire_id").day.max().rename("seen_last").to_frame().join(f[["first_day", "last_day"]], how="inner")
    span["end"] = np.maximum(span.last_day, span.seen_last)
    start, end = pd.to_datetime(span.first_day), pd.to_datetime(span.end)
    n = ((end - start).dt.days + 1).clip(lower=0).to_numpy()
    if n.sum() == 0:
        return pd.DataFrame()
    fid = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = pd.DataFrame({"fire_id": fid,
                      "day": (np.repeat(start.to_numpy(), n) + pd.to_timedelta(offs, unit="D")).strftime("%Y-%m-%d")})
    t = t.merge(day, on=["fire_id", "day"], how="left")
    g0 = t.groupby("fire_id")
    t["seen"] = t.pixels.fillna(0) > 0
    t["fine_fp"] = g0.fine_fp.ffill().fillna(0)
    t["all_fp"] = g0.all_fp.ffill().fillna(0)
    t["new_area"] = t.groupby("fire_id").fine_fp.diff().fillna(t.fine_fp)
    mk = merges.assign(merge=1).astype({"fire_id": t.fire_id.dtype})
    t = t.merge(mk, on=["fire_id", "day"], how="left")
    t["merge"] = t["merge"].fillna(0).astype(int)
    t["merges_so_far"] = t.groupby("fire_id")["merge"].cumsum() - t["merge"]
    t = t.join(f[["latitude", "longitude", "last_day"]].rename(columns={"latitude": "lat0", "longitude": "lon0"}), on="fire_id")
    t["run"] = run_dir.name
    t["run_last"] = run_last

    # history features from earlier days of the same fire
    t = t.sort_values(["fire_id", "day"], ignore_index=True)
    g = t.groupby("fire_id")
    for lag in (1, 2):
        t[f"new_area_l{lag}"] = g.new_area.shift(lag)
        t[f"frp_l{lag}"] = g.frp.shift(lag)
        lat_l, lon_l = g.clat.shift(lag), g.clon.shift(lag)
        t[f"dx_l{lag}"] = (t.clon - lon_l) * KM_LON * np.cos(np.radians(t.lat0))
        t[f"dy_l{lag}"] = (t.clat - lat_l) * KM_LAT
    t["age_days"] = g.cumcount()
    t = add_neighbours(t)

    # targets for d+1..d+3
    if with_targets:
        for k in range(1, H + 1):
            fut = g.shift(-k)
            t[f"growth_{k}"] = fut.fine_fp - t.fine_fp
            t[f"frp_{k}"] = fut.frp
            t[f"frp_peak_{k}"] = fut.frp_peak
            t[f"dx_{k}"] = (fut.clon - t.clon) * KM_LON * np.cos(np.radians(t.lat0))
            t[f"dy_{k}"] = (fut.clat - t.clat) * KM_LAT
            t[f"merge_{k}"] = sum(g["merge"].shift(-j).fillna(0) for j in range(1, k + 1)).clip(upper=1)
            plus = (pd.to_datetime(t.day) + pd.Timedelta(days=k)).dt.strftime("%Y-%m-%d")
            t[f"out_{k}"] = (t.last_day < plus).astype(int)
            # past the fire's last day there are no rows: it no longer grows
            t[f"growth_{k}"] = t[f"growth_{k}"].fillna(0)
        horizon = (pd.to_datetime(t.day) + pd.Timedelta(days=H)).dt.strftime("%Y-%m-%d")
        t = t[t.seen] if keep_all else t[t.seen & (horizon <= t.run_last)]
    else:
        t = t[t.seen]
    return t


def separate_parts(det: pd.DataFrame, cl: pd.DataFrame | None) -> pd.DataFrame:
    """Per fire and day: how many separate parts it has (groups of clusters that share no source fire, i.e. fires
    that have not joined yet) and the smallest gap between pixels of two different parts (km). Merges of separate
    fires happen between such parts."""
    cols = ["fire_id", "day", "sep_parts", "sep_gap_km"]
    if cl is None or det.empty:
        return pd.DataFrame(columns=cols)
    src = dict(zip(cl.cluster_id.astype(str), cl.sources.astype(str)))
    rows = []
    multi = det.groupby(["fire_id", "day"]).cluster_id.nunique()
    one = multi[multi < 2]
    rows += [(f, d, 1, np.nan) for (f, d) in one.index]
    sub = det.set_index(["fire_id", "day"]).loc[multi[multi >= 2].index].reset_index() if (multi >= 2).any() else det.iloc[0:0]
    for (f, d), g in sub.groupby(["fire_id", "day"]):
        cids = g.cluster_id.unique()
        parent = {c: c for c in cids}
        def root(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c
        owner = {}
        for c in cids:                                  # clusters sharing a source fire belong to the same part
            for s_ in src.get(c, c).split("+"):
                if s_ in owner:
                    parent[root(c)] = root(owner[s_])
                else:
                    owner[s_] = c
        lab = g.cluster_id.map(lambda c: root(c)).to_numpy()
        parts = len(set(lab))
        gap = np.nan
        if parts >= 2:
            lat0 = math.radians(g.latitude.mean())
            xy = np.c_[g.longitude.to_numpy() * KM_LON * math.cos(lat0), g.latitude.to_numpy() * KM_LAT]
            kq = min(len(g), 12)
            dd, jj = cKDTree(xy).query(xy, k=kq)
            diff = lab[jj] != lab[:, None]
            cand = np.where(diff, dd, np.inf).min()
            if not np.isfinite(cand):                     # no other part among the nearest pixels: compare part by part
                cand = min(cKDTree(xy[lab == a]).query(xy[lab != a], k=1)[0].min() for a in set(lab))
            gap = float(cand)
        rows.append((f, d, parts, gap))
    return pd.DataFrame(rows, columns=cols)


NB_COLS = ["nb_dist_km", "nb_gap_km", "nb_fp", "nb_new_area", "nb_frp", "nb_n10", "nb_n25", "nb_bearing"]


def add_neighbours(t: pd.DataFrame) -> pd.DataFrame:
    """Nearest other fire seen on the same day: distance between centres, gap between their footprints (as circles),
    its footprint, growth and power, how many fires are within 10 and 25 km, and the bearing towards it."""
    for c in NB_COLS:
        t[c] = np.nan
    lat = t.clat.fillna(t.lat0).to_numpy()
    lon = t.clon.fillna(t.lon0).to_numpy()
    r = np.sqrt(t.fine_fp.fillna(0).to_numpy() / math.pi)
    seen = t.seen.to_numpy()
    R_EARTH = 6371.0
    for day, idx in t[seen].groupby("day").indices.items():
        rows = np.flatnonzero(seen)[idx]
        if len(rows) < 2:
            continue
        la, lo = np.radians(lat[rows]), np.radians(lon[rows])
        xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * R_EARTH
        kq = min(6, len(rows))
        d, j = cKDTree(xyz).query(xyz, k=kq)
        d, j = d[:, 1:], j[:, 1:]
        nbr = rows[j[:, 0]]
        t.iloc[rows, t.columns.get_loc("nb_dist_km")] = d[:, 0]
        t.iloc[rows, t.columns.get_loc("nb_gap_km")] = np.maximum(0, d[:, 0] - r[rows] - r[nbr])
        for src, dst in (("fine_fp", "nb_fp"), ("new_area", "nb_new_area"), ("frp", "nb_frp")):
            t.iloc[rows, t.columns.get_loc(dst)] = t[src].to_numpy()[nbr]
        t.iloc[rows, t.columns.get_loc("nb_n10")] = (d <= 10).sum(axis=1)
        t.iloc[rows, t.columns.get_loc("nb_n25")] = (d <= 25).sum(axis=1)
        dx = (lon[nbr] - lon[rows]) * KM_LON * np.cos(np.radians(lat[rows]))
        dy = (lat[nbr] - lat[rows]) * KM_LAT
        t.iloc[rows, t.columns.get_loc("nb_bearing")] = np.degrees(np.arctan2(dx, dy)) % 360
    return t


def add_place_and_weather(t: pd.DataFrame, run_dir: Path, weather_daily: pd.DataFrame | None, cells: pd.DataFrame | None,
                          freq: pd.DataFrame | None) -> pd.DataFrame:
    """Place features, plus weather for days d..d+3 at the fire's weather cell (observed or forecast)."""
    t = t.copy()
    lat, lon = t.lat0.to_numpy(), t.lon0.to_numpy()
    t["abs_lat"] = np.abs(lat)
    doy = pd.to_datetime(t.day).dt.dayofyear
    t["doy_sin"], t["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    t["region"] = [REGIONS.index(region_of(a, b)) for a, b in zip(lat, lon)]
    t["month"] = pd.to_datetime(t.day).dt.month
    if freq is not None:
        t["fi"], t["fj"] = np.floor(lat / 0.1).astype(int), np.floor(lon / 0.1).astype(int)
        t = t.merge(freq, left_on=["fi", "fj", "month"], right_on=["i", "j", "month"], how="left").drop(columns=["i", "j", "fi", "fj"])
        t["risk_share"] = t.risk_share.fillna(0)
        t["risk_days"] = t.risk_days.fillna(0)
    t["weather_cell"] = nearest_cell(cells, lat, lon) if cells is not None else None
    if cells is not None and "elevation_m" in cells:
        t = t.merge(cells[["weather_cell", "elevation_m"]], on="weather_cell", how="left")
    else:
        t["elevation_m"] = np.nan
    if weather_daily is not None and len(weather_daily):
        for k in range(0, H + 1):
            dk = (pd.to_datetime(t.day) + pd.Timedelta(days=k)).dt.strftime("%Y-%m-%d")
            w = weather_daily.rename(columns={c: f"w{k}_{c}" for c in WX}).rename(columns={"day": "_day"})
            t = t.assign(_day=dk).merge(w, on=["weather_cell", "_day"], how="left").drop(columns=["_day"])
    else:
        for k in range(0, H + 1):
            for c in WX:
                t[f"w{k}_{c}"] = np.nan
    if "nb_bearing" in t:
        # wind component blowing from this fire towards its nearest neighbour (km/h; negative = away from it)
        b = np.radians(t.nb_bearing.to_numpy())
        for k in range(1, H + 1):
            t[f"w{k}_wind_to_nb"] = t[f"w{k}_u_mean"].to_numpy() * np.sin(b) + t[f"w{k}_v_mean"].to_numpy() * np.cos(b)
    return t


def load_freq() -> pd.DataFrame | None:
    p = DATA_DIR / "archive" / "frequency.csv.gz"
    if not p.exists():
        return None
    f = pd.read_csv(p, usecols=["i", "j", "month", "share_of_years", "mean_fire_days"])
    return f.rename(columns={"share_of_years": "risk_share", "mean_fire_days": "risk_days"})


CHAINS = False          # link fires that vanish and reappear into chains (prediction_labels.py)


def build_run(run_dir: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    if CHAINS:
        from prediction_labels import relabel
        t = fire_days(run_dir, keep_all=True)
        if t.empty:
            return t
        t = relabel(t, with_targets=True)
        horizon = (pd.to_datetime(t.day) + pd.Timedelta(days=H)).dt.strftime("%Y-%m-%d")
        t = t[horizon <= t.run_last]
    else:
        t = fire_days(run_dir)
    if t.empty:
        return t
    kg = run_dir / "kg"
    hours = read(kg, "nodes_WeatherHour")
    cells = read(kg, "nodes_WeatherCell")
    wd = daily_weather(hours) if hours is not None and len(hours) else None
    return add_place_and_weather(t, run_dir, wd, cells, FREQ)


FREQ = None


def _init(chains=False):
    global FREQ, CHAINS
    FREQ = load_freq()
    CHAINS = chains


HIST_COLS = [f"h{j}_{c}" for j in range(1, 5) for c in ("new_area", "frp", "sep_parts", "back")] +             ["area_trend3", "frp_trend3", "seen_last7"]


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """The fire's previous (up to) 4 fire-days as features: growth, power, separate parts and how many days back,
    plus the mean growth of the last 3 fire-days, power against its 3-day mean, and fire-days seen in the last week."""
    d = df.copy()
    who = d.chain if "chain" in d else d.fire_id                 # a chain of fires that vanished and came back
    key = d.run.astype(str) + "|" + who.astype(str)
    order = np.lexsort((d.day.to_numpy(), key.to_numpy()))
    s = d.iloc[order]
    k = key.iloc[order].to_numpy()
    day = pd.to_datetime(s.day).to_numpy()
    out = {}
    for j in range(1, 5):
        same = np.zeros(len(s), bool)
        same[j:] = k[j:] == k[:-j]
        back = np.full(len(s), np.nan)
        back[j:] = (day[j:] - day[:-j]).astype("timedelta64[D]").astype(float)
        ok = same & (back <= 7)
        for c in ("new_area", "frp", "sep_parts"):
            v = np.full(len(s), np.nan)
            src = s[c].to_numpy("float64")
            v[j:] = src[:-j]
            out[f"h{j}_{c}"] = np.where(ok, v, np.nan)
        out[f"h{j}_back"] = np.where(ok, back, np.nan)
    h = pd.DataFrame(out, index=s.index)
    h["area_trend3"] = h[["h1_new_area", "h2_new_area", "h3_new_area"]].mean(axis=1)
    h["frp_trend3"] = s.frp - h[["h1_frp", "h2_frp", "h3_frp"]].mean(axis=1)
    h["seen_last7"] = h[[f"h{j}_back" for j in range(1, 5)]].notna().sum(axis=1)
    for c in HIST_COLS:
        d[c] = h[c].reindex(d.index)
    return d


def feature_columns(df: pd.DataFrame, weather: bool = True) -> list:
    base = ["fine_fp", "all_fp", "new_area", "new_area_l1", "new_area_l2", "frp", "frp_l1", "frp_peak", "pixels", "passes",
            "clusters_alive", "clusters_seen", "front", "age_days", "dx_l1", "dy_l1", "dx_l2", "dy_l2", "merges_so_far",
            "lat0", "abs_lat", "doy_sin", "doy_cos", "region", "elevation_m", "risk_share", "risk_days",
            "nb_dist_km", "nb_gap_km", "nb_fp", "nb_new_area", "nb_frp", "nb_n10", "nb_n25", "sep_parts", "sep_gap_km",
            *HIST_COLS]
    wx = [f"w{k}_{c}" for k in range(0, H + 1) for c in WX] + [f"w{k}_wind_to_nb" for k in range(1, H + 1)]
    static = STATIC_COLS + [f"w{k}_wind_upslope" for k in range(1, H + 1)]
    return [c for c in base + static + (wx if weather else []) if c in df]


# terrain (Copernicus DEM) and fuel (ESA WorldCover) around the fire, from static_features.py
STATIC_COLS = ["elev_mean", "elev_std", "slope_mean", "slope_p90", "downslope_e", "downslope_n",
               "lc1_tree", "lc1_shrub", "lc1_grass", "lc1_crop", "lc1_built", "lc1_bare", "lc1_snow", "lc1_water", "lc1_wetland",
               "lc1_mangrove", "lc1_moss", "lc5_tree", "lc5_shrub", "lc5_grass", "lc5_crop", "lc5_built", "lc5_bare", "lc5_snow",
               "lc5_water", "lc5_wetland", "lc5_mangrove", "lc5_moss"]


def add_static(df: pd.DataFrame, sample_missing: bool = False) -> pd.DataFrame:
    """Terrain and land-cover columns for each fire-day (cache of static_features.py; optionally sample new places),
    plus the wind component blowing uphill on days +1..+3 (fire runs faster uphill and with the wind)."""
    import static_features as S
    d = df.copy()
    for kind in ("terrain", "landcover"):
        F = S.lookup(d, kind, sample_missing=sample_missing)
        for c in F.columns:
            d[c] = F[c].to_numpy()
    for k in range(1, H + 1):
        if f"w{k}_u_mean" in d and "downslope_e" in d:
            # uphill = against the downslope direction; positive when the wind pushes the fire uphill
            d[f"w{k}_wind_upslope"] = -(d[f"w{k}_u_mean"] * d.downslope_e + d[f"w{k}_v_mean"] * d.downslope_n)
    return d


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", action="store_true", help="link vanishing and reappearing fires into chains")
    ap.add_argument("--out", default="fire_days.csv.gz")
    args = ap.parse_args()
    t0 = time.time()
    runs = [r for r in sorted((DATA_DIR / "runs").glob("*")) if (r / "kg" / "nodes_WeatherHour.csv.gz").exists()]
    print(f"{len(runs)} runs with the weather part", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(args.chains,)) as pool:
        for r, df in zip(runs, pool.map(build_run, map(str, runs))):
            print(f"  {r.name}: {len(df):,} fire-days", flush=True)
            parts.append(df)
    data = pd.concat([p for p in parts if len(p)], ignore_index=True)
    data["year"] = data.day.str[:4].astype(int)
    data["split"] = np.where(data.year <= 2023, "train", np.where(data.year == 2024, "check", "test"))
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / args.out, index=False)
    wx_cov = data["w1_tmax"].notna().mean()
    summary = {"rows": len(data), "fires": int(data.groupby(["run", "fire_id"]).ngroups),
               "split": {k: int(v) for k, v in data.split.value_counts().items()},
               "rows_with_weather_d+1": round(float(wx_cov), 3),
               "targets": {f"out_{k}": round(float(data[f"out_{k}"].mean()), 3) for k in range(1, H + 1)} |
                          {f"merge_{k}": round(float(data[f"merge_{k}"].mean()), 4) for k in range(1, H + 1)},
               "seconds": round(time.time() - t0, 1)}
    if "chain" in data:
        summary["chains"] = int(data.groupby(["run", "chain"]).ngroups)
        summary["out_unknown_censored"] = {f"out_{k}": round(float(data[f"out_{k}"].isna().mean()), 3) for k in range(1, H + 1)}
        summary["grew_0.1"] = {f"growth_{k}": round(float((data[f"growth_{k}"] >= 0.1).mean()), 3) for k in range(1, H + 1)}
    (OUT / args.out.replace(".csv.gz", "_summary.json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
