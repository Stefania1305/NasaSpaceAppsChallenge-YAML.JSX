"""Terrain and fuel around each fire: Copernicus DEM (elevation, slope, aspect, ruggedness) and ESA WorldCover (land cover).

Both are free cloud-optimised GeoTIFFs on AWS open data, read window by window over HTTP (no download of whole tiles):
  terrain  Copernicus DEM GLO-90 (read at its 180 m overview), within about 3 km of the fire: mean elevation, elevation spread (ruggedness),
           mean and 90th-percentile slope (degrees), and the dominant downslope direction as east / north parts
           (weighted by slope; fire usually runs uphill, against it)
           https://registry.opendata.aws/copernicus-dem/
  fuel     ESA WorldCover 2021 (10 m map, read at its 40 m overview within about 1 km and 320 m overview within 5 km),
           share of each land-cover class: tree, shrub, grass, crop, built-up, bare, snow, water, wetland, mangrove, moss
           https://registry.opendata.aws/esa-worldcover-vito/
Locations are rounded to 0.01 degree (about 1 km) and cached, so each place is read once; the prediction engine uses
the same functions (and cache) for live fires.

Usage:  python static_features.py terrain|landcover [--threads 32]     (all fire locations of fire_days_v3.csv.gz)
Output: data/static_features/terrain.csv.gz, landcover.csv.gz
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import rasterio  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402

from firms import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "static_features"
GRID = 0.01
DEM_URL = "/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
WC_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif"
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
DEM_DECIM = 2
TERRAIN_COLS = ["elev_mean", "elev_std", "slope_mean", "slope_p90", "downslope_e", "downslope_n"]
LC_COLS = [f"lc1_{v}" for v in WC_CLASSES.values()] + [f"lc5_{v}" for v in WC_CLASSES.values()]


def dem_tile(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    return DEM_URL.format(ns="N" if la >= 0 else "S", lat=abs(la), ew="E" if lo >= 0 else "W", lon=abs(lo))


def wc_tile(lat, lon):
    la, lo = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return WC_URL.format(ns="N" if la >= 0 else "S", lat=abs(la), ew="E" if lo >= 0 else "W", lon=abs(lo))


def terrain_at(ds, lat, lon, half=0.03):
    # window clipped to the tile (a "boundless" read builds a virtual raster per read and is ~200x slower)
    w = from_bounds(lon - half, lat - half, lon + half, lat + half, ds.transform).round_offsets().round_lengths()
    w = w.intersection(Window(0, 0, ds.width, ds.height))
    # read from the file's 2x overview (180 m): a fraction of the download, enough for slope and ruggedness
    shape = (max(3, int(w.height // DEM_DECIM)), max(3, int(w.width // DEM_DECIM)))
    z = ds.read(1, window=w, out_shape=shape).astype("float64")
    z[z < -1000] = np.nan
    if np.isnan(z).all() or z.size < 9:
        return [np.nan] * len(TERRAIN_COLS)
    px_n = abs(ds.res[1]) * w.height / shape[0] * 110574.0
    px_e = abs(ds.res[0]) * w.width / shape[1] * 111320.0 * math.cos(math.radians(lat))
    dzdy, dzdx = np.gradient(z, px_n, px_e)                      # rows go south, so -dzdy is the northward gradient
    slope = np.degrees(np.arctan(np.hypot(dzdx, -dzdy)))
    ok = ~np.isnan(slope)
    sw = np.where(ok, np.tan(np.radians(slope)), 0)
    # downslope direction: where the ground falls (east / north parts), weighted by steepness
    de, dn = -np.nan_to_num(dzdx), np.nan_to_num(dzdy)
    norm = np.hypot(de, dn) + 1e-9
    e = float((sw * de / norm).sum() / max(sw.sum(), 1e-9))
    n = float((sw * dn / norm).sum() / max(sw.sum(), 1e-9))
    return [float(np.nanmean(z)), float(np.nanstd(z)), float(np.nanmean(slope)), float(np.nanpercentile(slope[ok], 90)) if ok.any() else np.nan, e, n]


def landcover_at(ds, lat, lon):
    vals = []
    for half, decim in ((0.01, 4), (0.045, 32)):             # file overviews: 40 m within 1 km, 320 m within 5 km
        w = from_bounds(lon - half, lat - half, lon + half, lat + half, ds.transform).round_offsets().round_lengths()
        w = w.intersection(Window(0, 0, ds.width, ds.height))
        shape = (max(1, int(w.height // decim)), max(1, int(w.width // decim)))
        a = ds.read(1, window=w, out_shape=shape)
        valid = a > 0
        n = max(int(valid.sum()), 1)
        vals += [float((a == c).sum() / n) if valid.any() else np.nan for c in WC_CLASSES]
    return vals


def sample(points: pd.DataFrame, kind: str, threads: int = 32, log=print) -> pd.DataFrame:
    """points: columns plat, plon (rounded). Returns the same rows with the feature columns."""
    tile_of = dem_tile if kind == "terrain" else wc_tile
    fn = terrain_at if kind == "terrain" else landcover_at
    cols = TERRAIN_COLS if kind == "terrain" else LC_COLS
    pts = points.copy()
    pts["tile"] = [tile_of(a, b) for a, b in zip(pts.plat, pts.plon)]
    groups = list(pts.groupby("tile"))
    res = {}
    done = [0, time.time()]

    def work(tile, g):
        out = []
        try:
            with rasterio.Env(), rasterio.open(tile) as ds:
                for i, a, b in zip(g.index, g.plat, g.plon):
                    try:
                        out.append((i, fn(ds, a, b)))
                    except Exception:
                        out.append((i, [np.nan] * len(cols)))
        except Exception:                                   # no tile: ocean, or outside the data
            out = [(i, [np.nan] * len(cols)) for i in g.index]
        return out

    with ThreadPoolExecutor(threads) as ex:
        futs = [ex.submit(work, t, g) for t, g in groups]
        for k, f in enumerate(as_completed(futs)):
            for i, v in f.result():
                res[i] = v
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(groups):
                log(f"  {kind}: {done[0]}/{len(groups)} tiles, {len(res):,}/{len(pts):,} places ({time.time() - done[1]:.0f} s)")
    F = pd.DataFrame.from_dict(res, orient="index", columns=cols)
    return pts.drop(columns=["tile"]).join(F)


def place_key(lat, lon):
    return np.round(np.asarray(lat) / GRID) * GRID, np.round(np.asarray(lon) / GRID) * GRID


def lookup(df: pd.DataFrame, kind: str, sample_missing: bool = True, threads: int = 16) -> pd.DataFrame:
    """Feature columns for the fires in df (uses the cache; samples places not in it). Used by the engine."""
    lat = df.clat.fillna(df.lat0) if "clat" in df else df.lat0
    lon = df.clon.fillna(df.lon0) if "clon" in df else df.lon0
    plat, plon = place_key(lat, lon)
    keys = pd.DataFrame({"plat": plat.round(2), "plon": plon.round(2)})
    path = OUT / f"{kind}.csv.gz"
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["plat", "plon"])
    cache["plat"], cache["plon"] = cache.plat.round(2), cache.plon.round(2)
    need = keys.drop_duplicates().merge(cache[["plat", "plon"]], how="left", indicator=True)
    need = need[need._merge == "left_only"][["plat", "plon"]].reset_index(drop=True)
    if sample_missing and len(need):
        new = sample(need, kind, threads, log=lambda m: None)
        cache = pd.concat([cache, new], ignore_index=True)
        OUT.mkdir(parents=True, exist_ok=True)
        cache.to_csv(path, index=False)
    cols = TERRAIN_COLS if kind == "terrain" else LC_COLS
    return keys.merge(cache.drop_duplicates(["plat", "plon"]), on=["plat", "plon"], how="left")[cols].set_index(df.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["terrain", "landcover"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--table", default="fire_days_v3.csv.gz")
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(DATA_DIR / "prediction" / args.table, usecols=["clat", "clon", "lat0", "lon0"])
    plat, plon = place_key(df.clat.fillna(df.lat0), df.clon.fillna(df.lon0))
    pts = pd.DataFrame({"plat": plat.round(2), "plon": plon.round(2)}).drop_duplicates().reset_index(drop=True)
    print(f"{len(pts):,} places ({args.kind}), {args.threads} threads", flush=True)
    out = sample(pts, args.kind, args.threads, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"{args.kind}.csv.gz", index=False)
    cols = TERRAIN_COLS if args.kind == "terrain" else LC_COLS
    print(f"saved {len(out):,} places, {out[cols[0]].notna().mean() * 100:.1f}% with data, in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
