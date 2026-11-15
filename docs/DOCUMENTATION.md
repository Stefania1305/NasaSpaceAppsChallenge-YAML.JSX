# Fire Pattern Predictor: technical documentation

**From raw NASA satellite fire detections to tracked fires, a knowledge graph and a calibrated 3-day forecast of every active fire, explored on an interactive 3D globe.**

Repository: <https://github.com/Stefania1305/NasaSpaceAppsChallenge-YAML.JSX> · Overview and quick start: [README](../README.md) · Launch video: [docs/media/fire-pattern-predictor.mp4](media/fire-pattern-predictor.mp4)

---

## Contents

1. [Team](#1-team)
2. [The problem and what the system does](#2-the-problem-and-what-the-system-does)
3. [System architecture](#3-system-architecture)
4. [Data ingestion and harmonisation](#4-data-ingestion-and-harmonisation)
5. [The workers (tracking pipeline)](#5-the-workers-tracking-pipeline)
6. [The knowledge graph](#6-the-knowledge-graph)
7. [The 10-year archive and training data](#7-the-10-year-archive-and-training-data)
8. [AI prediction engine](#8-ai-prediction-engine)
9. [Backend API](#9-backend-api)
10. [Frontend](#10-frontend)
11. [Metrics](#11-metrics)
12. [Testing](#12-testing)
13. [Limitations](#13-limitations)
14. [External data, services and libraries (with links)](#14-external-data-services-and-libraries)
15. [Credits and attribution](#15-credits-and-attribution)

---

## 1. Team

**YAML.JSX** is a high-octane engineering crew from UNITBV encapsulating the full spectrum of tech evolution. We connect the fresh creativity of Bogdan Diaconiță (1st Yr CE) and sharp drive of Cezar Sava (2nd Yr CE) with the battle-tested development architecture of Ștefania Roman and Adelin Olteanu (3rd Yr CE). The matrix is anchored by Valentin Floricescu, an Applied Electronics alumnus and PhD researcher. From freshman spark to doctoral depth, we deliver flawless execution.

| Member | Background | Role |
|---|---|---|
| Bogdan Diaconiță | 1st year, Computer Engineering, UNITBV | Frontend |
| Cezar Sava | 2nd year, Computer Engineering, UNITBV | Full stack |
| Ștefania Roman | 3rd year, Computer Engineering, UNITBV | Full stack |
| Adelin Olteanu | 3rd year, Computer Engineering, UNITBV | Full stack |
| Valentin Floricescu | Computer Engineering and Applied Electronics engineer, PhD researcher | Full stack |

---

## 2. The problem and what the system does

NASA's [FIRMS](https://firms.modaps.eosdis.nasa.gov/) publishes every hot spot that the MODIS, VIIRS and Landsat instruments detect, within hours of each satellite pass. Those are **points**, not **fires**: five satellites with different pixel sizes (30 m to about 5 km), different detection limits and overpass times report the same fire many times, miss it under clouds, and say nothing about where it is heading.

Fire Pattern Predictor turns that stream into answers a fire manager can act on:

| Question | Where it is answered |
|---|---|
| Which detections are the same fire, across satellites and passes? | Correlation layer, localisation worker |
| Where and when did each fire start? | Epicenter worker |
| How is it growing: ignitions, new bursts, merges, splits, extinction? | Fire expansion worker |
| How much energy does it release, and is it intensifying? | Fire power worker |
| What weather is it burning in? | Weather worker |
| How will it grow, move and burn over the next 3 days? Will it go out? Will it merge with a neighbour? Which 1 km cells will burn? | AI prediction engine |
| How do I see all of this for a region? | React frontend with a 3D globe |

Every intermediate result is kept in a **knowledge graph** (Neo4j bulk-import format), so each prediction can be traced back to the satellite pixels and the weather hours it came from.

---

## 3. System architecture

```
                        ┌────────────────────────── NASA FIRMS (MODIS · VIIRS S-NPP / NOAA-20 / NOAA-21 · Landsat) ─┐
                        │                                                                                          │
                firms.py (API client, key rotation)                                       archive_scan.py (10 years, SP + NRT)
                        │                                                                                          │
        Harmonisation: comparison.py · type_inference.py · swath.py                         fire_runs.py (2,000 fires, 40 batches)
                        │                                                                                          │
   ┌────────────────────┴───────────────────── run_pipeline.py ───────────────────────────────────────┐              │
   │ 1 correlation  →  2 interpreter  →  3a localisation ┐                                          │              │
   │                                     3b fire power   ┘→ 4 epicenter → 4b fire expansion         │ ◄────────────┘
   │                                                                          │                     │
   │                                                              KG feed 1 (kg_builder.py)          │
   │                                                                          │                     │
   │                                     Open-Meteo ──► 5 weather worker ──► KG feed 2 (kg_weather_feed.py)
   └──────────────────────────────────────────────────────────────────────────┬─────────────────────┘
                                                                              │  <run>/kg/  (one graph, two feeds)
                       Copernicus DEM, ESA WorldCover (AWS) ─► static_features.py
                                                                              ▼
                                  prediction_data.py + prediction_labels.py (fire chains) → training table
                                                                              ▼
                     prediction_v3.py (XGBoost, CUDA)   prediction_seq.py (GRU network, PyTorch)   prediction_spread.py
                                                                              ▼
                                         prediction_engine.py  ── blend → predictions.csv / spread.csv / merge_outlook.csv
                                                                              ▼
                               server.py (HTTP API) ──► frontend/ (React · MapLibre globe) ◄── weather.py (Open-Meteo forecast)
```

**Design principles**

- **Files between stages.** Each stage reads the previous stage's files in the run folder (`data/runs/<run>/<stage>/`) and writes its own. Any stage can be rerun alone (`--start-at`), and stage timings go to `timings.json`.
- **Parallel where it is safe.** The interpreter cuts the data into work packages that never split a fire complex, so localisation and fire power run in parallel processes, and side by side.
- **Same code for training and live use.** The training table is built only from the knowledge graph, and the engine builds the same features from the same graph at prediction time.
- **No data leakage.** The models train on 2016–2023, use 2024 only to stop training and calibrate, and are tested on 2025–2026.

**Main modules**

| Module | Role |
|---|---|
| `firms.py` | Shared NASA FIRMS API client; reads `FIRMS_MAP_KEY` (or several `FIRMS_MAP_KEYS`, rotated when one reaches its quota) from `.env` |
| `comparison.py`, `type_inference.py`, `swath.py`, `build_type_reference.py` | Sensor harmonisation |
| `correlation.py`, `interpreter.py`, `localisation_worker.py`, `fire_power_worker.py`, `epicenter_worker.py`, `fire_expansion_worker.py`, `weather_worker.py` | The pipeline workers |
| `kg_builder.py`, `kg_weather_feed.py` | Knowledge graph feeds 1 and 2 |
| `run_pipeline.py` | Runs and times all stages for a date range and area |
| `archive_scan.py`, `fire_runs.py`, `weather_fires.py` | 10-year archive, fire selection, weather for training |
| `prediction_data.py`, `prediction_labels.py`, `forecast_errors.py`, `static_features.py` | Training table, fire chains, forecast-error pool, terrain and fuel |
| `prediction_v3.py`, `prediction_seq.py`, `seq_numpy.py`, `prediction_spread.py` | Model training (trees, network, spread map) |
| `prediction_engine.py` | Live prediction for a run |
| `prediction_metrics.py`, `test_prediction_accuracy.py`, `test_small_fires.py` | Evaluation |
| `merges_dataset.py`, `prediction_animation.py` | Merge dataset, GIF animations |
| `server.py`, `weather.py`, `frontend/` | Backend API, forecast weather for the app, React app |
| `backtest/backtest.py` | End-to-end check of the app against later real detections |

---

## 4. Data ingestion and harmonisation

### 4.1 FIRMS sources

| Source | Instrument and platform | Pixel | Used for |
|---|---|---|---|
| `MODIS_NRT`, `MODIS_SP` | MODIS on Terra and Aqua | 1 km (up to about 4.8 × 2 km at the swath edge) | Pipeline, archive, 10-year trend |
| `VIIRS_SNPP_*`, `VIIRS_NOAA20_*`, `VIIRS_NOAA21_*` | VIIRS on Suomi NPP, NOAA-20, NOAA-21 | 375 m | Pipeline, archive, footprints, training targets |
| `LANDSAT_NRT` | OLI on Landsat 8 and 9 | 30 m | Localisation and expansion (no FRP) |

`SP` (standard processing) files are the definitive archive; `NRT` (near real time) covers the latest months. Downloads use the FIRMS [area API](https://firms.modaps.eosdis.nasa.gov/api/area/) and [data availability API](https://firms.modaps.eosdis.nasa.gov/api/data_availability/).

### 4.2 Making the sensors comparable (`comparison.py`)

| Sensor difference | How it is resolved |
|---|---|
| Pixel size (MODIS about 1 km, VIIRS 375 m) | Both aggregated onto a common grid (0.05°), compared cell by cell |
| FRP split (VIIRS spreads one fire over several sub-pixel rows) | FRP summed per cell, so each fire's energy is counted once |
| Raw counts (VIIRS 4–6× more pixels) | Fires counted as occupied cells, not pixels |
| Detection limit (VIIRS sees much smaller fires) | One common minimum FRP, calibrated from the data, applied to both |
| Confidence (MODIS 0–100, VIIRS low/nominal/high) | Mapped to low / nominal / high (MODIS < 30, 30–79, ≥ 80, per FIRMS docs) |
| Saturation (MODIS about 500 K, VIIRS I4 clips at 367 K) | Brightness kept at full precision, not used to decide comparability |
| Overpass times | Aqua compared with VIIRS on a similar schedule, day with day, night with night; gap reported |
| MODIS swath gaps near the equator | Each satellite's ground tracks reconstructed (`swath.py`); a fire seen by one sensor where the other did not look is left out |

### 4.3 Inferring the `type` field (`type_inference.py`)

Only SP files carry `type` (0 vegetation fire, 1 volcano, 2 other static land source such as gas flares, 3 offshore). FIRMS assigns it by location, so SP detections are binned onto a fine grid; an unlabelled detection takes its cell's majority type, then its neighbours', then "vegetation fire". The result carries a `type_confidence`.

### 4.4 Where a satellite actually looked (`swath.py`)

FIRMS lists only fire pixels, never "looked and saw nothing". A sun-synchronous orbit of one day is fixed by two numbers, the orbit phase and the local solar time of the ascending node. Both are estimated from the satellite's own detections (latitude and UTC time give the phase; the `scan` pixel size gives the distance from the ground track). With the tracks known, any location can be tested for coverage. This lets the epicenter and expansion workers tell "the satellite passed and saw no fire" from "no data".

---

## 5. The workers (tracking pipeline)

```
correlation → interpreter → localisation + fire power → epicenter → fire expansion → KG feed 1 → weather join → KG feed 2
```

### 5.1 Correlation layer (`correlation.py`, stage 1)

Measures, on each dataset and per instrument scope (all, MODIS only, VIIRS only, ...):

- **Field relationships:** Spearman rank correlation (numeric–numeric), correlation ratio η on ranks (categorical–numeric), bias-corrected Cramér's V (categorical–categorical). Pairs related by construction (for example `pixel_area` from `scan × track`) are flagged.
- **Exact rules** checked row by row (`pixel_area = scan × track`, ...).
- **Row links:** `cell_group` (same cell, date, day/night), `pass_id` (same orbit segment), `links` between detections whose pixel footprints touch, labelled `same_pass` / `repeat_pass` / `other_satellite` with distance and time gap, `patch_id` (connected pixels in one pass), and how strongly each field agrees across linked detections.

Output: `correlation/rows.csv.gz`, `links.csv.gz`, `correlations.json`.

### 5.2 Interpreter (`interpreter.py`, stage 2)

Decides, from the correlations, **which fields and rows each worker gets and how the work is split**:

- Each worker gets a fixed minimum set of fields plus every field related to its goal at |value| ≥ 0.4 in this dataset (FRP-related fields for fire power; fields that agree across linked detections for localisation). Duplicates and near-copies (|value| ≥ 0.98) are not sent.
- Non-wildfire detections (volcano, static source, offshore) are removed when the type is known or inferred with confidence ≥ 0.5; less certain ones are flagged. Rows without FRP (Landsat) go to localisation only.
- Linked detections form one unit that is never split; units are ordered by 10° tile and cut into packages of similar size.

Output: `interpreter/plan.json` and `interpreter/<worker>/package_NNN.csv.gz`.

### 5.3 Localisation worker (`localisation_worker.py`, stage 3a)

For every fire complex (linked detections within 24 h, any satellite):

- **Observation** per satellite pass: time, FRP-weighted centre, burning area, pixel count, FRP.
- **Extent:** bounding box and convex hull of all pixel footprints (km²).
- **Spread**, from consecutive passes of the same sensor family (VIIRS with VIIRS, MODIS with MODIS, since pixel sizes differ about 10×): new front pixels, spread direction (bearing to the new front), spread rate (km/h), centre drift (least squares), growth per hour.

### 5.4 Fire power worker (`fire_power_worker.py`, stage 3b, runs in parallel with 3a)

- **Per detection:** FRP density (MW/km², independent of pixel size) and an FRP anomaly z-score: log FRP fitted against log(pixel area × (T⁸ − T_bg⁸)) per sensor and day/night, residuals in standard deviations (flags cloud edges, sub-pixel hot spots, saturation).
- **Per pass:** FRP sum, max and mean intensity, pixels, local solar time.
- **Per fire:** peak FRP and intensity class (< 10 MW small, 10–100 moderate, 100–1,000 large, ≥ 1,000 extreme); FRP trend in % per hour from a least-squares fit of log FRP (VIIRS preferred, partial views removed, quality flag); **fire radiative energy** (MJ) and **biomass burned** = FRE × 0.368 kg/MJ ([Wooster et al. 2005](https://doi.org/10.1029/2005JD006318)); day/night FRP ratio.

### 5.5 Epicenter worker (`epicenter_worker.py`, stage 4)

Estimates **where and when each fire started**:

- **Where:** the earliest pass (plus passes within 30 min), centre weighted by FRP / pixel area, then moved halfway towards the back edge along the spread direction (the origin lies behind the front). Uncertainty radius = distance to the farthest edge of the earliest pixels. Groups of earliest pixels more than 5 km apart get separate epicenters (fires that merged later).
- **When:** latest possible start = first detection; earliest possible start = the last pass of a modelled satellite (Aqua, the three VIIRS) over the spot that saw nothing, counted only when that satellite has a detection within 3 min (proof its data exists). Fires burning when the data starts are marked "before data start" so their start is never learned as real.
- **Merging neighbours:** fires whose pixels touch at any time are merged into one fire (also through chains); totals are recomputed and each original keeps its epicenter.

### 5.6 Fire expansion worker (`fire_expansion_worker.py`, stage 4b)

Replays each merged fire's passes in time order and follows its **pixel clusters**:

| Cluster matches | Event |
|---|---|
| no existing track | **ignition** (first pixels) or **new burst** |
| one track | continues |
| several tracks | **merge** (new track inherits all histories) |
| one track → several clusters | **split** |
| several ↔ several | merge and split |

Only fine passes (VIIRS, Landsat) decide merges and splits; coarse MODIS pixels only continue tracks. Merges or splits undone within 2 h (VIIRS pixel growth at the swath edge) are marked `undone_within_2h`. A track ends ("extinguished") at the next pass that looked at the spot with proof of data and saw nothing; otherwise it is "burning at data end". Per pass the timeline records clusters, pixels, burning area, FRP, **cumulative footprint** (union of pixel squares on a 100 m raster, VIIRS/Landsat-only version too), new footprint and the front distance from the nearest epicenter.

### 5.7 Weather worker (`weather_worker.py`, stage 5)

- **Weather per 0.25° cell and hour** from [Open-Meteo](https://open-meteo.com/): observed ERA5 ([historical API](https://open-meteo.com/en/docs/historical-weather-api)) and the ECMWF IFS 0.25° forecast issued the day before ([previous-runs API](https://open-meteo.com/en/docs/previous-runs-api)), kept apart as `fc_*`. Optional Copernicus CEMS fire danger indices (FWI, FFMC, DMC, DC, ISI, BUI). Derived: wind east/north components, rain in the last 24 h, hours since rain, a weather condition class.
- **Attached to fires:** start-window weather for epicenters, weather at each pass (interpolated) with wind alignment of the spread, summaries per fire.
- **Fire–weather correlations:** spread and growth vs temperature, humidity, VPD, wind, rain; spread with the wind by wind-speed class; FRP vs weather and local time; extinction vs rain; ignition windows vs the same hours one day earlier (paired) and vs all hours (AUC).
- In per-fire mode (`weather_fires.py`) it fetches only each fire's cells and days, within one shared budget of Open-Meteo calls per minute, hour and day, with everything cached.

---

## 6. The knowledge graph

**One graph, fed by two feeds.** Output is [Neo4j bulk-import CSV](https://neo4j.com/docs/operations-manual/current/import/) (typed headers), one gzip file per node label and relationship type in `<run>/kg/`. Every id starts with the run name, so many runs load into one database without clashes:

```bash
neo4j-admin database import full neo4j --overwrite-destination --nodes=... --relationships=...
```

(the exact command for each run is written in `<run>/kg/summary.json`).

### 6.1 Feed 1: fire (`kg_builder.py`)

| Node | Meaning |
|---|---|
| `Run` | The dataset the run covered |
| `Satellite` | Terra, Aqua, Suomi NPP, NOAA-20, NOAA-21, Landsat 8/9 |
| `Pass` | One orbit segment of one satellite |
| `GridCell` | 0.05° cell, with its 0.25° weather cell |
| `Detection` | One harmonised FIRMS row (non-wildfire ones flagged) |
| `SourceFire` | A fire as localisation found it, with its fire-power values |
| `Fire` | The merged fire |
| `Epicenter` | A starting point, with its start window |
| `Observation` | One pass over one source fire |
| `Cluster` | A pixel cluster followed through time |
| `FireState` | One fire at one pass: a timeline step |
| `ExpansionEvent` | Ignition, new burst, merge, split, extinction |
| `Field`, `ExactRule`, `LinkKind` | Data fields, row-by-row rules, link kinds of the correlation layer |
| `Worker`, `DataSource`, `Package` | Pipeline workers, FIRMS sources, interpreter work packages |

| Relationship | Pattern |
|---|---|
| `SEEN_ON`, `BY` | `(Detection)-[:SEEN_ON]->(Pass)-[:BY]->(Satellite)` |
| `IN_CELL` | `(Detection)-[:IN_CELL]->(GridCell)`, `(Epicenter)-[:IN_CELL]->(GridCell)` |
| `LINKED` | `(Detection)-[:LINKED {kind, dist_km, gap_min}]->(Detection)` |
| `PART_OF`, `MERGED_INTO` | `(Detection)-[:PART_OF]->(SourceFire)-[:MERGED_INTO]->(Fire)` |
| `HAS_EPICENTER`, `FROM_SOURCE` | `(Fire)-[:HAS_EPICENTER]->(Epicenter)-[:FROM_SOURCE]->(SourceFire)` |
| `LAST_MISSED_BY` | `(Epicenter)-[:LAST_MISSED_BY {time}]->(Satellite)` |
| `OF_FIRE`, `DURING` | `(Observation)-[:OF_FIRE]->(SourceFire)`, `(Observation)-[:DURING]->(Pass)` |
| `IN_RUN` | `(Fire)-[:IN_RUN]->(Run)` |
| `HAS_CLUSTER`, `BECAME` | `(Fire)-[:HAS_CLUSTER]->(Cluster)-[:BECAME {how, time, undone_within_2h}]->(Cluster)` |
| `IN_CLUSTER`, `STARTED_CLUSTER` | `(Detection)-[:IN_CLUSTER]->(Cluster)`, `(Epicenter)-[:STARTED_CLUSTER]->(Cluster)` |
| `HAS_STATE`, `NEXT` | `(Fire)-[:HAS_STATE]->(FireState)-[:NEXT]->(FireState)`, `(FireState)-[:DURING]->(Pass)` |
| `HAD_EVENT`, `INVOLVES` | `(Fire)-[:HAD_EVENT]->(ExpansionEvent)-[:INVOLVES]->(Cluster)` |
| `CORRELATES` | `(Field)-[:CORRELATES {measure, value, scope, n}]->(Field)` |
| `USES`, `NOT_SENT` | `(Worker)-[:USES {reason}]->(Field)`, `(Worker)-[:NOT_SENT {reason}]->(Field)` |
| `RAN`, `USED_SOURCE`, `FOR_WORKER`, `IN_PACKAGE`, `AGREES_ACROSS` | Run provenance, sources, packages, cross-link agreement |

### 6.2 Feed 2: weather (`kg_weather_feed.py`)

It never rewrites a fire file. It checks that every fire node it links to exists and reuses feed 1's `Field` nodes.

| Node | Meaning |
|---|---|
| `WeatherCell` | 0.25° ERA5 / CEMS grid point, with elevation |
| `WeatherHour` | One cell, one hour: observed values, day-before forecast (`fc_*`), condition |
| `WeatherDay` | CEMS fire danger indices, when available |
| `StartWeather`, `FireWeather` | Weather over an epicenter's start window, over a fire's life |
| `WeatherFindings` | The run's fire–weather findings |

| Relationship | Pattern |
|---|---|
| `IN_WEATHER_CELL`, `HAS_HOUR`, `HAS_DAY` | `(GridCell)-[:IN_WEATHER_CELL]->(WeatherCell)-[:HAS_HOUR]->(WeatherHour)` |
| `STARTED_IN`, `HAD_START_WEATHER` | `(Epicenter)-[:STARTED_IN {role, hours_before_detection}]->(WeatherHour)` |
| `OBSERVED_IN` | `(Observation | FireState)-[:OBSERVED_IN {weather at the pass}]->(WeatherHour)` |
| `HAD_WEATHER` | `(Fire)-[:HAD_WEATHER]->(FireWeather)` |
| `CORRELATES` | Fire–weather relationships, all strengths |
| `HAS_WEATHER_FINDINGS`, `RAN` | Run findings and the weather worker's summary |

### 6.3 Size

| Scope | Nodes | Relationships | Compressed size |
|---|---|---|---|
| One archive batch (`fires_batch_039`) | 262,811 | 2,950,231 | 26.7 MB |
| All 40 archive batches | **18,659,826** | **190,890,398** | 1.76 GB |

Totals over the 40 batches: 10,487,617 detections, 50,340 satellite passes, 803,695 source fires merged into 189,819 fires, 805,670 epicenters, 1,204,365 clusters, 1,718,221 expansion events, 529,665 fire states, 3,488 weather cells and 969,528 weather hours.

---

## 7. The 10-year archive and training data

### 7.1 Archive scan (`archive_scan.py`)

Every MODIS and VIIRS detection worldwide from **14 Nov 2016 to 13 Nov 2026**: 731 five-day windows, 1,465 downloads (SP archive, then NRT for recent months), in 78 minutes. Aggregated to 0.1° cell-days, counted as **fire days** (independent of how many satellites were flying). 801,123 cells had fire. In NRT months, 17,302 cells that the archive shows are mostly static sources were masked out. The scan also produces a per-cell, per-month fire frequency (used as a feature) and a ranked list of big fires (largest: 18 days in Siberia in July–August 2021, about 17,800 km², 188,656 detections).

### 7.2 Fire selection (`fire_runs.py`)

- **1,000 major fires:** connected cell-days with FRP ≥ 500 MW each, at least 2 days, picked per region (boreal, savanna, tropical forest, Mediterranean, ...) and spread over the years so megafires do not dominate.
- **1,000 medium fires:** FRP ≥ 100 MW, 2–20 cells, never 500 MW, a random sample per region and year, so the models do not learn that every fire grows huge.
- The full pipeline runs on them in 40 batches (`data/runs/fires_batch_000…039`), each with its own knowledge graph; `weather_fires.py` adds weather and feed 2.

### 7.3 Training table (`prediction_data.py`)

Built **only from the knowledge graphs**: one row = one fire on one day d on which it was seen; targets are what happened on d+1, d+2, d+3.

| Feature group | Content |
|---|---|
| Fire now | Footprint (VIIRS-only cumulative, km²), new area on d, d−1, d−2, mean and peak FRP, pixels, passes, clusters alive, front distance, age, movement, merges so far |
| History | Previous 4 fire-days of the chain, 3-day trends |
| Separate parts | Not-yet-joined fires inside the fire that day and the gap between them |
| Neighbours | Nearest other fire: distance, gap, size, growth, power; fires within 10 and 25 km |
| Place | Latitude, region, season (sin/cos day of year), elevation, local 10-year fire frequency |
| Weather, d to d+3 | Max/mean temperature, min/mean humidity, max VPD, mean/max wind, gusts, wind east/north, rain, soil moisture 0–7/7–28/28–100 cm, hours since rain, wind towards the nearest fire |
| Terrain and fuel (v4) | See [8.6](#86-terrain-and-fuel-v4) |

118 features in v2/v3, 149 with weather in v4 (86 without). **333,452 fire-days** from 188,559 fires.

### 7.4 Fire chains (`prediction_labels.py`)

Satellites miss fires under clouds, smoke or between passes, so the tracking sees a "new" fire when the old one reappears. In 2025–2026, **43% of fire-days labelled "extinguished" were followed by a fire within 5 km in the next 3 days** (49% for fires under 1 km²). The first models learned those false extinctions and predicted that most small fires go out.

Fire B continues fire A when B is first seen **1–3 days** after A's last sighting, within **5 km plus A's radius**. Footprint, history and every target follow the chain; a fire is "out" only when nothing continues it within 3 days; when the data ends first, the row is **censored**. Result: 188,559 fires → **144,229 chains**; "out by next day" fell from 56% to 43% of fire-days, "grew ≥ 0.1 km²" rose from 27% to 34%.

### 7.5 Forecast-like weather (`forecast_errors.py`)

The models learn from observed weather (ERA5) but in use get forecasts. For fire-days from March 2024 on, the ECMWF IFS forecasts issued at the time were fetched and compared with ERA5: **4,590 error vectors** from 1,930 fire-days at 205 cells (mean absolute error on day +1: max temperature 1.2 °C, minimum humidity 4.6 points, mean wind 1.5 km/h, rain 0.46 mm). During training, each row's weather for days +1 to +3 gets a whole error vector added (same day ahead and region), so the models trust a forecast only as much as forecasts deserve.

---

## 8. AI prediction engine

### 8.1 Outputs, per active fire and day +1, +2, +3

| Output | Unit | Form |
|---|---|---|
| New burned area and total footprint | km² | low / likely / high (10th / 50th / 90th percentile) |
| Chance the fire grows by ≥ 0.1 km² (`p_grow`) | 0–1 | probability |
| Expected growth (`growth_expected`) | km² | mean, calibrated on unseen data |
| Centre movement, spread direction and rate | km, °, km/day | low / likely / high |
| Mean and peak fire power of a VIIRS pass | MW | low / likely / high |
| Chance the fire is out | 0–1 | survival model |
| Chance it merges with a separate neighbour | 0–1 | probability |
| Burn chance of every 1 km cell around the fire | 0–1 | spread map |

### 8.2 Model history

| Version | What changed | Status |
|---|---|---|
| Fast model | First XGBoost models, 87 features | Superseded |
| v2 | 118 features, XGBoost + sequence network blend, forecast-like weather | Superseded: on corrected labels its extinction chances were 6–14 points too high |
| **v3** | Fire chains, survival model with censoring, two-part growth, calibrated expected growth | **Production on `main`** |
| **v4** | v3 + terrain and land cover (149 features) | `model-v4` branch; best without a forecast |
| Frontend grid model | Logistic regression on 5.6 km cells, refitted per zone | Earlier app forecast |

### 8.3 Gradient-boosted trees (`prediction_v3.py`, XGBoost on CUDA)

Common settings: `hist` trees on the GPU, learning rate 0.05, max depth 8, min child weight 5, 80% row and column subsampling, early stopping on 2024. Two variants: **weather** and **no_weather** (used when no forecast is supplied). 25 models per variant:

| Target | Model |
|---|---|
| Extinction | **`survival:aft`** (accelerated failure time) on days until out, with censored rows when the data ends; three distribution scales (0.6 / 0.9 / 1.3) tried and the best Brier score kept; then **isotonic calibration** per day ahead on 2024 |
| Growth | **Two parts:** a classifier for "grows ≥ 0.1 km²" and quantile regression (10/50/90) for how much if it grows, combined into a mixture distribution; expected growth recalibrated on 2024 |
| Fire power (mean and peak) | Quantile regression 10/50/90 on log FRP |
| Movement (east, north) | Quantile regression on asinh(km) |
| Merging | Binary classifier, threshold chosen on 2024 |

Trained on an NVIDIA RTX 5060 (CUDA 12.8): about 28 minutes for both variants in parallel.

### 8.4 Sequence network (`prediction_seq.py`, PyTorch)

- **Architecture:** a GRU (64 units) reads the chain's previous 4 fire-days in order; its state joins the current features in an MLP (384 → 384 → 256, GELU, LayerNorm, dropout 0.1). Two heads predict all continuous targets as 3 quantiles each for all 3 days, and the yes/no targets (out, merge, grows).
- **Training:** AdamW (lr 4e-3, weight decay 1e-4), one-cycle schedule, batch 8,192, bf16 mixed precision, early stopping (patience 2). **Weather dropout:** weather hidden for 40% of rows, so one network handles missing forecasts. **Forecast noise** from the error pool. About 1 minute on the GPU.
- **Deployment:** weights exported to numpy (`seq_numpy.py`), so prediction needs no PyTorch.

### 8.5 Blend and spread map

- **Blend (production):** growth, chance it grows, fire power, movement and merging are averaged between XGBoost and the network; extinction comes from the calibrated XGBoost survival model.
- **Spread map (`prediction_spread.py`):** today's pixels are moved by the predicted shift and scaled to the predicted footprint; each 1 km cell within reach gets features (distance to the nearest moved pixel, distance to the centre in likely/high radii, "ahead of the fire" cosine, chance out, size, FRP) and a **logistic regression per day ahead** gives its burn chance. Fitted on 2024, checked on 2025–2026.
- **Merge outlook:** each fire's nearest neighbour, the gap between their predicted footprints and the merge chance.

### 8.6 Terrain and fuel (v4)

`static_features.py` reads cloud-optimised GeoTIFFs window by window over HTTP from AWS open data (no tile downloads), for all **283,905** fire locations (rounded to 0.01°, cached):

| Group | Features | Source |
|---|---|---|
| Terrain (≈ 3 km) | Mean elevation, ruggedness, mean and 90th-percentile slope, downslope direction (east/north); **wind blowing uphill** on days +1 to +3 | [Copernicus DEM GLO-90](https://registry.opendata.aws/copernicus-dem/) at its 180 m overview |
| Fuel (≈ 1 km and 5 km) | Share of tree, shrub, grass, crop, built-up, bare, snow, water, wetland, mangrove, moss | [ESA WorldCover 2021](https://registry.opendata.aws/esa-worldcover-vito/) (10 m) at its 40 m and 320 m overviews |

Terrain and land cover carry 8–12% of the trees' total gain (for example survival 3.8% + 4.0%, movement 4.9–5.5% + 5.4–6.2%). The live engine uses the same functions and cache.

### 8.7 Live prediction (`prediction_engine.py`)

1. Read the run's knowledge graph and build each fire's features exactly as in training, including fire-chain linking at prediction time and static features when the model card asks for them.
2. Take the **hourly weather forecast from the caller** (the frontend supplies it from Open-Meteo); each fire uses the forecast point in its 0.25° cell or the nearest within 0.5°. Without a forecast, the no-weather models are used.
3. Predict with the trees and the network, blend, and write `predictions.csv/.json`, `spread.csv` (cells ≥ 2%), `merge_outlook.csv` and `summary.json`.

```bash
python prediction_engine.py --run data/runs/fires_batch_039 --weather forecast.json --limit 200
```

---

## 9. Backend API

`server.py` (Python standard library HTTP server, port `PORT`, default 8000) serves the built app at `/app` and these endpoints:

| Endpoint | Returns |
|---|---|
| `/api/sensors` | FIRMS sensors and their available date ranges |
| `/api/harmonise` | All sensors harmonised for a date range and area |
| `/api/fires` | One sensor's raw detections |
| `/api/compare` | MODIS and VIIRS made comparable |
| `/api/weather` | Fire weather for a zone (`weather.py`, Open-Meteo, up to a 7 × 7 grid, daily max temperature, min humidity, max wind and direction, gusts, rain, a fire-weather rating) |
| `/api/predict` | The earlier grid forecast for a zone |
| `/api/pipeline/plan` | Which of the 3 days after the window already have real data |
| `/api/pipeline/harmonised` | The harmonised detections the pipeline starts from |
| `/api/pipeline/stage` | Runs one stage for the zone: `correlation`, `interpreter`, `localisation`, `fire_power`, `epicenter`, `expansion`, the knowledge graph, then `prediction_engine.py` with the Open-Meteo forecast |
| `/api/pipeline/results` | Fires, expansion, predictions and spread map for the map and sidebar |

---

## 10. Frontend

**Fire Pattern Predictor** (`frontend/`), a single-page app served by the backend at `http://localhost:8000/app`.

| Part | Technology |
|---|---|
| UI | [React](https://react.dev/) 18, TypeScript, [Vite](https://vite.dev/) 6 |
| Map | [MapLibre GL](https://maplibre.org/) 5.6 with globe projection, [Esri World Imagery](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer), [Natural Earth](https://www.naturalearthdata.com/) borders |
| State | [Zustand](https://github.com/pmndrs/zustand) 5 |
| Fonts | IBM Plex Sans, IBM Plex Mono |

| Feature | Description |
|---|---|
| Opening screen | Live status of each satellite's latest data (MODIS, VIIRS S-NPP / NOAA-20 / NOAA-21, Landsat) and a spinning globe with stars and embers |
| Zones | Draw a box, click a country or territory, drop a pin with a radius, or type coordinates |
| Analyse zone | Load all sensors harmonised into one dataset, or one sensor's raw detections, for a date range |
| Pipeline run | Runs the stages one by one with progress, then the AI prediction; if the following days already have real data, shows it instead |
| Forecast | Predicted spread for days +1 to +3, burn-chance heatmap, fire cards with each fire's ranges, merge and extinction chances, a timeline of observed and forecast days, charts in the sidebar |
| Fire weather | Temperature, humidity and animated wind-flow layers over the zone |

| Source file | Role |
|---|---|
| `src/engine.ts` | Globe, space backdrop, spin, transitions, layers, popups |
| `src/store.ts` | App state and actions (zones, analyse, pipeline, forecast, panel) |
| `src/api.ts` | Backend calls |
| `src/riskHeat.ts`, `weatherHeat.ts`, `windFlow.ts` | Burn-chance heatmap, weather heatmaps, wind animation |
| `src/fireCard.ts`, `timeline.ts`, `countries.ts`, `geo.ts`, `format.ts`, `mapImage.ts` | Fire forecast card, timeline, country shapes, geometry, formatting, image export |
| `src/components/` | `Splash`, `NavBar`, `Panel`, `PanelCharts`, `Charts`, `ZoneSection`, `Forecast`, `AnalyseModal`, `MapView`, `MapHud`, `Boundary` |

```bash
python server.py               # open http://localhost:8000/app
cd frontend && npm install && npm run dev    # development, http://localhost:5173/app/
```

Screenshots: [home](images/frontend_home.jpg), [globe](images/frontend_map.jpg), [analyse](images/frontend_analyse.jpg), [forecast](images/frontend_forecast.jpg).

---

## 11. Metrics

Unless stated: trained on 2016–2023, 2024 used to stop training and calibrate, tested on **2025–2026 (49,378 fire-days from 28,657 fires that no model saw)**, scored against the corrected fire-chain labels. Cells show **day +1 / +2 / +3**. *Persistence* = "the fire keeps doing what it did today". Produced by `prediction_metrics.py`, `test_prediction_accuracy.py`, `test_small_fires.py` and `backtest/backtest.py`.

### 11.1 Glossary

| Metric | Meaning | Better |
|---|---|---|
| Overall accuracy | Mean of 5 accuracies: growth class, FRP within ×2, direction within 45°, extinction, merge balanced accuracy | Higher |
| Accuracy | Share of yes/no predictions right (threshold 0.5 unless stated) | Higher |
| Balanced accuracy | Mean of hit rates on positives and negatives (fair for rare events) | Higher |
| Precision / Recall / F1 | Share of alarms that were right / share of events caught / their harmonic mean | Higher |
| ROC AUC | Chance a random positive ranks above a random negative (0.5 guessing, 1 perfect) | Higher |
| PR AUC | Average precision over all thresholds | Higher |
| Brier score | Mean squared error of predicted chances | Lower |
| Calibration error | Mean gap between predicted chance and observed share over 10 bins, percentage points | Lower |
| MAE | Mean absolute error of the likely value (km², MW) | Lower |
| Low–high coverage | Share of real values inside the 10th–90th percentile range (target about 80%) | ≈ 80% |
| Pinball loss | Quantile loss of the three percentiles (log scale) | Lower |
| Growth class accuracy, macro F1 | None (< 0.1 km²) / small (0.1–2) / large (≥ 2 km²) | Higher |
| Within ×2 | FRP predictions within a factor of 2 | Higher |
| Within 45° / 90°, median angle error | Spread direction vs real movement (fires that moved ≥ 1 km) | Higher / lower |
| Median centre error | Distance between predicted and real centre, km | Lower |

### 11.2 Headline: overall accuracy %

| Model | With weather | Without weather |
|---|---|---|
| v2 blend | 61.6 / 60.3 / 59.3 | 60.3 / 58.7 / 58.3 |
| v3 XGBoost | 62.1 / 61.0 / 60.6 | 60.9 / 59.3 / 58.9 |
| v3 sequence network | 61.9 / 60.7 / 60.4 | 60.4 / 58.9 / 58.9 |
| **v3 blend (production)** | **62.2 / 61.1 / 60.8** | 61.0 / 59.4 / 59.3 |
| v4 XGBoost | 62.1 / 60.8 / 60.6 | 61.8 / 60.5 / 60.1 |
| v4 sequence network | 61.7 / 60.4 / 60.3 | 60.3 / 58.9 / 58.7 |
| **v4 blend** | **62.4 / 61.0 / 60.9** | **61.7 / 60.4 / 60.0** |

### 11.3 Extinction

| Model | Accuracy % | Precision % | Recall % | F1 % | ROC AUC | PR AUC | Brier | Calibration error, pts |
|---|---|---|---|---|---|---|---|---|
| v2 blend | 71.9 / 75.9 / 79.0 | 62.9 / 73.6 / 79.6 | 87.0 / 90.5 / 92.4 | 73.0 / 81.1 / 85.5 | 0.819 / 0.845 / 0.857 | 0.749 / 0.869 / 0.918 | 0.198 / 0.169 / 0.146 | 13.54 / 9.54 / 6.27 |
| v3 blend | 74.4 / 77.3 / 79.1 | 68.8 / 78.7 / 82.9 | 75.8 / 82.9 / 86.8 | 72.1 / 80.8 / 84.8 | 0.833 / 0.855 / 0.858 | 0.792 / 0.883 / 0.918 | 0.165 / 0.153 / 0.141 | 1.91 / 0.88 / 1.25 |
| v3 sequence network | 74.2 / 76.7 / 79.5 | 70.5 / 77.6 / 81.4 | 70.7 / 83.4 / 89.8 | 70.6 / 80.4 / 85.4 | 0.832 / 0.854 / 0.864 | 0.793 / 0.885 / 0.926 | 0.165 / 0.153 / 0.137 | 1.39 / 0.80 / 1.18 |
| v4 blend | 74.4 / 77.3 / 79.3 | 69.5 / 78.7 / 82.4 | 73.9 / 83.0 / 88.0 | 71.7 / 80.8 / 85.1 | 0.832 / 0.855 / 0.858 | 0.793 / 0.883 / 0.919 | 0.166 / 0.153 / 0.141 | 1.92 / 0.92 / 1.26 |
| Persistence | 56.3 / 42.6 / 33.0 | – | 0 | 0 | – | – | – | – |

(The v3/v4 blend's extinction is the XGBoost survival model.)

### 11.4 Chance it grows (≥ 0.1 km²)

| Model | Accuracy % | Precision % | Recall % | F1 % | ROC AUC | PR AUC | Calibration error, pts |
|---|---|---|---|---|---|---|---|
| v3 XGBoost | 76.8 / 72.5 / 71.9 | 78.3 / 76.9 / 76.1 | 45.5 / 52.4 / 61.5 | 57.5 / 62.3 / 68.0 | 0.797 / 0.799 / 0.809 | 0.732 / 0.779 / 0.812 | 2.38 / 2.85 / 2.74 |
| v3 sequence network | 76.9 / 72.4 / 71.8 | 78.3 / 76.3 / 75.2 | 46.1 / 52.7 / 62.5 | 58.0 / 62.4 / 68.3 | 0.796 / 0.798 / 0.806 | 0.731 / 0.779 / 0.811 | 1.85 / 2.23 / 1.98 |
| **v3 blend** | 77.1 / 72.7 / 72.3 | 79.0 / 77.1 / 76.3 | 45.9 / 52.6 / 62.3 | 58.0 / 62.6 / 68.6 | 0.801 / 0.803 / 0.812 | 0.736 / 0.783 / 0.816 | 2.00 / 2.27 / 2.03 |
| **v4 blend** | 76.9 / 72.6 / 72.3 | 79.2 / 77.4 / 76.9 | 45.1 / 52.0 / 61.2 | 57.5 / 62.2 / 68.2 | 0.801 / 0.803 / 0.811 | 0.736 / 0.783 / 0.815 | 2.38 / 2.50 / 2.47 |

### 11.5 Merging with a separate fire (threshold for best F1 on 2024; merges are 4–8% of fire-days)

| Model | Balanced acc. % | Precision % | Recall % | F1 % | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|---|---|---|
| v2 blend | 71.1 / 78.6 / 77.4 | 53.1 / 50.2 / 58.8 | 43.9 / 61.2 / 58.4 | 48.1 / 55.2 / 58.6 | 0.947 / 0.938 / 0.931 | 0.518 / 0.599 / 0.645 | 0.027 / 0.038 / 0.044 |
| v3 blend | 69.7 / 73.8 / 80.1 | 56.2 / 59.0 / 52.4 | 40.6 / 50.0 / 65.4 | 47.2 / 54.1 / 58.2 | 0.947 / 0.939 / 0.934 | 0.519 / 0.597 / 0.645 | 0.027 / 0.038 / 0.045 |
| v4 blend | 69.2 / 73.7 / 77.8 | 56.8 / 59.5 / 58.1 | 39.6 / 49.7 / 59.3 | 46.7 / 54.2 / 58.7 | 0.946 / 0.938 / 0.933 | 0.517 / 0.597 / 0.644 | 0.027 / 0.038 / 0.045 |
| Persistence | 50.0 | 0 | 0 | 0 | – | – | – |

### 11.6 Growth, fire power, direction

| Model | Growth MAE km² | Growth class % | Class macro F1 % | Growth coverage % | Pinball (log) |
|---|---|---|---|---|---|
| v2 blend | 1.95 / 3.58 / 5.10 | 72.4 / 64.6 / 60.1 | 60.1 / 55.0 / 51.1 | 80.8 / 75.5 / 70.7 | 0.075 / 0.101 / 0.126 |
| **v3 blend** | 1.92 / 3.47 / 4.92 | 72.9 / 66.9 / 65.4 | 61.6 / 61.3 / 63.4 | 84.3 / 83.9 / 81.7 | 0.072 / 0.090 / 0.101 |
| v4 blend | 1.94 / 3.53 / 4.98 | 72.9 / 66.8 / 65.4 | – | 85.5 / 83.4 / 82.1 | – |
| Persistence | 2.85 / 5.87 / 9.07 | 32.5 / 35.9 / 36.4 | 37.8 / 37.3 / 35.8 | – | – |

| Model | FRP MAE MW | FRP within ×2 % | FRP coverage % | Peak FRP MAE MW | Within 45° % | Within 90° % | Median angle error ° | Median centre error km |
|---|---|---|---|---|---|---|---|---|
| v2 blend | 295 / 370 / 420 | 39.9 / 37.2 / 36.2 | 76.6 / 77.1 / 75.5 | 537 / 687 / 789 | 35.4 / 36.3 / 34.7 | 59.5 / 59.4 / 59.3 | 70.7 / 70.6 / 71.7 | 2.97 / 3.37 / 3.71 |
| **v3 blend** | 295 / 370 / 414 | 40.0 / 37.7 / 37.9 | 76.6 / 76.6 / 75.4 | 536 / 686 / 780 | 36.2 / 36.0 / 35.4 | 59.8 / 59.9 / 60.1 | 69.7 / 69.8 / 70.4 | 2.95 / 3.36 / 3.69 |
| v4 blend | 298 / 370 / 416 | 40.2 / 37.9 / 38.0 | – | – | 36.5 / 35.8 / 35.2 | – | – | 2.94 / 3.36 / 3.71 |
| Persistence | 420 / 579 / 645 | 33.2 / 30.1 / 30.0 | – | 766 / 1077 / 1232 | 21.1 / 19.8 / 20.5 | 38.7 / 37.8 / 38.8 | 115.6 / 116.4 / 114.2 | 5.48 / 9.13 / 12.76 |

### 11.7 Without a weather forecast

| Model | Extinction ROC AUC | Extinction calibration, pts | Merge ROC AUC | Growth MAE km² | FRP within ×2 % |
|---|---|---|---|---|---|
| v2 blend | 0.773 / 0.779 / 0.787 | 14.17 / 10.15 / 7.15 | 0.944 / 0.933 / 0.924 | 2.09 / 3.98 / 5.78 | 39.1 / 36.0 / 36.1 |
| v3 blend | 0.787 / 0.796 / 0.800 | 1.86 / 1.27 / 1.48 | 0.944 / 0.935 / 0.927 | 2.06 / 3.90 / 5.51 | 39.5 / 36.9 / 37.3 |
| **v4 blend** | **0.820 / 0.845 / 0.848** | 1.99 / 1.01 / 1.21 | 0.944 / 0.936 / 0.929 | **2.01 / 3.74 / 5.29** | 40.2 / 37.6 / 37.3 |

Terrain and land cover make up for much of a missing forecast: v4's extinction AUC without weather is close to v3's with weather.

### 11.8 What happens to fires by size (the check that found the false-extinction problem)

| Day | Fire size | Fire-days | Really out % | v2 says % | v3 says % | v4 says % | Really grew % | v3 chance it grows % | v4 chance it grows % |
|---|---|---|---|---|---|---|---|---|---|
| +1 | < 1 km² | 25,054 | 51.8 | 81.3 | 59.7 | 57.8 | 23.6 | 21.0 | 20.6 |
| +1 | 1–10 km² | 17,981 | 41.4 | 48.3 | 42.8 | 41.3 | 36.4 | 34.4 | 33.9 |
| +1 | 10–100 km² | 4,711 | 20.7 | 13.7 | 18.7 | 17.6 | 68.6 | 69.2 | 69.0 |
| +1 | > 100 km² | 1,632 | 12.9 | 10.2 | 14.8 | 13.7 | 84.3 | 85.1 | 85.1 |
| +3 | < 1 km² | 25,054 | 73.6 | 92.2 | 84.1 | 85.4 | 40.1 | 36.6 | 36.2 |
| +3 | 1–10 km² | 17,981 | 65.3 | 71.6 | 64.1 | 65.8 | 50.1 | 49.0 | 48.4 |
| +3 | 10–100 km² | 4,711 | 48.2 | 39.7 | 33.8 | 34.4 | 74.0 | 76.2 | 75.8 |
| +3 | > 100 km² | 1,632 | 36.8 | 34.0 | 25.2 | 26.8 | 85.2 | 86.2 | 86.2 |

v2's "likely growth" for fires under 1 km² was positive on only 1.7% of days (real: 23.6%); v3/v4 add `p_grow` and calibrated expected growth.

### 11.9 Robustness (retrained without one year or region, overall accuracy %, day +1 / +3)

| Left out | v3 | v4 |
|---|---|---|
| Year 2024 | 62.4 / 59.8 | 62.2 / 59.8 |
| Americas | 61.5 / 59.7 | 61.6 / 59.4 |
| Africa | 60.9 / 58.5 | 60.9 / 58.2 |

### 11.10 With real issued forecasts (earlier models, original labels)

| Overall accuracy %, day +1 / +2 / +3 | Fast model | v2 XGBoost |
|---|---|---|
| Observed weather | 66.1 / 65.3 / 65.9 | 67.0 / 65.7 / 66.4 |
| Issued forecasts (real use) | 64.1 / 62.1 / 64.5 | 65.0 / 63.8 / 64.8 |
| No-weather models | 64.9 / 62.8 / 64.7 | 65.2 / 63.1 / 64.4 |

### 11.11 Spread map (1 km cells, calibrated on 2024, checked on 2025–2026)

| Day | ROC AUC | Brier | Brier of base rate | Cells that burned % | Cells checked |
|---|---|---|---|---|---|
| +1 | 0.728 | 0.0477 | 0.0502 | 5.3 | 451,051 |
| +2 | 0.732 | 0.0338 | 0.0348 | 3.61 | 473,015 |
| +3 | 0.753 | 0.0250 | 0.0257 | 2.64 | 491,278 |

### 11.12 Behaviour checks

v3 and v4 blends pass **all 11**: more wind or drier air → more growth; drier air → more fire power; rain → more extinction and less growth; turning the wind turns the spread; growth never negative; ranges ordered; chances in 0–1; deterministic; works without weather.

### 11.13 Backtest in the app (end to end)

`backtest/backtest.py` (15 Nov 2026) drives the app's own backend: pipeline on 5 days of data, prediction with the Open-Meteo forecast, compared with the **real FIRMS detections of the following days**. Zones: Brazil (Mato Grosso / Bolivia border) and Central Africa (CAR / South Sudan savanna). Report: [`backtest/report.md`](../backtest/report.md).

**Over 1,503 fire-days, 68.6% of fire outcomes were called right, against 38.8% for "every fire keeps burning".**

| Zone · day | Fires | Really out | Predicted out | AUC | Accuracy | "All keep burning" |
|---|---|---|---|---|---|---|
| Brazil · today +1 | 32 | 66% | 46% | 0.75 | 0.72 | 0.34 |
| Brazil · +1 / +2 / +3 | 60 | 80 / 85 / 90% | 50 / 63 / 74% | 0.66 / 0.66 / 0.78 | 0.57 / 0.65 / 0.83 | 0.20 / 0.15 / 0.10 |
| Africa · today +1 | 325 | 60% | 64% | 0.70 | 0.68 | 0.40 |
| Africa · +1 / +2 / +3 | 322 | 47 / 56 / 68% | 67 / 73 / 78% | 0.70 / 0.72 / 0.65 | 0.65 / 0.71 / 0.70 | 0.53 / 0.44 / 0.32 |

| Finding | Value |
|---|---|
| Spread map AUC, next day | 0.83–0.87 Brazil, 0.61–0.67 Africa; near guessing by day +2/+3 |
| Real FRP inside predicted range | 76–84% in Africa (target ≈ 80%) |
| Centre movement vs "stays put" | No better (differences 0.0–0.15 km) |
| Detections on predicted days that were new fires | 66–76% Brazil, 53–71% Africa (new ignitions are outside the model's scope) |

### 11.14 Frontend grid model (earlier app forecast)

Southern Chad, 9–13 Nov 2026, 5.6 km cells: AUC 0.739, precision 28.5%, recall 33.0%, F1 30.6%, against persistence F1 40.7%. One zone and one day; it was replaced in the app by the trained engine.

### 11.15 History (original, uncorrected labels)

Before fire chains, v2 scored about 65–67% overall and an extinction ROC AUC of 0.917 / 0.913 / 0.907; those numbers rewarded predicting false extinctions. Rescored on corrected labels, v2 drops to 61.6 / 60.3 / 59.3% with extinction chances 13.6 / 9.5 / 6.2 points too high. The full original-label tables are in the [README](../README.md#history-metrics-on-the-original-labels).

---

## 12. Testing

| Command | Checks |
|---|---|
| `python test_correlation.py`, `test_interpreter.py`, `test_localisation.py`, `test_fire_power.py`, `test_epicenter.py`, `test_expansion.py`, `test_weather.py`, `test_prediction.py` | Offline unit tests of each stage (no key, no network) |
| `python prediction_metrics.py` | Every metric in section 11 for every model |
| `python test_small_fires.py` | Fires by size, real vs predicted |
| `python test_prediction_accuracy.py heldout / behaviour / cv / forecast` | Held-out accuracy by region, size, season; behaviour checks; leave-one-out retraining; real-forecast accuracy |
| `python backtest/backtest.py` | End-to-end check through the running server |

---

## 13. Limitations

- The engine predicts area, centre shift, direction and power, **not the fire outline**.
- **Direction and fire power** are the weakest outputs (about 36% within 45°, 38–40% within ×2), though about twice persistence.
- **New ignitions** are not predicted; only fires already burning.
- **Fire chains** use distance and timing only; nearby separate fires can be joined, and gaps over 3 days still split a fire.
- **Regional calibration:** the backtest shows correct ranking but extinction chances that are too low in Brazil and too high in the African savanna.
- **Weather coverage:** about half of the training rows have weather; v3/v4 have not yet been tested with archived real forecasts.
- **Satellite limits:** only actively burning pixels at overpass time are seen, so 3–5% of nearby cells register as burning on a given day.

---

## 14. External data, services and libraries

### 14.1 Satellite fire data (NASA)

| Source | Used for | Links |
|---|---|---|
| NASA FIRMS | All fire detections, NRT and SP | [FIRMS](https://firms.modaps.eosdis.nasa.gov/) · [API](https://firms.modaps.eosdis.nasa.gov/api/) · [Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) · [Data availability](https://firms.modaps.eosdis.nasa.gov/api/data_availability/) · [Map key](https://firms.modaps.eosdis.nasa.gov/api/map_key/) · [About FIRMS](https://www.earthdata.nasa.gov/data/tools/firms) · [FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq) |
| MODIS (Terra, Aqua) | 1 km active fires | [NASA Earthdata: MODIS](https://www.earthdata.nasa.gov/data/instruments/modis) |
| VIIRS (Suomi NPP, NOAA-20, NOAA-21) | 375 m active fires | [NASA Earthdata: VIIRS](https://www.earthdata.nasa.gov/data/instruments/viirs) · [NOAA JPSS](https://www.nesdis.noaa.gov/our-satellites/currently-flying/joint-polar-satellite-system) |
| Landsat 8/9 OLI | 30 m active fires | [NASA Landsat](https://landsat.gsfc.nasa.gov/) |

### 14.2 Weather

| Source | Used for | Links |
|---|---|---|
| Open-Meteo | Observed hourly weather, archived forecasts, live forecast for the app | [Open-Meteo](https://open-meteo.com/) · [Forecast API](https://open-meteo.com/en/docs) · [Historical API](https://open-meteo.com/en/docs/historical-weather-api) · [Previous runs API](https://open-meteo.com/en/docs/previous-runs-api) |
| ECMWF IFS 0.25° | Model behind the forecasts | [ECMWF datasets](https://www.ecmwf.int/en/forecasts/datasets/set-i) |
| ERA5 (Copernicus C3S) | Reanalysis behind the observed weather | [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) |
| Copernicus CEMS fire danger | Optional FWI indices (off by default) | [CEMS fire historical](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1) |

### 14.3 Terrain, land cover and maps

| Source | Used for | Links |
|---|---|---|
| Copernicus DEM GLO-90 | Elevation, slope, downslope direction (v4) | [AWS open data](https://registry.opendata.aws/copernicus-dem/) |
| ESA WorldCover 2021 | Land cover / fuel (v4) | [AWS open data](https://registry.opendata.aws/esa-worldcover-vito/) · [ESA WorldCover](https://esa-worldcover.org/) |
| Esri World Imagery | Globe imagery | [Service](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer) |
| Natural Earth | Country borders | [naturalearthdata.com](https://www.naturalearthdata.com/) |
| world-atlas | Outlook map basemap | [topojson/world-atlas](https://github.com/topojson/world-atlas) |

### 14.4 Science references

| Reference | Used for |
|---|---|
| Wooster et al. (2005), *Retrieval of biomass combustion rates and totals from fire radiative power observations*, JGR. [doi:10.1029/2005JD006318](https://doi.org/10.1029/2005JD006318) | Biomass burned = FRE × 0.368 kg/MJ |
| [FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq) | Confidence classes, `type` field, pixel sizes |

### 14.5 Software

| Library / tool | Used for |
|---|---|
| [XGBoost](https://xgboost.readthedocs.io/) (CUDA) | Tree models, survival AFT, quantile regression |
| [PyTorch](https://pytorch.org/) | Sequence network training |
| [scikit-learn](https://scikit-learn.org/) | Isotonic calibration, logistic regression, metrics |
| [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) | Data processing, KD-trees |
| [rasterio](https://rasterio.readthedocs.io/) / [GDAL](https://gdal.org/) | Reading cloud-optimised GeoTIFFs over HTTP |
| [Neo4j](https://neo4j.com/docs/operations-manual/current/import/) | Knowledge graph import format |
| [React](https://react.dev/), [Vite](https://vite.dev/), [TypeScript](https://www.typescriptlang.org/), [MapLibre GL](https://maplibre.org/), [Zustand](https://github.com/pmndrs/zustand), [d3](https://d3js.org/) | Frontend and outlook maps |
| [matplotlib](https://matplotlib.org/), [Pillow](https://python-pillow.org/) | Animations |
| [Git LFS](https://git-lfs.com/) | Trained models in the repository |

---

## 15. Credits and attribution

We acknowledge the use of data from NASA's Fire Information for Resource Management System ([FIRMS](https://www.earthdata.nasa.gov/data/tools/firms)), part of NASA's Earth Science Data and Information System. Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), based on ECMWF IFS forecasts and Copernicus ERA5. Imagery © Esri; borders from Natural Earth. Terrain: Copernicus DEM, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA. Land cover: © ESA WorldCover project 2021, contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium.
