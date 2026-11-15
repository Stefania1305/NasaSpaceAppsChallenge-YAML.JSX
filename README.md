# Nasa: wildfire tracking and 3-day fire prediction

**Turns raw NASA FIRMS satellite fire detections into tracked fires, a knowledge graph and a 3-day forecast of how each fire grows, moves, burns and merges.**

```bash
python run_pipeline.py --date 2026-11-01 --days 1 --area world --no-weather
python prediction_engine.py --run data/runs/2026-11-01_1d_world --limit 200 --weather forecast.json
```

The first command downloads one day of worldwide MODIS and VIIRS detections and runs the full tracking pipeline on it. The second predicts, for the 200 largest active fires, their footprint, spread direction and rate, fire power, chance of going out or merging with a neighbour, and a 1 km burn-chance map for each of the next 3 days, using the weather forecast you provide. Each value comes as a low / likely / high range.

## What it does

| Stage | What you get |
|---|---|
| Harmonisation | MODIS, VIIRS (Suomi NPP, NOAA-20, NOAA-21) and Landsat detections made directly comparable, with the FIRMS `type` field inferred where a file lacks it |
| Correlation and interpretation | Detections linked across satellites and passes, false and duplicate detections removed |
| Localisation and fire power | Each fire's position, movement and radiative power (FRP) per satellite pass |
| Epicenters | Where each fire started, checked against the satellite's actual swath |
| Fire expansion | Clusters followed pass by pass: ignitions, bursts, merges, splits and extinctions |
| Weather | Hourly weather for every fire (Open-Meteo: ERA5 observations and archived ECMWF forecasts) |
| Knowledge graph | Every result of every stage as Neo4j-import CSV files: fires, detections, clusters, events, weather and their relationships |
| Prediction | GPU-trained models that forecast each active fire 3 days ahead |
| Merge dataset | Every observed merge of two separate fires, with geometry and weather at the time |

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Prediction engine](#prediction-engine)
- [Accuracy](#accuracy)
- [Data layout](#data-layout)
- [Testing](#testing)
- [Limitations](#limitations)
- [Data sources](#data-sources)

## Requirements

- Python 3.14 (developed and tested on 3.14.7)
- A NASA FIRMS map key ([request one](https://firms.modaps.eosdis.nasa.gov/api/map_key/))
- An NVIDIA GPU with CUDA for training (an RTX 5060 was used). Prediction and training also run on the CPU, about 7.5 times slower to train.
- Disk space: the 10-year archive and its runs take tens of GB

## Installation

The project has no package manifest. Install the libraries it imports:

```bash
pip install pandas numpy requests scipy scikit-learn xgboost matplotlib pillow
```

The trained v2 models are in `models/v2/` and stored with [Git LFS](https://git-lfs.com/) (285 MB). Install Git LFS before cloning, or fetch them afterwards:

```bash
git lfs install
git lfs pull
```

Tested with pandas 3.0.6, numpy 2.5.4, xgboost 3.4.1 (CUDA build), scikit-learn 1.9.1, scipy 1.18.1, requests 2.34.2, matplotlib 3.11.2 and Pillow 12.3.0.

## Configuration

Settings are read from a `.env` file in the project folder. It is git-ignored; never commit it.

```ini
FIRMS_MAP_KEY=your_firms_key
```

| Variable | Required | Default | Used for |
|---|---|---|---|
| `FIRMS_MAP_KEY` | yes | none | All FIRMS downloads |
| `FIRMS_MAP_KEYS` | no | `FIRMS_MAP_KEY` | Comma-separated list of keys for the archive scan. When one key reaches its quota (5,000 transactions per 10 minutes), the next one is used |
| `PORT` | no | `8000` | Port of the local browsing server |
| `EWDS_API_KEY`, `EWDS_API_URL` | no | none | Copernicus CEMS fire danger indices. Not used by default: the weather stage uses Open-Meteo soil moisture instead |

Open-Meteo needs no key. Its free limits are 600 calls per minute, 5,000 per hour and about 10,000 per day; `weather_worker.py` stays under them.

## Usage

### Browse FIRMS data

```bash
python server.py
```

Open http://localhost:8000. The server proxies the FIRMS API so the key never reaches the browser, and adds the harmonised and inferred `type` columns to every table. `test.html` is a test bench for the harmonisation.

### Run the pipeline for a date and area

```bash
python run_pipeline.py --date 2026-11-01 --days 1 --area world --no-weather
```

| Option | Default | Meaning |
|---|---|---|
| `--date` | required | First day, `YYYY-MM-DD` |
| `--days` | `1` | Number of days |
| `--area` | `world` | `world` or a box `west,south,east,north`, for example `--area=-48,-12,-42,-6` |
| `--no-weather` | off | Skip the weather join and knowledge-graph feed 2. A world day needs about 32,000 Open-Meteo calls, more than the free daily quota |
| `--start-at` | none | Skip earlier stages and reuse their output, for example `--start-at interpreter` |

Output goes to `data/runs/<date>_<days>d_<area>/`, with stage timings in `timings.json`. On a one-day world run, the stages from the interpreter to KG feed 1 took about 7 minutes, not counting the FIRMS download and correlation.

### Build the 10-year archive and train the models

```bash
python archive_scan.py --start 2016-11-14 --end 2026-11-13
python fire_runs.py --major 1000 --medium 1000
python weather_fires.py
python prediction_data.py
python forecast_errors.py
python prediction_train.py --device cuda --full --noise
python prediction_seq.py --full --epochs 16 --out models_seq_full
python prediction_spread.py fit
```

| Step | What it does |
|---|---|
| `archive_scan.py` | Downloads 10 years of MODIS and VIIRS (Suomi NPP, then NOAA-20) worldwide, builds a fire frequency map (`frequency.csv.gz`) and ranks the biggest fires. Options: `--threads`, `--agg-workers`, `--top`, `--out` |
| `fire_runs.py` | Picks the major and medium fires across regions and runs the full pipeline on them in batches (`data/runs/fires_batch_000` to `_039`). Options: `--batch`, `--parallel`, `--limit`, `--reselect` |
| `weather_fires.py` | Adds weather to every batch and feeds it into the knowledge graph, within the Open-Meteo quota (`--budget`, default 9,000 calls). Skips batches that already have weather |
| `prediction_data.py` | Builds the training table `data/prediction/fire_days.csv.gz`: one row per fire per day, with the targets for days +1 to +3 |
| `forecast_errors.py` | Measures how wrong real weather forecasts were (forecast minus observed) at about 200 weather cells, using archived forecasts from Open-Meteo. Writes a pool of 4,590 error vectors used in training. Option: `--cells` |
| `prediction_train.py` | Trains the models with and without weather (118 and 55 features, 42 models) and writes `data/prediction/model_card.json`. `--noise` trains on forecast-like weather, `--full` uses all years (production), `--out` picks the model folder. About 25 minutes on the GPU |
| `prediction_seq.py` | Trains the sequence network with PyTorch on the GPU (about 1 minute) and saves it also as plain arrays for numpy. Without `--full` it trains on 2016–2023 and stops when 2024 stops improving |
| `prediction_spread.py fit` | Calibrates the 1 km burn-chance map on 2024 and checks it on 2025–2026 (`data/prediction/models/spread.json`) |

### Predict

```bash
python prediction_engine.py --run data/runs/fires_batch_039 --weather forecast.json
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | A pipeline run folder |
| `--weather` | none | Hourly weather forecast for the fires' area, as `.json` or `.csv`. Without it, the no-weather models are used |
| `--limit` | `0` (all) | Only the N largest active fires |

The engine does not download weather. The forecast comes from the caller (the frontend), as hourly rows in the units the models were trained on. In code, `prediction_engine.predict(run_dir, weather=rows)` also accepts a list of dicts or a DataFrame.

| Field | Unit | Required |
|---|---|---|
| `latitude`, `longitude` | degrees | yes |
| `time` | UTC hour, ISO 8601, e.g. `2026-11-14T13:00` | yes |
| `temperature_2m` | °C | recommended |
| `relative_humidity_2m` | % | recommended |
| `wind_speed_10m`, `wind_gusts_10m` | km/h | recommended |
| `wind_direction_10m` | degrees the wind comes from | recommended |
| `precipitation` | mm per hour | recommended |
| `vapour_pressure_deficit` | kPa | recommended |
| `soil_moisture_0_to_7cm`, `_7_to_28cm`, `_28_to_100cm` | m³/m³ | optional |
| `elevation` | m | optional |

```json
{"hourly": [
  {"latitude": 10.25, "longitude": 19.25, "time": "2026-11-14T13:00", "temperature_2m": 33.1,
   "relative_humidity_2m": 18, "wind_speed_10m": 14.2, "wind_direction_10m": 60, "wind_gusts_10m": 27.0,
   "precipitation": 0, "vapour_pressure_deficit": 4.1}
]}
```

- **Coverage:** the forecast should cover the run's last day through 3 days ahead. A few days before that let the engine count the hours since the last rain.
- **Matching:** each fire uses the forecast point in its 0.25° cell, or the nearest point within 0.5°. Points in the same cell are averaged. Fires with no point nearby keep empty weather features.
- **Gaps:** missing variables or hours are allowed; the models were trained with gaps.

Output, in `<run>/prediction/`:

| File | Content |
|---|---|
| `predictions.csv`, `predictions.json` | One row per fire and day ahead, with low / likely / high values |
| `spread.csv` | Burn chance per 1 km cell around each fire and day ahead (cells of at least 2%) |
| `summary.json` | Run, weather used and its coverage, models, timing |

### Animate a fire and build the merge dataset

```bash
python prediction_animation.py --run data/runs/fires_batch_039 --fire 1501
python merges_dataset.py
```

`prediction_animation.py` writes `<run>/prediction/spread_<fire>.gif`: the last observed days of satellite pixels, then the 3 predicted days. Options: `--half-km` (view size, default 15), `--past` (observed frames, default 4), `--ms` (milliseconds per frame).

`merges_dataset.py` writes `data/prediction/merges.csv.gz`, one row per observed merge, and `<run>/prediction/merge_outlook.csv` for runs with predictions. Use `--runs` to process only some runs.

## Pipeline

```
FIRMS download
  -> correlation          link detections across satellites and passes
  -> interpreter          clean and plan the work
  -> localisation  ┐      position and movement per pass
  -> fire power    ┘      radiative power per pass (both run at the same time)
  -> epicenter            where each fire started
  -> fire expansion       clusters followed pass by pass
  -> KG feed 1 (fire)     kg_builder.py
  -> weather join         weather_worker.py
  -> KG feed 2 (weather)  kg_weather_feed.py
```

Each stage reads the previous stage's files in the run folder. Workers never read the knowledge graph; it is an output that holds every worker's results.

The knowledge graph (`<run>/kg/`) is a set of `nodes_*.csv.gz` and `rels_*.csv.gz` files in Neo4j bulk-import format, with `schema.json` and `summary.json`. Node types include `Fire`, `Detection`, `Cluster`, `FireState`, `Epicenter`, `ExpansionEvent`, `Pass`, `Satellite`, `WeatherCell`, `WeatherHour` and `FireWeather`.

## Prediction engine

For every fire seen on the run's last day, the engine predicts days +1, +2 and +3:

| Output | Unit | Form |
|---|---|---|
| New burned area and total footprint | km² | low / likely / high (10th / 50th / 90th percentile) |
| Movement of the fire centre | km east, km north | low / likely / high |
| Spread direction and rate, predicted centre and radius | degrees, km/day, lat/lon, km | from the likely movement and footprint |
| Mean and peak fire power of a VIIRS pass | MW | low / likely / high |
| Chance the fire is out | 0 to 1 | probability |
| Chance it merges with a separate neighbouring fire | 0 to 1 | probability |
| Burn chance of each 1 km cell around the fire | 0 to 1 | spread map |

Two models are combined:

| Model | What it is | Used for |
|---|---|---|
| XGBoost | Gradient-boosted trees on the GPU, one quantile model per output and day ahead (42 models) | Every output |
| Sequence network | A GRU that reads the fire's previous 4 fire-days, followed by dense layers; predicts all outputs and days at once. Trained with PyTorch, run with numpy (`seq_numpy.py`), so the engine needs no PyTorch | Averaged with XGBoost for growth, fire power, movement and merging. The chance a fire is out comes from XGBoost alone, because averaging made it less well calibrated |

Both use the same 118 features:

| Group | Features |
|---|---|
| Fire now | Size, footprint, growth, fire power, pixels, passes, clusters, front distance, age, merges so far |
| History | The previous 4 fire-days: growth, power, separate parts, days back; 3-day growth and power trends |
| Separate parts | How many not-yet-joined fires the fire consists of that day, and the gap between them. Merges happen between these parts |
| Neighbours | Distance and gap to the nearest other fire, its size, growth and power, fires within 10 and 25 km |
| Place | Latitude, region, season, elevation, local 10-year fire frequency |
| Weather, days d to d+3 | Temperature, humidity, vapour pressure deficit, wind, gusts and wind direction, rain, hours since rain, soil moisture at three depths, wind blowing towards the nearest fire |

The models train on **forecast-like weather**: each training row's weather for days +1 to +3 gets a real forecast error added, drawn from the `forecast_errors.py` pool for the same day ahead and region. That way the models trust a forecast only as much as forecasts deserve. Days +1 to +3 also use day d's soil moisture, as in real use.

The spread map takes today's satellite pixels, moves and spreads them with the predicted centre shift and footprint, and turns each 1 km cell's distance and position into a burn chance with a logistic model. It was calibrated on 2024 and checked on 2025–2026: AUC 0.73–0.75, and cells marked 30% burned about 30% of the time.

The training data runs from 2016 to 2026: 333,452 fire-days from 188,559 fires. For testing it was split by time: train on 2016–2023, stop training on 2024, test on 2025–2026. The production models are then trained on all years, keeping 5% of fires aside only to decide when to stop.

## Accuracy

Measured on the 2025–2026 test years, which training never saw: 49,378 fire-days from 28,657 fires. Persistence means "the fire keeps doing what it did today". The production models use the same setup trained on all years, so these figures are the best estimate for them.

| Accuracy %, day +1 / +2 / +3 | With weather | No weather | Persistence |
|---|---|---|---|
| **Overall** (mean of the five rows below it) | **65.4 / 65.6 / 65.4** | 64.7 / 64.1 / 64.1 | 34.1 / 30.7 / 28.9 |
| *(this table: XGBoost alone; the blend is compared below)* | | | |
| Growth class right (none / small / large) | 79 / 82 / 84 | 78 / 79 / 81 | 26 / 23 / 20 |
| Right about extinction | 83 / 84 / 85 | 81 / 81 / 82 | 42 / 33 / 26 |
| Merge, balanced accuracy | 87 / 87 / 87 | 87 / 87 / 86 | 50 |
| Spread direction within 45° | 39 / 40 / 37 | 39 / 39 / 38 | 20 / 19 / 19 |
| Fire power within a factor of 2 | 38 / 36 / 35 | 37 / 35 / 34 | 32 / 29 / 29 |
| Real growth inside the low–high range | 88 / 87 / 88 | 88 / 88 / 89 | – |

- **Strong:** growth class, extinction and merges. Extinction AUC is 0.91–0.92 and merge AUC 0.94–0.95.
- **Calibrated chances:** predicted chances are within 0.6–0.9 percentage points of how often things happened.
- **Weak:** spread direction and fire power, though both are about twice as good as persistence.
- **Merges:** caught 86–90% of the time, but only 22–34% of merge alarms are right.

### XGBoost, sequence network and the blend

Same test years, with weather:

| Overall accuracy %, day +1 / +2 / +3 | |
|---|---|
| XGBoost | 65.4 / 65.6 / 65.4 |
| Sequence network | 64.9 / 65.4 / 65.6 |
| Blend, averaging both | **65.8 / 65.8 / 65.8** |
| **Blend with extinction from XGBoost (production)** | **about 65.9 / 65.9 / 65.8** |

The network is slightly better than XGBoost on growth, fire power and merges and worse at extinction (AUC 0.87–0.89 against 0.91–0.92). The blend beats both. The production row is worked out from the per-output results of the two rows above it. The blend passes 10 of 11 behaviour checks, including turning the spread when the wind turns.

### With real forecasts

On 600 fire-days from 2024–2026, predicted again with the weather forecasts that were actually issued at the time:

| Overall accuracy %, day +1 / +2 / +3 | |
|---|---|
| Observed weather | 67.0 / 65.7 / 66.4 |
| **Issued forecasts (real use)** | **65.0 / 63.8 / 64.8** |
| No-weather models | 65.2 / 63.1 / 64.4 |
| Persistence | 35.8 / 33.3 / 31.9 |

Training on forecast-like weather raised real-use accuracy by 0.3–1.7 points over the earlier models. With real forecasts, the weather models now match or beat the no-weather models on every day (day +1 is a tie within this sample's ±2 points).

### Robustness

The models were retrained without one year or one world region and tested on what was left out. These check models use a quicker setup (learning rate 0.1, days +1 and +3 only).

| Left out of training | Overall accuracy %, day +1 / +3 |
|---|---|
| Year 2024 | 65.5 / 64.7 |
| Americas | 62.5 / 62.5 |
| Africa | 64.9 / 64.5 |

Accuracy is stable from year to year. In a region the models never saw, it drops by up to 3 points; the Americas, with the least training data, are the hardest.

### Behaviour

10 of 11 behaviour checks pass. More wind or drier air gives more growth, drier air gives more fire power, rain raises the chance of extinction and cuts growth, and turning the wind turns the predicted spread the same way. The one failure: the raw growth model sometimes outputs tiny negative values, which the engine sets to zero before writing predictions.

Full results are in `data/prediction/accuracy/`, produced by `test_prediction_accuracy.py`.

## Data layout

```
data/
  archive/                 10-year scan: raw FIRMS files, frequency map, biggest and selected fires
  runs/<run>/              one folder per pipeline run
    correlation/ interpreter/ localisation/ fire_power/ epicenter/ expansion/
    kg/                    knowledge graph, Neo4j-import CSV
    prediction/            predictions, spread map, merge outlook, animations
    timings.json
  prediction/
    fire_days.csv.gz       training table
    models/weather/        21 models trained with weather (production)
    models/no_weather/     21 models trained without weather (production)
    models/spread.json     spread map calibration
    models_seq_full/       sequence network trained on all years (weights.npz for numpy, model.pt for PyTorch)
    models_split/          earlier models trained on 2016-2023, kept for comparison
    forecast_residuals.csv.gz  real forecast errors used in training
    model_card.json        training scores
    merges.csv.gz          observed merges
    accuracy/              accuracy test results
  weather_cache/           cached Open-Meteo responses
```

The repository ships the production models in `models/v2/`: `weather/` and `no_weather/` (21 compressed XGBoost models each), `seq/` (the sequence network as numpy arrays), `spread.json` and `model_card.json`. The engine uses `data/prediction/models/` when you have trained models locally, and `models/v2/` otherwise.

`data/` is git-ignored: it holds about 1 GB of models and tables plus the raw archive. Rebuild it with the commands in [Usage](#usage).

## Testing

Offline checks need no network or API key:

```bash
python test_correlation.py
python test_interpreter.py
python test_localisation.py
python test_fire_power.py
python test_epicenter.py
python test_expansion.py
python test_weather.py
python test_prediction.py
```

Accuracy tests of the prediction engine need the training table and trained models. `PRED_MODELS=<folder>` tests another model folder, and `PRED_SOIL_FROM_D=1` gives days +1 to +3 day d's soil moisture, as the production models expect:

```bash
python test_prediction_accuracy.py heldout
python test_prediction_accuracy.py behaviour
python test_prediction_accuracy.py cv
python test_prediction_accuracy.py forecast
```

| Test | Checks |
|---|---|
| `heldout` | Accuracy on 2025–2026, split by region, fire size, season and weather coverage, plus calibration |
| `behaviour` | Response to wind, humidity and rain changes, plus output rules |
| `cv` | Retrains without one year or one world region and tests on what was left out (GPU) |
| `forecast` | Accuracy when real issued forecasts replace observed weather (downloads archived forecasts from Open-Meteo) |

`python run_test_plan.py` runs the harmonisation test plan in `test.html` against the running server.

## Limitations

- **Shape:** the engine predicts each fire's area, centre shift, direction and power, not its outline.
- **Forecast errors are simulated in training:** they are drawn from real forecast errors at about 200 cells, which approximates but does not replace training on archived forecasts.
- **Weather coverage:** only about half of the training rows have weather. Small fires outside the fetched weather cells have none.
- **First day:** a fire seen for the first time has no history, so its predictions are less reliable.
- **Soil moisture:** days +1 to +3 use day d's value, because the forecast archive has none.
- **Spread map:** satellites only see actively burning pixels at overpass time, so only 3–5% of nearby cells register as burning on a given day. That limits how sharp the burn-chance map can be.
- **Training the network needs PyTorch** with CUDA (`pip install torch --index-url https://download.pytorch.org/whl/cu128`); running predictions does not.
- **Short runs:** runs shorter than 4 days produce no training rows for the 3-day horizon.

## Data sources

- Fire detections: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (MODIS, VIIRS, Landsat)
- Weather for the knowledge graph and training: [Open-Meteo](https://open-meteo.com/), with ERA5 reanalysis and archived ECMWF IFS forecasts
- Weather forecast for predictions: provided by the caller
- Fire danger indices (optional, off by default): Copernicus Emergency Management Service

## License

No license file yet. Until one is added, all rights are reserved by the authors.
