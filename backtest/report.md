# Backtest: the app's 3-day fire prediction vs what really happened

Run on 15 Nov 2026 with `backtest/backtest.py`, through the app's own backend (the same routes the frontend calls):
pipeline on 5 days of data (correlation → interpreter → localisation + fire power → epicenter → fire expansion →
knowledge graph → prediction with the Open-Meteo forecast; model: **XGBoost blended with the sequence network**),
then the predictions for the next days compared with the **real harmonised FIRMS detections** of those days, which
the model never saw. Raw numbers: `backtest/results.json`.

| Test | Data the model saw | Predicted days checked |
|---|---|---|
| Today | 10–14 Nov | 15 Nov (+1) — **partial**: today's passes so far only (Brazil up to 12:29 UTC, Africa up to 14:48 UTC) |
| 3-day | 7–11 Nov | 12, 13, 14 Nov (+1, +2, +3) — complete days |

Zones: **Brazil** (Mato Grosso / Bolivia border, −62,−16 → −56,−10) and **Central Africa** (CAR / South Sudan
savanna, 22,7 → 25,9.5).

## 1. Will the fire go out? (chance out)

AUC: 0.5 = guessing, 1 = perfect ranking. Accuracy at 50% vs the baseline "every fire keeps burning".
Brier: lower is better, vs always predicting the real share of fires that went out (a baseline that already
knows the answer's average).

| Zone · day | Fires | Really out | Predicted out (avg) | AUC | Accuracy | "All keep burning" | Brier | Base-rate Brier |
|---|---|---|---|---|---|---|---|---|
| Brazil · today +1 | 32 | 66% | 46% | **0.75** | **0.72** | 0.34 | 0.231 | 0.226 |
| Brazil · +1 | 60 | 80% | 50% | 0.66 | 0.57 | 0.20 | 0.295 | 0.160 |
| Brazil · +2 | 60 | 85% | 63% | 0.66 | 0.65 | 0.15 | 0.219 | 0.127 |
| Brazil · +3 | 60 | 90% | 74% | **0.78** | **0.83** | 0.10 | 0.137 | 0.090 |
| Africa · today +1 | 325 | 60% | 64% | **0.70** | **0.68** | 0.40 | 0.237 | 0.240 |
| Africa · +1 | 322 | 47% | 67% | 0.70 | 0.65 | 0.53 | 0.282 | 0.249 |
| Africa · +2 | 322 | 56% | 73% | **0.72** | **0.71** | 0.44 | 0.231 | 0.246 |
| Africa · +3 | 322 | 68% | 78% | 0.65 | 0.70 | 0.32 | 0.232 | 0.218 |

**Reading:** the model clearly knows *which* fires are more likely to go out (AUC 0.65–0.78 everywhere) and beats
"every fire keeps burning" by a wide margin. Its *percentages* are off in opposite directions by region: in Brazil
fires died faster than it expected (it said 50% out on day +1, really 80%); in the African savanna they lasted longer
(it said 67%, really 47%). So the Brier score is only about as good as a constant base rate — the ranking is useful,
the absolute chance needs regional calibration.

## 2. Spread map (chance that each 1 km cell burns)

| Zone · day | Cells | Cells that burned | AUC |
|---|---|---|---|
| Brazil · today +1 | 621 | 20 | **0.83** |
| Brazil · +1 | 1,072 | 9 | **0.87** |
| Brazil · +2 | 838 | 5 | 0.48 |
| Brazil · +3 | 782 | 9 | 0.68 |
| Africa · today +1 | 6,639 | 357 | 0.61 |
| Africa · +1 | 6,406 | 298 | 0.67 |
| Africa · +2 | 5,372 | 257 | 0.55 |
| Africa · +3 | 5,346 | 215 | 0.47 |

**Reading:** good for the next day (AUC 0.83–0.87 in Brazil, 0.61–0.67 in Africa), close to guessing by day +2/+3.
Calibration is reasonable at low chances (Africa: cells predicted ~3% burned 3–4%, ~7% burned 6%), but the
highest-chance cells (20–40%) burned far less often than predicted (0–14%).

## 3. Movement and fire power (fires seen again)

| Zone · day | Fires seen again | Centre error, model | Centre error, "stays put" | Real FRP inside predicted range |
|---|---|---|---|---|
| Brazil · today +1 | 11 | 0.46 km | 0.47 km | 27% |
| Brazil · +1 / +2 / +3 | 7 / 4 / 6 | 0.75 / 0.66 / 0.96 km | 0.73 / 0.47 / 0.88 km | 57% / 75% / 67% |
| Africa · today +1 | 130 | 1.57 km | 1.65 km | **76%** |
| Africa · +1 / +2 / +3 | 97 / 79 / 103 | 1.60 / 1.85 / 1.70 km | 1.75 / 1.90 / 1.70 km | **81% / 84% / 82%** |

**Reading:** the predicted movement is no better than assuming the fire stays where it is (differences of
0.0–0.15 km). The fire-power range is well calibrated in Africa (76–84% of real values inside, the model aims for
~80%); Brazil has too few fires seen again to judge.

## 4. What the model does not cover

Most real detections on the predicted days were **new fires**, away from every fire the model predicted
(Brazil 66–76%, Africa 53–71% of each day's detections). The model only predicts fires already burning on the last
day; new ignitions are outside what it forecasts.

## Caveats

- **Today's truth is incomplete** (only the morning/early-afternoon passes so far); some fires counted "out" today
  may simply not have been seen yet. Rerun the backtest tomorrow for a complete 15 Nov.
- A fire counts as "seen again" if a real detection falls within its predicted reach (likely-high radius + 1.5 km,
  at least 2 km) of its predicted or current centre; satellites miss some fires (clouds, no pass), which looks like
  "out" here and in the model's training data alike.
- Two zones, two windows: a first check, not a full evaluation (the model's own held-out evaluation is in the
  `model-v2` README).
