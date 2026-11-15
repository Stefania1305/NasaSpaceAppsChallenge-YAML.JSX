"""The check that found the extinction problem: what really happens to fires by size, against what each model says.

On the 2025-2026 test years, per fire size (under 1, 1-10, 10-100, over 100 km2) and day ahead: share of fires
really out / growing by at least 0.1 km2 / mean growth, against the model's share predicted out (chance >= 50 %),
share whose likely growth is at least 0.1 km2, its chance it grows, and its mean likely and expected growth.
Uses the fire-chain labels (fire_days_v3.csv.gz), so a fire that vanishes and comes back is not counted as out.

Usage:  python test_small_fires.py
Output: data/prediction/accuracy/small_fires.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from firms import DATA_DIR
from prediction_data import H, add_history
import prediction_metrics as PM

P_DIR = DATA_DIR / "prediction"
SIZES = [(0, 1, "under 1 km2"), (1, 10, "1-10 km2"), (10, 100, "10-100 km2"), (100, 1e9, "over 100 km2")]


def main():
    import prediction_v3 as V
    from seq_numpy import NumpySeq
    df = add_history(pd.read_csv(P_DIR / "fire_days_v3.csv.gz", low_memory=False))
    te = df[df.split == "test"].copy()
    soil = te.copy()
    for k in range(1, H + 1):
        for c in ("sm0", "sm1", "sm2"):
            soil[f"w{k}_{c}"] = soil[f"w0_{c}"]
    models = {
        "v2": PM.blend(PM.xgb_predict(PM.xgb_models(P_DIR / "models_v2", "weather"), soil),
                       NumpySeq.load(P_DIR / "models_seq").predict_df(soil, weather=True)),
        "v3": PM.blend(V.predict(P_DIR / "models_v3", soil, "weather"),
                       NumpySeq.load(P_DIR / "models_seq_v3").predict_df(soil, weather=True)),
    }
    res = {}
    for k in range(1, H + 1):
        g = te[f"growth_{k}"].to_numpy()
        out = te[f"out_{k}"].to_numpy("float64")
        for lo, hi, name in SIZES:
            m = ((te.fine_fp >= lo) & (te.fine_fp < hi)).to_numpy()
            ko = m & ~np.isnan(out)
            row = {"fire_days": int(m.sum()), "real_out_%": round(100 * float(out[ko].mean()), 1),
                   "real_grew_%": round(100 * float((g[m] >= 0.1).mean()), 1), "real_mean_growth_km2": round(float(g[m].mean()), 2)}
            for mn, P in models.items():
                po, gl = P[f"out_{k}"], P[f"growth_{k}"][:, 1]
                row[f"{mn}_says_out_%"] = round(100 * float((po[ko] >= 0.5).mean()), 1)
                row[f"{mn}_mean_chance_out_%"] = round(100 * float(po[ko].mean()), 1)
                row[f"{mn}_likely_grew_%"] = round(100 * float((gl[m] >= 0.1).mean()), 1)
                row[f"{mn}_mean_likely_growth_km2"] = round(float(gl[m].mean()), 2)
                if f"grow_{k}" in P:
                    row[f"{mn}_mean_chance_grows_%"] = round(100 * float(P[f"grow_{k}"][m].mean()), 1)
                    row[f"{mn}_mean_expected_growth_km2"] = round(float(P[f"growth_mean_{k}"][m].mean()), 2)
            res[f"day+{k} {name}"] = row
            print(f"day +{k} {name:13} n={m.sum():6} | out real {row['real_out_%']:5} v2 {row['v2_says_out_%']:5} v3 {row['v3_says_out_%']:5} | "
                  f"grew real {row['real_grew_%']:5} v2 likely {row['v2_likely_grew_%']:5} v3 likely {row['v3_likely_grew_%']:5} "
                  f"v3 chance {row.get('v3_mean_chance_grows_%', '-')} | growth real {row['real_mean_growth_km2']} v3 expected {row.get('v3_mean_expected_growth_km2', '-')}",
                  flush=True)
    (DATA_DIR / "prediction" / "accuracy" / "small_fires.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
