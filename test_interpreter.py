"""Offline checks for interpreter.py on a small synthetic run (no network).

Run: python test_interpreter.py
"""
import json
import tempfile
from pathlib import Path

import pandas as pd

import correlation as C
import interpreter as I
from test_correlation import COLS, row


def build_run(tmp: Path) -> Path:
    rows = []
    # fire complex A: 40 touching VIIRS pixels in one pass (a 5 x 8 block, ~0.4 km apart), FRP rising with brightness
    for k in range(40):
        rows.append(row(10 + (k // 8) * 0.0036, 20 + (k % 8) * 0.0036, frp=str(2 + k), bt=str(320 + k), bt2=str(290 + k / 2)))
    # fire complex B, far away, plus a MODIS pixel over it
    for k in range(40):
        rows.append(row(-5 + (k // 8) * 0.0036, 30 + (k % 8) * 0.0036, frp=str(1 + k / 2), bt=str(315 + k), bt2=str(288 + k / 3)))
    rows.append(row(-5.001, 30.001, t="1545", sat="Aqua", inst="MODIS", conf="85", scan="1.0", track="1.0"))
    # a static land source from the file (sure) and an isolated vegetation fire
    rows.append(row(40, 50, typ="2"))
    rows[-1][COLS.index("type_existed")] = "yes"
    rows.append(row(45, 60))
    res = C.run_correlation(COLS, rows)
    run = tmp / "run"
    C.save(res, run / "correlation")
    return run


def test_plan():
    with tempfile.TemporaryDirectory() as d:
        run = build_run(Path(d))
        plan = I.interpret(run, n_packages=2)
        assert plan["row_filter"]["removed_non_wildfire"] == 1, plan["row_filter"]
        for name, spec in I.WORKERS.items():
            w = plan["workers"][name]
            missing = [f for f in spec["required"] if f not in w["fields"]]
            assert not missing, f"{name} missing required {missing}"
            for f in I.ALWAYS_EXCLUDED:
                assert f not in w["fields"], f"{name} got excluded field {f}"
            # units never split across packages
            seen = {}
            for p in w["packages"]:
                df = pd.read_csv(run / "interpreter" / name / f"package_{p['package']:03d}.csv.gz")
                assert "type_label" not in df and "row_id" in df
                assert not (df["type"] == 2).any(), "static source must be removed"
                for u in df["unit_id"].unique():
                    assert u not in seen, f"unit {u} split across packages"
                    seen[u] = p["package"]
        fp = plan["workers"]["fire_power"]
        assert "bright_diff_k" in fp["fields"] or "frp_physics_proxy" in fp["fields"], fp["why"]
        c = plan["checks"]
        assert c["links_inside_packages"] == c["links_between_kept_rows"] > 0, c
        assert json.loads((run / "interpreter" / "plan.json").read_text())["rows_in"] == 83


if __name__ == "__main__":
    test_plan()
    print("PASS test_plan")
