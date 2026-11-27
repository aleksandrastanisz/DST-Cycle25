"""Forecasts of all models on calib (2010-2019) with the final models trained on train.

Reads train and calib only. Writes results/forecasts/calib.parquet (not in git) and prints the
number of hours with a forecast of each model and in the common set.

    python scripts/forecast_calib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import forecast  # noqa: E402
from dstc25.runinfo import session_info, sha256  # noqa: E402

OUT = ROOT / "results" / "forecasts" / "calib.parquet"


def main() -> None:
    session = session_info()
    if session["uncommitted_changes"]:
        print("Warning: tracked files have uncommitted changes; this is recorded in the results.")
    record = forecast.load_record(ROOT / "results" / "final_models.json")
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    calib = pd.read_parquet(ROOT / "data" / "processed" / "calib.parquet")
    out = forecast.forecast_split(train, calib, ROOT / "models", record)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    summary = {"session": session, "hours": len(out), "file_sha256": sha256(OUT)}
    for h in forecast.HORIZONS:
        counts = {m: int(out[f"{m}_h{h}"].notna().sum()) for m in "PEXL"}
        counts["target"] = int(out[f"y_h{h}"].notna().sum())
        counts["common"] = int(out[f"common_h{h}"].sum())
        summary[f"h{h}"] = counts
        print(f"h={h}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" of {len(out)} hours")
    (OUT.parent / "calib_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
