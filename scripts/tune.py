"""Tune models X and L by blocked cross-validation within train (protocol section 7.2).

Writes results/tuning/<model>_h<h>.json (every fold run and the summary) and out-of-fold
forecasts to results/tuning/oof/ for the R0 checks. An interrupted run resumes when started
again with the same arguments.

    python scripts/tune.py --model X
    python scripts/tune.py --model L --horizon 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import tuning  # noqa: E402
from dstc25.runinfo import session_info  # noqa: E402

OUT = ROOT / "results" / "tuning"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuning.MODELS, nargs="+", default=list(tuning.MODELS))
    parser.add_argument("--horizon", type=int, choices=tuning.HORIZONS, nargs="+", default=list(tuning.HORIZONS))
    args = parser.parse_args()

    session = session_info()
    if session["uncommitted_changes"]:
        print("Warning: tracked files have uncommitted changes; this is recorded in the results.")
    df = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    for model in args.model:
        for h in args.horizon:
            s = tuning.tune(df, model, h, OUT, session)
            c = s["configs"][s["selected"]]
            print(f"{model} h={h}: selected {s['selected_config']}, mean validation MSE "
                  f"{c['mean_val_mse']:.2f} nT^2, final iterations {s['final_iter']} "
                  f"(folds {c['best_iter']}), on grid edge: {s['on_grid_edge'] or 'none'}")


if __name__ == "__main__":
    main()
