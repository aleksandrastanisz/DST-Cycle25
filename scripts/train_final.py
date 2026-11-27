"""Train the final models X and L on all of train (protocol section 7).

For each model and horizon: the configuration selected in tuning, the number of trees or
epochs from tuning (median of the five folds), seeds 42-46. Models go to models/ (not in git);
results/final_models.json records the session, the settings and the SHA-256 of every file.
Models that already exist are not trained again, so an interrupted run resumes.

    python scripts/train_final.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import tuning  # noqa: E402
from dstc25.runinfo import session_info, sha256  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
MODELS_DIR = ROOT / "models"
RECORD = ROOT / "results" / "final_models.json"
SUFFIX = {"X": ".json", "L": ".pt"}


def model_path(model: str, h: int, seed: int) -> Path:
    return MODELS_DIR / f"{model}_h{h}_seed{seed}{SUFFIX[model]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuning.MODELS, nargs="+", default=list(tuning.MODELS))
    args = parser.parse_args()

    session = session_info()
    if session["uncommitted_changes"]:
        print("Warning: tracked files have uncommitted changes; this is recorded in the results.")
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {"sessions": [], "models": {}}
    record["sessions"].append(session)
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")

    for model in args.model:
        if model == "X":
            from dstc25 import xgb_model as M
        else:
            from dstc25 import lstm_model as M
        for h in tuning.HORIZONS:
            summary = json.loads((ROOT / "results" / "tuning" / f"{model}_h{h}.json").read_text())["summary"]
            config, n_iter = summary["selected_config"], summary["final_iter"]
            x, y, usable = tuning.prepare(df, model, h)
            x_tr, y_tr = tuning.take(x, usable), y[usable]
            for seed in SEEDS:
                path = model_path(model, h, seed)
                name = path.name
                if path.exists() and name in record["models"]:
                    continue
                t0 = time.perf_counter()
                if model == "X":
                    m = M.fit(x_tr, y_tr, config, seed, n_trees=n_iter)
                else:
                    m = M.fit(x_tr, y_tr, config, seed, n_epochs=n_iter)
                m.save(path)
                record["models"][name] = {
                    "model": model, "horizon": h, "seed": seed, "config": config,
                    "iterations": n_iter, "train_rows": int(usable.sum()),
                    "session": len(record["sessions"]) - 1, "sha256": sha256(path),
                }
                RECORD.write_text(json.dumps(record, indent=1))
                print(f"{name}: {n_iter} {'trees' if model == 'X' else 'epochs'}, "
                      f"{int(usable.sum())} rows, {(time.perf_counter() - t0) / 60:.1f} min")
    RECORD.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
