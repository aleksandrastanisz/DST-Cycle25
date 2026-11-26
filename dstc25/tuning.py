"""Tuning by blocked cross-validation within train (protocol section 7.2).

For one model and one horizon every configuration of the grid is fitted on each of the five
folds with seed 100 and early stopping on the fold's validation block. The record of each
(configuration, fold) run holds the validation MSE, the number of validation hours and the
best iteration (trees for X, epochs for L). Records are appended to a JSON file after every
run, so an interrupted tuning resumes where it stopped.

Each model is validated on the hours in which it issues a forecast and the target is known.
Configurations are only compared within one model, so this does not bias the selection.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dstc25 import cv
from dstc25 import features as F

MODELS = ("X", "L")
HORIZONS = (1, 6)


def grid(model: str) -> list[dict]:
    if model == "X":
        from dstc25.xgb_model import GRID
    else:
        from dstc25.lstm_model import GRID
    return GRID


def prepare(df: pd.DataFrame, model: str, h: int) -> tuple[object, np.ndarray, np.ndarray]:
    """Inputs, target and a mask of usable rows (model issues a forecast, target known)."""
    filled = F.fill_gaps(df)
    y = F.targets(df, (h,))[f"y_h{h}"].to_numpy(dtype=float)
    if model == "X":
        x = F.build_features(filled)[F.FEATURES]
        usable = ~np.isnan(y)
    else:
        x, valid = F.lstm_windows(filled)
        usable = valid & ~np.isnan(y)
    return x, y, usable


def take(x, rows: np.ndarray):
    return x.loc[rows] if isinstance(x, pd.DataFrame) else x[rows]


def run_fold(model: str, config: dict, x, y: np.ndarray, train: np.ndarray, val: np.ndarray):
    """Fit one configuration on one fold; return the fitted model and its record."""
    if model == "X":
        from dstc25 import xgb_model as M
    else:
        from dstc25 import lstm_model as M
    m = M.fit(take(x, train), y[train], config, cv.TUNING_SEED, x_val=take(x, val), y_val=y[val])
    pred = m.predict(take(x, val))
    best = m.n_trees if model == "X" else m.n_epochs
    record = {"val_mse": float(np.mean((pred - y[val]) ** 2)), "n_val": int(val.sum()), "best_iter": int(best)}
    return m, pred, record


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"runs": []}


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def summarize(runs: list[dict], configs: list[dict]) -> dict:
    """Mean validation MSE per configuration, the selected one and its final iteration count.

    The final count of trees or epochs is the median of the five best iterations of the
    selected configuration. A tie in mean MSE goes to the first configuration in grid order.
    """
    by_config = []
    for k, config in enumerate(configs):
        rs = sorted((r for r in runs if r["config"] == k), key=lambda r: r["block"])
        if len(rs) != len(cv.BLOCKS):
            raise ValueError(f"configuration {k} has {len(rs)} of {len(cv.BLOCKS)} folds")
        by_config.append({
            "config": config,
            "mean_val_mse": float(np.mean([r["val_mse"] for r in rs])),
            "fold_val_mse": [r["val_mse"] for r in rs],
            "best_iter": [r["best_iter"] for r in rs],
        })
    selected = int(np.argmin([c["mean_val_mse"] for c in by_config]))
    chosen = configs[selected]
    on_edge = [p for p in chosen
               if len({c[p] for c in configs}) > 2 and chosen[p] in (min(c[p] for c in configs), max(c[p] for c in configs))]
    return {
        "configs": by_config,
        "selected": selected,
        "selected_config": chosen,
        "final_iter": int(statistics.median(by_config[selected]["best_iter"])),
        "on_grid_edge": on_edge,
    }


def tune(df: pd.DataFrame, model: str, h: int, out_dir: Path, session: dict, log=print) -> dict:
    """Run (or resume) tuning of one model at one horizon; write the summary and OOF forecasts.

    `session` (commit, uncommitted-changes flag, start time) is appended to the record, so every
    resumed part of a tuning run is traceable (protocol section 14).
    """
    configs = grid(model)
    path = out_dir / f"{model}_h{h}.json"
    state = load(path)
    if state.get("configs_grid", configs) != configs:
        raise ValueError(f"{path} was written for a different grid")
    state["configs_grid"] = configs
    state.setdefault("sessions", []).append(session)
    done = {(r["config"], r["block"][0]) for r in state["runs"]}

    x, y, usable = prepare(df, model, h)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for k, config in enumerate(configs):
        oof_path = oof_dir / f"{model}_h{h}_config{k:02d}.parquet"
        oof = (pd.read_parquet(oof_path) if oof_path.exists()
               else pd.DataFrame({"t_start": df["t_start"], "pred": np.nan}))
        for block, train, val in cv.folds(df["t_start"]):
            if (k, block[0]) in done:
                continue
            t0 = time.perf_counter()
            tr, va = train & usable, val & usable
            _, pred, record = run_fold(model, config, x, y, tr, va)
            oof.loc[va, "pred"] = pred
            oof.to_parquet(oof_path, index=False)
            state["runs"].append({"config": k, "block": list(block), "session": len(state["sessions"]) - 1, **record})
            save(path, state)
            log(f"{model} h={h} config {k + 1}/{len(configs)} block {block[0]}-{block[1]}: "
                f"MSE {record['val_mse']:.2f} nT^2, best iteration {record['best_iter']}, "
                f"{(time.perf_counter() - t0) / 60:.1f} min")

    state["summary"] = summarize(state["runs"], configs)
    save(path, state)
    return state["summary"]
