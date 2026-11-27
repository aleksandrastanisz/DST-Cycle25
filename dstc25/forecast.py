"""Forecasts of all models for one split (protocol section 7).

Inputs at the start of a split need earlier hours (lags up to 24 h and gap filling), so the
split is appended to the preceding data and inputs are built on the joined series; only rows of
the split are returned. Targets come from the joined series too, so a target that lies after
the end of the split is missing: no later split is ever read.

The ensemble forecast of X and L is the mean of the five seeds (section 7); the forecasts of the
single seeds are kept for the spread across seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dstc25 import baselines
from dstc25 import features as F
from dstc25 import lstm_model, xgb_model
from dstc25.runinfo import sha256

HORIZONS = (1, 6)
SEEDS = (42, 43, 44, 45, 46)


def verified_path(models_dir: Path, record: dict, name: str) -> Path:
    path = models_dir / name
    if sha256(path) != record["models"][name]["sha256"]:
        raise ValueError(f"{name} does not match its SHA-256 in final_models.json")
    return path


def forecast_split(history: pd.DataFrame, split: pd.DataFrame, models_dir: Path, record: dict) -> pd.DataFrame:
    """Forecasts of P, E, X and L at both horizons for every hour of `split`.

    `history` is the data immediately preceding `split` (its last hour is the hour before the
    first hour of the split).
    """
    df = pd.concat([history, split], ignore_index=True)
    F.check_hourly(df)
    rows = np.arange(len(history), len(df))
    filled = F.fill_gaps(df)
    feats = F.build_features(filled)[F.FEATURES].iloc[rows]
    windows, valid = F.lstm_windows(filled)
    windows, valid = windows[rows], valid[rows]

    out = pd.DataFrame({"t_start": df["t_start"].iloc[rows].to_numpy(),
                        "dst_t": filled["Dst"].iloc[rows].to_numpy()})
    for h in HORIZONS:
        out[f"y_h{h}"] = F.targets(df, (h,))[f"y_h{h}"].iloc[rows].to_numpy()
        out[f"P_h{h}"] = baselines.persistence(filled, h).iloc[rows].to_numpy()
        out[f"E_h{h}"] = baselines.ak2(filled, h).iloc[rows].to_numpy()
        for seed in SEEDS:
            m = xgb_model.load(verified_path(models_dir, record, f"X_h{h}_seed{seed}.json"))
            out[f"X_h{h}_seed{seed}"] = m.predict(feats)
            m = lstm_model.load(verified_path(models_dir, record, f"L_h{h}_seed{seed}.pt"))
            pred = np.full(len(rows), np.nan)
            pred[valid] = m.predict(windows[valid])
            out[f"L_h{h}_seed{seed}"] = pred
        for model in ("X", "L"):
            out[f"{model}_h{h}"] = out[[f"{model}_h{h}_seed{s}" for s in SEEDS]].mean(axis=1, skipna=False)
        out[f"common_h{h}"] = out[[f"y_h{h}"] + [f"{m}_h{h}" for m in "PEXL"]].notna().all(axis=1)
    return out


def load_record(path: Path) -> dict:
    return json.loads(path.read_text())
