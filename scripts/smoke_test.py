"""Smoke test and compute-time estimate (protocol section 15, step 1).

Uses the train split only, on one cross-validation fold (validation block 2001-2003).
It prints timings, not validation errors, so it does not inform tuning.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import features as F  # noqa: E402

VAL_BLOCK = ("2001-01-01", "2003-12-31 23:00")
GAP = pd.Timedelta(days=27)
HORIZON = 1

# number of training runs in the protocol (sections 7.2, 9)
XGB_RUNS = 12 * 5 * 2 + 5 * 2 + 4 * 5 * 2      # tuning + final models + ablations A1-A4
LSTM_RUNS = 12 * 5 * 2 + 5 * 2 + 5 * 2 + 2     # tuning + final models + ablation A1 + X4


def fold_masks(t: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = pd.Timestamp(VAL_BLOCK[0], tz="UTC"), pd.Timestamp(VAL_BLOCK[1], tz="UTC")
    val = ((t >= lo) & (t <= hi)).to_numpy()
    train = ((t < lo - GAP) | (t > hi + GAP)).to_numpy()
    return train, val


def hms(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def smoke_xgboost(df: pd.DataFrame) -> float:
    import xgboost as xgb

    feats, y = F.build_features(F.fill_gaps(df)), F.targets(df, (HORIZON,))[f"y_h{HORIZON}"]
    train, val = fold_masks(df.t_start)
    ok = y.notna().to_numpy()
    dtrain = xgb.DMatrix(feats.loc[train & ok, F.FEATURES], label=y[train & ok])
    dval = xgb.DMatrix(feats.loc[val & ok, F.FEATURES], label=y[val & ok])
    times = []
    for depth, lr in ((6, 0.1), (8, 0.03)):  # a typical and the slowest configuration of the grid
        params = {
            "objective": "reg:squarederror", "max_depth": depth, "eta": lr, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "device": "cpu", "seed": 100,
        }
        t0 = time.perf_counter()
        booster = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dval, "val")],
                            early_stopping_rounds=50, verbose_eval=False)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  XGBoost depth={depth} lr={lr}: {hms(dt)}, best iteration {booster.best_iteration}")
    return float(np.mean(times))


def smoke_lstm(df: pd.DataFrame, epochs: int = 2) -> float:
    import torch
    from torch import nn

    torch.set_num_threads(os.cpu_count() or 1)
    filled = F.fill_gaps(df)
    x, valid = F.lstm_windows(filled)
    y = F.targets(df, (HORIZON,))[f"y_h{HORIZON}"].to_numpy(dtype=np.float32)
    train, _ = fold_masks(df.t_start)
    rows = train & valid & ~np.isnan(y)
    xtr = x[rows]
    mean, std = xtr.reshape(-1, xtr.shape[-1]).mean(0), xtr.reshape(-1, xtr.shape[-1]).std(0)
    xt = torch.from_numpy((xtr - mean) / std)
    yt = torch.from_numpy(y[rows]).unsqueeze(1)

    class Net(nn.Module):
        def __init__(self, hidden, layers):
            super().__init__()
            self.lstm = nn.LSTM(xt.shape[-1], hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, a):
            out, _ = self.lstm(a)
            return self.head(out[:, -1])

    per_epoch = []
    for hidden, layers in ((64, 1), (128, 2)):  # a typical and the slowest configuration of the grid
        torch.manual_seed(100)
        net = Net(hidden, layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        t0 = time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for b in range(0, len(xt), 256):
                idx = perm[b:b + 256]
                opt.zero_grad()
                loss = loss_fn(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        dt = (time.perf_counter() - t0) / epochs
        per_epoch.append(dt)
        print(f"  LSTM hidden={hidden} layers={layers}: {hms(dt)} per epoch ({len(xt)} windows)")
    return float(np.mean(per_epoch))


def main() -> None:
    import torch
    import xgboost

    print(f"Python {platform.python_version()}, CPU threads {os.cpu_count()}, "
          f"xgboost {xgboost.__version__}, torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    df = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    print(f"Train split: {len(df)} hours; fold with validation block {VAL_BLOCK[0][:4]}-{VAL_BLOCK[1][:4]}")

    print("\nXGBoost (h = 1 h):")
    xgb_run = smoke_xgboost(df)
    print("\nLSTM (h = 1 h):")
    lstm_epoch = smoke_lstm(df)

    print("\nEstimate for the whole protocol (training runs only):")
    print(f"  XGBoost: {XGB_RUNS} runs x {hms(xgb_run)} = {hms(XGB_RUNS * xgb_run)}")
    for ep in (20, 50, 100):
        print(f"  LSTM:    {LSTM_RUNS} runs x {ep} epochs = {hms(LSTM_RUNS * ep * lstm_epoch)}")


if __name__ == "__main__":
    main()
