"""Inputs, targets and features from protocol sections 5.2, 7 and 7.1.

Row convention: the record with t_start = s holds the mean over [s, s + 1 h). A forecast is
issued at t = s + 1 h, the end of that hour, so row i is the "issue row":
  - Dst(t) (lag 0) is the value in row i, lag k is row i - k;
  - the target at horizon h, Dst over [t + h - 1 h, t + h), is row i + h.
All functions take a DataFrame on a complete hourly grid sorted by t_start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_WIND = ["V", "Bz_gsm", "By_gsm", "B", "n", "Pdyn"]
FILL_LIMIT_H = 3
DST_LAGS = (0, 1, 2, 3, 6)
SW_LAGS = (0, 1, 2, 3)
LSTM_CHANNELS = ["V", "Bz_gsm", "By_gsm", "B", "n", "Pdyn", "VBs", "Dst"]
LSTM_WINDOW_H = 24

FEATURE_GROUPS = {
    "ring_current": [f"Dst_lag{k}" for k in DST_LAGS] + ["dDst_1h", "dDst_3h"],
    "solar_wind": [f"{v}_lag{k}" for v in SOLAR_WIND for k in SW_LAGS],
    "coupling": ["VBs", "Newell", "sqrtPdyn"],
    "driving_history": (
        [f"{v}_mean{w}h" for v in ("VBs", "Bz_gsm", "Pdyn", "Newell") for w in (3, 6)]
        + ["VBs_sum6h", "VBs_sum12h"]
    ),
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def check_hourly(df: pd.DataFrame) -> None:
    t = df["t_start"]
    if not t.is_monotonic_increasing or (t.diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("data must be on a complete, sorted hourly grid")


def fill_gaps(df: pd.DataFrame, columns=(*SOLAR_WIND, "Dst")) -> pd.DataFrame:
    """Fill gaps with the last known value, at most 3 h back and only from the past (section 7)."""
    out = df.copy()
    out[list(columns)] = out[list(columns)].ffill(limit=FILL_LIMIT_H)
    return out


def coupling(df: pd.DataFrame) -> pd.DataFrame:
    """VBs [mV/m], Newell coupling function and sqrt(Pdyn) for every row (section 7.1)."""
    bs = (-df["Bz_gsm"]).clip(lower=0)
    vbs = df["V"] * bs * 1e-3
    b_t = np.sqrt(df["By_gsm"] ** 2 + df["Bz_gsm"] ** 2)
    theta = np.arctan2(df["By_gsm"], df["Bz_gsm"])
    newell = df["V"] ** (4 / 3) * b_t ** (2 / 3) * np.abs(np.sin(theta / 2)) ** (8 / 3)
    return pd.DataFrame({"VBs": vbs, "Newell": newell, "sqrtPdyn": np.sqrt(df["Pdyn"])}, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 44 features of model X for every issue row. Expects gap-filled input (fill_gaps)."""
    check_hourly(df)
    c = coupling(df)
    base = pd.concat([df[SOLAR_WIND + ["Dst"]], c], axis=1)
    out = {}
    for k in DST_LAGS:
        out[f"Dst_lag{k}"] = base["Dst"].shift(k)
    out["dDst_1h"] = base["Dst"] - base["Dst"].shift(1)
    out["dDst_3h"] = base["Dst"] - base["Dst"].shift(3)
    for v in SOLAR_WIND:
        for k in SW_LAGS:
            out[f"{v}_lag{k}"] = base[v].shift(k)
    for v in ("VBs", "Newell", "sqrtPdyn"):
        out[v] = base[v]
    for v in ("VBs", "Bz_gsm", "Pdyn", "Newell"):
        for w in (3, 6):
            out[f"{v}_mean{w}h"] = base[v].rolling(w, min_periods=w).mean()  # complete windows only
    for w in (6, 12):
        out[f"VBs_sum{w}h"] = base["VBs"].rolling(w, min_periods=w).sum()
    feats = pd.DataFrame(out, index=df.index)[FEATURES]
    feats.insert(0, "t_start", df["t_start"])
    return feats


def targets(df: pd.DataFrame, horizons=(1, 6)) -> pd.DataFrame:
    """Target Dst for each horizon, aligned to the issue row (row i + h)."""
    check_hourly(df)
    out = pd.DataFrame({"t_start": df["t_start"]}, index=df.index)
    for h in horizons:
        out[f"y_h{h}"] = df["Dst"].shift(-h)
    return out


def lstm_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """24 h windows of the LSTM channels ending at each issue row (rows i - 23 .. i).

    Returns (X, valid): X has shape (n_rows, 24, n_channels); valid marks rows whose whole
    window is complete after gap filling (section 7: L needs a complete 24 h window).
    """
    check_hourly(df)
    base = pd.concat([df[SOLAR_WIND + ["Dst"]], coupling(df)[["VBs"]]], axis=1)[LSTM_CHANNELS]
    a = base.to_numpy(dtype=np.float32)
    n, w = len(a), LSTM_WINDOW_H
    x = np.full((n, w, a.shape[1]), np.nan, dtype=np.float32)
    if n >= w:
        x[w - 1:] = np.lib.stride_tricks.sliding_window_view(a, (w, a.shape[1]))[:, 0]
    valid = ~np.isnan(x).any(axis=(1, 2))
    return x, valid
