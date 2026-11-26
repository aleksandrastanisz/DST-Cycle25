"""Reference models P (persistence) and E (O'Brien-McPherron AK2) from protocol section 7.

Both take gap-filled data (features.fill_gaps) on a complete hourly grid and return the forecast
issued at every row, aligned with the row (see the row convention in features.py). A row whose
required inputs are missing gets NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25.features import check_hourly

AK2_VBS_THRESHOLD = 0.5  # mV/m
AK2_Q_SLOPE = -4.4  # nT/h per mV/m
AK2_TAU_A = 2.4  # h
AK2_TAU_B = 9.74  # mV/m
AK2_TAU_C = 4.69  # mV/m
PRESSURE_B = 7.26  # nT / sqrt(nPa)
PRESSURE_C = 11.0  # nT


def persistence(df: pd.DataFrame, h: int) -> pd.Series:
    """Model P: Dst(t + h) = Dst(t). Requires Dst(t)."""
    check_hourly(df)
    return df["Dst"].rename(f"P_h{h}")


def ak2_injection(vbs: np.ndarray) -> np.ndarray:
    """Injection Q [nT/h] for VBs [mV/m]."""
    return np.where(vbs > AK2_VBS_THRESHOLD, AK2_Q_SLOPE * (vbs - AK2_VBS_THRESHOLD), 0.0)


def ak2_tau(vbs: np.ndarray) -> np.ndarray:
    """Decay time tau [h] for VBs [mV/m]."""
    return AK2_TAU_A * np.exp(AK2_TAU_B / (AK2_TAU_C + vbs))


def pressure_correct(dst: np.ndarray, pdyn: np.ndarray) -> np.ndarray:
    """Dst* = Dst - 7.26 sqrt(Pdyn) + 11."""
    return dst - PRESSURE_B * np.sqrt(pdyn) + PRESSURE_C


def pressure_uncorrect(dst_star: np.ndarray, pdyn: np.ndarray) -> np.ndarray:
    """Inverse of pressure_correct."""
    return dst_star + PRESSURE_B * np.sqrt(pdyn) - PRESSURE_C


def ak2_integrate(dst_star: np.ndarray, vbs: np.ndarray, steps: int) -> np.ndarray:
    """Explicit Euler with a 1 h step under driving held constant over all steps."""
    q = ak2_injection(vbs)
    tau = ak2_tau(vbs)
    x = np.asarray(dst_star, dtype=float).copy()
    for _ in range(steps):
        x = x + q - x / tau
    return x


def ak2(df: pd.DataFrame, h: int) -> pd.Series:
    """Model E: AK2 from Dst*(t) over h steps, solar wind held at its value at t.

    Requires Dst(t), V, Bz and Pdyn at t. Pdyn is held constant, so the same Pdyn converts
    Dst(t) to Dst*(t) and the forecast Dst*(t + h) back to Dst(t + h).
    """
    check_hourly(df)
    vbs = (df["V"] * (-df["Bz_gsm"]).clip(lower=0) * 1e-3).to_numpy()
    pdyn = df["Pdyn"].to_numpy()
    dst_star = pressure_correct(df["Dst"].to_numpy(), pdyn)
    forecast = pressure_uncorrect(ak2_integrate(dst_star, vbs, h), pdyn)
    return pd.Series(forecast, index=df.index, name=f"E_h{h}")
