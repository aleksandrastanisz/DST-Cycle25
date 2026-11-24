"""Storm definition from protocol section 6, computed from the Dst series alone."""
from __future__ import annotations

import pandas as pd

STORM_THRESHOLD = -50.0      # nT, a storm minimum must be at or below this
INTENSE_THRESHOLD = -100.0   # nT
SUPERINTENSE_THRESHOLD = -250.0   # nT
HALF_SEARCH_H = 24           # minimum must be the lowest value within +-24 h
WINDOW_BEFORE_H = 24
WINDOW_AFTER_H = 48

HOUR = pd.Timedelta(hours=1)


def _hourly(dst: pd.Series) -> pd.Series:
    """Return Dst on a complete hourly UTC grid (missing hours become NaN)."""
    if not isinstance(dst.index, pd.DatetimeIndex) or dst.index.tz is None:
        raise ValueError("dst must be indexed by tz-aware hourly timestamps (t_start)")
    full = pd.date_range(dst.index.min(), dst.index.max(), freq="h")
    return dst.sort_index().reindex(full)


def find_minima(dst: pd.Series) -> pd.Series:
    """Storm minima (protocol 6.1): Dst <= -50 nT and the lowest value in [t - 24 h, t + 24 h].

    Ties go to the earliest hour: a later hour with the same value is not a minimum,
    because it is not strictly lower than the previous 24 h.
    Returns the minimum Dst values indexed by their t_start.
    """
    d = _hourly(dst)
    window = 2 * HALF_SEARCH_H + 1
    centred_min = d.rolling(window, center=True, min_periods=1).min()
    previous_min = d.shift(1).rolling(HALF_SEARCH_H, min_periods=1).min()
    is_min = (d <= STORM_THRESHOLD) & (d == centred_min) & ~(previous_min <= d)
    return d[is_min]


def storm_windows(dst: pd.Series) -> pd.DataFrame:
    """Merged storm windows (protocol 6.2): [t_min - 24 h, t_min + 48 h], overlapping windows merged.

    Columns: start, end (inclusive t_start of first and last hour), t_min and dst_min of the
    lowest minimum, n_minima, category ('moderate' or 'intense') and superintense (bool).
    """
    minima = find_minima(dst)
    rows = []
    for t, value in minima.items():
        start, end = t - WINDOW_BEFORE_H * HOUR, t + WINDOW_AFTER_H * HOUR
        if rows and start <= rows[-1]["end"]:
            last = rows[-1]
            last["end"] = max(last["end"], end)
            last["n_minima"] += 1
            if value < last["dst_min"]:
                last["dst_min"], last["t_min"] = value, t
        else:
            rows.append({"start": start, "end": end, "t_min": t, "dst_min": value, "n_minima": 1})
    out = pd.DataFrame(rows, columns=["start", "end", "t_min", "dst_min", "n_minima"])
    out["category"] = out.dst_min.le(INTENSE_THRESHOLD).map({True: "intense", False: "moderate"})
    out["superintense"] = out.dst_min.le(SUPERINTENSE_THRESHOLD)
    return out


def assign_to_split(windows: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """Windows whose lowest minimum lies in [lo, hi], clipped to that range (protocol 6.3)."""
    lo_t, hi_t = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
    w = windows[(windows.t_min >= lo_t) & (windows.t_min <= hi_t)].copy()
    w["start"] = w.start.clip(lower=lo_t)
    w["end"] = w.end.clip(upper=hi_t)
    return w.reset_index(drop=True)


def count_by_category(windows: pd.DataFrame) -> dict:
    return {
        "moderate": int((windows.category == "moderate").sum()),
        "intense": int((windows.category == "intense").sum()),
        "superintense": int(windows.superintense.sum()),
    }
