"""Cross-fitted C-Mond and C-DtACI on the two halves of calib (protocol section 7.3, M9).

Rows are split by target hour into 2010-2014 (half 0) and 2015-2019 (half 1), except rows of a
storm window that crosses the boundary, which go to the half containing the window's lowest
minimum (section 13, 2026-11-27). Intervals for one half use calibration residuals from the
other half only: residuals of the common set, in half 0 from issue time 2010-01-28. C-DtACI runs
online through the half from the initial state, with feedback delay h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25 import intervals, mondrian

BOUNDARY = pd.Timestamp("2015-01-01", tz="UTC")


def halves(t_issue: pd.Series, h: int, unit: pd.Series, units: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Half (0 or 1) of every row and the number of storm windows crossing the boundary."""
    target = pd.DatetimeIndex(t_issue) + pd.Timedelta(hours=h)
    half = np.asarray(target >= BOUNDARY).astype(int)
    storms = units[units.kind == "storm"]
    crossing = storms[(storms.start < BOUNDARY) & (storms.end >= BOUNDARY)]
    u = unit.to_numpy()
    for _, w in crossing.iterrows():
        half[u == w.unit] = int(w.t_min >= BOUNDARY)
    return half, len(crossing)


def coverage(t_issue: pd.Series, h: int, y: np.ndarray, forecast: np.ndarray, dst_t: np.ndarray,
             common: np.ndarray, half: np.ndarray, merged_into: list[int]) -> dict:
    """Cross-fitted coverage (1/0, NaN outside the common set) of C-Mond and C-DtACI per row."""
    score = np.abs(np.asarray(y, float) - np.asarray(forecast, float))
    eligible = np.asarray(common, bool) & (pd.DatetimeIndex(t_issue) >= mondrian.CALIB_RESIDUALS_FROM)
    cls = mondrian.classes(dst_t)
    cov = {"mond": np.full(len(y), np.nan), "dtaci": np.full(len(y), np.nan)}
    for this in (0, 1):
        res_rows = eligible & (half == 1 - this)
        rows = np.flatnonzero(half == this)  # time order
        ev = rows[np.asarray(common, bool)[rows]]
        q = intervals.mondrian_q(score[res_rows], cls[res_rows], cls[ev], merged_into)
        cov["mond"][ev] = (score[ev] <= q).astype(float)
        run = intervals.online(score[rows], np.asarray(common, bool)[rows], h, score[res_rows])
        cov["dtaci"][rows] = run["covered"]
    return cov
