"""Intervals for the main evaluation (protocol section 7.3).

Rows are the hourly forecasts of calib and test in time order (2010-01-01 onwards). Calibration
residuals are the common-set rows issued from 2010-01-28 whose target hour lies in calib. C-split
and C-Mond use these residuals; the adaptive methods start on 2010-01-01 with alpha = level
and equal weights and run through calib and on through test without reset, scored only on rows
of the common set. Results are kept for test rows of the common set; other rows get NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25 import intervals, mondrian

METHODS = ("split", "mond", "dtaci", "aci_0.005", "aci_0.05")


def residual_rows(t_issue: pd.Series, h: int, common: np.ndarray, calib_end: pd.Timestamp) -> np.ndarray:
    t = pd.DatetimeIndex(t_issue)
    target = t + pd.Timedelta(hours=h)
    return np.asarray(common, bool) & np.asarray(t >= mondrian.CALIB_RESIDUALS_FROM) & np.asarray(target <= calib_end)


def run(t_issue: pd.Series, y: np.ndarray, forecast: np.ndarray, dst_t: np.ndarray, common: np.ndarray,
        is_test: np.ndarray, h: int, calib_end: pd.Timestamp, merged_into: list[int],
        alpha: float = intervals.ALPHA, methods=METHODS) -> dict:
    """Half-width q and coverage (1/0) of each method, per row; NaN outside test rows of the
    common set. Also the number of test rows with an infinite or empty adaptive interval."""
    common = np.asarray(common, bool)
    is_test = np.asarray(is_test, bool)
    score = np.abs(np.asarray(y, float) - np.asarray(forecast, float))
    res_rows = residual_rows(t_issue, h, common, calib_end)
    if (res_rows & is_test).any():
        raise ValueError("a calibration residual lies in the test period")
    res = score[res_rows]
    ev = is_test & common
    out = {}

    def keep(q):
        q = np.where(ev, q, np.nan)
        cov = np.where(ev, (score <= q).astype(float), np.nan)
        return {"q": q, "covered": cov}

    if "split" in methods:
        out["split"] = keep(np.full(len(score), intervals.split_q(res, alpha)))
    if "mond" in methods:
        cls = mondrian.classes(dst_t)
        q = np.full(len(score), np.nan)
        q[ev] = intervals.mondrian_q(res, cls[res_rows], cls[ev], merged_into, alpha)
        out["mond"] = keep(q)
    adaptive = {"dtaci": intervals.DTACI_GAMMAS, "aci_0.005": (0.005,), "aci_0.05": (0.05,)}
    for name, gammas in adaptive.items():
        if name not in methods:
            continue
        r = intervals.online(score, common, h, res, gammas=gammas, alpha=alpha,
                             eta=intervals.dtaci_eta(len(gammas), alpha=alpha))
        d = keep(r["q"])
        d["alpha_t"] = np.where(ev, r["alpha_t"], np.nan)
        d["n_infinite"] = int(np.sum(ev & (r["q"] == np.inf)))
        d["n_empty"] = int(np.sum(ev & (r["q"] == -np.inf)))
        out[name] = d
    return out
