"""Per-unit statistics and the primary metrics (protocol sections 8 and 10.1).

A forecast row belongs to the unit of its target hour (dstc25.units.for_forecasts). A unit enters
the analysis when at least half of its target hours are in the common set (section 8). Every
metric is a mean over units with equal weight; the skill score aggregates errors, not ratios:
    M1 = 1 - mean_j MSE_model,j / mean_j MSE_P,j,      M2 = mean_j coverage_j.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_unit(unit: pd.Series, common: np.ndarray, unit_hours: pd.Series, values: dict) -> pd.DataFrame:
    """Mean of each array in `values` over the common hours of each unit.

    unit: unit id of every forecast row (None outside units); common: rows in the common set;
    unit_hours: number of target hours of each unit (index: unit id). Returns one row per unit
    of `unit_hours` with n_common, included and one column per value (NaN if no common hour).
    """
    rows = pd.DataFrame({"unit": unit.to_numpy()})
    for name, v in values.items():
        rows[name] = np.asarray(v, dtype=float)
    rows = rows[np.asarray(common) & rows["unit"].notna().to_numpy()]
    g = rows.groupby("unit")
    out = pd.DataFrame(index=unit_hours.index)
    out["n_hours"] = unit_hours
    out["n_common"] = g.size().reindex(out.index).fillna(0).astype(int)
    out["included"] = out["n_common"] >= 0.5 * out["n_hours"]
    for name in values:
        out[name] = g[name].mean().reindex(out.index)
    return out


def skill(mse_model: np.ndarray, mse_ref: np.ndarray) -> float:
    """M1 (or M6 with E as the reference): 1 - mean MSE of the model / mean MSE of the reference."""
    return float(1 - np.mean(mse_model) / np.mean(mse_ref))


def coverage(cov: np.ndarray) -> float:
    """M2: mean over units of the fraction of covered hours."""
    return float(np.mean(cov))
