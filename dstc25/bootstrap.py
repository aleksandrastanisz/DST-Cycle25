"""Bootstrap over units (protocol section 10.1).

A replicate draws units with replacement within one category. It is stored as counts: row b of
the count matrix says how many times each unit was drawn in replicate b. One count matrix per
category is shared by all models, horizons and interval methods, so comparisons are paired.

A statistic of a replicate is a weighted mean of per-unit values, the weights being the counts.
Units excluded for a given model and horizon (section 8, half-of-hours rule) are dropped from
the replicate, so the draws stay shared while each statistic uses only its included units.
"""
from __future__ import annotations

import numpy as np

B = 10_000
SEED_TEST = 2026
SEED_PERMUTATION = 2027
SEED_M9 = 2028
SEED_P0 = 2029


def draw_counts(rng: np.random.Generator, n_units: int, n_draws: int | None = None, b: int = B) -> np.ndarray:
    """Count matrix (b x n_units) of `n_draws` draws with replacement (default: n_units)."""
    n_draws = n_units if n_draws is None else n_draws
    idx = rng.integers(0, n_units, size=(b, n_draws))
    counts = np.zeros((b, n_units), dtype=np.int32)
    rows = np.repeat(np.arange(b), n_draws)
    np.add.at(counts, (rows, idx.ravel()), 1)
    return counts


def weighted_mean(counts: np.ndarray, values: np.ndarray, included: np.ndarray | None = None) -> np.ndarray:
    """Mean of per-unit values in each replicate, over drawn units that are included.

    A replicate that drew no included unit gives NaN."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v) if included is None else (np.asarray(included, bool) & np.isfinite(v))
    c = counts[:, keep].astype(float)
    total = c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, c @ v[keep] / total, np.nan)


def p_value(replicates: np.ndarray) -> float:
    """One-sided p = (b + 1) / (B + 1), b = replicates on or below the null boundary 0."""
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    return float((np.sum(r <= 0) + 1) / (len(r) + 1))


def percentile_ci(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def holm(p: dict) -> dict:
    """Holm-adjusted p-values within one family (keys: test names)."""
    names = sorted(p, key=lambda k: p[k])
    m = len(names)
    adjusted, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * p[name]))
        adjusted[name] = running
    return adjusted
