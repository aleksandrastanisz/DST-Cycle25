"""Prediction intervals (protocol section 7.3): C-split, C-Mond, C-DtACI and C-ACI.

The nonconformity score is |Dst - forecast|; an interval is forecast +- q. A quantile of order
1 - beta of n calibration residuals is the ceil((n + 1)(1 - beta))-th smallest residual; when that
index exceeds n the interval is infinite (q = inf, always covered), and when beta >= 1 it is empty
(q = -inf, never covered). The same rule gives the protocol's conventions for alpha(t) <= 0 and
alpha(t) >= 1 in the adaptive methods.

Adaptive methods run online over rows in time order. A forecast issued at row i is scored when
its target is known, which is at row i + h (feedback delay, section 7.3): the state used for the
interval at row j contains the updates from rows i <= j - h only. An update uses the interval
that was actually issued at row i (the experts' levels at row i). Rows without a forecast give
no interval and no update; time still advances, so pending feedback from earlier rows arrives.
"""
from __future__ import annotations

import math

import numpy as np

ALPHA = 0.1
DTACI_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
DTACI_I = 72  # h, the storm-window length
ACI_GAMMAS = (0.005, 0.05)
_EPS = 1e-9


def dtaci_sigma(interval: int = DTACI_I) -> float:
    return 1 / (2 * interval)


def dtaci_eta(k: int = len(DTACI_GAMMAS), interval: int = DTACI_I, alpha: float = ALPHA) -> float:
    return math.sqrt(3 / interval) * math.sqrt((math.log(k * interval) + 2) / ((1 - alpha) ** 2 * alpha ** 2))


def quantile(sorted_res: np.ndarray, beta) -> np.ndarray:
    """Conformal quantile of order 1 - beta of sorted residuals (vectorized over beta)."""
    n = len(sorted_res)
    beta = np.asarray(beta, dtype=float)
    k = np.ceil((n + 1) * (1 - beta) - _EPS).astype(int)
    q = np.where(k > n, np.inf, -np.inf)
    inside = (k >= 1) & (k <= n) & (beta < 1)
    q[inside] = sorted_res[k[inside] - 1]
    q[beta >= 1] = -np.inf
    return q


def split_q(residuals: np.ndarray, alpha: float = ALPHA) -> float:
    """C-split: one half-width for all hours."""
    return float(quantile(np.sort(residuals), alpha))


def mondrian_q(residuals: np.ndarray, res_class: np.ndarray, query_class: np.ndarray,
               merged_into: list[int], alpha: float = ALPHA) -> np.ndarray:
    """C-Mond: half-width from the residuals of the (merged) class of each query hour.

    Classes are 0-3 (dstc25.mondrian.classes); merged_into maps a class to its merged class.
    Query hours with class -1 (missing Dst(t)) get NaN; such hours have no forecast of P anyway.
    """
    m = np.asarray(merged_into)
    rc, qc = m[res_class], np.where(query_class >= 0, m[np.maximum(query_class, 0)], -1)
    out = np.full(len(query_class), np.nan)
    for c in np.unique(qc[qc >= 0]):
        out[qc == c] = quantile(np.sort(residuals[rc == c]), alpha)
    return out


def beta_observed(sorted_res: np.ndarray, score: float) -> float:
    """sup{beta : score <= quantile(1 - beta)}: the largest level whose interval still covers."""
    n = len(sorted_res)
    m = int(np.searchsorted(sorted_res, score, side="left")) + 1  # rank of the score
    return 1 - (m - 1) / (n + 1)


def pinball(beta: float, theta: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    d = beta - theta
    return alpha * d - np.minimum(0.0, d)


def online(scores: np.ndarray, has: np.ndarray, h: int, calib_res: np.ndarray,
           gammas=DTACI_GAMMAS, alpha: float = ALPHA, sigma: float | None = None,
           eta: float | None = None) -> dict:
    """C-DtACI (deterministic, Gibbs and Candes 2024, Algorithm 2) with feedback delay h.

    With a single gamma this is C-ACI: alpha(t+1) = alpha(t) + gamma (alpha - err(t)).
    scores: |y - forecast| per row (used only once the target is known); has: rows with a
    forecast and a known target. Returns per row: alpha_t (the level used), q (half-width),
    covered (1/0) and NaN where there is no forecast.
    """
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)
    sigma = dtaci_sigma() if sigma is None else sigma
    eta = dtaci_eta(k) if eta is None else eta
    res = np.sort(np.asarray(calib_res, dtype=float))
    n = len(scores)

    a_exp = np.full(k, alpha)  # experts' levels
    w = np.ones(k)
    issued_exp = np.full((n, k), np.nan)
    alpha_t = np.full(n, np.nan)
    q = np.full(n, np.nan)
    covered = np.full(n, np.nan)

    for j in range(n):
        i = j - h
        if i >= 0 and has[i]:  # feedback from row i arrives now
            b = beta_observed(res, scores[i])
            lvl = issued_exp[i]
            loss = pinball(b, lvl, alpha)
            w_bar = w * np.exp(-eta * loss)
            w = (1 - sigma) * w_bar + w_bar.sum() * sigma / k
            w = w / w.sum()  # rescaling does not change p_t and avoids underflow
            err = (scores[i] > quantile(res, lvl)).astype(float)
            a_exp = a_exp + gammas * (alpha - err)
        if has[j]:
            issued_exp[j] = a_exp
            alpha_t[j] = float(np.dot(w / w.sum(), a_exp))
            q[j] = float(quantile(res, alpha_t[j]))
            covered[j] = float(scores[j] <= q[j])
    return {"alpha_t": alpha_t, "q": q, "covered": covered}
