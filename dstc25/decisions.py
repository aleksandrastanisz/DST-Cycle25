"""Hypothesis statistics, p-values and decisions (protocol sections 10.1-10.3).

Inputs are per-unit tables (dstc25.metrics.per_unit) keyed by (model, horizon):
  test[(m, h)]  - test units with columns se_m, se_p, cov_mond, cov_dtaci and `included`;
  calib[(m, h)] - calib intense windows with cross-fitted cov_mond (M9) and `included`.
Unit lists fix the order of the bootstrap count matrices:
  ids["quiet"], ids["intense"] for test, ids["calib_intense"] for M9.

H1: Delta = M1(quiet) - M1(intense) >= 0.05 and Holm p < 0.05.
H2: delta = M9(C-Mond) - M2(test; intense; C-Mond) >= 0.05 and Holm p < 0.05.
H3: D = M2(intense; C-DtACI) - M2(intense; C-Mond) >= 0.03, Holm p < 0.05 and
    M2(intense; C-DtACI) >= 0.85.
The p-value counts replicates on or below 0 (the null boundary for all three).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25 import bootstrap as bs
from dstc25 import metrics

THRESHOLD = {"H1": 0.05, "H2": 0.05, "H3": 0.03}
H3_MIN_COVERAGE = 0.85
ALPHA_FAMILY = 0.05
N_PERMUTATIONS = 100_000


def _inc(t: pd.DataFrame, ids: list[str], drop: str | None) -> np.ndarray:
    inc = t.loc[ids, "included"].to_numpy().copy()
    if drop is not None and drop in ids:
        inc[ids.index(drop)] = False
    return inc


def _m1(t, ids, inc):
    return metrics.skill(t.loc[ids, "se_m"].to_numpy()[inc], t.loc[ids, "se_p"].to_numpy()[inc])


def _m1_reps(t, ids, inc, counts):
    return 1 - bs.weighted_mean(counts, t.loc[ids, "se_m"], inc) / bs.weighted_mean(counts, t.loc[ids, "se_p"], inc)


def _cov(t, ids, inc, col):
    return metrics.coverage(t.loc[ids, col].to_numpy()[inc])


def statistics(test: dict, calib: dict, ids: dict, counts: dict, drop: str | None = None) -> dict:
    """Point estimates and bootstrap replicates of the three statistics for every (model, horizon).

    counts: "quiet" and "intense" (test, seed 2026), "calib_intense" (M9, seed 2028).
    drop: a test storm window removed from the analysis (S6)."""
    out = {"H1": {}, "H2": {}, "H3": {}}
    for key, t in test.items():
        iq, ii = _inc(t, ids["quiet"], drop), _inc(t, ids["intense"], drop)
        c = calib[key]
        ic = _inc(c, ids["calib_intense"], None)
        out["H1"][key] = {
            "estimate": _m1(t, ids["quiet"], iq) - _m1(t, ids["intense"], ii),
            "reps": _m1_reps(t, ids["quiet"], iq, counts["quiet"]) - _m1_reps(t, ids["intense"], ii, counts["intense"]),
        }
        m9 = _cov(c, ids["calib_intense"], ic, "cov_mond")
        m2_mond = _cov(t, ids["intense"], ii, "cov_mond")
        out["H2"][key] = {
            "estimate": m9 - m2_mond, "m9": m9, "m2_test": m2_mond,
            "reps": bs.weighted_mean(counts["calib_intense"], c.loc[ids["calib_intense"], "cov_mond"], ic)
            - bs.weighted_mean(counts["intense"], t.loc[ids["intense"], "cov_mond"], ii),
        }
        d = t.loc[ids["intense"], "cov_dtaci"].to_numpy() - t.loc[ids["intense"], "cov_mond"].to_numpy()
        m2_dtaci = _cov(t, ids["intense"], ii, "cov_dtaci")
        out["H3"][key] = {
            "estimate": m2_dtaci - m2_mond, "m2_dtaci": m2_dtaci, "m2_mond": m2_mond,
            "reps": bs.weighted_mean(counts["intense"], d, ii), "d_units": d[ii],
        }
    return out


def decide(stats: dict) -> dict:
    """p-values, Holm adjustment within each family, percentile intervals and decisions."""
    result = {}
    for hyp, family in stats.items():
        p = {k: bs.p_value(v["reps"]) for k, v in family.items()}
        adj = bs.holm(p)
        result[hyp] = {}
        for k, v in family.items():
            ok = v["estimate"] >= THRESHOLD[hyp] and adj[k] < ALPHA_FAMILY
            if hyp == "H3":
                ok = ok and v["m2_dtaci"] >= H3_MIN_COVERAGE
            extra = {x: v[x] for x in ("m9", "m2_test", "m2_dtaci", "m2_mond") if x in v}
            result[hyp][k] = {"estimate": float(v["estimate"]), "p": p[k], "p_holm": adj[k],
                              "ci95": bs.percentile_ci(v["reps"]), "supported": bool(ok), **extra}
    return result


def sign_flip_p(d: np.ndarray, n: int = N_PERMUTATIONS, seed: int = bs.SEED_PERMUTATION,
                chunk: int = 10_000) -> float:
    """One-sided sign-flip permutation p-value for mean(d) > 0: (b + 1) / (n + 1), where b counts
    random sign flips whose mean is at least the observed mean."""
    d = np.asarray(d, dtype=float)
    observed = d.mean()
    rng = np.random.default_rng(seed)
    b, done = 0, 0
    while done < n:
        m = min(chunk, n - done)
        signs = rng.choice((-1.0, 1.0), size=(m, len(d)))
        b += int(np.sum(signs @ d / len(d) >= observed - 1e-12))
        done += m
    return float((b + 1) / (n + 1))


def fragility(test: dict, calib: dict, ids: dict, counts: dict, base: dict) -> dict:
    """S6: test storm windows whose removal changes any decision, per hypothesis and key."""
    changes = {}
    for w in ids["intense"]:
        dec = decide(statistics(test, calib, ids, counts, drop=w))
        for hyp in dec:
            for k in dec[hyp]:
                if dec[hyp][k]["supported"] != base[hyp][k]["supported"]:
                    changes.setdefault(hyp, {}).setdefault(f"{k[0]}_h{k[1]}", []).append(w)
    return changes
