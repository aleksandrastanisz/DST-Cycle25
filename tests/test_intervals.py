"""Intervals (protocol section 7.3), including the tests required in section 14: feedback delay in
C-ACI and C-DtACI, and agreement of C-DtACI with Algorithm 2 of Gibbs and Candes (2024)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import intervals as I  # noqa: E402


def test_conformal_quantile_index():
    r = np.arange(1.0, 10.0)  # n = 9: ceil(10 * 0.9) = 9 -> the largest
    assert I.split_q(r, 0.1) == 9.0
    assert I.split_q(np.arange(1.0, 9.0), 0.1) == math.inf  # n = 8: index 9 > n
    assert I.split_q(np.arange(1.0, 20.0), 0.1) == 18.0  # n = 19: ceil(18) = 18 despite rounding
    assert I.quantile(np.sort(r), 0.0) == math.inf and I.quantile(np.sort(r), -0.2) == math.inf
    assert I.quantile(np.sort(r), 1.0) == -math.inf and I.quantile(np.sort(r), 1.3) == -math.inf


def test_dtaci_constants_from_protocol():
    assert I.dtaci_sigma() == pytest.approx(0.0069, abs=1e-4)
    assert I.dtaci_eta() == pytest.approx(6.6, abs=0.05)


def test_split_coverage_exchangeable():
    # marginal coverage over calibration sets is exactly ceil(0.9 (n + 1)) / (n + 1) = 0.9 for n = 19
    rng = np.random.default_rng(0)
    data = np.abs(rng.normal(size=(40000, 20)))
    q = np.array([I.split_q(row[:19]) for row in data])
    assert np.mean(data[:, 19] <= q) == pytest.approx(0.9, abs=0.005)


def test_mondrian_per_class_and_merging():
    res = np.array([1.0] * 9 + [10.0] * 9)
    cls = np.array([0] * 9 + [3] * 9)
    q = I.mondrian_q(res, cls, np.array([0, 3, -1]), [0, 1, 2, 3])
    assert q[0] == 1.0 and q[1] == 10.0 and np.isnan(q[2])
    q = I.mondrian_q(res, cls, np.array([0, 3]), [0, 1, 2, 0])  # class 3 merged into 0
    assert q[0] == q[1] == 10.0  # 18 residuals: index ceil(19 * 0.9) = 18


def test_beta_observed_is_the_coverage_boundary():
    res = np.sort(np.random.default_rng(1).exponential(size=50))
    for s in (0.0, res[3], res[3] + 1e-6, 0.7, res[-1], 99.0):
        b = I.beta_observed(res, s)
        assert s <= I.quantile(res, b - 1e-9)       # just below beta: covered
        assert not s <= I.quantile(res, b + 1e-9)   # just above beta: not covered


def reference_dtaci(scores, calib, gammas, alpha, sigma, eta):
    """Algorithm 2 of Gibbs and Candes (2024) written out plainly, without delay (h = 1 means the
    score of step t is known before step t + 1). Independent of dstc25.intervals.online."""
    res = np.sort(calib)
    n, k = len(res), len(gammas)

    def q_of(b):
        if b >= 1:
            return -math.inf
        idx = math.ceil((n + 1) * (1 - b) - 1e-9)
        return math.inf if idx > n else res[idx - 1]

    a = [alpha] * k
    w = [1.0] * k
    out = []
    for s in scores:
        p = [wi / sum(w) for wi in w]
        a_t = sum(pi * ai for pi, ai in zip(p, a))
        out.append(a_t)
        rank = sum(r < s for r in res) + 1
        beta = 1 - (rank - 1) / (n + 1)
        loss = [alpha * (beta - ai) - min(0.0, beta - ai) for ai in a]
        w_bar = [wi * math.exp(-eta * li) for wi, li in zip(w, loss)]
        W = sum(w_bar)
        w = [(1 - sigma) * wb + W * sigma / k for wb in w_bar]
        err = [float(s > q_of(ai)) for ai in a]
        a = [ai + g * (alpha - e) for ai, g, e in zip(a, gammas, err)]
    return np.array(out)


def test_dtaci_matches_algorithm_2():
    rng = np.random.default_rng(2)
    calib = np.abs(rng.normal(size=300))
    scores = np.abs(rng.normal(size=600)) * np.r_[np.ones(300), 2.5 * np.ones(300)]  # a shift
    got = I.online(scores, np.ones(600, bool), 1, calib)["alpha_t"]
    want = reference_dtaci(scores, calib, I.DTACI_GAMMAS, I.ALPHA, I.dtaci_sigma(), I.dtaci_eta())
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_aci_single_expert_update():
    calib = np.arange(1.0, 20.0)  # q(0.1) = 18
    scores = np.array([30.0, 1.0, 1.0])
    out = I.online(scores, np.ones(3, bool), 1, calib, gammas=[0.05])
    np.testing.assert_allclose(out["alpha_t"], [0.1, 0.1 + 0.05 * (0.1 - 1), 0.1 + 0.05 * (0.1 - 1) + 0.05 * 0.1])
    assert out["covered"].tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("gammas", [I.DTACI_GAMMAS, (0.05,)])
@pytest.mark.parametrize("h", [1, 6])
def test_feedback_delay(gammas, h):
    rng = np.random.default_rng(3)
    calib = np.abs(rng.normal(size=200))
    scores = np.abs(rng.normal(size=400))
    has = np.ones(400, bool)
    base = I.online(scores, has, h, calib, gammas=gammas)["alpha_t"]
    j = 250
    later = scores.copy()
    later[j - h + 1:] = 0.0  # targets not yet known at row j
    np.testing.assert_array_equal(I.online(later, has, h, calib, gammas=gammas)["alpha_t"][:j + 1], base[:j + 1])
    # the most recent known target (row j - h) changes the interval at row j: covered vs missed
    hit, miss = scores.copy(), scores.copy()
    hit[j - h], miss[j - h] = 0.0, 100.0
    a_hit = I.online(hit, has, h, calib, gammas=gammas)["alpha_t"]
    a_miss = I.online(miss, has, h, calib, gammas=gammas)["alpha_t"]
    np.testing.assert_array_equal(a_hit[:j], a_miss[:j])
    assert a_hit[j] > a_miss[j]


def test_rows_without_forecast_are_skipped():
    rng = np.random.default_rng(4)
    calib, scores = np.abs(rng.normal(size=100)), np.abs(rng.normal(size=50)) * 3
    has = np.ones(50, bool)
    has[10:20] = False
    out = I.online(scores, has, 2, calib, gammas=[0.05])
    assert np.isnan(out["alpha_t"][10:20]).all() and np.isnan(out["covered"][10:20]).all()
    # with no forecasts at rows 10-19, the state changes only through feedback of rows 8 and 9
    # (arriving at rows 10 and 11); after that it is frozen until feedback of row 20 at row 22
    a = out["alpha_t"]
    assert a[20] == a[21]
    assert a[22] != a[21]
    # the result equals running ACI on the rows with a forecast, with the delay counted in rows
    # of the full grid: rows 8 and 9 update before row 20, nothing else does
    err = 1 - out["covered"]
    assert a[20] == pytest.approx(a[9] + 0.05 * (0.1 - err[8]) + 0.05 * (0.1 - err[9]))


def test_extreme_levels():
    calib = np.arange(1.0, 10.0)
    assert I.online(np.array([1e9, 0.0]), np.ones(2, bool), 1, calib, gammas=[2.0])["covered"].tolist() == [0.0, 1.0]
    # after a miss with gamma = 2, alpha = 0.1 + 2 * (0.1 - 1) < 0: infinite interval, covered
    out = I.online(np.array([1e9, 1e9]), np.ones(2, bool), 1, calib, gammas=[2.0])
    assert out["alpha_t"][1] < 0 and out["q"][1] == math.inf and out["covered"][1] == 1.0
    out = I.online(np.array([0.0, 0.0]), np.ones(2, bool), 1, calib, gammas=[20.0])
    assert out["alpha_t"][1] >= 1 and out["q"][1] == -math.inf and out["covered"][1] == 0.0


def test_dtaci_adapts_to_shift():
    rng = np.random.default_rng(5)
    calib = np.abs(rng.normal(size=2000))
    scores = np.abs(rng.normal(size=20000)) * 1.6  # all errors larger than in calibration
    out = I.online(scores, np.ones(20000, bool), 1, calib)
    assert np.mean(scores <= I.split_q(calib)) < 0.7  # a fixed quantile undercovers
    assert np.nanmean(out["covered"]) == pytest.approx(0.9, abs=0.01)
