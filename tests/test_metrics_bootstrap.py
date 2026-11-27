"""Per-unit metrics (section 8) and bootstrap machinery (section 10.1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import bootstrap as bs  # noqa: E402
from dstc25 import metrics  # noqa: E402


def test_per_unit_means_and_half_rule():
    unit = pd.Series(["a", "a", "a", "a", "b", "b", None, "c"])
    common = np.array([1, 1, 0, 1, 1, 0, 1, 0], bool)
    hours = pd.Series({"a": 4, "b": 4, "c": 1})
    v = np.array([1, 3, 100, 5, 7, 100, 100, 100], float)
    t = metrics.per_unit(unit, common, hours, {"x": v})
    assert t.loc["a", "x"] == 3 and t.loc["b", "x"] == 7 and np.isnan(t.loc["c", "x"])
    assert t.n_common.tolist() == [3, 1, 0]
    assert t.included.tolist() == [True, False, False]  # b: 1 of 4 hours, c: 0 of 1


def test_skill_aggregates_errors_not_ratios():
    assert metrics.skill(np.array([1.0, 9.0]), np.array([2.0, 18.0])) == pytest.approx(0.5)
    # the mean of per-unit ratios would also be 0.5 here, but not in general:
    assert metrics.skill(np.array([1.0, 1.0]), np.array([2.0, 100.0])) == pytest.approx(1 - 2 / 102)


def test_draw_counts_rows_sum_to_draws():
    c = bs.draw_counts(np.random.default_rng(0), 26, 24, b=500)
    assert c.shape == (500, 26) and (c.sum(axis=1) == 24).all()
    assert bs.draw_counts(np.random.default_rng(0), 10, b=3).sum(axis=1).tolist() == [10, 10, 10]


def test_weighted_mean_drops_excluded_units():
    counts = np.array([[2, 1, 0], [0, 0, 3], [1, 1, 1]])
    v = np.array([1.0, 4.0, 100.0])
    np.testing.assert_allclose(bs.weighted_mean(counts, v), [2.0, 100.0, 35.0])
    out = bs.weighted_mean(counts, v, included=np.array([1, 1, 0], bool))
    assert out[0] == 2.0 and np.isnan(out[1]) and out[2] == 2.5


def test_bootstrap_se_matches_theory():
    rng = np.random.default_rng(1)
    v = rng.normal(size=40)
    reps = bs.weighted_mean(bs.draw_counts(np.random.default_rng(2), 40, 25, b=20000), v)
    assert np.std(reps) == pytest.approx(np.std(v) / np.sqrt(25), rel=0.03)


def test_p_value_and_ci():
    assert bs.p_value(np.array([1.0, 2.0, -1.0, 0.0])) == pytest.approx(3 / 5)
    assert bs.p_value(np.ones(9999)) == pytest.approx(1 / 10000)  # never zero
    lo, hi = bs.percentile_ci(np.arange(1001.0))
    assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)


def test_holm():
    adj = bs.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})


def test_bootstrap_with_and_without_effect():
    """Section 14: the machinery detects a known effect and keeps the error rate without one."""
    counts = bs.draw_counts(np.random.default_rng(bs.SEED_TEST), 24, b=2000)
    effect = bs.weighted_mean(counts, np.random.default_rng(3).normal(0.3, 0.2, 24))
    assert bs.p_value(effect) < 0.001
    # under H0 (true mean 0) a one-sided test at 0.05 rejects in about 5 % of data sets
    rejections = sum(
        bs.p_value(bs.weighted_mean(counts, np.random.default_rng(100 + s).normal(0.0, 0.2, 24))) < 0.05
        for s in range(400)
    )
    assert 8 <= rejections <= 36  # 2-9 %; the percentile bootstrap with 24 units is slightly liberal
