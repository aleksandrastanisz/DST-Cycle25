"""Tests of blocked cross-validation (section 7.2) and model X (section 7)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import cv  # noqa: E402
from dstc25 import xgb_model as X  # noqa: E402

TRAIN_T = pd.Series(pd.date_range("1995-01-01", "2009-12-31 23:00", freq="h", tz="UTC"))


def test_validation_blocks_partition_train():
    count = np.zeros(len(TRAIN_T), dtype=int)
    for _, _, val in cv.folds(TRAIN_T):
        count += val
    assert (count == 1).all()


def test_train_and_validation_separated_by_gap():
    for block, train, val in cv.folds(TRAIN_T):
        assert not (train & val).any()
        t_train, t_val = TRAIN_T[train], TRAIN_T[val]
        lo, hi = t_val.min(), t_val.max()
        dist = np.minimum((t_train - lo).abs(), (t_train - hi).abs())
        assert dist.min() > cv.GAP
        # nothing else is dropped: train is everything outside the block and its two gaps
        outside = ~val & ((TRAIN_T < lo - cv.GAP) | (TRAIN_T > hi + cv.GAP)).to_numpy()
        assert (train == outside).all()


def test_block_bounds():
    lo, hi = cv.block_bounds((2001, 2003))
    assert lo == pd.Timestamp("2001-01-01", tz="UTC") and hi == pd.Timestamp("2003-12-31 23:00", tz="UTC")


def test_grid_has_12_distinct_configs():
    assert len(X.GRID) == 12
    assert len({tuple(sorted(c.items())) for c in X.GRID}) == 12


def regression_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = 3 * x["a"].to_numpy() - x["b"].to_numpy() ** 2 + rng.normal(0, 0.5, n)
    return x, y


def test_predict_uses_best_iteration():
    x, y = regression_data()
    cfg = {"max_depth": 4, "learning_rate": 0.3, "min_child_weight": 1}
    m = X.fit(x[:400], y[:400], cfg, seed=100, x_val=x[400:], y_val=y[400:])
    assert m.n_trees == m.booster.best_iteration + 1
    assert m.booster.num_boosted_rounds() == m.n_trees + X.PATIENCE  # trees after the best are kept
    expected = m.booster.predict(xgb.DMatrix(x[400:]), iteration_range=(0, m.n_trees))
    np.testing.assert_array_equal(m.predict(x[400:]), expected)


def test_fixed_trees_and_seed():
    x, y = regression_data()
    cfg = {"max_depth": 4, "learning_rate": 0.1, "min_child_weight": 1}
    a = X.fit(x, y, cfg, seed=42, n_trees=30)
    b = X.fit(x, y, cfg, seed=42, n_trees=30)
    c = X.fit(x, y, cfg, seed=43, n_trees=30)
    assert a.booster.num_boosted_rounds() == 30
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    assert not np.array_equal(a.predict(x), c.predict(x))  # subsampling depends on the seed


def test_missing_features_handled():
    x, y = regression_data()
    x.iloc[::7, 0] = np.nan
    m = X.fit(x, y, X.GRID[0], seed=42, n_trees=10)
    assert np.isfinite(m.predict(x)).all()


def test_fit_arguments_checked():
    x, y = regression_data()
    with pytest.raises(ValueError):
        X.fit(x, y, X.GRID[0], seed=42)
    with pytest.raises(ValueError):
        X.fit(x, y, X.GRID[0], seed=42, n_trees=5, x_val=x, y_val=y)
