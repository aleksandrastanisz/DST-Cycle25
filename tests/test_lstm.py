"""Tests of model L (protocol sections 7 and 7.2)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import lstm_model as L  # noqa: E402

CFG = {"hidden": 16, "lr": 3e-3, "layers": 1}


def windows(n=1500, seed=0):
    """Target = 2 * last value of channel 0 + noise, in nT-like units with an offset."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 24, 3)).astype(np.float32)
    x[:, :, 1] = x[:, :, 1] * 50 + 400  # a channel on a very different scale
    y = -20 + 30 * x[:, -1, 0] + rng.normal(0, 1, n)
    return x, y


def test_grid_has_12_distinct_configs():
    assert len(L.GRID) == 12
    assert len({tuple(sorted(c.items())) for c in L.GRID}) == 12


def test_learns_simple_target():
    x, y = windows()
    m = L.fit(x[:1200], y[:1200], CFG, seed=42, n_epochs=40)
    rmse = np.sqrt(np.mean((m.predict(x[1200:]) - y[1200:]) ** 2))
    assert rmse < 0.5 * np.std(y)  # predicting the mean would give rmse = std(y)


def test_scaler_uses_training_rows_only():
    x, y = windows()
    m1 = L.fit(x[:1000], y[:1000], CFG, seed=42, x_val=x[1000:], y_val=y[1000:], max_epochs=2)
    m2 = L.fit(x[:1000], y[:1000], CFG, seed=42, x_val=x[1000:] * 5, y_val=y[1000:] * 5, max_epochs=2)
    np.testing.assert_array_equal(m1.scaler.x_mean, m2.scaler.x_mean)
    assert m1.scaler.y_mean == m2.scaler.y_mean == pytest.approx(np.mean(y[:1000]))


def test_early_stopping_restores_best_epoch():
    x, y = windows()
    m = L.fit(x[:1000], y[:1000], CFG, seed=42, x_val=x[1000:], y_val=y[1000:], max_epochs=12, patience=2)
    assert m.n_epochs == int(np.argmin(m.val_mse)) + 1
    assert len(m.val_mse) <= 12
    if len(m.val_mse) < 12:
        assert len(m.val_mse) - m.n_epochs == 2  # stopped exactly `patience` epochs after the best
    mse = np.mean((m.predict(x[1000:]) - y[1000:]) ** 2)
    assert mse == pytest.approx(min(m.val_mse), rel=1e-5)  # weights of the best epoch are restored


def test_same_seed_same_model():
    x, y = windows(n=600)
    a = L.fit(x, y, CFG, seed=42, n_epochs=2).predict(x)
    b = L.fit(x, y, CFG, seed=42, n_epochs=2).predict(x)
    c = L.fit(x, y, CFG, seed=43, n_epochs=2).predict(x)
    np.testing.assert_allclose(a, b, atol=1e-4)
    assert not np.allclose(a, c, atol=1e-4)


def test_prediction_independent_of_batching():
    x, y = windows(n=600)
    m = L.fit(x, y, CFG, seed=42, n_epochs=1)
    whole = m.predict(x)
    parts = np.concatenate([m.predict(x[:7]), m.predict(x[7:])])
    np.testing.assert_allclose(whole, parts, atol=1e-4)


def test_inputs_checked():
    x, y = windows(n=300)
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        L.fit(bad, y, CFG, seed=42, n_epochs=1)
    with pytest.raises(ValueError):
        L.fit(x, y, CFG, seed=42)
    with pytest.raises(ValueError):
        L.fit(x, y, CFG, seed=42, n_epochs=1, x_val=x, y_val=y)
