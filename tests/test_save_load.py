"""A saved and loaded model predicts exactly as the original."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import lstm_model as L  # noqa: E402
from dstc25 import xgb_model as X  # noqa: E402


def test_xgb_round_trip_keeps_only_best_trees(tmp_path):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    y = 2 * x["a"].to_numpy() + rng.normal(0, 0.5, 500)
    m = X.fit(x[:350], y[:350], {"max_depth": 3, "learning_rate": 0.3, "min_child_weight": 1},
              seed=42, x_val=x[350:], y_val=y[350:])
    m.save(tmp_path / "x.json")
    loaded = X.load(tmp_path / "x.json")
    assert loaded.n_trees == m.n_trees == loaded.booster.num_boosted_rounds()
    assert loaded.features == ["a", "b", "c"]
    np.testing.assert_array_equal(loaded.predict(x), m.predict(x))


def test_lstm_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 24, 3)).astype(np.float32)
    y = 10 * x[:, -1, 0] - 20
    m = L.fit(x, y, {"hidden": 8, "lr": 3e-3, "layers": 2}, seed=42, n_epochs=2)
    m.save(tmp_path / "l.pt")
    loaded = L.load(tmp_path / "l.pt")
    assert loaded.n_epochs == 2
    np.testing.assert_array_equal(loaded.predict(x), m.predict(x))
