"""Model X: XGBoost on the 44 features of section 7.1 (protocol sections 7 and 7.2).

Uses the native xgboost API, so scikit-learn is not needed. A model fitted with early stopping
keeps the trees grown after the best iteration; predictions therefore always use only the
first `n_trees` trees, where n_trees = best_iteration + 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
}
MAX_TREES = 5000
PATIENCE = 50

GRID = [
    {"max_depth": d, "learning_rate": lr, "min_child_weight": mcw}
    for d, lr, mcw in product((4, 6, 8), (0.03, 0.1), (1, 10))
]


@dataclass
class FittedX:
    booster: xgb.Booster
    n_trees: int
    features: list[str]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(x[self.features])
        return self.booster.predict(d, iteration_range=(0, self.n_trees))

    def save(self, path) -> None:
        """Save only the first n_trees trees, so a loaded model predicts the same."""
        self.booster[: self.n_trees].save_model(str(path))


def load(path) -> FittedX:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return FittedX(booster, booster.num_boosted_rounds(), list(booster.feature_names))


def params(config: dict, seed: int) -> dict:
    return {**FIXED_PARAMS, **config, "seed": seed}


def fit(x: pd.DataFrame, y: np.ndarray, config: dict, seed: int, *,
        n_trees: int | None = None, x_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None) -> FittedX:
    """Fit with early stopping on (x_val, y_val), or with a fixed number of trees.

    Exactly one of `n_trees` and the validation pair must be given. Rows with a missing target
    must be removed by the caller; missing feature values are handled natively.
    """
    features = list(x.columns)
    dtrain = xgb.DMatrix(x, label=y)
    if n_trees is not None:
        if x_val is not None:
            raise ValueError("give either n_trees or a validation set, not both")
        booster = xgb.train(params(config, seed), dtrain, num_boost_round=n_trees)
        return FittedX(booster, n_trees, features)
    if x_val is None or y_val is None:
        raise ValueError("early stopping needs a validation set")
    dval = xgb.DMatrix(x_val[features], label=y_val)
    booster = xgb.train(params(config, seed), dtrain, num_boost_round=MAX_TREES,
                        evals=[(dval, "val")], early_stopping_rounds=PATIENCE, verbose_eval=False)
    return FittedX(booster, booster.best_iteration + 1, features)
