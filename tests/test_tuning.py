"""Tests of the tuning bookkeeping (protocol section 7.2): selection, final iterations, resume."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import cv, tuning  # noqa: E402

CONFIGS = [{"depth": d, "lr": lr} for d in (4, 6, 8) for lr in (0.03, 0.1)]


def runs(mse_by_config, iters=(10, 30, 20, 50, 40)):
    out = []
    for k, mse in enumerate(mse_by_config):
        for b, it in zip(cv.BLOCKS, iters):
            out.append({"config": k, "block": list(b), "val_mse": mse, "n_val": 100, "best_iter": it + k})
    return out


def test_selects_lowest_mean_and_median_iterations():
    s = tuning.summarize(runs([5, 4, 3, 6, 7, 8]), CONFIGS)
    assert s["selected"] == 2
    assert s["final_iter"] == 30 + 2  # median of (12, 32, 22, 52, 42)


def test_tie_goes_to_first_in_grid_order():
    assert tuning.summarize(runs([5, 3, 3, 6, 7, 8]), CONFIGS)["selected"] == 1


def test_grid_edge_only_for_parameters_with_three_or_more_values():
    assert tuning.summarize(runs([1, 2, 3, 4, 5, 6]), CONFIGS)["on_grid_edge"] == ["depth"]
    assert tuning.summarize(runs([3, 3, 1, 4, 5, 6]), CONFIGS)["on_grid_edge"] == []


def test_incomplete_folds_rejected():
    r = runs([5, 4, 3, 6, 7, 8])[:-1]
    with pytest.raises(ValueError):
        tuning.summarize(r, CONFIGS)


def test_resume_skips_finished_runs(tmp_path, monkeypatch):
    """A second call re-runs nothing and gives the same summary."""
    t = pd.date_range("1995-01-01", "2009-12-31 23:00", freq="24h", tz="UTC")
    df = pd.DataFrame({"t_start": t})
    calls = []

    monkeypatch.setattr(tuning, "grid", lambda model: CONFIGS[:2])
    monkeypatch.setattr(tuning, "prepare", lambda df, model, h: (np.zeros((len(df), 1)), np.zeros(len(df)), np.ones(len(df), bool)))

    def fake_run(model, config, x, y, train, val):
        calls.append((config["lr"], int(train.sum())))
        assert not (train & val).any()
        return None, np.full(val.sum(), config["lr"]), {"val_mse": config["lr"], "n_val": int(val.sum()), "best_iter": 7}

    monkeypatch.setattr(tuning, "run_fold", fake_run)
    s1 = tuning.tune(df, "X", 1, tmp_path, {"commit": "a"}, log=lambda *_: None)
    assert len(calls) == 2 * len(cv.BLOCKS)
    s2 = tuning.tune(df, "X", 1, tmp_path, {"commit": "b"}, log=lambda *_: None)
    assert len(calls) == 2 * len(cv.BLOCKS) and s1 == s2

    oof = pd.read_parquet(tmp_path / "oof" / "X_h1_config00.parquet")
    assert oof["pred"].notna().all()  # every train hour is predicted exactly once out of fold
    state = json.loads((tmp_path / "X_h1.json").read_text())
    assert [x["commit"] for x in state["sessions"]] == ["a", "b"]
