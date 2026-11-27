"""Forecasts for a split: alignment, no use of later data, seed mean (dstc25/forecast.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import features as F  # noqa: E402
from dstc25 import forecast  # noqa: E402
from dstc25 import lstm_model, xgb_model  # noqa: E402
from dstc25.runinfo import sha256  # noqa: E402


def synthetic(n, start, seed):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "t_start": t,
        "V": rng.uniform(300, 800, n), "Bz_gsm": rng.normal(0, 5, n), "By_gsm": rng.normal(0, 5, n),
        "B": rng.uniform(2, 20, n), "n": rng.uniform(1, 20, n), "Pdyn": rng.uniform(0.5, 10, n),
        "Dst": rng.normal(-20, 30, n),
    })


def tiny_models(tmp_path, history):
    """Train one tiny X and L per horizon and copy them under all seed names."""
    filled = F.fill_gaps(history)
    feats = F.build_features(filled)[F.FEATURES]
    windows, valid = F.lstm_windows(filled)
    record = {"models": {}}
    for h in forecast.HORIZONS:
        y = F.targets(history, (h,))[f"y_h{h}"].to_numpy()
        ok = ~np.isnan(y)
        mx = xgb_model.fit(feats[ok], y[ok], xgb_model.GRID[0], seed=42, n_trees=5)
        ok_l = ok & valid
        ml = lstm_model.fit(windows[ok_l], y[ok_l], {"hidden": 4, "lr": 1e-3, "layers": 1}, seed=42, n_epochs=1)
        for seed in forecast.SEEDS:
            for name, m in ((f"X_h{h}_seed{seed}.json", mx), (f"L_h{h}_seed{seed}.pt", ml)):
                m.save(tmp_path / name)
                record["models"][name] = {"sha256": sha256(tmp_path / name)}
    return record


def test_split_forecasts(tmp_path):
    history, split = synthetic(300, "2009-12-01", 0), synthetic(100, "2009-12-13 12:00", 1)
    record = tiny_models(tmp_path, history)
    out = forecast.forecast_split(history, split, tmp_path, record)

    assert len(out) == 100 and (out["t_start"].to_numpy() == split["t_start"].to_numpy()).all()
    # the first hour of the split already has a complete LSTM window, built from history
    assert out["L_h1"].notna().all()
    # targets after the end of the split are missing, not taken from anywhere else
    assert out["y_h1"].isna().sum() == 1 and out["y_h6"].isna().sum() == 6
    assert (out["y_h6"].iloc[:-6].to_numpy() == split["Dst"].iloc[6:].to_numpy()).all()
    # P is the current Dst; the ensemble is the mean of the seeds
    assert (out["P_h1"].to_numpy() == split["Dst"].to_numpy()).all()
    seeds = out[[f"X_h1_seed{s}" for s in forecast.SEEDS]].mean(axis=1)
    np.testing.assert_allclose(out["X_h1"], seeds)
    assert (out["common_h6"] == out["y_h6"].notna()).all()

    # forecasts at a split hour do not depend on later split hours
    later = split.copy()
    later.loc[50:, ["V", "Bz_gsm", "By_gsm", "B", "n", "Pdyn", "Dst"]] = 0.0
    out2 = forecast.forecast_split(history, later, tmp_path, record)
    cols = [c for c in out.columns if c[0] in "PEXL" and "_h" in c]
    pd.testing.assert_frame_equal(out.loc[:49, cols], out2.loc[:49, cols])


def test_model_file_checksum_verified(tmp_path):
    history, split = synthetic(300, "2009-12-01", 0), synthetic(30, "2009-12-13 12:00", 1)
    record = tiny_models(tmp_path, history)
    record["models"]["X_h1_seed42.json"]["sha256"] = "0" * 64
    try:
        forecast.forecast_split(history, split, tmp_path, record)
    except ValueError as e:
        assert "SHA-256" in str(e)
    else:
        raise AssertionError("a model file with a wrong checksum was accepted")
