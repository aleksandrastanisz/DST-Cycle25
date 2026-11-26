"""Tests of features (protocol sections 7 and 7.1), including the causality test from section 14."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import features as F  # noqa: E402


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2001-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "t_start": t,
        "V": rng.uniform(300, 800, n), "Bz_gsm": rng.normal(0, 5, n), "By_gsm": rng.normal(0, 5, n),
        "B": rng.uniform(2, 20, n), "n": rng.uniform(1, 20, n), "Pdyn": rng.uniform(0.5, 10, n),
        "Dst": rng.normal(-20, 30, n),
    })


def test_feature_count_and_groups():
    assert len(F.FEATURES) == 44
    assert [len(g) for g in F.FEATURE_GROUPS.values()] == [7, 24, 3, 10]


def test_every_feature_is_causal():
    """Changing data after row i must not change any feature at row i (section 14)."""
    df = synthetic()
    base = F.build_features(F.fill_gaps(df))
    i = 120
    changed = df.copy()
    cols = F.SOLAR_WIND + ["Dst"]
    changed.loc[i + 1:, cols] = changed.loc[i + 1:, cols] * 3 + 100
    new = F.build_features(F.fill_gaps(changed))
    pd.testing.assert_frame_equal(base.loc[:i], new.loc[:i])


def test_lstm_windows_are_causal_and_aligned():
    df = synthetic()
    x, valid = F.lstm_windows(df)
    i = 120
    assert valid[i] and not valid[22] and valid[23]
    assert x[i, -1, F.LSTM_CHANNELS.index("Dst")] == pytest.approx(df.Dst[i])
    assert x[i, 0, F.LSTM_CHANNELS.index("Dst")] == pytest.approx(df.Dst[i - 23])
    changed = df.copy()
    changed.loc[i + 1:, "Dst"] += 500
    x2, _ = F.lstm_windows(changed)
    np.testing.assert_array_equal(x[: i + 1], x2[: i + 1])


def test_targets_and_lags_alignment():
    df = synthetic()
    feats, y = F.build_features(df), F.targets(df)
    i = 50
    assert feats.Dst_lag0[i] == df.Dst[i] and feats.Dst_lag6[i] == df.Dst[i - 6]
    assert y.y_h1[i] == df.Dst[i + 1] and y.y_h6[i] == df.Dst[i + 6]


def test_gap_filling_limit_and_direction():
    df = synthetic(30)
    df.loc[10:14, "V"] = np.nan   # 5-hour gap
    filled = F.fill_gaps(df)
    assert (filled.V[10:13] == df.V[9]).all()
    assert filled.V[13:15].isna().all()
    assert filled.V[15] == df.V[15]


def test_rolling_features_need_complete_windows():
    df = synthetic(40)
    df.loc[20, "Pdyn"] = np.nan
    feats = F.build_features(df)
    assert feats.Pdyn_mean3h[20:23].isna().all() and not np.isnan(feats.Pdyn_mean3h[23])


def test_newell_is_finite_for_negative_clock_angle():
    df = synthetic(10)
    df["By_gsm"], df["Bz_gsm"] = -5.0, -5.0
    c = F.coupling(df)
    assert np.isfinite(c.Newell).all() and (c.Newell > 0).all()
    assert c.VBs.iloc[0] == pytest.approx(df.V.iloc[0] * 5 * 1e-3)
