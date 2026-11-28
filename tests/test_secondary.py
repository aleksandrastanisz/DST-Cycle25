"""Secondary per-window metrics (protocol section 8: M7, M8, M11, phases)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import secondary  # noqa: E402

H = pd.Timedelta(hours=1)


def series(n=200, seed=0):
    t = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    y = pd.Series(np.cumsum(np.random.default_rng(seed).normal(size=n)), index=t)
    return y


def test_m7_finds_hidden_persistence():
    y = series()
    for h in (1, 6):
        window = y.index[50:120]
        persistence_like = pd.Series(y.shift(h).reindex(window).to_numpy(), index=window)
        assert secondary.hidden_persistence_shift(persistence_like, y, h) == h
        perfect = pd.Series(y.reindex(window).to_numpy(), index=window)
        assert secondary.hidden_persistence_shift(perfect, y, h) == 0
    assert secondary.hidden_persistence_shift(pd.Series(1.0, index=y.index[:10]), y, 1) is None


def test_m8_minimum_error_and_timing_with_ties():
    t = pd.date_range("2021-01-01", periods=5, freq="h", tz="UTC")
    y = pd.Series([0, -80, -120, -120, -60.0], index=t)
    f = pd.Series([0, -100, -90, -100, -50.0], index=t)
    d, dt = secondary.minimum_error(f, y)
    assert d == 20.0 and dt == -1.0  # forecast minimum at hour 1, observed (earliest tie) at hour 2


def test_per_window_and_dose_response():
    t = pd.Series(pd.date_range("2021-01-01", periods=300, freq="h", tz="UTC"))
    h = 1
    target = t + H
    unit = pd.Series([None] * 300)
    unit[(target >= pd.Timestamp("2021-01-03", tz="UTC")) & (target < pd.Timestamp("2021-01-06 01:00", tz="UTC"))] = "w1"
    rng = np.random.default_rng(0)
    y = rng.normal(-80, 20, 300)
    rows = pd.DataFrame({"t_start": t, "unit": unit, "phase": "main", "common": True, "y": y,
                         "f": y + rng.normal(0, 5, 300), "q_dtaci": np.where(np.arange(300) % 10 == 0, np.inf, 8.0)})
    rows["se_m"] = (rows.f - rows.y) ** 2
    rows["se_p"] = 4 * rows.se_m
    rows["cov_dtaci"] = (np.abs(rows.f - rows.y) <= rows.q_dtaci).astype(float)
    units = pd.DataFrame([{"unit": "w1", "kind": "storm", "category": "intense", "superintense": False,
                           "dst_min": -130.0, "n_hours": 73}])
    w = secondary.per_window(rows, units, h, methods=("dtaci",))
    assert len(w) == 1 and w.n_common[0] == 73 and w.included[0]
    assert w.M1_window[0] == pytest.approx(0.75)
    assert 0 < w.infinite_dtaci[0] < 0.2
    bins = secondary.dose_response(pd.concat([w, w.assign(unit="w2", dst_min=-100.0), w.assign(unit="w3", dst_min=-50.0)]),
                                   methods=("dtaci",))
    assert [b["bin"] for b in bins] == [[-150.0, -100.0], [-100.0, -50.0]]
    assert [b["windows"] for b in bins] == [2, 1]
    ph = secondary.phase_metrics(rows, {"w1"}, methods=("dtaci",))
    assert ph["main"]["units"] == 1 and ph["main"]["M1"] == pytest.approx(0.75)
