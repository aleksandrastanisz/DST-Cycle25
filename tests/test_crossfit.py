"""Cross-fitting on the halves of calib (M9, protocol section 7.3)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import crossfit  # noqa: E402


def test_halves_by_target_hour_and_crossing_window():
    t = pd.Series(pd.date_range("2014-12-30", "2015-01-03", freq="h", tz="UTC"))
    unit = pd.Series([None] * len(t))
    w = (t >= pd.Timestamp("2014-12-31 10:00", tz="UTC")) & (t <= pd.Timestamp("2015-01-01 05:00", tz="UTC"))
    unit[w] = "c-S001"
    units = pd.DataFrame([{"unit": "c-S001", "kind": "storm",
                           "start": pd.Timestamp("2014-12-31 11:00", tz="UTC"),
                           "end": pd.Timestamp("2015-01-01 06:00", tz="UTC"),
                           "t_min": pd.Timestamp("2014-12-31 20:00", tz="UTC")}])
    half, n_cross = crossfit.halves(t, 1, unit, units)
    assert n_cross == 1
    assert (half[w.to_numpy()] == 0).all()  # whole window with its minimum, in 2014
    target = t + pd.Timedelta(hours=1)
    outside = ~w.to_numpy()
    assert (half[outside] == (target[outside] >= crossfit.BOUNDARY).astype(int)).all()


def test_residuals_come_only_from_the_other_half():
    n = 2000
    t = pd.Series(pd.date_range("2014-11-01", periods=n, freq="h", tz="UTC"))
    half = (t >= crossfit.BOUNDARY).astype(int).to_numpy()
    rng = np.random.default_rng(0)
    y, f = rng.normal(size=n), np.zeros(n)
    common = np.ones(n, bool)
    dst_t = np.zeros(n)
    base = crossfit.coverage(t, 1, y, f, dst_t, common, half, [0, 1, 2, 3])
    # inflating errors in half 1 changes the coverage in half 0 (residuals from half 1) but the
    # C-Mond coverage of half 1 itself only through its own scores, not through its quantile
    y2 = y.copy()
    y2[half == 1] *= 3
    out = crossfit.coverage(t, 1, y2, f, dst_t, common, half, [0, 1, 2, 3])
    assert np.nanmean(out["mond"][half == 0]) > np.nanmean(base["mond"][half == 0])
    q0 = np.sort(np.abs(y[half == 0]))[int(np.ceil((np.sum(half == 0) + 1) * 0.9)) - 1]
    np.testing.assert_array_equal(out["mond"][half == 1], (np.abs(y2[half == 1]) <= q0).astype(float))


def test_rows_outside_common_set_have_no_coverage():
    n = 1000
    t = pd.Series(pd.date_range("2014-12-01", periods=n, freq="h", tz="UTC"))
    half = (t >= crossfit.BOUNDARY).astype(int).to_numpy()
    common = np.ones(n, bool)
    common[::10] = False
    out = crossfit.coverage(t, 6, np.random.default_rng(1).normal(size=n), np.zeros(n), np.zeros(n),
                            common, half, [0, 1, 2, 3])
    for k in ("mond", "dtaci"):
        assert np.isnan(out[k][~common]).all() and np.isfinite(out[k][common]).all()
