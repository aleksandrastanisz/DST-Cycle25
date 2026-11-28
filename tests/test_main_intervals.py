"""Intervals for the main evaluation: residuals from calib only, adaptive runs without reset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import intervals, main_intervals  # noqa: E402

CALIB_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")


def data(n_calib=2000, n_test=1000, seed=0, test_scale=1.0):
    t = pd.Series(pd.date_range(CALIB_END - pd.Timedelta(hours=n_calib - 1), periods=n_calib + n_test,
                                freq="h", tz="UTC"))
    rng = np.random.default_rng(seed)
    err = rng.normal(size=len(t))
    err[n_calib:] *= test_scale
    y = rng.normal(-20, 20, len(t))
    return t, y, y - err, y, np.ones(len(t), bool), np.arange(len(t)) >= n_calib


def test_residuals_only_from_calib_and_results_only_on_test():
    t, y, f, dst, common, is_test = data()
    out = main_intervals.run(t, y, f, dst, common, is_test, 6, CALIB_END, [0, 1, 2, 3])
    res_rows = main_intervals.residual_rows(t, 6, common, CALIB_END)
    assert not (res_rows & is_test).any() and res_rows.sum() == 2000 - 6  # last 6 targets lie in test
    expected = intervals.split_q(np.abs(y - f)[res_rows])
    assert np.nanmax(out["split"]["q"]) == np.nanmin(out["split"]["q"]) == expected
    for m in main_intervals.METHODS:
        assert np.isnan(out[m]["covered"][~is_test]).all() and np.isfinite(out[m]["covered"][is_test]).all()


def test_test_errors_do_not_change_static_quantiles():
    t, y, f, dst, common, is_test = data()
    a = main_intervals.run(t, y, f, dst, common, is_test, 1, CALIB_END, [0, 1, 2, 3], methods=("split", "mond"))
    f2 = f.copy()
    f2[is_test] += 50
    b = main_intervals.run(t, y, f2, dst, common, is_test, 1, CALIB_END, [0, 1, 2, 3], methods=("split", "mond"))
    for m in ("split", "mond"):
        np.testing.assert_array_equal(a[m]["q"], b[m]["q"])


def test_adaptive_run_continues_from_calib_into_test():
    t, y, f, dst, common, is_test = data()
    out = main_intervals.run(t, y, f, dst, common, is_test, 1, CALIB_END, [0, 1, 2, 3], methods=("dtaci",))
    res = np.abs(y - f)[main_intervals.residual_rows(t, 1, common, CALIB_END)]
    full = intervals.online(np.abs(y - f), common, 1, res)
    np.testing.assert_array_equal(out["dtaci"]["alpha_t"][is_test], full["alpha_t"][is_test])
    assert full["alpha_t"][is_test][0] != intervals.ALPHA  # state carried over from calib, no reset


def test_dtaci_recovers_coverage_after_shift_while_split_does_not():
    t, y, f, dst, common, is_test = data(n_test=20000, test_scale=1.5)
    out = main_intervals.run(t, y, f, dst, common, is_test, 1, CALIB_END, [0, 1, 2, 3])
    assert np.nanmean(out["split"]["covered"]) < 0.8
    assert np.nanmean(out["dtaci"]["covered"]) == pytest.approx(0.9, abs=0.015)


def test_other_nominal_levels():
    t, y, f, dst, common, is_test = data(n_test=20000)
    out = main_intervals.run(t, y, f, dst, common, is_test, 1, CALIB_END, [0, 1, 2, 3], alpha=0.5)
    for m in ("split", "mond", "dtaci"):
        assert np.nanmean(out[m]["covered"]) == pytest.approx(0.5, abs=0.03)
