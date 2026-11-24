"""Tests of the storm definition (protocol section 6) on synthetic Dst series."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import storms  # noqa: E402

T0 = pd.Timestamp("2010-01-01", tz="UTC")


def series(values):
    idx = pd.date_range(T0, periods=len(values), freq="h")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def quiet(n):
    return [-10.0] * n


def dip(depth, width=11):
    """A V-shaped dip with a single minimum of `depth` in the middle."""
    half = width // 2
    return list(np.linspace(-10, depth, half + 1)) + list(np.linspace(depth, -10, half + 1))[1:]


def test_single_storm_minimum_window_and_category():
    d = series(quiet(100) + dip(-120) + quiet(100))
    w = storms.storm_windows(d)
    assert len(w) == 1
    t_min = T0 + pd.Timedelta(hours=105)
    assert w.t_min[0] == t_min and w.dst_min[0] == -120
    assert w.start[0] == t_min - pd.Timedelta(hours=24)
    assert w.end[0] == t_min + pd.Timedelta(hours=48)
    assert w.category[0] == "intense" and not w.extreme[0]


def test_below_threshold_is_not_a_storm():
    d = series(quiet(100) + dip(-49) + quiet(100))
    assert storms.storm_windows(d).empty


def test_tie_goes_to_earliest_hour():
    d = series(quiet(50) + [-60, -60, -60] + quiet(50))
    m = storms.find_minima(d)
    assert list(m.index) == [T0 + pd.Timedelta(hours=50)]


def test_second_dip_within_24h_is_not_a_separate_minimum():
    d = series(quiet(100) + dip(-80) + quiet(5) + dip(-60) + quiet(100))
    assert len(storms.find_minima(d)) == 1


def test_overlapping_windows_merge_and_take_lowest_minimum():
    # minima 40 h apart: separate minima, but windows [-24, +48] overlap
    d = series(quiet(100) + dip(-70) + quiet(29) + dip(-180) + quiet(150))
    w = storms.storm_windows(d)
    assert len(w) == 1
    assert w.n_minima[0] == 2 and w.dst_min[0] == -180 and w.category[0] == "intense"


def test_separate_windows_and_extreme_flag():
    d = series(quiet(100) + dip(-300) + quiet(200) + dip(-60) + quiet(100))
    w = storms.storm_windows(d)
    assert len(w) == 2
    assert w.extreme.tolist() == [True, False]
    assert storms.count_by_category(w) == {"moderate": 1, "intense": 1, "extreme": 1}


def test_split_assignment_by_minimum_and_clipping():
    d = series(quiet(100) + dip(-120) + quiet(100))
    w = storms.storm_windows(d)
    t_min = w.t_min[0]
    lo = (t_min - pd.Timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
    hi = (t_min + pd.Timedelta(hours=200)).strftime("%Y-%m-%d %H:%M")
    a = storms.assign_to_split(w, lo, hi)
    assert len(a) == 1 and a.start[0] == pd.Timestamp(lo, tz="UTC")
    assert storms.assign_to_split(w, "2009-01-01", "2009-12-31 23:00").empty
