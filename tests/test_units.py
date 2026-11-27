"""Units of analysis (protocol section 6): storm windows, quiet blocks, phases, assignment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import units  # noqa: E402

H = pd.Timedelta(hours=1)


def series(start, n, dips=()):
    """Quiet Dst of -10 nT with V-shaped storms (time, depth) of 12 h descent."""
    t = pd.date_range(start, periods=n, freq="h", tz="UTC")
    d = np.full(n, -10.0)
    for when, depth in dips:
        i = t.get_loc(pd.Timestamp(when, tz="UTC"))
        for k in range(-12, 13):
            if 0 <= i + k < n:
                d[i + k] = min(d[i + k], depth + abs(k) * (depth + 10) / -12)
    return pd.Series(d, index=t)


def test_merge_small_blocks():
    assert units.merge_small_blocks([100, 100, 100]) == [0, 1, 2]
    assert units.merge_small_blocks([100, 50, 100]) == [0, 0, 1]
    assert units.merge_small_blocks([50, 100, 100]) == [0, 0, 1]
    assert units.merge_small_blocks([10, 10, 100]) == [0, 0, 0]
    assert units.merge_small_blocks([100, 10, 10]) == [0, 0, 0]
    assert units.merge_small_blocks([100, 100, 0]) == [0, 1, 1]


def test_windows_phases_and_quiet_blocks():
    dst = series("2010-01-01", 24 * 120, dips=[("2010-02-01 12:00", -150), ("2010-03-10 00:00", -60)])
    hours, u = units.split_units(dst, "c", "2010-01-01", "2010-04-30 23:00")
    storm = u[u.kind == "storm"].reset_index(drop=True)
    assert storm.category.tolist() == ["intense", "moderate"]
    s0 = storm.iloc[0]
    assert s0.t_min == pd.Timestamp("2010-02-01 12:00", tz="UTC")
    assert s0.start == s0.t_min - 24 * H and s0.end == s0.t_min + 48 * H and s0.n_hours == 73
    w = hours[hours.unit == s0.unit]
    assert (w.phase == "main").sum() == 25 and (w.phase == "recovery").sum() == 48
    # every hour has exactly one unit; quiet hours are in blocks of at least 72 h
    assert hours.unit.notna().all()
    q = u[u.kind == "quiet"]
    assert (q.n_hours >= 72).all() and q.n_hours.sum() + storm.n_hours.sum() == len(hours)
    # 120 days = 4 full 27-day blocks + 12 days; the last part has >= 72 quiet hours, so 5 blocks
    assert len(q) == 5


def test_window_of_neighbouring_split_is_excluded():
    # minimum on the last hour of December lies in the previous split; its window reaches into January
    dst = series("2009-12-01", 24 * 100, dips=[("2009-12-31 20:00", -120)])
    hours, u = units.split_units(dst, "c", "2010-01-01", "2010-03-10 23:00")
    assert (u.kind == "storm").sum() == 0
    first = hours[hours.t_target <= pd.Timestamp("2009-12-31 20:00", tz="UTC") + 48 * H]
    assert first.unit.isna().all() and len(first) == 45  # 2010-01-01 00:00 .. 2010-01-02 20:00
    assert hours.unit.notna().sum() == len(hours) - 45


def test_window_clipped_to_own_split():
    dst = series("2009-12-01", 24 * 100, dips=[("2010-01-01 05:00", -120)])
    _, u = units.split_units(dst, "c", "2010-01-01", "2010-03-10 23:00")
    s = u[u.kind == "storm"].iloc[0]
    assert s.start == pd.Timestamp("2010-01-01", tz="UTC") and s.n_hours == 6 + 48


def test_forecasts_assigned_by_target_hour():
    dst = series("2010-01-01", 24 * 60, dips=[("2010-02-01 12:00", -150)])
    hours, u = units.split_units(dst, "c", "2010-01-01", "2010-03-01 23:00")
    s = u[u.kind == "storm"].iloc[0]
    issue = pd.Series(pd.date_range("2010-01-01", "2010-03-01 23:00", freq="h", tz="UTC"))
    a = units.for_forecasts(issue, 6, hours)
    in_storm = (a.unit == s.unit).to_numpy()
    assert issue[in_storm].min() == s.start - 6 * H and issue[in_storm].max() == s.end - 6 * H
    # the last 6 issue hours have targets after the split
    assert a.unit.iloc[-6:].isna().all() and a.unit.iloc[:-6].notna().all()
    assert not a.superintense.any()
