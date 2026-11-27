"""Tests of the download logic on synthetic data (no network).  Run: python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as dd  # noqa: E402

KYOTO_INFO = {
    "cadence": "PT1H",
    "timeStampLocation": "center",
    "parameters": [{"name": "Time"}, {"name": "dstValue", "fill": "99999"}, {"name": "versionCode"}],
}


def omni_line(year, doy, hour, bz=-5.0, v=450.0, dst=-20):
    vals = ["0"] * dd.OMNI_N_COLS
    vals[0], vals[1], vals[2] = str(year), str(doy), str(hour)
    vals[16] = str(bz)
    vals[24] = str(v)
    vals[40] = str(dst)
    return " ".join(vals)


def test_parse_omni_time_and_fill_values():
    text = "\n".join([
        omni_line(2001, 1, 0),
        omni_line(2001, 32, 23, bz=999.9, v=9999.0, dst=99999),
    ])
    df = dd.parse_omni(text)
    assert df.t_start.iloc[0] == pd.Timestamp("2001-01-01 00:00", tz="UTC")
    assert df.t_start.iloc[1] == pd.Timestamp("2001-02-01 23:00", tz="UTC")
    assert df.Bz_gsm.iloc[0] == -5.0 and df.V.iloc[0] == 450.0
    assert df[["Bz_gsm", "V", "Dst_omni"]].iloc[1].isna().all()


def test_parse_omni_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        dd.parse_omni("2001 1 0 5.0")


def test_parse_kyoto_centre_stamps_versions_and_fill():
    raw = pd.DataFrame({
        "Time": ["2020-12-31T23:29:30Z", "2021-01-01T00:29:30Z", "2026-01-01T00:29:30Z"],
        "dstValue": [-12, 99999, -40],
        "versionCode": [20, 11, 0],
    })
    out = dd.parse_kyoto(raw, KYOTO_INFO)
    assert out.t_kyoto.tolist() == list(pd.to_datetime(
        ["2020-12-31 23:00", "2021-01-01 00:00", "2026-01-01 00:00"], utc=True))
    assert out.dst_version.tolist() == ["final", "provisional", "realtime"]
    assert np.isnan(out.Dst.iloc[1]) and out.Dst.iloc[0] == -12


def test_parse_kyoto_rejects_off_centre_stamps():
    raw = pd.DataFrame({"Time": ["2020-01-01T00:00:00Z"], "dstValue": [0], "versionCode": [20]})
    with pytest.raises(SystemExit):
        dd.parse_kyoto(raw, KYOTO_INFO)


def test_align_finds_one_hour_shift():
    t = pd.date_range("2000-01-01", periods=500, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    dst = rng.integers(-150, 20, size=len(t)).astype(float)
    omni = pd.DataFrame({"t_start": t, "Dst_omni": dst})
    # Kyoto stamps the end of the hour: same value, timestamp +1 h
    kyoto = pd.DataFrame({"t_kyoto": t + pd.Timedelta(hours=1), "Dst": dst})
    aligned, info = dd.align_kyoto(omni, kyoto)
    assert info["lag_hours"] == -1
    assert aligned.t_start.tolist() == t.tolist()


def test_splits_are_disjoint_contiguous_and_as_in_protocol():
    assert dd.SPLITS == {
        "train": ("1995-01-01", "2009-12-31 23:00"),
        "calib": ("2010-01-01", "2019-12-31 23:00"),
        "test": ("2020-01-01", "2025-12-31 23:00"),
    }
    bounds = [(pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")) for lo, hi in dd.SPLITS.values()]
    for (lo1, hi1), (lo2, hi2) in zip(bounds, bounds[1:]):
        assert lo1 < hi1 < lo2 < hi2
        assert lo2 - hi1 == pd.Timedelta(hours=1)  # no gap and no shared hour
    assert bounds[0][0].year == dd.FIRST_YEAR and bounds[-1][1].year == dd.LAST_YEAR
