"""Testy logiki pobierania na sztucznych danych (bez sieci).  Uruchom: python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as dd  # noqa: E402


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
    try:
        dd.parse_omni("2001 1 0 5.0")
    except ValueError:
        return
    raise AssertionError("powinien odrzucić wiersz z inną liczbą kolumn")


def test_align_finds_one_hour_shift():
    t = pd.date_range("2000-01-01", periods=500, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    dst = rng.integers(-150, 20, size=len(t)).astype(float)
    omni = pd.DataFrame({"t_start": t, "Dst_omni": dst})
    # HAPI znakuje koniec godziny: ta sama wartość, znacznik +1 h
    kyoto = pd.DataFrame({"t_kyoto": t + pd.Timedelta(hours=1), "Dst": dst})
    aligned, info = dd.align_kyoto(omni, kyoto)
    assert info["lag_hours"] == -1
    assert (aligned.t_start.values == t.values).all()


def test_label_version_boundaries():
    ranges = {"final_until_exclusive": pd.Timestamp("2021-01-01", tz="UTC"),
              "provisional_until_exclusive": pd.Timestamp("2025-07-01", tz="UTC")}
    t = pd.Series(pd.to_datetime(["2020-12-31 23:00", "2021-01-01 00:00",
                                  "2025-06-30 23:00", "2025-07-01 00:00"], utc=True))
    assert dd.label_version(t, ranges).tolist() == ["final", "provisional", "provisional", "realtime"]
