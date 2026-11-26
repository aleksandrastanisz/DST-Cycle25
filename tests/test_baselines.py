import numpy as np
import pandas as pd
import pytest

from dstc25.baselines import (
    ak2,
    ak2_injection,
    ak2_integrate,
    ak2_tau,
    persistence,
    pressure_correct,
    pressure_uncorrect,
)


def frame(n=10, **cols):
    t = pd.date_range("2000-01-01", periods=n, freq="h")
    base = {"V": 400.0, "Bz_gsm": 0.0, "Pdyn": 2.0, "Dst": -20.0}
    base.update(cols)
    return pd.DataFrame({"t_start": t, **{k: np.broadcast_to(v, n).astype(float) for k, v in base.items()}})


def test_persistence_is_current_dst():
    df = frame(Dst=np.arange(10.0))
    np.testing.assert_array_equal(persistence(df, 6).to_numpy(), np.arange(10.0))


def test_ak2_steady_state_under_constant_driving():
    # R0 item 4a: VBs = 5 mV/m, Pdyn = 2 nPa; Dst* converges to Q * tau within 1 %.
    vbs = np.array([5.0])
    target = ak2_injection(vbs) * ak2_tau(vbs)
    x = ak2_integrate(np.array([0.0]), vbs, steps=200)
    assert abs(x[0] - target[0]) <= 0.01 * abs(target[0])


def test_ak2_units_through_forecast():
    # V = 500 km/s, Bz = -10 nT gives VBs = 5 mV/m; the long-horizon forecast is Q*tau mapped back.
    df = frame(V=500.0, Bz_gsm=-10.0, Pdyn=2.0, Dst=0.0)
    vbs = np.array([5.0])
    expected = pressure_uncorrect(ak2_injection(vbs) * ak2_tau(vbs), np.array([2.0]))
    np.testing.assert_allclose(ak2(df, 200).to_numpy(), expected[0], rtol=1e-6)


def test_ak2_no_injection_below_threshold():
    assert ak2_injection(np.array([0.0, 0.5]))[0] == 0.0
    assert ak2_injection(np.array([0.5]))[0] == 0.0
    assert ak2_injection(np.array([0.6]))[0] < 0.0


def test_ak2_quiet_decay():
    # Northward IMF: no injection, Dst* relaxes towards zero without overshoot.
    x0 = np.array([-100.0])
    x = ak2_integrate(x0, np.array([0.0]), steps=6)
    tau = ak2_tau(np.array([0.0]))[0]
    assert x[0] == pytest.approx(-100.0 * (1 - 1 / tau) ** 6)
    assert -100.0 < x[0] < 0.0


def test_pressure_correction_round_trip():
    dst = np.array([-50.0, 10.0])
    pdyn = np.array([1.0, 9.0])
    np.testing.assert_allclose(pressure_uncorrect(pressure_correct(dst, pdyn), pdyn), dst)
    assert pressure_correct(np.array([0.0]), np.array([4.0]))[0] == pytest.approx(-7.26 * 2 + 11)


def test_ak2_missing_inputs_give_nan():
    df = frame(n=4)
    for col in ("V", "Bz_gsm", "Pdyn", "Dst"):
        d = df.copy()
        d.loc[2, col] = np.nan
        out = ak2(d, 1).to_numpy()
        assert np.isnan(out[2]) and not np.isnan(out[[0, 1, 3]]).any()


def test_ak2_uses_only_row_inputs():
    # Changing any later row does not change the forecast issued at row 3.
    rng = np.random.default_rng(0)
    df = frame(n=10, V=rng.uniform(300, 700, 10), Bz_gsm=rng.normal(0, 5, 10),
               Pdyn=rng.uniform(1, 5, 10), Dst=rng.normal(-30, 20, 10))
    before = ak2(df, 6).iloc[3]
    df.loc[4:, ["V", "Bz_gsm", "Pdyn", "Dst"]] = 0.0
    assert ak2(df, 6).iloc[3] == before
