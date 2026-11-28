"""Decision rules for H1-H3 (protocol section 10.2) on synthetic per-unit tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import bootstrap as bs  # noqa: E402
from dstc25 import decisions  # noqa: E402

KEYS = [("X", 1), ("L", 1), ("X", 6), ("L", 6)]


def tables(seed=0, quiet_ratio=0.5, intense_ratio=0.9, m9=0.9, mond=0.75, dtaci=0.88, n_q=82, n_i=24, n_c=26,
           sd=0.08):
    rng = np.random.default_rng(seed)
    ids = {"quiet": [f"q{i}" for i in range(n_q)], "intense": [f"s{i}" for i in range(n_i)],
           "calib_intense": [f"c{i}" for i in range(n_c)]}
    test, calib = {}, {}
    for k in KEYS:
        se_p = np.r_[rng.uniform(5, 15, n_q), rng.uniform(200, 600, n_i)]
        ratio = np.r_[np.full(n_q, quiet_ratio), np.full(n_i, intense_ratio)] * rng.uniform(0.9, 1.1, n_q + n_i)
        t = pd.DataFrame({"se_p": se_p, "se_m": se_p * ratio,
                          "cov_mond": np.r_[np.full(n_q, 0.9), np.clip(rng.normal(mond, sd, n_i), 0, 1)],
                          "included": True}, index=ids["quiet"] + ids["intense"])
        t["cov_dtaci"] = np.r_[np.full(n_q, 0.9), np.clip(t.cov_mond.to_numpy()[n_q:] + rng.normal(dtaci - mond, 0.03, n_i), 0, 1)]
        test[k] = t
        calib[k] = pd.DataFrame({"cov_mond": np.clip(rng.normal(m9, sd, n_c), 0, 1), "included": True},
                                index=ids["calib_intense"])
    counts = {"quiet": bs.draw_counts(np.random.default_rng(1), n_q, b=2000),
              "intense": bs.draw_counts(np.random.default_rng(2), n_i, b=2000),
              "calib_intense": bs.draw_counts(np.random.default_rng(3), n_c, b=2000)}
    return test, calib, ids, counts


def test_clear_effects_are_supported():
    test, calib, ids, counts = tables()
    dec = decisions.decide(decisions.statistics(test, calib, ids, counts))
    for k in KEYS:
        assert dec["H1"][k]["estimate"] == pytest.approx(0.4, abs=0.05) and dec["H1"][k]["supported"]
        assert dec["H2"][k]["supported"] and dec["H2"][k]["estimate"] > 0.1
        assert dec["H3"][k]["supported"] and dec["H3"][k]["m2_dtaci"] >= 0.85


def test_effect_below_practical_threshold_is_not_supported():
    # Delta about 0.02: significant with these tight units but below 0.05
    test, calib, ids, counts = tables(quiet_ratio=0.88, intense_ratio=0.9)
    dec = decisions.decide(decisions.statistics(test, calib, ids, counts))
    assert all(not dec["H1"][k]["supported"] for k in KEYS)


def test_h3_needs_minimum_coverage():
    test, calib, ids, counts = tables(mond=0.70, dtaci=0.80)  # D = 0.1 but coverage 0.80 < 0.85
    dec = decisions.decide(decisions.statistics(test, calib, ids, counts))
    for k in KEYS:
        assert dec["H3"][k]["estimate"] > 0.05 and dec["H3"][k]["p_holm"] < 0.05
        assert not dec["H3"][k]["supported"]


def test_no_effect_not_supported():
    test, calib, ids, counts = tables(quiet_ratio=0.9, intense_ratio=0.9, m9=0.75, mond=0.75, dtaci=0.75, sd=0.01)
    dec = decisions.decide(decisions.statistics(test, calib, ids, counts))
    assert not any(dec[h][k]["supported"] for h in dec for k in KEYS)


def test_excluded_units_and_drop():
    test, calib, ids, counts = tables()
    t = test[("X", 1)]
    t.loc["s0", "included"] = False
    t.loc["s0", "cov_mond"] = -100.0  # would dominate the mean if it were used
    s = decisions.statistics(test, calib, ids, counts)
    expected = t.loc[ids["intense"][1:], "cov_mond"].mean()
    assert s["H2"][("X", 1)]["m2_test"] == pytest.approx(expected)
    assert np.nanmin(s["H2"][("X", 1)]["reps"]) > -1  # excluded unit never enters a replicate
    s3 = decisions.statistics(test, calib, ids, counts, drop="s1")
    assert len(s3["H3"][("X", 1)]["d_units"]) == 22


def test_sign_flip_permutation():
    d = np.full(20, 0.05)
    assert decisions.sign_flip_p(d, n=20000) == pytest.approx(1 / 2 ** 20 + 1 / 20001, abs=2e-4)
    rng = np.random.default_rng(0)
    p = [decisions.sign_flip_p(rng.normal(0, 1, 24), n=2000, seed=s) for s in range(200)]
    assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09


def test_fragility_names_the_decisive_storm():
    test, calib, ids, counts = tables()
    for k in KEYS:  # D >= 0.03 only because of window s3
        t = test[k]
        t.loc[ids["intense"], "cov_mond"] = 0.86
        t.loc[ids["intense"], "cov_dtaci"] = 0.872
        t.loc["s3", ["cov_mond", "cov_dtaci"]] = [0.0, 1.0]
    base = decisions.decide(decisions.statistics(test, calib, ids, counts))
    assert all(base["H3"][k]["supported"] for k in KEYS)
    frag = decisions.fragility(test, calib, ids, counts, base)
    assert frag["H3"] == {f"{m}_h{h}": ["s3"] for m, h in KEYS}
