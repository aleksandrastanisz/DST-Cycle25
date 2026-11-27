"""Correctness checks R0 (protocol section 9.1) on out-of-fold forecasts within train.

Needs finished tuning of X and L (scripts/tune.py) with their out-of-fold forecasts. Checks:
  1. positive control: X and L have lower MSE than P at both horizons, on the common set of
     hours in which P, E, X and L all forecast and the target is known;
  2. target leakage: X with the target added as a feature reaches out-of-fold MAE < 1 nT;
  3. negative control: X on a target shuffled within train does not beat a constant forecast
     (the mean of the fold's training target) by more than 1 % in MSE;
  4b. model E: correlation between forecast and observation > 0.8 at h = 1 h.
Check 4a (steady state of E) is the automated test in tests/test_baselines.py.
Checks 2 and 3 use the selected configuration of X with the tuning seed (100) and the same
folds and early stopping as tuning; the shuffle in check 3 also uses seed 100.

    python scripts/r0.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import baselines, cv  # noqa: E402
from dstc25 import features as F  # noqa: E402
from dstc25 import xgb_model as X  # noqa: E402

TUNING = ROOT / "results" / "tuning"
HORIZONS = (1, 6)


def selected(model: str, h: int) -> dict:
    state = json.loads((TUNING / f"{model}_h{h}.json").read_text())
    return state["summary"]


def oof(model: str, h: int, df: pd.DataFrame) -> np.ndarray:
    k = selected(model, h)["selected"]
    o = pd.read_parquet(TUNING / "oof" / f"{model}_h{h}_config{k:02d}.parquet")
    if len(o) != len(df) or not (o["t_start"].to_numpy() == df["t_start"].to_numpy()).all():
        raise ValueError(f"out-of-fold forecasts of {model} h={h} are not aligned with train")
    return o["pred"].to_numpy(dtype=float)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def xgb_oof(feats: pd.DataFrame, y: np.ndarray, config: dict, t_start: pd.Series) -> np.ndarray:
    """Out-of-fold X forecasts with tuning folds, seed and early stopping; NaN where y is missing."""
    pred = np.full(len(y), np.nan)
    known = ~np.isnan(y)
    for _, train, val in cv.folds(t_start):
        tr, va = train & known, val & known
        m = X.fit(feats.loc[tr], y[tr], config, cv.TUNING_SEED, x_val=feats.loc[va], y_val=y[va])
        pred[va] = m.predict(feats.loc[va])
    return pred


def constant_oof(y: np.ndarray, t_start: pd.Series) -> np.ndarray:
    """Mean of the fold's training target, as an out-of-fold forecast."""
    pred = np.full(len(y), np.nan)
    known = ~np.isnan(y)
    for _, train, val in cv.folds(t_start):
        pred[val & known] = y[train & known].mean()
    return pred


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    filled = F.fill_gaps(df)
    feats = F.build_features(filled)[F.FEATURES]
    report, passed = {}, True

    for h in HORIZONS:
        y = F.targets(df, (h,))[f"y_h{h}"].to_numpy(dtype=float)
        p = baselines.persistence(filled, h).to_numpy(dtype=float)
        e = baselines.ak2(filled, h).to_numpy(dtype=float)
        x, l = oof("X", h, df), oof("L", h, df)
        common = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(e) & ~np.isnan(x) & ~np.isnan(l)
        m = {name: mse(f[common], y[common]) for name, f in (("P", p), ("E", e), ("X", x), ("L", l))}
        ok1 = m["X"] < m["P"] and m["L"] < m["P"]
        print(f"h={h}: common hours {common.sum()} of {len(y)}; MSE [nT^2] "
              + ", ".join(f"{k} {v:.2f}" for k, v in m.items()) + f"; check 1 {'pass' if ok1 else 'FAIL'}")

        config = selected("X", h)["selected_config"]
        leak = feats.assign(target=y)
        xl = xgb_oof(leak, y, config, df["t_start"])
        mae_leak = float(np.nanmean(np.abs(xl - y)))
        ok2 = mae_leak < 1.0
        print(f"h={h}: X with the target as a feature, MAE {mae_leak:.3f} nT; check 2 {'pass' if ok2 else 'FAIL'}")

        known = ~np.isnan(y)
        ys = y.copy()
        ys[known] = np.random.default_rng(cv.TUNING_SEED).permutation(y[known])
        xs, c = xgb_oof(feats, ys, config, df["t_start"]), constant_oof(ys, df["t_start"])
        mse_s, mse_c = mse(xs[known], ys[known]), mse(c[known], ys[known])
        gain = 1 - mse_s / mse_c
        ok3 = gain <= 0.01
        print(f"h={h}: shuffled target, MSE X {mse_s:.2f}, constant {mse_c:.2f}, gain {100 * gain:.2f} %; "
              f"check 3 {'pass' if ok3 else 'FAIL'}")

        report[f"h{h}"] = {
            "common_hours": int(common.sum()), "hours": int(len(y)), "mse_common": m, "check1": ok1,
            "leak_mae": mae_leak, "check2": ok2,
            "shuffled_mse_x": mse_s, "shuffled_mse_const": mse_c, "shuffled_gain": gain, "check3": ok3,
        }
        passed &= ok1 and ok2 and ok3

        if h == 1:
            ok_e = ~np.isnan(e) & ~np.isnan(y)
            r = float(np.corrcoef(e[ok_e], y[ok_e])[0, 1])
            ok4 = r > 0.8
            print(f"h=1: correlation of E with observation {r:.3f} over {ok_e.sum()} hours; "
                  f"check 4b {'pass' if ok4 else 'FAIL'}")
            report["h1"].update({"e_corr": r, "e_hours": int(ok_e.sum()), "check4b": ok4})
            passed &= ok4

    report["all_passed"] = bool(passed)
    out = ROOT / "results" / "r0.json"
    out.write_text(json.dumps(report, indent=1))
    print("R0: all checks passed" if passed else "R0: at least one check FAILED")


if __name__ == "__main__":
    main()
