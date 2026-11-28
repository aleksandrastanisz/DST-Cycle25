"""Secondary metrics computed per storm window (protocol section 8): M7, M8, M11, phases and the
share of infinite adaptive intervals. Inputs are hourly rows of one model and horizon with
columns t_start (issue time), unit, phase, common, y, f, se_m, se_p and optionally q_<method>,
cov_<method>. Target time is t_start + h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25 import metrics


def hidden_persistence_shift(f: pd.Series, y_all: pd.Series, h: int) -> int | None:
    """M7: the shift k in {0, ..., h + 3} h maximizing corr(forecast(tau), observation(tau - k))
    over the window's target hours tau; None for a constant forecast or too few pairs.
    f is indexed by target time; y_all by target time over the whole period."""
    if f.nunique() < 2:
        return None
    best_k, best_r = None, -np.inf
    for k in range(0, h + 4):
        obs = y_all.reindex(f.index - pd.Timedelta(hours=k)).to_numpy()
        ok = np.isfinite(obs) & np.isfinite(f.to_numpy())
        if ok.sum() < 3 or np.std(obs[ok]) == 0 or np.std(f.to_numpy()[ok]) == 0:
            continue
        r = np.corrcoef(f.to_numpy()[ok], obs[ok])[0, 1]
        if r > best_r:  # strict: ties keep the smaller shift
            best_k, best_r = k, r
    return best_k


def minimum_error(f: pd.Series, y: pd.Series) -> tuple[float, float]:
    """M8: forecast minimum minus observed minimum [nT] and timing error of the minimum [h]
    (forecast time minus observed time); ties go to the earliest hour."""
    tf, ty = f.idxmin(), y.idxmin()
    return float(f.min() - y.min()), float((tf - ty) / pd.Timedelta(hours=1))


def per_window(rows: pd.DataFrame, units: pd.DataFrame, h: int, methods=()) -> pd.DataFrame:
    """One row per storm window (common hours only): M7 shift, M8 errors, window M1 and coverage."""
    target = rows["t_start"] + pd.Timedelta(hours=h)
    y_all = pd.Series(rows["y"].to_numpy(), index=pd.DatetimeIndex(target))
    out = []
    storms = units[units.kind == "storm"].set_index("unit")
    for uid, w in storms.iterrows():
        r = rows[(rows["unit"] == uid).to_numpy() & rows["common"].to_numpy()]
        if r.empty:
            continue
        idx = pd.DatetimeIndex(r["t_start"] + pd.Timedelta(hours=h))
        f, y = pd.Series(r["f"].to_numpy(), index=idx), pd.Series(r["y"].to_numpy(), index=idx)
        d_min, d_t = minimum_error(f, y)
        rec = {"unit": uid, "category": w.category, "superintense": bool(w.superintense),
               "dst_min": w.dst_min, "n_common": len(r), "n_hours": int(w.n_hours),
               "M7_shift": hidden_persistence_shift(f, y_all, h),
               "M8_min_error": d_min, "M8_timing_error": d_t,
               "mse_m": float(r.se_m.mean()), "mse_p": float(r.se_p.mean()),
               "M1_window": float(1 - r.se_m.mean() / r.se_p.mean())}
        for m in methods:
            rec[f"cov_{m}"] = float(r[f"cov_{m}"].mean())
            rec[f"infinite_{m}"] = float(np.mean(np.isinf(r[f"q_{m}"])))
        out.append(rec)
    t = pd.DataFrame(out)
    t["included"] = t["n_common"] >= 0.5 * t["n_hours"]
    return t


def m7_summary(t: pd.DataFrame, h: int) -> dict:
    k = t.loc[t.included, "M7_shift"].dropna()
    return {"windows": int(len(k)), "median_shift": float(np.median(k)) if len(k) else float("nan"),
            "fraction_shift_ge_h": float(np.mean(k >= h)) if len(k) else float("nan")}


def dose_response(t: pd.DataFrame, methods=(), width: float = 50.0) -> list[dict]:
    """M11: window M1 (aggregated as in section 8) and mean coverage in bins of the window
    minimum: (-100, -50], (-150, -100], ... nT, matching the category edges."""
    t = t[t.included]
    edges = np.ceil(t.dst_min / width) * width - width  # bin (lo, lo + 50]: -63 and -50 -> -100
    out = []
    for lo in sorted(edges.unique()):
        s = t[edges == lo]
        rec = {"bin": [float(lo), float(lo + width)], "windows": int(len(s)),
               "M1": metrics.skill(s.mse_m, s.mse_p)}
        for m in methods:
            rec[f"M2_{m}"] = float(s[f"cov_{m}"].mean())
        out.append(rec)
    return out


def phase_metrics(rows: pd.DataFrame, included_units: set, methods=()) -> dict:
    """M1-M3 and M2 per storm phase: per-unit means over common hours of the phase, then the
    aggregation of section 8 over included units that have hours in that phase."""
    out = {}
    r = rows[rows["common"].to_numpy() & rows["unit"].isin(included_units).to_numpy()]
    for phase in ("main", "recovery"):
        p = r[r["phase"] == phase]
        if p.empty:
            continue
        g = p.groupby("unit")
        rec = {"units": int(g.ngroups), "hours": int(len(p)),
               "M1": metrics.skill(g.se_m.mean(), g.se_p.mean()),
               "M3_rmse": float(np.sqrt(p.se_m.mean())), "M3_mae": float(np.sqrt(p.se_m).mean())}
        for m in methods:
            rec[f"M2_{m}"] = float(g[f"cov_{m}"].mean().mean())
            rec[f"M4_{m}"] = float(np.median(2 * p[f"q_{m}"]))
        out[phase] = rec
    return out
