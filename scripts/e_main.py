"""Main evaluation E-main on the test period 2020-2025 (protocol sections 8-10).

Runs only from a clean repository whose commit descends from the latest protocol tag and only
when every data file matches its SHA-256 (dstc25/guard.py); the opening is logged in
results/test_openings.json before the test file is read.

Steps: forecasts of P, E, X and L for calib and test with the archived final models (the calib
part must equal results/forecasts/calib.parquet); intervals C-split, C-Mond, C-DtACI and C-ACI
(S1) with residuals from calib; M9 cross-fitted on calib as in P0; per-unit metrics; bootstrap
over units (seed 2026 for test units, 2028 for calib windows of M9); decisions on H1-H3 with
Holm's correction; the sign-flip permutation test for H3; the single-storm check S6.

Writes results/e_main/summary.json, per-unit tables results/e_main/units_<model>_h<h>.csv and
the hourly rows to results/forecasts/test_rows.parquet (not in git; SHA-256 in the summary).

    python scripts/e_main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import bootstrap as bs  # noqa: E402
from dstc25 import crossfit, decisions, forecast, guard, main_intervals, metrics, units  # noqa: E402
from dstc25.intervals import ALPHA  # noqa: E402
from dstc25.runinfo import session_info, sha256  # noqa: E402

HORIZONS = (1, 6)
INTERVAL_MODELS = ("X", "L")
POINT_MODELS = ("E", "X", "L")
METHODS = main_intervals.METHODS
CATEGORIES = ("quiet", "moderate", "intense", "superintense")
M10_CAVEAT = 0.10
OUT = ROOT / "results" / "e_main"


def interval_score(y, f, q, alpha=ALPHA):
    lo, hi = f - q, f + q
    with np.errstate(invalid="ignore"):
        return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def category_rows(rows: pd.DataFrame, cat: str) -> np.ndarray:
    if cat == "superintense":
        return rows["superintense"].to_numpy()
    return (rows["category"] == cat).to_numpy()


def category_units(u: pd.DataFrame, cat: str) -> list[str]:
    if cat == "superintense":
        return sorted(u.loc[u.superintense, "unit"])
    return sorted(u.loc[u.category == cat, "unit"])


def secondary(rows: pd.DataFrame, table: pd.DataFrame, u: pd.DataFrame, model: str) -> dict:
    """M1, M1b, M2, M3-M6, M10 and hour-weighted M1 and M2 per category (section 8)."""
    out = {}
    for cat in CATEGORIES:
        ids = category_units(u, cat)
        t = table.loc[ids]
        inc = t.included.to_numpy()
        r = rows[category_rows(rows, cat)]
        rc = r[r.common]
        res = {"units": len(ids), "units_included": int(inc.sum()),
               "M10_hours_excluded": float(1 - len(rc) / len(r)) if len(r) else float("nan"),
               "hours_common": int(len(rc))}
        if inc.any():
            res["M1"] = metrics.skill(t.se_m[inc], t.se_p[inc])
            res["M1b"] = float(t.se_p[inc].mean() - t.se_m[inc].mean())
            res["M6"] = metrics.skill(t.se_m[inc], t.se_e[inc])
            res["M1_hour_weighted"] = float(1 - rc.se_m.mean() / rc.se_p.mean())
            res["M3_rmse"] = float(np.sqrt(rc.se_m.mean()))
            res["M3_mae"] = float(np.sqrt(rc.se_m).mean())
            if "cov_mond" in t.columns:
                res["M2"] = {m: metrics.coverage(t[f"cov_{m}"][inc]) for m in METHODS}
                res["M2_hour_weighted"] = {m: float(rc[f"cov_{m}"].mean()) for m in METHODS}
                res["M4_median_width"] = {m: float(np.median(2 * rc[f"q_{m}"])) for m in METHODS}
                res["M5_interval_score"] = {m: float(np.mean(interval_score(rc.y, rc.f, rc[f"q_{m}"])))
                                            for m in METHODS}
        out[cat] = res
    return out


def main() -> None:
    commit = guard.check(ROOT)
    n_openings = guard.log_opening(ROOT / "results" / "test_openings.json", commit, "E-main")
    session = session_info()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    splits = manifest["splits"]
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    calib = pd.read_parquet(ROOT / "data" / "processed" / "calib.parquet")
    test = pd.read_parquet(ROOT / "data" / "sealed" / "test.parquet")
    record = forecast.load_record(ROOT / "results" / "final_models.json")
    f = forecast.forecast_split(train, pd.concat([calib, test], ignore_index=True), ROOT / "models", record)
    test_lo = pd.Timestamp(splits["test"][0], tz="UTC")
    calib_end = pd.Timestamp(splits["calib"][1], tz="UTC")
    is_test = (f["t_start"] >= test_lo).to_numpy()

    saved = pd.read_parquet(ROOT / "results" / "forecasts" / "calib.parquet")
    for col in [c for c in saved.columns if c[0] in "PEXL" and "_h" in c]:
        a, b = f.loc[~is_test, col].to_numpy(), saved[col].to_numpy()
        if not np.allclose(a, b, atol=1e-3, rtol=0, equal_nan=True):
            raise SystemExit(f"calib forecasts in column {col} differ from results/forecasts/calib.parquet")

    dst = pd.concat([d[["t_start", "Dst"]] for d in (train, calib, test)]).set_index("t_start")["Dst"]
    hours_t, u_t = units.split_units(dst, "test", *splits["test"])
    hours_c, u_c = units.split_units(dst, "calib", *splits["calib"])
    merged = json.loads((ROOT / "results" / "mondrian_classes.json").read_text())

    ids = {"quiet": category_units(u_t, "quiet"), "moderate": category_units(u_t, "moderate"),
           "intense": category_units(u_t, "intense"), "calib_intense": category_units(u_c, "intense")}
    rng = np.random.default_rng(bs.SEED_TEST)  # draws in the order quiet, moderate, intense
    counts = {"quiet": bs.draw_counts(rng, len(ids["quiet"])),
              "moderate": bs.draw_counts(rng, len(ids["moderate"])),
              "intense": bs.draw_counts(rng, len(ids["intense"]))}
    counts["calib_intense"] = bs.draw_counts(np.random.default_rng(bs.SEED_M9), len(ids["calib_intense"]))

    unit_hours_t = u_t.set_index("unit")["n_hours"]
    unit_hours_c = u_c.set_index("unit")["n_hours"]
    ft = f[is_test].reset_index(drop=True)
    test_tables, calib_tables, summary_metrics, interval_info, all_rows = {}, {}, {}, {}, []
    for h in HORIZONS:
        a = units.for_forecasts(ft["t_start"], h, hours_t)
        common_t = ft[f"common_h{h}"].to_numpy()
        y = ft[f"y_h{h}"].to_numpy()
        ac = units.for_forecasts(saved["t_start"], h, hours_c)
        half, _ = crossfit.halves(saved["t_start"], h, ac["unit"], u_c)
        for m in POINT_MODELS:
            fc = ft[f"{m}_h{h}"].to_numpy()
            rows = pd.DataFrame({"t_start": ft["t_start"], "h": h, "model": m, "unit": a["unit"],
                                 "category": a["category"], "phase": a["phase"], "superintense": a["superintense"],
                                 "common": common_t, "y": y, "f": fc,
                                 "se_m": (fc - y) ** 2, "se_p": (ft[f"P_h{h}"].to_numpy() - y) ** 2,
                                 "se_e": (ft[f"E_h{h}"].to_numpy() - y) ** 2})
            values = {"se_m": rows.se_m, "se_p": rows.se_p, "se_e": rows.se_e}
            if m in INTERVAL_MODELS:
                run = main_intervals.run(f["t_start"], f[f"y_h{h}"].to_numpy(), f[f"{m}_h{h}"].to_numpy(),
                                         f["dst_t"].to_numpy(), f[f"common_h{h}"].to_numpy(), is_test, h,
                                         calib_end, merged[f"h{h}"]["merged_into"])
                for meth in METHODS:
                    rows[f"q_{meth}"] = run[meth]["q"][is_test]
                    rows[f"cov_{meth}"] = run[meth]["covered"][is_test]
                    values[f"cov_{meth}"] = rows[f"cov_{meth}"]
                interval_info[f"{m}_h{h}"] = {meth: {"n_infinite": run[meth]["n_infinite"], "n_empty": run[meth]["n_empty"]}
                                              for meth in METHODS if meth not in ("split", "mond")}
                cov_c = crossfit.coverage(saved["t_start"], h, saved[f"y_h{h}"].to_numpy(), saved[f"{m}_h{h}"].to_numpy(),
                                          saved["dst_t"].to_numpy(), saved[f"common_h{h}"].to_numpy(), half,
                                          merged[f"h{h}"]["merged_into"])
                calib_tables[(m, h)] = metrics.per_unit(ac["unit"], saved[f"common_h{h}"].to_numpy(), unit_hours_c,
                                                        {"cov_mond": cov_c["mond"], "cov_dtaci": cov_c["dtaci"]})
            table = metrics.per_unit(rows["unit"], common_t, unit_hours_t, values)
            table.to_csv(OUT / f"units_{m}_h{h}.csv")
            summary_metrics[f"{m}_h{h}"] = secondary(rows, table, u_t, m)
            if m in INTERVAL_MODELS:
                test_tables[(m, h)] = table
                c = calib_tables[(m, h)].loc[ids["calib_intense"]]
                summary_metrics[f"{m}_h{h}"]["M9"] = {
                    meth: metrics.coverage(c[f"cov_{meth}"][c.included]) for meth in ("mond", "dtaci")}
            all_rows.append(rows)

    stats = decisions.statistics(test_tables, calib_tables, ids, counts)
    dec = decisions.decide(stats)
    for k, v in stats["H3"].items():
        dec["H3"][k]["p_permutation"] = decisions.sign_flip_p(v["d_units"])
        dec["H3"][k]["width_ratio_dtaci_to_mond"] = (
            summary_metrics[f"{k[0]}_h{k[1]}"]["intense"]["M4_median_width"]["dtaci"]
            / summary_metrics[f"{k[0]}_h{k[1]}"]["intense"]["M4_median_width"]["mond"])
        dec["H3"][k]["permutation_disagrees"] = bool(
            (dec["H3"][k]["p_holm"] < 0.05) != (dec["H3"][k]["p_permutation"] < 0.05))
    fragile = decisions.fragility(test_tables, calib_tables, ids, counts, dec)
    caveat_m10 = {k: v["intense"]["M10_hours_excluded"] > M10_CAVEAT for k, v in summary_metrics.items()}

    rows_all = pd.concat(all_rows, ignore_index=True)
    rows_path = ROOT / "results" / "forecasts" / "test_rows.parquet"
    rows_all.to_parquet(rows_path, index=False)
    summary = {
        "session": session, "commit": commit, "test_openings": n_openings,
        "units_test": {c: len(category_units(u_t, c)) for c in CATEGORIES},
        "units_calib_intense": len(ids["calib_intense"]),
        "decisions": {h: {f"{k[0]}_h{k[1]}": v for k, v in d.items()} for h, d in dec.items()},
        "fragile_S6": fragile, "m10_caveat": caveat_m10,
        "metrics": summary_metrics, "adaptive_infinite_or_empty": interval_info,
        "test_rows_sha256": sha256(rows_path),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    print(f"test opening no. {n_openings}, commit {commit[:7]}")
    for hyp in ("H1", "H2", "H3"):
        for k, v in summary["decisions"][hyp].items():
            extra = ""
            if hyp == "H2":
                extra = f", M9 {v['m9']:.3f}, M2(test) {v['m2_test']:.3f}"
            if hyp == "H3":
                extra = (f", M2 C-DtACI {v['m2_dtaci']:.3f}, C-Mond {v['m2_mond']:.3f}, "
                         f"p permutation {v['p_permutation']:.5f}")
            print(f"{hyp} {k}: estimate {v['estimate']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}], "
                  f"p {v['p']:.5f}, Holm {v['p_holm']:.5f}{extra} -> {'supported' if v['supported'] else 'not supported'}")
    print(f"S6 fragile decisions: {fragile or 'none'}")
    print(f"M10 > 10 % in intense storms: {[k for k, v in caveat_m10.items() if v] or 'none'}")


if __name__ == "__main__":
    main()
