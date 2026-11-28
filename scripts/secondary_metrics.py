"""Secondary metrics after E-main (protocol section 8): M7, M8, M11, M12, metrics per storm phase
and the share of infinite adaptive intervals in storm windows.

Uses the hourly rows saved by E-main (results/forecasts/test_rows.parquet, checked against its
SHA-256 in results/e_main/summary.json) and the calib forecasts. Runs through dstc25/guard.py
and is logged in results/test_openings.json, because it reads test data.

    python scripts/secondary_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import guard, secondary, units  # noqa: E402
from dstc25.runinfo import session_info, sha256  # noqa: E402

HORIZONS = (1, 6)
INTERVAL_MODELS = ("X", "L")
METHODS = ("split", "mond", "dtaci", "aci_0.005", "aci_0.05")
OUT = ROOT / "results" / "secondary"


def main() -> None:
    commit = guard.check(ROOT)
    n_openings = guard.log_opening(ROOT / "results" / "test_openings.json", commit, "secondary metrics")
    session = session_info()
    OUT.mkdir(parents=True, exist_ok=True)

    e_main = json.loads((ROOT / "results" / "e_main" / "summary.json").read_text())
    rows_path = ROOT / "results" / "forecasts" / "test_rows.parquet"
    if sha256(rows_path) != e_main["test_rows_sha256"]:
        raise SystemExit("test_rows.parquet does not match its SHA-256 in results/e_main/summary.json")
    rows_all = pd.read_parquet(rows_path)
    splits = json.loads((ROOT / "data" / "MANIFEST.json").read_text())["splits"]
    files = [ROOT / "data" / "processed" / "train.parquet", ROOT / "data" / "processed" / "calib.parquet",
             ROOT / "data" / "sealed" / "test.parquet"]
    dst = pd.concat([pd.read_parquet(p, columns=["t_start", "Dst"]) for p in files]).set_index("t_start")["Dst"]
    _, u_t = units.split_units(dst, "test", *splits["test"])
    hours_c, u_c = units.split_units(dst, "calib", *splits["calib"])
    calib = pd.read_parquet(ROOT / "results" / "forecasts" / "calib.parquet")

    report = {"session": session, "commit": commit, "test_openings": n_openings, "results": {}}
    for (m, h), rows in rows_all.groupby(["model", "h"], sort=False):
        rows = rows.reset_index(drop=True)
        methods = METHODS if m in INTERVAL_MODELS else ()
        w = secondary.per_window(rows, u_t, h, methods)
        w.to_csv(OUT / f"windows_{m}_h{h}.csv", index=False)
        res = {"M7": {c: secondary.m7_summary(w[w.category == c], h) for c in ("moderate", "intense")},
               "M8": {c: {"median_min_error": float(w.loc[(w.category == c) & w.included, "M8_min_error"].median()),
                          "median_abs_timing_error": float(w.loc[(w.category == c) & w.included, "M8_timing_error"].abs().median())}
                      for c in ("moderate", "intense")},
               "M11": secondary.dose_response(w, methods)}
        included = set(w.loc[w.included, "unit"])
        res["phases"] = {c: secondary.phase_metrics(rows[rows["category"] == c], included, methods)
                         for c in ("moderate", "intense")}

        # M12: RMSE in test over RMSE in calib, per category (hour-pooled, common hours)
        ac = units.for_forecasts(calib["t_start"], h, hours_c)
        se_c = (calib[f"{m}_h{h}"] - calib[f"y_h{h}"]) ** 2
        common_c = calib[f"common_h{h}"].to_numpy()
        res["M12"] = {}
        for c in ("quiet", "moderate", "intense"):
            rmse_c = float(np.sqrt(se_c[common_c & (ac["category"] == c).to_numpy()].mean()))
            rmse_t = e_main["metrics"][f"{m}_h{h}"][c]["M3_rmse"]
            res["M12"][c] = {"rmse_test": rmse_t, "rmse_calib": rmse_c, "ratio": rmse_t / rmse_c}

        if methods:
            r = rows[(rows["category"] == "intense").to_numpy() & rows["common"].to_numpy()
                     & rows["unit"].isin(included).to_numpy()]
            res["infinite_in_intense"] = {}
            for meth in ("dtaci", "aci_0.005", "aci_0.05"):
                inf = np.isinf(r[f"q_{meth}"].to_numpy())
                finite = r.loc[~inf]
                res["infinite_in_intense"][meth] = {
                    "hours": int(len(r)), "infinite_hours": int(inf.sum()), "share": float(inf.mean()),
                    "windows_with_infinite": int((w.loc[w.included & (w.category == "intense"), f"infinite_{meth}"] > 0).sum()),
                    "coverage_finite_hours_only": float(finite[f"cov_{meth}"].mean()) if len(finite) else float("nan"),
                }
        report["results"][f"{m}_h{h}"] = res

    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"test opening no. {n_openings}, commit {commit[:7]}")
    for key, res in report["results"].items():
        m7 = res["M7"]["intense"]
        line = (f"{key}: M7 intense median shift {m7['median_shift']:.1f} h, shift >= h in "
                f"{m7['fraction_shift_ge_h']:.0%} of {m7['windows']} windows; M12 intense {res['M12']['intense']['ratio']:.2f}")
        if "infinite_in_intense" in res:
            d = res["infinite_in_intense"]["dtaci"]
            line += (f"; C-DtACI infinite in intense {d['infinite_hours']}/{d['hours']} h "
                     f"({d['share']:.1%}, {d['windows_with_infinite']} windows), coverage on finite hours "
                     f"{d['coverage_finite_hours_only']:.3f}")
        print(line)


if __name__ == "__main__":
    main()
