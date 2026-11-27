"""Power analysis P0 (protocol section 10.4; details in section 13, 2026-11-27). Informational only.

On calib, with the final models trained on train: point values of the H1-H3 statistics (H2 and
H3 from C-Mond and C-DtACI intervals cross-fitted on the halves of calib, as in M9), their
standard errors from a bootstrap that resamples as many units as the test period contains, and
the smallest effect detectable with 80 % power, MDE80 = max(d, z * SE) + z80 * SE.

Reads the calib forecasts and the Dst column of all splits (units are defined from Dst alone;
the test split contributes only through results/unit_counts.json). Writes results/p0.json.

    python scripts/p0.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import bootstrap as bs  # noqa: E402
from dstc25 import crossfit, metrics, units  # noqa: E402
from dstc25.runinfo import session_info  # noqa: E402

MODELS = ("X", "L")
HORIZONS = (1, 6)
THRESHOLD = {"H1": 0.05, "H2": 0.05, "H3": 0.03}
Z_HOLM = NormalDist().inv_cdf(1 - 0.05 / 4)  # strictest Holm level with 4 tests, about 2.24
Z_POWER = NormalDist().inv_cdf(0.80)        # about 0.84
FILES = {
    "train": ROOT / "data" / "processed" / "train.parquet",
    "calib": ROOT / "data" / "processed" / "calib.parquet",
    "test": ROOT / "data" / "sealed" / "test.parquet",
}


def mde80(se: float, d: float) -> float:
    return max(d, Z_HOLM * se) + Z_POWER * se


def main() -> None:
    session = session_info()
    if session["uncommitted_changes"]:
        print("Warning: tracked files have uncommitted changes; this is recorded in the results.")
    splits = json.loads((ROOT / "data" / "MANIFEST.json").read_text())["splits"]
    unit_counts = json.loads((ROOT / "results" / "unit_counts.json").read_text())
    merged_into = json.loads((ROOT / "results" / "mondrian_classes.json").read_text())
    dst = pd.concat([pd.read_parquet(p, columns=["t_start", "Dst"]) for p in FILES.values()]).set_index("t_start")["Dst"]
    hours, u = units.split_units(dst, "calib", *splits["calib"])
    u = u.sort_values("unit").reset_index(drop=True)
    if (u.kind == "quiet").sum() != unit_counts["calib"]["quiet_blocks"] or \
            (u.category == "intense").sum() != unit_counts["calib"]["intense"]:
        raise SystemExit("calib units differ from results/unit_counts.json")
    quiet_ids = u.loc[u.kind == "quiet", "unit"].tolist()
    intense_ids = u.loc[u.category == "intense", "unit"].tolist()
    n_test_quiet, n_test_intense = unit_counts["test"]["quiet_blocks"], unit_counts["test"]["intense"]

    rng = np.random.default_rng(bs.SEED_P0)  # draws in the order fixed in section 13
    c_quiet = bs.draw_counts(rng, len(quiet_ids), n_test_quiet)
    c_int24 = bs.draw_counts(rng, len(intense_ids), n_test_intense)
    c_int26 = bs.draw_counts(rng, len(intense_ids), len(intense_ids))

    f = pd.read_parquet(ROOT / "results" / "forecasts" / "calib.parquet")
    unit_hours = u.set_index("unit")["n_hours"]
    report = {"session": session, "z_holm": Z_HOLM, "z_power": Z_POWER, "thresholds": THRESHOLD,
              "draws": {"quiet": [len(quiet_ids), n_test_quiet], "intense_test": [len(intense_ids), n_test_intense],
                        "intense_m9": [len(intense_ids), len(intense_ids)]},
              "results": {}}
    rows_out = []
    for h in HORIZONS:
        a = units.for_forecasts(f["t_start"], h, hours)
        common = f[f"common_h{h}"].to_numpy()
        y = f[f"y_h{h}"].to_numpy()
        half, n_cross = crossfit.halves(f["t_start"], h, a["unit"], u)
        report[f"crossing_windows_h{h}"] = n_cross
        for m in MODELS:
            fc = f[f"{m}_h{h}"].to_numpy()
            cov = crossfit.coverage(f["t_start"], h, y, fc, f["dst_t"].to_numpy(), common, half,
                                    merged_into[f"h{h}"]["merged_into"])
            t = metrics.per_unit(a["unit"], common, unit_hours, {
                "se_m": (fc - y) ** 2, "se_p": (f[f"P_h{h}"].to_numpy() - y) ** 2,
                "cov_mond": cov["mond"], "cov_dtaci": cov["dtaci"],
            })
            tq, ti = t.loc[quiet_ids], t.loc[intense_ids]
            iq, ii = tq.included.to_numpy(), ti.included.to_numpy()

            m1q = metrics.skill(tq.se_m[iq], tq.se_p[iq])
            m1i = metrics.skill(ti.se_m[ii], ti.se_p[ii])
            m9_mond, m9_dtaci = metrics.coverage(ti.cov_mond[ii]), metrics.coverage(ti.cov_dtaci[ii])
            d_h3 = m9_dtaci - m9_mond

            def m1_reps(tt, inc, c):
                return 1 - bs.weighted_mean(c, tt.se_m, inc) / bs.weighted_mean(c, tt.se_p, inc)

            delta_reps = m1_reps(tq, iq, c_quiet) - m1_reps(ti, ii, c_int24)
            se = {
                "H1": float(np.nanstd(delta_reps)),
                "H2": float(np.hypot(np.nanstd(bs.weighted_mean(c_int26, ti.cov_mond, ii)),
                                     np.nanstd(bs.weighted_mean(c_int24, ti.cov_mond, ii)))),
                "H3": float(np.nanstd(bs.weighted_mean(c_int24, ti.cov_dtaci - ti.cov_mond, ii))),
            }
            res = {
                "units_included": {"quiet": int(iq.sum()), "quiet_all": len(iq),
                                   "intense": int(ii.sum()), "intense_all": len(ii)},
                "calib": {"M1_quiet": m1q, "M1_intense": m1i, "H1_delta": m1q - m1i,
                          "M9_mond": m9_mond, "M9_dtaci": m9_dtaci, "H3_D": d_h3},
                "se": se,
                "mde80": {k: mde80(se[k], THRESHOLD[k]) for k in se},
            }
            report["results"][f"{m}_h{h}"] = res
            rows_out.append((m, h, res))

    (ROOT / "results" / "p0.json").write_text(json.dumps(report, indent=1))
    for h in HORIZONS:
        print(f"h={h}: storm windows crossing the boundary of the calib halves: {report[f'crossing_windows_h{h}']}")
    for m, h, r in rows_out:
        c, s, d = r["calib"], r["se"], r["mde80"]
        print(f"{m} h={h}: units quiet {r['units_included']['quiet']}/{r['units_included']['quiet_all']}, "
              f"intense {r['units_included']['intense']}/{r['units_included']['intense_all']}")
        print(f"  H1: M1 quiet {c['M1_quiet']:.3f}, intense {c['M1_intense']:.3f}, delta {c['H1_delta']:.3f}; "
              f"SE {s['H1']:.3f}, MDE80 {d['H1']:.3f}")
        print(f"  H2: M9 C-Mond {c['M9_mond']:.3f}; SE {s['H2']:.3f}, MDE80 {d['H2']:.3f}")
        print(f"  H3: M9 C-DtACI {c['M9_dtaci']:.3f}, D {c['H3_D']:.3f}; SE {s['H3']:.3f}, MDE80 {d['H3']:.3f}")


if __name__ == "__main__":
    main()
