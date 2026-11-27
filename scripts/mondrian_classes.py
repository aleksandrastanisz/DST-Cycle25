"""Class sizes of C-Mond on calib and the merging decision (protocol sections 7.3 and 15).

Counts calibration hours from 2010-01-28 in the common set (all four models forecast and the
target is known), separately for each horizon, and applies the merging rule. Reads only the calib
forecasts; writes results/mondrian_classes.json.

    python scripts/mondrian_classes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import mondrian  # noqa: E402
from dstc25.forecast import HORIZONS  # noqa: E402

LABELS = ["Dst(t) > -30", "-50 < Dst(t) <= -30", "-100 < Dst(t) <= -50", "Dst(t) <= -100"]


def main() -> None:
    f = pd.read_parquet(ROOT / "results" / "forecasts" / "calib.parquet")
    k = mondrian.classes(f["dst_t"].to_numpy())
    report = {"min_hours": mondrian.MIN_HOURS, "from": str(mondrian.CALIB_RESIDUALS_FROM.date()), "labels": LABELS}
    for h in HORIZONS:
        use = (f["t_start"] >= mondrian.CALIB_RESIDUALS_FROM).to_numpy() & f[f"common_h{h}"].to_numpy()
        counts = [int(np.sum(use & (k == c))) for c in range(mondrian.N_CLASSES)]
        target = mondrian.merge_map(counts)
        report[f"h{h}"] = {"hours": int(use.sum()), "counts": counts, "merged_into": target}
        print(f"h={h}: {int(use.sum())} hours; " + "; ".join(f"{LABELS[c]}: {counts[c]}" for c in range(4)))
        merged = [c for c in range(mondrian.N_CLASSES) if target[c] != c]
        print(f"h={h}: " + ("no class merged" if not merged else
                             ", ".join(f"'{LABELS[c]}' merged into '{LABELS[target[c]]}'" for c in merged)))
    (ROOT / "results" / "mondrian_classes.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
