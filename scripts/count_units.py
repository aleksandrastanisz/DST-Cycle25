"""Units of analysis per split (protocol section 6): storm windows by category and quiet blocks.

Reads only the Dst column of all splits, as scripts/count_storms.py does; for the test split it
prints and saves counts only. The number of test quiet blocks is needed by P0 (section 10.4).
Checks that the storm counts agree with results/storm_counts.json.

    python scripts/count_units.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import units  # noqa: E402

FILES = {
    "train": ROOT / "data" / "processed" / "train.parquet",
    "calib": ROOT / "data" / "processed" / "calib.parquet",
    "test": ROOT / "data" / "sealed" / "test.parquet",
}


def main() -> None:
    splits = json.loads((ROOT / "data" / "MANIFEST.json").read_text())["splits"]
    storm_counts = json.loads((ROOT / "results" / "storm_counts.json").read_text())
    parts = [pd.read_parquet(path, columns=["t_start", "Dst"]) for path in FILES.values()]
    dst = pd.concat(parts).set_index("t_start")["Dst"]

    report = {}
    for name, (lo, hi) in splits.items():
        hours, u = units.split_units(dst, name, lo, hi)
        counts = {
            "quiet_blocks": int((u.kind == "quiet").sum()),
            "moderate": int((u.category == "moderate").sum()),
            "intense": int((u.category == "intense").sum()),
            "superintense": int(u.superintense.sum()),
            "hours": len(hours),
            "hours_without_unit": int(hours.unit.isna().sum()),
        }
        expected = storm_counts[name]
        if any(counts[k] != expected[k] for k in ("moderate", "intense", "superintense")):
            raise SystemExit(f"{name}: storm counts {counts} differ from results/storm_counts.json {expected}")
        report[name] = counts
        print(f"{name}: {counts}")
    (ROOT / "results" / "unit_counts.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
