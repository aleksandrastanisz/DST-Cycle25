"""Count storm windows per split (protocol section 6), from Dst alone.

Reads only the Dst column. For the test split it prints and saves counts only, no values or dates.

    python scripts/count_storms.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dstc25 import storms  # noqa: E402

FILES = {
    "train": ROOT / "data" / "processed" / "train.parquet",
    "calib": ROOT / "data" / "processed" / "calib.parquet",
    "test": ROOT / "data" / "sealed" / "test.parquet",
}


def main() -> None:
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    splits = manifest["splits"]
    parts = [pd.read_parquet(path, columns=["t_start", "Dst"]) for path in FILES.values()]
    dst = pd.concat(parts).set_index("t_start")["Dst"]
    windows = storms.storm_windows(dst)

    counts = {}
    for name, (lo, hi) in splits.items():
        counts[name] = storms.count_by_category(storms.assign_to_split(windows, lo, hi))
        print(f"{name}: {counts[name]}")

    out = ROOT / "results" / "storm_counts.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(counts, indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
