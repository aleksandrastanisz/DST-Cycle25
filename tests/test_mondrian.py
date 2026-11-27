"""Classes of C-Mond and their merging (protocol section 7.3)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import mondrian  # noqa: E402


def test_class_boundaries():
    d = np.array([10, -29.9, -30, -49.9, -50, -99.9, -100, -300, np.nan])
    assert mondrian.classes(d).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1]


def test_no_merge_when_all_large():
    assert mondrian.merge_map([5000, 800, 400, 250]) == [0, 1, 2, 3]


def test_small_extreme_class_merged_towards_zero():
    assert mondrian.merge_map([5000, 800, 400, 150]) == [0, 1, 2, 2]


def test_merge_cascades():
    # class 3 joins class 2 (120 + 50 < 200), so the merged class joins class 1
    assert mondrian.merge_map([5000, 800, 120, 50]) == [0, 1, 1, 1]


def test_middle_class_merged():
    assert mondrian.merge_map([5000, 150, 400, 250]) == [0, 0, 2, 3]
