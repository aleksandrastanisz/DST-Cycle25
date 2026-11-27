"""Classes of C-Mond and their merging (protocol section 7.3).

Classes of Dst(t) at issue time, from the class closest to zero:
  0: Dst(t) > -30,  1: -50 < Dst(t) <= -30,  2: -100 < Dst(t) <= -50,  3: Dst(t) <= -100 nT.
A class with fewer than 200 calibration hours is merged with its neighbour closer to zero.
Merging runs from the most negative class towards zero, so a class that is still too small
after taking in its neighbour is merged again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UPPER_EDGES = (-30.0, -50.0, -100.0)  # a class k > 0 holds UPPER_EDGES[k] < ... <= UPPER_EDGES[k-1]
N_CLASSES = 4
MIN_HOURS = 200
CALIB_RESIDUALS_FROM = pd.Timestamp("2010-01-28", tz="UTC")


def classes(dst_t: np.ndarray) -> np.ndarray:
    """Class 0-3 of each Dst(t); -1 where Dst(t) is missing."""
    d = np.asarray(dst_t, dtype=float)
    k = np.zeros(len(d), dtype=int)
    for i, edge in enumerate(UPPER_EDGES, start=1):
        k[d <= edge] = i
    k[np.isnan(d)] = -1
    return k


def merge_map(counts: list[int], min_hours: int = MIN_HOURS) -> list[int]:
    """For each original class, the class it belongs to after merging (labelled by its member
    closest to zero)."""
    target = list(range(N_CLASSES))
    sizes = list(counts)
    for k in range(N_CLASSES - 1, 0, -1):
        if sizes[k] < min_hours:
            sizes[k - 1] += sizes[k]
            sizes[k] = 0
            for j in range(N_CLASSES):
                if target[j] == k:
                    target[j] = k - 1
    return target
