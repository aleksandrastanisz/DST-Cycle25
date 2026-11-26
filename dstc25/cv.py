"""Blocked cross-validation within train (protocol section 7.2).

Train is split into five 3-year blocks. In each fold one block is the validation block; the
model is fitted on the other blocks without the 27 days adjacent to the validation block on
either side. Rows are assigned by their issue time (t_start of the issue row). The 27-day gap
is far longer than the longest look-back (24 h) and horizon (6 h), so no training row sees a
validation input or target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCKS = [(1995, 1997), (1998, 2000), (2001, 2003), (2004, 2006), (2007, 2009)]
GAP = pd.Timedelta(days=27)
TUNING_SEED = 100


def block_bounds(block: tuple[int, int]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hourly t_start of a block of years, in UTC."""
    first, last = block
    return pd.Timestamp(f"{first}-01-01", tz="UTC"), pd.Timestamp(f"{last}-12-31 23:00", tz="UTC")


def fold_masks(t_start: pd.Series, block: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, validation) masks for the fold whose validation block is `block`."""
    lo, hi = block_bounds(block)
    val = ((t_start >= lo) & (t_start <= hi)).to_numpy()
    train = ((t_start < lo - GAP) | (t_start > hi + GAP)).to_numpy()
    return train, val


def folds(t_start: pd.Series):
    """Yield (block, train mask, validation mask) for the five folds in block order."""
    for block in BLOCKS:
        train, val = fold_masks(t_start, block)
        yield block, train, val
