"""Units of analysis (protocol section 6): storm windows and quiet blocks, for one split.

Every target hour of a split gets at most one unit: a storm window assigned to the split, a
quiet block, or nothing (hours of a window assigned to a neighbouring split). A forecast belongs
to the unit of its target hour, so a forecast issued at t_start = s for horizon h belongs to the
unit of target hour s + h (section 6, item 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dstc25 import storms

QUIET_BLOCK = pd.Timedelta(days=27)
MIN_QUIET_HOURS = 72


def merge_small_blocks(counts: list[int], min_hours: int = MIN_QUIET_HOURS) -> list[int]:
    """Final block index of each raw block. A block with fewer than `min_hours` quiet hours is
    merged into the previous block, or into the next one if it is the first block. Blocks with
    no quiet hours at all disappear (they are merged like any small block)."""
    n = len(counts)
    owner = list(range(n))
    sizes = list(counts)
    for i in range(n):
        if sizes[i] >= min_hours:
            continue
        if i == 0 or owner[i - 1] == i:  # no earlier block left to merge into
            if i == n - 1:
                continue
            j = i + 1
        else:
            j = owner[i - 1]
        sizes[j] += sizes[i]
        sizes[i] = 0
        for k in range(n):
            if owner[k] == i:
                owner[k] = j
    labels = {b: r for r, b in enumerate(sorted(set(owner)))}
    return [labels[b] for b in owner]


def split_units(dst: pd.Series, name: str, lo: str, hi: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Units of one split from the Dst series of all splits (indexed by tz-aware t_start).

    Returns (hours, units):
      hours: one row per target hour in [lo, hi] with columns unit, kind ('storm', 'quiet' or
        None), category ('quiet', 'moderate', 'intense' or None), phase ('main', 'recovery' or
        None) and superintense (bool);
      units: one row per unit with columns unit, kind, category, superintense, start, end,
        t_min, dst_min, n_hours.
    """
    lo_t, hi_t = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
    every = storms.storm_windows(dst)
    own = storms.assign_to_split(every, lo, hi).sort_values("start").reset_index(drop=True)
    t = pd.date_range(lo_t, hi_t, freq="h")
    hours = pd.DataFrame({"t_target": t, "unit": None, "kind": None, "category": None,
                          "phase": None, "superintense": False})

    in_any = np.zeros(len(t), dtype=bool)
    for _, w in every.iterrows():
        in_any |= ((t >= w.start) & (t <= w.end))
    rows = []
    for k, w in own.iterrows():
        uid = f"{name}-S{k:03d}"
        m = (t >= w.start) & (t <= w.end)
        hours.loc[m, ["unit", "kind", "category", "superintense"]] = [uid, "storm", w.category, bool(w.superintense)]
        hours.loc[m & (t <= w.t_min), "phase"] = "main"
        hours.loc[m & (t > w.t_min), "phase"] = "recovery"
        rows.append({"unit": uid, "kind": "storm", "category": w.category, "superintense": bool(w.superintense),
                     "start": w.start, "end": w.end, "t_min": w.t_min, "dst_min": w.dst_min, "n_hours": int(m.sum())})

    quiet = ~in_any
    raw = ((t - lo_t) // QUIET_BLOCK).to_numpy().astype(int)
    n_raw = int(raw.max()) + 1
    counts = [int(np.sum(quiet & (raw == b))) for b in range(n_raw)]
    final = np.array(merge_small_blocks(counts))[raw]
    for b in np.unique(final[quiet]):
        uid = f"{name}-Q{b:03d}"
        m = quiet & (final == b)
        hours.loc[m, ["unit", "kind", "category"]] = [uid, "quiet", "quiet"]
        rows.append({"unit": uid, "kind": "quiet", "category": "quiet", "superintense": False,
                     "start": t[m].min(), "end": t[m].max(), "t_min": pd.NaT, "dst_min": np.nan,
                     "n_hours": int(m.sum())})
    return hours, pd.DataFrame(rows)


def for_forecasts(t_issue: pd.Series, h: int, hours: pd.DataFrame) -> pd.DataFrame:
    """Unit, kind, category, phase and superintense flag of forecasts issued at `t_issue` for
    horizon h, by their target hour; rows whose target hour lies outside the split get None."""
    target = pd.DatetimeIndex(t_issue) + pd.Timedelta(hours=h)
    table = hours.set_index("t_target")
    out = table.reindex(target)[["unit", "kind", "category", "phase", "superintense"]]
    out["superintense"] = out["superintense"].eq(True)
    return out.reset_index(drop=True)
