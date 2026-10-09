"""
modules/stacking.py — CDP Stacking
===========================================================
Fold-normalized CDP stacking. A naive (un-normalized) mean stack is
also provided purely as a labelled comparison (fold-normalization
matters wherever fold varies across the gather, e.g. after a stretch
mute zeroes some near-time, far-offset samples).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def fold_normalized_stack(traces: np.ndarray, live_mask: np.ndarray | None = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack traces, dividing by the number of *live* traces at each time
    sample (not the nominal fold), so a stretch-muted or partially-dead
    gather doesn't have its amplitude diluted by zeroed samples."""
    if live_mask is None:
        live_mask = ~np.isclose(traces, 0.0)
    fold = live_mask.sum(axis=0)
    summed = np.where(live_mask, traces, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stack = np.where(fold > 0, summed / np.maximum(fold, 1), 0.0)
    return stack, fold


def naive_mean_stack(traces: np.ndarray) -> np.ndarray:
    """Un-normalized mean stack — kept only as a labelled before/after
    comparison; dilutes amplitude wherever live fold is less than the
    nominal trace count (e.g. after an NMO stretch mute)."""
    return np.mean(traces, axis=0)


def build_stack_section(gathers: Dict[int, dict], cdp_list: List[int],
                         traces_key: str, live_mask_key: str
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack every CDP gather and assemble the results into 2-D
    (n_cdp, n_samples) matrices, plus a naive-mean comparison.

    Also writes `stack_dmo`/`stack_naive_mean`/`fold` back into each
    gather dict so single-CDP before/after QC plots can reach into the
    per-gather values too.
    """
    stacks, folds, naive = [], [], []
    for cdp in cdp_list:
        g = gathers[cdp]
        traces = g[traces_key]
        live = g.get(live_mask_key)
        stack, fold = fold_normalized_stack(traces, live)
        g["stack_dmo"] = stack
        g["fold"] = fold
        g["stack_naive_mean"] = naive_mean_stack(traces)
        stacks.append(stack)
        folds.append(fold)
        naive.append(g["stack_naive_mean"])
    return np.vstack(stacks), np.vstack(folds), np.vstack(naive)
