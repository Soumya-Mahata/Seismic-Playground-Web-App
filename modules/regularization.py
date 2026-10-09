"""
modules/regularization.py — Data Regularisation
====================================================================
Transforms irregularly sampled seismic data onto a regular offset grid.
A regular grid is a prerequisite for reliable subsequent velocity
analysis, stacking, migration, and attribute analysis — all of which
assume (or work best with) consistent offset sampling across CDPs.

Here, the irregularity being fixed is in the OFFSET dimension within
each CDP gather: real fold is rarely a clean, evenly spaced set of
offsets (near-trace gaps, dropped channels, split-spread geometry
irregularities), which biases both semblance velocity analysis (which
implicitly assumes reasonably even offset sampling for a fair NMO scan)
and any migration operator that sums over offset. `regularize_cdp_gather`
interpolates each gather onto a common, regularly spaced target-offset
grid built once for the whole survey; `build_regular_offset_grid` builds
that shared target grid from the union of all live offsets actually
present.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def build_regular_offset_grid(gathers: Dict[int, dict], n_offsets: int = 31,
                                offsets_key: str = "offsets") -> np.ndarray:
    """Build one shared, regularly spaced target-offset axis spanning the
    survey's actual min/max offset — every gather is later interpolated
    onto this same axis so downstream steps see a consistent geometry."""
    all_off = np.concatenate([np.asarray(g[offsets_key], dtype=float) for g in gathers.values()])
    lo, hi = float(np.min(all_off)), float(np.max(all_off))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_offsets)


def regularize_cdp_gather(traces: np.ndarray, offsets: np.ndarray, target_offsets: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate one CDP gather's irregularly sampled offsets onto
    `target_offsets`, sample-by-sample (i.e. trace interpolation across
    offset, not across time) using linear interpolation with no
    extrapolation (target offsets outside this gather's actual live
    offset range are returned as exact zero and flagged dead in
    `live_mask`, rather than silently extrapolated).

    Returns (regularized_traces, live_mask) where `regularized_traces`
    has shape (len(target_offsets), n_samples) and `live_mask` is a
    boolean array of the same length marking which target offsets fall
    inside this gather's actual fold.
    """
    offsets = np.asarray(offsets, dtype=float)
    order = np.argsort(offsets)
    offs_sorted = offsets[order]
    tr_sorted = np.asarray(traces, dtype=float)[order]
    ns = tr_sorted.shape[1]
    n_tgt = len(target_offsets)
    out = np.zeros((n_tgt, ns))
    if len(offs_sorted) >= 2:
        for it in range(ns):
            out[:, it] = np.interp(target_offsets, offs_sorted, tr_sorted[:, it], left=0.0, right=0.0)
    elif len(offs_sorted) == 1:
        # A single-fold gather can't be interpolated across offset — hold
        # its one trace flat across the whole target axis rather than
        # producing an all-zero (and therefore misleadingly "dead") panel.
        out[:, :] = tr_sorted[0, :][None, :]
    live_mask = (target_offsets >= offs_sorted.min()) & (target_offsets <= offs_sorted.max()) \
        if len(offs_sorted) else np.zeros(n_tgt, dtype=bool)
    return out, live_mask


def regularize_all_gathers(gathers: Dict[int, dict], input_key: str, offsets_key: str = "offsets",
                            n_offsets: int = 31, out_traces_key: str = "traces_regularized",
                            out_offsets_key: str = "offsets_regularized",
                            out_live_key: str = "live_regularized") -> np.ndarray:
    """Regularizes every CDP gather onto one shared target-offset axis,
    writing the results back into each gather dict (same accumulation
    pattern as every other pipeline stage). Returns the shared target
    offset axis."""
    target_offsets = build_regular_offset_grid(gathers, n_offsets=n_offsets, offsets_key=offsets_key)
    for g in gathers.values():
        reg, live = regularize_cdp_gather(g[input_key], g[offsets_key], target_offsets)
        g[out_traces_key] = reg
        g[out_offsets_key] = target_offsets
        g[out_live_key] = live
    return target_offsets


def fold_before_after(gathers: Dict[int, dict], offsets_key: str = "offsets",
                       live_key: str = "live_regularized") -> Tuple[np.ndarray, np.ndarray]:
    """Per-CDP fold before (actual live traces) vs. after (regularized,
    interpolation-supported offsets) regularisation, for QC."""
    before = np.array([len(g[offsets_key]) for g in gathers.values()])
    after = np.array([int(np.sum(g[live_key])) for g in gathers.values()])
    return before, after
