"""
modules/statics.py — Step 12 of the reference pipeline
==========================================================
Surface-consistent (shot + receiver) residual statics: cross-correlate
each 1st-pass-NMO trace against its CDP's fold-normalized pilot trace,
then solve for per-shot and per-receiver static shifts by least squares.

Ported directly from `Seismic_Processing_CORRECTED.ipynb`
(`crosscorr_lag`, the shot/receiver least-squares system, and
`apply_static_shift`). No change to the underlying math.

Documented limitation (carried over from the notebook): field statics
are not computed here — the SEG-Y headers this pipeline was built
against have no weathering-velocity / uphole-time information, so there
is nothing to compute a field static from. Only the data-driven,
surface-consistent residual is applicable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import correlate


def crosscorr_lag(trace: np.ndarray, pilot: np.ndarray, dt: float, max_shift_s: float) -> float:
    """Best-fit time lag (seconds) that aligns `trace` to `pilot`, search
    restricted to +/- max_shift_s."""
    max_shift = int(max_shift_s / dt)
    if np.allclose(trace, 0) or np.allclose(pilot, 0):
        return 0.0
    xc = correlate(trace, pilot, mode="full")
    lags = np.arange(-len(trace) + 1, len(trace))
    center = len(xc) // 2
    lo, hi = center - max_shift, center + max_shift + 1
    window = xc[lo:hi]
    best = np.argmax(window)
    return lags[lo + best] * dt


def apply_static_shift(trace: np.ndarray, shift_s: float, dt: float) -> np.ndarray:
    """Shift a trace in time by `shift_s` via linear interpolation
    (zero-padded outside the original range)."""
    n = len(trace)
    t_orig = np.arange(n) * dt
    return np.interp(t_orig - shift_s, t_orig, trace, left=0.0, right=0.0)


def solve_residual_statics(gathers: Dict[int, dict], input_key: str, live_mask_key: str,
                            dt: float, max_shift_s: float = 0.02
                            ) -> Tuple[Dict[int, float], Dict[int, float], dict]:
    """Estimate surface-consistent (shot + receiver) residual static
    shifts from a set of 1st-pass-NMO-corrected gathers.

    Returns (shot_static, receiver_static, diagnostics). Mutates nothing;
    call `apply_residual_statics` separately to apply the result to
    (typically different, pre-NMO) gathers.
    """
    rows_shot, rows_recv, obs_shift = [], [], []
    all_shots = sorted(set(int(s) for g in gathers.values() for s in g["shot_ids"]))
    all_recv = sorted(set(int(r) for g in gathers.values() for r in g["receiver_ids"]))
    shot_idx = {s: i for i, s in enumerate(all_shots)}
    recv_idx = {r: i for i, r in enumerate(all_recv)}

    for cdp, g in gathers.items():
        live = g[live_mask_key]
        traces = g[input_key]
        pilot = np.where(live, traces, 0.0).sum(axis=0)
        fold = live.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pilot = np.where(fold > 0, pilot / np.maximum(fold, 1), 0.0)
        for ix in range(traces.shape[0]):
            lag = crosscorr_lag(traces[ix, :], pilot, dt, max_shift_s)
            rows_shot.append(shot_idx[int(g["shot_ids"][ix])])
            rows_recv.append(recv_idx[int(g["receiver_ids"][ix])])
            obs_shift.append(lag)

    obs_shift = np.array(obs_shift)
    n_obs, n_shot, n_recv = len(obs_shift), len(all_shots), len(all_recv)
    A = np.zeros((n_obs, n_shot + n_recv))
    A[np.arange(n_obs), rows_shot] = 1.0
    A[np.arange(n_obs), n_shot + np.array(rows_recv)] = 1.0
    # zero-mean constraint on the shot statics removes the shot/receiver null space
    A_reg = np.vstack([A, np.r_[np.ones(n_shot), np.zeros(n_recv)] * 10.0])
    b_reg = np.r_[obs_shift, 0.0]
    solution, *_ = np.linalg.lstsq(A_reg, b_reg, rcond=None)

    shot_static = dict(zip(all_shots, solution[:n_shot]))
    receiver_static = dict(zip(all_recv, solution[n_shot:]))
    diagnostics = {
        "n_shots": n_shot, "n_receivers": n_recv, "n_observations": n_obs,
        "rms_shot_static_ms": float(np.std(solution[:n_shot]) * 1000),
        "rms_receiver_static_ms": float(np.std(solution[n_shot:]) * 1000),
        "rms_observed_shift_ms": float(np.std(obs_shift) * 1000),
    }
    return shot_static, receiver_static, diagnostics


def apply_residual_statics(gathers: Dict[int, dict], input_key: str, output_key: str,
                            shot_static: Dict[int, float], receiver_static: Dict[int, float],
                            dt: float) -> None:
    """Apply solved shot+receiver static shifts to `input_key`, writing
    `output_key` into every gather (mutates `gathers` in place)."""
    for cdp, g in gathers.items():
        traces = g[input_key]
        corrected = np.zeros_like(traces)
        for ix in range(traces.shape[0]):
            total_static = (shot_static[int(g["shot_ids"][ix])]
                             + receiver_static[int(g["receiver_ids"][ix])])
            corrected[ix] = apply_static_shift(traces[ix], total_static, dt)
        g[output_key] = corrected
