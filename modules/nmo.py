"""
modules/nmo.py — Steps 11, 13, 14, 16 of the reference pipeline
===================================================================
Normal moveout (NMO) correction, its inverse, and the simplified DMO
approximation.

Contains one deliberate, documented correction relative to the literal
source notebook, affecting two functions — see "Bug found and fixed"
below.

Bug found and fixed (relative to Seismic_Processing_CORRECTED.ipynb)
------------------------------------------------------------------
For each trace at offset x, the recorded two-way time at zero-offset
time t0 is tn(t0) = sqrt(t0^2 + x^2/vrms(t0)^2). Forward NMO maps the
*recorded* amplitude at tn back to t0: out(t0) = recorded(tn(t0)). Since
a trace is only known at its own sample grid, this means building an
interpolant from (t0_grid, trace_values) pairs and evaluating it at the
query points tn — `np.interp(tn, t0_grid, trace)`.

The source notebook's `nmo_correction` instead calls
`np.interp(t0[valid], tn[valid], traces[ix, valid])` — query and
reference-grid arguments swapped. Verified empirically on a synthetic
trace with a single reflection event at a known time: the notebook's
exact formula places the corrected event at the wrong output time
(mapped a reflector that should flatten to t0=0.500 s to t=0.754 s
instead), while `np.interp(tn, t0_grid, trace)` recovers 0.500 s exactly.

The natural next question — since two functions in this file are meant
to be exact inverses of each other — is whether `inverse_nmo_correction`
was affected too. A direct round-trip test (apply forward NMO, then
inverse NMO, and check the original trace comes back) shows it was:
the notebook's `inverse_nmo_correction` uses the *same* query/reference
convention as the (corrected) forward `nmo_correction` above
(`np.interp(tn, t0_grid, trace)`), which is self-consistent for forward
warping but is **not** actually the inverse of it — applying it after a
forward NMO pass compounds the same warp a second time rather than
undoing it (confirmed numerically: round-tripping a spike through both
functions as originally written moves it further from its start, not
back to it).

The actual inverse of `out[k] = interp(tn[k], t0_grid, v)` is
`out[k] = interp(t0[k], tn_grid, v)` — i.e. exactly the swapped
convention the notebook's (buggy) `nmo_correction` happened to use. In
other words: the notebook's two NMO functions each implement the
*other* one's stated purpose. `nmo_correction` here uses the forward
convention; `inverse_nmo_correction` here uses the swapped convention —
verified with a direct round-trip test
(`tests/test_pipeline.py::test_inverse_nmo_undoes_nmo`) that recovers an
original trace's spikes to within one sample after forward-then-inverse
NMO. `dmo_correction` is left unchanged: it doesn't claim to be an
inverse of anything, and consistently uses the same forward-style
convention to add stretch on top of already-flat data, which is
self-consistent with what it's meant to do.

Because `nmo_correction` is called at both NMO passes (steps 11 and 16)
and `inverse_nmo_correction` feeds directly into the 2nd-pass velocity
analysis (step 15), both bugs would otherwise propagate through most of
the rest of the pipeline. Documented in README.md under "Deviations from
the notebook".
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def nmo_correction(traces: np.ndarray, offsets: np.ndarray, vrms: np.ndarray,
                    dt: float, stretch_mute: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """Forward NMO: flattens a moveout-containing gather to zero-offset
    time t0. See module docstring for the formula and the fix relative
    to the source notebook.

    Returns (corrected_traces, live_mask) — `live_mask` marks samples
    kept after the stretch mute (NMO stretch = (tn-t0)/t0 exceeding
    `stretch_mute` distorts the wavelet too much to be usable and is
    zeroed/excluded from later fold-normalized stacking).
    """
    ntr, nt = traces.shape
    t0 = np.arange(nt) * dt
    out = np.zeros_like(traces)
    live = np.zeros((ntr, nt), dtype=bool)

    for ix in range(ntr):
        tn = np.sqrt(t0 ** 2 + (offsets[ix] ** 2) / (vrms ** 2))
        stretch = np.zeros_like(t0)
        nz = t0 > 0
        stretch[nz] = (tn[nz] - t0[nz]) / t0[nz]
        valid = (stretch < stretch_mute) & (tn <= t0[-1])
        out[ix, valid] = np.interp(tn[valid], t0, traces[ix, :])
        live[ix, :] = valid
    return out, live


def inverse_nmo_correction(nmo_traces: np.ndarray, offsets: np.ndarray,
                            vrms: np.ndarray, dt: float) -> np.ndarray:
    """Exact inverse of `nmo_correction`: re-introduces the moveout that
    a forward NMO pass with the same offsets/velocities would remove.
    See module docstring — this uses the swapped interpolation
    convention relative to the source notebook's `inverse_nmo_correction`,
    which is required for it to be an actual inverse of (the corrected)
    `nmo_correction` rather than a repeat of the same forward warp."""
    ntr, nt = nmo_traces.shape
    t0 = np.arange(nt) * dt
    out = np.zeros_like(nmo_traces)
    for ix in range(ntr):
        tn = np.sqrt(t0 ** 2 + (offsets[ix] ** 2) / (vrms ** 2))
        valid = tn <= t0[-1]
        out[ix, valid] = np.interp(t0[valid], tn[valid], nmo_traces[ix, valid])
    return out


def dmo_correction(nmo_traces: np.ndarray, offsets: np.ndarray,
                    vrms: np.ndarray, dt: float) -> np.ndarray:
    """Simplified single-gather time-stretch DMO approximation.

    Documented limitation (carried over from the notebook): this is a
    time-stretch stand-in for true dip-moveout, not a full common-offset
    log-stretch/Kirchhoff DMO operator. It corrects for the leading-order
    time shift DMO introduces but does not perform a proper spatial
    (common-offset-domain) energy redistribution across CDPs. A full DMO
    operator is noted as future work (see README "Known limitations").

    Uses the same forward-style interpolation convention as
    `nmo_correction` (adds stretch on top of already-NMO'd data), which
    is self-consistent with its purpose and was not affected by the fix
    above — it never claimed to invert anything."""
    ntr, nt = nmo_traces.shape
    t = np.arange(nt) * dt
    out = np.zeros_like(nmo_traces)
    for ix in range(ntr):
        h = offsets[ix] / 2.0
        td = np.sqrt(t ** 2 + (h / vrms) ** 2)
        valid = (t > 0) & (td < t[-1])
        out[ix, valid] = np.interp(td[valid], t, nmo_traces[ix, :])
    return out
