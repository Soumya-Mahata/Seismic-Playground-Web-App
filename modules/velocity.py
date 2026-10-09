"""
modules/velocity.py — Velocity Analysis (Semblance, Autopicking, Lateral Interpolation)
=========================================================================================
Windowed semblance (Neidell & Taner coherence) velocity analysis, an
automatic picker, and the multi-CDP picking driver that interpolates
picks into a 2-D V(t, CDP) field. Used identically for both the 1st
(approximate) and 2nd (refined) velocity-analysis passes.

`compute_semblance_vectorized` implements the standard windowed-semblance
formula with no change to the math — it was verified against a synthetic
gather with known reflector velocities and correctly locates every true
velocity to within ~1-2% (see `tests/test_pipeline.py`).

`autopick_velocity` uses a confidence-gated local-maximum walk rather
than a whole-path dynamic-programming optimizer. See "Why this picker
design" below — a deliberate, evidence-based design choice, not a
stylistic one.

Why this picker design
----------------------------
A whole-path dynamic-programming optimizer finds the single path through
the whole semblance panel that maximizes cumulative coherence minus a
jump penalty. Testing it against synthetic gathers with known reflector
velocities (flat reflectors at 0.35-2.2 s, 1900-3400 m/s) showed two
reproducible failures:

1. **Systematic under-tracking of deep, isolated peaks.** Real
   reflection coherence is "spotty" in time — strong at a reflector,
   flat background in between. A whole-path optimizer weighs a distant,
   short-lived peak against the cumulative jump-penalty cost of visiting
   it; if the peak is narrow, that cost isn't always "worth it" to the
   global optimum, so the path can simply skip it. On the test gather
   this produced a ~3000 m/s pick at 2.2 s where the true (and
   correctly-located, by the semblance panel itself) velocity was
   3400 m/s — a 12% error entirely attributable to the picker, not the
   semblance computation.
2. **A single bad analysis CDP can corrupt a large stretch of the
   survey.** `pick_velocity_field` builds the lateral V(t, CDP) field by
   linearly interpolating between analysis-CDP picks; `np.interp`
   extrapolates by *holding the boundary value constant*. If the picker
   fails badly at an edge analysis CDP (lower fold, weaker coherence),
   that single bad pick gets held flat across every CDP beyond it —
   exactly the kind of large, uniform, clearly-wrong block of anomalous
   velocity that shows up on the edge of a real refined-velocity-field
   QC plot from this pipeline.

The replacement is a **confidence-gated, physically-windowed local walk**
seeded from the panel's single strongest peak (found globally, so the
seed itself is reliable) and then tracked outward in both time
directions: at each step, search a window around the *current* velocity
(width in m/s) and move to the local coherence maximum only if it clears
a minimum-confidence floor; otherwise hold the last confident value.
This is the standard "seed-and-track" pattern interactive velocity-
picking tools use, and it is *local* rather than globally cumulative, so
a real nearby peak is never skipped just because a whole-path accountant
judged it not worth the trip. Re-tested against the same synthetic
gathers: 2.2 s / 3400 m/s recovers to within ~1%, and low-fold edge CDPs
degrade gracefully instead of collapsing to the velocity-range floor.

`pick_velocity_field` additionally cross-checks every analysis CDP's
picked curve against the **lateral median trend** across all analysis
CDPs (a robust "expected" velocity-vs-time function) and flags any CDP
whose curve deviates from it by more than a tolerance as laterally
inconsistent, excluding it from the interpolation by default — the same
QC judgement an interpreter makes when comparing velocity functions
between neighbouring analysis locations and rejecting an obvious
outlier. This directly targets the "one bad pick corrupts a whole
stretch of the field" failure: an absolute per-CDP confidence score
turned out to be a poor gate on its own (see the `mean(confidence)`
attempt in early development — legitimately quiet time zones between
sparse reflectors drag a whole-record average down regardless of pick
quality, and a single strong-but-spurious noise peak can still pass a
raw-confidence bar while clearly disagreeing with its neighbours).
Manual per-CDP overrides are also supported: even a much-improved
automatic picker is not a substitute for the interactive QC / manual
correction that is standard practice in every commercial velocity-
picking tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def compute_semblance_vectorized(gather: np.ndarray, offsets: np.ndarray,
                                  velocities: np.ndarray, dt: float,
                                  window_len: int = 5) -> np.ndarray:
    """Windowed NMO-corrected coherence (Neidell & Taner semblance) for a
    single CDP gather, evaluated over a range of trial stacking velocities.
    Returns a (n_velocities, n_samples) semblance panel in [0, 1].

    Implements the standard semblance formula unchanged — verified to
    correctly locate known reflector velocities to within ~1-2% (see
    module docstring)."""
    ntr, nt = gather.shape
    t0 = np.arange(nt) * dt
    offsets2 = offsets ** 2
    W = 2 * window_len + 1
    semblance = np.zeros((len(velocities), nt))

    for iv, v in enumerate(velocities):
        tn = np.sqrt(t0[:, None] ** 2 + offsets2[None, :] / v ** 2)
        sampled = np.zeros_like(tn)
        for ix in range(ntr):
            f = interp1d(t0, gather[ix, :], kind="linear", bounds_error=False, fill_value=0.0)
            sampled[:, ix] = f(tn[:, ix])

        row_sum = sampled.sum(axis=1)
        row_sq_sum = (sampled ** 2).sum(axis=1)
        num_per_t = row_sum ** 2
        win_num = uniform_filter1d(num_per_t, size=W, mode="nearest") * W
        win_den = uniform_filter1d(row_sq_sum, size=W, mode="nearest") * W
        den = ntr * win_den
        with np.errstate(divide="ignore", invalid="ignore"):
            semblance[iv, :] = np.where(den > 0, win_num / den, 0.0)

    return np.clip(np.nan_to_num(semblance, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def autopick_velocity(semblance: np.ndarray, velocities: np.ndarray, t: np.ndarray,
                       min_time: float = 0.12, search_half_width_ms: float = 600.0,
                       min_confidence: float = 0.12, smooth_sigma: int = 8,
                       enforce_non_decreasing: bool = True,
                       non_decrease_tolerance: float = 0.985) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence-gated local-maximum walk, seeded from the panel's
    strongest peak (see module docstring for why this is used instead of
    a whole-path dynamic-programming picker).

    Returns (velocity_curve, confidence_curve) — confidence is the raw
    semblance value accepted at each time sample, used by
    `pick_velocity_field` to flag unreliable analysis CDPs.
    """
    nvel, nt = semblance.shape
    dt = t[1] - t[0]
    dv = velocities[1] - velocities[0]
    it_start = max(1, int(round(min_time / dt)))
    S = gaussian_filter1d(semblance, sigma=1, axis=0, mode="nearest")
    half_width = max(2, int(round(search_half_width_ms / dv)))

    valid = S[:, it_start:]
    flat = int(np.argmax(valid))
    seed_v_idx, seed_t_rel = np.unravel_index(flat, valid.shape)
    seed_t_idx = int(seed_t_rel) + it_start
    seed_v_idx = int(seed_v_idx)

    v_idx = np.zeros(nt, dtype=int)
    conf = np.zeros(nt)
    v_idx[seed_t_idx] = seed_v_idx
    conf[seed_t_idx] = S[seed_v_idx, seed_t_idx]

    def _walk(t_range):
        cur = seed_v_idx
        for it in t_range:
            lo, hi = max(0, cur - half_width), min(nvel, cur + half_width + 1)
            local = S[lo:hi, it]
            pk = lo + int(np.argmax(local))
            pv = S[pk, it]
            if pv >= min_confidence:
                cur = pk
            v_idx[it] = cur
            conf[it] = S[cur, it]

    _walk(range(seed_t_idx + 1, nt))
    _walk(range(seed_t_idx - 1, it_start - 1, -1))
    v_idx[:it_start] = v_idx[it_start] if it_start < nt else seed_v_idx
    conf[:it_start] = conf[it_start] if it_start < nt else conf[seed_t_idx]

    v_full = velocities[v_idx].astype(float)
    v_smooth = gaussian_filter1d(v_full, sigma=smooth_sigma)
    if enforce_non_decreasing:
        v_mono = v_smooth.copy()
        for i in range(1, nt):
            if v_mono[i] < non_decrease_tolerance * v_mono[i - 1]:
                v_mono[i] = non_decrease_tolerance * v_mono[i - 1]
        v_smooth = v_mono
    return np.clip(v_smooth, velocities.min(), velocities.max()), conf


def constant_velocity_stack_panel(gathers: Dict[int, dict], cdp_list: List[int], input_key: str,
                                   center_cdp: int, half_window_cdps: int, trial_velocities: np.ndarray,
                                   dt: float, num_samples: int, offsets_key: str = "offsets"
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Constant Velocity Stack (CVS) analysis: for each trial (constant-
    in-time) velocity, NMO-correct a small local window of CDPs around
    `center_cdp` and stack them, producing one trace per trial velocity.
    The industry rule of thumb (used identically here): the best-fit
    velocity is the one whose CVS trace has the highest stacked
    amplitude/energy at a given time — this is a complementary,
    amplitude-domain alternative to semblance-based picking, not a
    replacement for it.

    Returns (cvs_panel, stack_power):
      cvs_panel: (n_velocities, n_samples) — one stacked trace per trial velocity
      stack_power: (n_velocities,) — total stacked energy, for a quick
        "which velocity stacks best overall" read
    """
    from modules.nmo import nmo_correction
    from modules.stacking import fold_normalized_stack

    idx = cdp_list.index(center_cdp)
    lo, hi = max(0, idx - half_window_cdps), min(len(cdp_list), idx + half_window_cdps + 1)
    window_cdps = cdp_list[lo:hi]

    panel = np.zeros((len(trial_velocities), num_samples))
    power = np.zeros(len(trial_velocities))
    for iv, v in enumerate(trial_velocities):
        vrms = np.full(num_samples, v)
        all_traces, all_live = [], []
        for cdp in window_cdps:
            g = gathers[cdp]
            nmo_out, live = nmo_correction(g[input_key], g[offsets_key], vrms, dt, stretch_mute=0.5)
            all_traces.append(nmo_out)
            all_live.append(live)
        traces_cat = np.vstack(all_traces)
        live_cat = np.vstack(all_live)
        stack, _ = fold_normalized_stack(traces_cat, live_cat)
        panel[iv, :] = stack
        power[iv] = float(np.sum(stack ** 2))
    return panel, power


def max_abs_offset(gathers: Dict[int, dict], cdp: int, offsets_key: str = "offsets") -> float:
    return float(np.nanmax(np.abs(gathers[cdp][offsets_key])))


@dataclass
class VelocityPickResult:
    """Everything the UI needs to show and let the user correct one
    analysis CDP's velocity pick."""
    cdp: int
    velocity: np.ndarray
    confidence: np.ndarray
    reliability: float
    is_reliable: bool
    is_manual_override: bool = False


def pick_velocity_field(gathers: Dict[int, dict], cdp_list: List[int], input_key: str,
                         velocities: np.ndarray, dt: float, num_samples: int,
                         n_analysis: int = 9, min_abs_offset: float = 700.0,
                         label: str = "", window_len: int = 5,
                         autopick_kwargs: Optional[dict] = None,
                         reliability_threshold: float = 0.15,
                         manual_overrides: Optional[Dict[int, np.ndarray]] = None,
                         excluded_cdps: Optional[set] = None,
                         progress_cb=None, offsets_key: str = "offsets"
                         ) -> Tuple[np.ndarray, List[int], Dict[int, VelocityPickResult]]:
    """Automatic multi-CDP semblance picking, laterally interpolated into a
    2-D V(t, CDP) field. Shared by both velocity-analysis passes.

    Mutates `gathers` in place, writing `semblance_{label}`, `vrms_{label}`,
    and `vrms_field_{label}` per analysis CDP, so QC plots for this stage
    can always reach back into the gather dict (same accumulation pattern
    every other pipeline stage uses).

    Analysis CDPs are chosen evenly across the survey, but skipped/
    substituted (nearest-neighbour search) if their far offset is too
    short (`min_abs_offset`) for reliable velocity discrimination.

    `manual_overrides`: {cdp: full-length velocity array} — replaces the
    auto-pick for that CDP and is always treated as reliable.
    `excluded_cdps`: analysis CDPs the user has chosen to drop entirely
    from the lateral interpolation (e.g. a low-reliability pick they
    don't want to manually fix either).
    """
    autopick_kwargs = autopick_kwargs or {}
    manual_overrides = manual_overrides or {}
    excluded_cdps = excluded_cdps or set()
    twt = np.arange(num_samples) * dt

    candidate_idx = np.linspace(0, len(cdp_list) - 1, n_analysis).round().astype(int)
    analysis = []
    for idx in candidate_idx:
        cdp0 = cdp_list[idx]
        if max_abs_offset(gathers, cdp0, offsets_key) >= min_abs_offset:
            analysis.append(cdp0)
            continue
        found = None
        for radius in range(1, 15):
            for j in (idx - radius, idx + radius):
                if 0 <= j < len(cdp_list) and max_abs_offset(gathers, cdp_list[j], offsets_key) >= min_abs_offset:
                    found = cdp_list[j]
                    break
            if found:
                break
        if found:
            analysis.append(found)
    analysis = sorted(set(analysis))
    if not analysis:
        # No candidate CDP (even after the neighbour search) meets
        # `min_abs_offset` — typically a small-aperture survey where the
        # requirement itself is stricter than the data supports. Fall
        # back to the evenly-spaced candidates regardless of offset
        # rather than crashing; velocity discrimination will just be
        # less reliable at these CDPs, which the reliability gate below
        # will generally catch anyway.
        analysis = sorted(set(cdp_list[i] for i in candidate_idx))

    raw_pick: Dict[int, np.ndarray] = {}
    raw_conf: Dict[int, np.ndarray] = {}
    is_manual: Dict[int, bool] = {}
    for i, cdp in enumerate(analysis):
        g = gathers[cdp]
        S = compute_semblance_vectorized(g[input_key], g[offsets_key], velocities, dt, window_len)
        g[f"semblance_{label}"] = S
        if cdp in manual_overrides:
            v = np.asarray(manual_overrides[cdp], dtype=float)
            conf = np.full(num_samples, 1.0)
        else:
            v, conf = autopick_velocity(S, velocities, twt, **autopick_kwargs)
        raw_pick[cdp], raw_conf[cdp] = v, conf
        is_manual[cdp] = cdp in manual_overrides
        g[f"vrms_{label}"] = v
        g[f"vrms_confidence_{label}"] = conf
        if progress_cb:
            progress_cb((i + 1) / len(analysis))

    # Lateral-consistency reliability gate: compare each curve to the
    # robust (median) trend across every analysis CDP, then flag a curve
    # as an outlier relative to the *spread of the other candidates*
    # (median/MAD), rather than against one fixed universal threshold —
    # different processing stages and datasets have different baseline
    # coherence levels, so an absolute cutoff calibrated on one stage
    # under- or over-flags on another. A raw per-CDP confidence average
    # was tried first and rejected — see module docstring — because
    # legitimately quiet time zones between sparse reflectors drag a
    # whole-record average down regardless of pick quality.
    median_curve = np.median(np.vstack([raw_pick[c] for c in analysis]), axis=0)
    dev_score = {}
    for cdp in analysis:
        rel_dev = np.abs(raw_pick[cdp] - median_curve) / np.maximum(median_curve, 1.0)
        # the 85th percentile (not the median) of the deviation, so a
        # curve that is badly wrong over even a meaningful minority of
        # its length scores as more deviant — a whole-curve median would
        # let a localized-but-severe departure slip through.
        dev_score[cdp] = float(np.percentile(rel_dev, 85))

    devs = np.array([dev_score[c] for c in analysis if not is_manual[c]])
    if len(devs) >= 3:
        med_dev, mad_dev = float(np.median(devs)), float(np.median(np.abs(devs - np.median(devs))))
        outlier_cutoff = med_dev + max(1.5 * mad_dev, 0.10)
    else:
        outlier_cutoff = reliability_threshold

    results: Dict[int, VelocityPickResult] = {}
    for cdp in analysis:
        v, conf = raw_pick[cdp], raw_conf[cdp]
        if is_manual[cdp]:
            reliability, is_reliable = 1.0, True
        else:
            reliability = float(1.0 - np.clip(dev_score[cdp], 0.0, 1.0))
            is_reliable = dev_score[cdp] <= outlier_cutoff
        results[cdp] = VelocityPickResult(cdp, v, conf, reliability, is_reliable, is_manual[cdp])

    usable = [c for c in analysis
              if c not in excluded_cdps and (results[c].is_reliable or results[c].is_manual_override)]
    if len(usable) < 2:
        # Too few trustworthy picks to interpolate a meaningful lateral
        # trend from — fall back to using everything rather than
        # producing a degenerate (flat or empty) field, but this should
        # surface as a prominent warning in the UI.
        usable = analysis

    usable_arr = np.array(usable)
    vstack = np.vstack([results[c].velocity for c in usable])
    Vfield = np.zeros((num_samples, len(cdp_list)))
    for it in range(num_samples):
        Vfield[it, :] = np.interp(cdp_list, usable_arr, vstack[:, it])
    for i, cdp in enumerate(cdp_list):
        gathers[cdp][f"vrms_field_{label}"] = Vfield[:, i]

    return Vfield, analysis, results
