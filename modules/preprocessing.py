"""
modules/preprocessing.py — Amplitude Recovery, Deconvolution, Balancing, Mute
========================================================================
Amplitude recovery (AGC), spiking deconvolution before stack, trace
balancing (scalar normalization), and top mute.

Each function takes explicit array arguments (rather than closing over a
shared `gathers` dict) so each step is a pure, independently-testable,
cacheable function.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import correlate, lfilter


# ─────────────────────────────────────────────────────────────────────────
# 4) AMPLITUDE RECOVERY (AGC)
# ─────────────────────────────────────────────────────────────────────────

def agc(data: np.ndarray, window: int, mode: str = "rms") -> np.ndarray:
    """Automatic gain control along the time axis (axis=1).

    mode: "rms" (sliding RMS, self-normalizing — the preferred
    amplitude-recovery method, since raw t^2 spherical-
    divergence gain over-amplifies the late-time noise floor), "mean",
    or "median".
    """
    data = np.asarray(data, dtype=float)
    if mode == "rms":
        gain = np.sqrt(uniform_filter1d(data ** 2, size=window, axis=1, mode="reflect"))
    elif mode == "mean":
        gain = uniform_filter1d(np.abs(data), size=window, axis=1, mode="reflect")
    elif mode == "median":
        gain = median_filter(np.abs(data), size=(1, window), mode="reflect")
    else:
        raise ValueError("mode must be one of: rms, mean, median")
    gain = np.where(gain == 0, 1e-10, gain)
    return data / gain


def tp_gain(data: np.ndarray, twt: np.ndarray, power: float = 2.0) -> np.ndarray:
    """t^power spherical-divergence-correction gain (comparison only —
    this over-amplifies the late-time noise floor on a full 3 s record,
    which is why AGC is used for the actual amplitude-recovery step
    instead)."""
    curve = twt ** power
    return data * curve[np.newaxis, :]


def late_time_noise_ratio(raw: np.ndarray, gained: np.ndarray, dt: float, late_time_s: float = 2.0) -> float:
    """Quantifies how much a gain method inflates the noise floor after
    `late_time_s` — a useful diagnostic for catching over-amplification
    from a t^2-type gain (compare single-trace amplitude at each stage,
    not the final stack's appearance)."""
    idx = int(late_time_s / dt)
    num = np.abs(gained[:, idx:]).mean()
    den = np.abs(raw[:, idx:]).mean() + 1e-15
    return float(num / den)


# ─────────────────────────────────────────────────────────────────────────
# TRACE EDIT / KILL AND DE-SPIKING
# Standard early-QC noise-attenuation steps (de-spiking, then trace
# kill/edit) that run before ground-roll/multiple attenuation and
# deconvolution — see README for the industry noise-attenuation ordering
# this follows.
# ─────────────────────────────────────────────────────────────────────────

def despike_traces(data: np.ndarray, window: int = 15, threshold: float = 6.0,
                    isolation_ratio: float = 3.0) -> tuple:
    """De-spiking / amplitude-burst-noise removal: flags samples whose
    amplitude exceeds `threshold` times the local (sliding-median-based)
    robust noise level AND are genuinely isolated in time (their
    immediate neighbours' residuals average less than 1/`isolation_ratio`
    of their own) — replacing flagged samples with the local median.

    The isolation check matters: a naive outlier test alone
    misclassifies real, sharp reflection wavelets as "spikes" wherever
    the local window is otherwise quiet (their MAD-based robust-noise
    estimate collapses toward zero in a quiet window, so *any* genuine
    reflection peak there reads as an extreme outlier). A true burst/
    glitch spike is a 1-2 sample transient with small immediate
    neighbours; a real wavelet peak's immediate neighbours ramp up and
    down smoothly at comparable amplitude. Verified against clean
    synthetic data with no injected noise: the outlier-only test flagged
    ~2% of all samples (entirely false positives, visibly distorting
    real reflections); adding the isolation check drops that to <0.05%
    while still catching an injected single-sample spike (see
    `tests/test_pipeline.py::test_despike_removes_burst_without_disturbing_rest`
    and the module's own development notes).

    Returns (cleaned_data, spike_mask)."""
    local_med = median_filter(data, size=(1, window), mode="reflect")
    residual = data - local_med
    mad = median_filter(np.abs(residual), size=(1, window), mode="reflect")
    robust_std = 1.4826 * np.maximum(mad, 1e-12)
    is_outlier = np.abs(residual) > threshold * robust_std

    neighbor_avg = (np.abs(np.roll(residual, 1, axis=1)) + np.abs(np.roll(residual, -1, axis=1))) / 2.0
    is_isolated = np.abs(residual) > isolation_ratio * (neighbor_avg + 1e-9)

    spike_mask = is_outlier & is_isolated
    cleaned = np.where(spike_mask, local_med, data)
    return cleaned, spike_mask


def trace_energy(data: np.ndarray) -> np.ndarray:
    """RMS energy per trace — the basis for dead/noisy trace detection."""
    return np.sqrt(np.mean(data ** 2, axis=1))


def flag_bad_traces(data: np.ndarray, dead_frac: float = 0.02, noisy_mult: float = 8.0) -> dict:
    """Flags dead (near-zero energy) and abnormally noisy (energy far
    above the ensemble median) traces within a gather — the "trace edit
    / kill" step of a standard noise-attenuation sequence.

    IMPORTANT — call this on gain-recovery data that still preserves
    relative amplitude BETWEEN traces (e.g. the band-passed data,
    `traces_bp`), never on AGC output (`traces_agc`). AGC is a per-trace
    self-normalizing gain: it divides each trace by a sliding estimate
    of its OWN local energy, so after AGC every trace — dead, noisy, or
    perfectly healthy — ends up at very nearly the same RMS level by
    construction. Confirmed empirically: a channel that is 1000x weaker
    than its neighbors before AGC becomes indistinguishable from them
    afterward, so this function would silently flag nothing. AGC
    completely removes the relative amplitude information embedded in
    the seismic data — that applies just as much to a QC step that
    depends on relative amplitude as it does to AVO/inversion.

    Returns a dict of boolean masks and the underlying energy values so
    a QC view can show exactly why each trace was flagged, rather than
    silently dropping traces.
    """
    energy = trace_energy(data)
    median_energy = np.median(energy[energy > 0]) if np.any(energy > 0) else 0.0
    dead = energy < dead_frac * (median_energy + 1e-12)
    noisy = energy > noisy_mult * (median_energy + 1e-12)
    return {"energy": energy, "dead": dead, "noisy": noisy, "kill": dead | noisy,
            "median_energy": median_energy}


def apply_trace_kill(data: np.ndarray, kill_mask: np.ndarray) -> np.ndarray:
    """Zero out killed traces (rather than deleting them, which would
    disturb offset/CDP bookkeeping downstream)."""
    out = data.copy()
    out[kill_mask, :] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────
# 5/11) DECONVOLUTION — spiking (Wiener-Hopf) and predictive (gap), used
# both for pre-stack resolution enhancement and for multiple suppression
# ─────────────────────────────────────────────────────────────────────────

def predictive_decon_single_trace(trace: np.ndarray, operator_length: int = 30,
                                   prediction_lag: int = 1, pre_whitening: float = 0.01) -> np.ndarray:
    """General predictive-deconvolution operator (Peacock & Treitel /
    Wiener-Levinson prediction-error filter) with an arbitrary prediction
    lag (gap).

    `prediction_lag=1` is exactly spiking deconvolution (collapses the
    whole wavelet to a zero-lag spike). A `prediction_lag` set larger
    than one sample — a practical rule of thumb is the second zero-
    crossing of the autocorrelation — preserves the primary wavelet
    while predicting and subtracting the periodic reverberation tail
    caused by short-period multiples: this is the standard tool for
    multiple suppression on land data (the marine equivalent, SRME,
    needs a free-surface/multiple-generator model this app doesn't have
    access to — see README).
    """
    trace = trace.astype(float)
    if np.allclose(trace, 0):
        return trace
    scale = np.max(np.abs(trace)) + 1e-12
    trace_n = trace / scale
    filter_len = max(1, operator_length - 1)

    ac = correlate(trace_n, trace_n, mode="full")
    mid = len(ac) // 2
    need = prediction_lag + filter_len
    tail = ac[mid: mid + need]
    if len(tail) < need:
        tail = np.pad(tail, (0, need - len(tail)))

    r0 = tail[:filter_len].copy()
    r0[0] *= (1.0 + pre_whitening)
    R = toeplitz(r0)
    b = tail[prediction_lag: prediction_lag + filter_len]
    try:
        a = np.linalg.solve(R, b)
    except np.linalg.LinAlgError:
        return trace

    inv_filter = np.concatenate(([1.0], np.zeros(prediction_lag - 1), -a))
    return lfilter(inv_filter, 1.0, trace_n) * scale


def predictive_decon_gather(gather: np.ndarray, operator_length: int = 30,
                             prediction_lag: int = 1, pre_whitening: float = 0.01) -> np.ndarray:
    """Apply `predictive_decon_single_trace` to every trace in a
    (n_traces, n_samples) array."""
    out = np.zeros_like(gather, dtype=float)
    for i in range(gather.shape[0]):
        out[i] = predictive_decon_single_trace(gather[i], operator_length, prediction_lag, pre_whitening)
    return out


def second_zero_crossing_lag(trace: np.ndarray, dt: float, max_lag_s: float = 0.2) -> float:
    """Practical rule-of-thumb prediction-lag suggestion: the time of the
    second zero-crossing of a trace's autocorrelation, which separates
    the primary wavelet from its reverberation tail. Returned in
    seconds; convert to samples for `prediction_lag`."""
    trace = trace.astype(float)
    if np.allclose(trace, 0):
        return max_lag_s / 2
    ac = correlate(trace, trace, mode="full")
    mid = len(ac) // 2
    max_lag = int(max_lag_s / dt)
    r = ac[mid: mid + max_lag]
    signs = np.sign(r)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) >= 2:
        return float(crossings[1] * dt)
    return max_lag_s / 4


def spiking_decon_single_trace(trace: np.ndarray, operator_length: int = 30,
                                pre_whitening: float = 0.01) -> np.ndarray:
    """Spiking deconvolution — predictive deconvolution with a
    prediction lag of exactly one sample (see
    `predictive_decon_single_trace`). Kept as a named, explicit special
    case of the general predictive-deconvolution operator."""
    return predictive_decon_single_trace(trace, operator_length, prediction_lag=1, pre_whitening=pre_whitening)


def spiking_decon_gather(gather: np.ndarray, operator_length: int = 30,
                          pre_whitening: float = 0.01) -> np.ndarray:
    """Apply `spiking_decon_single_trace` to every trace in a (n_traces, n_samples) array."""
    out = np.zeros_like(gather, dtype=float)
    for i in range(gather.shape[0]):
        out[i] = spiking_decon_single_trace(gather[i], operator_length, pre_whitening)
    return out


# ─────────────────────────────────────────────────────────────────────────
# 7) TRACE BALANCING — scalar-per-trace RMS normalization
# ─────────────────────────────────────────────────────────────────────────

def trace_normalize(data: np.ndarray, ref_window: slice | None = None) -> np.ndarray:
    """Divide each trace by a single RMS scalar (optionally computed over
    a reference time window), preserving relative amplitude within the
    trace. Distinct from the sliding-window AGC used for amplitude
    recovery — this is a separate "Balance" step."""
    data = np.asarray(data, dtype=float)
    seg = data if ref_window is None else data[:, ref_window]
    scalar = np.sqrt(np.mean(seg ** 2, axis=1, keepdims=True))
    scalar = np.where(scalar == 0, 1e-10, scalar)
    return data / scalar


# ─────────────────────────────────────────────────────────────────────────
# 8) TOP / BOTTOM MUTE
# ─────────────────────────────────────────────────────────────────────────

def apply_top_bottom_mute(data: np.ndarray, dt: float, t_top: float = 0.05,
                           t_bottom: float | None = None, taper_s: float = 0.05) -> np.ndarray:
    """Cosine-tapered top (and optional bottom) mute — suppresses the
    earliest, physically meaningless samples (direct arrival / pre-first-
    break noise) with a smooth taper rather than a hard cut."""
    ns = data.shape[-1]
    time = np.arange(ns) * dt
    mask = np.ones_like(time)
    if t_top and t_top > 0:
        it = int(t_top / dt)
        it_tap = int(taper_s / dt)
        mask[:max(0, it - it_tap)] = 0.0
        if it_tap > 0:
            mask[max(0, it - it_tap):it] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, it_tap)))
    if t_bottom is not None:
        ib = min(ns - 1, int(t_bottom / dt))
        ib_tap = int(taper_s / dt)
        mask[ib + ib_tap:] = 0.0
        if ib_tap > 0:
            mask[ib:ib + ib_tap] = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, ib_tap)))
    return data * mask


def apply_linear_mute(data: np.ndarray, offsets: np.ndarray, dt: float, velocity: float,
                       t_intercept: float = 0.0, taper_s: float = 0.05) -> tuple:
    """Offset-dependent (linear-moveout) mute: zeroes each trace before
    `t_mute(x) = t_intercept + |offset| / velocity`, with a cosine taper,
    rather than the same fixed time on every trace.

    This is the correct mute geometry for removing a direct arrival or
    ground-roll cone, whose arrival time increases roughly linearly with
    offset. A single global top-mute time (`apply_top_bottom_mute`)
    cannot follow that moveout at all — it either mutes too little at
    far offsets (the cone is still there) or too much at near offsets
    (killing real shallow reflections). A flat cutoff regardless of
    offset cannot follow the diagonal noise trend visible in a full shot
    gather, which is exactly why an offset-dependent mute is needed.

    Returns (muted_data, mute_times) where `mute_times` is one cutoff
    time per trace, for QC overlay on the section display.
    """
    n_tr, ns = data.shape
    time = np.arange(ns) * dt
    mute_times = np.maximum(0.0, t_intercept + np.abs(offsets) / max(velocity, 1e-6))
    it_tap = max(1, int(round(taper_s / dt)))
    out = np.empty_like(data, dtype=float)
    for i in range(n_tr):
        it = int(round(mute_times[i] / dt))
        mask = np.ones(ns)
        lo = min(max(0, it - it_tap), ns)
        hi = min(max(it, lo), ns)
        mask[:lo] = 0.0
        if hi > lo:
            mask[lo:hi] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, hi - lo)))
        out[i, :] = data[i, :] * mask
    return out, mute_times


def apply_surgical_mute(data: np.ndarray, offsets: np.ndarray, dt: float, *,
                         t_min: float, t_max: float, x_min: float, x_max: float,
                         taper_s: float = 0.02) -> np.ndarray:
    """Surgical mute: zeroes a specific rectangular (offset, time)
    region of noise *inside* the gather — unlike a top mute, this does
    not need to start from t=0, and unlike trace kill it does not need
    to remove a whole trace. This is useful for noise such as
    distinctive high-frequency harmonic noise confined to a triangular
    zone along the near-offset channels, which is not easy to remove
    using conventional filtering, and is better handled with a surgical
    mute instead.

    Traces with |offset| in [x_min, x_max] are zeroed for time in
    [t_min, t_max], with a cosine taper of `taper_s` seconds on both the
    time and offset edges of the box so the cut doesn't ring.
    """
    n_tr, ns = data.shape
    time = np.arange(ns) * dt
    it_tap = max(1, int(round(taper_s / dt)))
    it0, it1 = int(round(t_min / dt)), int(round(t_max / dt))
    out = data.copy()
    in_offset = (np.abs(offsets) >= x_min) & (np.abs(offsets) <= x_max)
    time_mask = np.ones(ns)
    lo, hi = max(0, it0 - it_tap), min(ns, it1 + it_tap)
    time_mask[it0:it1] = 0.0
    if it0 - lo > 0:
        time_mask[lo:it0] = np.cos(np.linspace(0, np.pi / 2, it0 - lo)) ** 2
    if hi - it1 > 0:
        time_mask[it1:hi] = np.cos(np.linspace(np.pi / 2, 0, hi - it1)) ** 2
    out[in_offset, :] *= time_mask[None, :]
    return out, in_offset


def mute_mask(ns: int, dt: float, t_top: float = 0.05, t_bottom: float | None = None,
              taper_s: float = 0.05) -> np.ndarray:
    """Return the 1-D mute mask itself (used to re-apply the same mute to
    a section after migration)."""
    return apply_top_bottom_mute(np.ones((1, ns)), dt, t_top, t_bottom, taper_s)[0]
