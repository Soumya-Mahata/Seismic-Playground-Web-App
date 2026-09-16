"""
modules/attributes.py — Seismic Attributes
=============================================
Not present in the source processing notebooks (which cover processing
only, load through migration), but directly supported by the
interpretation reference PDF and by the attribute taxonomy/formulas in
the industrial-training reference report (Section 8.6, "Seismic
Attribute Analysis" — RMS amplitude, variance, sweetness, coherence,
instantaneous phase/frequency all follow that report's definitions
directly). All follow standard textbook/industry definitions (Taner,
Koehler & Sheriff, 1979, for the instantaneous-attribute family) — no
notebook algorithm to preserve here.

All attributes operate on a 2-D (n_cdp, n_samples) section and return an
array of the same shape (curvature is the exception — it's computed
along a 1-D picked horizon line, in `modules/interpretation.py`, since
it's a property of an interpreted surface rather than the seismic volume
itself).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert


def analytic_signal(section: np.ndarray) -> np.ndarray:
    """Complex analytic signal via the Hilbert transform, trace by trace."""
    return hilbert(section, axis=1)


def instantaneous_amplitude(section: np.ndarray) -> np.ndarray:
    """Envelope / reflection-strength: |analytic signal|. Highlights
    bright spots and gross impedance contrasts, independent of phase."""
    return np.abs(analytic_signal(section))


# Alias — "envelope" and "instantaneous amplitude" are the same quantity
# under two common names in the literature.
envelope = instantaneous_amplitude


def instantaneous_phase(section: np.ndarray) -> np.ndarray:
    """Instantaneous phase (radians, -pi..pi): emphasizes reflector
    continuity/geometry regardless of amplitude — useful for tracing
    weak-but-continuous reflectors and subtle stratigraphic terminations."""
    return np.angle(analytic_signal(section))


def instantaneous_frequency(section: np.ndarray, dt: float, smooth_samples: int = 5) -> np.ndarray:
    """Time derivative of the unwrapped instantaneous phase, converted to
    Hz. Smoothed slightly (`smooth_samples`) — the raw derivative of phase
    is very sensitive to amplitude notches (phase spins rapidly near a
    zero-crossing of the envelope) and is unusable without some averaging,
    a well-known practical caveat of this attribute."""
    phase = np.unwrap(instantaneous_phase(section), axis=1)
    freq = np.diff(phase, axis=1) / (2 * np.pi * dt)
    freq = np.pad(freq, ((0, 0), (0, 1)), mode="edge")
    if smooth_samples > 1:
        freq = uniform_filter1d(freq, size=smooth_samples, axis=1, mode="nearest")
    return freq


def rms_amplitude(section: np.ndarray, window_samples: int = 21) -> np.ndarray:
    """Sliding-window RMS amplitude — a smoother, less phase-sensitive
    energy indicator than instantaneous amplitude; commonly extracted in
    a window around a horizon for bright-spot / DHI mapping. One of the
    most widely used attributes in industry: localized high-amplitude
    zones frequently correspond to significant acoustic-impedance
    contrasts (channel sands, lithology changes, direct hydrocarbon
    indicators)."""
    return np.sqrt(uniform_filter1d(section ** 2, size=window_samples, axis=1, mode="nearest"))


def variance(section: np.ndarray, trace_window: int = 5, time_window: int = 5) -> np.ndarray:
    """Variance (discontinuity) attribute: local statistical variance of
    amplitude across a small ensemble of neighbouring traces, in a small
    time window, at every sample. Where the subsurface is laterally
    continuous, neighbouring traces are nearly identical and variance is
    low; where continuity breaks — a fault, a channel edge, a pinch-out —
    neighbouring traces diverge and variance rises. One of the primary
    attributes used for fault/fracture delineation. Conceptually the
    inverse counterpart of coherence/similarity below (same discontinuity,
    opposite scale)."""
    n_cdp, ns = section.shape
    mean_local = uniform_filter1d(section, size=trace_window, axis=0, mode="nearest")
    sq_mean_local = uniform_filter1d(section ** 2, size=trace_window, axis=0, mode="nearest")
    var_spatial = np.clip(sq_mean_local - mean_local ** 2, 0, None)
    var_smoothed = uniform_filter1d(var_spatial, size=time_window, axis=1, mode="nearest")
    v = var_smoothed - var_smoothed.min()
    return v / (v.max() + 1e-12)


def coherence_similarity(section: np.ndarray, trace_window: int = 3, time_window: int = 11) -> np.ndarray:
    """Coherence / similarity: trace-to-trace waveform similarity within
    a small analysis window, via a normalized cross-correlation-style
    ratio (a simple two-trace formulation, extended here to a symmetric
    multi-trace window — industry tools generally use a more elaborate
    eigenstructure formulation for greater robustness, noted as a future
    improvement). High coherence marks continuous, undisturbed
    reflectivity; low coherence marks faults/channel edges — the same
    discontinuities `variance` highlights, on an inverted scale."""
    n_cdp, ns = section.shape
    half = trace_window // 2
    num = np.zeros((n_cdp, ns))
    den = np.zeros((n_cdp, ns)) + 1e-12
    for d in range(-half, half + 1):
        if d == 0:
            continue
        shifted = np.roll(section, -d, axis=0)
        if d > 0:
            shifted[-d:, :] = 0
        else:
            shifted[:-d, :] = 0
        cross = uniform_filter1d(section * shifted, size=time_window, axis=1, mode="nearest")
        e1 = uniform_filter1d(section ** 2, size=time_window, axis=1, mode="nearest")
        e2 = uniform_filter1d(shifted ** 2, size=time_window, axis=1, mode="nearest")
        num += cross ** 2
        den += e1 * e2
    return np.clip(num / den, 0.0, 1.0)


def sweetness(section: np.ndarray, dt: float, freq_floor_hz: float = 3.0) -> np.ndarray:
    """Sweetness: a hybrid attribute combining instantaneous amplitude
    (envelope) and instantaneous frequency, S(t) = A(t) / sqrt(f(t)),
    designed to highlight "sweet spots" combining high reflectivity
    strength with low frequency content — frequently associated with
    clean, gas-charged sands (gas both raises the impedance contrast,
    increasing A(t), and preferentially attenuates high frequencies,
    lowering f(t)). `freq_floor_hz` avoids dividing by a near-zero or
    negative instantaneous frequency, which is a well-known practical
    caveat of this attribute at low-amplitude/noisy samples."""
    amp = instantaneous_amplitude(section)
    freq = np.clip(instantaneous_frequency(section, dt), freq_floor_hz, None)
    return amp / np.sqrt(freq)


def dominant_frequency_map(section: np.ndarray, dt: float, window_samples: int = 51) -> np.ndarray:
    """Locally-windowed spectral dominant frequency along each trace —
    a coarse frequency-attenuation / tuning indicator, computed with a
    sliding FFT window (stride = window_samples // 4 for efficiency,
    then interpolated back to full sample rate)."""
    n_cdp, ns = section.shape
    stride = max(1, window_samples // 4)
    centers = np.arange(window_samples // 2, ns - window_samples // 2, stride)
    freqs = np.fft.rfftfreq(window_samples, d=dt)
    out_centers = np.zeros((n_cdp, len(centers)))
    for j, c in enumerate(centers):
        seg = section[:, c - window_samples // 2: c + window_samples // 2]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[1])[None, :], axis=1))
        out_centers[:, j] = freqs[np.argmax(spec, axis=1)]
    x_full = np.arange(ns)
    out = np.zeros((n_cdp, ns))
    for i in range(n_cdp):
        out[i, :] = np.interp(x_full, centers, out_centers[i, :])
    return out


ATTRIBUTE_REGISTRY = {
    "RMS Amplitude": lambda sec, dt: rms_amplitude(sec),
    "Instantaneous Amplitude (Envelope)": lambda sec, dt: instantaneous_amplitude(sec),
    "Instantaneous Phase": lambda sec, dt: instantaneous_phase(sec),
    "Instantaneous Frequency": lambda sec, dt: instantaneous_frequency(sec, dt),
    "Sweetness": lambda sec, dt: sweetness(sec, dt),
    "Variance": lambda sec, dt: variance(sec),
    "Coherence / Similarity": lambda sec, dt: coherence_similarity(sec),
    "Dominant Frequency": lambda sec, dt: dominant_frequency_map(sec, dt),
}

ATTRIBUTE_CATEGORY = {
    "RMS Amplitude": "Amplitude",
    "Instantaneous Amplitude (Envelope)": "Complex-trace / Envelope",
    "Instantaneous Phase": "Phase",
    "Instantaneous Frequency": "Frequency",
    "Sweetness": "Hybrid (Amplitude + Frequency)",
    "Variance": "Geometric (discontinuity)",
    "Coherence / Similarity": "Geometric (discontinuity, inverse-scaled)",
    "Dominant Frequency": "Frequency",
}

ATTRIBUTE_USE = {
    "RMS Amplitude": "Reflectivity strength; lithology/DHI mapping; channel geometry",
    "Instantaneous Amplitude (Envelope)": "Bright-spot boundaries independent of polarity",
    "Instantaneous Phase": "Reflector continuity; subtle stratigraphic/depositional boundaries",
    "Instantaneous Frequency": "Thin-bed tuning; thickness variation; feeds into sweetness",
    "Sweetness": "\"Sweet spot\" / clean gas-sand identification; channel & lobe delineation",
    "Variance": "Fault and fracture delineation; stratigraphic edge detection",
    "Coherence / Similarity": "Fault and fracture delineation (inverse-scaled counterpart of variance)",
    "Dominant Frequency": "Bandwidth/attenuation QC; coarse tuning indicator",
}
