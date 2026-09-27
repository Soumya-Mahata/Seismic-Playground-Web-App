"""
modules/filtering.py — Steps 6, 9, 18, 20 of the reference pipeline
======================================================================
Band-pass filter (frequency-domain trapezoid -> time-domain operator),
F-K dip filtering (ground-roll / coherent-noise attenuation), post-stack
random noise attenuation (spatial median filter), and the time-variant
filter (TVF).

Ported directly from `Seismic_Processing_CORRECTED.ipynb`
(`design_bandpass` — the notebook builds this same trapezoid design twice,
once inlined for step 6 and once as a named function for step 20's TVF
zones; consolidated here into one function used by both — `apply_time_domain_conv`,
`fk_filter_gather`). No change to the underlying math.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import windows

import config as cfg


# ─────────────────────────────────────────────────────────────────────────
# 6) BAND-PASS FILTER
# ─────────────────────────────────────────────────────────────────────────

def design_bandpass(fmin: float, fmax: float, dt: float, ns: int,
                     low_slope: float = 6.0, high_slope: float = 10.0,
                     smooth_hz: float = cfg.SMOOTH_WIDTH_HZ_DEFAULT) -> np.ndarray:
    """Design a zero-phase band-pass operator: a trapezoidal amplitude
    spectrum (linear tapers below `fmin` and above `fmax`), smoothed with
    a Hann kernel to avoid Gibbs ringing, then converted to a time-domain
    operator via the inverse FFT. Returned as a centered (`ifftshift`ed)
    real time-domain filter ready for `apply_time_domain_conv`."""
    fs = 1.0 / dt
    freqs = np.fft.fftfreq(ns, d=dt)
    af = np.abs(freqs)
    low_start, low_pass = max(0.0, fmin - low_slope), fmin
    high_pass, high_end = fmax, fmax + high_slope

    H = np.zeros(ns)
    rise = (af >= low_start) & (af < low_pass)
    H[rise] = (af[rise] - low_start) / max(low_pass - low_start, 1e-9)
    H[(af >= low_pass) & (af <= high_pass)] = 1.0
    fall = (af > high_pass) & (af <= high_end)
    H[fall] = 1.0 - (af[fall] - high_pass) / max(high_end - high_pass, 1e-9)

    df = fs / ns
    smooth_samples = max(1, int(round(smooth_hz / df)))
    kernel = windows.hann(2 * smooth_samples + 1)
    kernel /= kernel.sum()
    H = np.convolve(H, kernel, mode="same")
    return np.fft.ifftshift(np.real(np.fft.ifft(H)))


def apply_time_domain_conv(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve every trace in `data` (n_traces, n_samples) with the
    time-domain operator `h`, same-length output."""
    return np.array([np.convolve(tr, h, mode="same") for tr in data])


def amplitude_spectrum(data: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean positive-frequency amplitude spectrum across all traces in
    `data`, normalized to a 0-100 scale — used for the before/after
    spectrum QC plots and to help choose FMIN/FMAX."""
    ns = data.shape[1]
    freqs = np.fft.fftfreq(ns, d=dt)
    pos = freqs >= 0
    fft_d = np.fft.fft(data, axis=1)
    amp = np.abs(fft_d[:, pos]) * (2.0 / ns)
    mean_amp = amp.mean(axis=0)
    mean_amp = mean_amp / (mean_amp.max() + 1e-12) * 100.0
    return freqs[pos], mean_amp


def dominant_frequency(data: np.ndarray, dt: float) -> float:
    """Frequency of peak mean amplitude in the spectrum — a simple,
    standard bandwidth/QC metric."""
    f, amp = amplitude_spectrum(data, dt)
    return float(f[np.argmax(amp)])


# ─────────────────────────────────────────────────────────────────────────
# 9) F-K DIP FILTER (ground-roll / coherent linear-noise attenuation)
# ─────────────────────────────────────────────────────────────────────────

def fk_filter_gather(gather: np.ndarray, dt: float, dx: float,
                      polygon_vertices: List[Tuple[float, float]], taper_sigma_px: float = 1.5):
    """2-D FFT the gather into the F-K domain, REJECT everything *inside*
    the drawn polygon (a wedge in (k, f) space corresponding to the
    ground-roll's characteristic low apparent velocity — see the summer
    report Sec 5.2.2/6.6, "ground roll ... attenuated using f-k
    filtering"), and inverse-FFT back to the time-offset domain, passing
    everything outside the polygon untouched.

    Bug fixed here: the polygon passed into this function is documented
    everywhere (UI copy, config name FK_POLYGON_DEFAULT) as a *reject*
    polygon, but the mask previously applied used `contains_points(...)`
    directly as the multiplier — i.e. it KEPT only the energy *inside*
    the polygon (the ground-roll cone) and zeroed everything outside
    (every real reflection). That is backwards: a "reject" polygon must
    be inverted (kept = 1 - inside) before being used as a multiplicative
    mask. This was silently turning the F-K "filter" into a filter that
    kept noise and deleted signal.

    The binary polygon edge is also lightly smoothed (`taper_sigma_px`,
    in pixel/bin units of the F-K grid) with a Gaussian blur before being
    applied, rather than used as a hard 0/1 cut — a hard rectangular
    cutoff in the wavenumber domain rings (Gibbs phenomenon) back into
    the time-offset domain as spurious oscillatory artifacts; a soft
    taper at the polygon boundary is standard practice for F-K/dip
    filtering and keeps the QC spectra honest about what is passed vs.
    attenuated rather than a binary on/off.

    Returns (filtered, pass_mask, f_axis, k_axis, fk_before, fk_after) —
    `pass_mask` is the actual multiplier applied (1 = passed, 0 =
    rejected, with a smooth transition at the polygon edge); the
    intermediate F-K spectra are returned for QC display.
    """
    data = np.nan_to_num(np.asarray(gather, dtype=float), nan=0.0)
    nx, nt = data.shape
    f = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    fk = np.fft.fftshift(np.fft.fft2(data))
    K, F = np.meshgrid(k, f, indexing="ij")
    inside_reject_polygon = MplPath(polygon_vertices).contains_points(
        np.column_stack((K.ravel(), F.ravel()))).reshape(nx, nt).astype(float)
    pass_mask = 1.0 - inside_reject_polygon
    if taper_sigma_px and taper_sigma_px > 0:
        pass_mask = gaussian_filter(pass_mask, sigma=taper_sigma_px, mode="nearest")
    fk_filt = fk * pass_mask
    filtered = np.fft.ifft2(np.fft.ifftshift(fk_filt)).real
    return filtered, pass_mask, f, k, fk, fk_filt


# ─────────────────────────────────────────────────────────────────────────
# 18) POST-STACK RANDOM NOISE ATTENUATION (spatial median filter)
# ─────────────────────────────────────────────────────────────────────────

def random_noise_attenuation(stack_matrix: np.ndarray, spatial_window: int = 3) -> np.ndarray:
    """Running median across the CDP (spatial) direction. Coherent
    reflections correlate trace-to-trace; random noise doesn't, so a
    spatial median suppresses incoherent energy while preserving
    continuous, dipping events.

    Documented simplification (carried over from the notebook): this
    stands in for industry F-X deconvolution, which needs more regularly
    sampled, higher-fold data than a typical small teaching/demo line
    provides to avoid its own artifacts.
    """
    return median_filter(stack_matrix, size=(spatial_window, 1), mode="nearest")


# ─────────────────────────────────────────────────────────────────────────
# 20) TIME-VARIANT FILTER (TVF)
# ─────────────────────────────────────────────────────────────────────────

def time_variant_filter(data: np.ndarray, dt: float, zones: List[Tuple[float, float, float, float]],
                         taper_s: float = 0.15) -> np.ndarray:
    """Apply a different band-pass in each (t_start, t_end, fmin, fmax)
    zone and blend the results with a raised-cosine taper across zone
    boundaries — higher frequencies attenuate faster with depth/time, so
    a single fixed band-pass is a compromise; TVF widens the pass-band at
    shallow times and narrows it with depth.
    """
    n_traces, ns = data.shape
    out = np.zeros_like(data, dtype=float)
    for (t0, t1, fmin, fmax) in zones:
        h = design_bandpass(fmin, fmax, dt, ns)
        filtered = apply_time_domain_conv(data, h)
        it0, it1 = int(t0 / dt), min(ns, int(t1 / dt))
        taper = int(taper_s / dt)
        w = np.zeros(ns)
        w[it0:it1] = 1.0
        if taper > 0:
            lo_a, lo_b = max(0, it0 - taper // 2), it0 + taper // 2
            hi_a, hi_b = it1 - taper // 2, min(ns, it1 + taper // 2)
            if lo_b > lo_a:
                w[lo_a:lo_b] = np.maximum(w[lo_a:lo_b], np.linspace(0, 1, lo_b - lo_a))
            if hi_b > hi_a:
                w[hi_a:hi_b] = np.minimum(w[hi_a:hi_b], np.linspace(1, 0, hi_b - hi_a))
        out += filtered * w[np.newaxis, :]
    return out
