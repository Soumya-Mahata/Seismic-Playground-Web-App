"""
modules/filtering.py — Steps 6, 9, 18, 20 of the processing pipeline
======================================================================
Band-pass filter (frequency-domain trapezoid -> time-domain operator),
F-K dip filtering (ground-roll / coherent-noise attenuation), post-stack
random noise attenuation (spatial median filter), and the time-variant
filter (TVF).

`design_bandpass` builds the same trapezoid design for both the main
band-pass in step 6 and each of step 20's TVF zones, consolidated here
into one function used by both, alongside `apply_time_domain_conv` and
`fk_filter_gather`.
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

def apparent_velocity_wedge(v_cutoff: float, f_max: float, k_max: float) -> List[Tuple[float, float]]:
    """Builds the central f-k wedge bounded by the two lines |k| = f / V
    for a single apparent-velocity cutoff `v_cutoff`, from f=0 up to
    `f_max`, clipped to +/-`k_max`: a linear-moveout event of apparent
    velocity V traces the straight line k = f/V in the f-k domain, so
    "reject everything slower than V" and "accept everything faster than
    V" are both exactly this one wedge boundary — both conventions give
    the same result.

    Everything with |apparent velocity| > `v_cutoff` (primary reflections)
    falls inside this wedge; everything slower (ground roll, direct wave,
    low-velocity refractions) falls outside it. The wedge widens with
    frequency because the same velocity difference produces a larger
    wavenumber separation at higher frequency — this is why the polygon
    narrows to a point at f=0 and widens toward the Nyquist frequency,
    rather than a fixed-width band.

    Only the f>=0 half needs to be given here — `fk_filter_gather`
    automatically mirrors whatever region it is given through the origin
    to cover negative frequencies too, since a real signal's spectrum is
    symmetric there.
    """
    k_at_fmax = f_max / max(v_cutoff, 1e-6)
    if k_at_fmax <= k_max:
        return [(0.0, 0.0), (-k_at_fmax, f_max), (k_at_fmax, f_max)]
    f_at_kmax = v_cutoff * k_max
    return [(0.0, 0.0), (-k_max, f_at_kmax), (-k_max, f_max),
            (k_max, f_max), (k_max, f_at_kmax)]


def apparent_velocity_noise_wedges(v_cutoff: float, f_max: float, k_max: float
                                     ) -> List[List[Tuple[float, float]]]:
    """Builds the two noise-flank polygons (left and right of the signal
    wedge) that are the *exact geometric complement*, within the f>=0
    half of the [-k_max, k_max] x [0, f_max] rectangle, of
    `apparent_velocity_wedge`'s signal wedge for the same `v_cutoff`.

    Why this exists: an accept polygon encloses the signal amplitudes
    while a reject polygon surrounds the noise area, and both approaches
    are meant to normally produce the same results — but that
    equivalence only holds if the reject polygon actually traces the
    NOISE region's own boundary. Reusing the signal wedge's shape as a
    "reject" polygon would tell the masking logic in `fk_filter_gather`
    that the signal fan itself is the noise to be attenuated (and the
    ground roll outside it is the signal to be kept) — exactly backward.
    This function gives the reject path its own, correctly-shaped
    polygon so accept and reject produce the same (correct) filtered
    result instead of opposite ones.

    The complement of a single wedge inside a rectangle is two disjoint
    pieces (left and right of the wedge), not one simple polygon, so this
    returns a list of two rings rather than one vertex list.
    `fk_filter_gather` accepts a list of rings for exactly this case,
    testing membership in their union.
    """
    k_at_fmax = f_max / max(v_cutoff, 1e-6)
    if k_at_fmax <= k_max:
        left = [(-k_max, 0.0), (-k_max, f_max), (-k_at_fmax, f_max), (0.0, 0.0)]
        right = [(0.0, 0.0), (k_at_fmax, f_max), (k_max, f_max), (k_max, 0.0)]
    else:
        # The signal wedge already saturates to the full k-width at
        # f_at_kmax < f_max, so above that frequency there is no noise
        # flank left at all — each flank is just the triangle below it.
        f_at_kmax = v_cutoff * k_max
        left = [(-k_max, 0.0), (-k_max, f_at_kmax), (0.0, 0.0)]
        right = [(0.0, 0.0), (k_max, f_at_kmax), (k_max, 0.0)]
    return [left, right]


def default_fk_polygon(dt: float, dx: float, v_cutoff: float = cfg.FK_VELOCITY_CUTOFF_DEFAULT
                        ) -> List[Tuple[float, float]]:
    """Convenience wrapper: builds the apparent-velocity wedge spanning
    this gather's actual Nyquist frequency and Nyquist wavenumber, so
    callers don't need to compute the F-K axes themselves first."""
    return apparent_velocity_wedge(v_cutoff, f_max=0.5 / dt, k_max=0.5 / dx)


def fk_filter_gather(gather: np.ndarray, dt: float, dx: float,
                      polygon_vertices: List[Tuple[float, float]], *,
                      polygon_mode: str = cfg.FK_POLYGON_MODE_DEFAULT,
                      reject_coefficient: float = cfg.FK_REJECT_COEFFICIENT_DEFAULT,
                      taper_sigma_px: float = 1.5):
    """2-D FFT the gather into the F-K domain and apply an accept or
    reject polygon:

    - **accept** (`polygon_mode="accept"`, generally preferred in
      practice because it is easier to enclose the signal zone):
      `polygon_vertices` must enclose the SIGNAL; amplitudes OUTSIDE it
      are attenuated.
    - **reject** (`polygon_mode="reject"`): `polygon_vertices` must
      enclose the NOISE instead; amplitudes INSIDE it are attenuated.
      Both approaches normally produce the same results, although accept
      polygons are preferred in practice.

    **Critical**: accept and reject are not two ways of describing the
    same boundary — they require DIFFERENT polygons (the signal region
    and its geometric complement, the noise region) to produce an
    equivalent result. Passing the signal-enclosing wedge with
    `polygon_mode="reject"` does not "flip" a label — it tells this
    function the signal fan IS the noise, and attenuates it while
    passing the ground roll through untouched. Callers that offer both
    modes for the same underlying apparent-velocity cutoff must build
    the complementary polygon for reject (see
    `apparent_velocity_noise_wedges`), not reuse `apparent_velocity_wedge`
    with the mode flag flipped.

    Either way, the attenuated side is multiplied by `reject_coefficient`
    — a small nonzero constant, **not an exact zero**: multiplying by a
    constant in the frequency domain is equivalent to scaling by the
    inverse of that constant in the time domain, and the inverse of zero
    is simply infinite. The polygon boundary is also lightly
    Gaussian-tapered (`taper_sigma_px`, in pixel/bin units of the F-K
    grid) rather than used as a hard 0/1 cut, since a hard rectangular
    cutoff in wavenumber rings (Gibbs phenomenon) back into the
    time-offset domain as spurious oscillatory artifacts.

    Only the amplitude spectrum is touched — the complex F-K spectrum is
    scaled by a real-valued mask, so the phase spectrum is left
    unchanged.

    Returns (filtered, pass_mask, f_axis, k_axis, fk_before, fk_after) —
    `pass_mask` is the actual real-valued multiplier applied at every
    (k, f) bin, so it can be displayed directly as QC; the intermediate
    F-K spectra are returned for before/after spectrum QC.
    """
    if polygon_mode not in ("accept", "reject"):
        raise ValueError(f"polygon_mode must be 'accept' or 'reject', got {polygon_mode!r}")
    data = np.nan_to_num(np.asarray(gather, dtype=float), nan=0.0)
    nx, nt = data.shape
    f = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    fk = np.fft.fftshift(np.fft.fft2(data))
    K, F = np.meshgrid(k, f, indexing="ij")
    pts = np.column_stack((K.ravel(), F.ravel()))
    # `polygon_vertices` is normally one ring: a flat list of (x, y)
    # points. It can also be a list of several disjoint rings (e.g.
    # `apparent_velocity_noise_wedges`'s two flanking polygons) when the
    # region of interest isn't a single simple polygon — detected by
    # whether the first element is itself a point or a whole ring.
    is_multi_ring = (len(polygon_vertices) > 0
                      and isinstance(polygon_vertices[0], (list, tuple))
                      and len(polygon_vertices[0]) > 0
                      and isinstance(polygon_vertices[0][0], (list, tuple, np.ndarray)))
    if is_multi_ring:
        path = MplPath.make_compound_path(*[MplPath(ring) for ring in polygon_vertices])
    else:
        path = MplPath(polygon_vertices)
    # A real-valued signal's 2-D spectrum is Hermitian-symmetric:
    # X(-k,-f) = conj(X(k,f)). A polygon is normally only drawn/labeled in
    # the positive-frequency half of the spectrum — exactly as any
    # geophysicist would draw one — but this app's f-k grid (from
    # fftshift) spans BOTH positive and negative frequencies. A polygon
    # defined only for f>0 previously left every negative-frequency bin
    # outside it, which an "accept" mask then
    # floored to `reject_coefficient` — silently discarding half of a
    # real signal's necessary spectral content (verified: this cut a pure
    # flat reflector's amplitude by ~50%, not the "leave the signal
    # alone" an accept wedge is supposed to do). The fix: automatically
    # OR the requested region with its own point-reflection through the
    # origin, so a polygon drawn (or generated) only in positive-f space
    # is honored correctly on both halves without requiring the caller to
    # draw a mirrored polygon by hand.
    inside_direct = path.contains_points(pts).reshape(nx, nt)
    inside_mirrored = path.contains_points(-pts).reshape(nx, nt)
    inside = (inside_direct | inside_mirrored).astype(float)
    if taper_sigma_px and taper_sigma_px > 0:
        # Taper ONLY along the wavenumber axis (axis 0), never along
        # frequency (axis 1). An isotropic 2-D blur was used here
        # originally, but the wedge's own apex sits exactly at (k=0,
        # f=0) by construction (the boundary narrows to a point at
        # f=0) — smoothing across BOTH axes rounds off that
        # point, which disproportionately attenuates genuine low-
        # frequency, near-zero-dip reflection energy that legitimately
        # belongs at k=0 for f just above 0 (verified: a purely flat,
        # k=0 reflector lost ~45% of its energy to this, concentrated
        # exactly in the 0-5 Hz tail of its wavelet, even though a flat
        # reflector is in no sense noise). A real f-k filter's taper
        # exists to stop a hard wavenumber cutoff from ringing
        # (Gibbs phenomenon) in the along-trace (space) direction — it
        # was never meant to soften anything in the frequency direction,
        # so restricting it to axis 0 removes the artifact while still
        # doing its actual job.
        inside = gaussian_filter(inside, sigma=(taper_sigma_px, 0), mode="nearest")
    coeff = float(np.clip(reject_coefficient, 0.0, 1.0))
    if polygon_mode == "accept":
        # kept (~1) inside the polygon, floor of `coeff` outside it
        pass_mask = coeff + (1.0 - coeff) * inside
    else:
        # kept (~1) outside the polygon, floor of `coeff` inside it
        pass_mask = coeff + (1.0 - coeff) * (1.0 - inside)
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

    Documented simplification: this stands in for industry F-X
    deconvolution, which needs more regularly sampled, higher-fold data
    than a typical small teaching/demo line provides to avoid its own
    artifacts.
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
