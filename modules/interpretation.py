"""
modules/interpretation.py — Horizon/Fault Picking & Well Tie
================================================================
This goes beyond pure processing (which ends at migration) into
interpretation — horizon/fault correlation, synthetic seismograms, and
wavelet extraction. Implemented as honest, standard, clearly-scoped
features operating on the app's own processed section — nothing here is
faked or pre-filled.

Horizon and fault picking is *manual* (the person clicks/enters points on
the section); there is no automatic horizon tracker, because a
fabricated auto-tracker would risk presenting invented picks as real
interpretation.

The well-tie / synthetic-seismogram module is a genuine, standard
implementation (reflectivity -> wavelet convolution -> correlation) but
is clearly scoped as approximate: it needs a sonic (or velocity) and
density log plus, optionally, a checkshot survey for depth-to-time
conversion; without a checkshot it falls back to direct log integration,
which is the standard simplification when no checkshot is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# HORIZON PICKING
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Horizon:
    name: str
    color: str = "#facc15"
    picks: Dict[int, float] = field(default_factory=dict)   # {cdp: time_s}


@dataclass
class FaultLine:
    name: str
    color: str = "#ef4444"
    points: List[Tuple[int, float]] = field(default_factory=list)  # [(cdp, time_s), ...]


def interpolate_horizon(picks: Dict[int, float], cdp_list: List[int]) -> np.ndarray:
    """Linearly interpolate a sparse set of (cdp, time) picks across the
    full CDP range; NaN outside the picked range (no extrapolation — a
    horizon should not be implied where nobody actually picked it)."""
    out = np.full(len(cdp_list), np.nan)
    if len(picks) < 1:
        return out
    pcdp = np.array(sorted(picks.keys()))
    ptime = np.array([picks[c] for c in pcdp])
    cdp_arr = np.array(cdp_list)
    if len(pcdp) == 1:
        mask = cdp_arr == pcdp[0]
        out[mask] = ptime[0]
        return out
    interp = np.interp(cdp_arr, pcdp, ptime, left=np.nan, right=np.nan)
    in_range = (cdp_arr >= pcdp.min()) & (cdp_arr <= pcdp.max())
    out[in_range] = interp[in_range]
    return out


def extract_attribute_along_horizon(attribute_section: np.ndarray, cdp_list: List[int],
                                     horizon_time: np.ndarray, dt: float,
                                     half_window_s: float = 0.01, agg: str = "mean") -> np.ndarray:
    """Extract an attribute value at (or in a small window around) a
    horizon, per CDP — the standard "attribute-along-horizon" map."""
    n_cdp, ns = attribute_section.shape
    out = np.full(n_cdp, np.nan)
    half_w = max(0, int(round(half_window_s / dt)))
    for i in range(n_cdp):
        t = horizon_time[i]
        if not np.isfinite(t):
            continue
        it = int(round(t / dt))
        lo, hi = max(0, it - half_w), min(ns, it + half_w + 1)
        if hi <= lo:
            continue
        seg = attribute_section[i, lo:hi]
        out[i] = {"mean": np.mean, "max": np.max, "min": np.min, "rms": lambda s: np.sqrt(np.mean(s ** 2))}[agg](seg)
    return out


def horizon_to_dataframe(horizon: Horizon, cdp_list: List[int], group_interval: float) -> pd.DataFrame:
    times = interpolate_horizon(horizon.picks, cdp_list)
    return pd.DataFrame({
        "cdp": cdp_list,
        "distance_m": np.array(cdp_list) * group_interval,
        "picked": [c in horizon.picks for c in cdp_list],
        "time_s": times,
    })


def fault_to_dataframe(fault: FaultLine) -> pd.DataFrame:
    pts = sorted(fault.points)
    return pd.DataFrame({"cdp": [p[0] for p in pts], "time_s": [p[1] for p in pts]})


# ─────────────────────────────────────────────────────────────────────────
# WELL TIE / SYNTHETIC SEISMOGRAM (approximate, standard method)
# ─────────────────────────────────────────────────────────────────────────

def horizon_curvature(horizon_time: np.ndarray, cdp_list: List[int], group_interval: float,
                       smooth_samples: int = 5) -> np.ndarray:
    """Curvature of a picked horizon line: kappa(x) = y''(x) / (1+y'(x)^2)^1.5,
    the standard 2-D analogue of the surface-curvature attributes used in
    3D interpretation (there, computed from a gridded time/depth surface's
    second derivatives in two lateral directions; here, from the single
    lateral direction a 2D line has). Useful for spotting subtle folds,
    flexures, and fracture-prone zones from a picked horizon alone,
    without needing a full 3D volume. NaN where the horizon has no pick
    (see `interpolate_horizon`).
    """
    x = np.asarray(cdp_list, dtype=float) * group_interval
    y = horizon_time.copy()
    valid = np.isfinite(y)
    if valid.sum() < 5:
        return np.full_like(y, np.nan)
    y_filled = np.interp(x, x[valid], y[valid])
    if smooth_samples > 1:
        from scipy.ndimage import uniform_filter1d
        y_filled = uniform_filter1d(y_filled, size=smooth_samples, mode="nearest")
    dy = np.gradient(y_filled, x)
    d2y = np.gradient(dy, x)
    kappa = d2y / (1 + dy ** 2) ** 1.5
    kappa[~valid] = np.nan
    return kappa


def _ricker(freq_hz: float, dt: float, length_s: float = 0.15) -> np.ndarray:
    n_half = int(round(length_s / dt / 2))
    t = np.arange(-n_half, n_half + 1) * dt
    a = (np.pi * freq_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w / np.max(np.abs(w))


def depth_to_time(depth_m: np.ndarray, vp_m_s: np.ndarray,
                   checkshot: Optional[pd.DataFrame] = None) -> np.ndarray:
    """One-way traveltime at each depth sample, by direct integration of
    slowness (dt = dz / v), then doubled for two-way time; if a checkshot
    survey (columns 'depth_m', 'twt_s') is supplied, it's used to
    calibrate (linearly correct) the integrated curve instead of trusting
    the sonic log's absolute drift alone — standard practice, since sonic
    logs are well known to accumulate a *systematic* time error with
    depth relative to a checkshot."""
    depth_m = np.asarray(depth_m, dtype=float)
    vp_m_s = np.asarray(vp_m_s, dtype=float)
    dz = np.diff(depth_m, prepend=depth_m[0])
    dz[0] = 0.0
    owt = np.cumsum(dz / np.maximum(vp_m_s, 1.0))
    twt = 2.0 * owt

    if checkshot is not None and len(checkshot) >= 2:
        cs = checkshot.sort_values("depth_m")
        twt_calib = np.interp(depth_m, cs["depth_m"], cs["twt_s"])
        # blend: honor the checkshot at its control points, keep the
        # sonic-derived *shape* (high-frequency detail) between them by
        # correcting with a smooth drift curve rather than overwriting outright
        drift = twt_calib - np.interp(depth_m, cs["depth_m"],
                                       np.interp(cs["depth_m"], depth_m, twt))
        twt = twt + drift
    return twt


def extract_statistical_wavelet(traces: np.ndarray, dt: float, length_s: float = 0.15,
                                 t_window: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Statistical (zero-phase) wavelet extraction directly from seismic
    traces, for use in the well-to-seismic tie: extracted statistically
    from the seismic data in a window around the well.

    Method: the same white-reflectivity assumption underlying spiking
    deconvolution (the autocorrelation of the seismogram approximates
    the autocorrelation of the wavelet) means
    the ensemble-averaged POWER spectrum of a group of traces around the
    well approximates the wavelet's power spectrum. Taking the square
    root gives the wavelet's amplitude spectrum; assuming zero phase
    (the standard default for a statistically extracted wavelet, since
    the reflectivity assumption alone gives no phase information) and
    inverse-transforming gives a symmetric, centered zero-phase wavelet,
    which is then windowed to `length_s` and Hann-tapered to suppress
    truncation artifacts.

    `t_window`, if given, restricts the trace samples used to a
    (t_min_s, t_max_s) window — typically bracketing the well/zone of
    interest, since the wavelet's effective shape can change with time
    (a single "the" wavelet is itself an approximation).
    """
    traces = np.nan_to_num(np.asarray(traces, dtype=float), nan=0.0)
    if traces.ndim == 1:
        traces = traces[None, :]
    if t_window is not None:
        i0 = max(0, int(round(t_window[0] / dt)))
        i1 = min(traces.shape[1], int(round(t_window[1] / dt)))
        if i1 - i0 >= 8:
            traces = traces[:, i0:i1]
    ns = traces.shape[1]
    win = np.hanning(ns) if ns > 4 else np.ones(ns)
    tapered = traces * win[None, :]
    power = np.abs(np.fft.fft(tapered, axis=1)) ** 2
    mean_power = power.mean(axis=0)
    amp_spectrum = np.sqrt(mean_power)
    wavelet_full = np.real(np.fft.ifftshift(np.fft.ifft(amp_spectrum)))

    n_half = max(4, int(round(length_s / dt / 2)))
    center = len(wavelet_full) // 2
    lo, hi = max(0, center - n_half), min(len(wavelet_full), center + n_half + 1)
    wav = wavelet_full[lo:hi]
    wav = wav * np.hanning(len(wav))
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak
    return wav


def well_tie_correlation(synthetic: np.ndarray, seismic: np.ndarray) -> float:
    """Zero-lag correlation coefficient between the (normalized) synthetic
    and real seismic trace over their common length — values above
    ~0.7-0.8 are generally considered a good tie, though the threshold
    is data- and zone-dependent."""
    n = min(len(synthetic), len(seismic))
    s, x = synthetic[:n], seismic[:n]
    denom = np.sqrt(np.sum(s ** 2) * np.sum(x ** 2))
    return float(np.sum(s * x) / denom) if denom > 0 else 0.0


def read_las_curves(file_obj) -> dict:
    """Reads an LAS well-log file via `lasio` and returns
    {curve_mnemonic: numpy array}, plus the depth index under key
    "__DEPTH__". Raises ImportError with a clear message if `lasio`
    isn't installed (it's an optional dependency, only needed for LAS
    support — CSV well logs work without it)."""
    try:
        import lasio
    except ImportError as e:
        raise ImportError(
            "Reading .las files needs the 'lasio' package (`pip install lasio`), "
            "which isn't installed in this environment. CSV well logs work without it."
        ) from e
    las = lasio.read(file_obj)
    curves = {mnemonic: np.asarray(las[mnemonic], dtype=float) for mnemonic in las.curves.keys()}
    curves["__DEPTH__"] = np.asarray(las.index, dtype=float)
    return curves


def sonic_to_velocity(dt_log: np.ndarray, unit: str) -> np.ndarray:
    """Converts a sonic (interval transit time) log to P-wave velocity.
    `unit` is one of "us_per_ft" or "us_per_m" (microseconds per unit
    length). Vp = 1 / delta-t, expressed per the log's native unit
    before converting to m/s."""
    dt_log = np.asarray(dt_log, dtype=float)
    safe = np.where(np.abs(dt_log) < 1e-6, np.nan, dt_log)
    if unit == "us_per_ft":
        vp_ft_s = 1e6 / safe
        return vp_ft_s * 0.3048
    elif unit == "us_per_m":
        return 1e6 / safe
    raise ValueError(f"Unknown sonic unit: {unit}")



def synthetic_seismogram(depth_m: np.ndarray, vp_m_s: np.ndarray, rho_kg_m3: np.ndarray,
                          dt: float, wavelet_freq_hz: float = 30.0,
                          checkshot: Optional[pd.DataFrame] = None,
                          t_max_s: float = 3.0, wavelet: Optional[np.ndarray] = None) -> dict:
    """Standard well-tie synthetic: acoustic impedance -> normal-incidence
    reflectivity -> resample to a uniform time axis -> convolve with a
    wavelet. Pass `wavelet` (e.g. from `extract_statistical_wavelet`) to
    use a wavelet extracted from the real seismic data instead of the
    default Ricker wavelet.

    Returns a dict with the time axis, reflectivity-in-time, synthetic
    trace, and the depth->time curve used, for QC display.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    vp_m_s = np.asarray(vp_m_s, dtype=float)
    rho_kg_m3 = np.asarray(rho_kg_m3, dtype=float)
    impedance = vp_m_s * rho_kg_m3
    rc = np.zeros(len(depth_m))
    rc[1:] = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1] + 1e-9)

    twt = depth_to_time(depth_m, vp_m_s, checkshot)
    time_axis = np.arange(0, t_max_s, dt)
    rc_time = np.zeros_like(time_axis)
    # place each depth-domain reflection coefficient at its two-way time
    t_idx = np.clip(np.round(twt / dt).astype(int), 0, len(time_axis) - 1)
    np.add.at(rc_time, t_idx, rc)

    wavelet_used = wavelet if wavelet is not None else _ricker(wavelet_freq_hz, dt)
    synthetic = np.convolve(rc_time, wavelet_used, mode="same")

    return {
        "time_axis": time_axis,
        "reflectivity_time": rc_time,
        "synthetic_trace": synthetic,
        "depth_time_curve": twt,
        "wavelet": wavelet_used,
    }
