"""
modules/synthetic_data.py — Synthetic Seismic Survey Generator
=================================================================
Builds a small, fully synthetic 2D land seismic survey (shots, receivers,
CDP geometry, dipping-free flat reflectors, ground roll, a point diffractor,
and ambient noise), written out as a genuine SEG-Y file.

This exists ONLY because no real SEG-Y file was supplied with the source
notebooks — every processing algorithm the app runs on this data is the
same real computation it would run on a real survey (nothing about the
*processing* is faked). The data itself is clearly labeled everywhere in
the UI as synthetic test data, never presented as a real acquired survey.

Ground truth (for validating the pipeline end-to-end):
  * 5 flat primary reflectors with increasing velocity with depth
    (a simple compaction trend) — used to sanity-check velocity analysis
    and NMO flattening.
  * Low-velocity, low-frequency ground roll — a legitimate target for the
    F-K dip filter.
  * One point diffractor — a legitimate target for Kirchhoff migration
    (a real diffraction hyperbola that should collapse to a point).
  * Mild time-dependent amplitude decay — a legitimate target for the
    amplitude-recovery (AGC) step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import fftconvolve

from modules.segy_io import SeismicSurvey, write_segy_traces, read_segy

RNG_SEED_DEFAULT = 42


@dataclass
class Reflector:
    t0: float          # zero-offset two-way time, s
    velocity: float     # NMO/stacking velocity, m/s
    amplitude: float    # relative reflection strength
    polarity: int = 1   # +1 or -1


DEFAULT_REFLECTORS: List[Reflector] = [
    Reflector(t0=0.35, velocity=1900, amplitude=1.00, polarity=1),
    Reflector(t0=0.65, velocity=2200, amplitude=0.85, polarity=-1),
    Reflector(t0=1.05, velocity=2600, amplitude=0.70, polarity=1),
    Reflector(t0=1.55, velocity=3000, amplitude=0.55, polarity=-1),
    Reflector(t0=2.20, velocity=3400, amplitude=0.40, polarity=1),
]

DEFAULT_DIFFRACTOR = dict(t0=1.20, velocity=2500.0, amplitude=0.5)


def _ricker(f0_hz: float, dt: float, length_s: float = 0.2) -> np.ndarray:
    """Zero-phase Ricker (Mexican-hat) wavelet, unit peak amplitude."""
    n_half = int(round(length_s / dt / 2))
    t = np.arange(-n_half, n_half + 1) * dt
    a = (np.pi * f0_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w / np.max(np.abs(w))


def _survey_geometry(n_stations: int, dx: float, shot_interval_stations: int,
                      channels_per_side: int, near_offset_stations: int = 1):
    """Build the shot/receiver/offset/CDP geometry table for a standard
    split-spread 2D land line."""
    station_x = (np.arange(n_stations) + 1) * dx  # 1-indexed station positions
    shot_stations = np.arange(1, n_stations + 1, shot_interval_stations)

    xs_list, xr_list, off_list, shot_list, rstat_list = [], [], [], [], []
    for s in shot_stations:
        xs = station_x[s - 1]
        lo = max(1, s - channels_per_side)
        hi = min(n_stations, s + channels_per_side)
        for r in range(lo, hi + 1):
            if abs(r - s) < near_offset_stations:
                continue
            xr = station_x[r - 1]
            xs_list.append(xs)
            xr_list.append(xr)
            off_list.append(xr - xs)
            shot_list.append(int(s))
            rstat_list.append(int(r))

    xs_arr = np.array(xs_list, dtype=float)
    xr_arr = np.array(xr_list, dtype=float)
    offsets = np.array(off_list, dtype=float)
    shot_id = np.array(shot_list, dtype=int)
    receiver_station = np.array(rstat_list, dtype=int)

    midpoint = (xs_arr + xr_arr) / 2.0
    cdp_bin = dx / 2.0
    cdp_number = np.round(midpoint / cdp_bin).astype(int) + 1

    return xs_arr, xr_arr, offsets, shot_id, receiver_station, cdp_number


def generate_synthetic_survey(
    n_stations: int = 160,
    dx: float = 25.0,
    shot_interval_stations: int = 2,
    channels_per_side: int = 60,
    dt: float = 0.002,
    record_length: float = 3.0,
    reflectors: Optional[List[Reflector]] = None,
    diffractor: Optional[dict] = None,
    primary_wavelet_freq: float = 32.0,
    ground_roll_velocity: float = 450.0,
    ground_roll_freq: float = 8.0,
    ground_roll_amplitude: float = 3.0,
    attenuation_beta: float = 1.2,
    noise_std: float = 0.03,
    seed: int = RNG_SEED_DEFAULT,
    source_name: str = "synthetic_survey.sgy",
) -> SeismicSurvey:
    """Generate a complete synthetic 2D land seismic survey in memory."""
    reflectors = reflectors if reflectors is not None else DEFAULT_REFLECTORS
    diffractor = diffractor if diffractor is not None else DEFAULT_DIFFRACTOR
    rng = np.random.default_rng(seed)

    xs, xr, offsets, shot_id, receiver_station, cdp_number = _survey_geometry(
        n_stations, dx, shot_interval_stations, channels_per_side)
    n_traces = len(offsets)
    n_samples = int(round(record_length / dt)) + 1
    twt = np.arange(n_samples) * dt

    reflectivity = np.zeros((n_traces, n_samples))

    def deposit(sample_time: np.ndarray, amplitude: np.ndarray):
        valid = np.isfinite(sample_time) & (sample_time >= 0) & (sample_time < twt[-1])
        idx_f = sample_time[valid] / dt
        idx0 = np.floor(idx_f).astype(int)
        frac = idx_f - idx0
        rows = np.nonzero(valid)[0]
        np.add.at(reflectivity, (rows, idx0), amplitude[valid] * (1 - frac))
        np.add.at(reflectivity, (rows, np.minimum(idx0 + 1, n_samples - 1)), amplitude[valid] * frac)

    # Primary flat reflectors: hyperbolic moveout t(x) = sqrt(t0^2 + x^2/v^2)
    for refl in reflectors:
        t_x = np.sqrt(refl.t0 ** 2 + (offsets ** 2) / (refl.velocity ** 2))
        amp = np.full(n_traces, refl.amplitude * refl.polarity)
        deposit(t_x, amp)

    # Point diffractor: two-way time = (dist_source + dist_receiver) / v
    z0 = diffractor["velocity"] * diffractor["t0"] / 2.0
    x0 = (n_stations * dx) / 2.0
    dist_s = np.sqrt((xs - x0) ** 2 + z0 ** 2)
    dist_r = np.sqrt((xr - x0) ** 2 + z0 ** 2)
    t_diff = (dist_s + dist_r) / diffractor["velocity"]
    deposit(t_diff, np.full(n_traces, diffractor["amplitude"]))

    # Convolve reflectivity with the primary wavelet -> primary signal
    wavelet_primary = _ricker(primary_wavelet_freq, dt)
    primary = fftconvolve(reflectivity, wavelet_primary[None, :], mode="same", axes=1)

    # Mild time-dependent amplitude decay (spherical divergence + intrinsic
    # attenuation stand-in) -- gives the AGC / amplitude-recovery step
    # something real to correct for.
    decay = 1.0 / (1.0 + attenuation_beta * twt)
    primary *= decay[None, :]

    # Ground roll: low-velocity, low-frequency, offset-decaying surface wave
    t_gr = np.abs(offsets) / ground_roll_velocity
    amp_gr = ground_roll_amplitude / np.sqrt(1.0 + np.abs(offsets) / 60.0)
    gr_spikes = np.zeros((n_traces, n_samples))
    valid = t_gr < twt[-1]
    idx_f = t_gr[valid] / dt
    idx0 = np.floor(idx_f).astype(int)
    frac = idx_f - idx0
    rows = np.nonzero(valid)[0]
    np.add.at(gr_spikes, (rows, idx0), amp_gr[valid] * (1 - frac))
    np.add.at(gr_spikes, (rows, np.minimum(idx0 + 1, n_samples - 1)), amp_gr[valid] * frac)
    wavelet_gr = _ricker(ground_roll_freq, dt, length_s=0.4)
    ground_roll = fftconvolve(gr_spikes, wavelet_gr[None, :], mode="same", axes=1)

    noise = rng.normal(scale=noise_std, size=(n_traces, n_samples))

    raw_data = primary + ground_roll + noise

    group_interval = float(dx)
    survey = SeismicSurvey(
        source_name=source_name,
        num_traces=n_traces,
        sample_interval=dt,
        num_samples=n_samples,
        record_length=record_length,
        twt=twt,
        raw_data=raw_data,
        offsets=offsets,
        cdp=cdp_number,
        shot_id=shot_id,
        receiver_station=receiver_station,
        source_x=xs,
        source_y=np.zeros(n_traces),
        group_x=xr,
        group_y=np.zeros(n_traces),
        trace_seq=np.arange(1, n_traces + 1),
        group_interval=group_interval,
        text_header_lines=["SYNTHETIC SEISMIC TEST DATA - FOR SOFTWARE TESTING ONLY"],
        binary_header={},
        is_synthetic=True,
        notes=(
            f"Synthetic 2D line: {n_stations} stations @ {dx} m, "
            f"{len(np.unique(shot_id))} shots, {n_traces} traces. "
            f"Ground truth reflectors (t0 s / v m/s): "
            + ", ".join(f"{r.t0:.2f}/{r.velocity:.0f}" for r in reflectors)
            + f". Point diffractor at t0={diffractor['t0']:.2f}s, v={diffractor['velocity']:.0f} m/s."
        ),
    )
    return survey


def save_synthetic_segy(path: str, **kwargs) -> SeismicSurvey:
    """Generate a synthetic survey and write it to `path` as real SEG-Y,
    then re-read it through the same path a real file would take, so the
    shipped sample data exercises the exact production read path."""
    survey = generate_synthetic_survey(**kwargs)
    write_segy_traces(
        path, survey.raw_data, survey.sample_interval,
        offsets=survey.offsets, cdp=survey.cdp, shot_id=survey.shot_id,
        source_x=survey.source_x, source_y=survey.source_y,
        group_x=survey.group_x, group_y=survey.group_y,
        text_header="SYNTHETIC SEISMIC TEST DATA - FOR SOFTWARE TESTING ONLY",
    )
    reloaded = read_segy(path, source_name=kwargs.get("source_name", "synthetic_survey.sgy"))
    reloaded.is_synthetic = True
    reloaded.notes = survey.notes
    return reloaded
