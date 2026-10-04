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


def _inject_field_noise(raw_data: np.ndarray, receiver_station: np.ndarray, shot_id: np.ndarray,
                         dt: float, rng: np.random.Generator, *,
                         dead_stations: set, powerline_stations: set, powerline_freq: float,
                         n_spikes: int, spike_amplitude_mult: float) -> dict:
    """Injects the realistic field-noise conditions the report's Trace
    Edit / De-spike / Mute sections are actually written around (Sec 5.7,
    Figs 5.50-5.56) directly into `raw_data` in place, and returns the
    ground truth (which stations/traces were corrupted) for QC.

    Without conditions like these, a clean synthetic gather gives the
    Trace Edit / De-spike step nothing to actually fix — running it does
    nothing, which is indistinguishable from it being broken. Each
    condition here mirrors one the report documents explicitly:
    - **Dead channels** (Fig 5.51): "the channel numbers of dead traces
      do not change for a specific survey" — so these are fixed receiver
      stations, dead on every shot that records them, not random
      per-trace dropouts.
    - **Powerline harmonic noise** (Fig 5.50): a monofrequency (here 50
      Hz) sinusoid on a contiguous run of channels — the report notes
      trace kill is preferred over a notch filter here since a notch
      "produces amplitude notches in the spectrum which may ultimately
      cause issues during deconvolution."
    - **Spike-like noise bursts** (Fig 5.52): isolated, extremely high-
      amplitude single-sample transients scattered through the dataset.
    """
    twt = np.arange(raw_data.shape[1]) * dt
    info: dict = {"dead_stations": sorted(dead_stations), "powerline_stations": sorted(powerline_stations),
                  "powerline_freq_hz": powerline_freq, "spike_locations": []}

    if dead_stations:
        dead_mask = np.isin(receiver_station, list(dead_stations))
        # A real dead channel isn't analytically exact zero, just a tiny
        # instrument noise floor with no coherent signal in it at all.
        raw_data[dead_mask, :] = rng.normal(scale=1e-4, size=(int(dead_mask.sum()), raw_data.shape[1]))

    if powerline_stations:
        pl_mask = np.isin(receiver_station, list(powerline_stations))
        n_pl = int(pl_mask.sum())
        if n_pl:
            phase = rng.uniform(0, 2 * np.pi, size=n_pl)
            pl_signal = 2.5 * np.sin(2 * np.pi * powerline_freq * twt[None, :] + phase[:, None])
            # a touch of the 2nd harmonic too, matching "harmonic noise"
            pl_signal += 1.0 * np.sin(2 * np.pi * (2 * powerline_freq) * twt[None, :] + phase[:, None])
            raw_data[pl_mask, :] += pl_signal

    if n_spikes > 0:
        n_tr, n_s = raw_data.shape
        base_amp = float(np.std(raw_data)) + 1e-9
        tr_idx = rng.integers(0, n_tr, size=n_spikes)
        samp_idx = rng.integers(int(0.05 / dt), n_s, size=n_spikes)
        for ti, si in zip(tr_idx, samp_idx):
            burst = float(rng.choice([-1.0, 1.0]) * base_amp * rng.uniform(8.0, 15.0))
            raw_data[ti, si] += burst
            info["spike_locations"].append({"trace_index": int(ti), "sample_index": int(si),
                                              "shot_id": int(shot_id[ti]),
                                              "receiver_station": int(receiver_station[ti])})
    return info


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
    inject_field_noise: bool = False,
    n_dead_channels: int = 2,
    n_powerline_channels: int = 4,
    powerline_freq_hz: float = 50.0,
    n_spikes: int = 20,
) -> SeismicSurvey:
    """Generate a complete synthetic 2D land seismic survey in memory.

    `inject_field_noise=True` adds a small, fixed set of dead channels,
    a contiguous powerline-noise-affected channel range, and scattered
    high-amplitude spikes (report Sec 5.7) — see `_inject_field_noise` —
    so the Trace Edit / De-spike / Mute steps have real, documented
    problems to find and fix rather than a perfectly clean gather.
    Defaults to False (a clean survey) so velocity-analysis and other
    accuracy tests that call this function directly aren't perturbed by
    noise they aren't testing for; the app's own "Generate synthetic
    survey" button explicitly opts in.
    """
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

    field_noise_info: dict = {}
    if inject_field_noise:
        all_stations = np.unique(receiver_station)
        dead = set()
        if n_dead_channels > 0 and len(all_stations) > n_dead_channels:
            start = int(len(all_stations) * 0.33)
            dead = set(int(s) for s in all_stations[start:start + n_dead_channels])
        powerline = set()
        if n_powerline_channels > 0 and len(all_stations) > n_powerline_channels:
            start = int(len(all_stations) * 0.66)
            powerline = set(int(s) for s in all_stations[start:start + n_powerline_channels])
        field_noise_info = _inject_field_noise(
            raw_data, receiver_station, shot_id, dt, rng,
            dead_stations=dead, powerline_stations=powerline, powerline_freq=powerline_freq_hz,
            n_spikes=n_spikes, spike_amplitude_mult=1.0)

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
        field_noise_info=field_noise_info,
        notes=(
            f"Synthetic 2D line: {n_stations} stations @ {dx} m, "
            f"{len(np.unique(shot_id))} shots, {n_traces} traces. "
            f"Ground truth reflectors (t0 s / v m/s): "
            + ", ".join(f"{r.t0:.2f}/{r.velocity:.0f}" for r in reflectors)
            + f". Point diffractor at t0={diffractor['t0']:.2f}s, v={diffractor['velocity']:.0f} m/s."
            + (f" Injected field noise: dead stations {field_noise_info.get('dead_stations')}, "
               f"powerline-noise stations {field_noise_info.get('powerline_stations')}, "
               f"{len(field_noise_info.get('spike_locations', []))} spike bursts."
               if field_noise_info else "")
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
    reloaded.field_noise_info = survey.field_noise_info
    return reloaded
