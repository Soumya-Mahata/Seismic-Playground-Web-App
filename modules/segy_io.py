"""
modules/segy_io.py — SEG-Y I/O, Header Inspection, CDP Gather Construction
============================================================================
Pure Python / NumPy / segyio. No Streamlit dependency.

This is the "single source of truth for dt / ns / record length" step from
`Seismic_Processing_CORRECTED.ipynb` (cell 3), plus the CDP-gather builder
from cell 5 (`make_cdp_gathers`), reimplemented on top of `segyio` instead
of `obspy` + `segyio` together.

Library note (documented assumption)
-------------------------------------
The reference notebooks use `obspy.read()` to pull trace samples into a
NumPy array and `segyio` for named-field header access. `segyio` alone can
do both (trace data via `f.trace.raw[:]`, headers via `f.attributes(...)`)
and can also *write* SEG-Y, which `obspy`'s SEG-Y writer does not support
as robustly for arbitrary custom headers. Dropping the `obspy` dependency
keeps the app's dependency footprint smaller without changing any of the
notebook's actual processing logic — only how trace samples and header
words are pulled out of the file.

A SEG-Y file can be read in one of two geometry modes here:
  * `ignore_geometry=True` (default, matches the notebook) — every trace is
    read in file order; CDP/offset/shot come from trace headers. Works for
    any 2D line regardless of whether the file declares a regular 3D grid.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import segyio
from segyio.tracefield import TraceField as TF


# ─────────────────────────────────────────────────────────────────────────
# DATA CONTAINERS
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SeismicSurvey:
    """Everything read from one SEG-Y file — the raw, immutable input."""
    source_name: str
    num_traces: int
    sample_interval: float          # seconds
    num_samples: int
    record_length: float            # seconds
    twt: np.ndarray                 # (num_samples,) two-way time axis, seconds
    raw_data: np.ndarray            # (num_traces, num_samples)
    offsets: np.ndarray             # (num_traces,) meters, signed as stored
    cdp: np.ndarray                 # (num_traces,) int ensemble/CDP number
    shot_id: np.ndarray             # (num_traces,) int FieldRecord
    receiver_station: np.ndarray    # (num_traces,) int
    source_x: np.ndarray
    source_y: np.ndarray
    group_x: np.ndarray
    group_y: np.ndarray
    trace_seq: np.ndarray
    group_interval: float           # meters
    text_header_lines: list         # 40 lines, 80 chars each
    binary_header: dict
    is_synthetic: bool = False
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────
# FILE HANDLING
# ─────────────────────────────────────────────────────────────────────────

def materialize_upload(uploaded_file) -> str:
    """
    segyio needs a real file path. Streamlit's UploadedFile is an in-memory
    buffer, so write it to a temp file once and return the path. Caller is
    responsible for cleanup (or just let the OS temp dir reclaim it).
    """
    suffix = os.path.splitext(getattr(uploaded_file, "name", "upload.sgy"))[1] or ".sgy"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        tmp.write(uploaded_file.read())
    else:
        tmp.write(uploaded_file)
    tmp.flush()
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────
# 1) READ SEG-Y  (mirrors notebook cell 3: single source of truth for
#    dt / ns / record length, plus shot & receiver identifiers)
# ─────────────────────────────────────────────────────────────────────────

def read_segy(path: str, source_name: Optional[str] = None) -> SeismicSurvey:
    """Read a SEG-Y file and return a fully-populated SeismicSurvey."""
    source_name = source_name or os.path.basename(path)

    with segyio.open(path, "r", ignore_geometry=True) as f:
        f.mmap()

        num_traces = f.tracecount
        num_samples = len(f.samples)
        sample_interval = float(f.bin[segyio.BinField.Interval]) / 1_000_000.0
        if sample_interval <= 0:
            # fall back to the trace header if the binary header is blank
            sample_interval = float(f.header[0][TF.TRACE_SAMPLE_INTERVAL]) / 1_000_000.0
        record_length = sample_interval * num_samples
        twt = np.arange(num_samples) * sample_interval

        raw_data = f.trace.raw[:].astype(np.float64)
        if raw_data.shape != (num_traces, num_samples):
            raw_data = raw_data.reshape(num_traces, num_samples)

        offsets = f.attributes(TF.offset)[:].astype(float).copy()
        cdp = f.attributes(TF.CDP)[:].astype(int).copy()
        shot_id = f.attributes(TF.FieldRecord)[:].astype(int).copy()
        source_x = f.attributes(TF.SourceX)[:].astype(float).copy()
        source_y = f.attributes(TF.SourceY)[:].astype(float).copy()
        group_x = f.attributes(TF.GroupX)[:].astype(float).copy()
        group_y = f.attributes(TF.GroupY)[:].astype(float).copy()
        trace_seq = f.attributes(TF.TRACE_SEQUENCE_LINE)[:].astype(int).copy()

        text_bytes = f.text[0]
        binary_header = {str(k): int(v) for k, v in dict(f.bin).items()}

    # Group interval: notebook uses the difference between the first two
    # stored offsets. Robust fallback: median absolute step between unique
    # GroupX positions, used only if the primary estimate is degenerate
    # (zero or NaN — e.g. offset field left blank in the file).
    group_interval = float(offsets[1] - offsets[0]) if num_traces > 1 else 0.0
    if not np.isfinite(group_interval) or abs(group_interval) < 1e-9:
        gx_unique = np.unique(group_x[np.isfinite(group_x)])
        steps = np.diff(np.sort(gx_unique))
        group_interval = float(np.median(steps)) if len(steps) else 1.0

    receiver_station = np.round(group_x / max(abs(group_interval), 1e-6)).astype(int)

    text_header_lines = _decode_text_header(text_bytes)

    return SeismicSurvey(
        source_name=source_name,
        num_traces=num_traces,
        sample_interval=sample_interval,
        num_samples=num_samples,
        record_length=record_length,
        twt=twt,
        raw_data=raw_data,
        offsets=offsets,
        cdp=cdp,
        shot_id=shot_id,
        receiver_station=receiver_station,
        source_x=source_x,
        source_y=source_y,
        group_x=group_x,
        group_y=group_y,
        trace_seq=trace_seq,
        group_interval=group_interval,
        text_header_lines=text_header_lines,
        binary_header=binary_header,
    )


def _decode_text_header(text_bytes: bytes) -> list:
    """segyio already returns the textual header as ASCII bytes (it does
    the EBCDIC conversion internally when the file is EBCDIC-encoded).
    Split into the standard 40 x 80-character card image."""
    try:
        s = text_bytes.decode("ascii", errors="replace")
    except Exception:
        s = str(text_bytes)
    lines = [s[i * 80:(i + 1) * 80] for i in range(40)] if len(s) >= 3200 else [s]
    return lines


# ─────────────────────────────────────────────────────────────────────────
# TRACE HEADER SCAN (QC table) — mirrors Final_task.ipynb's scan_headers()
# ─────────────────────────────────────────────────────────────────────────

def scan_trace_headers(survey: SeismicSurvey, n: Optional[int] = None) -> pd.DataFrame:
    """Build a per-trace header QC table (first `n` traces, or all)."""
    n = survey.num_traces if n is None else min(n, survey.num_traces)
    return pd.DataFrame({
        "trace": np.arange(n),
        "cdp": survey.cdp[:n],
        "shot_id": survey.shot_id[:n],
        "offset_m": survey.offsets[:n],
        "source_x": survey.source_x[:n],
        "source_y": survey.source_y[:n],
        "group_x": survey.group_x[:n],
        "group_y": survey.group_y[:n],
        "receiver_station": survey.receiver_station[:n],
    })


def header_qc_summary(survey: SeismicSurvey) -> dict:
    """High-level dataset QC numbers for the Data Management dashboard."""
    n_dead = int(np.sum(np.all(survey.raw_data == 0, axis=1)))
    n_nan = int(np.sum(~np.isfinite(survey.raw_data)))
    amp = survey.raw_data
    finite = amp[np.isfinite(amp)]
    return {
        "n_traces": survey.num_traces,
        "n_samples": survey.num_samples,
        "sample_interval_ms": survey.sample_interval * 1000.0,
        "record_length_s": survey.record_length,
        "n_shots": int(len(np.unique(survey.shot_id))),
        "n_receiver_stations": int(len(np.unique(survey.receiver_station))),
        "n_cdp": int(len(np.unique(survey.cdp))),
        "group_interval_m": survey.group_interval,
        "offset_min_m": float(np.nanmin(survey.offsets)) if survey.num_traces else np.nan,
        "offset_max_m": float(np.nanmax(survey.offsets)) if survey.num_traces else np.nan,
        "dead_traces": n_dead,
        "non_finite_samples": n_nan,
        "amplitude_min": float(np.min(finite)) if finite.size else np.nan,
        "amplitude_max": float(np.max(finite)) if finite.size else np.nan,
        "amplitude_rms": float(np.sqrt(np.mean(finite ** 2))) if finite.size else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────
# 2) BUILD CDP GATHERS  (mirrors notebook cell 5: make_cdp_gathers)
# ─────────────────────────────────────────────────────────────────────────

def build_cdp_gathers(survey: SeismicSurvey, n_min_traces: int = 10, sort_by: str = "offset",
                       data_override: Optional[np.ndarray] = None) -> dict:
    """
    Group traces by CDP ensemble number into a dict:
        {cdp_number: {"traces": (fold, ns), "offsets": (fold,),
                      "shot_ids": (fold,), "receiver_ids": (fold,),
                      "trace_indices": (fold,)}}
    CDPs with fewer than `n_min_traces` traces are dropped (edge-of-survey
    low-fold gathers that are unreliable for velocity analysis / stacking).

    `data_override`: use this (num_traces, num_samples) array instead of
    `survey.raw_data` for the trace samples, while still using the
    survey's headers for grouping/offsets/IDs. This is how CDP sort/
    binning picks up data that has already been pre-stack-conditioned in
    the shot domain (see `build_shot_gathers` / `scatter_to_flat`) — the
    geometry headers never change, only the sample values do.
    """
    trace_data = survey.raw_data if data_override is None else data_override
    tmp = defaultdict(list)
    for i in range(survey.num_traces):
        tmp[int(survey.cdp[i])].append(i)

    gathers = {}
    for cdp_num, idx_list in tmp.items():
        idx = np.array(idx_list, dtype=int)
        offs = survey.offsets[idx]
        if sort_by == "offset":
            order = np.argsort(np.nan_to_num(offs, nan=1e9))
            idx = idx[order]
            offs = offs[order]
        gathers[cdp_num] = {
            "traces": trace_data[idx, :].copy(),
            "offsets": offs.astype(float),
            "shot_ids": survey.shot_id[idx].astype(int),
            "receiver_ids": survey.receiver_station[idx].astype(int),
            "trace_indices": idx,
        }

    gathers = {cdp: g for cdp, g in gathers.items() if g["traces"].shape[0] >= n_min_traces}
    return dict(sorted(gathers.items()))


def build_shot_gathers(survey: SeismicSurvey, n_min_traces: int = 1) -> dict:
    """
    Group traces by shot (FieldRecord) into a dict, mirroring
    `build_cdp_gathers` exactly but keyed by shot ID instead of CDP —
    this is the actual first grouping raw field data arrives in, and is
    where pre-stack conditioning (band-pass, AGC, trace edit/kill, mute,
    balance, F-K, multiple suppression, deconvolution) is properly
    applied in a standard workflow, *before* CDP sort/binning:

        {shot_id: {"traces": (n_ch, ns), "offsets": (n_ch,),
                   "receiver_ids": (n_ch,), "cdp": (n_ch,),
                   "trace_indices": (n_ch,)}}

    Sorted by receiver station within each shot for a sensible display
    order (near-to-far or vice versa depending on survey layout).
    """
    tmp = defaultdict(list)
    for i in range(survey.num_traces):
        tmp[int(survey.shot_id[i])].append(i)

    gathers = {}
    for shot_num, idx_list in tmp.items():
        idx = np.array(idx_list, dtype=int)
        order = np.argsort(survey.receiver_station[idx])
        idx = idx[order]
        gathers[shot_num] = {
            "traces": survey.raw_data[idx, :].copy(),
            "offsets": survey.offsets[idx].astype(float),
            "receiver_ids": survey.receiver_station[idx].astype(int),
            "cdp": survey.cdp[idx].astype(int),
            "trace_indices": idx,
        }
    gathers = {s: g for s, g in gathers.items() if g["traces"].shape[0] >= n_min_traces}
    return dict(sorted(gathers.items()))


def scatter_to_flat(gathers: dict, trace_key: str, num_traces: int, num_samples: int) -> np.ndarray:
    """Inverse of grouping into gathers: scatter a per-gather trace array
    (keyed by `trace_key`, e.g. after pre-stack conditioning) back into a
    flat (num_traces, num_samples) array indexed exactly like
    `survey.raw_data`, using each gather's stored `trace_indices`. This
    is what lets CDP sort/binning re-group *conditioned* shot-domain data
    by CDP via `build_cdp_gathers(..., data_override=...)`.
    """
    flat = np.zeros((num_traces, num_samples))
    for g in gathers.values():
        flat[g["trace_indices"], :] = g[trace_key]
    return flat


def demux_check(raw_data: np.ndarray, num_traces: int, num_samples: int) -> tuple:
    """Confirm (traces, samples) orientation; transpose if it looks swapped.
    Mirrors the demultiplexing sanity check in Final_task.ipynb."""
    if raw_data.shape == (num_traces, num_samples):
        return raw_data, "Data already demultiplexed (traces, samples)."
    if raw_data.shape == (num_samples, num_traces):
        return raw_data.T, "Data appeared multiplexed (samples, traces); transposed."
    raise ValueError(f"Unexpected data shape: {raw_data.shape}")


# ─────────────────────────────────────────────────────────────────────────
# EXPORT — write a processed 2D section (e.g. final stack/migration) as
# a new SEG-Y file so results can leave the app in a standard format.
# ─────────────────────────────────────────────────────────────────────────

def merge_surveys(survey_a: SeismicSurvey, survey_b: SeismicSurvey,
                   cdp_gap: int = 100000, shot_gap: int = 100000) -> SeismicSurvey:
    """
    Simplified geometry merging: concatenate two SEG-Y surveys' traces
    into one, renumbering the second survey's CDP and shot IDs upward by
    a fixed gap so they can never collide with the first survey's
    numbering.

    Documented simplification (see README): a full industry geometry
    merge reconciles source/receiver coordinates from SPS navigation
    files across multiple independently-acquired lines/vessels into one
    consistent coordinate system. This app has no SPS-style external
    navigation to merge against — the shipped/synthetic data's geometry
    is self-consistent already — so this performs the part of geometry
    merging that *is* well-defined without that external file: combining
    two already-geometry-complete surveys into one working dataset
    without ID collisions, e.g. two adjacent 2D lines or a repeat shoot.

    Requires both surveys to share the same sample interval and sample
    count (raises `ValueError` otherwise, rather than silently
    resampling one to match).
    """
    if not np.isclose(survey_a.sample_interval, survey_b.sample_interval):
        raise ValueError(
            f"Sample intervals differ ({survey_a.sample_interval * 1000:.2f} ms vs "
            f"{survey_b.sample_interval * 1000:.2f} ms) — merging surveys with different "
            f"sample rates isn't supported (would require resampling one of them first).")
    if survey_a.num_samples != survey_b.num_samples:
        raise ValueError(
            f"Sample counts differ ({survey_a.num_samples} vs {survey_b.num_samples}) — "
            f"trim or pad one survey to match before merging.")

    n_a = survey_a.num_traces
    return SeismicSurvey(
        source_name=f"{survey_a.source_name} + {survey_b.source_name} (merged)",
        num_traces=n_a + survey_b.num_traces,
        sample_interval=survey_a.sample_interval,
        num_samples=survey_a.num_samples,
        record_length=survey_a.record_length,
        twt=survey_a.twt,
        raw_data=np.vstack([survey_a.raw_data, survey_b.raw_data]),
        offsets=np.concatenate([survey_a.offsets, survey_b.offsets]),
        cdp=np.concatenate([survey_a.cdp, survey_b.cdp + cdp_gap]),
        shot_id=np.concatenate([survey_a.shot_id, survey_b.shot_id + shot_gap]),
        receiver_station=np.concatenate([survey_a.receiver_station, survey_b.receiver_station]),
        source_x=np.concatenate([survey_a.source_x, survey_b.source_x]),
        source_y=np.concatenate([survey_a.source_y, survey_b.source_y]),
        group_x=np.concatenate([survey_a.group_x, survey_b.group_x]),
        group_y=np.concatenate([survey_a.group_y, survey_b.group_y]),
        trace_seq=np.arange(1, n_a + survey_b.num_traces + 1),
        group_interval=survey_a.group_interval,
        text_header_lines=survey_a.text_header_lines,
        binary_header=survey_a.binary_header,
        is_synthetic=survey_a.is_synthetic or survey_b.is_synthetic,
        notes=f"Merged survey: '{survey_a.source_name}' ({n_a} traces) + "
              f"'{survey_b.source_name}' ({survey_b.num_traces} traces, CDP/shot IDs offset by "
              f"{cdp_gap}/{shot_gap} to avoid collisions).",
    )


def write_segy_traces(path: str, data: np.ndarray, dt: float, *, offsets: np.ndarray,
                       cdp: np.ndarray, shot_id: np.ndarray, source_x: np.ndarray,
                       source_y: np.ndarray, group_x: np.ndarray, group_y: np.ndarray,
                       text_header: str = "") -> None:
    """
    Write full pre-stack trace gathers (raw or partially processed) with a
    complete set of geometry headers. Used by the synthetic-data generator
    so synthetic surveys flow through exactly the same `read_segy()` /
    `build_cdp_gathers()` path as a real acquired SEG-Y file.
    """
    n_traces, n_samples = data.shape
    spec = segyio.spec()
    spec.samples = np.arange(n_samples) * (dt * 1000.0)
    spec.tracecount = n_traces
    spec.format = 5

    with segyio.create(path, spec) as f:
        card = (text_header or "SYNTHETIC SEISMIC TEST DATA - FOR SOFTWARE TESTING ONLY")
        f.text[0] = segyio.tools.wrap(card[:3200].ljust(3200))
        f.bin[segyio.BinField.Interval] = int(round(dt * 1_000_000))
        f.bin[segyio.BinField.Samples] = n_samples
        for i in range(n_traces):
            f.trace[i] = data[i, :].astype(np.float32)
            h = f.header[i]
            h[TF.TRACE_SEQUENCE_LINE] = i + 1
            h[TF.FieldRecord] = int(shot_id[i])
            h[TF.CDP] = int(cdp[i])
            h[TF.offset] = int(round(offsets[i]))
            h[TF.SourceX] = int(round(source_x[i]))
            h[TF.SourceY] = int(round(source_y[i]))
            h[TF.GroupX] = int(round(group_x[i]))
            h[TF.GroupY] = int(round(group_y[i]))
            h[TF.TRACE_SAMPLE_INTERVAL] = int(round(dt * 1_000_000))


def write_segy_section(path: str, data: np.ndarray, dt: float, cdp_numbers: np.ndarray,
                        group_interval: float = 25.0, text_header: str = "") -> None:
    """
    Write a post-stack section (n_traces, n_samples) — one trace per CDP —
    as a standard SEG-Y file. Offset is 0 (zero-offset / stacked section);
    CDP number is carried in the CDP trace header word.
    """
    n_traces, n_samples = data.shape
    spec = segyio.spec()
    spec.samples = np.arange(n_samples) * (dt * 1000.0)  # ms, for header consistency
    spec.tracecount = n_traces
    spec.format = 5  # IEEE float32

    with segyio.create(path, spec) as f:
        card = (text_header or "SEISMIC PROCESSING & INTERPRETATION PLATFORM - EXPORTED SECTION")
        f.text[0] = segyio.tools.wrap(card[:3200].ljust(3200))
        f.bin[segyio.BinField.Interval] = int(round(dt * 1_000_000))
        f.bin[segyio.BinField.Samples] = n_samples
        for i in range(n_traces):
            f.trace[i] = data[i, :].astype(np.float32)
            f.header[i][TF.CDP] = int(cdp_numbers[i])
            f.header[i][TF.TRACE_SEQUENCE_LINE] = i + 1
            f.header[i][TF.offset] = 0
            f.header[i][TF.GroupX] = int(cdp_numbers[i] * group_interval)
            f.header[i][TF.TRACE_SAMPLE_INTERVAL] = int(round(dt * 1_000_000))
