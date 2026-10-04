"""
visualization/seismic_plots.py — Core Seismic Display
=========================================================
Section (image) displays, gather displays, wiggle/variable-area traces,
before/after QC comparisons, and fold maps. Plotly throughout for
interactivity (zoom, pan, hover amplitude, colorbar) — matching the
reference app's visualization approach — with matplotlib reserved for
nothing here (wiggle/VA is implemented as filled Plotly traces, which
stays interactive and is legible for the moderate trace counts a
"trace viewer" window realistically shows).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg
from visualization import theme


def _polygon_rings(polygon_points) -> list:
    """Normalizes an F-K polygon overlay into a list of rings (each a
    list of (x, y) tuples) for plotting — `polygon_points` may be a
    single flat ring (the usual case, e.g. the accept wedge) or a list
    of several disjoint rings (e.g. `apparent_velocity_noise_wedges`'s
    two reject-mode flanking polygons), mirroring what
    `modules.filtering.fk_filter_gather` itself accepts."""
    if not polygon_points:
        return []
    first = polygon_points[0]
    is_multi_ring = (isinstance(first, (list, tuple))
                      and len(first) > 0
                      and isinstance(first[0], (list, tuple)))
    return [list(r) for r in polygon_points] if is_multi_ring else [list(polygon_points)]


def plot_section(section: np.ndarray, x_values: np.ndarray, twt: np.ndarray, *,
                  title: str = "", x_title: str = "CDP", pclip: float = cfg.PCLIP_DEFAULT,
                  colorscale: str = None, height: int = 550,
                  gain: float = 1.0, equalize: bool = False) -> go.Figure:
    """Standard seismic-section image display: time down, CDP/trace
    across, symmetric percentile amplitude clip, diverging colormap.

    `colorscale` is resolved through `theme.resolve_colorscale` so a bare
    name like "Seismic" (or "seismic") always becomes a real Plotly-
    understood colorscale — passing that string straight through is what
    used to crash with `ValueError: Invalid value ... colorscale`.
    `equalize` applies a per-trace *display-only* amplitude normalization
    (see `theme.equalize_traces`) before clipping, which is usually what
    makes a variable-density image of early/raw data actually look like a
    seismic section instead of a faint, mostly-blank field.
    """
    colorscale = theme.resolve_colorscale(colorscale or theme.SEISMIC_COLORSCALE)
    data = np.nan_to_num(section, nan=0.0) * gain
    if equalize:
        data = theme.equalize_traces(data, pclip=95.0)
    clip = theme.percentile_clip(data, pclip)
    fig = go.Figure(go.Heatmap(
        z=data.T, x=x_values, y=twt, zmin=-clip, zmax=clip, colorscale=colorscale, zmid=0.0,
        zsmooth="best", colorbar=dict(title="Amplitude", **theme.COLORBAR),
    ))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis(x_title))
    fig.update_yaxes(**theme.time_axis())
    return fig


def plot_gather(traces: np.ndarray, offsets: np.ndarray, twt: np.ndarray, *,
                 title: str = "", pclip: float = cfg.PCLIP_DEFAULT, colorscale: str = None,
                 height: int = 550, equalize: bool = False) -> go.Figure:
    """A single CDP gather, offset on x."""
    order = np.argsort(offsets)
    return plot_section(traces[order], offsets[order], twt, title=title,
                         x_title="Offset (m)", pclip=pclip, colorscale=colorscale, height=height,
                         equalize=equalize)


def plot_before_after(before: np.ndarray, after: np.ndarray, x_values: np.ndarray, twt: np.ndarray, *,
                       title_before: str = "Before", title_after: str = "After",
                       x_title: str = "CDP", pclip: float = cfg.PCLIP_DEFAULT, height: int = 550,
                       colorscale: str = None, equalize: bool = False,
                       x_values_after: np.ndarray = None) -> go.Figure:
    """Side-by-side before/after section comparison sharing one clip
    level (computed from the "after" section) and one time axis.
    `x_values_after` lets the two panels use different x-axes (e.g. a
    regularized offset grid with a different trace count than the
    original) — defaults to `x_values` when the two panels share the
    same trace positions."""
    x_values_after = x_values if x_values_after is None else x_values_after
    colorscale = theme.resolve_colorscale(colorscale or theme.SEISMIC_COLORSCALE)
    before_d = np.nan_to_num(before, nan=0.0)
    after_d = np.nan_to_num(after, nan=0.0)
    if equalize:
        before_d = theme.equalize_traces(before_d, pclip=95.0)
        after_d = theme.equalize_traces(after_d, pclip=95.0)
    clip = theme.percentile_clip(after_d, pclip) or theme.percentile_clip(before_d, pclip)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                         subplot_titles=(f"<b>{title_before}</b>", f"<b>{title_after}</b>"),
                         horizontal_spacing=0.06)
    fig.add_trace(go.Heatmap(z=before_d.T, x=x_values, y=twt,
                              zmin=-clip, zmax=clip, colorscale=colorscale, zmid=0.0, zsmooth="best",
                              showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=after_d.T, x=x_values_after, y=twt,
                              zmin=-clip, zmax=clip, colorscale=colorscale, zmid=0.0, zsmooth="best",
                              colorbar=dict(title="Amplitude", **theme.COLORBAR)), row=1, col=2)
    fig.update_layout(**theme.base_layout(title="", height=height))
    for c in (1, 2):
        fig.update_xaxes(**theme.x_axis(x_title), row=1, col=c)
    fig.update_yaxes(**theme.time_axis(), row=1, col=1)
    fig.update_yaxes(**theme.time_axis(), row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=theme.TITLE_BLUE)
    return fig


def plot_mute_qc(traces: np.ndarray, x_positions: np.ndarray, twt: np.ndarray,
                  mute_times: np.ndarray, *, title: str = "Mute QC", x_title: str = "Receiver station",
                  pclip: float = cfg.PCLIP_DEFAULT, colorscale: str = None, height: int = 550,
                  equalize: bool = True) -> go.Figure:
    """Section display with the mute cutoff time (one value per trace)
    drawn as an explicit line, so it is unambiguous exactly which part of
    the data a mute removed — the QC the mute step was missing entirely
    (a before/after pair that looks identical gives no way to see *where*
    the cut actually is)."""
    fig = plot_section(traces, x_positions, twt, title=title, x_title=x_title, pclip=pclip,
                        colorscale=colorscale, height=height, equalize=equalize)
    order = np.argsort(x_positions)
    fig.add_trace(go.Scatter(x=np.asarray(x_positions)[order], y=np.asarray(mute_times)[order],
                              mode="lines", line=dict(color="#111111", width=2.5, dash="dash"),
                              name="Mute cutoff"))
    fig.update_layout(showlegend=True)
    return fig


def plot_variable_density_with_wiggle(traces: np.ndarray, x_positions: np.ndarray, twt: np.ndarray, *,
                                       title: str = "", max_traces: int = 60, pclip: float = cfg.PCLIP_DEFAULT,
                                       colorscale: str = None, height: int = 600,
                                       equalize: bool = True) -> go.Figure:
    """Variable-density (color image) background with wiggle traces
    overlaid — "VD+WT" display, a standard industry combination that
    shows both the continuous amplitude field and individual trace
    shape at once. Same `max_traces` safety cap as `plot_wiggle_va`.

    `equalize` defaults to True here (unlike `plot_section`): a raw or
    early-stage gather commonly spans a huge dynamic range across offset
    (near-trace amplitude can be 10-100x a far trace before AGC), and a
    single global clip level leaves most of a VD image looking blank —
    exactly the "doesn't look like a seismic section" complaint. This is
    a display-only per-trace balance, identical in spirit to a
    workstation's "display AGC" toggle; it never touches the processing
    data returned to the caller.
    """
    n = min(traces.shape[0], max_traces)
    traces_w = traces[:n]
    x_positions = np.asarray(x_positions[:n], dtype=float)
    diffs = np.diff(np.sort(x_positions))
    trace_spacing = float(np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 1.0

    fig = plot_section(traces_w, x_positions, twt, title=title, pclip=pclip,
                        colorscale=colorscale, height=height, equalize=equalize)
    disp = theme.equalize_traces(traces_w, pclip=95.0) if equalize else traces_w
    peak = np.percentile(np.abs(disp), 98) or 1.0
    scale = 0.45 * trace_spacing / peak
    for i in range(n):
        x = x_positions[i] + disp[i] * scale
        fig.add_trace(go.Scatter(x=x, y=twt, mode="lines", line=dict(width=0.8, color="black"),
                                  showlegend=False, hoverinfo="skip", opacity=0.75))
    return fig


def plot_wiggle_va(traces: np.ndarray, x_positions: np.ndarray, twt: np.ndarray, *,
                    title: str = "", max_traces: int = 60, trace_spacing: Optional[float] = None,
                    fill_color: str = "rgba(13,43,94,0.75)", line_color: str = "#0D2B5E",
                    height: int = 600, equalize: bool = False) -> go.Figure:
    """Classic wiggle + variable-area trace display. Only sensible for a
    modest number of traces at once (a "trace viewer" window), so the
    caller should pass an already-windowed subset — traces beyond
    `max_traces` are dropped with the excess simply not drawn (the
    caller is expected to have already limited the selection; this is a
    hard safety cap, not a feature)."""
    n = min(traces.shape[0], max_traces)
    traces = traces[:n]
    x_positions = np.asarray(x_positions[:n], dtype=float)
    if equalize:
        traces = theme.equalize_traces(traces, pclip=95.0)
    if trace_spacing is None:
        diffs = np.diff(np.sort(x_positions))
        trace_spacing = float(np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 1.0

    peak = np.percentile(np.abs(traces), 98) or 1.0
    scale = 0.45 * trace_spacing / peak

    fig = go.Figure()
    for i in range(n):
        y = twt
        x = x_positions[i] + traces[i] * scale
        baseline = np.full_like(x, x_positions[i])
        pos = np.where(traces[i] > 0, x, x_positions[i])
        fig.add_trace(go.Scatter(x=pos, y=y, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=baseline, y=y, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor=fill_color, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1, color=line_color),
                                  showlegend=False,
                                  hovertemplate=f"trace {i}<br>t=%{{y:.3f}} s<extra></extra>"))

    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("Position"), tickvals=x_positions)
    fig.update_yaxes(**theme.time_axis())
    return fig


def plot_fk_spectrum(f: np.ndarray, k: np.ndarray, magnitude: np.ndarray, *,
                      polygon_points: list | None = None, polygon_mode: str = "accept",
                      title: str = "F-K Amplitude Spectrum", height: int = 550) -> go.Figure:
    """F-K amplitude spectrum display (log-scaled magnitude) with an
    optional accept/reject polygon overlay — used both for passive F-K QC
    and for manual polygon definition. `polygon_mode` only changes the
    label/color (green=accept/signal, red=reject/noise); the boundary
    itself is drawn identically either way.

    Only the non-negative-frequency half (f >= 0) is displayed. A real
    gather's F-K spectrum is Hermitian-symmetric (f<0 is a mirror image
    of f>0, see `fk_filter_gather`'s own `inside_mirrored` handling), and
    every reference figure for this filter (report Figs. 5.64B, 5.67)
    plots only f=0..f_max — never the redundant negative-frequency half.
    Showing the full -f..+f range made the accept/reject polygon (which
    is only generated/drawn for f>=0) look like it covered just half the
    picture instead of the whole spectrum, which was the actual source of
    the reported "shape doesn't match the reference" mismatch; cropping
    to f>=0 here fixes that without changing any filtering math."""
    f_pos = f >= 0
    f = f[f_pos]
    magnitude = magnitude[:, f_pos]
    mag_db = 20 * np.log10(magnitude.T + 1e-6)
    mag_db -= mag_db.max()
    fig = go.Figure(go.Heatmap(z=mag_db, x=k, y=f, colorscale="Turbo", zmin=-60, zmax=0, zsmooth="best",
                                colorbar=dict(title="dB", **theme.COLORBAR)))
    rings = _polygon_rings(polygon_points)
    color = "#22c55e" if polygon_mode == "accept" else "#ef4444"
    label = "Accept polygon (signal, kept)" if polygon_mode == "accept" else "Reject polygon (noise, attenuated)"
    for i, ring in enumerate(rings):
        if len(ring) < 1:
            continue
        pts = ring + ([ring[0]] if len(ring) >= 3 else [])
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                  line=dict(color=color, width=2.5), marker=dict(size=8, color=color),
                                  fill="toself", fillcolor=_hex_to_rgba(color, 0.15),
                                  name=label if i == 0 else None, showlegend=(i == 0),
                                  legendgroup="fk_polygon"))
    fig.update_layout(**theme.base_layout(title, height), showlegend=bool(rings))
    fig.update_xaxes(**theme.x_axis("Wavenumber k (cycles/m)"))
    fig.update_yaxes(**theme.x_axis("Frequency (Hz)"))
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_fk_before_after(f: np.ndarray, k: np.ndarray, mag_before: np.ndarray, mag_after: np.ndarray, *,
                          polygon_points: list | None = None, polygon_mode: str = "accept",
                          height: int = 500) -> go.Figure:
    """Side-by-side F-K amplitude spectra before/after applying the
    accept/reject polygon mask — the direct QC for "is the filter
    actually removing the ground-roll energy and leaving the signal
    zone alone", which a single spectrum plot cannot show.

    Cropped to f >= 0 for the same reason as `plot_fk_spectrum` — matches
    the reference figures and keeps the polygon overlay (only ever
    generated for f>=0) filling the whole visible panel."""
    f_pos = f >= 0
    f = f[f_pos]
    mag_before = mag_before[:, f_pos]
    mag_after = mag_after[:, f_pos]
    db_before = 20 * np.log10(mag_before.T + 1e-6)
    db_before -= db_before.max()
    db_after = 20 * np.log10(mag_after.T + 1e-6)
    db_after -= db_after.max()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                         subplot_titles=("<b>F-K spectrum — before</b>", "<b>F-K spectrum — after</b>"),
                         horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=db_before, x=k, y=f, colorscale="Turbo", zmin=-60, zmax=0,
                              zsmooth="best", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=db_after, x=k, y=f, colorscale="Turbo", zmin=-60, zmax=0, zsmooth="best",
                              colorbar=dict(title="dB", **theme.COLORBAR)), row=1, col=2)
    rings = [r for r in _polygon_rings(polygon_points) if len(r) >= 3]
    color = "#22c55e" if polygon_mode == "accept" else "#ef4444"
    for ring in rings:
        pts = ring + [ring[0]]
        for col in (1, 2):
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                                      line=dict(color=color, width=2), showlegend=False),
                          row=1, col=col)
    fig.update_layout(**theme.base_layout("", height))
    for c in (1, 2):
        fig.update_xaxes(**theme.x_axis("Wavenumber k (cycles/m)"), row=1, col=c)
    fig.update_yaxes(**theme.x_axis("Frequency (Hz)"), row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=theme.TITLE_BLUE)
    return fig


def plot_fk_pass_mask(f: np.ndarray, k: np.ndarray, pass_mask: np.ndarray, *,
                       title: str = "F-K Filter — Applied Multiplier", height: int = 480) -> go.Figure:
    """Shows the actual real-valued multiplier applied at every (k, f)
    bin (0..1) — the most direct possible QC for an F-K filter: rather
    than inferring what happened from before/after spectra, this is
    exactly what was multiplied onto the data, including the taper and
    the small non-zero floor on the attenuated side (report Fig. 5.66)."""
    fig = go.Figure(go.Heatmap(z=pass_mask.T, x=k, y=f, colorscale="Greys", zmin=0, zmax=1, zsmooth="best",
                                colorbar=dict(title="Multiplier", **theme.COLORBAR)))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("Wavenumber k (cycles/m)"))
    fig.update_yaxes(**theme.x_axis("Frequency (Hz)"))
    return fig


def plot_fold_map(fold: np.ndarray, cdp_list: Sequence[int], *, title: str = "Fold Map",
                   height: int = 300) -> go.Figure:
    """CDP fold profile — the standard early-QC "Fold Map" deliverable."""
    fig = go.Figure(go.Bar(x=list(cdp_list), y=fold, marker_color=theme.TITLE_BLUE))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.x_axis("Fold (traces)"))
    return fig


def plot_amplitude_time_map(matrix: np.ndarray, cdp_list: Sequence[int], twt: np.ndarray, *,
                             title: str = "", colorscale: str = None, height: int = 300,
                             colorbar_title: str = "Fold") -> go.Figure:
    """Generic (n_cdp, n_samples) map display (fold-vs-time, static
    corrections, etc.) sharing the section colour conventions."""
    fig = go.Figure(go.Heatmap(z=matrix.T, x=list(cdp_list), y=twt, colorscale=colorscale or "Viridis",
                                colorbar=dict(title=colorbar_title, **theme.COLORBAR)))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.time_axis())
    return fig
