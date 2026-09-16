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


def plot_section(section: np.ndarray, x_values: np.ndarray, twt: np.ndarray, *,
                  title: str = "", x_title: str = "CDP", pclip: float = cfg.PCLIP_DEFAULT,
                  colorscale: str = None, height: int = 550,
                  gain: float = 1.0) -> go.Figure:
    """Standard seismic-section image display: time down, CDP/trace
    across, symmetric percentile amplitude clip, diverging colormap."""
    colorscale = colorscale or theme.SEISMIC_COLORSCALE
    data = np.nan_to_num(section, nan=0.0) * gain
    clip = theme.percentile_clip(data, pclip)
    fig = go.Figure(go.Heatmap(
        z=data.T, x=x_values, y=twt, zmin=-clip, zmax=clip, colorscale=colorscale,
        colorbar=dict(title="Amplitude", **theme.COLORBAR),
    ))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis(x_title))
    fig.update_yaxes(**theme.time_axis())
    return fig


def plot_gather(traces: np.ndarray, offsets: np.ndarray, twt: np.ndarray, *,
                 title: str = "", pclip: float = cfg.PCLIP_DEFAULT, colorscale: str = None,
                 height: int = 550) -> go.Figure:
    """A single CDP gather, offset on x."""
    order = np.argsort(offsets)
    return plot_section(traces[order], offsets[order], twt, title=title,
                         x_title="Offset (m)", pclip=pclip, colorscale=colorscale, height=height)


def plot_before_after(before: np.ndarray, after: np.ndarray, x_values: np.ndarray, twt: np.ndarray, *,
                       title_before: str = "Before", title_after: str = "After",
                       x_title: str = "CDP", pclip: float = cfg.PCLIP_DEFAULT, height: int = 550) -> go.Figure:
    """Side-by-side before/after section comparison sharing one clip
    level (computed from the "after" section) and one time axis."""
    clip = theme.percentile_clip(np.nan_to_num(after, nan=0.0), pclip)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                         subplot_titles=(f"<b>{title_before}</b>", f"<b>{title_after}</b>"),
                         horizontal_spacing=0.06)
    fig.add_trace(go.Heatmap(z=np.nan_to_num(before, nan=0.0).T, x=x_values, y=twt,
                              zmin=-clip, zmax=clip, colorscale=theme.SEISMIC_COLORSCALE,
                              showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=np.nan_to_num(after, nan=0.0).T, x=x_values, y=twt,
                              zmin=-clip, zmax=clip, colorscale=theme.SEISMIC_COLORSCALE,
                              colorbar=dict(title="Amplitude", **theme.COLORBAR)), row=1, col=2)
    fig.update_layout(**theme.base_layout(title="", height=height))
    for c in (1, 2):
        fig.update_xaxes(**theme.x_axis(x_title), row=1, col=c)
    fig.update_yaxes(**theme.time_axis(), row=1, col=1)
    fig.update_yaxes(**theme.time_axis(), row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=theme.TITLE_BLUE)
    return fig


def plot_variable_density_with_wiggle(traces: np.ndarray, x_positions: np.ndarray, twt: np.ndarray, *,
                                       title: str = "", max_traces: int = 40, pclip: float = cfg.PCLIP_DEFAULT,
                                       colorscale: str = None, height: int = 600) -> go.Figure:
    """Variable-density (color image) background with wiggle traces
    overlaid — "VD+WT" display, a standard industry combination that
    shows both the continuous amplitude field and individual trace
    shape at once. Same `max_traces` safety cap as `plot_wiggle_va`."""
    n = min(traces.shape[0], max_traces)
    traces_w = traces[:n]
    x_positions = np.asarray(x_positions[:n], dtype=float)
    diffs = np.diff(np.sort(x_positions))
    trace_spacing = float(np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 1.0

    fig = plot_section(traces_w, x_positions, twt, title=title, pclip=pclip,
                        colorscale=colorscale, height=height)
    peak = np.percentile(np.abs(traces_w), 98) or 1.0
    scale = 0.45 * trace_spacing / peak
    for i in range(n):
        x = x_positions[i] + traces_w[i] * scale
        fig.add_trace(go.Scatter(x=x, y=twt, mode="lines", line=dict(width=0.8, color="black"),
                                  showlegend=False, hoverinfo="skip", opacity=0.7))
    return fig


def plot_wiggle_va(traces: np.ndarray, x_positions: np.ndarray, twt: np.ndarray, *,
                    title: str = "", max_traces: int = 40, trace_spacing: Optional[float] = None,
                    fill_color: str = "rgba(13,43,94,0.65)", line_color: str = "#0D2B5E",
                    height: int = 600) -> go.Figure:
    """Classic wiggle + variable-area trace display. Only sensible for a
    modest number of traces at once (a "trace viewer" window), so the
    caller should pass an already-windowed subset — traces beyond
    `max_traces` are dropped with the excess simply not drawn (the
    caller is expected to have already limited the selection; this is a
    hard safety cap, not a feature)."""
    n = min(traces.shape[0], max_traces)
    traces = traces[:n]
    x_positions = np.asarray(x_positions[:n], dtype=float)
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
                      polygon_points: list | None = None, title: str = "F-K Amplitude Spectrum",
                      height: int = 550) -> go.Figure:
    """F-K amplitude spectrum display (log-scaled magnitude) with an
    optional reject-polygon overlay — used both for passive F-K QC and
    for manual (click-to-draw) polygon definition."""
    mag_db = 20 * np.log10(magnitude.T + 1e-6)
    mag_db -= mag_db.max()
    fig = go.Figure(go.Heatmap(z=mag_db, x=k, y=f, colorscale="Viridis", zmin=-60, zmax=0,
                                colorbar=dict(title="dB", **theme.COLORBAR)))
    if polygon_points and len(polygon_points) >= 1:
        pts = list(polygon_points) + ([polygon_points[0]] if len(polygon_points) >= 3 else [])
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                  line=dict(color="#ef4444", width=2.5), marker=dict(size=8, color="#ef4444"),
                                  name="Reject polygon"))
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
