"""
visualization/velocity_plots.py — Velocity Analysis Displays
=================================================================
Semblance panels with the auto-pick overlay (and, when present, manual
control points), plus the final 2-D V(t, CDP) field map.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization import theme


def plot_semblance_panel(semblance: np.ndarray, velocities: np.ndarray, twt: np.ndarray, *,
                          picked_velocity: Optional[np.ndarray] = None,
                          manual_points: Optional[Sequence[tuple]] = None,
                          title: str = "Semblance", height: int = 550) -> go.Figure:
    """One analysis-CDP semblance panel: velocity across, time down,
    coherence as color, with the picked velocity function overlaid."""
    fig = go.Figure(go.Heatmap(z=semblance.T, x=velocities, y=twt, colorscale=theme.SEMBLANCE_COLORSCALE,
                                zmin=0, zmax=float(np.percentile(semblance, 99.5)) or 1.0,
                                colorbar=dict(title="Semblance", **theme.COLORBAR)))
    if picked_velocity is not None:
        fig.add_trace(go.Scatter(x=picked_velocity, y=twt, mode="lines",
                                  line=dict(color="white", width=2.5), name="Auto pick"))
    if manual_points:
        pts_v = [p[1] for p in manual_points]
        pts_t = [p[0] for p in manual_points]
        fig.add_trace(go.Scatter(x=pts_v, y=pts_t, mode="markers", marker=dict(color="#facc15", size=10,
                                  line=dict(color="black", width=1)), name="Manual control points"))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("Velocity (m/s)"))
    fig.update_yaxes(**theme.time_axis())
    return fig


def plot_velocity_field(Vfield: np.ndarray, cdp_list: Sequence[int], twt: np.ndarray, *,
                         title: str = "Stacking-Velocity Field", height: int = 480) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=Vfield, x=list(cdp_list), y=twt, colorscale=theme.VELOCITY_COLORSCALE,
                                colorbar=dict(title="Velocity (m/s)", **theme.COLORBAR)))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.time_axis())
    return fig


def plot_cvs_panel(cvs_panel: np.ndarray, power: np.ndarray, trial_velocities: np.ndarray,
                    twt: np.ndarray, *, title: str = "Constant Velocity Stack (CVS)",
                    height: int = 550) -> go.Figure:
    """Stacked CVS display: the stacked-trace panel on top (velocity
    across, time down, amplitude as color) and the total stack-power
    curve below, sharing the same velocity axis — the industry rule of
    thumb is "the velocity that stacks best has the highest power"."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                         vertical_spacing=0.08,
                         subplot_titles=("<b>CVS panel</b>", "<b>Stack power vs. trial velocity</b>"))
    clip = float(np.percentile(np.abs(cvs_panel), 98)) or 1.0
    fig.add_trace(go.Heatmap(z=cvs_panel.T, x=trial_velocities, y=twt, zmin=-clip, zmax=clip,
                              colorscale=theme.SEISMIC_COLORSCALE, showscale=False), row=1, col=1)
    best_v = float(trial_velocities[np.argmax(power)])
    fig.add_trace(go.Scatter(x=trial_velocities, y=power, mode="lines+markers",
                              line=dict(color=theme.TITLE_BLUE, width=2), marker=dict(size=4),
                              showlegend=False), row=2, col=1)
    fig.add_vline(x=best_v, line_dash="dash", line_color="#c62828", row=2, col=1)
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_yaxes(**theme.time_axis(), row=1, col=1)
    fig.update_xaxes(**theme.x_axis("Trial velocity (m/s)"), row=2, col=1)
    fig.update_yaxes(**theme.x_axis("Stack power"), row=2, col=1)
    fig.add_annotation(text=f"Best overall: {best_v:.0f} m/s", xref="paper", yref="paper",
                        x=1.0, y=1.08, showarrow=False, font=dict(color=theme.TITLE_BLUE, size=12))
    return fig


def plot_reliability_summary(results: Dict, height: int = 320) -> go.Figure:
    """Bar chart of each analysis CDP's reliability score, colored by
    whether it passed the lateral-consistency gate — the at-a-glance QC
    the person needs before trusting the field built from these picks."""
    cdps = list(results.keys())
    rel = [results[c].reliability for c in cdps]
    colors = ["#2e7d32" if results[c].is_reliable else "#c62828" for c in cdps]
    labels = ["manual" if results[c].is_manual_override else ("reliable" if results[c].is_reliable else "flagged")
              for c in cdps]
    fig = go.Figure(go.Bar(x=[str(c) for c in cdps], y=rel, marker_color=colors,
                            text=labels, textposition="outside"))
    fig.add_hline(y=0.85, line_dash="dash", line_color="#888",
                  annotation_text="typical reliability threshold", annotation_font_color=theme.TEXT_DARK)
    fig.update_layout(**theme.base_layout("Analysis-CDP Pick Reliability", height))
    fig.update_xaxes(**theme.x_axis("Analysis CDP"))
    fig.update_yaxes(**theme.x_axis("Reliability score"), range=[0, 1.05])
    return fig
