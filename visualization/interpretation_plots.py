"""
visualization/interpretation_plots.py — Interpretation Displays
====================================================================
Horizon/fault overlays on a seismic section, and attribute-along-horizon
maps.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import plotly.graph_objects as go

from modules.interpretation import FaultLine, Horizon, interpolate_horizon
from visualization import seismic_plots, theme


def plot_section_with_picks(section: np.ndarray, cdp_list: Sequence[int], twt: np.ndarray, *,
                             horizons: Dict[str, Horizon] = None, faults: Dict[str, FaultLine] = None,
                             title: str = "", pclip: float = 98.0, height: int = 600,
                             colorscale: str = None, equalize: bool = False) -> go.Figure:
    fig = seismic_plots.plot_section(section, np.array(cdp_list), twt, title=title,
                                      x_title="CDP", pclip=pclip, height=height,
                                      colorscale=colorscale, equalize=equalize)
    for name, h in (horizons or {}).items():
        interp = interpolate_horizon(h.picks, list(cdp_list))
        fig.add_trace(go.Scatter(x=list(cdp_list), y=interp, mode="lines", name=name,
                                  line=dict(color=h.color, width=2.5)))
        picked_cdp = list(h.picks.keys())
        fig.add_trace(go.Scatter(x=picked_cdp, y=[h.picks[c] for c in picked_cdp], mode="markers",
                                  marker=dict(color=h.color, size=7, line=dict(color="black", width=1)),
                                  name=f"{name} (picks)", showlegend=False))
    for name, fl in (faults or {}).items():
        if len(fl.points) < 1:
            continue
        pts = sorted(fl.points)
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                  name=name, line=dict(color=fl.color, width=2.5, dash="dot"),
                                  marker=dict(symbol="x", size=8)))
    fig.update_layout(legend=theme.LEGEND, showlegend=True)
    return fig


def plot_structure_profile(values: np.ndarray, cdp_list: Sequence[int], *, y_title: str = "TWT (s)",
                            title: str = "", invert_y: bool = True, color: str = "#0D2B5E") -> go.Figure:
    """A time/depth structure "profile" along CDP — the 2D-line analogue
    of an areal time/depth structure map (report Sec 8.5): this app only
    ever works with a single 2D line, so there is no areal grid to
    contour, but the same quantity (a picked/converted horizon's
    time-or-depth value at every CDP) is shown as a continuous profile,
    which is the correct 2D equivalent and is not a fabricated stand-in
    for a 3D map."""
    fig = go.Figure(go.Scatter(x=list(cdp_list), y=values, mode="lines", line=dict(color=color, width=2.5),
                                fill="tozeroy", fillcolor="rgba(13,43,94,0.12)"))
    fig.update_layout(**theme.base_layout(title, 420))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.time_axis(title=y_title, reversed_axis=invert_y))
    return fig


def plot_attribute_along_horizon(values: np.ndarray, cdp_list: Sequence[int], *,
                                  title: str = "Attribute Along Horizon", colorbar_title: str = "Value",
                                  colorscale: str = None, height: int = 320) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(cdp_list), y=values, mode="lines+markers",
                                marker=dict(size=5, color=values, colorscale=colorscale or theme.ATTRIBUTE_COLORSCALE,
                                            colorbar=dict(title=colorbar_title, **theme.COLORBAR)),
                                line=dict(color="rgba(100,100,100,0.4)")))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.x_axis(colorbar_title))
    return fig
