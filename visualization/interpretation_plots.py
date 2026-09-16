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
                             title: str = "", pclip: float = 98.0, height: int = 600) -> go.Figure:
    fig = seismic_plots.plot_section(section, np.array(cdp_list), twt, title=title,
                                      x_title="CDP", pclip=pclip, height=height)
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


def plot_attribute_along_horizon(values: np.ndarray, cdp_list: Sequence[int], *,
                                  title: str = "Attribute Along Horizon", colorbar_title: str = "Value",
                                  height: int = 320) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(cdp_list), y=values, mode="lines+markers",
                                marker=dict(size=5, color=values, colorscale=theme.ATTRIBUTE_COLORSCALE,
                                            colorbar=dict(title=colorbar_title, **theme.COLORBAR)),
                                line=dict(color="rgba(100,100,100,0.4)")))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("CDP"))
    fig.update_yaxes(**theme.x_axis(colorbar_title))
    return fig
