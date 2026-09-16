"""
visualization/spectra_plots.py — Amplitude Spectrum Displays
=================================================================
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from modules.filtering import amplitude_spectrum
from visualization import theme


def plot_spectrum(data: np.ndarray, dt: float, *, title: str = "Amplitude Spectrum",
                   fmax_display: float = None, height: int = 380) -> go.Figure:
    f, amp = amplitude_spectrum(data, dt)
    fmax_display = fmax_display or (0.5 / dt)
    mask = f <= fmax_display
    fig = go.Figure(go.Scatter(x=f[mask], y=amp[mask], mode="lines",
                                line=dict(color=theme.TITLE_BLUE, width=2), fill="tozeroy",
                                fillcolor="rgba(13,43,94,0.15)"))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("Frequency (Hz)"))
    fig.update_yaxes(**theme.x_axis("Relative amplitude"))
    return fig


def plot_spectrum_comparison(before: np.ndarray, after: np.ndarray, dt: float, *,
                              label_before: str = "Before", label_after: str = "After",
                              title: str = "Amplitude Spectrum — Before vs After",
                              fmax_display: float = None, height: int = 380) -> go.Figure:
    fb, ab = amplitude_spectrum(before, dt)
    fa, aa = amplitude_spectrum(after, dt)
    fmax_display = fmax_display or (0.5 / dt)
    mb, ma = fb <= fmax_display, fa <= fmax_display
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fb[mb], y=ab[mb], mode="lines", name=label_before,
                              line=dict(color="#9e9e9e", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=fa[ma], y=aa[ma], mode="lines", name=label_after,
                              line=dict(color=theme.TITLE_BLUE, width=2)))
    fig.update_layout(**theme.base_layout(title, height))
    fig.update_xaxes(**theme.x_axis("Frequency (Hz)"))
    fig.update_yaxes(**theme.x_axis("Relative amplitude"))
    return fig
