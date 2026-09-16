"""
visualization/theme.py — Shared Plotly Style Tokens
=======================================================
Mirrors the visual language of the reference Petrophysics app's
`plots.py`: forced white plot backgrounds with dark text so scientific
plots stay legible regardless of whether the surrounding Streamlit app
is in light or dark mode (a plot that inherits the app's dark background
loses most colormaps' contrast).
"""

import numpy as np
import plotly.graph_objects as go

import config as cfg

TEXT_DARK = "#212121"
TITLE_BLUE = "#0D2B5E"

GRID = dict(showgrid=True, gridcolor="#d4d4d4", gridwidth=1,
            zeroline=True, zerolinecolor="#bbbbbb", zerolinewidth=1)

MARGIN = dict(l=60, r=30, t=60, b=50)

LEGEND = dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
              bgcolor="rgba(255,255,255,0.95)", bordercolor="#888888", borderwidth=1,
              font=dict(color=TEXT_DARK, size=11))

COLORBAR = dict(tickfont=dict(color=TEXT_DARK, size=11), title_font=dict(color=TEXT_DARK, size=12),
                 bgcolor="white", bordercolor="#aaaaaa", borderwidth=1,
                 outlinecolor="#aaaaaa", outlinewidth=1, thickness=14, len=0.8)

SEISMIC_COLORSCALE = cfg.SEISMIC_CMAP        # diverging, colorblind-friendlier than the classic red/blue "seismic"
SEMBLANCE_COLORSCALE = "Jet"          # standard in the geophysics community for coherence panels
VELOCITY_COLORSCALE = "Turbo"
ATTRIBUTE_COLORSCALE = "Viridis"


def base_layout(title: str = "", height: int = 550) -> dict:
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=TITLE_BLUE), x=0, xanchor="left"),
        height=height, margin=MARGIN, legend=LEGEND,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT_DARK),
    )


def time_axis(title: str = "Two-way time (s)", reversed_axis: bool = True) -> dict:
    ax = dict(title_text=title, title_font=dict(size=12, color=TEXT_DARK),
              tickfont=dict(color=TEXT_DARK, size=10), **GRID)
    if reversed_axis:
        ax["autorange"] = "reversed"
    return ax


def x_axis(title: str) -> dict:
    return dict(title_text=title, title_font=dict(size=12, color=TEXT_DARK),
                tickfont=dict(color=TEXT_DARK, size=10), **GRID)


def fix_colorbar(fig: go.Figure) -> go.Figure:
    """Force every colorbar/trace text on a figure to the dark-on-white
    scheme (Plotly's per-trace styling can otherwise default to grey)."""
    fig.update_layout(font=dict(color=TEXT_DARK))
    return fig


def percentile_clip(data: np.ndarray, pclip: float = cfg.PCLIP_DEFAULT) -> float:
    """Symmetric amplitude-clip value for section display, matching the
    notebook's own visualization convention (percentile of |amplitude|)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    v = np.percentile(np.abs(finite), pclip)
    return float(v) if v > 0 else 1.0
