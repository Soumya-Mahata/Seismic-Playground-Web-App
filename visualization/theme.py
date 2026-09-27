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

SEMBLANCE_COLORSCALE = "Jet"          # standard in the geophysics community for coherence panels
VELOCITY_COLORSCALE = "Turbo"
ATTRIBUTE_COLORSCALE = "Viridis"

# ── "Seismic" colormap ──────────────────────────────────────────────────
# Plotly has NO built-in colorscale literally named "seismic" (that name
# only exists in matplotlib) — passing the bare string "seismic" straight
# into go.Heatmap raises a ValueError, which is the exact crash reported
# against this app. Reproduced here as an explicit [value, color] list so
# selecting "seismic" in the UI works, and so it looks like a real
# industry seismic-section display: a hard, dark navy-blue -> white ->
# dark maroon-red diverging ramp (matplotlib's own "seismic" control
# points), which — unlike a pastel scale such as plain "RdBu" — keeps
# strong reflectors visually "black-and-white-ish" and high contrast the
# way a variable-density display on a real workstation looks.
SEISMIC_RWB = [
    [0.00, "#00004d"], [0.10, "#00008f"], [0.25, "#0000ff"], [0.40, "#7f7fff"],
    [0.48, "#e6e6ff"], [0.50, "#ffffff"], [0.52, "#ffe6e6"], [0.60, "#ff7f7f"],
    [0.75, "#ff0000"], [0.90, "#8f0000"], [1.00, "#4d0000"],
]

# Named colorscales the display controls expose. Everything except
# "Seismic" is resolved straight through to Plotly's own built-in name;
# "Seismic" is resolved to SEISMIC_RWB by `resolve_colorscale` below.
NAMED_COLORSCALES = {"Seismic": SEISMIC_RWB}

SEISMIC_COLORSCALE = SEISMIC_RWB if cfg.SEISMIC_CMAP.lower() == "seismic" else cfg.SEISMIC_CMAP


def resolve_colorscale(name):
    """Turn a colorscale *name* chosen in the UI into something Plotly's
    go.Heatmap will actually accept. Plotly only understands its own
    built-in names (RdBu, Turbo, ...) or an explicit [value, color] list
    — it does NOT understand "seismic" (a matplotlib-only name), which is
    exactly the bug this resolves: any caller that used to hand the bare
    string "seismic" straight to a Heatmap would crash with
    `ValueError: Invalid value ... seismic`. Case-insensitive so old
    session state / bookmarked names still resolve.
    """
    if name is None:
        return SEISMIC_RWB
    if isinstance(name, (list, tuple)):
        return name
    key = str(name).strip().lower()
    for label, scale in NAMED_COLORSCALES.items():
        if label.lower() == key:
            return scale
    return name


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


def equalize_traces(data: np.ndarray, pclip: float = 95.0) -> np.ndarray:
    """Per-trace DISPLAY normalization: divide each trace by its own
    robust peak amplitude, then rescale by the ensemble's median peak so
    the section keeps a sensible overall amplitude level. This is a
    *display-only* transform (never written back into the processing
    data) that fixes the single biggest reason a variable-density image
    of early/raw gathers "doesn't look like a seismic section": true
    relative amplitude decays hugely with offset/time before AGC, so a
    single global clip level leaves far offsets looking like blank white
    while the near offset saturates. Real seismic workstations offer the
    identical toggle (often called "trace-balanced display" or "display
    AGC") for exactly this reason — it is a viewing aid, not a processing
    step, which is why it lives here rather than in `modules/`.
    """
    finite = np.nan_to_num(data, nan=0.0)
    scale = np.percentile(np.abs(finite), pclip, axis=1, keepdims=True)
    scale = np.where(scale <= 0, 1.0, scale)
    med = np.median(scale[scale > 0]) if np.any(scale > 0) else 1.0
    return finite / scale * med


def percentile_clip(data: np.ndarray, pclip: float = cfg.PCLIP_DEFAULT) -> float:
    """Symmetric amplitude-clip value for section display, matching the
    notebook's own visualization convention (percentile of |amplitude|)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    v = np.percentile(np.abs(finite), pclip)
    return float(v) if v > 0 else 1.0
