"""
visualization/controls.py — Interactive Section-Display Controls
=====================================================================
A small, reusable Streamlit control panel (colormap, amplitude clip,
gain, polarity) that every major section-viewing spot in the app uses,
so the person can adjust *how* a section is displayed without re-running
any processing — this was under-exposed in the first version of the app
(the underlying plot functions already accepted a clip/colormap, but no
page actually let the user change them).
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

import config as cfg

# "Seismic" here resolves to a real, high-contrast custom colorscale (see
# visualization/theme.SEISMIC_RWB) — the bare string "seismic" is never
# passed to Plotly directly, which is what used to crash with
# `ValueError: Invalid value ... colorscale ... seismic`.
COLORSCALE_OPTIONS = ["Seismic", "RdBu", "RdGy", "Picnic", "Portland", "Turbo", "Greys"]


@dataclass
class DisplayParams:
    colorscale: str
    pclip: float
    gain: float
    polarity: int
    equalize: bool = False


def section_display_controls(key_prefix: str, default_colorscale: str = None) -> DisplayParams:
    """Renders a compact, columns-based control row and returns the
    chosen display parameters. Call once per section viewer; pass a
    unique `key_prefix` per call site."""
    default_colorscale = default_colorscale or cfg.SEISMIC_CMAP
    c1, c2, c3, c4 = st.columns(4)
    colorscale = c1.selectbox("Colormap", COLORSCALE_OPTIONS,
                               index=COLORSCALE_OPTIONS.index(default_colorscale)
                               if default_colorscale in COLORSCALE_OPTIONS else 0,
                               key=f"{key_prefix}_colorscale")
    pclip = c2.slider("Amplitude clip (percentile)", 80.0, 100.0, cfg.PCLIP_DEFAULT, step=0.5,
                       key=f"{key_prefix}_pclip", help="Lower values clip more, boosting weaker events "
                                                        "at the cost of saturating the strongest ones.")
    gain = c3.slider("Display gain", 0.1, 5.0, 1.0, step=0.1, key=f"{key_prefix}_gain")
    polarity_label = c4.radio("Polarity", ["Normal", "Reversed"], horizontal=True, key=f"{key_prefix}_polarity")
    polarity = 1 if polarity_label == "Normal" else -1
    equalize = st.checkbox(
        "Trace-balanced display (equalize each trace's display amplitude)", value=False,
        key=f"{key_prefix}_equalize",
        help="Display-only aid — never changes the underlying data. Normalizes each trace's own "
             "peak before applying the clip above, so far-offset/weak-amplitude traces (e.g. before "
             "AGC) are still visible instead of looking blank. Turn this off to see true relative "
             "amplitude.")
    return DisplayParams(colorscale=colorscale, pclip=pclip, gain=gain * polarity, polarity=polarity,
                          equalize=equalize)
