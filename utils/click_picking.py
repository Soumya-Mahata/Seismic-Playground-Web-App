"""
utils/click_picking.py — Click-to-Pick Helper for Plotly Charts
====================================================================
Requires Streamlit >= 1.35 (`st.plotly_chart(..., on_select="rerun")`).

A single click on a Plotly chart triggers a script rerun with the
clicked point's data coordinates available as
`event.selection.points[-1]["x"/"y"]` (confirmed against the installed
Streamlit's `PlotlySelectionState` schema — see
`streamlit.elements.plotly_chart`). Each click *replaces* the prior
selection rather than accumulating it, so building up a multi-point
pick (a horizon, a fault line, an F-K polygon) requires tracking
"have I already consumed this particular click" across reruns —
that's what `consume_new_click` does.

Used identically for horizon picking, fault picking, F-K polygon
drawing, and manual velocity control points — one click-tracking
mechanism, four different pages/purposes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st


def consume_new_click(event, dedup_key: str) -> Optional[Tuple[float, float]]:
    """Return (x, y) of a newly-clicked chart point, or None if there's
    no selection or it's the same click already consumed on a prior
    rerun (Streamlit re-delivers the current selection on every rerun
    until the person clicks again, so without de-duplication the same
    point would be re-added on every unrelated interaction on the page).
    """
    try:
        points = event.selection.points if event is not None else []
    except Exception:
        points = []
    if not points:
        return None
    pt = points[-1]
    try:
        xy = (float(pt["x"]), float(pt["y"]))
    except (KeyError, TypeError, ValueError):
        return None
    if st.session_state.get(dedup_key) == xy:
        return None
    st.session_state[dedup_key] = xy
    return xy


def click_help_caption() -> str:
    return ("🖱️ Click directly on the plot to add a point. Each click adds one point; "
            "use the controls below to undo or clear.")
