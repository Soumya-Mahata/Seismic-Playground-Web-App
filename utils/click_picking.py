"""
utils/click_picking.py — Point-Picking Helpers for Plotly Charts
====================================================================
Why this no longer relies on click-on-the-chart
------------------------------------------------
Streamlit's `st.plotly_chart(..., on_select="rerun")` only delivers
selection *events* for a handful of trace types — scatter/scattergl,
bar, and histogram markers. It does **not** fire point-selection events
for `go.Heatmap` traces (there is no per-cell "marker" for the browser
to hit-test against the way there is for a scatter point), and every
picker in this app (F-K polygon, horizon/fault picking, velocity
semblance picking) is drawn on a Heatmap background. Practically, this
means the old "click directly on the plot" instruction never actually
worked for these panels — no error is raised, the callback is just
never invoked.

There's a second, more disruptive side effect: turning on
`on_select=...` for a chart also switches Plotly's default *drag* mode
from "zoom" to "select" for that chart, so dragging on the plot draws a
selection rectangle instead of zooming — which is exactly the "it's
neither clickable nor zoomable any more" regression. Removing
`on_select` restores normal zoom/pan/box-zoom/double-click-to-reset
behavior immediately.

The fix used throughout this app: keep the chart as a plain
`st.plotly_chart(fig, theme=None)` (full native zoom/pan restored), and
add a small, explicit "Add point" control (dropdown/number-input +
button) beneath it instead of relying on a click event that this
combination of chart type + Streamlit version cannot deliver. This is
slower than a genuine click-to-pick would be, but it is honest and it
reliably works, which a silently-broken click handler does not.
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st


def consume_new_click(event, dedup_key: str) -> Optional[Tuple[float, float]]:
    """Kept for any chart built on a genuinely click-selectable trace
    type (scatter/bar/histogram) that still wants on_select behavior.
    Returns (x, y) of a newly-clicked point, or None if there's no
    selection or it's the same click already consumed on a prior rerun.
    Do NOT use this on a Heatmap-backed figure — see module docstring;
    it will silently never fire.
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
    return ("🔍 Zoom/pan freely on the plot below (drag to zoom, double-click to reset). "
            "Add points precisely with the controls beneath it.")


def numeric_point_form(key_prefix: str, x_label: str, y_label: str, *,
                        x_value: float = 0.0, y_value: float = 0.0,
                        x_min: Optional[float] = None, x_max: Optional[float] = None,
                        y_min: Optional[float] = None, y_max: Optional[float] = None,
                        x_step: float = 1.0, y_step: float = 0.01,
                        x_format: Optional[str] = None, y_format: Optional[str] = None,
                        button_label: str = "Add point") -> Optional[Tuple[float, float]]:
    """A small (x, y) entry row + "Add point" button, replacing a click
    on a chart that cannot deliver click events (see module docstring).
    Returns (x, y) only on the run where the button was pressed."""
    c1, c2, c3 = st.columns([2, 2, 1])
    kwargs_x = dict(value=float(x_value), step=x_step, key=f"{key_prefix}_xval")
    kwargs_y = dict(value=float(y_value), step=y_step, key=f"{key_prefix}_yval")
    if x_min is not None:
        kwargs_x["min_value"] = float(x_min)
    if x_max is not None:
        kwargs_x["max_value"] = float(x_max)
    if y_min is not None:
        kwargs_y["min_value"] = float(y_min)
    if y_max is not None:
        kwargs_y["max_value"] = float(y_max)
    if x_format:
        kwargs_x["format"] = x_format
    if y_format:
        kwargs_y["format"] = y_format
    x = c1.number_input(x_label, **kwargs_x)
    y = c2.number_input(y_label, **kwargs_y)
    c3.markdown("&nbsp;")
    if c3.button(button_label, key=f"{key_prefix}_addbtn"):
        return (float(x), float(y))
    return None
