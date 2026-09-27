"""views/attributes_view.py — Seismic Attributes."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules.attributes import ATTRIBUTE_REGISTRY
from modules.interpretation import extract_attribute_along_horizon, interpolate_horizon
from utils import session_state as ss
from visualization import interpretation_plots, seismic_plots, theme


def _best_available_section():
    for key, label in [("stack_final", "Final processed section"), ("stack_migrated", "Migrated section"),
                        ("stack_tvf", "Post-stack (pre-migration)"), ("stack_matrix", "Stack")]:
        if st.session_state.get(key) is not None:
            return st.session_state[key], label
    return None, None


def render():
    st.header("🧬 Seismic Attributes")
    st.caption("Step 5 of the interpretation workflow (report Ch.8.6) — Well Tie → Horizon Picking → "
                "Fault Interpretation → Surface Map (see the **Interpretation** page) → **Attribute "
                "Analysis**. Standard Hilbert-transform-based instantaneous attributes, computed on "
                "the processed section.")

    section, section_label = _best_available_section()
    if section is None:
        st.info("Run at least the Stacking step to get a section to compute attributes on.", icon="ℹ️")
        return

    survey = st.session_state["survey"]
    cdp_list = st.session_state["cdp_list"]
    dt, twt = survey.sample_interval, survey.twt

    st.caption(f"Computing on: **{section_label}**")
    attr_name = st.selectbox("Attribute", list(ATTRIBUTE_REGISTRY.keys()))
    if st.button("Compute attribute", type="primary"):
        with st.spinner(f"Computing {attr_name}..."):
            values = ATTRIBUTE_REGISTRY[attr_name](section, dt)
        st.session_state["attributes"][attr_name] = values
        st.rerun()

    if attr_name in st.session_state["attributes"]:
        values = st.session_state["attributes"][attr_name]
        st.plotly_chart(seismic_plots.plot_amplitude_time_map(
            values, cdp_list, twt, title=attr_name, colorscale=theme.ATTRIBUTE_COLORSCALE,
            colorbar_title=attr_name, height=550), theme=None, key="attributes_view_chart_1001")

        horizons = st.session_state["horizons"]
        if horizons:
            st.divider()
            st.markdown("**Extract along a horizon**")
            hname = st.selectbox("Horizon", list(horizons.keys()))
            half_window_ms = st.slider("Extraction window (± ms)", 0, 100, 10)
            h = horizons[hname]
            horizon_time = interpolate_horizon(h.picks, cdp_list)
            if np.all(np.isnan(horizon_time)):
                st.warning("This horizon has no picks yet — add some in the Interpretation page first.")
            else:
                vals = extract_attribute_along_horizon(values, cdp_list, horizon_time, dt,
                                                         half_window_s=half_window_ms / 1000.0)
                st.plotly_chart(interpretation_plots.plot_attribute_along_horizon(
                    vals, cdp_list, title=f"{attr_name} along '{hname}'", colorbar_title=attr_name), theme=None, key="attributes_view_chart_1")
