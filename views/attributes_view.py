"""views/attributes_view.py — Seismic Attributes."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules.attributes import ATTRIBUTE_COLORSCALE_MAP, ATTRIBUTE_REGISTRY
from modules.interpretation import extract_attribute_along_horizon, interpolate_horizon
from utils import session_state as ss
from visualization import interpretation_plots, seismic_plots, theme


def render():
    st.header("🧬 Seismic Attributes")
    st.caption("Step 5 of the interpretation workflow — Well Tie → Horizon Picking → "
                "Fault Interpretation → Surface Map (see the **Interpretation** page) → **Attribute "
                "Analysis**. Standard Hilbert-transform-based instantaneous attributes, computed on "
                "the processed section.")

    section_state, section_label = ss.best_available_state()
    section = ss.section(section_state)
    if section is None:
        st.info("Attributes need a section: run Stacking (or a pre-stack migration) first.", icon="ℹ️")
        return

    survey = st.session_state["survey"]
    cdp_list = ss.cdp_list(section_state)
    dt, twt = survey.sample_interval, survey.twt

    st.caption(f"Computing on: **{section_label}** (state {section_state.id})")
    attr_name = st.selectbox("Attribute", list(ATTRIBUTE_REGISTRY.keys()))
    if st.button("Compute attribute", type="primary"):
        with st.spinner(f"Computing {attr_name}..."):
            values = ATTRIBUTE_REGISTRY[attr_name](section, dt)
        st.session_state["attributes"][attr_name] = values
        st.rerun()

    if attr_name in st.session_state["attributes"]:
        values = st.session_state["attributes"][attr_name]
        attr_colorscale = ATTRIBUTE_COLORSCALE_MAP.get(attr_name, theme.ATTRIBUTE_COLORSCALE)
        st.caption(f"Color scale: **{attr_name}** is displayed with the Petrel-convention scale for "
                    "this attribute family — see the Interpretation & Attributes section of the "
                    "README for the full per-attribute color mapping.")
        st.plotly_chart(seismic_plots.plot_amplitude_time_map(
            values, cdp_list, twt, title=attr_name, colorscale=attr_colorscale,
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
                    vals, cdp_list, title=f"{attr_name} along '{hname}'", colorbar_title=attr_name,
                    colorscale=attr_colorscale), theme=None, key="attributes_view_chart_1")
