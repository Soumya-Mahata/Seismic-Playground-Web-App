"""views/poststack_view.py — Post-Stack Processing (steps 18-20)."""

from __future__ import annotations

import streamlit as st

import config as cfg
from modules import filtering, preprocessing
from utils import session_state as ss
from visualization import seismic_plots, spectra_plots


def render():
    st.header("🧹 Post-Stack Processing")
    st.caption("Steps 18-20 of the reference pipeline.")

    if not ss.require_stage("stack"):
        return

    survey = st.session_state["survey"]
    cdp_list = st.session_state["cdp_list"]
    dt, twt = survey.sample_interval, survey.twt
    stack_matrix = st.session_state["stack_matrix"]

    tabs = st.tabs(["18. Random Noise Attenuation", "19. Deconvolution", "20. Time-Variant Filter"])

    with tabs[0]:
        st.markdown("A spatial (across-CDP) median filter suppresses energy that doesn't correlate "
                     "trace-to-trace — random noise — while preserving continuous, dipping events. "
                     "Documented simplification (from the notebook): stands in for F-X deconvolution, "
                     "which needs higher fold than a small teaching/demo line typically provides.")
        window = st.slider("Spatial window (traces)", 3, 15, cfg.RNA_SPATIAL_WINDOW_DEFAULT, step=2, key="rna_window")
        if st.button("Run random noise attenuation", type="primary", key="run_rna"):
            st.session_state["stack_rna"] = filtering.random_noise_attenuation(stack_matrix, window)
            ss.mark_stage_done("rna", {"spatial_window": window})
            st.rerun()
        if ss.is_stage_done("rna"):
            st.success("✅ Random noise attenuation applied.")
            st.plotly_chart(seismic_plots.plot_before_after(
                stack_matrix, st.session_state["stack_rna"], cdp_list, twt,
                title_before="Stack", title_after="After RNA"), theme=None, key="poststack_view_chart_1001")

    with tabs[1]:
        if not ss.require_stage("rna"):
            pass
        else:
            st.markdown("A second, milder spiking-deconvolution pass on the stacked section sharpens "
                         "the effective wavelet further now that stacking has improved signal-to-noise.")
            oplen = st.slider("Operator length (samples)", 4, 60, cfg.DECON_AFTER_OPLEN_DEFAULT, key="decon2_oplen")
            pw = st.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_AFTER_PW_DEFAULT,
                                  step=0.005, format="%.3f", key="decon2_pw")
            if st.button("Run post-stack deconvolution", type="primary", key="run_decon2"):
                st.session_state["stack_decon_post"] = preprocessing.spiking_decon_gather(
                    st.session_state["stack_rna"], oplen, pw)
                ss.mark_stage_done("decon_post", {"operator_length": oplen, "pre_whitening": pw})
                st.rerun()
            if ss.is_stage_done("decon_post"):
                st.success("✅ Post-stack deconvolution applied.")
                st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                    st.session_state["stack_rna"], st.session_state["stack_decon_post"], dt,
                    title="Spectrum — Post-Stack Deconvolution"), theme=None, key="poststack_view_chart_1002")
                st.plotly_chart(seismic_plots.plot_before_after(
                    st.session_state["stack_rna"], st.session_state["stack_decon_post"], cdp_list, twt,
                    title_before="Before", title_after="After decon"), theme=None, key="poststack_view_chart_1003")

    with tabs[2]:
        if not ss.require_stage("decon_post"):
            pass
        else:
            st.markdown("Applies a different band-pass in each time zone (higher frequencies "
                         "attenuate faster with depth), blended with a smooth taper across zone "
                         "boundaries.")
            st.dataframe(
                {"Zone start (s)": [z[0] for z in cfg.TVF_ZONES_DEFAULT],
                 "Zone end (s)": [z[1] for z in cfg.TVF_ZONES_DEFAULT],
                 "Low-cut (Hz)": [z[2] for z in cfg.TVF_ZONES_DEFAULT],
                 "High-cut (Hz)": [z[3] for z in cfg.TVF_ZONES_DEFAULT]})
            if st.button("Run time-variant filter", type="primary", key="run_tvf"):
                with st.spinner("Applying time-variant filter..."):
                    st.session_state["stack_tvf"] = filtering.time_variant_filter(
                        st.session_state["stack_decon_post"], dt, cfg.TVF_ZONES_DEFAULT, cfg.TVF_TAPER_S_DEFAULT)
                ss.mark_stage_done("tvf", {"zones": len(cfg.TVF_ZONES_DEFAULT)})
                st.rerun()
            if ss.is_stage_done("tvf"):
                st.success("✅ Time-variant filter applied — ready for migration.")
                st.plotly_chart(seismic_plots.plot_before_after(
                    st.session_state["stack_decon_post"], st.session_state["stack_tvf"], cdp_list, twt,
                    title_before="Before TVF", title_after="After TVF"), theme=None, key="poststack_view_chart_1004")
