"""views/poststack_view.py — Post-Stack Processing.

Random-noise attenuation, a second deconvolution and a time-variant filter
are three independent operators on a stacked section. Each needs only a
section to exist; it reads the *current head* of the post-stack chain, so
they can be applied in any order, in any combination, or not at all. The
usual order (RNA → decon → TVF) is a recommendation: running them in
another order shows a ⚠ and asks for confirmation instead of blocking.
"""

from __future__ import annotations

import streamlit as st

import config as cfg
from modules import stage_ops
from utils import session_state as ss
from utils import workflow_ui as wf
from visualization import seismic_plots, spectra_plots


def _before_label(entry) -> str:
    return wf.input_name(entry)


def render():
    st.header("🧹 Post-Stack Processing")
    st.caption("Random noise attenuation, a second deconvolution pass, and a time-variant filter. "
                "All three are optional and independent: each acts on the current stacked section, "
                "so apply any of them, in any order, or skip them and go straight to migration.")

    if st.session_state["stack_matrix"] is None or not ss.is_stage_done("stack"):
        wf.show_status("rna")          # ⛔ — a stacked section is the one genuine input
        return

    survey = st.session_state["survey"]
    cdp_list = st.session_state["cdp_list"]
    dt, twt = survey.sample_interval, survey.twt

    wf.data_state_caption("section", "Current stacked section")
    wf.stale_banner(["rna", "decon_post", "tvf", "migration", "final_scale"])
    wf.undo_panel(["rna", "decon_post", "tvf"], "undo_poststack")

    tabs = st.tabs(["18. Random Noise Attenuation", "19. Deconvolution", "20. Time-Variant Filter"])

    with tabs[0]:
        chk = wf.show_status("rna")
        if chk.can_run:
            st.markdown("A spatial (across-CDP) median filter suppresses energy that doesn't correlate "
                         "trace-to-trace — random noise — while preserving continuous, dipping events. "
                         "This stands in for industry F-X deconvolution, which needs higher fold than a "
                         "small teaching/demo line typically provides.")
            window = st.slider("Spatial window (traces)", 3, 15, cfg.RNA_SPATIAL_WINDOW_DEFAULT, step=2, key="rna_window")
            if wf.run_button("rna", "Run random noise attenuation", key="run_rna"):
                wf.execute(stage_ops.run_rna, spatial_window=window)
        if ss.is_stage_done("rna"):
            e = ss.entry("rna")
            st.success(f"✅ Random noise attenuation applied to the {_before_label(e)} section.")
            st.plotly_chart(seismic_plots.plot_before_after(
                st.session_state[e.input_key], st.session_state["stack_rna"], cdp_list, twt,
                title_before=_before_label(e), title_after="After RNA"), theme=None, key="poststack_view_chart_1001")

    with tabs[1]:
        chk = wf.show_status("decon_post")
        if chk.can_run:
            st.markdown("A second, milder spiking-deconvolution pass on the stacked section sharpens "
                         "the effective wavelet further now that stacking has improved signal-to-noise.")
            oplen = st.slider("Operator length (samples)", 4, 60, cfg.DECON_AFTER_OPLEN_DEFAULT, key="decon2_oplen")
            pw = st.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_AFTER_PW_DEFAULT,
                                  step=0.005, format="%.3f", key="decon2_pw")
            if wf.run_button("decon_post", "Run post-stack deconvolution", key="run_decon2"):
                wf.execute(stage_ops.run_decon_post, operator_length=oplen, pre_whitening=pw)
        if ss.is_stage_done("decon_post"):
            e = ss.entry("decon_post")
            before = st.session_state[e.input_key]
            st.success(f"✅ Post-stack deconvolution applied to the {_before_label(e)} section.")
            st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                before, st.session_state["stack_decon_post"], dt,
                title="Spectrum — Post-Stack Deconvolution"), theme=None, key="poststack_view_chart_1002")
            st.plotly_chart(seismic_plots.plot_before_after(
                before, st.session_state["stack_decon_post"], cdp_list, twt,
                title_before="Before", title_after="After decon"), theme=None, key="poststack_view_chart_1003")

    with tabs[2]:
        chk = wf.show_status("tvf")
        if chk.can_run:
            st.markdown("Applies a different band-pass in each time zone (higher frequencies "
                         "attenuate faster with depth), blended with a smooth taper across zone "
                         "boundaries.")
            st.dataframe(
                {"Zone start (s)": [z[0] for z in cfg.TVF_ZONES_DEFAULT],
                 "Zone end (s)": [z[1] for z in cfg.TVF_ZONES_DEFAULT],
                 "Low-cut (Hz)": [z[2] for z in cfg.TVF_ZONES_DEFAULT],
                 "High-cut (Hz)": [z[3] for z in cfg.TVF_ZONES_DEFAULT]})
            if wf.run_button("tvf", "Run time-variant filter", key="run_tvf"):
                with st.spinner("Applying time-variant filter..."):
                    wf.execute(stage_ops.run_tvf)
        if ss.is_stage_done("tvf"):
            e = ss.entry("tvf")
            st.success("✅ Time-variant filter applied — ready for migration.")
            st.plotly_chart(seismic_plots.plot_before_after(
                st.session_state[e.input_key], st.session_state["stack_tvf"], cdp_list, twt,
                title_before=f"Before TVF ({_before_label(e)})", title_after="After TVF"), theme=None, key="poststack_view_chart_1004")
