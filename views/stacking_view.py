"""views/stacking_view.py — Fold-Normalized Stack."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules import stage_ops
from utils import session_state as ss
from utils import workflow_ui as wf
from visualization import controls, seismic_plots


def render():
    st.header("📚 Stacking")
    st.caption("Fold-normalized CDP stack of NMO-corrected gathers. Stacking un-flattened gathers "
                "would smear every reflection, so a moveout-corrected input is genuinely required; "
                "which NMO pass supplies it is your choice.")

    chk = wf.show_status("stack")
    if not chk.can_run:
        return

    gathers = st.session_state["gathers"]
    cdp_list = st.session_state["cdp_list"]
    survey = st.session_state["survey"]
    twt = survey.twt
    st.caption(f"Will stack: **{wf.input_name(chk)}** gathers.")
    wf.stale_banner(["stack", "rna", "decon_post", "tvf", "migration", "final_scale"])

    st.markdown("Fold-normalized stacking sums the live (non-muted) NMO-corrected traces at each "
                 "CDP and divides by the *live* fold at each time sample — not the nominal trace "
                 "count — so a stretch-muted near-time zone isn't diluted by zeroed samples.")

    if wf.run_button("stack", "Run stack", key="run_stack"):
        with st.spinner("Stacking..."):
            wf.execute(stage_ops.run_stack)

    if not ss.is_stage_done("stack"):
        return

    stack_matrix = st.session_state["stack_matrix"]
    fold_matrix = st.session_state["fold_matrix"]
    naive_matrix = st.session_state["naive_matrix"]
    st.success(f"✅ Stack complete — stacked the {wf.input_name(ss.entry('stack'))} gathers.")

    fold_per_cdp = np.array([g["fold"].max() for g in gathers.values()])
    st.plotly_chart(seismic_plots.plot_fold_map(fold_per_cdp, cdp_list), theme=None, key="stacking_view_chart_2001")

    st.markdown("**Display controls**")
    disp = controls.section_display_controls("stack_display")
    st.plotly_chart(seismic_plots.plot_section(stack_matrix, np.array(cdp_list), twt,
                     title="Fold-Normalized Stack", colorscale=disp.colorscale, pclip=disp.pclip,
                     gain=disp.gain, height=600), theme=None, key="stacking_view_chart_2002")

    with st.expander("Compare against a naive (un-normalized) mean stack"):
        st.plotly_chart(seismic_plots.plot_before_after(
            naive_matrix, stack_matrix, np.array(cdp_list), twt,
            title_before="Naive mean stack", title_after="Fold-normalized stack"), theme=None, key="stacking_view_chart_1")
