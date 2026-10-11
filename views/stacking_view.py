"""views/stacking_view.py — Fold-Normalized Stack."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules import stage_ops
from modules.dependencies import state_name
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

    survey = st.session_state["survey"]
    twt = survey.twt
    st.caption(f"Will stack: **{wf.input_name(chk)}** gathers.")
    wf.stale_banner(wf.stale_for_ops(["stack", "rna", "decon_post", "tvf", "migration", "final_scale"]))

    st.markdown("Fold-normalized stacking sums the live (non-muted) NMO-corrected traces at each "
                 "CDP and divides by the *live* fold at each time sample — not the nominal trace "
                 "count — so a stretch-muted near-time zone isn't diluted by zeroed samples.")

    if wf.run_button("stack", "Run stack", key="run_stack"):
        with st.spinner("Stacking..."):
            wf.execute(stage_ops.run_stack)

    wf.states_panel(["section"], "states_section", "🗂 Stacked / post-stack data states · undo · redo")

    e = wf.result("stack")
    if e is None:
        return

    wf.result_note(e)
    cdp_list = ss.cdp_list(e.state)
    stack_matrix = ss.blob(e.state, "section")
    fold_matrix = ss.blob(e.state, "fold")
    naive_matrix = ss.blob(e.state, "naive")
    st.success(f"✅ Stack complete — **{e.name}**, stacked the {state_name(e.src)} gathers.")

    fold_per_cdp = fold_matrix.max(axis=1)   # live fold per CDP (rows of the fold matrix)
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
