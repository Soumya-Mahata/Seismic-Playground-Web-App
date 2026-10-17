"""views/migration_view.py — Migration & Final Scaling.

Two migration implementations with genuinely different inputs:

* Post-stack Kirchhoff  — needs a stacked section + a velocity field.
* Pre-stack Kirchhoff (PSTM) — needs pre-stack, offset-preserving CDP
  gathers + a velocity field, and performs migration *and* stack in one
  pass, so no stack is required.

Everything else (post-stack random-noise attenuation / deconvolution /
time-variant filter, data regularisation, the second velocity pass) is a
recommendation, shown as a ⚠ and never as a block. Final scaling is its
own step and can be applied to any section, migrated or not.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import config as cfg
from modules import stage_ops
from modules.dependencies import state_name
from utils import session_state as ss
from utils import workflow_ui as wf
from visualization import controls, seismic_plots

_POST, _PRE = "Post-stack (Kirchhoff time migration)", "Pre-stack (PSTM, on CDP gathers)"


def _comparison_section():
    """The un-migrated section to compare a migration against, if one
    exists (it is optional: PSTM does not need a stack)."""
    head = ss.current("section")
    return (ss.section(head), state_name(head)) if head is not None else (None, None)


def _migration_block(survey, cdp_list):
    domain = st.radio(
        "Migration domain", [_POST, _PRE], key="mig_domain", horizontal=True,
        help="Post-stack migrates the already-stacked (approximately zero-offset) section. "
             "Pre-stack time migration (PSTM) instead sums directly from the offset-preserving "
             "CDP gathers before stacking — this matters because a "
             "CMP/DMO stack is not a true zero-offset section wherever CMP != CDP (dipping "
             "reflectors) or lateral velocity varies strongly, so post-stack migration of it is "
             "an approximation that PSTM avoids.")
    variant = "poststack" if domain == _POST else "pstm"
    chk = wf.show_status("migration", variant)
    aperture_m = st.slider("Migration aperture (m, half-width)", 100.0, 3000.0, cfg.APERTURE_M_DEFAULT, key="mig_aperture")
    aperture_traces = max(1, int(round(aperture_m / survey.group_interval))) if survey else 1
    if survey:
        st.caption(f"≈ {aperture_traces} traces each side at this survey's {survey.group_interval:.1f} m CDP spacing.")

    if variant == "poststack":
        st.markdown("Post-stack Kirchhoff time migration sums each output trace's diffraction "
                     "contributions from neighbouring traces within a lateral aperture, using the "
                     "best stacking-velocity field available (pass 2, else pass 1) as the migration "
                     "velocity. This collapses diffractions and repositions dipping reflectors.")
        if chk.can_run:
            st.caption(f"Will migrate: **{wf.input_name(chk)}** section · velocity: "
                        f"**{wf.input_name(chk, 'velocity')}** field.")
            if wf.run_button("migration", "Run Kirchhoff migration", key="run_postmig", variant="poststack"):
                with st.spinner(f"Migrating {len(cdp_list)} traces..."):
                    wf.execute(stage_ops.run_migration_poststack, aperture_m=aperture_m)
    else:
        st.markdown(
            "**Pre-Stack Time Migration (PSTM)** sums directly from every live pre-stack trace "
            "(offset-preserved) using the double-square-root traveltime "
            "`t = [sqrt(z^2+(xm-h-x0)^2) + sqrt(z^2+(xm+h-x0)^2)] / V` for a diffractor at "
            "`z = V*t0/2`, then fold-normalizes the result — the migration and the stack happen "
            "in one pass, on offset-preserved data, so **no stack is needed**. This is far more "
            "compute per output trace than the post-stack version (it sums over every offset in "
            "the aperture, not one stacked trace per CDP), so a **CDP stride** lets you migrate "
            "every Nth CDP directly and linearly fill the rest, trading lateral detail for runtime."
        )
        if chk.can_run:
            c1, c2 = st.columns(2)
            stride = c1.slider("Output CDP stride (1 = every CDP, slow)", 1, 10, cfg.PSTM_CDP_STRIDE_DEFAULT,
                               key="pstm_stride")
            c2.caption(f"Migrating from the {wf.input_name(chk)} gathers "
                        f"(offset-preserved), velocity: **{wf.input_name(chk, 'velocity')}** field. "
                        f"**Not** the NMO-corrected/stacked data, since PSTM needs the true offsets.")
            est_cost = (len(cdp_list) // stride) * (2 * aperture_traces + 1) * 20
            if est_cost > 400_000:
                st.warning(f"⚠️ This configuration does ~{est_cost:,} trace-summation operations — "
                            f"consider a larger stride or smaller aperture if it takes too long.")
            if wf.run_button("migration", "Run PSTM", key="run_pstm", variant="pstm"):
                prog = st.progress(0.0, text="Pre-stack time migrating...")
                with st.spinner(f"Migrating {len(cdp_list)} CDPs (stride {stride})..."):
                    wf.execute(stage_ops.run_migration_pstm, aperture_m=aperture_m, cdp_stride=stride,
                               progress_cb=lambda f: prog.progress(f, text="Pre-stack time migrating..."))

    e = wf.result("migration")
    if e is None:
        return
    wf.result_note(e)
    cdp_list = ss.cdp_list(e.state)
    migrated = ss.section(e.state)
    ran = "pre-stack (PSTM)" if e.state.variant == "pstm" else "post-stack"
    st.success(f"✅ Migration complete ({ran}) — **{e.name}**.")
    disp = controls.section_display_controls("mig_display", default_colorscale="Seismic")
    twt = survey.twt
    before, before_name = (ss.section(e.src), state_name(e.src)) \
        if e.state.variant != "pstm" else _comparison_section()
    if before is not None:
        st.plotly_chart(seismic_plots.plot_before_after(
            before, migrated, cdp_list, twt,
            title_before=f"Pre-migration ({before_name})", title_after="Migrated",
            colorscale=disp.colorscale, equalize=disp.equalize),
            theme=None, key="migration_view_chart_1001")
    else:
        st.plotly_chart(seismic_plots.plot_section(
            migrated, np.array(cdp_list), twt, title="Migrated section",
            colorscale=disp.colorscale, pclip=disp.pclip, gain=disp.gain, equalize=disp.equalize,
            height=600), theme=None, key="migration_view_chart_1001")
        st.caption("No stacked section exists to compare against — the pre-stack migration did not need one.")
    if ss.blob(e.state, "fold") is not None:
        st.plotly_chart(seismic_plots.plot_section(
            ss.blob(e.state, "fold"), np.array(cdp_list), twt, title="PSTM Fold (QC)",
            colorscale="Viridis", pclip=100.0), theme=None, key="pstm_fold_qc")


def _final_scaling_block(survey, cdp_list):
    st.divider()
    st.subheader("Final scaling")
    st.markdown("A light final AGC pass restores a balanced, presentation-ready amplitude range, then "
                 "re-applies the top mute (migration's diffraction summation can pull energy back "
                 "into the originally-muted zone). It acts on the migrated section when there is one, "
                 "otherwise on the current stacked section.")
    chk = wf.show_status("final_scale")
    if chk.can_run:
        st.caption(f"Will scale: **{wf.input_name(chk)}**.")
        window = st.slider("Final AGC window (samples)", 50, 800, cfg.FINAL_AGC_WINDOW_DEFAULT, key="final_agc_window")
        if wf.run_button("final_scale", "Run final scaling", key="run_final_scale"):
            wf.execute(stage_ops.run_final_scale, window=window)

    fe = wf.result("final_scale")
    if fe is not None:
        wf.result_note(fe)
        cdp_list = ss.cdp_list(fe.state)
        st.success(f"🎉 Final scaling applied — **{fe.name}** (from {state_name(fe.src)}) is the final "
                    f"processed section.")
        st.markdown("**Display controls**")
        disp2 = controls.section_display_controls("final_display", default_colorscale="Seismic")
        st.plotly_chart(seismic_plots.plot_section(
            ss.section(fe.state), np.array(cdp_list), survey.twt,
            title="Final Processed Section", colorscale=disp2.colorscale, pclip=disp2.pclip,
            gain=disp2.gain, equalize=disp2.equalize, height=650), theme=None, key="migration_view_chart_1002")


def render():
    st.header("🗺️ Migration")

    survey = st.session_state["survey"]
    if survey is None or st.session_state["gathers"] is None:
        wf.show_status("migration", "poststack")     # ⛔ names the missing CDP gathers / velocity
        return
    cdp_list = st.session_state["cdp_list"]
    wf.data_state_caption("section", "Current stacked section")
    wf.stale_banner(wf.stale_for_ops(["migration", "final_scale"]))
    wf.states_panel(["image", "final"], "states_image", "🗂 Migrated / final data states · undo · redo")

    _migration_block(survey, cdp_list)
    _final_scaling_block(survey, cdp_list)
