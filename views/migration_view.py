"""views/migration_view.py — Migration & Final Scaling (steps 21-22)."""

from __future__ import annotations

import numpy as np
import streamlit as st

import config as cfg
from modules import migration as migration_mod
from utils import session_state as ss
from visualization import controls, seismic_plots


def render():
    st.header("🗺️ Migration")
    st.caption("Steps 21-22 of the reference pipeline.")

    if not ss.require_stage("tvf"):
        return

    survey = st.session_state["survey"]
    cdp_list = st.session_state["cdp_list"]
    dt, twt = survey.sample_interval, survey.twt
    stack_tvf = st.session_state["stack_tvf"]
    Vfield_2 = st.session_state["Vfield_2"]

    domain = st.radio(
        "Migration domain", ["Post-stack (Kirchhoff time migration)", "Pre-stack (PSTM, on CDP gathers)"],
        key="mig_domain", horizontal=True,
        help="Post-stack migrates the already-stacked (approximately zero-offset) section. "
             "Pre-stack time migration (PSTM) instead sums directly from the offset-preserving "
             "CDP gathers before stacking — report Sec 6.13.1 explains why this matters: a "
             "CMP/DMO stack is not a true zero-offset section wherever CMP != CDP (dipping "
             "reflectors) or lateral velocity varies strongly, so post-stack migration of it is "
             "an approximation that PSTM avoids.")
    aperture_m = st.slider("Migration aperture (m, half-width)", 100.0, 3000.0, cfg.APERTURE_M_DEFAULT, key="mig_aperture")
    aperture_traces = max(1, int(round(aperture_m / survey.group_interval)))
    st.caption(f"≈ {aperture_traces} traces each side at this survey's {survey.group_interval:.1f} m CDP spacing.")

    if domain.startswith("Post-stack"):
        st.markdown("Post-stack Kirchhoff time migration sums each output trace's diffraction "
                     "contributions from neighbouring traces within a lateral aperture, using the "
                     "refined (pass-2) stacking-velocity field as the migration velocity. This "
                     "collapses diffractions and repositions dipping reflectors.")
        if st.button("Run Kirchhoff migration", type="primary", key="run_postmig"):
            x_positions = np.array(cdp_list, dtype=float) * survey.group_interval
            with st.spinner(f"Migrating {len(cdp_list)} traces..."):
                migrated = migration_mod.kirchhoff_migration(stack_tvf, x_positions, twt, Vfield_2, aperture_traces)
            st.session_state["stack_migrated"] = migrated
            st.session_state["pstm_image"] = None
            st.session_state["pstm_fold"] = None
            ss.mark_stage_done("migration", {"domain": "post-stack", "aperture_m": aperture_m})
            st.rerun()
    else:
        st.markdown(
            "**Pre-Stack Time Migration (PSTM)** sums directly from every live pre-stack trace "
            "(offset-preserved, post-regularization) using the double-square-root traveltime "
            "`t = [sqrt(z^2+(xm-h-x0)^2) + sqrt(z^2+(xm+h-x0)^2)] / V` for a diffractor at "
            "`z = V*t0/2`, then fold-normalizes the result — the migration and the stack happen "
            "in one pass, on offset-preserved data, per report Sec 6.13.3. This is far more "
            "compute per output trace than the post-stack version (it sums over every offset in "
            "the aperture, not one stacked trace per CDP), so a **CDP stride** lets you migrate "
            "every Nth CDP directly and linearly fill the rest, trading lateral detail for runtime."
        )
        gathers = st.session_state["gathers"]
        c1, c2 = st.columns(2)
        stride = c1.slider("Output CDP stride (1 = every CDP, slow)", 1, 10, 3, key="pstm_stride")
        input_key = "traces_regularized"
        offsets_key = "offsets_regularized"
        c2.caption(f"Migrating from `{input_key}` (offset-preserved, post-regularization) — "
                    f"**not** the NMO-corrected/stacked data, since PSTM needs the true offsets.")
        needs_reg = any(input_key not in g for g in gathers.values())
        if needs_reg:
            st.error("Run **Data Regularisation** (Velocity, NMO & DMO page) before PSTM — "
                      "it needs every gather on a common offset grid.")
        est_cost = (len(cdp_list) // stride) * (2 * aperture_traces + 1) * 20
        if est_cost > 400_000:
            st.warning(f"⚠️ This configuration does ~{est_cost:,} trace-summation operations — "
                        f"consider a larger stride or smaller aperture if it takes too long.")
        if st.button("Run PSTM", type="primary", key="run_pstm", disabled=needs_reg):
            prog = st.progress(0.0, text="Pre-stack time migrating...")
            with st.spinner(f"Migrating {len(cdp_list)} CDPs (stride {stride})..."):
                migrated, fold = migration_mod.prestack_kirchhoff_time_migration(
                    gathers, cdp_list, input_key, twt, Vfield_2, survey.group_interval,
                    aperture_traces, offsets_key=offsets_key, cdp_stride=stride,
                    progress_cb=lambda f: prog.progress(f, text="Pre-stack time migrating..."))
            prog.progress(1.0)
            st.session_state["pstm_image"] = migrated
            st.session_state["pstm_fold"] = fold
            st.session_state["stack_migrated"] = migrated
            ss.mark_stage_done("migration", {"domain": "pre-stack (PSTM)", "aperture_m": aperture_m,
                                                "cdp_stride": stride})
            st.rerun()

    if ss.is_stage_done("migration"):
        st.success("✅ Migration complete.")
        disp = controls.section_display_controls("mig_display", default_colorscale="Seismic")
        st.plotly_chart(seismic_plots.plot_before_after(
            stack_tvf, st.session_state["stack_migrated"], cdp_list, twt,
            title_before="Pre-migration stack", title_after="Migrated",
            colorscale=disp.colorscale, equalize=disp.equalize),
            theme=None, key="migration_view_chart_1001")
        if st.session_state.get("pstm_fold") is not None:
            st.plotly_chart(seismic_plots.plot_section(
                st.session_state["pstm_fold"], np.array(cdp_list), twt, title="PSTM Fold (QC)",
                colorscale="Viridis", pclip=100.0), theme=None, key="pstm_fold_qc")

        st.divider()
        st.markdown("**Final scaling** — a light final AGC pass restores a balanced, "
                     "presentation-ready amplitude range, then re-applies the top mute (migration's "
                     "diffraction summation can pull energy back into the originally-muted zone).")
        window = st.slider("Final AGC window (samples)", 50, 800, cfg.FINAL_AGC_WINDOW_DEFAULT, key="final_agc_window")
        if st.button("Run final scaling", type="primary"):
            st.session_state["stack_final"] = migration_mod.final_scaling(
                st.session_state["stack_migrated"], dt, window=window, t_top_mute=cfg.MUTE_TTOP_DEFAULT)
            ss.mark_stage_done("final_scale", {"agc_window": window})
            st.rerun()

        if ss.is_stage_done("final_scale"):
            st.success("🎉 Processing complete — this is the final processed section.")
            st.markdown("**Display controls**")
            disp2 = controls.section_display_controls("final_display", default_colorscale="Seismic")
            st.plotly_chart(seismic_plots.plot_section(
                st.session_state["stack_final"], np.array(cdp_list), twt,
                title="Final Processed Section", colorscale=disp2.colorscale, pclip=disp2.pclip,
                gain=disp2.gain, equalize=disp2.equalize, height=650), theme=None, key="migration_view_chart_1002")
