"""views/migration_view.py — Kirchhoff Migration & Final Scaling (steps 21-22)."""

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

    st.markdown("Post-stack Kirchhoff time migration sums each output trace's diffraction "
                 "contributions from neighbouring traces within a lateral aperture, using the "
                 "refined (pass-2) stacking-velocity field as the migration velocity. This "
                 "collapses diffractions and repositions dipping reflectors.")
    aperture_m = st.slider("Migration aperture (m, half-width)", 100.0, 3000.0, cfg.APERTURE_M_DEFAULT, key="mig_aperture")
    aperture_traces = max(1, int(round(aperture_m / survey.group_interval)))
    st.caption(f"≈ {aperture_traces} traces each side at this survey's {survey.group_interval:.1f} m CDP spacing.")

    if st.button("Run Kirchhoff migration", type="primary"):
        x_positions = np.array(cdp_list, dtype=float) * survey.group_interval
        with st.spinner(f"Migrating {len(cdp_list)} traces..."):
            migrated = migration_mod.kirchhoff_migration(stack_tvf, x_positions, twt, Vfield_2, aperture_traces)
        st.session_state["stack_migrated"] = migrated
        ss.mark_stage_done("migration", {"aperture_m": aperture_m})
        st.rerun()

    if ss.is_stage_done("migration"):
        st.success("✅ Migration complete.")
        st.plotly_chart(seismic_plots.plot_before_after(
            stack_tvf, st.session_state["stack_migrated"], cdp_list, twt,
            title_before="Pre-migration stack", title_after="Migrated"), theme=None, key="migration_view_chart_1001")

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
            disp = controls.section_display_controls("final_display")
            st.plotly_chart(seismic_plots.plot_section(
                st.session_state["stack_final"], np.array(cdp_list), twt,
                title="Final Processed Section", colorscale=disp.colorscale, pclip=disp.pclip,
                gain=disp.gain, height=650), theme=None, key="migration_view_chart_1002")
