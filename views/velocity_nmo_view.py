"""
views/velocity_nmo_view.py — Velocity Analysis, NMO, Statics, DMO (10-16)
=============================================================================
The most interactive part of the app: automatic velocity picking with
per-analysis-CDP reliability flagging, manual pick correction/exclusion,
and the two NMO passes / residual statics / DMO sandwich between them.

Every tab states its own readiness (✓ / ⚠ / ⛔) from the dependency
engine instead of waiting for the previous tab to have been run. What a
tab genuinely needs is shown in its ⛔ message — e.g. NMO needs a
velocity field; velocity analysis needs CDP gathers — and anything that
is merely the usual order (statics before DMO, regularisation before the
second velocity pass, ...) is a ⚠ the person can override.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from modules import regularization as reg_mod
from modules import stage_ops
from modules import velocity as vel_mod
from utils import session_state as ss
from utils import workflow_ui as wf
from utils.click_picking import numeric_point_form
from visualization import seismic_plots, velocity_plots


def _velocity_pass_ui(*, pass_label: str, stage_id: str, field_key: str,
                       analysis_key: str, results_key: str, overrides_key: str, excluded_key: str):
    """One semblance velocity-analysis pass. What it reads (and with which
    offsets) is whatever the dependency engine resolved for this stage —
    recorded in the history entry — rather than a key the page hard-codes."""
    chk = wf.show_status(stage_id)
    if not chk.can_run and not ss.is_stage_done(stage_id):
        return False
    survey = st.session_state["survey"]
    gathers = st.session_state["gathers"]
    cdp_list = st.session_state["cdp_list"]
    dt, ns, twt = survey.sample_interval, survey.num_samples, survey.twt
    if chk.can_run:
        st.caption(f"Will analyse: **{wf.input_name(chk)}** gathers.")

    with st.expander("Parameters", expanded=st.session_state[field_key] is None):
        c1, c2, c3 = st.columns(3)
        vmin = c1.number_input("Min trial velocity (m/s)", 800.0, 6000.0, cfg.VELOCITY_MIN_DEFAULT, key=f"{stage_id}_vmin")
        vmax = c2.number_input("Max trial velocity (m/s)", 1000.0, 8000.0, cfg.VELOCITY_MAX_DEFAULT, key=f"{stage_id}_vmax")
        n_analysis = c3.number_input("Analysis CDPs", 3, 25, cfg.N_ANALYSIS_CDPS_DEFAULT, key=f"{stage_id}_nanalysis")
        c1, c2 = st.columns(2)
        min_offset = c1.number_input("Min far offset required (m)", 100.0, 3000.0, cfg.MIN_ABS_OFFSET_DEFAULT, key=f"{stage_id}_minoff")
        window_len = c2.number_input("Semblance half-window (samples)", 1, 30, cfg.SEMBLANCE_WINDOW_LEN_DEFAULT, key=f"{stage_id}_winlen")
        with st.expander("Advanced auto-picker settings"):
            c1, c2, c3, c4 = st.columns(4)
            min_time = c1.number_input("Ignore before (s)", 0.0, 0.5,
                                        cfg.AUTOPICK_MIN_TIME_DEFAULT, step=0.01, key=f"{stage_id}_mintime")
            search_w = c2.number_input("Search half-width (m/s)", 100.0, 2000.0,
                                        cfg.AUTOPICK_SEARCH_HALF_WIDTH_MS_DEFAULT, key=f"{stage_id}_searchw")
            min_conf = c3.number_input("Min confidence to move", 0.01, 0.9,
                                        cfg.AUTOPICK_MIN_CONFIDENCE_DEFAULT, key=f"{stage_id}_minconf")
            smooth = c4.number_input("Smoothing (samples)", 1, 40, cfg.AUTOPICK_SMOOTH_SIGMA_DEFAULT, key=f"{stage_id}_smooth")
            reliability_threshold = st.slider(
                "Outlier tolerance (max relative deviation from neighbours)", 0.05, 0.5,
                cfg.RELIABILITY_THRESHOLD_DEFAULT, step=0.05, key=f"{stage_id}_reltol",
                help="Analysis CDPs whose pick deviates from the lateral median trend by more "
                     "than this fraction are flagged and excluded from the field by default.")

    if wf.run_button(stage_id, f"Run {pass_label}", key=f"run_{stage_id}"):
        prog = st.progress(0.0, text="Picking velocities...")
        wf.execute(stage_ops.run_velocity, stage_id,
                   vmin=vmin, vmax=vmax, n_analysis=int(n_analysis), min_offset=min_offset,
                   window_len=int(window_len), min_time=min_time, search_w=search_w,
                   min_conf=min_conf, smooth=int(smooth), reliability_threshold=reliability_threshold,
                   progress_cb=lambda f: prog.progress(f, text="Picking velocities..."))

    if not ss.is_stage_done(stage_id):
        return False

    entry = ss.entry(stage_id)
    input_key, offsets_key = entry.input_key, entry.offsets_key
    st.caption(f"Analysed: **{wf.input_name(entry)}** gathers.")
    Vfield = st.session_state[field_key]
    analysis = st.session_state[analysis_key]
    results: Dict[int, vel_mod.VelocityPickResult] = st.session_state[results_key]
    velocities = st.session_state["velocities"]

    n_reliable = sum(r.is_reliable for r in results.values())
    if n_reliable < len(results):
        st.warning(f"⚠️ {len(results) - n_reliable}/{len(results)} analysis-CDP picks were flagged as "
                    f"laterally inconsistent with their neighbours and excluded from the field below. "
                    f"Review them in **Manual pick review** — a flagged pick isn't necessarily wrong, "
                    f"but it disagreed enough with its neighbours to warrant a look.")
    st.plotly_chart(velocity_plots.plot_reliability_summary(results), theme=None, key=f"{stage_id}_reliability_chart")

    st.markdown("**Manual pick review / correction**")
    cdp_pick = st.selectbox("Analysis CDP", analysis, key=f"{stage_id}_review_cdp",
                             format_func=lambda c: f"CDP {c}" + (" ⚠️" if not results[c].is_reliable else " ✅"))
    r = results[cdp_pick]
    g = gathers[cdp_pick]
    semblance = g[f"semblance_{stage_id}"]

    manual_pts = st.session_state.get(f"{stage_id}_manual_pts_{cdp_pick}", [])
    fig = velocity_plots.plot_semblance_panel(
        semblance, velocities, twt, picked_velocity=r.velocity, manual_points=manual_pts,
        title=f"Semblance — CDP {cdp_pick} (reliability {r.reliability:.2f})")
    st.plotly_chart(fig, theme=None, key=f"{stage_id}_semblance_click_{cdp_pick}")
    st.caption("🔍 Zoom into the panel above to read off a (velocity, time) coordinate on a coherent "
                "trend, then add it below. (A heatmap-backed plot like this can't deliver reliable "
                "click events in Streamlit, so points are entered here instead of by clicking on it.)")
    new_pt = numeric_point_form(
        f"{stage_id}_vpt_{cdp_pick}", "Velocity (m/s)", "Time (s)",
        x_value=float(np.median(velocities)), y_value=float(twt[len(twt) // 2]),
        x_min=float(velocities.min()), x_max=float(velocities.max()),
        y_min=0.0, y_max=float(twt[-1]), x_step=10.0, y_step=0.01)
    if new_pt is not None:
        v_click, t_click = new_pt
        manual_pts = manual_pts + [(round(t_click, 4), round(v_click, 1))]
        st.session_state[f"{stage_id}_manual_pts_{cdp_pick}"] = manual_pts
        rev_key = f"{stage_id}_manual_rev_{cdp_pick}"
        st.session_state[rev_key] = st.session_state.get(rev_key, 0) + 1
        st.rerun()

    with st.expander("Constant Velocity Stack (CVS) — a complementary, amplitude-domain check"):
        st.markdown(
            "Stacks a small local window of CDPs after NMO-correcting with each trial (constant) "
            "velocity in turn — the velocity that stacks with the highest power at a given time is "
            "a good candidate stacking velocity, independent of the semblance calculation above."
        )
        half_win = st.slider("CDP window half-width", 1, 15, 5, key=f"{stage_id}_cvs_halfwin_{cdp_pick}")
        if st.button("Compute CVS panel", key=f"{stage_id}_cvs_btn_{cdp_pick}"):
            with st.spinner("Stacking at each trial velocity..."):
                panel, power = vel_mod.constant_velocity_stack_panel(
                    gathers, cdp_list, input_key, cdp_pick, half_win, velocities, dt, ns,
                    offsets_key=offsets_key)
            st.session_state[f"{stage_id}_cvs_{cdp_pick}"] = (panel, power)
        cvs_result = st.session_state.get(f"{stage_id}_cvs_{cdp_pick}")
        if cvs_result is not None:
            panel, power = cvs_result
            st.plotly_chart(velocity_plots.plot_cvs_panel(panel, power, velocities, twt,
                             title=f"CVS — CDP {cdp_pick} ± {half_win} CDPs"),
                             theme=None, key=f"{stage_id}_cvs_chart_{cdp_pick}")

    c1, c2 = st.columns(2)
    with c1:
        excluded = st.checkbox("Exclude this CDP from the field", value=cdp_pick in st.session_state[excluded_key],
                                key=f"{stage_id}_exclude_{cdp_pick}")
    with c2:
        st.caption("Or add manual control points (time, velocity) below to override the auto-pick entirely.")

    edited = st.data_editor(
        pd.DataFrame(manual_pts, columns=["time_s", "velocity_m_s"]) if manual_pts
        else pd.DataFrame(columns=["time_s", "velocity_m_s"]),
        num_rows="dynamic", key=f"{stage_id}_editor_{cdp_pick}_{st.session_state.get(f'{stage_id}_manual_rev_{cdp_pick}', 0)}")

    b1, b2 = st.columns(2)
    if b1.button("Apply manual points to this CDP", key=f"{stage_id}_apply_{cdp_pick}"):
        pts = [(float(row.time_s), float(row.velocity_m_s)) for row in edited.itertuples()
               if pd.notna(row.time_s) and pd.notna(row.velocity_m_s)]
        st.session_state[f"{stage_id}_manual_pts_{cdp_pick}"] = pts
        if len(pts) >= 2:
            pts_sorted = sorted(pts)
            t_ctrl = [p[0] for p in pts_sorted]
            v_ctrl = [p[1] for p in pts_sorted]
            v_full = np.interp(twt, t_ctrl, v_ctrl)
            st.session_state[overrides_key][cdp_pick] = v_full
            st.session_state[excluded_key].discard(cdp_pick)
        elif cdp_pick in st.session_state[overrides_key]:
            del st.session_state[overrides_key][cdp_pick]
        st.rerun()
    if b2.button("Update exclusion & rebuild field", key=f"{stage_id}_rebuild_{cdp_pick}"):
        if excluded:
            st.session_state[excluded_key].add(cdp_pick)
        else:
            st.session_state[excluded_key].discard(cdp_pick)
        st.rerun()

    st.plotly_chart(velocity_plots.plot_velocity_field(Vfield, cdp_list, twt,
                     title=f"Stacking-Velocity Field — {pass_label}"), theme=None, key=f"{stage_id}_velocity_field_chart")
    return True


def _example_cdp(cdp_list, key):
    return st.selectbox("Example CDP", cdp_list, index=len(cdp_list) // 2, key=key)


def render():
    st.header("🎯 Velocity Analysis, NMO & DMO")
    st.caption("Velocity analysis (auto-pick, manual click-pick, or CVS), NMO, residual statics, "
                "and DMO — operating on the CDP gathers built in CDP Sort & Binning. Each tab shows "
                "whether it can run now; you do not have to run the tabs in order — only the inputs "
                "a tab genuinely needs must exist.")

    if st.session_state["gathers"] is None:
        wf.show_status("vel1")          # ⛔ names the missing CDP gathers and where to make them
        return
    survey = st.session_state["survey"]
    gathers = st.session_state["gathers"]
    cdp_list = st.session_state["cdp_list"]
    twt = survey.twt

    wf.data_state_caption("cdp", "Current pre-NMO CDP gathers")
    wf.stale_banner(["vel1", "nmo1", "statics", "dmo", "inv_nmo", "data_reg", "vel2", "nmo_final"])
    wf.undo_panel(["vel1", "nmo1", "statics", "dmo", "inv_nmo", "data_reg", "vel2", "nmo_final"],
                  "undo_cdp_domain")

    tabs = st.tabs(["Velocity Pass 1", "NMO Pass 1", "Residual Statics",
                    "DMO Correction", "Inverse NMO", "Data Regularisation",
                    "RMS Velocity Analysis (Pass 2)", "Final NMO"])

    with tabs[0]:
        _velocity_pass_ui(pass_label="Velocity Analysis — Pass 1 (approximate)", stage_id="vel1",
                           field_key="Vfield_1", analysis_key="analysis_1",
                           results_key="vel_results_1", overrides_key="manual_overrides_1",
                           excluded_key="excluded_cdps_1")

    with tabs[1]:
        chk = wf.show_status("nmo1")
        if chk.can_run:
            st.markdown("Applies the pass-1 stacking-velocity field, flattening reflections so they "
                         "can be identified/stacked coherently. A stretch mute removes samples where "
                         "NMO would distort the wavelet too much to be usable.")
            stretch = st.slider("Stretch mute (fractional)", 0.05, 1.0, cfg.NMO_STRETCH_MUTE_DEFAULT, key="nmo1_stretch")
            if wf.run_button("nmo1", "Run NMO — Pass 1", key="run_nmo1"):
                wf.execute(stage_ops.run_nmo1, stretch_mute=stretch)
        if ss.is_stage_done("nmo1"):
            e = ss.entry("nmo1")
            st.success(f"✅ NMO pass 1 applied to {wf.input_name(e)} gathers.")
            cdp = _example_cdp(cdp_list, "nmo1_cdp")
            g = gathers[cdp]
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.input_key], g["traces_nmo1"], np.sort(g[e.offsets_key]), twt,
                title_before="Before NMO", title_after="After NMO — flattened", x_title="Offset (m)"), theme=None, key="velocity_nmo_view_chart_2")

    with tabs[2]:
        chk = wf.show_status("statics")
        if chk.can_run:
            st.markdown("Surface-consistent residual statics: cross-correlates each NMO-flattened "
                         "trace against its CDP's fold-normalized pilot trace, then solves for "
                         "per-shot and per-receiver static shifts by least squares.")
            max_shift = st.slider("Max static shift search (ms)", 2.0, 100.0, cfg.MAX_STATIC_SHIFT_S_DEFAULT * 1000,
                                   key="statics_maxshift") / 1000.0
            if wf.run_button("statics", "Solve & apply residual statics", key="run_statics"):
                with st.spinner("Solving surface-consistent statics..."):
                    wf.execute(stage_ops.run_statics, max_shift_s=max_shift)
        if ss.is_stage_done("statics"):
            e = ss.entry("statics")
            diag = st.session_state["static_diag"]
            st.success("✅ Residual statics solved and applied.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Shots solved", diag["n_shots"])
            c2.metric("Receivers solved", diag["n_receivers"])
            c3.metric("RMS residual shift", f"{diag['rms_observed_shift_ms']:.2f} ms")
            cdp = _example_cdp(cdp_list, "statics_cdp")
            g = gathers[cdp]
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.input_key], g["traces_statics"], np.sort(g[e.offsets_key]), twt,
                title_before="Before statics", title_after="After statics", x_title="Offset (m)"), theme=None, key="velocity_nmo_view_chart_3")

    with tabs[3]:
        chk = wf.show_status("dmo")
        if chk.can_run:
            st.markdown("**Simplified DMO** (documented limitation: a common-offset time-stretch "
                         "approximation, not a full log-stretch/Kirchhoff DMO operator — see README). "
                         "It NMO-corrects its own input with the pass-1 velocities first, so it does not "
                         "need NMO Pass 1 to have been run.")
            stretch_dmo = st.slider("Stretch mute (fractional)", 0.05, 1.0,
                                     float(st.session_state.get("nmo1_stretch", cfg.NMO_STRETCH_MUTE_DEFAULT)),
                                     key="dmo_stretch")
            if wf.run_button("dmo", "Run DMO correction", key="run_dmo"):
                wf.execute(stage_ops.run_dmo, stretch_mute=stretch_dmo)
        if ss.is_stage_done("dmo"):
            e = ss.entry("dmo")
            st.success(f"✅ DMO correction applied to {wf.input_name(e)} gathers.")
            cdp = _example_cdp(cdp_list, "dmo_cdp")
            g = gathers[cdp]
            st.plotly_chart(seismic_plots.plot_before_after(
                g["traces_nmo1_statics"], g["traces_dmo"], np.sort(g[e.offsets_key]), twt,
                title_before="NMO'd (pass 1 velocities)", title_after="After DMO", x_title="Offset (m)"), theme=None, key="velocity_nmo_view_chart_4")

    with tabs[4]:
        chk = wf.show_status("inv_nmo")
        if chk.can_run:
            st.markdown("Undoes the pass-1 NMO on the DMO-corrected gathers, so a refined velocity "
                         "analysis (next step) starts from data with the DMO improvement folded in "
                         "but without the pass-1 (approximate) velocity field's bias.")
            if wf.run_button("inv_nmo", "Run inverse NMO", key="run_invnmo"):
                wf.execute(stage_ops.run_inverse_nmo)
        if ss.is_stage_done("inv_nmo"):
            e = ss.entry("inv_nmo")
            st.success("✅ Inverse NMO applied.")
            cdp = _example_cdp(cdp_list, "invnmo_cdp")
            g = gathers[cdp]
            st.plotly_chart(seismic_plots.plot_gather(g["traces_inv_nmo"], g[e.offsets_key], twt,
                             title=f"CDP {cdp} — after inverse NMO"), theme=None, key="velocity_nmo_view_chart_2003")

    with tabs[5]:
        chk = wf.show_status("data_reg")
        if chk.can_run:
            st.markdown(
                "**Data Regularisation** interpolates every CDP gather's "
                "irregularly sampled offsets onto one shared, regularly spaced offset grid for the "
                "whole survey. It is *recommended* before a refined velocity analysis, pre-stack "
                "migration and stacking — real fold is rarely evenly spaced across offset (gaps, "
                "dropped channels, split-spread irregularities), which biases a semblance velocity "
                "scan and any migration operator that sums over offset — but it is optional: those "
                "steps can run without it."
            )
            n_offsets = st.slider("Regularized offsets per CDP", 11, 61, 31, step=2, key="reg_noffsets")
            if wf.run_button("data_reg", "Run data regularisation", key="run_datareg"):
                with st.spinner("Interpolating gathers onto a regular offset grid..."):
                    wf.execute(stage_ops.run_data_reg, n_offsets=int(n_offsets))
        if ss.is_stage_done("data_reg"):
            e = ss.entry("data_reg")
            # entry.offsets_key is the regularised (output) grid; the "before" panel needs the
            # offsets of the irregular gathers this step read.
            in_off_key = stage_ops.data_keys(ss.workflow(), e.consumed["data"])[1]
            before_fold, after_fold = reg_mod.fold_before_after(gathers, offsets_key=in_off_key)
            st.success(f"✅ Data regularised — fold before: {before_fold.min()}-{before_fold.max()} "
                        f"(irregular offsets) → after: {after_fold.min()}-{after_fold.max()} live "
                        f"traces on a common {len(st.session_state['regular_offsets'])}-offset grid.")
            cdp = _example_cdp(cdp_list, "datareg_cdp")
            g = gathers[cdp]
            order = np.argsort(g[in_off_key])
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.input_key][order], g["traces_regularized"],
                np.sort(g[in_off_key]), twt, x_values_after=g["offsets_regularized"],
                title_before="Before (irregular offsets)", title_after="After (regularized offsets)",
                x_title="Offset (m)"), theme=None, key="datareg_before_after")

    with tabs[6]:
        st.markdown("Refined semblance velocity analysis. It is normally run on regularised gathers so "
                     "the scan sees a consistent offset sampling across every CDP; if they have not "
                     "been regularised you will be told, and may continue.")
        _velocity_pass_ui(pass_label="RMS Velocity Analysis (Pass 2)", stage_id="vel2",
                           field_key="Vfield_2", analysis_key="analysis_2",
                           results_key="vel_results_2", overrides_key="manual_overrides_2",
                           excluded_key="excluded_cdps_2")

    with tabs[7]:
        chk = wf.show_status("nmo_final")
        if chk.can_run:
            st.markdown("Applies the best stacking-velocity field available (pass 2, else pass 1) to "
                         "the current pre-NMO gathers — this is the gather that will be stacked next.")
            stretch2 = st.slider("Stretch mute (fractional)", 0.05, 1.0, cfg.NMO_STRETCH_MUTE_DEFAULT, key="nmofinal_stretch")
            if wf.run_button("nmo_final", "Run final NMO", key="run_nmofinal"):
                wf.execute(stage_ops.run_nmo_final, stretch_mute=stretch2)
        if ss.is_stage_done("nmo_final"):
            e = ss.entry("nmo_final")
            st.success(f"✅ Final NMO applied to {wf.input_name(e)} gathers — ready to stack.")
            cdp = _example_cdp(cdp_list, "nmofinal_cdp")
            g = gathers[cdp]
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.input_key], g["traces_nmo_final"], g[e.offsets_key], twt,
                title_before="Before final NMO", title_after="After final NMO", x_title="Offset (m)"), theme=None, key="velocity_nmo_view_chart_5")
