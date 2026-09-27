"""
views/preprocessing_view.py — Pre-Stack Conditioning (shot domain)
======================================================================
Band-pass -> AGC -> Trace edit/kill/de-spike -> Mute -> Balance ->
F-K filter (auto or click-drawn polygon) -> Multiple suppression
(predictive deconvolution) -> Deconvolution, applied to every SHOT
gather (the domain the data was actually recorded in), each with a
"Run" button and before/after QC on a representative shot gather.

CDP sort/binning — re-grouping this conditioned data by CDP — happens
afterward, in `views/cdp_sort_view.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from modules import filtering, preprocessing
from utils import session_state as ss
from utils.click_picking import numeric_point_form
from visualization import controls, seismic_plots, spectra_plots


def _example_shot_selector(shot_list, key):
    return st.selectbox("Example shot for QC", shot_list, index=len(shot_list) // 2, key=key)


def render():
    st.header("⚙️ Pre-Stack Conditioning")
    st.caption("Applied to shot gathers — the domain the data was actually recorded in. "
                "CDP sort/binning happens afterward, right before velocity analysis.")

    if not ss.require_stage("shot_gathers"):
        return
    shot_gathers = st.session_state["shot_gathers"]
    shot_list = list(shot_gathers.keys())
    survey = st.session_state["survey"]
    dt = survey.sample_interval
    twt = survey.twt

    n_traces_in_shots = sum(g["traces"].shape[0] for g in shot_gathers.values())
    with st.expander(f"Trace-count check: {n_traces_in_shots:,} traces across {len(shot_list)} shots "
                      f"(survey has {survey.num_traces:,})", expanded=False):
        st.markdown(
            "The data was already in the shot domain, so every step in this page runs **in place** "
            "on each shot's `(channels, samples)` array — no step here concatenates, duplicates, or "
            "re-sorts traces, so this total should stay exactly the same through every tab below. "
            "If a shot has a split-spread layout (receivers on both sides of the source), its channel "
            "count is roughly `2 x channels-per-side` by design (see the survey-generation settings) "
            "— that is a real split-spread geometry, not a duplication bug."
        )
        counts = pd.DataFrame({"stage": ["Survey (all traces)", "Sum across shot gathers"],
                                "traces": [survey.num_traces, n_traces_in_shots]})
        st.dataframe(counts, hide_index=True)
        if n_traces_in_shots != survey.num_traces:
            st.warning("⚠️ These two totals differ — some traces were dropped when building shot "
                        "gathers (e.g. a minimum-fold filter) or the shot-grouping key doesn't cover "
                        "every trace. This is worth checking against the survey's FieldRecord header.")

    tabs = st.tabs(["1. Band-Pass Filter", "2. Amplitude Recovery", "3. Trace Edit/Kill/De-spike",
                    "4. Top Mute", "5. Trace Balancing", "6. F-K Filter", "7. Multiple Suppression",
                    "8. Deconvolution"])

    # ---- 1) Band-pass filter ----------------------------------------
    with tabs[0]:
        st.markdown("A trapezoidal frequency-domain band-pass, smoothed to avoid ringing, "
                     "removes the low-frequency ground-roll/DC drift and high-frequency noise "
                     "outside the useful reflection bandwidth — applied first, before any "
                     "amplitude-sensitive step, per standard practice.")
        nyq = 0.5 / dt
        c1, c2 = st.columns(2)
        fmin = c1.slider("Low-cut (Hz)", 1.0, nyq / 2, cfg.FMIN_DEFAULT, key="bp_fmin")
        fmax = c2.slider("High-cut (Hz)", fmin + 5, nyq * 0.95, min(cfg.FMAX_DEFAULT, nyq * 0.9), key="bp_fmax")
        low_slope = c1.slider("Low-cut taper width (Hz)", 1.0, 20.0, cfg.LOW_SLOPE_DEFAULT, key="bp_lowslope")
        high_slope = c2.slider("High-cut taper width (Hz)", 1.0, 30.0, cfg.HIGH_SLOPE_DEFAULT, key="bp_highslope")
        if st.button("Run band-pass filter", type="primary", key="run_bp"):
            h = filtering.design_bandpass(fmin, fmax, dt, survey.num_samples, low_slope, high_slope)
            for g in shot_gathers.values():
                g["traces_bp"] = filtering.apply_time_domain_conv(g["traces"], h)
            ss.mark_stage_done("bandpass", {"fmin_hz": fmin, "fmax_hz": fmax})
            st.rerun()
        if ss.is_stage_done("bandpass"):
            st.success("✅ Band-pass filter applied.")
            shot = _example_shot_selector(shot_list, "bp_shot")
            g = shot_gathers[shot]
            st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                g["traces"], g["traces_bp"], dt, title="Spectrum — Band-Pass Filter"), theme=None, key="bp_spectrum")
            disp = controls.section_display_controls("bp_display", default_colorscale="Seismic")
            st.plotly_chart(seismic_plots.plot_before_after(
                g["traces"], g["traces_bp"], g["receiver_ids"], twt,
                title_before="Raw shot", title_after="After band-pass", x_title="Receiver station",
                colorscale=disp.colorscale, equalize=disp.equalize),
                theme=None, key="bp_before_after")

    # ---- 2) Amplitude recovery (AGC) --------------------------------
    with tabs[1]:
        if not ss.require_stage("bandpass"):
            pass
        else:
            st.markdown("Sliding-window automatic gain control compensates for amplitude decay "
                         "with time (spherical divergence + intrinsic attenuation). Only **one** "
                         "AGC configuration is ever active at a time — running this again with new "
                         "parameters replaces the version every later step uses; there is no hidden "
                         "second copy floating around.")
            window = st.slider("AGC window (samples)", 20, 800, cfg.AGC_WINDOW_DEFAULT, key="agc_window")
            mode = st.radio("Statistic", ["rms", "mean", "median"],
                             index=["rms", "mean", "median"].index(cfg.AGC_MODE_DEFAULT),
                             horizontal=True, key="agc_mode")
            if st.button("Run amplitude recovery", type="primary", key="run_agc"):
                for g in shot_gathers.values():
                    g["traces_agc"] = preprocessing.agc(g["traces_bp"], window=window, mode=mode)
                st.session_state["agc_config"] = {"window": window, "mode": mode}
                ss.mark_stage_done("agc", {"window": window, "mode": mode})
                st.rerun()
            if ss.is_stage_done("agc"):
                cfg_active = st.session_state.get("agc_config") or {}
                st.success(f"✅ Amplitude recovery applied — **active version: {cfg_active.get('mode', '?').upper()} "
                            f"AGC, window={cfg_active.get('window', '?')} samples**. This is the version every "
                            f"later step (trace edit, mute, F-K, ..., stacking) uses; it also appears in the "
                            f"sidebar for the rest of the workflow.")
                shot = _example_shot_selector(shot_list, "agc_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("agc_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_bp"], g["traces_agc"], g["receiver_ids"], twt,
                    title_before="Before AGC", title_after="After AGC", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="agc_before_after")

    # ---- 3) Trace edit / kill / de-spike ----------------------------
    with tabs[2]:
        if not ss.require_stage("agc"):
            pass
        else:
            st.markdown("Two separate, sequential operations — run and QC'd independently so it's "
                         "obvious each one is actually doing something to the data, not just a single "
                         "opaque combined button:")
            st.markdown("**3a. De-spiking** — removes burst-noise transients (electronic glitches, "
                         "blast interference) sample-by-sample, replacing only the flagged samples "
                         "with the local median (an isolation check keeps this from misfiring on real, "
                         "sharp reflection peaks — see `modules/preprocessing.despike_traces`).")
            c1, c2 = st.columns(2)
            spike_window = c1.slider("De-spike window (samples)", 5, 60, 15, key="spike_window")
            spike_thresh = c2.slider("De-spike threshold (x robust noise level)", 3.0, 15.0, 6.0, key="spike_thresh")
            if st.button("Run de-spike", type="primary", key="run_despike"):
                total_spikes = 0
                for g in shot_gathers.values():
                    cleaned, spike_mask = preprocessing.despike_traces(g["traces_agc"], spike_window, spike_thresh)
                    g["traces_despiked"] = cleaned
                    g["_spike_mask"] = spike_mask
                    total_spikes += int(spike_mask.sum())
                st.session_state["_despike_total"] = total_spikes
                st.rerun()

            despike_done = all("traces_despiked" in g for g in shot_gathers.values())
            if despike_done:
                st.success(f"✅ De-spike applied — {st.session_state.get('_despike_total', 0)} sample(s) "
                            f"flagged and replaced across all shots.")
                shot = _example_shot_selector(shot_list, "despike_shot")
                g = shot_gathers[shot]
                n_spikes_shot = int(g["_spike_mask"].sum())
                st.caption(f"{n_spikes_shot} spike sample(s) flagged in shot {shot} "
                            f"(marked below with ×).")
                disp = controls.section_display_controls("despike_display", default_colorscale="Seismic")
                fig = seismic_plots.plot_before_after(
                    g["traces_agc"], g["traces_despiked"], g["receiver_ids"], twt,
                    title_before="Before de-spike", title_after="After de-spike", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize)
                if n_spikes_shot:
                    rows, cols = np.nonzero(g["_spike_mask"])
                    import plotly.graph_objects as go
                    fig.add_trace(go.Scatter(x=g["receiver_ids"][rows], y=twt[cols], mode="markers",
                                              marker=dict(symbol="x", size=6, color="#f59e0b"),
                                              name="Flagged spikes", showlegend=True), row=1, col=2)
                st.plotly_chart(fig, theme=None, key="despike_before_after")

                st.divider()
                st.markdown("**3b. Trace edit / kill** — flags and zeros whole traces that are dead "
                             "(near-zero energy) or abnormally noisy relative to the gather's median "
                             "trace energy, computed on the de-spiked data above.")
                c1, c2 = st.columns(2)
                dead_frac = c1.slider("Dead-trace energy threshold (x median)", 0.001, 0.5, 0.02, key="dead_frac")
                noisy_mult = c2.slider("Noisy-trace energy threshold (x median)", 2.0, 30.0, 8.0, key="noisy_mult")
                if st.button("Run trace kill", type="primary", key="run_kill"):
                    total_killed = 0
                    for g2 in shot_gathers.values():
                        flags = preprocessing.flag_bad_traces(g2["traces_despiked"], dead_frac, noisy_mult)
                        g2["traces_edited"] = preprocessing.apply_trace_kill(g2["traces_despiked"], flags["kill"])
                        g2["_kill_flags"] = flags
                        total_killed += int(flags["kill"].sum())
                    st.session_state["_kill_total"] = total_killed
                    ss.mark_stage_done("trace_edit", {"traces_killed": total_killed,
                                                        "dead_frac": dead_frac, "noisy_mult": noisy_mult})
                    st.rerun()

            if ss.is_stage_done("trace_edit"):
                st.success(f"✅ Trace kill applied — {st.session_state.get('_kill_total', 0)} "
                            f"trace(s) zeroed across all shots.")
                shot = _example_shot_selector(shot_list, "edit_shot")
                g = shot_gathers[shot]
                flags = g["_kill_flags"]
                c1, c2 = st.columns(2)
                c1.metric("Dead traces (this shot)", int(flags["dead"].sum()))
                c2.metric("Noisy traces (this shot)", int(flags["noisy"].sum()))
                if flags["kill"].any():
                    killed_ids = g["receiver_ids"][flags["kill"]]
                    st.caption(f"Killed receiver station(s) in shot {shot}: {list(killed_ids)}")
                disp2 = controls.section_display_controls("kill_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_despiked"], g["traces_edited"], g["receiver_ids"], twt,
                    title_before="Before kill", title_after="After kill (killed traces now flat)",
                    x_title="Receiver station", colorscale=disp2.colorscale, equalize=disp2.equalize),
                    theme=None, key="edit_before_after")

    # ---- 4) Mute --------------------------------------------------------
    with tabs[3]:
        if not ss.require_stage("trace_edit"):
            pass
        else:
            st.markdown(
                "**Mute type** matters here: a constant (flat) top mute only removes the same early "
                "time window on *every* trace regardless of offset — it does nothing to the direct "
                "arrival / ground-roll cone, whose arrival time increases with offset (see the report, "
                "Sec 5.2.2/5.11), so on a full-offset shot gather it visibly fails to touch that "
                "diagonal noise trend at all. The **offset-dependent (linear moveout) mute** instead "
                "follows a straight line `t_mute(x) = t0 + |offset| / v_mute` — the standard way to mute "
                "a direct arrival/ground-roll cone that moves out linearly with offset.")
            mute_type = st.radio("Mute type", ["Offset-dependent (linear moveout)", "Flat top mute (constant time)"],
                                  key="mute_type", horizontal=True)
            taper = st.slider("Taper width (s)", 0.005, 0.3, cfg.MUTE_TAPER_DEFAULT, step=0.005, key="mute_taper")

            if mute_type == "Flat top mute (constant time)":
                t_top = st.slider("Mute time (s)", 0.0, min(0.5, survey.record_length), cfg.MUTE_TTOP_DEFAULT,
                                   step=0.005, key="mute_ttop")
                mute_kwargs = dict(kind="flat", t_top=t_top, taper=taper)
                summary = f"flat, t_top={t_top * 1000:.0f} ms"
            else:
                c1, c2 = st.columns(2)
                v_mute = c1.slider("Mute velocity (m/s)", 200.0, 3000.0, 500.0, step=50.0, key="mute_velocity",
                                    help="Should roughly match the apparent velocity of the direct "
                                         "arrival / ground roll you want to remove — see its slope on "
                                         "the QC plot below.")
                t0_mute = c2.slider("Time intercept t0 (s)", 0.0, 0.3, 0.0, step=0.005, key="mute_t0")
                mute_kwargs = dict(kind="linear", velocity=v_mute, t0=t0_mute, taper=taper)
                summary = f"linear, v={v_mute:.0f} m/s, t0={t0_mute * 1000:.0f} ms"

            if st.button("Run mute", type="primary", key="run_mute"):
                for g in shot_gathers.values():
                    if mute_kwargs["kind"] == "flat":
                        muted = preprocessing.apply_top_bottom_mute(
                            g["traces_edited"], dt, mute_kwargs["t_top"], taper_s=mute_kwargs["taper"])
                        mute_times = np.full(g["traces_edited"].shape[0], mute_kwargs["t_top"])
                    else:
                        muted, mute_times = preprocessing.apply_linear_mute(
                            g["traces_edited"], g["offsets"], dt, mute_kwargs["velocity"],
                            t_intercept=mute_kwargs["t0"], taper_s=mute_kwargs["taper"])
                    g["traces_muted"] = muted
                    g["_mute_times"] = mute_times
                st.session_state["mute_config"] = {**mute_kwargs, "summary": summary}
                ss.mark_stage_done("mute", {"type": mute_kwargs["kind"], **{k: v for k, v in mute_kwargs.items() if k != "kind"}})
                st.rerun()

            if ss.is_stage_done("mute"):
                st.success(f"✅ Mute applied — **active mute: {summary}**.")
                shot = _example_shot_selector(shot_list, "mute_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("mute_display", default_colorscale="Seismic")
                st.markdown("**Before, with the mute cutoff line overlaid** — everything below/right "
                             "of the dashed line is removed.")
                st.plotly_chart(seismic_plots.plot_mute_qc(
                    g["traces_edited"], g["receiver_ids"], twt, g["_mute_times"],
                    title="Before mute (cutoff overlaid)", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="mute_qc_overlay")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_edited"], g["traces_muted"], g["receiver_ids"], twt,
                    title_before="Before", title_after="After mute", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="mute_before_after")

    # ---- 5) Trace balancing ------------------------------------------
    with tabs[4]:
        if not ss.require_stage("mute"):
            pass
        else:
            st.markdown("Scales each trace by a single RMS scalar (over a reference window from "
                         "t=0) so no one trace dominates purely due to a gain difference.")
            ref_s = st.slider("Reference window (s, from t=0)", 0.2, min(3.0, survey.record_length),
                               min(cfg.NORM_WINDOW_S_DEFAULT, survey.record_length), key="bal_ref")
            if st.button("Run trace balancing", type="primary", key="run_bal"):
                ref_window = slice(0, int(ref_s / dt))
                for g in shot_gathers.values():
                    g["traces_balanced"] = preprocessing.trace_normalize(g["traces_muted"], ref_window)
                ss.mark_stage_done("balance", {"ref_window_s": ref_s})
                st.rerun()
            if ss.is_stage_done("balance"):
                st.success("✅ Trace balancing applied.")
                shot = _example_shot_selector(shot_list, "bal_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("bal_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_muted"], g["traces_balanced"], g["receiver_ids"], twt,
                    title_before="Before", title_after="After balancing", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="bal_before_after")

    # ---- 6) F-K filter (auto polygon or click-drawn) -----------------
    with tabs[5]:
        if not ss.require_stage("balance"):
            pass
        else:
            st.markdown("A 2-D FFT transforms a gather into frequency-wavenumber (F-K) space, "
                         "where coherent linear noise (ground roll) separates from reflections by "
                         "its low apparent velocity (report Sec 5.11/6.6). The red polygon below "
                         "marks the region that is **rejected** (attenuated); everything outside it "
                         "passes through unchanged, with a soft taper at the polygon edge to avoid "
                         "ringing artifacts.")
            shot_fk = _example_shot_selector(shot_list, "fk_preview_shot")
            g_preview = shot_gathers[shot_fk]
            _, _, f_axis, k_axis, fk_before, _ = filtering.fk_filter_gather(
                g_preview["traces_balanced"], dt, survey.group_interval,
                st.session_state["fk_polygon"] or cfg.FK_POLYGON_DEFAULT)
            magnitude = np.abs(fk_before)

            poly_mode = st.radio("Reject-polygon source", ["Default polygon", "Build my own (numeric points)"],
                                  key="fk_poly_mode", horizontal=True)

            if poly_mode == "Default polygon":
                active_polygon = cfg.FK_POLYGON_DEFAULT
                st.plotly_chart(seismic_plots.plot_fk_spectrum(
                    f_axis, k_axis, magnitude, polygon_points=active_polygon,
                    title=f"F-K Spectrum — shot {shot_fk} (default reject polygon)"),
                    theme=None, key="fk_spectrum_default")
            else:
                st.plotly_chart(seismic_plots.plot_fk_spectrum(
                    f_axis, k_axis, magnitude, polygon_points=st.session_state["fk_click_points"],
                    title=f"F-K Spectrum — shot {shot_fk} (zoom in to read off coordinates)"),
                    theme=None, key="fk_click_chart")
                st.caption("Zoom into the spectrum above to read off (wavenumber, frequency) "
                            "coordinates that trace the boundary of the ground-roll energy, then "
                            "add each one below. (Streamlit can't deliver reliable click events on a "
                            "heatmap-backed plot, so points are entered here instead of by clicking.)")
                new_pt = numeric_point_form(
                    "fk_pt", "Wavenumber k (cycles/m)", "Frequency (Hz)",
                    x_value=0.0, y_value=float(f_axis.max()) * 0.3,
                    x_min=float(k_axis.min()), x_max=float(k_axis.max()),
                    y_min=0.0, y_max=float(f_axis.max()), x_step=0.001, y_step=1.0, x_format="%.4f")
                if new_pt is not None:
                    st.session_state["fk_click_points"].append(new_pt)
                    st.rerun()
                c1, c2, c3 = st.columns(3)
                if c1.button("Undo last point", key="fk_undo") and st.session_state["fk_click_points"]:
                    st.session_state["fk_click_points"].pop()
                    st.rerun()
                if c2.button("Clear polygon", key="fk_clear"):
                    st.session_state["fk_click_points"] = []
                    st.rerun()
                c3.metric("Points drawn", len(st.session_state["fk_click_points"]))
                active_polygon = (st.session_state["fk_click_points"]
                                   if len(st.session_state["fk_click_points"]) >= 3 else None)
                if active_polygon is None:
                    st.info("Add at least 3 points to define a usable polygon.", icon="ℹ️")

            if st.button("Run F-K filter", type="primary", key="run_fk", disabled=active_polygon is None):
                st.session_state["fk_polygon"] = list(active_polygon)
                prog = st.progress(0.0, text="F-K filtering gathers...")
                items = list(shot_gathers.items())
                for i, (shot_id, g) in enumerate(items):
                    filt, pass_mask, f, k, fk, fkf = filtering.fk_filter_gather(
                        g["traces_balanced"], dt, survey.group_interval, active_polygon)
                    g["traces_fk_filtered"] = filt
                    if i % 20 == 0:
                        prog.progress((i + 1) / len(items))
                prog.progress(1.0)
                ss.mark_stage_done("fk", {"polygon_vertices": len(active_polygon),
                                            "source": poly_mode})
                st.rerun()
            if ss.is_stage_done("fk"):
                st.success("✅ F-K filter applied.")
                shot = _example_shot_selector(shot_list, "fk_shot")
                g = shot_gathers[shot]
                _, _, f_ax2, k_ax2, fk_b, fk_a = filtering.fk_filter_gather(
                    g["traces_balanced"], dt, survey.group_interval, st.session_state["fk_polygon"])
                st.plotly_chart(seismic_plots.plot_fk_before_after(
                    f_ax2, k_ax2, np.abs(fk_b), np.abs(fk_a), polygon_points=st.session_state["fk_polygon"]),
                    theme=None, key="fk_spectrum_before_after")
                disp = controls.section_display_controls("fk_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_balanced"], g["traces_fk_filtered"], g["receiver_ids"], twt,
                    title_before="Before F-K", title_after="After F-K", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="fk_before_after")

    # ---- 7) Multiple suppression (predictive/gap deconvolution) ------
    with tabs[6]:
        if not ss.require_stage("fk"):
            pass
        else:
            st.markdown("**Predictive (gap) deconvolution**: rather than collapsing the whole "
                         "wavelet (spiking decon, next step), a prediction lag greater than one "
                         "sample preserves the primary wavelet while predicting and subtracting "
                         "the periodic reverberation tail caused by short-period multiples — the "
                         "standard land-data multiple-suppression tool (the marine equivalent, "
                         "SRME, needs a free-surface model this app doesn't have — see README). "
                         "A practical starting point for the prediction lag is the *second "
                         "zero-crossing* of the autocorrelation.")
            shot_ms = _example_shot_selector(shot_list, "ms_suggest_shot")
            suggested_lag_s = preprocessing.second_zero_crossing_lag(
                shot_gathers[shot_ms]["traces_fk_filtered"][0], dt)
            st.caption(f"Suggested prediction lag for shot {shot_ms}, trace 0: "
                        f"≈{suggested_lag_s * 1000:.1f} ms ({int(round(suggested_lag_s / dt))} samples)")
            c1, c2, c3 = st.columns(3)
            oplen = c1.slider("Operator length (samples)", 8, 150, 60, key="ms_oplen")
            lag_samples = c2.slider("Prediction lag (samples)", 2, 100,
                                     max(2, int(round(suggested_lag_s / dt))), key="ms_lag")
            pw = c3.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_PW_DEFAULT,
                                  step=0.005, format="%.3f", key="ms_pw")
            if st.button("Run multiple suppression", type="primary", key="run_ms"):
                prog = st.progress(0.0, text="Suppressing multiples...")
                items = list(shot_gathers.items())
                for i, (shot_id, g) in enumerate(items):
                    g["traces_demultiple"] = preprocessing.predictive_decon_gather(
                        g["traces_fk_filtered"], oplen, lag_samples, pw)
                    if i % 20 == 0:
                        prog.progress((i + 1) / len(items))
                prog.progress(1.0)
                ss.mark_stage_done("mult_supp", {"operator_length": oplen, "prediction_lag_samples": lag_samples})
                st.rerun()
            if ss.is_stage_done("mult_supp"):
                st.success("✅ Multiple suppression applied.")
                shot = _example_shot_selector(shot_list, "ms_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("ms_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_fk_filtered"], g["traces_demultiple"], g["receiver_ids"], twt,
                    title_before="Before", title_after="After multiple suppression", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="ms_before_after")

    # ---- 8) Deconvolution (spiking) -----------------------------------
    with tabs[7]:
        if not ss.require_stage("mult_supp"):
            pass
        else:
            st.markdown("Spiking deconvolution (prediction lag = 1 sample) compresses the "
                         "remaining source wavelet toward a zero-lag spike, sharpening temporal "
                         "resolution — the final pre-stack-conditioning step before CDP sort/binning.")
            oplen = st.slider("Operator length (samples)", 4, 120, cfg.DECON_OPLEN_DEFAULT, key="decon_oplen")
            pw = st.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_PW_DEFAULT,
                                  step=0.005, format="%.3f", key="decon_pw")
            if st.button("Run deconvolution", type="primary", key="run_decon"):
                prog = st.progress(0.0, text="Deconvolving gathers...")
                items = list(shot_gathers.items())
                for i, (shot_id, g) in enumerate(items):
                    g["traces_decon"] = preprocessing.spiking_decon_gather(g["traces_demultiple"], oplen, pw)
                    if i % 20 == 0:
                        prog.progress((i + 1) / len(items))
                prog.progress(1.0)
                ss.mark_stage_done("decon", {"operator_length": oplen, "pre_whitening": pw})
                st.rerun()
            if ss.is_stage_done("decon"):
                st.success("✅ Deconvolution applied — pre-stack conditioning complete. "
                            "Continue to **CDP Sort & Binning**.")
                shot = _example_shot_selector(shot_list, "decon_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("decon_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_demultiple"], g["traces_decon"], g["receiver_ids"], twt,
                    title_before="Before decon", title_after="After decon", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="decon_before_after")
                st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                    g["traces_demultiple"], g["traces_decon"], dt, label_before="Before", label_after="After",
                    title="Spectrum — Deconvolution whitens the passband"), theme=None, key="decon_spectrum")
