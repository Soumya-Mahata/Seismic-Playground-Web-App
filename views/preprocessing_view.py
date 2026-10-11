"""
views/preprocessing_view.py — Pre-Stack Conditioning (shot domain)
======================================================================
Band-pass, AGC, de-spike, trace edit/kill, mute, trace balancing, F-K
filter (auto or numeric-polygon), multiple suppression (predictive
deconvolution) and deconvolution, applied to every SHOT gather (the
domain the data was actually recorded in), each with a "Run" button and
before/after QC on a representative shot gather.

These steps are INDEPENDENT of each other. The tab numbers show the
*recommended* order only: every step needs just the shot gathers, so any
subset can be run in any order (`Raw → AGC → Band-pass → Mute` and
`Raw → Band-pass → F-K → CDP Sort` are both valid). Each step reads the
current state of the shot data — whatever was applied last — and warns
(never blocks) if a commonly-preceding step has not been applied.

CDP sort/binning — re-grouping this conditioned data by CDP — happens
afterward, in `views/cdp_sort_view.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from modules import filtering, preprocessing, stage_ops
from modules.dependencies import state_name
from utils import session_state as ss
from utils import workflow_ui as wf
from utils.click_picking import numeric_point_form
from visualization import controls, seismic_plots, spectra_plots, theme

SHOT_STAGES = ["bandpass", "agc", "despike", "trace_edit", "mute", "balance", "fk", "mult_supp", "decon"]


def _example_shot_selector(shot_list, key, default_shot=None):
    """Shot picker keyed by the real FFID/shot-number from the trace headers
    (`shot_id`/FieldRecord) — not a synthetic 1..N index. The label spells out
    "shot i of n" alongside the FFID so the dropdown's total count is never
    mistaken for the FFID's numeric value (FFIDs can skip, e.g. 1, 3, 5, ...
    for a shot interval of 2 stations — that is a real, headers-driven survey
    design choice, not extra/duplicated shots)."""
    n = len(shot_list)
    idx = shot_list.index(default_shot) if default_shot in shot_list else n // 2
    return st.selectbox(
        f"Example shot for QC ({n} shots total, from the FFID/shot-number header)",
        shot_list, index=idx, key=key,
        format_func=lambda ffid: f"Shot {shot_list.index(ffid) + 1} of {n}  (FFID {ffid})")


def _before_label(res) -> str:
    """'Before — Raw (S1)' / 'Before — AGC (S2)': the state this run actually read."""
    return res.before_label


def render():
    st.header("⚙️ Pre-Stack Conditioning")
    st.caption("Applied to shot gathers — the domain the data was actually recorded in. Every step "
                "below is independent: run any subset, in any order. The numbering is the "
                "**recommended** order only. CDP sort/binning happens afterward.")

    if not ss.is_stage_done("shot_gathers"):
        # Every conditioning step needs the shot gathers — that single genuine
        # dependency is the only reason this page is not usable yet.
        wf.show_status("bandpass")
        return
    shot_gathers = st.session_state["shot_gathers"]
    shot_list = list(shot_gathers.keys())
    survey = st.session_state["survey"]
    dt = survey.sample_interval
    twt = survey.twt

    wf.data_state_caption("shot", "Current shot-domain data")
    wf.stale_banner(wf.stale_for_ops(["cdp_sort"]))
    with st.expander("Status of every step on this page"):
        rows = [r for r in wf.status_rows() if r["stage"] in SHOT_STAGES]
        st.dataframe(pd.DataFrame([{
            "step": r["label"], "status": r["level"].badge, "done": "✅" if r["done"] else "",
            "note": r["note"]} for r in rows]), hide_index=True)
    wf.states_panel(["shot"], "states_shot", "🗂 Shot-domain data states · undo · redo")

    n_traces_in_shots = sum(g["traces"].shape[0] for g in shot_gathers.values())
    fn_info = getattr(survey, "field_noise_info", None) or {}
    spike_shots = sorted({loc["shot_id"] for loc in fn_info.get("spike_locations", [])})
    dead_stations = set(fn_info.get("dead_stations", []))
    powerline_stations = set(fn_info.get("powerline_stations", []))
    dead_or_pl_shots = [sid for sid in shot_list
                         if dead_stations & set(shot_gathers[sid]["receiver_ids"])
                         or powerline_stations & set(shot_gathers[sid]["receiver_ids"])]
    default_spike_shot = spike_shots[0] if spike_shots else None
    default_edit_shot = dead_or_pl_shots[0] if dead_or_pl_shots else None
    if fn_info:
        st.info(
            f"🧪 This synthetic survey has injected field-noise conditions to give Trace Edit / "
            f"De-spike / Mute real problems to fix: **dead channels** at receiver "
            f"station(s) {sorted(dead_stations) or 'none'}, **powerline harmonic noise** "
            f"({fn_info.get('powerline_freq_hz', '?')} Hz) at station(s) {sorted(powerline_stations) or 'none'}, "
            f"and **{len(fn_info.get('spike_locations', []))} spike bursts** scattered through the "
            f"dataset. The QC shot selectors below default to a shot that actually has one of these.",
            icon="🧪")

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

    tabs = st.tabs(["1. Band-Pass Filter", "2. Amplitude Recovery", "3a. De-spike",
                    "3b. Trace Edit/Kill", "4. Mute", "5. Trace Balancing", "6. F-K Filter",
                    "7. Multiple Suppression", "8. Deconvolution"])

    # ---- 1) Band-pass filter ----------------------------------------
    with tabs[0]:
        st.markdown("A trapezoidal frequency-domain band-pass, smoothed to avoid ringing, "
                     "removes the low-frequency ground-roll/DC drift and high-frequency noise "
                     "outside the useful reflection bandwidth — commonly applied first, before any "
                     "amplitude-sensitive step.")
        wf.show_status("bandpass")
        nyq = 0.5 / dt
        c1, c2 = st.columns(2)
        fmin = c1.slider("Low-cut (Hz)", 1.0, nyq / 2, cfg.FMIN_DEFAULT, key="bp_fmin")
        fmax = c2.slider("High-cut (Hz)", fmin + 5, nyq * 0.95, min(cfg.FMAX_DEFAULT, nyq * 0.9), key="bp_fmax")
        low_slope = c1.slider("Low-cut taper width (Hz)", 1.0, 20.0, cfg.LOW_SLOPE_DEFAULT, key="bp_lowslope")
        high_slope = c2.slider("High-cut taper width (Hz)", 1.0, 30.0, cfg.HIGH_SLOPE_DEFAULT, key="bp_highslope")
        if wf.run_button("bandpass", "Run band-pass filter", key="run_bp"):
            wf.execute(stage_ops.run_bandpass, fmin, fmax, low_slope, high_slope)
        e = wf.result("bandpass")
        if e is not None:
            wf.result_note(e)
            st.success("✅ Band-pass filter applied.")
            shot = _example_shot_selector(shot_list, "bp_shot")
            g = e.gathers[shot]
            st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                g[e.in_key], g[e.out_key], dt, title="Spectrum — Band-Pass Filter"), theme=None, key="bp_spectrum")
            disp = controls.section_display_controls("bp_display", default_colorscale="Seismic")
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                title_before=_before_label(e), title_after="After band-pass", x_title="Receiver station",
                colorscale=disp.colorscale, equalize=disp.equalize),
                theme=None, key="bp_before_after")

    # ---- 2) Amplitude recovery (AGC) --------------------------------
    with tabs[1]:
        st.markdown(
            "Sliding-window automatic gain control compensates for amplitude decay with time "
            "(spherical divergence + intrinsic attenuation). The scalar driving the gain can be "
            "computed three ways — **mean**, **RMS**, and **median** AGC — which normally "
            "produce similar but not identical results. Compare all three below on an "
            "example shot, then pick the one to use. The result is a new data state; the next step "
            "reads whichever state is current — running AGC again with another mode adds a state "
            "rather than replacing this one.")
        chk = wf.show_status("agc")
        if chk.can_run:
            pg, in_key, _ = wf.input_gathers(chk)
            window = st.slider("AGC window (samples)", 20, 800, cfg.AGC_WINDOW_DEFAULT, key="agc_window")
            shot_preview = _example_shot_selector(shot_list, "agc_preview_shot")
            g_preview = pg[shot_preview]
            previews = {m: preprocessing.agc(g_preview[in_key], window=window, mode=m)
                        for m in ("mean", "rms", "median")}

            disp = controls.section_display_controls("agc_display", default_colorscale="Seismic")
            cols = st.columns(3)
            for col, m in zip(cols, ("mean", "rms", "median")):
                with col:
                    st.markdown(f"**{m.upper()} AGC**")
                    st.plotly_chart(seismic_plots.plot_section(
                        previews[m], g_preview["receiver_ids"], twt, title="", height=420,
                        colorscale=disp.colorscale, pclip=disp.pclip, equalize=disp.equalize),
                        theme=None, key=f"agc_preview_{m}")

            mode = st.radio("AGC version to use", ["mean", "rms", "median"],
                             index=["mean", "rms", "median"].index(cfg.AGC_MODE_DEFAULT),
                             horizontal=True, key="agc_mode",
                             help="Mean AGC is generally preferred in seismic data processing "
                                  "applications, though all three are offered here since "
                                  "the difference can matter for a given dataset.")

            if wf.run_button("agc", f"Apply {mode.upper()} AGC to all shots", key="run_agc"):
                wf.execute(stage_ops.run_agc, window, mode)
            e = wf.result("agc")
            if e is not None:
                wf.result_note(e)
                cfg_active = e.params
                st.success(f"✅ Amplitude recovery applied — **active version: {cfg_active.get('mode', '?').upper()} "
                            f"AGC, window={cfg_active.get('window', '?')} samples**. It also appears in the "
                            f"sidebar for the rest of the workflow.")
                shot = _example_shot_selector(shot_list, "agc_shot")
                g = e.gathers[shot]
                st.plotly_chart(seismic_plots.plot_before_after(
                    g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                    title_before=_before_label(e), title_after=f"After {cfg_active.get('mode', '?').upper()} AGC",
                    x_title="Receiver station", colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="agc_before_after")

    # ---- 3a) De-spike -------------------------------------------------
    with tabs[2]:
        st.markdown("**De-spiking** removes burst-noise transients (electronic glitches, "
                     "blast interference) sample-by-sample, replacing only the flagged samples "
                     "with the local median (an isolation check keeps this from misfiring on real, "
                     "sharp reflection peaks — see `modules/preprocessing.despike_traces`). It is "
                     "independent of trace kill — run either, both, or neither.")
        wf.show_status("despike")
        c1, c2 = st.columns(2)
        spike_window = c1.slider("De-spike window (samples)", 5, 60, 15, key="spike_window")
        spike_thresh = c2.slider("De-spike threshold (x robust noise level)", 3.0, 15.0, 6.0, key="spike_thresh")
        if wf.run_button("despike", "Run de-spike", key="run_despike"):
            wf.execute(stage_ops.run_despike, spike_window, spike_thresh)
        e = wf.result("despike")
        if e is not None:
            wf.result_note(e)
            st.success(f"✅ De-spike applied — {e.info.get('samples_flagged', 0)} sample(s) "
                        f"flagged and replaced across all shots.")
            shot = _example_shot_selector(shot_list, "despike_shot", default_shot=default_spike_shot)
            g = e.gathers[shot]
            n_spikes_shot = int(g[e.key("spike_mask")].sum())
            st.caption(f"{n_spikes_shot} spike sample(s) flagged in shot {shot} "
                        f"(marked below with ×).")
            disp = controls.section_display_controls("despike_display", default_colorscale="Seismic")
            fig = seismic_plots.plot_before_after(
                g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                title_before=_before_label(e), title_after="After de-spike", x_title="Receiver station",
                colorscale=disp.colorscale, equalize=disp.equalize)
            if n_spikes_shot:
                rows, cols = np.nonzero(g[e.key("spike_mask")])
                fig.add_trace(go.Scatter(x=g["receiver_ids"][rows], y=twt[cols], mode="markers",
                                          marker=dict(symbol="x", size=6, color="#f59e0b"),
                                          name="Flagged spikes", showlegend=True), row=1, col=2)
            st.plotly_chart(fig, theme=None, key="despike_before_after")

    # ---- 3b) Trace edit / kill ----------------------------------------
    with tabs[3]:
        st.markdown(
            "**Trace edit / kill** — flags whole traces that are dead (near-zero energy) "
            "or abnormally noisy relative to the gather's median trace energy, then zeroes them. "
            "**Detection runs on amplitude-preserving data** — the data as it stood *before* any "
            "AGC or trace balancing — because those are per-trace self-normalizing gains: after "
            "them every trace (dead, noisy, or healthy) sits at very nearly the same level by "
            "construction, which would make this energy-based detection blind to real dead/noisy "
            "channels. The resulting kill mask is then applied to the data actually being processed.")
        chk = wf.show_status("trace_edit")
        if chk.can_run:
            lineage = ss.graph().lineage(chk.inputs["data"])
            det_state = lineage[stage_ops.kill_detection_index([x.op for x in lineage])]
            st.caption(f"Detection will measure: **{state_name(det_state)}** data · "
                        f"the mask will be applied to: **{wf.input_name(chk)}** data.")
            c1, c2 = st.columns(2)
            dead_frac = c1.slider("Dead-trace energy threshold (x median)", 0.001, 0.5, 0.02, key="dead_frac")
            noisy_mult = c2.slider("Noisy-trace energy threshold (x median)", 2.0, 30.0, 8.0, key="noisy_mult")
            if wf.run_button("trace_edit", "Run trace kill", key="run_kill"):
                wf.execute(stage_ops.run_trace_kill, dead_frac, noisy_mult)
            e = wf.result("trace_edit")
            if e is not None:
                wf.result_note(e)
                st.success(f"✅ Trace kill applied — {e.info.get('traces_killed', 0)} "
                            f"trace(s) zeroed across all shots (detected on: {e.info.get('detected_on', '?')}).")
                shot = _example_shot_selector(shot_list, "edit_shot", default_shot=default_edit_shot)
                g = e.gathers[shot]
                flags = g[e.key("kill_flags")]
                c1, c2 = st.columns(2)
                c1.metric("Dead traces (this shot)", int(flags["dead"].sum()))
                c2.metric("Noisy traces (this shot)", int(flags["noisy"].sum()))
                if flags["kill"].any():
                    killed_ids = g["receiver_ids"][flags["kill"]]
                    st.caption(f"Killed receiver station(s) in shot {shot}: {list(killed_ids)}")
                else:
                    st.caption("No dead/noisy traces flagged in this shot — try the shot the info box "
                                "above points to if you expected to see one flagged.")
                with st.expander("Why these traces were flagged (per-trace energy vs. median)"):
                    energy_fig = go.Figure()
                    colors = np.where(flags["dead"], "#ef4444", np.where(flags["noisy"], "#f59e0b", "#0D2B5E"))
                    energy_fig.add_trace(go.Bar(x=[str(r) for r in g["receiver_ids"]], y=flags["energy"],
                                                  marker_color=list(colors)))
                    energy_fig.add_hline(y=float(np.median(flags["energy"])), line_dash="dash",
                                          annotation_text="median")
                    energy_fig.update_layout(**theme.base_layout(
                        f"Per-trace energy ({e.info.get('detected_on', '?')}) — red=dead, orange=noisy, blue=kept", 320))
                    energy_fig.update_xaxes(**theme.x_axis("Receiver station"))
                    energy_fig.update_yaxes(**theme.x_axis("Energy"), type="log")
                    st.plotly_chart(energy_fig, theme=None, key="kill_energy_bar")
                disp2 = controls.section_display_controls("kill_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                    title_before=_before_label(e), title_after="After kill (killed traces now flat)",
                    x_title="Receiver station", colorscale=disp2.colorscale, equalize=disp2.equalize),
                    theme=None, key="edit_before_after")

    # ---- 4) Mute --------------------------------------------------------
    with tabs[4]:
        st.markdown(
            "There are three standard mute types. Two are implemented here, in the "
            "shot domain where they are typically applied; the third happens automatically later:\n"
            "- **Top mute**: removes the water-column / pre-first-break noise. Its "
            "cutoff follows the **direct arrival's moveout** — a straight line that gets later "
            "with offset — not the same fixed time on every trace; a flat top mute does nothing "
            "to that diagonal noise trend at all.\n"
            "- **Surgical mute**: zeroes a specific (offset, time) box of noise "
            "*inside* the gather that isn't easy to remove with a filter (e.g. localized "
            "high-frequency harmonic noise).\n"
            "- **NMO-stretch mute** is applied automatically to CDP gathers right "
            "after NMO correction, in the Velocity, NMO & DMO page — it doesn't belong here since "
            "it needs NMO-corrected data that doesn't exist yet at this stage of the pipeline."
        )
        chk = wf.show_status("mute")
        if chk.can_run:
            pg, in_key, _ = wf.input_gathers(chk)
            mute_type = st.radio("Mute type", ["Top mute (follows direct-arrival moveout)",
                                                  "Top mute (flat — simpler, not offset-aware)",
                                                  "Surgical mute (remove a noisy box region)"],
                                  key="mute_type")
            taper = st.slider("Taper width (s)", 0.005, 0.3, cfg.MUTE_TAPER_DEFAULT, step=0.005, key="mute_taper")

            if mute_type.startswith("Top mute (flat"):
                t_top = st.slider("Mute time (s)", 0.0, min(0.5, survey.record_length), cfg.MUTE_TTOP_DEFAULT,
                                   step=0.005, key="mute_ttop")
                mute_kwargs = dict(kind="flat", t_top=t_top, taper=taper)
            elif mute_type.startswith("Top mute (follows"):
                c1, c2 = st.columns(2)
                v_mute = c1.slider("Mute velocity (m/s)", 200.0, 3000.0, 500.0, step=50.0, key="mute_velocity",
                                    help="Should roughly match the apparent velocity of the direct "
                                         "arrival / ground roll you want to remove — see its slope on "
                                         "the QC plot below.")
                t0_mute = c2.slider("Time intercept t0 (s)", 0.0, 0.3, 0.0, step=0.005, key="mute_t0")
                mute_kwargs = dict(kind="linear", velocity=v_mute, t0=t0_mute, taper=taper)
            else:
                st.caption("Pick an offset range and a time range to zero out — e.g. draw a box "
                            "around a localized noise burst visible on the QC plot for an example "
                            "shot below.")
                shot_sm = _example_shot_selector(shot_list, "surgical_preview_shot")
                g_sm = pg[shot_sm]
                max_off = float(np.max(np.abs(g_sm["offsets"])))
                c1, c2 = st.columns(2)
                x_range = c1.slider("Offset range (m, absolute)", 0.0, max_off, (0.0, max_off * 0.3),
                                     key="surgical_xrange")
                t_range = c2.slider("Time range (s)", 0.0, float(survey.record_length),
                                     (0.0, min(0.3, float(survey.record_length))), key="surgical_trange")
                mute_kwargs = dict(kind="surgical", x_min=x_range[0], x_max=x_range[1],
                                     t_min=t_range[0], t_max=t_range[1], taper=taper)
                disp_sm = controls.section_display_controls("surgical_display", default_colorscale="Seismic")
                fig_sm = seismic_plots.plot_section(g_sm[in_key], g_sm["offsets"], twt,
                                                      title=f"Shot {shot_sm} — surgical mute box preview",
                                                      x_title="Offset (m)", colorscale=disp_sm.colorscale,
                                                      equalize=disp_sm.equalize)
                fig_sm.add_shape(type="rect", x0=x_range[0], x1=x_range[1], y0=t_range[0], y1=t_range[1],
                                  line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                fig_sm.add_shape(type="rect", x0=-x_range[1], x1=-x_range[0], y0=t_range[0], y1=t_range[1],
                                  line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                st.plotly_chart(fig_sm, theme=None, key="surgical_preview")

            if wf.run_button("mute", "Run mute", key="run_mute"):
                kind = mute_kwargs.pop("kind")
                wf.execute(stage_ops.run_mute, kind, **mute_kwargs)

            e = wf.result("mute")

            if e is not None:

                wf.result_note(e)
                mcfg = {**e.params, "summary": e.info["summary"]}
                st.success(f"✅ Mute applied — **active mute: {mcfg['summary']}**.")
                shot = _example_shot_selector(shot_list, "mute_shot")
                g = e.gathers[shot]
                disp = controls.section_display_controls("mute_display", default_colorscale="Seismic")
                if mcfg["kind"] in ("flat", "linear"):
                    st.markdown("**Before, with the mute cutoff line overlaid** — everything below/right "
                                 "of the dashed line is removed.")
                    st.plotly_chart(seismic_plots.plot_mute_qc(
                        g[e.in_key], g["receiver_ids"], twt, g[e.key("mute_times")],
                        title="Before mute (cutoff overlaid)", x_title="Receiver station",
                        colorscale=disp.colorscale, equalize=disp.equalize),
                        theme=None, key="mute_qc_overlay")
                else:
                    st.markdown(f"**Before, with the surgical mute box overlaid** "
                                 f"({int(np.sum(g[e.key('surgical_box')]))} of {len(g[e.key('surgical_box')])} "
                                 f"traces in this shot fall inside the chosen offset range).")
                    fig_box = seismic_plots.plot_section(g[e.in_key], g["offsets"], twt,
                                                           title="Before mute (box overlaid)",
                                                           x_title="Offset (m)", colorscale=disp.colorscale,
                                                           equalize=disp.equalize)
                    fig_box.add_shape(type="rect", x0=mcfg["x_min"], x1=mcfg["x_max"],
                                        y0=mcfg["t_min"], y1=mcfg["t_max"],
                                        line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                    fig_box.add_shape(type="rect", x0=-mcfg["x_max"], x1=-mcfg["x_min"],
                                        y0=mcfg["t_min"], y1=mcfg["t_max"],
                                        line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                    st.plotly_chart(fig_box, theme=None, key="mute_qc_overlay")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g[e.in_key], g[e.out_key],
                    g["offsets"] if mcfg["kind"] == "surgical" else g["receiver_ids"], twt,
                    title_before=_before_label(e), title_after="After mute",
                    x_title="Offset (m)" if mcfg["kind"] == "surgical" else "Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="mute_before_after")

    # ---- 5) Trace balancing ------------------------------------------
    with tabs[5]:
        st.markdown("Scales each trace by a single RMS scalar (over a reference window from "
                     "t=0) so no one trace dominates purely due to a gain difference.")
        wf.show_status("balance")
        ref_s = st.slider("Reference window (s, from t=0)", 0.2, min(3.0, survey.record_length),
                           min(cfg.NORM_WINDOW_S_DEFAULT, survey.record_length), key="bal_ref")
        if wf.run_button("balance", "Run trace balancing", key="run_bal"):
            wf.execute(stage_ops.run_balance, ref_s)
        e = wf.result("balance")
        if e is not None:
            wf.result_note(e)
            st.success("✅ Trace balancing applied.")
            shot = _example_shot_selector(shot_list, "bal_shot")
            g = e.gathers[shot]
            disp = controls.section_display_controls("bal_display", default_colorscale="Seismic")
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                title_before=_before_label(e), title_after="After balancing", x_title="Receiver station",
                colorscale=disp.colorscale, equalize=disp.equalize),
                theme=None, key="bal_before_after")

    # ---- 6) F-K dip filter (accept / reject polygon) -------------------
    with tabs[6]:
        st.markdown(
            "A 2-D FFT turns a gather into frequency-wavenumber (F-K) space, where an event of "
            "apparent velocity *V* lies on the line `|k| = f / V`. "
            "Primary reflections are fast (high apparent velocity) and cluster in a fan around the "
            "frequency axis; ground roll, direct wave and refractions are slow and appear as steep "
            "diagonal lines far from it. The filter multiplies the amplitude spectrum by a mask "
            "and leaves the phase untouched. The polygon is either an **accept** "
            "polygon (encloses the signal — preferred in practice, since it is easier to enclose "
            "the signal zone) or a **reject** polygon (encloses the noise); the attenuated side is "
            "multiplied by a small coefficient, not exactly zero, since multiplying the frequency-"
            "domain spectrum by exactly zero would correspond to an infinite scaling in the time "
            "domain."
        )
        chk = wf.show_status("fk")
        if chk.can_run:
            pg, in_key, _ = wf.input_gathers(chk)
            shot_fk = _example_shot_selector(shot_list, "fk_preview_shot")
            g_preview = pg[shot_fk]
            f_nyq, k_nyq = 0.5 / dt, 0.5 / survey.group_interval
            st.caption(f"This survey's actual F-K domain: k = ±{k_nyq:.3f} cycles/m "
                       f"(station spacing {survey.group_interval:.2f} m), f = 0–{f_nyq:.0f} Hz. "
                       f"The default wedge below is drawn to this exact range — a coarser station "
                       f"spacing (larger `dx`) gives a smaller k_max, so the same velocity cutoff "
                       f"saturates the wedge to full width at a lower frequency (generate the "
                       f"synthetic survey with a finer spacing on the Data Management page for a "
                       f"more gradually-tapering default wedge, or use the custom-polygon option "
                       f"below to hand-draw any shape regardless of spacing).")

            c1, c2 = st.columns(2)
            polygon_mode_label = c1.radio(
                "Polygon type", ["Accept polygon (encloses signal — preferred)", "Reject polygon (encloses noise)"],
                key="fk_polygon_type",
                help="These are opposite operations for the SAME boundary drawn below: Accept keeps "
                     "the inside of the polygon and attenuates everything outside it; Reject does the "
                     "reverse — keeps the outside and attenuates the inside. Both approaches normally "
                     "produce the same results, but only when the two *polygon shapes* are drawn as "
                     "complements of each other to describe the same filter design — not when flipping "
                     "this toggle while keeping the exact same boundary, which does change the output. "
                     "Watch the 'Multiplier that will be applied' panel below: it visibly inverts "
                     "(black/white swap) when you toggle this.")
            polygon_mode = "accept" if polygon_mode_label.startswith("Accept") else "reject"
            poly_source = c2.radio("Polygon source", ["Apparent-velocity wedge (recommended)",
                                                        "Custom (numeric points)"],
                                    key="fk_poly_source")
            reject_coeff = st.slider(
                "Attenuation coefficient (multiplier on the attenuated side)", 0.0, 0.3,
                cfg.FK_REJECT_COEFFICIENT_DEFAULT, step=0.005, key="fk_reject_coeff",
                help="A small non-zero constant, not exactly 0 — multiplying by a "
                     "constant in the frequency domain scales the time-domain function by its "
                     "inverse, and the inverse of zero is infinite. Lower = stronger "
                     "attenuation.")

            v_cutoff = None
            if poly_source.startswith("Apparent-velocity"):
                v_cutoff = st.slider(
                    "Apparent-velocity cutoff (m/s)", 200.0, 4000.0, cfg.FK_VELOCITY_CUTOFF_DEFAULT,
                    step=50.0, key="fk_vcutoff",
                    help="Events slower than this (ground roll, direct wave, low-velocity "
                         "refractions) fall OUTSIDE the wedge; faster events (reflections) fall "
                         "INSIDE it. Read the ground-roll velocity off the slope of the linear "
                         "noise trend in the F-K spectrum below (V = f / |k|).")
                if polygon_mode == "accept":
                    # The wedge IS the signal region by construction — correct as-is for accept.
                    active_polygon = filtering.apparent_velocity_wedge(v_cutoff, f_nyq, k_nyq)
                else:
                    # Reusing the signal wedge here would tell the filter the SIGNAL is the
                    # noise (backward). Build the wedge's own geometric complement — the two
                    # flanking noise regions — so reject traces the actual noise zone and
                    # produces an equivalent result to the accept-mode wedge.
                    active_polygon = filtering.apparent_velocity_noise_wedges(v_cutoff, f_nyq, k_nyq)
                poly_desc = f"velocity wedge, cutoff {v_cutoff:.0f} m/s"
            else:
                st.caption("Add vertices of a polygon in (k, f) space. For an **accept** polygon "
                            "trace the boundary of the reflection fan; for a **reject** polygon "
                            "trace the boundary of the noise (NOT the same shape as the accept "
                            "polygon). Zoom into the spectrum to read "
                            "off coordinates — Streamlit cannot deliver click events on a heatmap, "
                            "so points are entered numerically or edited in the table below.")
                new_pt = numeric_point_form(
                    "fk_pt", "Wavenumber k (cycles/m)", "Frequency (Hz)",
                    x_value=0.0, y_value=float(f_nyq) * 0.3,
                    x_min=-float(k_nyq), x_max=float(k_nyq),
                    y_min=0.0, y_max=float(f_nyq), x_step=0.001, y_step=1.0, x_format="%.4f")
                if new_pt is not None:
                    st.session_state["fk_click_points"].append(new_pt)
                    st.rerun()
                if st.button("Clear all points", key="fk_clear"):
                    st.session_state["fk_click_points"] = []
                    st.rerun()

                pts = st.session_state["fk_click_points"]

                # Editable table: review, hand-correct any vertex, delete a row, add rows
                # at the end, or REORDER via the explicit "order" column — vertex order
                # traces the polygon boundary, so being able to fix that order (not just
                # the k/f values) matters as much as editing the coordinates themselves.
                # Plain Streamlit has no drag-and-drop row reordering for `st.data_editor`,
                # so "order" is a plain editable number: set two rows to the same relative
                # order you want them traced in (e.g. swap their numbers, or type any value
                # between two existing ones) and the table re-sorts by it immediately below.
                st.caption("Edit k/f values, delete a row (trash icon), or add a new row at "
                            "the bottom. To REORDER vertices, edit the **order** column — "
                            "rows are always re-sorted by it, so typing a value between two "
                            "existing ones moves a vertex between them without retyping "
                            "everything else. Changes apply immediately below.")
                pts_df = pd.DataFrame(
                    # "order" is built as float64 from the start (not plain ints) so that
                    # typing a fractional value (e.g. 1.5) into the column doesn't hit a
                    # pandas int64->float64 LossySetitemError inside st.data_editor's own
                    # cell-edit handling — verified directly against this exact pattern.
                    {"order": [float(i) for i in range(1, len(pts) + 1)],
                     "k (cycles/m)": [p[0] for p in pts],
                     "f (Hz)": [p[1] for p in pts]})
                edited_df = st.data_editor(
                    pts_df, num_rows="dynamic", key="fk_points_editor", use_container_width=True,
                    hide_index=True,
                    column_config={
                        "order": st.column_config.NumberColumn(
                            step=0.5, format="%.2f",
                            help="Rows are sorted by this value — edit it to reorder vertices "
                                 "(fractional values, e.g. 1.5, insert a row between 1 and 2)."),
                        "k (cycles/m)": st.column_config.NumberColumn(
                            min_value=-float(k_nyq), max_value=float(k_nyq), step=0.001, format="%.4f"),
                        "f (Hz)": st.column_config.NumberColumn(
                            min_value=0.0, max_value=float(f_nyq), step=1.0, format="%.1f"),
                    })
                edited_df = edited_df.dropna(subset=["k (cycles/m)", "f (Hz)"])
                edited_df = edited_df.sort_values(
                    "order", kind="stable", na_position="last").reset_index(drop=True)
                st.session_state["fk_click_points"] = [
                    (float(r["k (cycles/m)"]), float(r["f (Hz)"])) for _, r in edited_df.iterrows()
                ]

                # Friendlier alternative to hand-editing "order" numbers: pick a gap in
                # human-readable terms ("Between row 2 and row 3") and insert a new vertex
                # there directly — covers the common case without needing to think in
                # fractional order values at all.
                pts = st.session_state["fk_click_points"]
                if len(pts) >= 1:
                    st.markdown("**Insert a point between existing rows**")
                    n = len(pts)
                    position_labels = ["Before row 1"] + [
                        f"Between row {i} and row {i + 1}" for i in range(1, n)] + [f"After row {n}"]
                    ic1, ic2, ic3, ic4 = st.columns([2, 1, 1, 1])
                    pos_label = ic1.selectbox("Insert position", position_labels, key="fk_insert_pos")
                    ins_k = ic2.number_input("k", -float(k_nyq), float(k_nyq), 0.0, step=0.001,
                                              format="%.4f", key="fk_insert_k")
                    ins_f = ic3.number_input("f", 0.0, float(f_nyq), float(f_nyq) * 0.3, step=1.0,
                                              key="fk_insert_f")
                    if ic4.button("Insert", key="fk_insert_btn"):
                        insert_idx = position_labels.index(pos_label)  # 0 = before row 1, n = after last
                        pts.insert(insert_idx, (float(ins_k), float(ins_f)))
                        st.session_state["fk_click_points"] = pts
                        st.rerun()

                st.caption(f"{len(st.session_state['fk_click_points'])} point(s). Need at least 3 "
                            "to form a usable polygon.")
                active_polygon = (list(st.session_state["fk_click_points"])
                                   if len(st.session_state["fk_click_points"]) >= 3 else None)
                poly_desc = f"custom polygon, {len(st.session_state['fk_click_points'])} vertices"
                if active_polygon is None:
                    st.info("Add at least 3 points to define a usable polygon.", icon="ℹ️")

            # live preview on the chosen shot (spectrum + polygon + applied multiplier)
            if active_polygon is not None:
                preview_poly = active_polygon
            elif polygon_mode == "accept":
                preview_poly = filtering.apparent_velocity_wedge(
                    cfg.FK_VELOCITY_CUTOFF_DEFAULT, f_nyq, k_nyq)
            else:
                preview_poly = filtering.apparent_velocity_noise_wedges(
                    cfg.FK_VELOCITY_CUTOFF_DEFAULT, f_nyq, k_nyq)
            _, pmask, f_axis, k_axis, fk_before, _ = filtering.fk_filter_gather(
                g_preview[in_key], dt, survey.group_interval, preview_poly,
                polygon_mode=polygon_mode, reject_coefficient=reject_coeff)
            st.plotly_chart(seismic_plots.plot_fk_spectrum(
                f_axis, k_axis, np.abs(fk_before), polygon_points=preview_poly,
                polygon_mode=polygon_mode,
                title=f"F-K Spectrum — shot {shot_fk} ({poly_desc}; {polygon_mode})"),
                theme=None, key="fk_spectrum_preview")
            # Spectral-energy-weighted readout: for the recommended wedge source, this should
            # now land close to the SAME value for accept vs reject (same cutoff, complementary
            # polygons) — confirming that both approaches normally produce the same results.
            # For a hand-drawn custom polygon, it will only match if you drew the
            # correct (complementary) shape for whichever mode you picked; reusing the exact
            # same hand-drawn points for both modes inverts the result, same as the wedge case
            # used to before this fix.
            spec_power = np.abs(fk_before) ** 2
            pct_kept = 100.0 * float(np.sum(spec_power * pmask) / max(np.sum(spec_power), 1e-12))
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Spectral energy kept by current mask", f"{pct_kept:.1f}%")
            mcol2.metric("Spectral energy attenuated", f"{100.0 - pct_kept:.1f}%")
            with st.expander("Multiplier that will be applied (0 = removed, 1 = passed)", expanded=True):
                st.plotly_chart(seismic_plots.plot_fk_pass_mask(f_axis, k_axis, pmask),
                                theme=None, key="fk_mask_preview")
                st.caption(f"Attenuated side is multiplied by {reject_coeff:.3f} (not zero); the "
                            f"polygon edge is Gaussian-tapered to avoid Gibbs ringing. Accept keeps "
                            f"the wedge WHITE (≈1) and floors everything outside it to {reject_coeff:.3f} "
                            f"(BLACK); Reject does the opposite — compare by toggling the radio above.")

            if wf.run_button("fk", "Run F-K filter", key="run_fk", disabled=active_polygon is None):
                prog = st.progress(0.0, text="F-K filtering gathers...")
                wf.execute(stage_ops.run_fk, active_polygon, polygon_mode, float(reject_coeff), poly_desc,
                           lambda f: prog.progress(min(1.0, f), text="F-K filtering gathers..."))

            e = wf.result("fk")

            if e is not None:

                wf.result_note(e)
                fkc = {"mode": e.params["mode"], "coeff": e.params["coeff"], "source": e.info.get("source", "")}
                st.success(f"✅ F-K filter applied — {fkc.get('mode', '?')} polygon, "
                            f"{fkc.get('source', '')}, attenuation coefficient {fkc.get('coeff', 0):.3f}.")
                shot = _example_shot_selector(shot_list, "fk_shot")
                g = e.gathers[shot]
                _, _, f_ax2, k_ax2, fk_b, fk_a = filtering.fk_filter_gather(
                    g[e.in_key], dt, survey.group_interval, e.params["polygon"],
                    polygon_mode=fkc.get("mode", "accept"),
                    reject_coefficient=fkc.get("coeff", cfg.FK_REJECT_COEFFICIENT_DEFAULT))
                st.plotly_chart(seismic_plots.plot_fk_before_after(
                    f_ax2, k_ax2, np.abs(fk_b), np.abs(fk_a), polygon_points=e.params["polygon"],
                    polygon_mode=fkc.get("mode", "accept")),
                    theme=None, key="fk_spectrum_before_after")
                disp = controls.section_display_controls("fk_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                    title_before=_before_label(e), title_after="After F-K", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="fk_before_after")
                st.markdown("**Removed energy** (before − after). This should look like the linear "
                             "ground-roll / direct-wave trend and contain no reflection hyperbolas; "
                             "if reflections appear here the polygon is cutting into the signal.")
                st.plotly_chart(seismic_plots.plot_section(
                    g[e.in_key] - g[e.out_key], g["receiver_ids"], twt,
                    title="Removed by F-K filter", x_title="Receiver station",
                    colorscale=disp.colorscale, pclip=disp.pclip, equalize=disp.equalize),
                    theme=None, key="fk_removed")

    # ---- 7) Multiple suppression (predictive/gap deconvolution) ------
    with tabs[7]:
        st.markdown("**Predictive (gap) deconvolution**: rather than collapsing the whole "
                     "wavelet (spiking decon, next step), a prediction lag greater than one "
                     "sample preserves the primary wavelet while predicting and subtracting "
                     "the periodic reverberation tail caused by short-period multiples — the "
                     "standard land-data multiple-suppression tool (the marine equivalent, "
                     "SRME, needs a free-surface model this app doesn't have — see README). "
                     "A practical starting point for the prediction lag is the *second "
                     "zero-crossing* of the autocorrelation.")
        chk = wf.show_status("mult_supp")
        if chk.can_run:
            pg, in_key, _ = wf.input_gathers(chk)
            shot_ms = _example_shot_selector(shot_list, "ms_suggest_shot")
            suggested_lag_s = preprocessing.second_zero_crossing_lag(
                pg[shot_ms][in_key][0], dt)
            st.caption(f"Suggested prediction lag for shot {shot_ms}, trace 0: "
                        f"≈{suggested_lag_s * 1000:.1f} ms ({int(round(suggested_lag_s / dt))} samples)")
            c1, c2, c3 = st.columns(3)
            oplen = c1.slider("Operator length (samples)", 8, 150, 60, key="ms_oplen")
            # The suggested lag is measured on whatever data this step will read, so the widget is
            # keyed by that source: if the shot-domain data changes (another step is applied, or
            # an earlier one re-run) the default is re-derived instead of staying at the value it
            # had when the tab was first drawn on different data.
            lag_samples = c2.slider("Prediction lag (samples)", 2, 100,
                                     max(2, int(round(suggested_lag_s / dt))),
                                     key=f"ms_lag__{chk.inputs['data']}")
            pw = c3.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_PW_DEFAULT,
                                  step=0.005, format="%.3f", key="ms_pw")
            if wf.run_button("mult_supp", "Run multiple suppression", key="run_ms"):
                prog = st.progress(0.0, text="Suppressing multiples...")
                wf.execute(stage_ops.run_mult_supp, oplen, lag_samples, pw,
                           lambda f: prog.progress(min(1.0, f), text="Suppressing multiples..."))
            e = wf.result("mult_supp")
            if e is not None:
                wf.result_note(e)
                st.success("✅ Multiple suppression applied.")
                shot = _example_shot_selector(shot_list, "ms_shot")
                g = e.gathers[shot]
                disp = controls.section_display_controls("ms_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                    title_before=_before_label(e), title_after="After multiple suppression",
                    x_title="Receiver station", colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="ms_before_after")

    # ---- 8) Deconvolution (spiking) -----------------------------------
    with tabs[8]:
        st.markdown("Spiking deconvolution (prediction lag = 1 sample) compresses the "
                     "remaining source wavelet toward a zero-lag spike, sharpening temporal "
                     "resolution. Optional — and independent of every other step here.")
        wf.show_status("decon")
        oplen = st.slider("Operator length (samples)", 4, 120, cfg.DECON_OPLEN_DEFAULT, key="decon_oplen")
        pw = st.number_input("Pre-whitening (fraction)", 0.001, 0.5, cfg.DECON_PW_DEFAULT,
                              step=0.005, format="%.3f", key="decon_pw")
        if wf.run_button("decon", "Run deconvolution", key="run_decon"):
            prog = st.progress(0.0, text="Deconvolving gathers...")
            wf.execute(stage_ops.run_decon, oplen, pw,
                       lambda f: prog.progress(min(1.0, f), text="Deconvolving gathers..."))
        e = wf.result("decon")
        if e is not None:
            wf.result_note(e)
            st.success("✅ Deconvolution applied. Continue to **CDP Sort & Binning** whenever the "
                        "conditioning you want is done.")
            shot = _example_shot_selector(shot_list, "decon_shot")
            g = e.gathers[shot]
            disp = controls.section_display_controls("decon_display", default_colorscale="Seismic")
            st.plotly_chart(seismic_plots.plot_before_after(
                g[e.in_key], g[e.out_key], g["receiver_ids"], twt,
                title_before=_before_label(e), title_after="After decon", x_title="Receiver station",
                colorscale=disp.colorscale, equalize=disp.equalize),
                theme=None, key="decon_before_after")
            st.plotly_chart(spectra_plots.plot_spectrum_comparison(
                g[e.in_key], g[e.out_key], dt, label_before="Before", label_after="After",
                title="Spectrum — Deconvolution whitens the passband"), theme=None, key="decon_spectrum")
