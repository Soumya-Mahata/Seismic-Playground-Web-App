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
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from modules import filtering, preprocessing
from utils import session_state as ss
from utils.click_picking import numeric_point_form
from visualization import controls, seismic_plots, spectra_plots, theme


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
            f"De-spike / Mute real problems to fix (report Sec 5.7): **dead channels** at receiver "
            f"station(s) {sorted(dead_stations) or 'none'}, **powerline harmonic noise** "
            f"({fn_info.get('powerline_freq_hz', '?')} Hz) at station(s) {sorted(powerline_stations) or 'none'}, "
            f"and **{len(fn_info.get('spike_locations', []))} spike bursts** scattered through the "
            f"dataset. The QC shot selectors below default to a shot that actually has one of these.",
            icon="🧪")

    with st.expander(f"Trace-count check: {n_traces_in_shots:,} traces across {len(shot_list)} shots "
                      f"(survey has {survey.num_traces:,})", expanded=True):
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
                    "4. Mute", "5. Trace Balancing", "6. F-K Filter", "7. Multiple Suppression",
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
            st.markdown(
                "Sliding-window automatic gain control compensates for amplitude decay with time "
                "(spherical divergence + intrinsic attenuation). Report Sec 5.6.3 describes three "
                "scalar-calculation variants — **mean**, **RMS**, and **median** AGC — noting they "
                "\"normally produce similar\" but not identical results. Compare all three below on an "
                "example shot, then pick the one to use for every later step; only **one** version is "
                "ever active downstream — no hidden second copy.")
            window = st.slider("AGC window (samples)", 20, 800, cfg.AGC_WINDOW_DEFAULT, key="agc_window")
            shot_preview = _example_shot_selector(shot_list, "agc_preview_shot")
            g_preview = shot_gathers[shot_preview]
            previews = {m: preprocessing.agc(g_preview["traces_bp"], window=window, mode=m)
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

            mode = st.radio("AGC version to use for every later step", ["mean", "rms", "median"],
                             index=["mean", "rms", "median"].index(cfg.AGC_MODE_DEFAULT),
                             horizontal=True, key="agc_mode",
                             help="Report Sec 5.6.3: \"the mean AGC is preferred in seismic data "
                                  "processing applications\", though all three are offered here since "
                                  "the difference can matter for a given dataset.")

            if st.button(f"Apply {mode.upper()} AGC to all shots", type="primary", key="run_agc"):
                for g in shot_gathers.values():
                    for m in ("mean", "rms", "median"):
                        g.pop(f"traces_agc_{m}", None)
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
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_bp"], g["traces_agc"], g["receiver_ids"], twt,
                    title_before="Before AGC", title_after=f"After {cfg_active.get('mode', '?').upper()} AGC",
                    x_title="Receiver station", colorscale=disp.colorscale, equalize=disp.equalize),
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
                shot = _example_shot_selector(shot_list, "despike_shot", default_shot=default_spike_shot)
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
                st.markdown(
                    "**3b. Trace edit / kill** — flags whole traces that are dead (near-zero energy) "
                    "or abnormally noisy relative to the gather's median trace energy. **Detection runs "
                    "on the pre-AGC band-passed data (`traces_bp`), not the AGC output** — AGC is a "
                    "per-trace self-normalizing gain, so after it every trace (dead, noisy, or healthy) "
                    "ends up at very nearly the same level by construction, which would make this "
                    "energy-based detection blind to real dead/noisy channels (report Sec 5.6.3: AGC "
                    "\"completely removes relative amplitude information\"). The resulting kill mask is "
                    "then applied to the de-spiked data so the two steps still compose correctly.")
                c1, c2 = st.columns(2)
                dead_frac = c1.slider("Dead-trace energy threshold (x median)", 0.001, 0.5, 0.02, key="dead_frac")
                noisy_mult = c2.slider("Noisy-trace energy threshold (x median)", 2.0, 30.0, 8.0, key="noisy_mult")
                if st.button("Run trace kill", type="primary", key="run_kill"):
                    total_killed = 0
                    for g2 in shot_gathers.values():
                        flags = preprocessing.flag_bad_traces(g2["traces_bp"], dead_frac, noisy_mult)
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
                shot = _example_shot_selector(shot_list, "edit_shot", default_shot=default_edit_shot)
                g = shot_gathers[shot]
                flags = g["_kill_flags"]
                c1, c2 = st.columns(2)
                c1.metric("Dead traces (this shot)", int(flags["dead"].sum()))
                c2.metric("Noisy traces (this shot)", int(flags["noisy"].sum()))
                if flags["kill"].any():
                    killed_ids = g["receiver_ids"][flags["kill"]]
                    st.caption(f"Killed receiver station(s) in shot {shot}: {list(killed_ids)}")
                else:
                    st.caption("No dead/noisy traces flagged in this shot — try the shot the info box "
                                "above points to if you expected to see one flagged.")
                with st.expander("Why these traces were flagged (energy vs. median, on pre-AGC data)"):
                    energy_fig = go.Figure()
                    colors = np.where(flags["dead"], "#ef4444", np.where(flags["noisy"], "#f59e0b", "#0D2B5E"))
                    energy_fig.add_trace(go.Bar(x=[str(r) for r in g["receiver_ids"]], y=flags["energy"],
                                                  marker_color=list(colors)))
                    energy_fig.add_hline(y=float(np.median(flags["energy"])), line_dash="dash",
                                          annotation_text="median")
                    energy_fig.update_layout(**theme.base_layout(
                        "Per-trace energy (traces_bp) — red=dead, orange=noisy, blue=kept", 320))
                    energy_fig.update_xaxes(**theme.x_axis("Receiver station"))
                    energy_fig.update_yaxes(**theme.x_axis("Energy"), type="log")
                    st.plotly_chart(energy_fig, theme=None, key="kill_energy_bar")
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
                "Report Sec 5.8 distinguishes three mute types. Two are implemented here, in the "
                "shot domain where the report applies them; the third happens automatically later:\n"
                "- **Top mute** (Fig 5.57A): removes the water-column / pre-first-break noise. Its "
                "cutoff follows the **direct arrival's moveout** — a straight line that gets later "
                "with offset — not the same fixed time on every trace; a flat top mute does nothing "
                "to that diagonal noise trend at all.\n"
                "- **Surgical mute** (Fig 5.57B/5.60): zeroes a specific (offset, time) box of noise "
                "*inside* the gather that isn't easy to remove with a filter (e.g. localized "
                "high-frequency harmonic noise).\n"
                "- **NMO-stretch mute** (Fig 5.57C) is applied automatically to CDP gathers right "
                "after NMO correction, in the Velocity, NMO & DMO page — it doesn't belong here since "
                "it needs NMO-corrected data that doesn't exist yet at this stage of the pipeline."
            )
            mute_type = st.radio("Mute type", ["Top mute (follows direct-arrival moveout)",
                                                  "Top mute (flat — simpler, not offset-aware)",
                                                  "Surgical mute (remove a noisy box region)"],
                                  key="mute_type")
            taper = st.slider("Taper width (s)", 0.005, 0.3, cfg.MUTE_TAPER_DEFAULT, step=0.005, key="mute_taper")

            if mute_type.startswith("Top mute (flat"):
                t_top = st.slider("Mute time (s)", 0.0, min(0.5, survey.record_length), cfg.MUTE_TTOP_DEFAULT,
                                   step=0.005, key="mute_ttop")
                mute_kwargs = dict(kind="flat", t_top=t_top, taper=taper)
                summary = f"flat top mute, t_top={t_top * 1000:.0f} ms"
            elif mute_type.startswith("Top mute (follows"):
                c1, c2 = st.columns(2)
                v_mute = c1.slider("Mute velocity (m/s)", 200.0, 3000.0, 500.0, step=50.0, key="mute_velocity",
                                    help="Should roughly match the apparent velocity of the direct "
                                         "arrival / ground roll you want to remove — see its slope on "
                                         "the QC plot below.")
                t0_mute = c2.slider("Time intercept t0 (s)", 0.0, 0.3, 0.0, step=0.005, key="mute_t0")
                mute_kwargs = dict(kind="linear", velocity=v_mute, t0=t0_mute, taper=taper)
                summary = f"top mute (moveout), v={v_mute:.0f} m/s, t0={t0_mute * 1000:.0f} ms"
            else:
                st.caption("Pick an offset range and a time range to zero out — e.g. draw a box "
                            "around a localized noise burst visible on the QC plot for an example "
                            "shot below.")
                shot_sm = _example_shot_selector(shot_list, "surgical_preview_shot")
                g_sm = shot_gathers[shot_sm]
                max_off = float(np.max(np.abs(g_sm["offsets"])))
                c1, c2 = st.columns(2)
                x_range = c1.slider("Offset range (m, absolute)", 0.0, max_off, (0.0, max_off * 0.3),
                                     key="surgical_xrange")
                t_range = c2.slider("Time range (s)", 0.0, float(survey.record_length),
                                     (0.0, min(0.3, float(survey.record_length))), key="surgical_trange")
                mute_kwargs = dict(kind="surgical", x_min=x_range[0], x_max=x_range[1],
                                     t_min=t_range[0], t_max=t_range[1], taper=taper)
                summary = (f"surgical mute, offset [{x_range[0]:.0f},{x_range[1]:.0f}] m, "
                            f"time [{t_range[0]*1000:.0f},{t_range[1]*1000:.0f}] ms")
                disp_sm = controls.section_display_controls("surgical_display", default_colorscale="Seismic")
                fig_sm = seismic_plots.plot_section(g_sm["traces_edited"], g_sm["offsets"], twt,
                                                      title=f"Shot {shot_sm} — surgical mute box preview",
                                                      x_title="Offset (m)", colorscale=disp_sm.colorscale,
                                                      equalize=disp_sm.equalize)
                fig_sm.add_shape(type="rect", x0=x_range[0], x1=x_range[1], y0=t_range[0], y1=t_range[1],
                                  line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                fig_sm.add_shape(type="rect", x0=-x_range[1], x1=-x_range[0], y0=t_range[0], y1=t_range[1],
                                  line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)")
                st.plotly_chart(fig_sm, theme=None, key="surgical_preview")

            if st.button("Run mute", type="primary", key="run_mute"):
                for g in shot_gathers.values():
                    if mute_kwargs["kind"] == "flat":
                        muted = preprocessing.apply_top_bottom_mute(
                            g["traces_edited"], dt, mute_kwargs["t_top"], taper_s=mute_kwargs["taper"])
                        mute_times = np.full(g["traces_edited"].shape[0], mute_kwargs["t_top"])
                        in_box = None
                    elif mute_kwargs["kind"] == "linear":
                        muted, mute_times = preprocessing.apply_linear_mute(
                            g["traces_edited"], g["offsets"], dt, mute_kwargs["velocity"],
                            t_intercept=mute_kwargs["t0"], taper_s=mute_kwargs["taper"])
                        in_box = None
                    else:
                        muted, in_box = preprocessing.apply_surgical_mute(
                            g["traces_edited"], g["offsets"], dt,
                            t_min=mute_kwargs["t_min"], t_max=mute_kwargs["t_max"],
                            x_min=mute_kwargs["x_min"], x_max=mute_kwargs["x_max"],
                            taper_s=mute_kwargs["taper"])
                        mute_times = np.full(g["traces_edited"].shape[0], mute_kwargs["t_max"])
                    g["traces_muted"] = muted
                    g["_mute_times"] = mute_times
                    g["_surgical_box"] = in_box
                st.session_state["mute_config"] = {**mute_kwargs, "summary": summary}
                ss.mark_stage_done("mute", {"type": mute_kwargs["kind"],
                                              **{k: v for k, v in mute_kwargs.items() if k != "kind"}})
                st.rerun()

            if ss.is_stage_done("mute"):
                st.success(f"✅ Mute applied — **active mute: {summary}**.")
                shot = _example_shot_selector(shot_list, "mute_shot")
                g = shot_gathers[shot]
                disp = controls.section_display_controls("mute_display", default_colorscale="Seismic")
                mcfg = st.session_state["mute_config"]
                if mcfg["kind"] in ("flat", "linear"):
                    st.markdown("**Before, with the mute cutoff line overlaid** — everything below/right "
                                 "of the dashed line is removed.")
                    st.plotly_chart(seismic_plots.plot_mute_qc(
                        g["traces_edited"], g["receiver_ids"], twt, g["_mute_times"],
                        title="Before mute (cutoff overlaid)", x_title="Receiver station",
                        colorscale=disp.colorscale, equalize=disp.equalize),
                        theme=None, key="mute_qc_overlay")
                else:
                    st.markdown(f"**Before, with the surgical mute box overlaid** "
                                 f"({int(np.sum(g['_surgical_box']))} of {len(g['_surgical_box'])} "
                                 f"traces in this shot fall inside the chosen offset range).")
                    fig_box = seismic_plots.plot_section(g["traces_edited"], g["offsets"], twt,
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
                    x_title_ba = "Offset (m)"
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_edited"], g["traces_muted"],
                    g["offsets"] if mcfg["kind"] == "surgical" else g["receiver_ids"], twt,
                    title_before="Before", title_after="After mute",
                    x_title="Offset (m)" if mcfg["kind"] == "surgical" else "Receiver station",
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

    # ---- 6) F-K dip filter (accept / reject polygon) -------------------
    with tabs[5]:
        if not ss.require_stage("balance"):
            pass
        else:
            st.markdown(
                "A 2-D FFT turns a gather into frequency-wavenumber (F-K) space, where an event of "
                "apparent velocity *V* lies on the line `|k| = f / V` (report Sec 5.9, Fig. 5.64). "
                "Primary reflections are fast (high apparent velocity) and cluster in a fan around the "
                "frequency axis; ground roll, direct wave and refractions are slow and appear as steep "
                "diagonal lines far from it. The filter multiplies the amplitude spectrum by a mask "
                "and leaves the phase untouched (Fig. 5.66). The polygon is either an **accept** "
                "polygon (encloses the signal — preferred in practice, *\"it is easier to enclose the "
                "signal zone\"*) or a **reject** polygon (encloses the noise); the attenuated side is "
                "multiplied by a small coefficient, not exactly zero, per the Fourier-theory note in "
                "Fig. 5.66."
            )
            shot_fk = _example_shot_selector(shot_list, "fk_preview_shot")
            g_preview = shot_gathers[shot_fk]
            f_nyq, k_nyq = 0.5 / dt, 0.5 / survey.group_interval
            st.caption(
                f"This survey's F-K axes span k = ±{k_nyq:.4f} cycles/m and f = 0–{f_nyq:.0f} Hz — "
                f"k_max = 1/(2·dx) and f_max = 1/(2·dt) for THIS survey's actual trace spacing "
                f"(dx = {survey.group_interval:.2f} m) and sample interval (dt = {dt*1000:.1f} ms). "
                "The default 6.25 m trace spacing already matches the ±0.08 cycles/m wavenumber "
                "range used in the reference material's own f-k polygon figures (fk_filter.pdf "
                "Figs. 5.64/5.67), so the 'Apparent-velocity wedge' polygon below tapers across the "
                "spectrum the same way theirs does. A coarser spacing (a larger dx on the Data "
                "Management page, e.g. a real 25 m survey) narrows this k-axis and makes the wedge "
                "saturate to full width much earlier — physics, not a display setting."
            )

            c1, c2 = st.columns(2)
            polygon_mode_label = c1.radio(
                "Polygon type", ["Accept polygon (encloses signal — preferred)", "Reject polygon (encloses noise)"],
                key="fk_polygon_type",
                help="These are opposite operations for the SAME boundary drawn below: Accept keeps "
                     "the inside of the polygon and attenuates everything outside it; Reject does the "
                     "reverse — keeps the outside and attenuates the inside. The report's note that "
                     "'both approaches normally produce the same results' (Sec 5.9) means the two "
                     "*polygon shapes* can be drawn as complements of each other to describe the same "
                     "filter design — not that flipping this toggle while keeping the exact same "
                     "boundary leaves the output unchanged. Watch the 'Multiplier that will be "
                     "applied' panel below: it visibly inverts (black/white swap) when you toggle this.")
            polygon_mode = "accept" if polygon_mode_label.startswith("Accept") else "reject"
            poly_source = c2.radio("Polygon source", ["Apparent-velocity wedge (recommended)",
                                                        "Custom (numeric points)"],
                                    key="fk_poly_source")
            reject_coeff = st.slider(
                "Attenuation coefficient (multiplier on the attenuated side)", 0.0, 0.3,
                cfg.FK_REJECT_COEFFICIENT_DEFAULT, step=0.005, key="fk_reject_coeff",
                help="A small non-zero constant, not exactly 0 (report Fig. 5.66: multiplying by a "
                     "constant in the frequency domain scales the time-domain function by its "
                     "inverse, 'and the inverse of zero is simply infinite'). Lower = stronger "
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
                    # produces the report's claimed equivalent result (fk_filter.pdf Sec 5.9).
                    active_polygon = filtering.apparent_velocity_noise_wedges(v_cutoff, f_nyq, k_nyq)
                poly_desc = f"velocity wedge, cutoff {v_cutoff:.0f} m/s"
            else:
                st.caption("Add vertices of a polygon in (k, f) space. For an **accept** polygon "
                            "trace the boundary of the reflection fan; for a **reject** polygon "
                            "trace the boundary of the noise (NOT the same shape as the accept "
                            "polygon — see Fig. 5.67 panels A vs C). Zoom into the spectrum to read "
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

                # Note on drag-and-drop: plain Streamlit's st.data_editor has no built-in
                # drag-to-reorder handle for table rows — that interaction isn't something this
                # framework exposes, so it can't be added here. Two concrete alternatives are
                # provided instead: a guided "insert at position" control (below, no arithmetic
                # needed) and a direct numeric "order" column in the table itself.
                st.markdown("**Insert a point between existing rows**")
                pts_list = st.session_state["fk_click_points"]
                ins_c1, ins_c2, ins_c3, ins_c4 = st.columns([1.3, 1, 1, 0.8])
                insert_positions = list(range(len(pts_list) + 1))
                insert_labels = [
                    "At the start" if p == 0
                    else "At the end" if p == len(pts_list)
                    else f"Between row {p} and row {p + 1}"
                    for p in insert_positions
                ]
                insert_at = ins_c1.selectbox("Position", insert_positions,
                                              format_func=lambda p: insert_labels[insert_positions.index(p)],
                                              index=len(insert_positions) - 1, key="fk_insert_pos")
                insert_k = ins_c2.number_input("k (cycles/m)", -float(k_nyq), float(k_nyq), 0.0,
                                                step=0.001, format="%.4f", key="fk_insert_k")
                insert_f = ins_c3.number_input("f (Hz)", 0.0, float(f_nyq), float(f_nyq) * 0.3,
                                                step=1.0, format="%.1f", key="fk_insert_f")
                if ins_c4.button("Insert", key="fk_insert_btn"):
                    pts_list.insert(insert_at, (float(insert_k), float(insert_f)))
                    st.session_state["fk_click_points"] = pts_list
                    st.rerun()

                # Editable table: review, hand-correct any vertex, delete a row, or add new
                # rows directly — far smoother than entering points one at a time with no way
                # to see or fix what's already been added (the cramped, erratic polygon shape
                # from manual-only entry showed exactly this need).
                #
                # The table itself still has no native drag-to-reorder handle (see note above),
                # so reordering/inserting a point between two existing rows straight from the
                # table is done through an explicit, editable "order" column instead: type 1.5
                # to place a point between rows 1 and 2, or renumber rows outright. Rows are
                # re-sorted by this column below and it is redrawn as a clean 0..n-1 sequence
                # every run — this preserves whatever order the user set no matter how they
                # edited, deleted, or added rows in the same pass.
                st.caption("Drag-and-drop reordering isn't available in this table widget. Use "
                            "the 'Insert a point' control above, or edit k/f values directly, "
                            "delete a row (checkbox + Delete), or add a row with the '+' button "
                            "at the bottom (it appears at the END — give it an 'order' value to "
                            "place it anywhere, e.g. 1.5 to insert between rows 1 and 2). Vertex "
                            "order matters: it traces the polygon boundary.")
                pts_list = st.session_state["fk_click_points"]
                pts_df = pd.DataFrame({
                    "order": list(range(len(pts_list))),
                    "k (cycles/m)": [p[0] for p in pts_list],
                    "f (Hz)": [p[1] for p in pts_list],
                })
                edited_df = st.data_editor(
                    pts_df, num_rows="dynamic", key="fk_points_editor", use_container_width=True,
                    column_config={
                        "order": st.column_config.NumberColumn(
                            "order (edit to reorder/insert)", step=0.5, format="%.2f"),
                        "k (cycles/m)": st.column_config.NumberColumn(
                            min_value=-float(k_nyq), max_value=float(k_nyq), step=0.001, format="%.4f"),
                        "f (Hz)": st.column_config.NumberColumn(
                            min_value=0.0, max_value=float(f_nyq), step=1.0, format="%.1f"),
                    })
                edited_df = edited_df.dropna(subset=["k (cycles/m)", "f (Hz)"])
                # A brand-new row from the "+" button has no order value yet — send it to the
                # end rather than dropping it or letting it sort unpredictably as NaN.
                missing = edited_df["order"].isna()
                if missing.any():
                    next_order = edited_df["order"].max()
                    next_order = 0.0 if pd.isna(next_order) else next_order + 1.0
                    for idx in edited_df.index[missing]:
                        edited_df.loc[idx, "order"] = next_order
                        next_order += 1.0
                edited_df = edited_df.sort_values("order", kind="stable")
                st.session_state["fk_click_points"] = [
                    (float(r["k (cycles/m)"]), float(r["f (Hz)"]))
                    for _, r in edited_df.iterrows()
                ]
                st.caption(f"{len(st.session_state['fk_click_points'])} point(s), traced in the "
                            "order shown above (top to bottom). Need at least 3 to form a usable "
                            "polygon.")
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
                g_preview["traces_balanced"], dt, survey.group_interval, preview_poly,
                polygon_mode=polygon_mode, reject_coefficient=reject_coeff)
            st.plotly_chart(seismic_plots.plot_fk_spectrum(
                f_axis, k_axis, np.abs(fk_before), polygon_points=preview_poly,
                polygon_mode=polygon_mode,
                title=f"F-K Spectrum — shot {shot_fk} ({poly_desc}; {polygon_mode})"),
                theme=None, key="fk_spectrum_preview")
            # Spectral-energy-weighted readout: for the recommended wedge source, this should
            # now land close to the SAME value for accept vs reject (same cutoff, complementary
            # polygons) — confirming the report's "both approaches normally produce the same
            # results." For a hand-drawn custom polygon, it will only match if you drew the
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

            if st.button("Run F-K filter", type="primary", key="run_fk", disabled=active_polygon is None):
                st.session_state["fk_polygon"] = list(active_polygon)
                st.session_state["fk_config"] = {"mode": polygon_mode, "coeff": float(reject_coeff),
                                                   "source": poly_desc}
                prog = st.progress(0.0, text="F-K filtering gathers...")
                items = list(shot_gathers.items())
                for i, (shot_id, g) in enumerate(items):
                    filt, _, _, _, _, _ = filtering.fk_filter_gather(
                        g["traces_balanced"], dt, survey.group_interval, active_polygon,
                        polygon_mode=polygon_mode, reject_coefficient=reject_coeff)
                    g["traces_fk_filtered"] = filt
                    if i % 20 == 0:
                        prog.progress((i + 1) / len(items))
                prog.progress(1.0)
                n_vertices = (sum(len(ring) for ring in active_polygon)
                              if active_polygon and isinstance(active_polygon[0][0], (list, tuple))
                              else len(active_polygon))
                ss.mark_stage_done("fk", {"polygon_vertices": n_vertices, "mode": polygon_mode,
                                            "coefficient": reject_coeff, "source": poly_desc})
                st.rerun()

            if ss.is_stage_done("fk"):
                fkc = st.session_state.get("fk_config") or {}
                st.success(f"✅ F-K filter applied — {fkc.get('mode', '?')} polygon, "
                            f"{fkc.get('source', '')}, attenuation coefficient {fkc.get('coeff', 0):.3f}.")
                shot = _example_shot_selector(shot_list, "fk_shot")
                g = shot_gathers[shot]
                _, _, f_ax2, k_ax2, fk_b, fk_a = filtering.fk_filter_gather(
                    g["traces_balanced"], dt, survey.group_interval, st.session_state["fk_polygon"],
                    polygon_mode=fkc.get("mode", "accept"),
                    reject_coefficient=fkc.get("coeff", cfg.FK_REJECT_COEFFICIENT_DEFAULT))
                st.plotly_chart(seismic_plots.plot_fk_before_after(
                    f_ax2, k_ax2, np.abs(fk_b), np.abs(fk_a), polygon_points=st.session_state["fk_polygon"],
                    polygon_mode=fkc.get("mode", "accept")),
                    theme=None, key="fk_spectrum_before_after")
                disp = controls.section_display_controls("fk_display", default_colorscale="Seismic")
                st.plotly_chart(seismic_plots.plot_before_after(
                    g["traces_balanced"], g["traces_fk_filtered"], g["receiver_ids"], twt,
                    title_before="Before F-K", title_after="After F-K", x_title="Receiver station",
                    colorscale=disp.colorscale, equalize=disp.equalize),
                    theme=None, key="fk_before_after")
                st.markdown("**Removed energy** (before − after). This should look like the linear "
                             "ground-roll / direct-wave trend and contain no reflection hyperbolas; "
                             "if reflections appear here the polygon is cutting into the signal "
                             "(report Sec 5.9.2, QC in f-k filtering).")
                st.plotly_chart(seismic_plots.plot_section(
                    g["traces_balanced"] - g["traces_fk_filtered"], g["receiver_ids"], twt,
                    title="Removed by F-K filter", x_title="Receiver station",
                    colorscale=disp.colorscale, pclip=disp.pclip, equalize=disp.equalize),
                    theme=None, key="fk_removed")

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
