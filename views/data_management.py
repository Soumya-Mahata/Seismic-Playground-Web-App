"""
views/data_management.py — Load Data, Inspect Headers, Build Shot Gathers
============================================================================
Covers the "upload" and "shot_gathers" pipeline stages, plus early raw-
data QC (trace viewer, amplitude spectrum, amplitude stats), an optional
geometry merge, and an optional "Brute Stack" / "Fold Map" early
geometry sanity-check — standard-workflow steps that sit before any real
processing (see README's industry-standard-workflow cross-check).

Note: CDP gathers are *not* built here. In a standard workflow, pre-
stack conditioning runs on shot gathers first; the formal CDP sort/
binning happens later, right before velocity analysis (see
`views/cdp_sort_view.py`). The "brute stack" below builds a throwaway
CDP grouping purely for an early QC picture — it is never used by, or
confused with, the official CDP gathers built later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from modules import nmo, segy_io, synthetic_data
from modules.stacking import fold_normalized_stack
from modules.validation import ValidationError, validate_survey
from utils import session_state as ss
from utils.formatting import fmt_ms
from visualization import seismic_plots, spectra_plots


def _load_survey(survey, source_label: str):
    ss.invalidate_from("upload", include_self=True)
    st.session_state["survey"] = survey
    ss.mark_stage_done("upload", params={"source": source_label, "traces": survey.num_traces})
    st.rerun()


def render():
    st.header("📂 Data Management")
    st.caption("Load a SEG-Y survey, inspect its headers and geometry, then build shot gathers "
                "for pre-stack conditioning.")

    survey = st.session_state["survey"]

    with st.expander("① Load data", expanded=survey is None):
        tab_upload, tab_demo = st.tabs(["Upload SEG-Y", "Synthetic demo data"])

        with tab_upload:
            uploaded = st.file_uploader("SEG-Y file (.sgy / .segy / .su)", type=["sgy", "segy", "su"])
            if uploaded is not None and uploaded.size > cfg.LARGE_DATASET_WARN_MB * 1_000_000:
                st.warning(f"⚠️ This file is {uploaded.size / 1e6:.0f} MB — loading and processing "
                            f"may take a while in a browser-based app.", icon="🐢")
            if uploaded is not None and st.button("Load this file", type="primary"):
                try:
                    path = segy_io.materialize_upload(uploaded)
                    new_survey = segy_io.read_segy(path, source_name=uploaded.name)
                    validate_survey(new_survey.num_traces, new_survey.num_samples, new_survey.sample_interval)
                    _load_survey(new_survey, uploaded.name)
                except ValidationError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Couldn't read this file as SEG-Y: {e}")

        with tab_demo:
            st.markdown(
                "No SEG-Y file handy? Generate a small **synthetic** 2D land survey with known "
                "ground-truth reflectors — useful for trying the whole workflow end to end. "
                "Every processing step still runs a real computation on this data; only the "
                "input itself is synthetic, and it's labeled as such everywhere in the app."
            )
            col1, col2, col3 = st.columns(3)
            n_stations = col1.number_input("Receiver stations", 40, 300, 160, step=10)
            shot_interval = col2.number_input("Shot interval (stations)", 1, 10, 2)
            channels = col3.number_input("Channels per side (≈ half of channels per shot)", 10, 90, 60, step=5,
                                          help="This is a split/centre-spread survey (summer report "
                                               "Sec 5.5.1.1): each shot sits in the middle of its "
                                               "receiver patch and records on BOTH sides at once, so "
                                               "the actual channel count for one shot is roughly 2x "
                                               "this number (fewer near the survey edges, where the "
                                               "patch runs off the end of the line). That is real "
                                               "acquisition geometry, not a duplication bug — the "
                                               "'Build shot gathers' step below verifies the total "
                                               "trace count explicitly.")
            st.caption(f"→ a mid-line shot will show up to ~{2*int(channels)} channels/shot below "
                       f"(2 × {int(channels)}), tapering down near the survey's edges.")
            col4, col5 = st.columns(2)
            dx = col4.number_input(
                "Trace/group spacing dx (m)", 5.0, 50.0, 6.25, step=1.25, format="%.2f",
                help="Receiver spacing along the line. This sets the F-K spectrum's wavenumber "
                     "Nyquist limit seen on the Pre-Stack Conditioning page's F-K filter tab: "
                     "k_max = 1 / (2·dx). The default 6.25 m gives k_max = 0.08 cycles/m, matching "
                     "the wavenumber range used in the reference material's own f-k polygon figures "
                     "(fk_filter.pdf Figs. 5.64/5.67) so the default 'Apparent-velocity wedge' "
                     "polygon there tapers across the spectrum the same way theirs does, rather "
                     "than saturating to the full wavenumber width within the first ~15% of the "
                     "frequency axis the way a coarser real-world spacing (e.g. 25 m, k_max = 0.02) "
                     "would. Raise dx here for a coarser, more aliasing-prone survey instead.")
            dt_ms = col5.number_input(
                "Sample interval dt (ms)", 1.0, 4.0, 2.0, step=0.5, format="%.1f",
                help="Time sampling. Sets the F-K spectrum's frequency Nyquist: f_max = 1 / (2·dt). "
                     "2 ms (the default) gives f_max = 250 Hz.")
            k_nyq_preview = 0.5 / dx
            f_nyq_preview = 0.5 / (dt_ms / 1000.0)
            st.caption(f"→ this survey's F-K spectrum will span k = ±{k_nyq_preview:.4f} cycles/m "
                       f"and f = 0–{f_nyq_preview:.0f} Hz.")
            inject_noise = st.checkbox(
                "Inject realistic field-noise conditions (dead channels, powerline harmonic noise, "
                "spike bursts — report Sec 5.7)", value=True,
                help="Without this, the synthetic data is perfectly clean and the Trace Edit / "
                     "De-spike / Mute steps have nothing to actually fix. On by default so those "
                     "steps are demonstrably doing something.")
            if st.button("Generate synthetic survey", type="primary"):
                with st.spinner("Generating..."):
                    new_survey = synthetic_data.generate_synthetic_survey(
                        n_stations=int(n_stations), shot_interval_stations=int(shot_interval),
                        channels_per_side=int(channels), inject_field_noise=inject_noise,
                        dx=float(dx), dt=float(dt_ms) / 1000.0)
                _load_survey(new_survey, "synthetic")

    if survey is None:
        st.info("Load a SEG-Y file or generate synthetic demo data to continue.", icon="ℹ️")
        return

    if survey.is_synthetic:
        st.warning(f"⚠️ **Synthetic test data** — {survey.notes}", icon="🧪")

    st.markdown(f"**Loaded:** `{survey.source_name}`")

    with st.expander("② Geometry merging (optional)"):
        st.markdown(
            "Combine a second SEG-Y survey with the one already loaded — e.g. an adjacent 2D "
            "line or a repeat shoot — renumbering its CDP/shot IDs so they can't collide with "
            "the first survey's. **Simplified relative to a full industry geometry merge**: a "
            "real merge reconciles source/receiver coordinates from external SPS navigation "
            "files across independently-acquired lines; this just needs both surveys to already "
            "carry consistent geometry in their own headers (true of anything this app can load "
            "or generate), and combines them without ID collisions. Requires matching sample "
            "interval and sample count."
        )
        second = st.file_uploader("Second SEG-Y file to merge in", type=["sgy", "segy", "su"], key="merge_upload")
        if second is not None and st.button("Merge into current survey"):
            try:
                path = segy_io.materialize_upload(second)
                survey_b = segy_io.read_segy(path, source_name=second.name)
                merged = segy_io.merge_surveys(survey, survey_b)
                _load_survey(merged, f"merged: {survey.source_name} + {second.name}")
            except ValidationError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Couldn't merge this file: {e}")

    qc = segy_io.header_qc_summary(survey)
    if qc["n_traces"] > cfg.MAX_TRACES_FULL_PROCESS:
        st.warning(
            f"⚠️ This survey has {qc['n_traces']:,} traces, above the "
            f"{cfg.MAX_TRACES_FULL_PROCESS:,}-trace guideline for smooth interactive "
            f"processing in this app. Consider raising the minimum-fold threshold later, or "
            f"expect each processing step to take noticeably longer.", icon="🐢")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traces", f"{qc['n_traces']:,}")
    c2.metric("Samples/trace", qc["n_samples"])
    c3.metric("Sample interval", fmt_ms(survey.sample_interval))
    c4.metric("Record length", f"{qc['record_length_s']:.2f} s")
    c1.metric("Shots", qc["n_shots"])
    c2.metric("Receiver stations", qc["n_receiver_stations"])
    c3.metric("CDP ensembles (nominal)", qc["n_cdp"])
    c4.metric("Group interval", f"{qc['group_interval_m']:.1f} m")

    if qc["dead_traces"] > 0 or qc["non_finite_samples"] > 0:
        st.warning(f"⚠️ {qc['dead_traces']} dead trace(s), {qc['non_finite_samples']} non-finite sample(s) detected.")

    with st.expander("③ Header inspection"):
        tab_txt, tab_bin, tab_trace = st.tabs(["Textual (EBCDIC) header", "Binary header", "Trace headers"])
        with tab_txt:
            st.code("\n".join(survey.text_header_lines), language=None)
        with tab_bin:
            if survey.binary_header:
                st.dataframe(pd.DataFrame(list(survey.binary_header.items()), columns=["Field", "Value"]), height=300)
            else:
                st.caption("No binary header fields captured for this file.")
        with tab_trace:
            n_show = st.slider("Traces to preview", 10, min(2000, survey.num_traces), min(200, survey.num_traces))
            st.dataframe(segy_io.scan_trace_headers(survey, n_show), height=320)

    with st.expander("④ Raw data QC"):
        col1, col2 = st.columns(2)
        with col1:
            amp = survey.raw_data
            st.plotly_chart(spectra_plots.plot_spectrum(amp[:min(200, amp.shape[0])], survey.sample_interval,
                                                         title="Raw Amplitude Spectrum (first 200 traces)"), theme=None, key="dm_raw_spectrum")
        with col2:
            finite = amp[np.isfinite(amp)]
            st.markdown("**Amplitude statistics**")
            st.write(pd.DataFrame({
                "stat": ["min", "max", "mean", "RMS", "std"],
                "value": [finite.min(), finite.max(), finite.mean(), qc["amplitude_rms"], finite.std()],
            }).set_index("stat"))

        st.markdown("**Trace viewer**")
        max_tr = survey.num_traces
        start = st.slider("Start trace", 0, max(0, max_tr - 5), 0)
        n_traces_view = st.slider("Traces to show", 5, min(60, max_tr), min(20, max_tr))
        display_style = st.radio("Display style", ["Wiggle / variable-area", "Variable-density + wiggle"],
                                  horizontal=True, key="dm_trace_style")
        window = slice(start, min(max_tr, start + n_traces_view))
        if display_style == "Wiggle / variable-area":
            fig = seismic_plots.plot_wiggle_va(
                survey.raw_data[window], np.arange(window.start, window.stop), survey.twt,
                title=f"Raw traces {window.start}-{window.stop}")
        else:
            fig = seismic_plots.plot_variable_density_with_wiggle(
                survey.raw_data[window], np.arange(window.start, window.stop), survey.twt,
                title=f"Raw traces {window.start}-{window.stop} (VD + wiggle)")
        st.plotly_chart(fig, theme=None, key="dm_wiggle")

    with st.expander("⑤ Brute stack (optional early geometry QC)"):
        st.markdown(
            "A quick-look stack and CDP fold map, built with a single rough **constant-velocity "
            "NMO** correction (no full velocity analysis) — purely to sanity-check geometry (CDP "
            "assignment, fold taper at survey edges, gross statics problems) before investing time "
            "in real processing. This is what 'brute stack' means industry-wide: a fast, "
            "approximate-velocity look, not a stack of raw unaligned traces. This CDP grouping is "
            "**temporary and separate** from the official CDP sort/binning done later in the "
            "workflow, after pre-stack conditioning."
        )
        brute_velocity = st.slider(
            "Brute-stack NMO velocity (constant, m/s)", min_value=float(cfg.VELOCITY_MIN_DEFAULT),
            max_value=float(cfg.VELOCITY_MAX_DEFAULT), value=2500.0, step=50.0,
            help="A single rough velocity used only to flatten moveout for this QC stack — without "
                 "it, reflections at different offsets destructively interfere and the stack looks "
                 "like noise rather than a seismic section, even though the geometry is fine. Full "
                 "velocity analysis (per-CDP, time-varying) happens later in the real workflow.",
        )
        if st.button("Compute brute stack"):
            with st.spinner("Building a throwaway CDP grouping, applying constant-velocity NMO, and stacking it..."):
                brute_gathers = segy_io.build_cdp_gathers(survey, n_min_traces=max(1, st.session_state["n_min_traces"] // 2))
                folds = np.array([g["traces"].shape[0] for g in brute_gathers.values()])
                brute_cdp_list = list(brute_gathers.keys())
                dt = survey.sample_interval
                n_samples = survey.num_samples
                vrms_const = np.full(n_samples, brute_velocity)
                stacked_rows = []
                for g in brute_gathers.values():
                    corrected, live = nmo.nmo_correction(
                        g["traces"], g["offsets"], vrms_const, dt,
                        stretch_mute=cfg.NMO_STRETCH_MUTE_DEFAULT)
                    stacked_rows.append(fold_normalized_stack(corrected, live)[0])
                brute = np.vstack(stacked_rows)
            st.session_state["_brute_stack"] = (brute, brute_cdp_list, folds)
        if st.session_state.get("_brute_stack") is not None:
            brute, brute_cdp_list, folds = st.session_state["_brute_stack"]
            st.plotly_chart(seismic_plots.plot_fold_map(folds, brute_cdp_list), theme=None, key="dm_fold_map")
            st.plotly_chart(seismic_plots.plot_section(
                brute, np.array(brute_cdp_list), survey.twt,
                title=f"Brute Stack (constant-velocity NMO @ {brute_velocity:.0f} m/s, quick-look only)",
                equalize=True),
                theme=None, key="dm_brute_stack")
            st.caption(
                "Display-equalized (per-trace, time-domain) so the strong near-surface ground-roll "
                "energy — still fully present here since no preconditioning/mute has run yet — "
                "doesn't wash out the deeper reflectors in the color scale. A single constant "
                "velocity cannot flatten every reflector at once, so shallower/mid events will look "
                "more coherent than deep ones; that's expected for a quick-look QC stack."
            )

    st.divider()
    st.subheader("⑥ Build shot gathers")
    st.markdown(
        "This is the real starting point for **pre-stack conditioning** (band-pass → AGC → "
        "trace edit/kill → mute → balance → F-K filter → multiple suppression → deconvolution), "
        "all applied in the shot domain — the domain the data was actually recorded in. CDP "
        "sort/binning happens later, once conditioning is complete."
    )
    if st.button("Build shot gathers", type="primary"):
        shot_gathers = segy_io.build_shot_gathers(survey, n_min_traces=1)
        st.session_state["shot_gathers"] = shot_gathers
        ss.mark_stage_done("shot_gathers", params={"n_shots": len(shot_gathers)})
        st.rerun()

    if ss.is_stage_done("shot_gathers"):
        shot_gathers = st.session_state["shot_gathers"]
        ch_counts = np.array([g["traces"].shape[0] for g in shot_gathers.values()])
        total_in_shots = int(ch_counts.sum())
        st.success(f"✅ {len(shot_gathers)} shot gathers built "
                    f"({ch_counts.min()}-{ch_counts.max()} channels/shot). "
                    f"Continue to **Pre-Stack Conditioning**.")
        match = total_in_shots == survey.num_traces
        icon = "✅" if match else "⚠️"
        st.caption(
            f"{icon} Trace-count check: {total_in_shots:,} traces across all shot gathers "
            f"vs. {survey.num_traces:,} traces in the loaded survey — "
            + ("**no traces were duplicated or dropped**; building shot gathers only "
               "*reorganizes* the existing traces by their recorded shot/FieldRecord "
               "number, one trace in exactly one shot. "
               if match else
               "**these don't match — investigate before trusting downstream results.** "
               )
            + ("A per-shot channel count noticeably larger than a 'channels per side' "
               "setting is expected for a split/centre-spread survey (shot in the middle "
               "of its receiver patch, recording both flanks at once) — that is real "
               "acquisition geometry, confirmed against the summer report's spread-type "
               "definitions (Sec 5.5.1.1), not a doubling bug." if survey.is_synthetic else
               "A wide range of channels/shot across different shots is normal (edge "
               "shots record fewer live channels than mid-line shots) and reflects "
               "whatever live-receiver patch is recorded in the file's own headers for "
               "each shot, not anything this app adds.")
        )
