"""
views/interpretation_view.py — Well Tie, Horizon/Fault Picking, Surface Map
====================================================================
Implements the interpretation workflow in the order the summer report
lays it out (Ch.8): (1) Well-to-Seismic Tie -> (2) Seismic Horizon
Picking -> (3) Fault Interpretation -> (4) Surface (Time/Depth) Map
Generation. Step (5), Seismic Attribute Analysis, is its own page
(Attributes) since it operates on the finished interpretation.

Horizon/fault picking is click-*entered* rather than click-*on-the-
plot*: Streamlit cannot deliver reliable point-selection events for a
Heatmap-backed figure (see `utils/click_picking` module docstring), so
each pick is typed in via a small numeric form under a fully zoomable/
pannable reference plot, backed by an editable table for precise
review/correction. Every pick is exactly what the person entered or
edited, never inferred or auto-generated.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.interpretation import (FaultLine, Horizon, extract_statistical_wavelet,
                                     fault_to_dataframe, horizon_curvature, horizon_to_dataframe,
                                     interpolate_horizon, read_las_curves, sonic_to_velocity,
                                     synthetic_seismogram, well_tie_correlation)
from utils import session_state as ss
from utils.click_picking import numeric_point_form
from visualization import controls, interpretation_plots, seismic_plots, theme

_HORIZON_COLORS = ["#facc15", "#22d3ee", "#a78bfa", "#f472b6", "#34d399"]
_FAULT_COLORS = ["#ef4444", "#f97316", "#dc2626"]


def _best_available_section():
    for key, label in [("stack_final", "Final processed section"), ("stack_migrated", "Migrated section"),
                        ("stack_tvf", "Post-stack (pre-migration)"), ("stack_matrix", "Stack")]:
        if st.session_state.get(key) is not None:
            return st.session_state[key], label
    return None, None


def _nearest_cdp(cdp_list, x_clicked: float) -> int:
    arr = np.asarray(cdp_list)
    return int(arr[np.argmin(np.abs(arr - x_clicked))])


def _step_caption(current: int):
    steps = ["1. Well-to-Seismic Tie", "2. Seismic Horizon Picking", "3. Fault Interpretation",
              "4. Surface (Time/Depth) Map Generation"]
    parts = [f"**{s}**" if i == current else s for i, s in enumerate(steps)]
    st.caption(" → ".join(parts) + "  → 5. Seismic Attribute Analysis (see the **Attributes** page)")


def render():
    st.header("🧭 Interpretation")
    st.caption("Well-to-seismic tie, horizon and fault picking, and surface map generation — "
                "in the order the report lays them out (Ch.8).")

    if not ss.require_stage("cdp_sort"):
        return
    section, section_label = _best_available_section()
    if section is None:
        st.info("Run at least the Stacking step to get a section to interpret.", icon="ℹ️")
        return

    survey = st.session_state["survey"]
    cdp_list = st.session_state["cdp_list"]
    twt = survey.twt
    horizons: dict = st.session_state["horizons"]
    faults: dict = st.session_state["faults"]

    tab_tie, tab_horizon, tab_fault, tab_map = st.tabs(
        ["① Well Tie", "② Horizon Picking", "③ Fault Interpretation", "④ Surface Map"])

    # =====================================================================
    # 1) WELL-TO-SEISMIC TIE
    # =====================================================================
    with tab_tie:
        _step_caption(0)
        st.markdown(
            "Upload a well log as **CSV** (columns `depth_m`, `vp_m_s`, `rho_kg_m3`) or a **LAS** "
            "file (curve names vary between wells/vendors, so LAS curves are mapped explicitly "
            "below rather than assumed). A checkshot CSV (`depth_m`, `twt_s`) is optional — without "
            "one, depth-to-time comes from integrating the sonic/velocity log directly, the standard "
            "fallback (report Sec 8.2.1.1, Step 2)."
        )
        log_file = st.file_uploader("Well log", type=["csv", "las"], key="welllog_file")
        checkshot_file = st.file_uploader("Checkshot CSV (optional)", type=["csv"], key="checkshot_csv")

        depth_m = vp_m_s = rho_kg_m3 = None
        if log_file is not None:
            if log_file.name.lower().endswith(".las"):
                try:
                    raw_bytes = log_file.getvalue()
                    curves = read_las_curves(io.StringIO(raw_bytes.decode("utf-8", errors="replace")))
                except ImportError as e:
                    st.error(str(e))
                    curves = None
                except Exception as e:
                    st.error(f"Couldn't parse this LAS file: {e}")
                    curves = None
                if curves:
                    mnemonics = [k for k in curves if k != "__DEPTH__"]
                    st.markdown("**Map LAS curves** (this file's mnemonics, not assumed fixed names):")
                    c1, c2, c3 = st.columns(3)
                    depth_choice = c1.selectbox("Depth", ["(LAS index)"] + mnemonics, key="las_depth")
                    vel_curve = c2.selectbox("Velocity or sonic curve", mnemonics, key="las_vel")
                    rho_curve = c3.selectbox("Density curve", mnemonics, key="las_rho")
                    c1, c2, c3 = st.columns(3)
                    depth_unit = c1.selectbox("Depth unit", ["m", "ft"], key="las_depth_unit")
                    vel_kind = c2.selectbox("Velocity curve is...",
                                             ["Direct velocity (m/s)", "Direct velocity (ft/s)",
                                              "Sonic slowness (µs/ft)", "Sonic slowness (µs/m)"],
                                             key="las_vel_kind")
                    rho_unit = c3.selectbox("Density unit", ["kg/m3", "g/cc"], key="las_rho_unit")

                    depth_raw = curves["__DEPTH__"] if depth_choice == "(LAS index)" else curves[depth_choice]
                    depth_m = depth_raw * 0.3048 if depth_unit == "ft" else depth_raw
                    vel_raw = curves[vel_curve]
                    if vel_kind == "Direct velocity (m/s)":
                        vp_m_s = vel_raw
                    elif vel_kind == "Direct velocity (ft/s)":
                        vp_m_s = vel_raw * 0.3048
                    elif vel_kind == "Sonic slowness (µs/ft)":
                        vp_m_s = sonic_to_velocity(vel_raw, "us_per_ft")
                    else:
                        vp_m_s = sonic_to_velocity(vel_raw, "us_per_m")
                    rho_raw = curves[rho_curve]
                    rho_kg_m3 = rho_raw * 1000.0 if rho_unit == "g/cc" else rho_raw
                    valid = np.isfinite(depth_m) & np.isfinite(vp_m_s) & np.isfinite(rho_kg_m3)
                    depth_m, vp_m_s, rho_kg_m3 = depth_m[valid], vp_m_s[valid], rho_kg_m3[valid]
                    order = np.argsort(depth_m)
                    depth_m, vp_m_s, rho_kg_m3 = depth_m[order], vp_m_s[order], rho_kg_m3[order]
                    st.caption(f"{len(depth_m)} valid depth samples after unit conversion and cleaning.")
            else:
                try:
                    log_df = pd.read_csv(log_file)
                    required = {"depth_m", "vp_m_s", "rho_kg_m3"}
                    if not required.issubset(log_df.columns):
                        st.error(f"Well log CSV must have columns {sorted(required)}.")
                    else:
                        depth_m = log_df["depth_m"].values
                        vp_m_s = log_df["vp_m_s"].values
                        rho_kg_m3 = log_df["rho_kg_m3"].values
                except Exception as e:
                    st.error(f"Couldn't read this CSV: {e}")

        st.divider()
        st.markdown("**Wavelet** — used to convolve the depth-domain reflectivity into a synthetic trace.")
        wavelet_source = st.radio("Wavelet source", ["Ricker (choose frequency)",
                                                        "Statistical (extracted from the seismic data)"],
                                    key="tie_wavelet_source", horizontal=True)
        tie_cdp = st.selectbox("Seismic trace to compare (nearest CDP)", cdp_list,
                                index=len(cdp_list) // 2, key="tie_cdp")

        wavelet_freq = None
        extracted_wavelet = None
        if wavelet_source.startswith("Ricker"):
            wavelet_freq = st.slider("Wavelet peak frequency (Hz)", 10.0, 80.0, 30.0, key="tie_freq")
        else:
            st.markdown(
                "Extracts a zero-phase wavelet directly from a window of the seismic section around "
                "the tie CDP, using the same white-reflectivity assumption that underlies statistical "
                "deconvolution (report Sec 6.8.2/8.2.1.1): the ensemble-averaged power spectrum of "
                "nearby traces approximates the wavelet's own power spectrum."
            )
            c1, c2, c3 = st.columns(3)
            n_cdps = c1.slider("CDPs to average around tie CDP", 1, 41, 15, step=2, key="tie_wav_ncdp")
            t_win = c2.slider("Extraction time window (s)", 0.0, float(twt[-1]),
                               (0.0, float(twt[-1])), key="tie_wav_twin")
            wav_len = c3.slider("Wavelet length (ms)", 40, 400, 150, step=10, key="tie_wav_len")
            idx = cdp_list.index(tie_cdp)
            lo, hi = max(0, idx - n_cdps // 2), min(len(cdp_list), idx + n_cdps // 2 + 1)
            trace_block = np.atleast_2d(section[lo:hi, :])
            extracted_wavelet = extract_statistical_wavelet(
                trace_block, survey.sample_interval, length_s=wav_len / 1000.0, t_window=t_win)
            wav_t = (np.arange(len(extracted_wavelet)) - len(extracted_wavelet) // 2) * survey.sample_interval
            spec = np.abs(np.fft.rfft(extracted_wavelet))
            freqs = np.fft.rfftfreq(len(extracted_wavelet), survey.sample_interval)
            c1, c2 = st.columns(2)
            with c1:
                figw = go.Figure(go.Scatter(x=wav_t, y=extracted_wavelet, mode="lines",
                                              line=dict(color="#0D2B5E", width=2)))
                figw.update_layout(**theme.base_layout("Extracted wavelet (time domain)", 300))
                figw.update_xaxes(**theme.x_axis("Time (s)"))
                figw.update_yaxes(**theme.x_axis("Amplitude"))
                st.plotly_chart(figw, theme=None, key="tie_wavelet_time")
            with c2:
                figs = go.Figure(go.Scatter(x=freqs, y=spec, mode="lines", line=dict(color="#c62828", width=2)))
                figs.update_layout(**theme.base_layout("Amplitude spectrum", 300))
                figs.update_xaxes(**theme.x_axis("Frequency (Hz)"))
                figs.update_yaxes(**theme.x_axis("Amplitude"))
                st.plotly_chart(figs, theme=None, key="tie_wavelet_spectrum")

        shift_ms = st.slider("Manual bulk shift (ms)", -200, 200, 0, key="tie_shift")

        if depth_m is not None and st.button("Compute synthetic & tie", type="primary"):
            try:
                checkshot_df = pd.read_csv(checkshot_file) if checkshot_file is not None else None
                result = synthetic_seismogram(
                    depth_m, vp_m_s, rho_kg_m3, survey.sample_interval,
                    wavelet_freq_hz=wavelet_freq or 30.0, checkshot=checkshot_df,
                    t_max_s=survey.record_length, wavelet=extracted_wavelet)
                st.session_state["synthetic_tie"] = result
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't build the synthetic: {e}")

        result = st.session_state.get("synthetic_tie")
        if result is not None:
            seismic_trace = section[cdp_list.index(tie_cdp), :]
            shift_s = shift_ms / 1000.0
            synth_shifted = np.interp(result["time_axis"] - shift_s, result["time_axis"],
                                       result["synthetic_trace"], left=0, right=0)
            syn_n = synth_shifted / (np.max(np.abs(synth_shifted)) + 1e-9)
            sei_n = seismic_trace / (np.max(np.abs(seismic_trace)) + 1e-9)
            corr = well_tie_correlation(syn_n, sei_n)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=syn_n, y=result["time_axis"], name="Synthetic", line=dict(color="#c62828")))
            fig.add_trace(go.Scatter(x=sei_n, y=twt[:len(sei_n)], name=f"Seismic (CDP {tie_cdp})",
                                      line=dict(color="#0D2B5E")))
            fig.update_layout(**theme.base_layout("Synthetic-to-Seismic Tie", 600))
            fig.update_xaxes(**theme.x_axis("Normalized amplitude"))
            fig.update_yaxes(**theme.time_axis())
            st.plotly_chart(fig, theme=None, key="tie_chart")
            badge = "✅ good tie" if corr >= 0.7 else ("⚠️ marginal" if corr >= 0.4 else "❌ poor tie")
            st.metric("Correlation coefficient (report Sec 8.2.1.1, Step 4)", f"{corr:.2f}", badge)
            st.caption("Adjust the manual bulk shift above until the strongest synthetic events line "
                        "up with the strongest seismic events and the correlation coefficient is "
                        "maximized (≳0.7-0.8 is generally considered a good tie).")

    # =====================================================================
    # 2) SEISMIC HORIZON PICKING
    # =====================================================================
    with tab_horizon:
        _step_caption(1)
        st.caption(f"Interpreting on: **{section_label}**")
        c1, c2 = st.columns([3, 1])
        new_name = c1.text_input("New horizon name", placeholder="e.g. Top Reservoir", key="new_horizon_name")
        if c2.button("Add horizon") and new_name:
            if new_name not in horizons:
                horizons[new_name] = Horizon(new_name, _HORIZON_COLORS[len(horizons) % len(_HORIZON_COLORS)])

        if not horizons:
            st.info("Name and add a horizon above to start picking.", icon="ℹ️")
        else:
            name = st.selectbox("Editing horizon", list(horizons.keys()))
            h = horizons[name]
            disp = controls.section_display_controls("horizon_display", default_colorscale="Seismic")

            fig = interpretation_plots.plot_section_with_picks(
                section, cdp_list, twt, horizons=horizons, faults={}, title=section_label,
                colorscale=disp.colorscale, equalize=disp.equalize)
            st.plotly_chart(fig, theme=None, key=f"h_click_{name}")
            st.caption("🔍 Zoom in on a reflector above to read off its (CDP, time) position, then add "
                        "the pick below.")
            new_pt = numeric_point_form(
                f"h_pt_{name}", "CDP", "Time (s)",
                x_value=float(np.median(cdp_list)), y_value=float(twt[len(twt) // 3]),
                x_min=float(min(cdp_list)), x_max=float(max(cdp_list)),
                y_min=0.0, y_max=float(twt[-1]), x_step=1.0, y_step=0.005)
            if new_pt is not None:
                cdp_clicked, t_clicked = new_pt
                h.picks[_nearest_cdp(cdp_list, cdp_clicked)] = round(float(t_clicked), 4)
                st.rerun()

            df = horizon_to_dataframe(h, cdp_list, survey.group_interval)
            picked_df = df[df["picked"]][["cdp", "time_s"]].reset_index(drop=True)
            rev = st.session_state.get(f"h_rev_{name}", 0)
            edited = st.data_editor(picked_df, num_rows="dynamic", key=f"h_editor_{name}_{rev}")
            c1, c2, c3 = st.columns(3)
            if c1.button("Save table edits", key=f"h_save_{name}"):
                h.picks = {int(r.cdp): float(r.time_s) for r in edited.itertuples()
                           if pd.notna(r.cdp) and pd.notna(r.time_s)}
                st.session_state[f"h_rev_{name}"] = rev + 1
                st.rerun()
            if c2.button("Clear all picks", key=f"h_clear_{name}"):
                h.picks = {}
                st.session_state[f"h_rev_{name}"] = rev + 1
                st.rerun()
            if c3.button(f"Delete horizon '{name}'", key=f"h_delete_{name}"):
                del horizons[name]
                st.rerun()

            if len(h.picks) >= 5:
                with st.expander("Horizon curvature"):
                    st.markdown(
                        "Curvature of the picked horizon line — the 2-D analogue of the "
                        "surface-curvature attribute used in 3D interpretation (report Sec 8.6.4: "
                        "\"computed from the second derivatives of the surface's time or depth "
                        "values\"; here from the single lateral direction a 2D line has)."
                    )
                    horizon_time = interpolate_horizon(h.picks, cdp_list)
                    kappa = horizon_curvature(horizon_time, cdp_list, survey.group_interval)
                    st.plotly_chart(interpretation_plots.plot_attribute_along_horizon(
                        kappa, cdp_list, title=f"Curvature — '{name}'", colorbar_title="Curvature (1/m)"),
                        theme=None, key=f"h_curvature_{name}")

    # =====================================================================
    # 3) FAULT INTERPRETATION
    # =====================================================================
    with tab_fault:
        _step_caption(2)
        c1, c2 = st.columns([3, 1])
        new_fname = c1.text_input("New fault name", placeholder="e.g. Fault A", key="new_fault_name")
        if c2.button("Add fault") and new_fname:
            if new_fname not in faults:
                faults[new_fname] = FaultLine(new_fname, _FAULT_COLORS[len(faults) % len(_FAULT_COLORS)])

        if not faults:
            st.info("Name and add a fault above to start picking.", icon="ℹ️")
        else:
            fname = st.selectbox("Editing fault", list(faults.keys()))
            fl = faults[fname]
            disp = controls.section_display_controls("fault_display", default_colorscale="Seismic")

            fig = interpretation_plots.plot_section_with_picks(
                section, cdp_list, twt, horizons={}, faults=faults, title=section_label,
                colorscale=disp.colorscale, equalize=disp.equalize)
            st.plotly_chart(fig, theme=None, key=f"f_click_{fname}")
            st.caption("🔍 Zoom in on the reflector offset/discontinuity above, then add each point "
                        "along the fault trace below, in order from shallow to deep.")
            new_pt = numeric_point_form(
                f"f_pt_{fname}", "CDP", "Time (s)",
                x_value=float(np.median(cdp_list)), y_value=float(twt[len(twt) // 3]),
                x_min=float(min(cdp_list)), x_max=float(max(cdp_list)),
                y_min=0.0, y_max=float(twt[-1]), x_step=1.0, y_step=0.005)
            if new_pt is not None:
                cdp_clicked, t_clicked = new_pt
                fl.points.append((_nearest_cdp(cdp_list, cdp_clicked), round(float(t_clicked), 4)))
                st.rerun()

            df = fault_to_dataframe(fl)
            rev = st.session_state.get(f"f_rev_{fname}", 0)
            edited = st.data_editor(df, num_rows="dynamic", key=f"f_editor_{fname}_{rev}")
            c1, c2, c3 = st.columns(3)
            if c1.button("Save table edits", key=f"f_save_{fname}"):
                fl.points = [(int(r.cdp), float(r.time_s)) for r in edited.itertuples()
                             if pd.notna(r.cdp) and pd.notna(r.time_s)]
                st.session_state[f"f_rev_{fname}"] = rev + 1
                st.rerun()
            if c2.button("Clear all points", key=f"f_clear_{fname}"):
                fl.points = []
                st.session_state[f"f_rev_{fname}"] = rev + 1
                st.rerun()
            if c3.button(f"Delete fault '{fname}'", key=f"f_delete_{fname}"):
                del faults[fname]
                st.rerun()

    # =====================================================================
    # 4) SURFACE (TIME/DEPTH) MAP GENERATION
    # =====================================================================
    with tab_map:
        _step_caption(3)
        st.markdown(
            "Report Sec 8.5: grids a picked horizon into a continuous surface and, given a velocity "
            "model, converts it from two-way time to true depth. This app works with a single 2D "
            "line rather than a 3D volume, so there is no areal grid to contour — the equivalent "
            "quantity here is the horizon's time (or depth) value as a continuous **profile** along "
            "the line, which is what's shown below, honestly labeled as a 2D line rather than a map."
        )
        if not horizons or not any(h.picks for h in horizons.values()):
            st.info("Pick at least one horizon (previous tab) before generating a structure profile.", icon="ℹ️")
        else:
            picked_horizons = {n: h for n, h in horizons.items() if h.picks}
            name = st.selectbox("Horizon", list(picked_horizons.keys()), key="map_horizon")
            h = picked_horizons[name]
            horizon_time = interpolate_horizon(h.picks, cdp_list)

            st.plotly_chart(interpretation_plots.plot_structure_profile(
                horizon_time, cdp_list, y_title="TWT (s)",
                title=f"Time Structure Profile — '{name}'", color=h.color),
                theme=None, key="map_time_profile")

            st.divider()
            st.markdown("**Time-to-depth conversion** (report Eq. 8.5: `Z = Vavg(t) * t / 2`).")
            vfield = st.session_state.get("Vfield_2")
            vel_source = st.radio("Average velocity source",
                                    ["Constant (enter a value)", "From RMS velocity field (Pass 2)"]
                                    if vfield is not None else ["Constant (enter a value)"],
                                    key="map_vel_source", horizontal=True)
            if vel_source.startswith("Constant"):
                v_avg = st.slider("Average velocity (m/s)", 1500.0, 5500.0, 2800.0, step=50.0, key="map_vavg")
                depth = v_avg * horizon_time / 2.0
            else:
                dt = survey.sample_interval
                t_idx = np.clip(np.round(horizon_time / dt).astype(int), 0, vfield.shape[0] - 1)
                cdp_idx = np.arange(len(cdp_list))
                v_at_horizon = vfield[t_idx, cdp_idx]
                depth = v_at_horizon * horizon_time / 2.0
            st.plotly_chart(interpretation_plots.plot_structure_profile(
                depth, cdp_list, y_title="Depth (m, TVDSS-equivalent)",
                title=f"Depth Structure Profile — '{name}'", color="#8B5E00"),
                theme=None, key="map_depth_profile")

            out_df = pd.DataFrame({"cdp": cdp_list, "twt_s": horizon_time, "depth_m": depth})
            st.download_button("Download structure profile (CSV)", out_df.to_csv(index=False),
                                file_name=f"structure_profile_{name}.csv", mime="text/csv")
