"""
views/interpretation_view.py — Horizon/Fault Picking & Well Tie
====================================================================
Not present in the source processing notebooks (which end at
migration), but supported by the interpretation reference PDF. Picking
is primarily **click-on-the-section** (via
`st.plotly_chart(..., on_select="rerun")`), backed by an editable table
for precise review/correction — click for speed, table for precision.
Every pick is exactly what the person clicked or typed, never inferred
or auto-generated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.interpretation import (FaultLine, Horizon, fault_to_dataframe,
                                     horizon_curvature, horizon_to_dataframe,
                                     interpolate_horizon, synthetic_seismogram)
from utils import session_state as ss
from utils.click_picking import click_help_caption, consume_new_click
from visualization import interpretation_plots, seismic_plots

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


def render():
    st.header("🧭 Interpretation")
    st.caption("Horizon and fault picking (click on the section to pick), and an approximate "
                "well-to-seismic tie.")

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

    tab_horizon, tab_fault, tab_tie = st.tabs(["Horizons", "Faults", "Well Tie (approximate)"])

    with tab_horizon:
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
            st.caption(click_help_caption())

            fig = interpretation_plots.plot_section_with_picks(
                section, cdp_list, twt, horizons=horizons, faults={}, title=section_label)
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                                     key=f"h_click_{name}", theme=None)
            new_pt = consume_new_click(event, f"h_dedup_{name}")
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
                        "surface-curvature attribute used in 3D interpretation (there, computed "
                        "from a gridded surface's second derivatives in two lateral directions; "
                        "here, from the single lateral direction a 2D line has). Useful for "
                        "spotting subtle folds and flexures directly from the pick."
                    )
                    horizon_time = interpolate_horizon(h.picks, cdp_list)
                    kappa = horizon_curvature(horizon_time, cdp_list, survey.group_interval)
                    st.plotly_chart(interpretation_plots.plot_attribute_along_horizon(
                        kappa, cdp_list, title=f"Curvature — '{name}'", colorbar_title="Curvature (1/m)"),
                        theme=None, key=f"h_curvature_{name}")

    with tab_fault:
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
            st.caption(click_help_caption() + " (points are kept in the order you add them)")

            fig = interpretation_plots.plot_section_with_picks(
                section, cdp_list, twt, horizons={}, faults=faults, title=section_label)
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                                     key=f"f_click_{fname}", theme=None)
            new_pt = consume_new_click(event, f"f_dedup_{fname}")
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

    with tab_tie:
        st.markdown(
            "**Approximate well-to-seismic tie.** Upload a simple well-log CSV with columns "
            "`depth_m`, `vp_m_s`, `rho_kg_m3` (density). A checkshot CSV (`depth_m`, `twt_s`) is "
            "optional — without one, the depth-time conversion comes directly from integrating the "
            "sonic log, the standard fallback. This computes a genuine reflectivity-and-convolution "
            "synthetic; it does **not** extract a statistical wavelet from the seismic itself "
            "(a Ricker wavelet at a frequency you choose is used instead) — see README for what a "
            "fuller well-tie workflow would add."
        )
        log_file = st.file_uploader("Well log CSV", type=["csv"], key="welllog_csv")
        checkshot_file = st.file_uploader("Checkshot CSV (optional)", type=["csv"], key="checkshot_csv")
        wavelet_freq = st.slider("Wavelet peak frequency (Hz)", 10.0, 80.0, 30.0, key="tie_freq")
        tie_cdp = st.selectbox("Seismic trace to compare (nearest CDP)", cdp_list,
                                index=len(cdp_list) // 2, key="tie_cdp")
        shift_ms = st.slider("Manual bulk shift (ms)", -200, 200, 0, key="tie_shift")

        if log_file is not None and st.button("Compute synthetic & tie", type="primary"):
            try:
                log_df = pd.read_csv(log_file)
                required = {"depth_m", "vp_m_s", "rho_kg_m3"}
                if not required.issubset(log_df.columns):
                    st.error(f"Well log CSV must have columns {sorted(required)}.")
                else:
                    checkshot_df = pd.read_csv(checkshot_file) if checkshot_file is not None else None
                    result = synthetic_seismogram(log_df["depth_m"].values, log_df["vp_m_s"].values,
                                                   log_df["rho_kg_m3"].values, survey.sample_interval,
                                                   wavelet_freq_hz=wavelet_freq, checkshot=checkshot_df,
                                                   t_max_s=survey.record_length)
                    st.session_state["synthetic_tie"] = result
                    st.rerun()
            except Exception as e:
                st.error(f"Couldn't build the synthetic: {e}")

        result = st.session_state.get("synthetic_tie")
        if result is not None:
            import plotly.graph_objects as go
            from visualization import theme
            g = st.session_state["gathers"][tie_cdp]
            seismic_trace = g.get("stack_dmo", g["traces"].mean(axis=0))
            shift_s = shift_ms / 1000.0
            synth_shifted = np.interp(result["time_axis"] - shift_s, result["time_axis"],
                                       result["synthetic_trace"], left=0, right=0)
            syn_n = synth_shifted / (np.max(np.abs(synth_shifted)) + 1e-9)
            sei_n = seismic_trace / (np.max(np.abs(seismic_trace)) + 1e-9)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=syn_n, y=result["time_axis"], name="Synthetic", line=dict(color="#c62828")))
            fig.add_trace(go.Scatter(x=sei_n, y=twt[:len(sei_n)], name=f"Seismic (CDP {tie_cdp})",
                                      line=dict(color="#0D2B5E")))
            fig.update_layout(**theme.base_layout("Synthetic-to-Seismic Tie", 600))
            fig.update_xaxes(**theme.x_axis("Normalized amplitude"))
            fig.update_yaxes(**theme.time_axis())
            st.plotly_chart(fig, theme=None, key="tie_chart")
            st.caption("Adjust the manual bulk shift above until the strongest synthetic events line "
                        "up with the strongest seismic events, then read off the shift as your "
                        "check on the depth-time relationship.")
