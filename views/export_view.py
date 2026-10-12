"""views/export_view.py — Export processed data, picks, attributes, figures, and a report."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import streamlit as st

from modules import segy_io
from modules.dependencies import stale_info
from modules.interpretation import fault_to_dataframe, horizon_to_dataframe
from utils import io_export, session_state as ss
from utils.formatting import human_bytes
from visualization import controls, seismic_plots


_TAGS = {"Final processed section": "final", "Migrated section": "migrated",
         "Stack": "stack"}


def _section_tag(label: str) -> str:
    if label in _TAGS:
        return _TAGS[label]
    return "poststack"          # any post-stack chain head (RNA / decon / TVF)


def render():
    st.header("💾 Export")
    survey = st.session_state["survey"]
    if survey is None:
        st.info("Load data first.", icon="ℹ️")
        return
    twt = survey.twt

    section_state, section_label = ss.best_available_state()
    section = ss.section(section_state)
    cdp_list = (ss.cdp_list(section_state) if section_state is not None
                else st.session_state["cdp_list"] or [])
    section_tag = _section_tag(section_label) if section_label else None

    tab_section, tab_picks, tab_attr, tab_report = st.tabs(
        ["Processed Section", "Horizons & Faults", "Attributes", "Processing Report"])

    with tab_section:
        if section is None:
            st.info("Run Stacking (or a pre-stack migration) to have a section to export.", icon="ℹ️")
        else:
            st.caption(f"Exporting the most advanced available section: **{section_label}** "
                        f"(state {section_state.id}).")
            stale = stale_info(ss.workflow()).get(section_state.id)
            if stale:
                st.warning(f"⟳ This section is out of date: {stale[0].detail} It is still exported as it "
                            f"is — re-run the steps named above (or use *Data states* on the processing "
                            f"pages to choose another state) if you want the latest inputs.")
            base_name = f"{survey.source_name.rsplit('.', 1)[0]}_{section_tag}"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("⬇ SEG-Y", io_export.segy_bytes_from_section(
                    section, survey.sample_interval, cdp_list, survey.group_interval),
                    file_name=f"{base_name}.sgy", mime="application/octet-stream")
            with col2:
                st.download_button("⬇ CSV", io_export.section_to_csv_bytes(section, cdp_list, twt),
                                    file_name=f"{base_name}.csv", mime="text/csv")
            with col3:
                st.download_button("⬇ NumPy (.npy)", io_export.section_to_npy_bytes(section),
                                    file_name=f"{base_name}.npy", mime="application/octet-stream")

            st.markdown("**Display controls** (affects the preview and the exported HTML figure only — not the SEG-Y/CSV/NumPy data)")
            disp = controls.section_display_controls("export_display")
            fig = seismic_plots.plot_section(section, np.array(cdp_list), twt, title=f"Exported section ({section_tag})",
                                              colorscale=disp.colorscale, pclip=disp.pclip, gain=disp.gain)
            st.plotly_chart(fig, theme=None, key="export_view_chart_1001")
            st.download_button("⬇ Figure (interactive HTML)", io_export.figure_html_bytes(fig),
                                file_name=f"{base_name}.html", mime="text/html")

    with tab_picks:
        horizons, faults = st.session_state["horizons"], st.session_state["faults"]
        if not horizons and not faults:
            st.info("No horizons or faults picked yet — see the Interpretation page.", icon="ℹ️")
        for name, h in horizons.items():
            df = horizon_to_dataframe(h, cdp_list, survey.group_interval)
            st.download_button(f"⬇ Horizon '{name}' (CSV)", io_export.horizon_csv_bytes(df),
                                file_name=f"horizon_{name}.csv", mime="text/csv", key=f"dl_h_{name}")
        for name, fl in faults.items():
            df = fault_to_dataframe(fl)
            st.download_button(f"⬇ Fault '{name}' (CSV)", io_export.horizon_csv_bytes(df),
                                file_name=f"fault_{name}.csv", mime="text/csv", key=f"dl_f_{name}")

    with tab_attr:
        attrs = st.session_state["attributes"]
        if not attrs:
            st.info("No attributes computed yet — see the Attributes page.", icon="ℹ️")
        for name, values in attrs.items():
            st.download_button(f"⬇ {name} (CSV)", io_export.section_to_csv_bytes(values, cdp_list, twt),
                                file_name=f"attribute_{name.replace(' ', '_')}.csv", mime="text/csv",
                                key=f"dl_attr_{name}")

    with tab_report:
        qc = segy_io.header_qc_summary(survey)
        survey_info = {
            "Source file": survey.source_name,
            "Synthetic test data": survey.is_synthetic,
            "Traces": qc["n_traces"], "Samples/trace": qc["n_samples"],
            "Sample interval (ms)": round(qc["sample_interval_ms"], 3),
            "Record length (s)": qc["record_length_s"],
            "CDP ensembles processed": len(cdp_list) if cdp_list else 0,
        }
        graph = ss.graph()
        only_this = section_state is not None and st.checkbox(
            f"Only what produced the exported section ({section_state.id}): its operations and states", value=False)
        report = io_export.processing_report_text(
            survey_info, graph, state_id=section_state.id if only_this else None)
        st.text_area("Report preview", report, height=400)
        base_name = survey.source_name.rsplit(".", 1)[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.download_button("⬇ Report (.txt)", report.encode("utf-8"),
                              file_name=f"{base_name}_processing_report.txt", mime="text/plain")
        col2.download_button("⬇ Data states (.json)", io_export.params_json_bytes(graph),
                              file_name=f"{base_name}_states.json", mime="application/json")
        col3.download_button("⬇ History (.csv)", io_export.action_log_csv_bytes(graph.actions),
                              file_name=f"{base_name}_history.csv", mime="text/csv",
                              help="Every action in the order performed — chronological, regardless of "
                                   "the display order chosen in the app.")
        if section_state is not None:
            col4.download_button(f"⬇ Recipe for {section_state.id} (.json)",
                                  io_export.recipe_json_bytes(graph, section_state.id),
                                  file_name=f"{base_name}_{section_state.id}_recipe.json",
                                  mime="application/json",
                                  help="The exact operations and parameters that produce the exported "
                                       "section from the loaded survey — replayable and fingerprint-checked.")
