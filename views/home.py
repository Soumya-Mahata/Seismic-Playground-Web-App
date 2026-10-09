"""views/home.py — Dashboard."""

from __future__ import annotations

import streamlit as st

import config as cfg
from modules.pipeline import PIPELINE_STAGES
from utils import workflow_ui as wf


def render():
    st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
    st.caption(cfg.APP_TAGLINE)
    st.markdown(
        "**From raw seismic data → processing → QC → interpretation → insights.** "
        "A guided, interactive 2D land seismic-processing and interpretation "
        "workflow, built as a proper application with parameter controls and QC "
        "at every step."
    )

    survey = st.session_state["survey"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📂 Dataset")
        if survey is None:
            st.warning("No data loaded yet.")
        else:
            st.success(f"`{survey.source_name}`")
            st.caption(f"{survey.num_traces:,} traces · {survey.num_samples} samples · "
                        f"{survey.sample_interval * 1000:.1f} ms")
            if survey.is_synthetic:
                st.caption("🧪 Synthetic test data")
    with c2:
        st.markdown("### ⚙️ Processing")
        done = len(st.session_state["completed_stages"])
        total = len(PIPELINE_STAGES)
        st.progress(done / total, text=f"{done} of {total} steps run")
        st.caption("Many steps are optional, so this is activity, not a score.")
    with c3:
        st.markdown("### 🧭 Interpretation")
        st.caption(f"{len(st.session_state['horizons'])} horizon(s), "
                    f"{len(st.session_state['faults'])} fault(s), "
                    f"{len(st.session_state['attributes'])} attribute(s) computed")

    st.divider()
    st.markdown("### Workflow status")
    st.caption("The numbered order is the *recommended* order. Only a step's genuine inputs gate "
                "it; optional steps can be skipped or reordered.")
    wf.home_status()
    with st.expander("Processing history (in the order steps were run)"):
        wf.history_table()

    st.divider()
    st.markdown("### Quick start")
    st.markdown(
        "1. **Data Management** — upload a SEG-Y file, or generate synthetic demo data. "
        "Optionally merge in a second survey; optionally check geometry with a quick brute "
        "stack. Then build shot gathers.\n"
        "2. **Pre-Stack Conditioning** — band-pass, AGC, trace edit/kill/de-spike, mute, "
        "balance, F-K filter (auto or click-drawn polygon), multiple suppression, "
        "deconvolution — all in the shot domain, and **all optional**: run any subset, in the "
        "order that suits the data. A step that is normally done first but wasn't gets a ⚠ "
        "and a *Continue?* prompt, never a block.\n"
        "3. **CDP Sort & Binning** — re-group the (conditioned or raw) shot data by CDP, with a "
        "fold map and a semblance preview.\n"
        "4. **Velocity, NMO & DMO** — pick stacking velocities by auto-pick, **click directly "
        "on the semblance panel**, or Constant-Velocity-Stack QC; apply NMO, solve residual "
        "statics, DMO. NMO needs a velocity field; the refinement steps are optional.\n"
        "5. **Stacking → Post-Stack → Migration** — stacking needs NMO-corrected gathers; "
        "post-stack steps are independent; migration needs a section + velocity (post-stack) or "
        "pre-stack gathers + velocity (PSTM).\n"
        "6. **Interpretation & Attributes** — click to pick horizons/faults; compute "
        "industry-standard attributes (RMS, variance, sweetness, coherence, curvature, ...).\n"
        "7. **Export** — download processed SEG-Y, picks, attributes, figures, and a report."
    )

    with st.expander("Supported data formats"):
        st.markdown(
            "- **Input:** SEG-Y (`.sgy`, `.segy`), SU (`.su`, read via the SEG-Y path)\n"
            "- **Well tie:** simple CSV well logs (`depth_m`, `vp_m_s`, `rho_kg_m3`) and "
            "optional checkshot CSV (`depth_m`, `twt_s`)\n"
            "- **Output:** SEG-Y, CSV, NumPy (`.npy`), interactive HTML figures, text/JSON reports"
        )

    with st.expander("ℹ️ About this app"):
        st.markdown(
            "A full 2D land seismic-processing and interpretation workflow — pre-stack "
            "conditioning (band-pass, AGC, trace edit, mute, balance, F-K, multiple "
            "suppression, deconvolution) applied in the **shot domain**, CDP sort/binning, "
            "two-pass velocity analysis/NMO/DMO, stacking, post-stack conditioning, and "
            "migration — matching a standard industry processing flow, plus an "
            "interpretation layer (well-to-seismic tie, horizon/fault picking, surface "
            "mapping, and industry-standard attribute analysis).\n\n"
            "See `README.md` for the full list of documented design decisions, including "
            "a documented interpolation-convention fix in the NMO correction and a "
            "velocity-picker design that avoids systematic tracking failures at depth and "
            "at survey edges."
        )
