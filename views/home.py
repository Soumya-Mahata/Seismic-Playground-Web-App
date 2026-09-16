"""views/home.py — Dashboard."""

from __future__ import annotations

import streamlit as st

import config as cfg
from modules.pipeline import PIPELINE_STAGES
from utils import session_state as ss


def render():
    st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
    st.caption(cfg.APP_TAGLINE)
    st.markdown(
        "**From raw seismic data → processing → QC → interpretation → insights.** "
        "A guided, interactive re-implementation of a 2D land seismic-processing "
        "workflow — the same processing sequence you'd otherwise run cell-by-cell "
        "in a notebook, as a proper application with parameter controls and QC at "
        "every step."
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
        st.markdown("### ⚙️ Processing progress")
        done = len(st.session_state["completed_stages"])
        total = len(PIPELINE_STAGES)
        st.progress(done / total, text=f"{done}/{total} stages complete")
    with c3:
        st.markdown("### 🧭 Interpretation")
        st.caption(f"{len(st.session_state['horizons'])} horizon(s), "
                    f"{len(st.session_state['faults'])} fault(s), "
                    f"{len(st.session_state['attributes'])} attribute(s) computed")

    st.divider()
    st.markdown("### Workflow")
    cols = st.columns(4)
    groups = [
        ("📂 Data & Shot Gathers", PIPELINE_STAGES[0:2]),
        ("⚙️ Pre-Stack Conditioning", PIPELINE_STAGES[2:10]),
        ("🔀 CDP Sort → 🎯 Velocity/NMO/DMO", PIPELINE_STAGES[10:18]),
        ("📚 Stack → 🗺️ Migration", PIPELINE_STAGES[18:24]),
    ]
    for col, (label, stages) in zip(cols, groups):
        with col:
            st.markdown(f"**{label}**")
            for s in stages:
                mark = "✅" if ss.is_stage_done(s.id) else "⬜"
                st.caption(f"{mark} {s.label}")

    st.divider()
    st.markdown("### Quick start")
    st.markdown(
        "1. **Data Management** — upload a SEG-Y file, or generate synthetic demo data. "
        "Optionally merge in a second survey; optionally check geometry with a quick brute "
        "stack. Then build shot gathers.\n"
        "2. **Pre-Stack Conditioning** — band-pass, AGC, trace edit/kill/de-spike, mute, "
        "balance, F-K filter (auto or click-drawn polygon), multiple suppression, "
        "deconvolution — all in the shot domain.\n"
        "3. **CDP Sort & Binning** — re-group the conditioned data by CDP, with a fold map "
        "and a semblance preview.\n"
        "4. **Velocity, NMO & DMO** — pick stacking velocities by auto-pick, **click directly "
        "on the semblance panel**, or Constant-Velocity-Stack QC; apply NMO, solve residual "
        "statics, DMO.\n"
        "5. **Stacking → Post-Stack → Migration** — produce the final processed section.\n"
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
            "Built from a reference 2D land seismic-processing notebook — pre-stack conditioning "
            "(band-pass, AGC, trace edit, mute, balance, F-K, multiple suppression, "
            "deconvolution) applied in the **shot domain**, CDP sort/binning, two-pass velocity "
            "analysis/NMO/DMO, stacking, post-stack conditioning, and Kirchhoff migration — "
            "cross-checked against a standard industry processing flow, plus an interpretation "
            "layer (horizon/fault picking, industry-standard attributes, an approximate well "
            "tie) not present in the source notebook.\n\n"
            "See `README.md` for the full list of documented assumptions and deliberate "
            "deviations from the source notebook — including one real bug that was found and "
            "fixed in the NMO correction, and a picker redesign that fixed a systematic "
            "velocity-tracking failure at depth and at survey edges."
        )
