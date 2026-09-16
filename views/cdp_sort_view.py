"""
views/cdp_sort_view.py — CDP Sort / Binning (after pre-stack conditioning)
=============================================================================
Re-groups the fully pre-stack-conditioned shot-domain data by CDP
ensemble number — the standard point in a real workflow where processing
switches from the shot domain to the CDP domain, right before velocity
analysis. Scatters the conditioned traces back to a flat, survey-indexed
array (`segy_io.scatter_to_flat`) using each shot gather's stored
`trace_indices`, then re-groups that array by CDP
(`segy_io.build_cdp_gathers(..., data_override=...)`) — the geometry
headers never changed, only the sample values did.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules import segy_io, velocity as vel_mod
from modules.validation import ValidationError, validate_cdp_gathers
from utils import session_state as ss
from visualization import seismic_plots, velocity_plots


def render():
    st.header("🔀 CDP Sort & Binning")
    st.caption("Re-group the now fully pre-stack-conditioned data by CDP ensemble, ready for "
                "velocity analysis.")

    if not ss.require_stage("decon"):
        return

    survey = st.session_state["survey"]
    shot_gathers = st.session_state["shot_gathers"]

    n_min = st.slider("Minimum fold to keep a CDP", 1, 40, st.session_state["n_min_traces"],
                       help="CDPs with fewer live traces than this are dropped as unreliable "
                            "edge-of-survey ensembles.")
    if st.button("Sort into CDP gathers", type="primary"):
        try:
            flat = segy_io.scatter_to_flat(shot_gathers, "traces_decon", survey.num_traces, survey.num_samples)
            gathers = segy_io.build_cdp_gathers(survey, n_min_traces=n_min, data_override=flat)
            validate_cdp_gathers(gathers, n_min)
            st.session_state["n_min_traces"] = n_min
            st.session_state["gathers"] = gathers
            st.session_state["cdp_list"] = list(gathers.keys())
            ss.mark_stage_done("cdp_sort", params={"n_min_traces": n_min, "n_cdp": len(gathers)})
            st.rerun()
        except ValidationError as e:
            st.error(str(e))

    if not ss.is_stage_done("cdp_sort"):
        return

    gathers = st.session_state["gathers"]
    cdp_list = st.session_state["cdp_list"]
    folds = np.array([g["traces"].shape[0] for g in gathers.values()])
    st.success(f"✅ {len(cdp_list)} CDP gathers built (fold {folds.min()}-{folds.max()}) from "
                f"pre-stack-conditioned data. Continue to **Velocity, NMO & DMO**.")
    st.plotly_chart(seismic_plots.plot_fold_map(folds, cdp_list), theme=None, key="cdpsort_fold_map")

    with st.expander("Semblance preview (full velocity picking is on the next page)"):
        cdp_preview = st.selectbox("Preview CDP", cdp_list, index=len(cdp_list) // 2, key="cdpsort_preview_cdp")
        g = gathers[cdp_preview]
        st.plotly_chart(seismic_plots.plot_gather(g["traces"], g["offsets"], survey.twt,
                         title=f"CDP {cdp_preview} gather ({g['traces'].shape[0]} traces)"),
                         theme=None, key="cdpsort_gather_preview")
        if st.button("Compute semblance for this CDP", key="cdpsort_semblance_btn"):
            velocities = np.linspace(1500, 4500, 60)
            S = vel_mod.compute_semblance_vectorized(g["traces"], g["offsets"], velocities, survey.sample_interval)
            st.session_state["_semblance_preview"] = (S, velocities)
        if st.session_state.get("_semblance_preview") is not None:
            S, velocities = st.session_state["_semblance_preview"]
            st.plotly_chart(velocity_plots.plot_semblance_panel(
                S, velocities, survey.twt, title=f"Semblance — CDP {cdp_preview} (preview)"),
                theme=None, key="cdpsort_semblance_chart")
