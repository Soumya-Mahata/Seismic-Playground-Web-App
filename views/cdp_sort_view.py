"""
views/cdp_sort_view.py — CDP Sort / Binning
=============================================
Re-groups the shot-domain data by CDP ensemble number — the point where
processing switches from the shot domain to the CDP domain, right before
velocity analysis. Scatters the traces back to a flat, survey-indexed
array (`segy_io.scatter_to_flat`) using each shot gather's stored
`trace_indices`, then re-groups that array by CDP
(`segy_io.build_cdp_gathers(..., data_override=...)`) — the geometry
headers never changed, only the sample values did.

Which shot-domain data is sorted is whatever the *current head* of the
shot-domain data is: every conditioning step is optional, so this page
needs only the survey (sorting just re-groups traces). Skipped
conditioning is shown as a note, not a barrier.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules import stage_ops, velocity as vel_mod
from utils import session_state as ss
from utils import workflow_ui as wf
from visualization import seismic_plots, velocity_plots


def render():
    st.header("🔀 CDP Sort & Binning")
    st.caption("Re-group the shot-domain data by CDP ensemble, ready for velocity analysis. "
                "Any pre-stack conditioning you chose to apply is carried into the CDP gathers; "
                "conditioning you skipped is simply not applied.")

    chk = wf.show_status("cdp_sort")
    if not chk.can_run:
        return

    survey = st.session_state["survey"]
    wf.data_state_caption("shot", "Shot-domain data that will be sorted")
    wf.stale_banner(["vel1", "nmo1", "statics", "dmo", "inv_nmo", "data_reg", "vel2", "nmo_final",
                      "stack"])

    src = chk.inputs.get("data")
    if src == "upload":
        n_in = survey.num_traces
        st.caption(f"No shot gathers were built, so the raw survey ({n_in:,} traces) is sorted directly "
                    f"— sorting only needs the headers.")
    else:
        shot_gathers = st.session_state["shot_gathers"]
        n_in = sum(g["traces"].shape[0] for g in shot_gathers.values())
        st.caption(f"Sanity check: {n_in:,} traces going in "
                    f"(survey has {survey.num_traces:,}). Sorting by CDP only re-groups these — it "
                    f"cannot change the total.")

    n_min = st.slider("Minimum fold to keep a CDP", 1, 40, st.session_state["n_min_traces"],
                       help="CDPs with fewer live traces than this are dropped as unreliable "
                            "edge-of-survey ensembles.")
    if wf.run_button("cdp_sort", "Sort into CDP gathers", key="run_cdp_sort"):
        wf.execute(stage_ops.run_cdp_sort, n_min_traces=n_min)

    wf.undo_panel(["cdp_sort"], "undo_cdp")

    if not ss.is_stage_done("cdp_sort"):
        return

    gathers = st.session_state["gathers"]
    cdp_list = st.session_state["cdp_list"]
    folds = np.array([g["traces"].shape[0] for g in gathers.values()])
    e = ss.entry("cdp_sort")
    st.success(f"✅ {len(cdp_list)} CDP gathers built (fold {folds.min()}-{folds.max()}, "
                f"{int(folds.sum()):,} traces total) from: **{wf.input_name(e)}**. "
                f"Continue to **Velocity, NMO & DMO**.")
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
