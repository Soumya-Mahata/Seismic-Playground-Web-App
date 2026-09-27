"""
utils/session_state.py — Session-State Schema & Pipeline-State Machine
==========================================================================
Requires Streamlit. This is the only place that owns `st.session_state`
structure, so every view reads/writes through these helpers rather than
touching `st.session_state` keys directly — keeps the schema in one place
and makes the invalidation rule ("re-running an earlier stage clears
everything after it, for reproducibility") enforceable in one function.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

import config as cfg
from modules.pipeline import PIPELINE_STAGES, log_step, stages_from

# Which session_state keys a given stage "owns" — cleared on invalidation.
STAGE_STATE_KEYS = {
    "upload": ["survey"],
    "shot_gathers": ["shot_gathers"],
    "bandpass": [],   # per-gather keys live inside shot_gathers[id][...], overwritten in place
    "agc": ["agc_config"],
    "trace_edit": [],
    "mute": ["mute_config"],
    "balance": [],
    "fk": ["fk_polygon"],
    "mult_supp": [],
    "decon": [],
    "cdp_sort": ["gathers", "cdp_list"],
    "vel1": ["velocities", "Vfield_1", "analysis_1", "vel_results_1"],
    "nmo1": [],
    "statics": ["shot_static", "receiver_static", "static_diag"],
    "dmo": [],
    "inv_nmo": [],
    "data_reg": ["regular_offsets"],
    "vel2": ["Vfield_2", "analysis_2", "vel_results_2"],
    "nmo_final": [],
    "stack": ["stack_matrix", "fold_matrix", "naive_matrix"],
    "rna": ["stack_rna"],
    "decon_post": ["stack_decon_post"],
    "tvf": ["stack_tvf"],
    "migration": ["stack_migrated", "pstm_image", "pstm_fold"],
    "final_scale": ["stack_final"],
}

SESSION_DEFAULTS: Dict[str, Any] = {
    "survey": None,
    "shot_gathers": None,
    "gathers": None,
    "cdp_list": None,
    "n_min_traces": cfg.N_MIN_TRACES_DEFAULT,
    "completed_stages": set(),
    "history": [],
    "velocities": None,
    "Vfield_1": None, "analysis_1": None, "vel_results_1": None,
    "Vfield_2": None, "analysis_2": None, "vel_results_2": None,
    "manual_overrides_1": {}, "manual_overrides_2": {},
    "excluded_cdps_1": set(), "excluded_cdps_2": set(),
    "shot_static": None, "receiver_static": None, "static_diag": None,
    "stack_matrix": None, "fold_matrix": None, "naive_matrix": None,
    "stack_rna": None, "stack_decon_post": None, "stack_tvf": None,
    "stack_migrated": None, "stack_final": None,
    "horizons": {}, "faults": {},
    "attributes": {},
    "well_data": None,
    "synthetic_tie": None,
    "active_page": "🏠 Home",
    "_brute_stack": None,
    "fk_polygon": None,
    "fk_click_points": [],
    "_semblance_preview": None,
    "agc_config": None,
    "mute_config": None,
    "regular_offsets": None,
    "pstm_image": None,
    "pstm_fold": None,
}


def init_session_state() -> None:
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (dict, list, set)) else v


def is_stage_done(stage_id: str) -> bool:
    return stage_id in st.session_state["completed_stages"]


def mark_stage_done(stage_id: str, params: Optional[dict] = None, notes: str = "") -> None:
    """Record a stage as complete, invalidating every later stage first
    (so re-running step N with new parameters can never leave stale
    step-N+2 output sitting around looking valid)."""
    invalidate_from(stage_id, include_self=False)
    st.session_state["completed_stages"].add(stage_id)
    st.session_state["history"] = log_step(st.session_state["history"], stage_id, params, notes)


def invalidate_from(stage_id: str, include_self: bool = True) -> None:
    ids = stages_from(stage_id)
    if not include_self:
        ids = ids[1:]
    for sid in ids:
        st.session_state["completed_stages"].discard(sid)
        for key in STAGE_STATE_KEYS.get(sid, []):
            st.session_state[key] = SESSION_DEFAULTS.get(key)
        st.session_state["history"] = [h for h in st.session_state["history"] if h.stage_id != sid]


def require_stage(stage_id: str, friendly_message: Optional[str] = None) -> bool:
    """Guard clause for a view: `if not require_stage("gathers"): return`.
    Shows an informational message (not an error — this is normal
    workflow sequencing, not a fault) and returns False if the
    prerequisite stage hasn't run yet."""
    if is_stage_done(stage_id):
        return True
    from modules.pipeline import stage_label
    msg = friendly_message or (
        f"This step needs **{stage_label(stage_id)}** to be completed first. "
        f"Head to that step in the sidebar and run it, then come back."
    )
    st.info(msg, icon="ℹ️")
    return False


def progress_fraction() -> float:
    return len(st.session_state["completed_stages"]) / len(PIPELINE_STAGES)
