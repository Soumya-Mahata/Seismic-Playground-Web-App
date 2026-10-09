"""
utils/session_state.py — Session-State Schema & Workflow-State Adapter
=========================================================================
Requires Streamlit. This is the only place that owns `st.session_state`
structure, so every view reads/writes through these helpers rather than
touching `st.session_state` keys directly.

What changed from the old linear design
---------------------------------------
Previously this module owned a linear "state machine": marking stage N
done cleared every stage after N *in the recommended order*, and
`require_stage` made each page refuse to render until the previous
stage's flag was set. Both conflated the recommended order with what a
stage actually needs.

Now this module is a thin adapter over the pure dependency engine:

  * `check()` / `workflow()`  — ask `modules/dependencies` what a stage
    needs and what it recommends, against the data state derived from
    `st.session_state["history"]`.
  * `invalidate_from()`        — drops a stage and *only the stages that
    consumed its output* (see `WorkflowState.dependents`).
  * all execution lives in `modules/stage_ops.py`, which takes
    `st.session_state` as its data store.

`require_stage` survives only as a hard "has this been run" check for
pages that consume a finished product; the pages that *run* stages use
`utils/workflow_ui.py` instead (status banner + confirm-to-continue).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from modules import stage_ops
from modules.dependencies import StageCheck, WorkflowState
from modules.pipeline import (PIPELINE_STAGES, PROCESSING_STORE_DEFAULTS, STAGE_STORE_KEYS,
                              HistoryEntry, stage_label)

# Kept under its old name: the store-level keys each stage owns.
STAGE_STATE_KEYS = STAGE_STORE_KEYS

SESSION_DEFAULTS: Dict[str, Any] = {
    **PROCESSING_STORE_DEFAULTS,
    # ---- interpretation / UI-only state ----
    "horizons": {}, "faults": {},
    "attributes": {},
    "well_data": None,
    "synthetic_tie": None,
    "active_page": "🏠 Home",
    "_brute_stack": None,
    "fk_click_points": [],
    "_semblance_preview": None,
}


def init_session_state() -> None:
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (dict, list, set)) else v


# ---- workflow state -----------------------------------------------------

def workflow() -> WorkflowState:
    """Live view of the processing history as a dependency-aware state."""
    return stage_ops.workflow(st.session_state)


def check(stage_id: str, variant: Optional[str] = None) -> StageCheck:
    """✓ / ⚠ / ⛔ for a stage against the current data state."""
    return stage_ops.check(st.session_state, stage_id, variant)


def is_stage_done(stage_id: str) -> bool:
    return workflow().has(stage_id)


def entry(stage_id: str) -> Optional[HistoryEntry]:
    return workflow().entry(stage_id)


def invalidate_from(stage_id: str, include_self: bool = True) -> list:
    """Drop `stage_id` and every stage that consumed its output (and
    free their data). Stages that merely come later in the recommended
    order, but never read this stage's output, are left alone."""
    return stage_ops.invalidate_stage(st.session_state, stage_id, include_self=include_self)


def best_available_section():
    return stage_ops.best_available_section(st.session_state)


def require_stage(stage_id: str, friendly_message: Optional[str] = None) -> bool:
    """Hard presence check: has `stage_id` been run (and is it still
    valid)? Use ONLY where a page consumes that finished product. Pages
    that run a stage should show its readiness with
    `utils.workflow_ui.show_status` instead — a stage that has not been
    run is not a reason to hide the page of a stage that does not need it."""
    if is_stage_done(stage_id):
        return True
    msg = friendly_message or (
        f"This page needs **{stage_label(stage_id)}** to have been run. "
        f"Run it first, then come back.")
    st.info(msg, icon="ℹ️")
    return False


def progress_fraction() -> float:
    """Share of all registered steps run so far. Many are optional, so
    100% is not the goal — the number is only a rough activity gauge."""
    return len(st.session_state["completed_stages"]) / len(PIPELINE_STAGES)
