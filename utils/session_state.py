"""
utils/session_state.py — Session-State Schema & Data-State Adapter
=====================================================================
Requires Streamlit. This is the only place that owns the structure of
`st.session_state`, so every view reads/writes through these helpers rather
than touching session keys directly.

What lives in the session
-------------------------
  graph        the `ProcessingGraph`: every data state ever produced, each with the
               state(s) it consumed, the operation, its parameters and a lineage
               fingerprint (`modules/state_graph.py`).
  containers   the per-gather arrays of gather-based states   ┐ the payloads
  blobs        sections, images, velocity fields …            ┘ (`modules/stage_ops.py`)
  <aliases>    `stack_matrix`, `Vfield_1`, `gathers` … — references to the payload of
               the current/latest state of a stage, kept so simple readers don't need
               the graph. Read-only for pages.
  input_picks  which input state a person chose for an operation (UI convenience).
  <UI state>   horizons, faults, attributes, well data … (nothing to do with processing).

This module is a thin adapter: the questions "can this run?", "what would it
read?", "what is stale?" are answered by the pure engine in
`modules/dependencies.py`; everything that changes the graph goes through
`modules/stage_ops.py`, which takes `st.session_state` as its store.

`require_stage` survives only as a hard "has this been run" check for pages
that *consume* a finished product; pages that run an operation use
`utils/workflow_ui.py` (status banner + confirm-to-continue).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from modules import stage_ops
from modules.dependencies import StageCheck, Workflow
from modules.pipeline import PIPELINE_STAGES, PROCESSING_STORE_DEFAULTS, stage_label
from modules.state_graph import DataState, ProcessingGraph

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
    "last_run": None,          # {"state", "op", "variant", "reused"} of the latest execution
    "_flash": None,            # one-shot message shown after a rerun
}


def init_session_state() -> None:
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (dict, list, set)) else v
    if st.session_state.get("graph") is None:
        st.session_state["graph"] = ProcessingGraph()


# ---- the graph, and questions about it ---------------------------------------

def graph() -> ProcessingGraph:
    return stage_ops.graph(st.session_state)


def workflow() -> Workflow:
    """The dependency engine's view over the session's graph."""
    return stage_ops.workflow(st.session_state)


def check(stage_id: str, variant: Optional[str] = None,
          inputs: Optional[Dict[str, str]] = None) -> StageCheck:
    """✓ / ⚠ / ⛔ for an operation against the current data states."""
    return stage_ops.check(st.session_state, stage_id, variant, inputs)


def is_stage_done(stage_id: str, variant: Optional[str] = None) -> bool:
    return workflow().has(stage_id, variant)


def latest(stage_id: str, variant: Optional[str] = None) -> Optional[DataState]:
    """The most recent state this operation produced (or that was made current)."""
    return graph().latest(stage_id, variant)


def current(kind: str) -> Optional[DataState]:
    """The current data state of a kind ('shot', 'cdp', 'section' …)."""
    return graph().current(kind)


def source(state: Optional[DataState], role: str = "data") -> Optional[DataState]:
    """The state `state` was made from (its input for `role`)."""
    return graph().get(state.inputs.get(role)) if state is not None else None


def result(stage_id: str, variant: Optional[str] = None) -> Optional[DataState]:
    """The latest state of an operation with its arrays in memory — re-computing
    them first if they were released to save memory. None if never run."""
    state = latest(stage_id, variant)
    if state is None:
        return None
    if not state.materialized:
        with st.spinner(f"Re-computing {stage_label(state.op)} ({state.id}) from its recorded "
                        f"operation …"):
            stage_ops.materialize(st.session_state, state.id)
    return state


def gathers(state: DataState) -> dict:
    return stage_ops.gathers_of(st.session_state, state)


def section(state: Optional[DataState]):
    return stage_ops.section_of(st.session_state, state)


def blob(state: Optional[DataState], name: str, default: Any = None) -> Any:
    return stage_ops.blob_of(st.session_state, state, name, default)


def cdp_list(state: DataState) -> List[int]:
    return stage_ops.cdp_list_of(st.session_state, state)


def best_available_state():
    return stage_ops.best_available_state(st.session_state)


def best_available_section():
    return stage_ops.best_available_section(st.session_state)


# ---- moving around the graph (undo / redo / select / free / discard) ----------

def select(state_id: str) -> DataState:
    return stage_ops.select_state(st.session_state, state_id)


def undo(kind: str) -> Optional[DataState]:
    return stage_ops.undo(st.session_state, kind)


def redo(kind: str) -> Optional[DataState]:
    return stage_ops.redo(st.session_state, kind)


def discard(state_id: str) -> List[str]:
    """Delete a state and every state built on it (and free their arrays)."""
    return stage_ops.discard_state(st.session_state, state_id)


def release(state_id: str) -> int:
    """Free a state's arrays (it is re-computed from its record when next needed)."""
    return stage_ops.release_payload(st.session_state, state_id)


def set_pick(stage_id: str, role: str, state_id: Optional[str], variant: Optional[str] = None) -> None:
    stage_ops.set_pick(st.session_state, stage_id, role, state_id, variant)


def require_stage(stage_id: str, friendly_message: Optional[str] = None) -> bool:
    """Hard presence check: has `stage_id` produced a state? Use ONLY where a
    page consumes that finished product. Pages that run a stage should show
    its readiness with `utils.workflow_ui.show_status` instead — a stage that
    has not been run is not a reason to hide the page of a stage that does
    not need it."""
    if is_stage_done(stage_id):
        return True
    msg = friendly_message or (
        f"This page needs **{stage_label(stage_id)}** to have been run. "
        f"Run it first, then come back.")
    st.info(msg, icon="ℹ️")
    return False


def progress_fraction() -> float:
    """Share of all registered steps run so far. Many are optional, so 100%
    is not the goal — the number is only a rough activity gauge."""
    return len(st.session_state["completed_stages"]) / len(PIPELINE_STAGES)


__all__ = ["SESSION_DEFAULTS", "init_session_state", "graph", "workflow", "check", "is_stage_done",
           "latest", "current", "source", "result", "gathers", "section", "blob", "cdp_list",
           "best_available_state", "best_available_section", "select", "undo", "redo", "discard",
           "release", "set_pick", "require_stage", "progress_fraction"]
