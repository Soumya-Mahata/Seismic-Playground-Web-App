"""
modules/pipeline.py — the recommended order, and the data-store layout
======================================================================
Operation metadata is declared ONCE, in `modules/registry.py`. This file is
the *view* of it that the rest of the application has always imported
(`PIPELINE_STAGES`, `STAGE_KIND`, `ROOT_STAGES`, `STAGE_GROUPS` …): nothing
here is declared a second time, it is all derived.

Four things that used to be fused into "the pipeline" are kept apart:

  1. RECOMMENDED PROCESSING ORDER   -> `PIPELINE_STAGES` (this file, from the
     registry's declaration order). The conventional sequence, used for the
     sidebar/home layout and as documentation. Advice, never an execution
     constraint.
  2. GENUINE COMPUTATIONAL DEPENDENCIES -> declared in the registry
     (`Operation.needs`), evaluated by `modules/dependencies.py`.
     What a stage actually needs to exist before it can run.
  3. DATA STATES                    -> `modules/state_graph.py`.
     What data actually exists: every operation consumes states and
     produces a new one (INPUT STATE -> OPERATION + PARAMETERS -> OUTPUT
     STATE), recorded with a stable id and lineage fingerprint.
  4. PROCESSING HISTORY             -> `modules/action_log.py`, the immutable
     chronological log of what the person did.

Also here: the data-store defaults (the schema of the store `stage_ops` works on).

No Streamlit dependency — pure data + small helpers.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import config as cfg
from modules import registry


# ─────────────────────────────────────────────────────────────────────────
# 1) RECOMMENDED ORDER  (derived from the registry)
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    description: str
    step_number: str  # this stage's numbered position in the *recommended* workflow
    group: str = ""   # "data" | "prestack" | "velocity" | "stack" — for display grouping
    optional: bool = False  # True: no other stage *needs* this one, so it can always be skipped


# NOTE: this list is the RECOMMENDED order. Nothing in the app requires a
# stage to run just because the stages before it in this list have run.
# Which stages genuinely need which others is declared in the registry.
PIPELINE_STAGES: List[Stage] = [
    Stage(o.id, o.display_name, registry.stage_description(o.id), o.step_number, o.group, o.optional)
    for o in registry.stages()]

STAGE_INDEX = dict(registry.STAGE_INDEX)
STAGE_BY_ID = {s.id: s for s in PIPELINE_STAGES}

# Display grouping (home page): (heading, group id)
STAGE_GROUPS: List[Tuple[str, str]] = [(heading, gid) for gid, heading in registry.GROUPS]


def stages_from(stage_id: str) -> List[str]:
    """Every stage at or after `stage_id` in the RECOMMENDED order.

    Retained for display/documentation only. Nothing is invalidated by
    position in the recommended order: states are never overwritten, and
    what depends on a state is read from the graph (`ProcessingGraph.dependents`)."""
    i = STAGE_INDEX[stage_id]
    return [s.id for s in PIPELINE_STAGES[i:]]


stage_label = registry.stage_label


# ─────────────────────────────────────────────────────────────────────────
# 2) WHAT EACH STAGE PRODUCES  (derived from the registry)
# ─────────────────────────────────────────────────────────────────────────
# The KIND of data state a stage's output is (see `state_graph.KINDS`). A
# stage that wants "the shot-domain data as it is now" asks for '@shot'
# and gets whichever state of that kind is current — it never names the
# stage that produced it, which is what lets operations run in any order.
STAGE_KIND: Dict[str, str] = dict(registry.STAGE_KIND)

# Stages that start a new set of gathers (their output owns a gather
# container; states derived from it share that container).
ROOT_STAGES: Tuple[str, ...] = registry.ROOT_STAGES


# ─────────────────────────────────────────────────────────────────────────
# 3) DATA STORE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────
# `graph`, `containers` and `blobs` are the real state (see
# `modules/stage_ops.py`): the graph records every data state; `containers`
# holds the per-gather arrays of gather-based states; `blobs` the arrays of
# everything else. All other keys below are *reference aliases* to the
# payload of the current/latest state of each stage, refreshed after every
# change (`stage_ops.sync_aliases`) — they cost no memory and exist so
# the pages and tests that read e.g. `store["stack_matrix"]` keep working —
# EXCEPT the user inputs (`manual_overrides_*`, `excluded_cdps_*`,
# `input_picks`), which are inputs, not results.
PROCESSING_STORE_DEFAULTS: Dict[str, Any] = {
    "survey": None,
    "graph": None,
    "containers": {},
    "blobs": {},
    "input_picks": {},
    "manual_overrides_1": {}, "manual_overrides_2": {},
    "excluded_cdps_1": set(), "excluded_cdps_2": set(),
    "n_min_traces": cfg.N_MIN_TRACES_DEFAULT,
    # ---- aliases of current/latest payloads (read-only for pages) ----
    "completed_stages": set(),
    "shot_gathers": None, "gathers": None, "cdp_list": None,
    "velocities": None,
    "Vfield_1": None, "analysis_1": None, "vel_results_1": None,
    "Vfield_2": None, "analysis_2": None, "vel_results_2": None,
    "shot_static": None, "receiver_static": None, "static_diag": None,
    "stack_matrix": None, "fold_matrix": None, "naive_matrix": None,
    "stack_rna": None, "stack_decon_post": None, "stack_tvf": None,
    "stack_migrated": None, "stack_final": None,
    "fk_polygon": None, "fk_config": None,
    "agc_config": None, "mute_config": None,
    "regular_offsets": None,
    "pstm_image": None, "pstm_fold": None,
}

# The alias keys (everything above the user inputs) — cleared and rebuilt by
# `sync_aliases`.
ALIAS_KEYS: Tuple[str, ...] = tuple(
    k for k in PROCESSING_STORE_DEFAULTS
    if k not in ("survey", "graph", "containers", "blobs", "input_picks", "manual_overrides_1",
                 "manual_overrides_2", "excluded_cdps_1", "excluded_cdps_2", "n_min_traces"))


def history_to_report_lines(graph) -> List[str]:
    """The processing history as plain text lines, in creation order, with
    the state each step consumed and produced."""
    lines = []
    for s in graph:
        lines.append(f"{s.id}  {stage_label(s.op)}"
                     + (f" ({s.variant})" if s.variant else "")
                     + (f" v{s.version}" if s.version > 1 else "") + f"   [{s.timestamp}]")
        if s.inputs:
            lines.append("     consumed: " + ", ".join(f"{role}={sid}" for role, sid in s.inputs.items()))
        lines.append(f"     fingerprint: {s.fingerprint}")
        for k, v in s.params.items():
            lines.append(f"     {k}: {v}")
        for k, v in s.info.items():
            lines.append(f"     ({k}: {v})")
        if s.notes:
            lines.append(f"     note: {s.notes}")
    return lines
