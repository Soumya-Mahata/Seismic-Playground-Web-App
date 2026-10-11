"""
modules/pipeline.py — Stage Registry (recommended order) & what each stage produces
======================================================================================
The *descriptive* half of the workflow model. Four things that used to be
fused into "the pipeline" are kept apart, and this file owns the first:

  1. RECOMMENDED PROCESSING ORDER   -> `PIPELINE_STAGES` (this file).
     The conventional sequence, used for the sidebar/home layout and as
     documentation. It is advice, never an execution constraint.
  2. GENUINE COMPUTATIONAL DEPENDENCIES -> `modules/dependencies.py`.
     What a stage actually needs to exist before it can run.
  3. DATA STATES                    -> `modules/state_graph.py`.
     What data actually exists: every operation consumes states and
     produces a new one (INPUT STATE -> OPERATION + PARAMETERS -> OUTPUT
     STATE), recorded with a stable id and lineage fingerprint.
  4. PROCESSING HISTORY             -> the states, in creation order.

Also here: the *kind* of data each stage produces (`STAGE_KIND`), which is
how a stage asks for "the current shot-domain data" without naming the
stage that happened to make it, and the data-store defaults.

No Streamlit dependency — pure data + small helpers.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import config as cfg


# ─────────────────────────────────────────────────────────────────────────
# 1) RECOMMENDED ORDER
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
# Which stages genuinely need which others is declared in
# `modules/dependencies.STAGE_SPECS`.
PIPELINE_STAGES: List[Stage] = [
    Stage("upload", "Load Data", "Read a SEG-Y survey (or generate synthetic test data).", "1", "data"),
    Stage("shot_gathers", "Build Shot Gathers", "Group traces by shot (FieldRecord) — the starting "
          "point for shot-domain pre-stack conditioning.", "1", "data", optional=True),
    Stage("bandpass", "Band-Pass Filter", "Trapezoidal frequency-domain band-pass (shot domain).",
          "2", "prestack", optional=True),
    Stage("agc", "Amplitude Recovery (AGC)", "Sliding-window automatic gain control (shot domain).",
          "3", "prestack", optional=True),
    Stage("despike", "De-spike", "Remove burst-noise spikes sample-by-sample (shot domain).",
          "4a", "prestack", optional=True),
    Stage("trace_edit", "Trace Edit / Kill", "Zero dead/abnormally noisy traces (shot domain).",
          "4b", "prestack", optional=True),
    Stage("mute", "Mute", "Top mute (flat or offset-dependent) and surgical mute (shot domain).",
          "5", "prestack", optional=True),
    Stage("balance", "Trace Balancing", "Scalar RMS normalization per trace (shot domain).",
          "6", "prestack", optional=True),
    Stage("fk", "F-K Dip Filter", "2-D FFT polygon mask — ground-roll attenuation (shot domain).",
          "7", "prestack", optional=True),
    Stage("mult_supp", "Multiple Suppression", "Predictive (gap) deconvolution — attenuates short-period "
          "reverberations/multiples.", "8", "prestack", optional=True),
    Stage("decon", "Deconvolution", "Spiking deconvolution per trace (shot domain).",
          "9", "prestack", optional=True),
    Stage("cdp_sort", "CDP Sort / Binning", "Re-group the traces (conditioned, if any conditioning "
          "was applied) by CDP ensemble number.", "10", "velocity"),
    Stage("vel1", "Velocity Analysis — Pass 1", "Semblance velocity analysis (approximate); auto-pick, "
          "manual click-pick, or Constant-Velocity Stack.", "11", "velocity"),
    Stage("nmo1", "NMO Correction — Pass 1", "Apply pass-1 stacking velocities to the CDP-sorted gathers.",
          "12", "velocity"),
    Stage("statics", "Residual Statics", "Surface-consistent shot/receiver statics.",
          "13", "velocity", optional=True),
    Stage("dmo", "DMO Correction", "Simplified dip-moveout time-stretch.", "14", "velocity", optional=True),
    Stage("inv_nmo", "Inverse NMO", "Undo pass-1 NMO on the DMO-corrected data.", "15", "velocity",
          optional=True),
    Stage("data_reg", "Data Regularisation", "Interpolate each CDP gather onto a common, regularly "
          "spaced offset grid — recommended before velocity analysis, stacking, and migration.",
          "15b", "velocity", optional=True),
    Stage("vel2", "RMS Velocity Analysis (Pass 2)", "Refined semblance velocity analysis on the "
          "refined (statics/DMO/regularised) gathers.", "16", "velocity", optional=True),
    Stage("nmo_final", "Final NMO", "Apply the best available stacking velocities (pass 2, else pass 1) "
          "to the current pre-NMO gathers.", "17", "velocity"),
    Stage("stack", "Stack", "Fold-normalized CDP stack of NMO-corrected gathers.", "18", "stack"),
    Stage("rna", "Random Noise Attenuation", "Spatial median filter across CDPs.", "19", "stack",
          optional=True),
    Stage("decon_post", "Deconvolution (post-stack)", "A milder second spiking-deconvolution pass.",
          "20", "stack", optional=True),
    Stage("tvf", "Time-Variant Filter", "Depth-zoned band-pass blending.", "21", "stack", optional=True),
    Stage("migration", "Migration", "Post-stack Kirchhoff time migration, or pre-stack Kirchhoff "
          "time migration (PSTM) on CDP gathers before stack.", "22", "stack", optional=True),
    Stage("final_scale", "Final Scaling", "Display AGC + re-applied top mute.", "23", "stack",
          optional=True),
]

STAGE_INDEX = {s.id: i for i, s in enumerate(PIPELINE_STAGES)}
STAGE_BY_ID = {s.id: s for s in PIPELINE_STAGES}

# Display grouping (home page): (heading, group id)
STAGE_GROUPS: List[Tuple[str, str]] = [
    ("📂 Data & Shot Gathers", "data"),
    ("⚙️ Pre-Stack Conditioning", "prestack"),
    ("🔀 CDP Sort → 🎯 Velocity/NMO/DMO", "velocity"),
    ("📚 Stack → 🗺️ Migration", "stack"),
]


def stages_from(stage_id: str) -> List[str]:
    """Every stage at or after `stage_id` in the RECOMMENDED order.

    Retained for display/documentation only. Nothing is invalidated by
    position in the recommended order: states are never overwritten, and
    what depends on a state is read from the graph (`ProcessingGraph.dependents`)."""
    i = STAGE_INDEX[stage_id]
    return [s.id for s in PIPELINE_STAGES[i:]]


def stage_label(stage_id: str) -> str:
    s = STAGE_BY_ID.get(stage_id)
    return s.label if s else stage_id


# ─────────────────────────────────────────────────────────────────────────
# 2) WHAT EACH STAGE PRODUCES
# ─────────────────────────────────────────────────────────────────────────
# The KIND of data state a stage's output is (see `state_graph.KINDS`). A
# stage that wants "the shot-domain data as it is now" asks for '@shot'
# and gets whichever state of that kind is current — it never names the
# stage that produced it, which is what lets operations run in any order.
STAGE_KIND: Dict[str, str] = {
    "upload": "raw",
    "shot_gathers": "shot",
    "bandpass": "shot", "agc": "shot", "despike": "shot", "trace_edit": "shot", "mute": "shot",
    "balance": "shot", "fk": "shot", "mult_supp": "shot", "decon": "shot",
    "cdp_sort": "cdp",
    "vel1": "velocity", "vel2": "velocity",
    "nmo1": "nmo", "nmo_final": "nmo",
    "statics": "cdp", "inv_nmo": "cdp", "data_reg": "cdp",
    "dmo": "dmo",
    "stack": "section", "rna": "section", "decon_post": "section", "tvf": "section",
    "migration": "image",
    "final_scale": "final",
}

# Stages that start a new set of gathers (their output owns a gather
# container; states derived from it share that container).
ROOT_STAGES: Tuple[str, ...] = ("shot_gathers", "cdp_sort")


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
