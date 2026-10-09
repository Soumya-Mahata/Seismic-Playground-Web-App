"""
modules/pipeline.py — Stage Registry (recommended order) & Processing History
================================================================================
This module deliberately holds only the *descriptive* half of the
workflow model. Four things that used to be fused into "the pipeline" are
kept separate, and this file owns two of them:

  1. RECOMMENDED PROCESSING ORDER   -> `PIPELINE_STAGES` (this file).
     The conventional sequence, used for the sidebar/home layout and as
     documentation. It is advice, never an execution constraint.
  2. GENUINE COMPUTATIONAL DEPENDENCIES -> `modules/dependencies.py`.
     What a stage actually needs to exist before it can run.
  3. CURRENT DATA STATE             -> derived from the processing history
     by `modules/dependencies.WorkflowState` (which products exist, and
     what has been applied to the data a stage would read next).
  4. PROCESSING HISTORY             -> `HistoryEntry` / `log_step` (this
     file). An *execution-order* log: what was run, in what order, on
     which upstream outputs, with which parameters.

Also here: the per-stage data keys (which dict key a stage writes), so a
stage can find "the output of the stage I am reading from" without
hard-coding the previous stage's name.

No Streamlit dependency — pure data + small helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

    Retained for display/documentation only. It is NOT used for
    invalidation any more: re-running a stage invalidates exactly the
    stages that consumed its output (see `WorkflowState.dependents`),
    which is a different — and smaller — set than "everything later in
    the recommended order"."""
    i = STAGE_INDEX[stage_id]
    return [s.id for s in PIPELINE_STAGES[i:]]


def stage_label(stage_id: str) -> str:
    s = STAGE_BY_ID.get(stage_id)
    return s.label if s else stage_id


# ─────────────────────────────────────────────────────────────────────────
# 2) WHERE EACH STAGE'S OUTPUT LIVES
# ─────────────────────────────────────────────────────────────────────────
# The per-gather dict key each stage's *trace* output is stored under (an
# accumulation pattern: every stage adds a new key so earlier stages'
# outputs stay available for before/after QC). Pre-stack conditioning
# (despike..decon) runs on *shot* gathers; cdp_sort re-groups the result
# by CDP, after which the CDP-domain stages operate on *CDP* gathers.
# Because a stage reads "the output of whichever stage is upstream of it
# in the data it is actually processing", these keys are how a stage finds
# its input — no stage hard-codes its predecessor's key.
STAGE_TRACE_KEY = {
    "shot_gathers": "traces",
    "bandpass": "traces_bp",
    "agc": "traces_agc",
    "despike": "traces_despiked",
    "trace_edit": "traces_edited",
    "mute": "traces_muted",
    "balance": "traces_balanced",
    "fk": "traces_fk_filtered",
    "mult_supp": "traces_demultiple",
    "decon": "traces_decon",
    "cdp_sort": "traces",              # first CDP-domain key, after re-sorting
    "nmo1": "traces_nmo1",
    "statics": "traces_statics",
    "dmo": "traces_dmo",
    "inv_nmo": "traces_inv_nmo",
    "data_reg": "traces_regularized",
    "nmo_final": "traces_nmo_final",
}

# Offsets key paired with a CDP-domain trace key (regularisation replaces
# the per-trace offsets with a shared, synthetic offset grid).
STAGE_OFFSETS_KEY = {"data_reg": "offsets_regularized"}
DEFAULT_OFFSETS_KEY = "offsets"

# Stacked-section products live in the store (not per gather).
SECTION_KEY = {
    "stack": "stack_matrix",
    "rna": "stack_rna",
    "decon_post": "stack_decon_post",
    "tvf": "stack_tvf",
    "migration": "stack_migrated",
    "final_scale": "stack_final",
}

# Velocity-analysis products: store key of the 2-D V(t, CDP) field, and
# the per-gather key of that CDP's interpolated velocity curve.
VELOCITY_FIELD_KEY = {"vel1": "Vfield_1", "vel2": "Vfield_2"}
VELOCITY_GATHER_KEY = {"vel1": "vrms_field_vel1", "vel2": "vrms_field_vel2"}

# NMO-corrected gather products that stacking can consume:
# stage -> (trace key, live-mask key)
NMO_PRODUCT_KEYS = {
    "nmo_final": ("traces_nmo_final", "nmo_final_live_mask"),
    "nmo1": ("traces_nmo1", "nmo1_live_mask"),
}

# Store-level keys a stage owns (reset to None when the stage is invalidated).
STAGE_STORE_KEYS: Dict[str, List[str]] = {
    "upload": ["survey"],
    "shot_gathers": ["shot_gathers"],
    "agc": ["agc_config"],
    "mute": ["mute_config"],
    "fk": ["fk_polygon", "fk_config"],
    "cdp_sort": ["gathers", "cdp_list"],
    "vel1": ["velocities", "Vfield_1", "analysis_1", "vel_results_1"],
    "statics": ["shot_static", "receiver_static", "static_diag"],
    "data_reg": ["regular_offsets"],
    "vel2": ["Vfield_2", "analysis_2", "vel_results_2"],
    "stack": ["stack_matrix", "fold_matrix", "naive_matrix"],
    "rna": ["stack_rna"],
    "decon_post": ["stack_decon_post"],
    "tvf": ["stack_tvf"],
    "migration": ["stack_migrated", "pstm_image", "pstm_fold"],
    "final_scale": ["stack_final"],
}

# Per-gather keys a stage owns, inside the shot-gather or CDP-gather dicts
# (cleared when the stage is invalidated so stale arrays never linger).
# stage -> (container store key, per-gather keys)
STAGE_GATHER_KEYS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "bandpass": ("shot_gathers", ("traces_bp",)),
    "agc": ("shot_gathers", ("traces_agc",)),
    "despike": ("shot_gathers", ("traces_despiked", "_spike_mask")),
    "trace_edit": ("shot_gathers", ("traces_edited", "_kill_flags")),
    "mute": ("shot_gathers", ("traces_muted", "_mute_times", "_surgical_box")),
    "balance": ("shot_gathers", ("traces_balanced",)),
    "fk": ("shot_gathers", ("traces_fk_filtered",)),
    "mult_supp": ("shot_gathers", ("traces_demultiple",)),
    "decon": ("shot_gathers", ("traces_decon",)),
    "vel1": ("gathers", ("semblance_vel1", "vrms_vel1", "vrms_confidence_vel1", "vrms_field_vel1")),
    "nmo1": ("gathers", ("traces_nmo1", "nmo1_live_mask")),
    "statics": ("gathers", ("traces_statics",)),
    "dmo": ("gathers", ("traces_nmo1_statics", "traces_dmo")),
    "inv_nmo": ("gathers", ("traces_inv_nmo",)),
    "data_reg": ("gathers", ("traces_regularized", "offsets_regularized", "live_regularized")),
    "vel2": ("gathers", ("semblance_vel2", "vrms_vel2", "vrms_confidence_vel2", "vrms_field_vel2")),
    "nmo_final": ("gathers", ("traces_nmo_final", "nmo_final_live_mask")),
    "stack": ("gathers", ("stack_dmo", "fold", "stack_naive_mean")),
}

# Processing-related defaults of the data store. (The Streamlit layer adds
# its own UI-only keys on top of these in `utils/session_state.py`.)
PROCESSING_STORE_DEFAULTS: Dict[str, Any] = {
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
    "fk_polygon": None, "fk_config": None,
    "agc_config": None, "mute_config": None,
    "regular_offsets": None,
    "pstm_image": None, "pstm_fold": None,
}


# ─────────────────────────────────────────────────────────────────────────
# 3) PROCESSING HISTORY (execution order)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    stage_id: str
    label: str
    params: Dict[str, Any]
    timestamp: str
    notes: str = ""
    # --- provenance (what makes the history a record of the *data flow*) ---
    seq: int = 0                                   # execution order, 1-based
    consumed: Dict[str, str] = field(default_factory=dict)  # role -> upstream stage id
    input_key: str = ""                            # data key this run read
    output_key: str = ""                           # data key this run wrote
    offsets_key: str = ""                          # offsets key paired with `output_key`
    variant: str = ""                              # e.g. migration: "poststack" | "pstm"


def log_step(history: List[HistoryEntry], stage_id: str, params: Optional[Dict[str, Any]] = None,
             notes: str = "", *, consumed: Optional[Dict[str, str]] = None, input_key: str = "",
             output_key: str = "", offsets_key: str = "", variant: str = "") -> List[HistoryEntry]:
    """Append a step to the processing history.

    The history is in EXECUTION order — it records what was actually done,
    not where a stage sits in the recommended sequence. Re-running a stage
    supersedes its previous entry (the stage is then the most recent step).
    """
    next_seq = max((h.seq for h in history), default=0) + 1
    history = [h for h in history if h.stage_id != stage_id]
    history.append(HistoryEntry(
        stage_id=stage_id, label=stage_label(stage_id), params=params or {},
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes=notes,
        seq=next_seq, consumed=dict(consumed or {}), input_key=input_key,
        output_key=output_key, offsets_key=offsets_key, variant=variant,
    ))
    return history


def history_to_report_lines(history: List[HistoryEntry]) -> List[str]:
    lines = []
    for i, h in enumerate(sorted(history, key=lambda e: e.seq), 1):
        lines.append(f"{i}. {h.label}  ({h.timestamp})")
        if h.consumed:
            src = ", ".join(f"{role}←{stage_label(sid)}" for role, sid in h.consumed.items())
            lines.append(f"     inputs: {src}")
        for k, v in h.params.items():
            lines.append(f"     {k}: {v}")
        if h.notes:
            lines.append(f"     note: {h.notes}")
    return lines
