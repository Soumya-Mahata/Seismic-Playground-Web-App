"""
modules/pipeline.py — Pipeline Stage Registry & Processing History
======================================================================
A single ordered list of every stage in the 22-step reference pipeline,
with human-readable labels/descriptions and the session-state key each
stage's output is stored under. This is the "spine" the whole app is
built around:
  * the sidebar/page flow follows this order,
  * `utils/session_state.py` uses it to know what to invalidate when an
    earlier stage is re-run with different parameters,
  * the Processing History / report views read it to render a
    human-readable log.

No Streamlit dependency — pure data + small helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    description: str
    notebook_step: str  # which numbered step of the reference notebook this implements


PIPELINE_STAGES: List[Stage] = [
    Stage("upload", "Load Data", "Read a SEG-Y survey (or generate synthetic test data).", "1"),
    Stage("shot_gathers", "Build Shot Gathers", "Group traces by shot (FieldRecord) — the real starting "
          "point for pre-stack conditioning.", "1"),
    Stage("bandpass", "Band-Pass Filter", "Trapezoidal frequency-domain band-pass (shot domain).", "2"),
    Stage("agc", "Amplitude Recovery (AGC)", "Sliding-window automatic gain control (shot domain).", "3"),
    Stage("trace_edit", "Trace Edit / Kill / De-spike", "Kill dead/noisy traces; remove burst-noise spikes.", "4"),
    Stage("mute", "Mute", "Top mute (moveout-following or flat) or surgical mute, cosine-tapered "
          "(shot domain).", "5"),
    Stage("balance", "Trace Balancing", "Scalar RMS normalization per trace (shot domain).", "6"),
    Stage("fk", "F-K Dip Filter", "2-D FFT polygon mask — ground-roll attenuation (shot domain).", "7"),
    Stage("mult_supp", "Multiple Suppression", "Predictive (gap) deconvolution — attenuates short-period "
          "reverberations/multiples.", "8"),
    Stage("decon", "Deconvolution", "Spiking deconvolution per trace (shot domain).", "9"),
    Stage("cdp_sort", "CDP Sort / Binning", "Re-group the now pre-stack-conditioned traces by CDP "
          "ensemble number.", "10"),
    Stage("vel1", "Velocity Analysis — Pass 1", "Semblance velocity analysis (approximate); auto-pick, "
          "manual click-pick, or Constant-Velocity Stack.", "11"),
    Stage("nmo1", "NMO Correction — Pass 1", "Apply pass-1 stacking velocities.", "12"),
    Stage("statics", "Residual Statics", "Surface-consistent shot/receiver statics.", "13"),
    Stage("dmo", "DMO Correction", "Simplified dip-moveout time-stretch.", "14"),
    Stage("inv_nmo", "Inverse NMO", "Undo pass-1 NMO on the DMO-corrected data.", "15"),
    Stage("data_reg", "Data Regularisation", "Interpolate each CDP gather onto a common, regularly "
          "spaced offset grid — the prerequisite for reliable velocity analysis, stacking, and "
          "migration (report Sec 6.11).", "15b"),
    Stage("vel2", "RMS Velocity Analysis (Pass 2)", "Refined semblance velocity analysis on "
          "regularized data.", "16"),
    Stage("nmo_final", "Final NMO", "Apply pass-2 (refined) stacking velocities.", "17"),
    Stage("stack", "Stack", "Fold-normalized CDP stack.", "18"),
    Stage("rna", "Random Noise Attenuation", "Spatial median filter across CDPs.", "19"),
    Stage("decon_post", "Deconvolution (post-stack)", "A milder second spiking-deconvolution pass.", "20"),
    Stage("tvf", "Time-Variant Filter", "Depth-zoned band-pass blending.", "21"),
    Stage("migration", "Migration", "Post-stack Kirchhoff time migration, or pre-stack Kirchhoff "
          "time migration (PSTM) on CDP gathers before stack.", "22"),
    Stage("final_scale", "Final Scaling", "Display AGC + re-applied top mute.", "23"),
]

STAGE_INDEX = {s.id: i for i, s in enumerate(PIPELINE_STAGES)}

# The per-gather dict key each stage's *trace* output is stored under
# (mirrors the notebook's own "g[key] = ..." accumulation pattern —
# every stage adds a new key so earlier stages' outputs stay available
# for before/after QC). Pre-stack conditioning (bandpass..decon) runs on
# *shot* gathers; cdp_sort re-groups the result by CDP, after which
# NMO/DMO stages operate on *CDP* gathers. Stages that produce a single
# assembled section rather than per-gather traces (stack onward) are not
# listed here; see `utils/session_state.STAGE_STATE_KEYS` for those.
STAGE_TRACE_KEY = {
    "shot_gathers": "traces",
    "bandpass": "traces_bp",
    "agc": "traces_agc",
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


def stages_from(stage_id: str) -> List[str]:
    """Every stage at or after `stage_id`, in pipeline order — used to
    know what must be invalidated when a stage is re-run."""
    i = STAGE_INDEX[stage_id]
    return [s.id for s in PIPELINE_STAGES[i:]]


def stage_label(stage_id: str) -> str:
    for s in PIPELINE_STAGES:
        if s.id == stage_id:
            return s.label
    return stage_id


@dataclass
class HistoryEntry:
    stage_id: str
    label: str
    params: Dict[str, Any]
    timestamp: str
    notes: str = ""


def log_step(history: List[HistoryEntry], stage_id: str, params: Optional[Dict[str, Any]] = None,
             notes: str = "") -> List[HistoryEntry]:
    """Append a step to the processing history, replacing any prior entry
    for the same stage (re-running a stage supersedes its old log line)."""
    history = [h for h in history if h.stage_id != stage_id]
    history.append(HistoryEntry(
        stage_id=stage_id, label=stage_label(stage_id), params=params or {},
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes=notes,
    ))
    order = {s.id: i for i, s in enumerate(PIPELINE_STAGES)}
    history.sort(key=lambda h: order.get(h.stage_id, 999))
    return history


def history_to_report_lines(history: List[HistoryEntry]) -> List[str]:
    lines = []
    for i, h in enumerate(history, 1):
        lines.append(f"{i}. {h.label}  ({h.timestamp})")
        for k, v in h.params.items():
            lines.append(f"     {k}: {v}")
        if h.notes:
            lines.append(f"     note: {h.notes}")
    return lines
