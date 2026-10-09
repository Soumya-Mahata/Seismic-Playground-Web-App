"""
modules/dependencies.py — Dependency-aware workflow engine
=============================================================
Answers, for any stage at any moment: *can it run, and is it advisable?*
No Streamlit dependency — pure Python, so every rule here is unit-tested
without a UI (see tests/test_dependencies.py).

The model keeps four ideas apart (they used to be one hard-coded chain):

  RECOMMENDED ORDER      `pipeline.PIPELINE_STAGES` — advice, never enforced.
  GENUINE DEPENDENCIES   `StageSpec.needs`  — an input the computation
                         cannot run without. Missing  ->  ⛔ blocked.
  RECOMMENDATIONS        `StageSpec.advice` — scientifically preferable
                         ordering. Missing / out of order  ->  ⚠ warning,
                         never a block.
  CURRENT DATA STATE     `WorkflowState` — derived from the processing
                         history: which products exist and what has been
                         applied to the data a stage would read next.
  PROCESSING HISTORY     the same history, in execution order.

Core idea — provenance instead of position
------------------------------------------
Every executed stage records which upstream stage outputs it *consumed*
(`HistoryEntry.consumed`: role -> stage id). From those edges:

  * a stage's input is "the head of its data stream" — the most recent
    valid output of the stages that transform that data — not "the stage
    before me in the list". So AGC can read raw traces, or band-passed
    ones, whichever is the current state of the shot gathers;
  * "has X been applied to this data?" is an ancestry query, not a
    check on a global done-flag;
  * re-running a stage invalidates exactly its *dependents* (the stages
    that consumed its output, transitively), no more.

Streams
-------
  shot     shot_gathers -> {despike, bandpass, agc, trace_edit, mute,
                            balance, fk, mult_supp, decon}
  cdp      cdp_sort     -> {statics, inv_nmo, data_reg}   (pre-NMO gathers)
  section  stack        -> {rna, decon_post, tvf}         (stacked section)

NMO (nmo1, nmo_final), DMO, velocity analysis, stacking, migration and
final scaling *consume* a stream's head but are not members of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from modules.pipeline import HistoryEntry, PIPELINE_STAGES, log_step, stage_label


# ─────────────────────────────────────────────────────────────────────────
# Status levels
# ─────────────────────────────────────────────────────────────────────────

class Level(IntEnum):
    AVAILABLE = 0   # ✓ can run, nothing to flag
    WARNING = 1     # ⚠ can run; a recommended prerequisite was not performed
    BLOCKED = 2     # ⛔ cannot run; a required input is missing

    @property
    def symbol(self) -> str:
        return {0: "✓", 1: "⚠", 2: "⛔"}[int(self)]

    @property
    def label(self) -> str:
        return {0: "Available",
                1: "Recommended prerequisite not performed",
                2: "Required input missing"}[int(self)]

    @property
    def badge(self) -> str:
        return f"{self.symbol} {self.label}"


@dataclass(frozen=True)
class Issue:
    level: Level
    kind: str          # missing_input | fallback_input | missing_recommended | order_inversion
                       # | orphaned_step | geometry
    message: str
    stages: Tuple[str, ...] = ()   # stage ids this issue is about
    fix: str = ""                  # where / how to resolve it


@dataclass
class StageCheck:
    stage_id: str
    variant: Optional[str]
    issues: List[Issue]
    inputs: Dict[str, Optional[str]]       # role -> resolved upstream stage id (None = missing)
    done: bool = False

    @property
    def blockers(self) -> List[Issue]:
        return [i for i in self.issues if i.level == Level.BLOCKED]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == Level.WARNING]

    @property
    def level(self) -> Level:
        return max((i.level for i in self.issues), default=Level.AVAILABLE)

    @property
    def can_run(self) -> bool:
        return self.level != Level.BLOCKED

    @property
    def symbol(self) -> str:
        return self.level.symbol

    def blocker_text(self) -> str:
        return " ".join(i.message + (f" → {i.fix}" if i.fix else "") for i in self.blockers)


# ─────────────────────────────────────────────────────────────────────────
# Streams
# ─────────────────────────────────────────────────────────────────────────

STREAM_BASE: Dict[str, str] = {"shot": "shot_gathers", "cdp": "cdp_sort", "section": "stack"}
STREAM_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "shot": ("despike", "bandpass", "agc", "trace_edit", "mute", "balance", "fk", "mult_supp", "decon"),
    "cdp": ("statics", "inv_nmo", "data_reg"),
    "section": ("rna", "decon_post", "tvf"),
}
SHOT_OPERATORS: Tuple[str, ...] = STREAM_MEMBERS["shot"]

# Where in the app each stage lives — used to tell a person where to go.
STAGE_LOCATION: Dict[str, str] = {
    "upload": "Data Management → ① Load data",
    "shot_gathers": "Data Management → ⑥ Build shot gathers",
    "bandpass": "Pre-Stack Conditioning → Band-Pass",
    "agc": "Pre-Stack Conditioning → Amplitude Recovery",
    "despike": "Pre-Stack Conditioning → De-spike",
    "trace_edit": "Pre-Stack Conditioning → Trace Edit/Kill",
    "mute": "Pre-Stack Conditioning → Mute",
    "balance": "Pre-Stack Conditioning → Trace Balancing",
    "fk": "Pre-Stack Conditioning → F-K Filter",
    "mult_supp": "Pre-Stack Conditioning → Multiple Suppression",
    "decon": "Pre-Stack Conditioning → Deconvolution",
    "cdp_sort": "CDP Sort & Binning",
    "vel1": "Velocity, NMO & DMO → Velocity Pass 1",
    "nmo1": "Velocity, NMO & DMO → NMO Pass 1",
    "statics": "Velocity, NMO & DMO → Residual Statics",
    "dmo": "Velocity, NMO & DMO → DMO Correction",
    "inv_nmo": "Velocity, NMO & DMO → Inverse NMO",
    "data_reg": "Velocity, NMO & DMO → Data Regularisation",
    "vel2": "Velocity, NMO & DMO → RMS Velocity Analysis (Pass 2)",
    "nmo_final": "Velocity, NMO & DMO → Final NMO",
    "stack": "Stacking",
    "rna": "Post-Stack Processing → Random Noise Attenuation",
    "decon_post": "Post-Stack Processing → Deconvolution",
    "tvf": "Post-Stack Processing → Time-Variant Filter",
    "migration": "Migration",
    "final_scale": "Migration → Final scaling",
}

# Short names for the data-state chain ("Raw → Band-pass → F-K").
SHORT: Dict[str, str] = {
    "upload": "Load", "shot_gathers": "Raw", "bandpass": "Band-pass", "agc": "AGC",
    "despike": "De-spike", "trace_edit": "Trace kill", "mute": "Mute", "balance": "Balance",
    "fk": "F-K", "mult_supp": "Multiple suppression", "decon": "Decon", "cdp_sort": "CDP sort",
    "vel1": "Velocity 1", "nmo1": "NMO 1", "statics": "Statics", "dmo": "DMO",
    "inv_nmo": "Inverse NMO", "data_reg": "Regularised", "vel2": "Velocity 2",
    "nmo_final": "Final NMO", "stack": "Stack", "rna": "RNA", "decon_post": "Post-stack decon",
    "tvf": "TVF", "migration": "Migration", "final_scale": "Final scaling",
}

# How a stage is named inside a recommendation sentence.
PHRASE: Dict[str, str] = {
    "bandpass": "band-pass filtering", "agc": "amplitude recovery (AGC)", "despike": "de-spiking",
    "trace_edit": "trace edit/kill", "mute": "muting", "balance": "trace balancing",
    "fk": "F-K filtering", "mult_supp": "multiple suppression", "decon": "deconvolution",
    "statics": "the residual-statics correction", "dmo": "DMO", "inv_nmo": "inverse NMO",
    "data_reg": "data regularisation", "vel2": "pass-2 velocity analysis",
    "nmo_final": "final NMO", "rna": "random noise attenuation",
    "decon_post": "post-stack deconvolution", "tvf": "time-variant filtering",
    "migration": "migration", "cdp_sort": "CDP sort",
}


def _phrase(stage_id: str) -> str:
    return PHRASE.get(stage_id, stage_label(stage_id))


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ─────────────────────────────────────────────────────────────────────────
# Workflow state (provenance)
# ─────────────────────────────────────────────────────────────────────────

class WorkflowState:
    """The current data state, derived from the processing history.

    Wraps (and mutates in place) a list of `HistoryEntry`. That list is
    the single source of truth: a stage is "available as an input" exactly
    when it has a valid entry in it. Streamlit's session state simply
    holds the list; tests hold a plain list."""

    def __init__(self, history: Optional[List[HistoryEntry]] = None):
        self.history: List[HistoryEntry] = history if history is not None else []

    # ---- simple queries ---------------------------------------------------
    def entry(self, stage_id: str) -> Optional[HistoryEntry]:
        for h in self.history:
            if h.stage_id == stage_id:
                return h
        return None

    def has(self, stage_id: str) -> bool:
        return self.entry(stage_id) is not None

    def completed(self) -> Set[str]:
        return {h.stage_id for h in self.history}

    # ---- stream heads ------------------------------------------------------
    def stream_head(self, stream: str, exclude: Iterable[str] = ()) -> Optional[str]:
        """Most recently produced valid output of `stream` (its base counts
        as the oldest member). `exclude` hides stages (used when re-running
        a stage, so it never reads its own output or a descendant's)."""
        exclude = set(exclude)
        ids = (STREAM_BASE[stream],) + STREAM_MEMBERS[stream]
        best: Optional[HistoryEntry] = None
        for h in self.history:
            if h.stage_id in ids and h.stage_id not in exclude:
                if best is None or h.seq > best.seq:
                    best = h
        return best.stage_id if best else None

    # ---- provenance --------------------------------------------------------
    def data_ancestry(self, stage_id: Optional[str]) -> List[str]:
        """Stages applied to the data that `stage_id` holds, oldest first,
        ending with `stage_id` itself (follows the 'data' input edges)."""
        out: List[str] = []
        seen: Set[str] = set()
        cur = stage_id
        while cur and cur not in seen:
            seen.add(cur)
            out.append(cur)
            e = self.entry(cur)
            cur = e.consumed.get("data") if e else None
        return list(reversed(out))

    def dependents(self, stage_id: str) -> List[str]:
        """Every stage that consumed `stage_id`'s output, directly or
        through other stages (any input role), in execution order."""
        frontier = {stage_id}
        found: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for h in self.history:
                if h.stage_id in found or h.stage_id == stage_id:
                    continue
                if any(src in frontier for src in h.consumed.values()):
                    found.add(h.stage_id)
                    frontier.add(h.stage_id)
                    changed = True
        return [h.stage_id for h in sorted(self.history, key=lambda e: e.seq) if h.stage_id in found]

    # ---- mutation ----------------------------------------------------------
    def invalidate(self, stage_id: str, include_self: bool = True) -> List[str]:
        """Drop `stage_id` (optionally) and everything built on it.
        Returns the stage ids removed, so the caller can free their data."""
        removed = self.dependents(stage_id)
        if include_self and self.has(stage_id):
            removed = [stage_id] + removed
        self.history[:] = [h for h in self.history if h.stage_id not in set(removed)]
        return removed

    def record(self, stage_id: str, *, consumed: Optional[Dict[str, str]] = None,
               params: Optional[dict] = None, notes: str = "", input_key: str = "",
               output_key: str = "", offsets_key: str = "", variant: str = "") -> List[str]:
        """Record a completed run. Invalidates the stage's dependents (they
        were built from its previous output) and supersedes its own entry.
        Returns the stage ids that were invalidated."""
        removed = self.invalidate(stage_id, include_self=False)
        self.history[:] = log_step(self.history, stage_id, params, notes, consumed=consumed,
                                   input_key=input_key, output_key=output_key,
                                   offsets_key=offsets_key, variant=variant)
        return removed


# ─────────────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Need:
    """One genuine input of a stage.

    `candidates` are tried in order of preference. A token starting with
    '@' means "the head of that stream" ('@shot', '@cdp', '@section');
    anything else is a stage id whose output is the input. If a candidate
    other than the first supplies the input and `fallback` is set, the
    stage is still runnable but gets a ⚠ with that text."""
    role: str
    candidates: Tuple[str, ...]
    what: str
    why: str = ""
    fix: str = ""
    fallback: str = ""
    track: bool = False   # report "built from earlier data" for a finished stage


@dataclass(frozen=True)
class Advice:
    """A scientifically preferable ordering — never blocks.

    mode "all": each listed stage should be in the ancestry of the data
    this stage reads (one ⚠ naming whichever are missing).
    mode "any": at least one should be."""
    stages: Tuple[str, ...]
    reason: str
    mode: str = "all"
    message: str = ""     # overrides the generated sentence


@dataclass(frozen=True)
class StageSpec:
    needs: Tuple[Need, ...] = ()
    advice: Tuple[Advice, ...] = ()
    checks: Tuple[Callable[["WorkflowState", Dict[str, Optional[str]]], List[Issue]], ...] = ()


def _statics_geometry(state: WorkflowState, inputs: Dict[str, Optional[str]]) -> List[Issue]:
    data = inputs.get("data")
    if data and "data_reg" in state.data_ancestry(data):
        return [Issue(
            Level.BLOCKED, "geometry",
            "Residual statics are surface-consistent: they need each trace's own shot and receiver. "
            "Regularised gathers hold interpolated traces on a synthetic offset grid, so that "
            "identity is gone.",
            stages=("data_reg",),
            fix="Apply residual statics before Data Regularisation (or discard Data Regularisation "
                "and what was built on it).")]
    return []


_SHOT = Need("data", ("@shot",), "shot gathers", fix="Build shot gathers (Data Management)",
             why="Pre-stack conditioning runs on shot gathers.")
_CDP_HEAD = Need("data", ("@cdp",), "CDP gathers", fix="Run CDP Sort & Binning")
_VEL_BEST_FALLBACK = ("Pass-2 velocity analysis has not been performed — the pass-1 (approximate) "
                      "velocities will be used.")

STAGE_SPECS: Dict[str, StageSpec] = {
    "upload": StageSpec(),
    "shot_gathers": StageSpec(needs=(
        Need("data", ("upload",), "a loaded survey", fix="Load a SEG-Y file or generate synthetic "
             "data (Data Management)"),)),

    # ---- shot-domain conditioning: each only needs shot gathers -------------
    "bandpass": StageSpec(needs=(_SHOT,)),
    "agc": StageSpec(needs=(_SHOT,), advice=(
        Advice(("bandpass",), "AGC amplifies everything in the trace, including out-of-band noise, so "
                              "the gain is driven by the useful reflection band only if the data "
                              "were filtered first."),)),
    "despike": StageSpec(needs=(_SHOT,)),
    "trace_edit": StageSpec(needs=(_SHOT,)),
    "mute": StageSpec(needs=(_SHOT,)),
    "balance": StageSpec(needs=(_SHOT,)),
    "fk": StageSpec(needs=(_SHOT,)),
    "mult_supp": StageSpec(needs=(_SHOT,), advice=(
        Advice(("bandpass", "agc"), "Predictive deconvolution assumes a stationary, noise-limited "
                                    "trace; band-pass filtering and amplitude recovery are normally "
                                    "applied first."),)),
    "decon": StageSpec(needs=(_SHOT,), advice=(
        Advice(("bandpass", "agc"), "Spiking deconvolution assumes a stationary, noise-limited trace: a "
                                    "wide band-pass and amplitude recovery are normally applied "
                                    "first (4 Deconvolution, 'Preconditioning')."),
        Advice(("mute", "fk"), "Coherent noise (ground roll, direct arrival) is normally removed "
                               "before deconvolution so it is not treated as part of the wavelet.",
               mode="any"),)),

    # ---- CDP sort: only a survey is genuinely required ----------------------
    "cdp_sort": StageSpec(
        needs=(Need("data", ("@shot", "upload"), "traces to sort (a loaded survey)",
                    fix="Load data (Data Management)", track=True),),
        advice=(Advice(SHOT_OPERATORS, "", mode="any",
                       message="No pre-stack conditioning has been applied, so the CDP gathers will be "
                               "built from the raw, unconditioned traces. (Fine for data that were "
                               "already processed.)"),)),

    # ---- velocity / NMO / statics / DMO / regularisation --------------------
    "vel1": StageSpec(needs=(Need("data", ("cdp_sort",), "CDP gathers", fix="Run CDP Sort & Binning",
                                  why="Velocity analysis scans offset-sorted CDP gathers."),)),
    "nmo1": StageSpec(needs=(
        Need("data", ("cdp_sort",), "CDP gathers", fix="Run CDP Sort & Binning"),
        Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=STAGE_LOCATION["vel1"],
             why="NMO correction is defined by a velocity function."))),
    "statics": StageSpec(
        needs=(_CDP_HEAD,
               Need("nmo", ("nmo1",), "NMO-flattened gathers (NMO Pass 1)", fix=STAGE_LOCATION["nmo1"],
                    why="Residual statics are estimated by cross-correlating NMO-flattened traces "
                        "with a pilot trace.")),
        checks=(_statics_geometry,)),
    "dmo": StageSpec(
        needs=(_CDP_HEAD,
               Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=STAGE_LOCATION["vel1"],
                    why="DMO acts on NMO-corrected data, so it needs the velocities that flatten the "
                        "gathers first.")),
        advice=(Advice(("statics",), "Residual statics are normally applied before DMO so the DMO "
                                     "stretch does not smear uncorrected shifts."),)),
    "inv_nmo": StageSpec(needs=(
        Need("data", ("dmo",), "DMO-corrected gathers", fix=STAGE_LOCATION["dmo"],
             why="Inverse NMO undoes pass-1 NMO on the DMO output."),
        Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=STAGE_LOCATION["vel1"]))),
    "data_reg": StageSpec(needs=(_CDP_HEAD,)),
    "vel2": StageSpec(needs=(_CDP_HEAD,), advice=(
        Advice(("statics", "inv_nmo", "data_reg"), "", mode="any",
               message="No refinement (residual statics, DMO or regularisation) has been applied since "
                       "pass 1, so pass-2 velocity analysis will re-analyse essentially the same "
                       "data."),)),
    "nmo_final": StageSpec(needs=(
        Need("data", ("@cdp",), "CDP gathers", fix="Run CDP Sort & Binning", track=True),
        Need("velocity", ("vel2", "vel1"), "stacking velocities", fix=STAGE_LOCATION["vel1"],
             why="NMO correction is defined by a velocity function.", fallback=_VEL_BEST_FALLBACK,
             track=True))),

    # ---- stack and post-stack ------------------------------------------------
    "stack": StageSpec(needs=(
        Need("data", ("nmo_final", "nmo1"), "NMO-corrected CDP gathers",
             fix="Apply NMO (Velocity, NMO & DMO → NMO Pass 1 or Final NMO)",
             why="Stacking sums traces at each CDP; without moveout correction reflections stack "
                 "out of phase.",
             fallback="Final NMO has not been applied — the pass-1 NMO output will be stacked "
                      "(approximate moveout correction, and none of the later refinements).",
             track=True),)),
    "rna": StageSpec(needs=(Need("data", ("@section",), "a stacked section", fix="Run Stacking"),)),
    "decon_post": StageSpec(
        needs=(Need("data", ("@section",), "a stacked section", fix="Run Stacking"),),
        advice=(Advice(("rna",), "Noise attenuation is normally applied first so deconvolution does not "
                                 "sharpen random noise."),)),
    "tvf": StageSpec(
        needs=(Need("data", ("@section",), "a stacked section", fix="Run Stacking"),),
        advice=(Advice(("decon_post",), "The time-variant filter normally follows deconvolution, "
                                        "removing the high-frequency noise it boosts."),)),

    # ---- migration (two implementations, two sets of inputs) ----------------
    "migration:poststack": StageSpec(
        needs=(Need("data", ("@section",), "a stacked section", fix="Run Stacking",
                    why="Post-stack migration operates on a stacked (≈ zero-offset) section.",
                    track=True),
               Need("velocity", ("vel2", "vel1"), "a stacking-velocity field (the migration velocity)",
                    fix=STAGE_LOCATION["vel1"], fallback=_VEL_BEST_FALLBACK.replace(
                        "will be used", "will be used as the migration velocity"), track=True)),
        advice=(Advice(("rna", "decon_post", "tvf"), "Post-stack conditioning is normally applied "
                       "before migration so noise is not migrated into the image."),)),
    "migration:pstm": StageSpec(
        needs=(Need("data", ("@cdp",), "pre-stack CDP gathers with their offsets preserved",
                    fix="Run CDP Sort & Binning", track=True,
                    why="PSTM sums pre-stack traces using their true offsets, so it cannot use "
                        "stacked or NMO-corrected data."),
               Need("velocity", ("vel2", "vel1"), "a stacking-velocity field (the migration velocity)",
                    fix=STAGE_LOCATION["vel1"], fallback=_VEL_BEST_FALLBACK.replace(
                        "will be used", "will be used as the migration velocity"), track=True)),
        advice=(Advice(("data_reg",), "Regularised offsets give the migration a consistent offset "
                                      "sampling at every CDP."),)),
    "final_scale": StageSpec(needs=(
        Need("data", ("migration", "@section"), "a section to scale", fix="Run Stacking",
             fallback="Migration has not been applied — final scaling will be applied to the "
                      "un-migrated section.", track=True),)),
}

MIGRATION_VARIANTS = {"poststack": "Post-stack", "pstm": "Pre-stack (PSTM)"}


def get_spec(stage_id: str, variant: Optional[str] = None) -> StageSpec:
    key = f"{stage_id}:{variant}" if variant else stage_id
    if key not in STAGE_SPECS:
        if stage_id == "migration":
            raise ValueError("Migration needs a variant: 'poststack' or 'pstm'.")
        raise KeyError(f"No dependency spec for stage '{stage_id}'")
    return STAGE_SPECS[key]


# ─────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _Resolved:
    stage: Optional[str]
    index: int                 # which candidate supplied it (0 = most preferred), -1 = none
    via_stream: Optional[str]  # stream name if supplied by a '@stream' head


def _resolve(state: WorkflowState, need: Need, exclude: Set[str]) -> _Resolved:
    for i, tok in enumerate(need.candidates):
        if tok.startswith("@"):
            head = state.stream_head(tok[1:], exclude)
            if head:
                return _Resolved(head, i, tok[1:])
        elif state.has(tok) and tok not in exclude:
            return _Resolved(tok, i, None)
    return _Resolved(None, -1, None)


def _exclusions(state: WorkflowState, stage_id: str) -> Set[str]:
    """A stage never reads its own output or anything built on it."""
    ex = {stage_id}
    if state.has(stage_id):
        ex |= set(state.dependents(stage_id))
    return ex


def resolve_inputs(state: WorkflowState, stage_id: str, variant: Optional[str] = None
                   ) -> Dict[str, Optional[str]]:
    spec = get_spec(stage_id, variant)
    ex = _exclusions(state, stage_id)
    return {n.role: _resolve(state, n, ex).stage for n in spec.needs}


# ─────────────────────────────────────────────────────────────────────────
# The check
# ─────────────────────────────────────────────────────────────────────────

def check_stage(state: WorkflowState, stage_id: str, variant: Optional[str] = None) -> StageCheck:
    """Evaluate one stage against the current data state.

    BLOCKED  — a `Need` has no valid supplier (genuine dependency).
    WARNING  — runnable, but: a recommended predecessor is not part of
               the data it would read; a recommended successor is already
               part of that data (the step would run out of order); an
               input comes from a less-preferred fallback; or a step has
               been run whose result is not in the data this step reads.
    """
    spec = get_spec(stage_id, variant)
    ex = _exclusions(state, stage_id)
    issues: List[Issue] = []
    inputs: Dict[str, Optional[str]] = {}
    data_res: Optional[_Resolved] = None

    # 1) genuine dependencies -----------------------------------------------
    for need in spec.needs:
        res = _resolve(state, need, ex)
        inputs[need.role] = res.stage
        if need.role == "data":
            data_res = res
        if res.stage is None:
            msg = f"Required input missing: {need.what}."
            if need.why:
                msg += f" {need.why}"
            issues.append(Issue(Level.BLOCKED, "missing_input", msg,
                                stages=tuple(t for t in need.candidates if not t.startswith("@")),
                                fix=need.fix))
        elif res.index > 0 and need.fallback:
            issues.append(Issue(Level.WARNING, "fallback_input", need.fallback, stages=(res.stage,)))

    # 2) extra structural checks (e.g. trace geometry) ------------------------
    for chk in spec.checks:
        issues.extend(chk(state, inputs))

    # 3) recommendations — only meaningful once the data input is known -------
    data_stage = inputs.get("data")
    if data_stage is not None:
        ancestry = state.data_ancestry(data_stage)
        in_data = set(ancestry)

        # 3a) recommended predecessors missing from the data being read
        for adv in spec.advice:
            present = [s for s in adv.stages if s in in_data]
            missing = [s for s in adv.stages if s not in in_data]
            if adv.mode == "any":
                trigger = not present
            else:
                trigger = bool(missing)
            if not trigger:
                continue
            names = _join([_phrase(s) for s in (adv.stages if adv.mode == "any" else missing)])
            if adv.message:
                msg = adv.message
            elif adv.mode == "any":
                msg = f"None of {names} has been applied to the current data. {adv.reason}"
            else:
                many = len(missing) > 1
                msg = (f"{names[0].upper() + names[1:]} {'are' if many else 'is'} commonly applied "
                       f"before {_phrase(stage_id)}, but {'they have' if many else 'it has'} not been "
                       f"applied to the current data. {adv.reason}")
            issues.append(Issue(Level.WARNING, "missing_recommended", msg.strip(),
                                stages=tuple(adv.stages)))

        # 3b) running out of order: a stage that is normally applied AFTER
        #     this one is already part of the data this one will read
        for done_stage in ancestry:
            if done_stage == stage_id or done_stage not in STAGE_SPECS:
                continue
            for adv in STAGE_SPECS[done_stage].advice:
                if stage_id in adv.stages:
                    issues.append(Issue(
                        Level.WARNING, "order_inversion",
                        f"{_phrase(stage_id)[0].upper() + _phrase(stage_id)[1:]} is commonly applied "
                        f"before {_phrase(done_stage)}, but {_phrase(done_stage)} is already part of "
                        f"the current data — it will be applied after it.",
                        stages=(stage_id, done_stage)))
                    break

        # 3c) orphaned steps: valid steps in the same stream whose result is
        #     not in the data this step will read
        if data_res is not None and data_res.via_stream:
            stream = data_res.via_stream
            for member in STREAM_MEMBERS[stream]:
                if member in in_data or member in ex or not state.has(member):
                    continue
                issues.append(Issue(
                    Level.WARNING, "orphaned_step",
                    f"{stage_label(member)} has been run, but its result is not part of the data this "
                    f"step will read (that data was produced before it ran).",
                    stages=(member,),
                    fix="Re-run the step that produced the data this reads, or continue as is."))

    issues.sort(key=lambda i: -int(i.level))
    return StageCheck(stage_id=stage_id, variant=variant, issues=issues, inputs=inputs,
                      done=state.has(stage_id))


# ─────────────────────────────────────────────────────────────────────────
# Whole-workflow views
# ─────────────────────────────────────────────────────────────────────────

# (stage id, variant, display label) for every row of the status panel.
STATUS_ROWS: List[Tuple[str, Optional[str], str]] = []
for _s in PIPELINE_STAGES:
    if _s.id == "migration":
        STATUS_ROWS.append(("migration", "poststack", "Migration — post-stack"))
        STATUS_ROWS.append(("migration", "pstm", "Migration — pre-stack (PSTM)"))
    else:
        STATUS_ROWS.append((_s.id, None, _s.label))


def check_all(state: WorkflowState) -> List[Tuple[str, Optional[str], str, StageCheck]]:
    return [(sid, v, label, check_stage(state, sid, v)) for sid, v, label in STATUS_ROWS]


def stale_notes(state: WorkflowState) -> Dict[str, str]:
    """For finished stages whose input has since been superseded — e.g. CDP
    gathers sorted before a later shot-domain step, a stack built from
    pass-1 NMO after Final NMO was run. Nothing is invalidated (the data
    are still valid, just not the latest); this is a note, not an error."""
    notes: Dict[str, str] = {}
    for h in state.history:
        key = f"{h.stage_id}:{h.variant}" if h.variant else h.stage_id
        spec = STAGE_SPECS.get(key)
        if spec is None:
            continue
        ex = _exclusions(state, h.stage_id)
        for need in spec.needs:
            if not need.track:
                continue
            recorded = h.consumed.get(need.role)
            current = _resolve(state, need, ex).stage
            if current and recorded and current != recorded:
                notes[h.stage_id] = (
                    f"{h.label} was built from {stage_label(recorded)}, but the current "
                    f"{need.what} now comes from {stage_label(current)}. Re-run {h.label} to "
                    f"include it.")
                break
    return notes


def describe_ancestry(state: WorkflowState, stage_id: Optional[str],
                      start_after: Sequence[str] = ("upload",)) -> str:
    """'Raw → Band-pass → F-K' for the data a stage holds."""
    if stage_id is None:
        return "—"
    chain = [s for s in state.data_ancestry(stage_id) if s not in start_after]
    return " → ".join(SHORT.get(s, s) for s in chain) or "—"


def describe_stream(state: WorkflowState, stream: str) -> str:
    head = state.stream_head(stream)
    if head is None:
        return "— (not available yet)"
    base = STREAM_BASE[stream]
    chain = state.data_ancestry(head)
    if base in chain:
        chain = chain[chain.index(base):]
    return " → ".join(SHORT.get(s, s) for s in chain)


def advice_pairs() -> List[Tuple[str, str]]:
    """(earlier, later) pairs implied by every `Advice` — used by tests to
    assert the recommendations agree with the recommended order."""
    pairs = []
    for key, spec in STAGE_SPECS.items():
        later = key.split(":")[0]
        for adv in spec.advice:
            for earlier in adv.stages:
                pairs.append((earlier, later))
    return pairs
