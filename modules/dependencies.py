"""
modules/dependencies.py — Dependency-aware workflow engine, on data states
===========================================================================
Answers, for any operation at any moment: *which data states would it
read, can it run, and is it advisable?* No Streamlit dependency — pure
Python, so every rule here is unit-tested without a UI.

Five ideas kept apart (they used to be one hard-coded chain):

  RECOMMENDED ORDER      `pipeline.PIPELINE_STAGES` — advice, never enforced.
  GENUINE DEPENDENCIES   `Operation.needs` — an input state the computation
                         cannot run without. Missing  ->  ⛔ blocked.
                         `Operation.exclusions` — data it must not read  ->  ⛔.
  RECOMMENDATIONS        `Operation.advice` — scientifically preferable
                         ordering. Missing / out of order  ->  ⚠ warning.
  DATA STATES            `state_graph.ProcessingGraph` — what data exists and
                         how each state was made: INPUT STATE -> OPERATION +
                         PARAMETERS -> OUTPUT STATE.
  STALENESS              derived here from the graph: a state is stale when
                         an input it was built from has since been superseded.

How an operation finds its input
--------------------------------
A `Need` lists *candidates*. `'@shot'` means "whichever shot-domain state is
current"; a bare stage id (`'vel1'`) means "the latest state that stage
produced". The default is the most preferred candidate that exists; a
person (or a test) can name any other candidate state explicitly
(`inputs={'data': 'S4'}`) — that is how an operation is applied to an
earlier state, or redone with different parameters.

What this file does NOT contain: the declarations. Every need, piece of advice,
exclusion, name and location is declared once, in `modules/registry.py`; this file
*evaluates* those declarations against a graph. (The old names — `STAGE_SPECS`,
`STREAM_BASE`, `SHORT`, `PHRASE`, `STAGE_LOCATION`, `KIND_LABELS`, `STATUS_ROWS` —
remain importable from here as views of the registry.)

Nothing is ever overwritten. Re-running an operation creates a new state;
states built from the old one are flagged *stale* (still valid data, just
not built from the latest inputs) rather than discarded.

Kinds, roots and "same gathers"
-------------------------------
Gather-based states (shot, cdp, nmo, …) store their per-gather arrays in the
container of their *root* (the `shot_gathers` or `cdp_sort` state they descend
from). Two inputs of one operation must come from the same root — a velocity
field picked on one CDP sort cannot flatten the gathers of another. Resolution
therefore picks `data` first and then prefers inputs with the same root; if
only mismatching ones exist the operation is ⛔ with a message saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

from modules import registry
from modules.registry import Advice, Need, Operation        # noqa: F401 — re-exported (declared in the registry)
from modules.registry import stage_label
from modules.state_graph import DataState, ProcessingGraph


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
    kind: str          # missing_input | container_mismatch | invalid_input | fallback_input
                       # | missing_recommended | order_inversion | orphaned_step
                       # | repeated_operation | stale_input | geometry
    message: str
    stages: Tuple[str, ...] = ()   # stage ids this issue is about
    states: Tuple[str, ...] = ()   # state ids this issue is about
    fix: str = ""                  # where / how to resolve it


@dataclass
class StageCheck:
    stage_id: str
    variant: Optional[str]
    issues: List[Issue]
    inputs: Dict[str, Optional[str]]       # role -> resolved input STATE id (None = missing)
    done: bool = False                     # has this operation produced any state?
    options: Dict[str, List[DataState]] = field(default_factory=dict)   # role -> every usable state
    states: Dict[str, Optional[DataState]] = field(default_factory=dict)  # role -> resolved state

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

    def state(self, role: str = "data") -> Optional[DataState]:
        return self.states.get(role)


# ─────────────────────────────────────────────────────────────────────────
# Names and streams — views of the registry (kept under their old names)
# ─────────────────────────────────────────────────────────────────────────
# A "stream" is a kind of data whose current state is the next input.

STREAM_BASE: Dict[str, str] = dict(registry.STREAM_BASE)
STREAM_MEMBERS: Dict[str, Tuple[str, ...]] = dict(registry.STREAM_MEMBERS)
SHOT_OPERATORS: Tuple[str, ...] = STREAM_MEMBERS["shot"]

# Where in the app each stage lives — used to tell a person where to go.
STAGE_LOCATION: Dict[str, str] = {sid: o.location for sid, o in zip(registry.stage_ids(), registry.stages())}

# Short names for the data-state chain ("Raw → Band-pass → F-K").
SHORT: Dict[str, str] = {sid: o.short_name for sid, o in zip(registry.stage_ids(), registry.stages())}

# How a stage is named inside a recommendation sentence (only where it differs from its label).
PHRASE: Dict[str, str] = {sid: o.phrase for sid, o in zip(registry.stage_ids(), registry.stages()) if o.phrase}


def _phrase(stage_id: str) -> str:
    return registry.phrase(stage_id)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def state_name(state: Optional[DataState]) -> str:
    """'Band-pass (S3)' — or 'AGC v2 (S7)' for the second state an operation made."""
    if state is None:
        return "—"
    v = f" v{state.version}" if state.version > 1 else ""
    return f"{SHORT.get(state.op, state.op)}{v} ({state.id})"


# ─────────────────────────────────────────────────────────────────────────
# Workflow = the graph, seen through the question "what would an operation read?"
# ─────────────────────────────────────────────────────────────────────────

class Cand(NamedTuple):
    state: DataState
    index: int               # which candidate token supplied it (0 = most preferred)
    via: Optional[str]       # '@kind' if supplied by a current-state token


class Workflow:
    """A view over a `ProcessingGraph` (which is the single source of truth;
    Streamlit's session state holds the graph, tests hold a plain one)."""

    def __init__(self, graph: Optional[ProcessingGraph] = None):
        self.graph: ProcessingGraph = graph if graph is not None else ProcessingGraph()

    # ---- simple queries -----------------------------------------------------
    def has(self, stage_id: str, variant: Optional[str] = None) -> bool:
        return bool(self.graph.of_op(stage_id, variant))

    def completed(self) -> Set[str]:
        return {s.op for s in self.graph}

    def current(self, kind: str) -> Optional[DataState]:
        return self.graph.current(kind)

    def latest(self, stage_id: str, variant: Optional[str] = None) -> Optional[DataState]:
        return self.graph.latest(stage_id, variant)

    def lineage_ops(self, state_id: Optional[str]) -> List[str]:
        """Operations applied to the data a state holds, oldest first."""
        return [s.op for s in self.graph.lineage(state_id)]

    # ---- candidates ---------------------------------------------------------------
    def candidates(self, need: "Need") -> List[Cand]:
        out: List[Cand] = []
        seen: Set[str] = set()
        for i, tok in enumerate(need.candidates):
            if tok.startswith("@"):
                states, via = self.graph.of_kind(tok[1:]), tok
            else:
                states, via = self.graph.of_op(tok), None
            for s in states:
                if s.id not in seen:
                    seen.add(s.id)
                    out.append(Cand(s, i, via))
        return out

    def dependents(self, state_id: str) -> List[DataState]:
        return self.graph.dependents(state_id)


# ─────────────────────────────────────────────────────────────────────────
# Rules — declared in the registry (`Need`, `Advice`, `Exclusion` on each `Operation`)
# ─────────────────────────────────────────────────────────────────────────

StageSpec = Operation        # a "spec" is the registry's declaration of the operation

STAGE_SPECS: Dict[str, Operation] = dict(registry.BY_KEY)

MIGRATION_VARIANTS = {o.variant: o.variant_label[:1].upper() + o.variant_label[1:]
                      for o in registry.variants("migration")}


def get_spec(stage_id: str, variant: Optional[str] = None) -> Operation:
    return registry.get(stage_id, variant)


# ─────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────

def _resolve_all(wf: Workflow, spec: StageSpec, overrides: Mapping[str, str], lenient: bool = False
                 ) -> Tuple[Dict[str, Optional[Cand]], Dict[str, List[Cand]], List[Issue]]:
    """Pick one state per need. 'data' first; every other role must come from
    the same gather root as it. Returns (chosen, options, issues).

    An explicit `overrides[role]` that is not a usable state is a ⛔ — unless
    `lenient` (a remembered choice from the UI that has since gone stale),
    in which case it is ignored and the default is used."""
    chosen: Dict[str, Optional[Cand]] = {}
    options: Dict[str, List[Cand]] = {}
    issues: List[Issue] = []
    root: Optional[str] = None
    for need in sorted(spec.needs, key=lambda n: n.role != "data"):
        cands = wf.candidates(need)
        usable = cands
        if need.role != "data" and root is not None:
            usable = [c for c in cands if c.state.root in (None, root)]
        options[need.role] = usable
        pick = overrides.get(need.role)
        choice: Optional[Cand] = None
        if pick:
            choice = next((c for c in usable if c.state.id == pick), None)
            if choice is None and lenient:
                pick = None
        if pick and choice is None:
            issues.append(Issue(
                Level.BLOCKED, "invalid_input",
                f"{pick} cannot be used as the {need.what} input here"
                + (" (it is not one of the usable states)." if pick in wf.graph else
                   " (that state no longer exists)."),
                states=(pick,), fix="Choose one of the listed input states."))
            chosen[need.role] = None
            continue
        if choice is None:
            if usable:
                choice = usable[0]
            elif cands:                      # they exist, but on different gathers
                other = cands[0].state
                issues.append(Issue(
                    Level.BLOCKED, "container_mismatch",
                    f"The {need.what} ({state_name(other)}) were made on different gathers "
                    f"({state_name(wf.graph.get(other.root))}) than the data this step reads "
                    f"({state_name(wf.graph.get(root))}), so the two cannot be combined.",
                    stages=tuple(t for t in need.candidates if not t.startswith("@")),
                    states=(other.id,),
                    fix=f"Run {STAGE_LOCATION.get(other.op, stage_label(other.op))} again on the "
                        f"current gathers."))
                chosen[need.role] = None
                continue
            else:
                issues.append(Issue(
                    Level.BLOCKED, "missing_input",
                    f"Required input missing: {need.what}." + (f" {need.why}" if need.why else ""),
                    stages=tuple(t for t in need.candidates if not t.startswith("@")),
                    fix=need.fix))
                chosen[need.role] = None
                continue
        chosen[need.role] = choice
        if need.role == "data":
            root = choice.state.root
    return chosen, options, issues


def resolve_inputs(wf: Workflow, stage_id: str, variant: Optional[str] = None,
                   inputs: Optional[Mapping[str, str]] = None) -> Dict[str, Optional[str]]:
    """role -> state id this stage would read (None where nothing usable exists)."""
    chosen, _, _ = _resolve_all(wf, get_spec(stage_id, variant), inputs or {}, lenient=True)
    return {r: (c.state.id if c else None) for r, c in chosen.items()}


# ─────────────────────────────────────────────────────────────────────────
# The check
# ─────────────────────────────────────────────────────────────────────────

def check_stage(wf: Workflow, stage_id: str, variant: Optional[str] = None,
                inputs: Optional[Mapping[str, str]] = None, lenient: bool = False) -> StageCheck:
    """Evaluate one operation against the current data states.

    BLOCKED  — a `Need` has no usable state (genuine dependency), or the
               inputs come from different gathers.
    WARNING  — runnable, but: a recommended predecessor is not part of the
               data it would read; a recommended successor already is (it
               would run out of order); an input comes from a less-
               preferred fallback; the operation was already applied to
               that data (it would be applied twice); a step's result is
               not in the data this step reads; or the input state is stale.

    `inputs` names explicit states for some roles ({'data': 'S4'}); any role
    not named takes its default. An explicit state that is not usable is a ⛔
    (or, with `lenient`, ignored — for a remembered UI choice gone stale).
    """
    spec = get_spec(stage_id, variant)
    g = wf.graph
    chosen, options, issues = _resolve_all(wf, spec, inputs or {}, lenient)
    states = {r: (c.state if c else None) for r, c in chosen.items()}

    for need in spec.needs:
        c = chosen.get(need.role)
        if c is not None and c.index > 0 and need.fallback:
            issues.append(Issue(Level.WARNING, "fallback_input", need.fallback,
                                stages=(c.state.op,), states=(c.state.id,)))

    # exclusions: data this operation must not read (e.g. trace geometry) --------------
    for exc in spec.exclusions:
        src = states.get(exc.role)
        if src is None:
            continue
        hit = tuple(s_ for s_ in exc.stages if s_ in wf.lineage_ops(src.id))
        if hit:
            issues.append(Issue(Level.BLOCKED, exc.kind, exc.message, stages=hit, states=(src.id,),
                                fix=exc.fix))

    data_c = chosen.get("data")
    data = data_c.state if data_c else None
    if data is not None:
        lineage = g.lineage(data.id)
        in_data = [s.op for s in lineage]
        in_data_set = set(in_data)

        # a) recommended predecessors missing from the data being read ----------------
        for adv in spec.advice:
            present = [s for s in adv.stages if s in in_data_set]
            missing = [s for s in adv.stages if s not in in_data_set]
            trigger = (not present) if adv.mode == "any" else bool(missing)
            if not trigger:
                continue
            names = _join([_phrase(s) for s in (adv.stages if adv.mode == "any" else missing)])
            if adv.message:
                msg = adv.message
            elif adv.mode == "any":
                msg = f"None of {names} has been applied to the current data. {adv.reason}"
            else:
                many = len(missing) > 1
                msg = (f"{_cap(names)} {'are' if many else 'is'} commonly applied "
                       f"before {_phrase(stage_id)}, but {'they have' if many else 'it has'} not been "
                       f"applied to the current data. {adv.reason}")
            issues.append(Issue(Level.WARNING, "missing_recommended", msg.strip(),
                                stages=tuple(adv.stages)))

        # b) running out of order: a stage normally applied AFTER this one is already
        #    part of the data this one will read -------------------------------------------
        for done_op in dict.fromkeys(in_data):
            if done_op == stage_id or done_op not in STAGE_SPECS:
                continue
            for adv in STAGE_SPECS[done_op].advice:
                if stage_id in adv.stages:
                    issues.append(Issue(
                        Level.WARNING, "order_inversion",
                        f"{_cap(_phrase(stage_id))} is commonly applied "
                        f"before {_phrase(done_op)}, but {_phrase(done_op)} is already part of "
                        f"the current data — it will be applied after it.",
                        stages=(stage_id, done_op)))
                    break

        # c) repeated operation: this operation is already in the data's lineage -------------
        again = [s for s in lineage if s.op == stage_id]
        if again:
            prev = again[-1]
            before = g.get(prev.inputs.get("data"))
            redo_hint = (f" To redo it with different settings instead, run it on the state before it "
                         f"({state_name(before)})." if before is not None else "")
            issues.append(Issue(
                Level.WARNING, "repeated_operation",
                f"{_cap(_phrase(stage_id))} has already been applied to the data this step will read "
                f"({state_name(prev)}), so it would be applied twice.{redo_hint}",
                stages=(stage_id,), states=(prev.id,)))

        # d) orphaned steps: results in the same stream that are not part of this data ----------
        if data_c is not None and data_c.via:
            kind = data_c.via[1:]
            ahead = {d.id for d in g.dependents(data.id)}      # undone / redo-able work is not orphaned
            for member in STREAM_MEMBERS.get(kind, ()):
                if member in in_data_set:
                    continue                                    # applied here (maybe another version)
                orphans = [s for s in g.of_op(member) if s.id not in ahead]
                if not orphans:
                    continue
                issues.append(Issue(
                    Level.WARNING, "orphaned_step",
                    f"{stage_label(member)} has been run ({state_name(orphans[0])}), but its result is "
                    f"not part of the data this step will read (that data was produced without it).",
                    stages=(member,), states=(orphans[0].id,),
                    fix="Run the missing step on the current data, choose a different input state, "
                        "or continue as is."))

    # e) stale inputs ------------------------------------------------------------------------------
    stale = stale_info(wf)
    for role, st_ in states.items():
        if st_ is not None and st_.id in stale:
            issues.append(Issue(
                Level.WARNING, "stale_input",
                f"The {role} input ({state_name(st_)}) is out of date: {stale[st_.id][0].detail} "
                f"Re-run the steps that produced it, or choose another input.",
                states=(st_.id,)))

    issues.sort(key=lambda i: -int(i.level))
    return StageCheck(
        stage_id=stage_id, variant=variant, issues=issues,
        inputs={r: (s.id if s else None) for r, s in states.items()},
        done=wf.has(stage_id, variant), states=states,
        options={r: [c.state for c in cs] for r, cs in options.items()})


# ─────────────────────────────────────────────────────────────────────────
# Staleness
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StaleReason:
    kind: str          # behind | replaced | rolled_back | diverged | inherited
    detail: str
    source: str = ""   # id of the input state that is out of date


def stale_info(wf: Workflow) -> Dict[str, List[StaleReason]]:
    """State id -> why it is stale (only stale states appear).

    A state is stale when an input it was built from has since been
    superseded:
      behind       that input has been processed further since;
      replaced     the same operation was re-run (new parameters or input);
      rolled_back  the current data was undone to before that input;
      diverged     a more preferred source now exists (e.g. Final NMO after a
                   stack of NMO Pass 1);
      inherited    an input is itself stale (so this is too, transitively).
    Nothing is discarded: a stale state is valid data, just not built from the
    latest inputs. States are visited in creation order, so inputs are
    classified before the states built on them."""
    g = wf.graph
    out: Dict[str, List[StaleReason]] = {}
    for s in g:
        spec = STAGE_SPECS.get(s.op_key())
        reasons: List[StaleReason] = []
        if spec is not None:
            mine = {d.id for d in g.dependents(s.id)} | {s.id}
            for need in spec.needs:
                if not need.track:
                    continue
                rec = g.get(s.inputs.get(need.role))
                if rec is None:
                    continue
                cur = next((c.state for c in wf.candidates(need) if c.state.id not in mine), None)
                if cur is None or cur.id == rec.id:
                    continue
                me, then, now = state_name(s), state_name(rec), state_name(cur)
                if g.is_ancestor(rec.id, cur.id):
                    chain = [x for x in g.lineage(cur.id) if x.seq > rec.seq]
                    ops = " → ".join(SHORT.get(x.op, x.op) for x in chain) or "further steps"
                    kind = "behind"
                    detail = (f"{me} was built from {then}, but that data has since been processed "
                              f"further ({ops}); the current {need.what} is {now}.")
                elif g.is_ancestor(cur.id, rec.id):
                    kind = "rolled_back"
                    detail = (f"{me} was built from {then}, but the current {need.what} was rolled "
                              f"back to {now}.")
                elif cur.op == rec.op and cur.variant == rec.variant:
                    kind = "replaced"
                    detail = (f"{me} was built from {then}, which has since been re-run; the current "
                              f"{need.what} is {now}.")
                else:
                    kind = "diverged"
                    detail = (f"{me} was built from {then}, but the current {need.what} now comes "
                              f"from {now}.")
                reasons.append(StaleReason(kind, detail + f" Re-run {_phrase(s.op)} to include it.",
                                           source=rec.id))
        for role, src in s.inputs.items():
            if src in out and not any(r.source == src for r in reasons):
                reasons.append(StaleReason(
                    "inherited", f"{state_name(s)} was built from {state_name(g[src])}, which is out "
                                 f"of date.", source=src))
        if reasons:
            out[s.id] = reasons
    return out


def stale_notes(wf: Workflow) -> Dict[str, str]:
    """state id -> one-sentence note for display (the first reason)."""
    return {sid: rs[0].detail for sid, rs in stale_info(wf).items()}


# ─────────────────────────────────────────────────────────────────────────
# Whole-workflow views
# ─────────────────────────────────────────────────────────────────────────

# (stage id, variant, display label) for every row of the status panel: one per registered
# operation, in the recommended order (migration appears once per implementation).
STATUS_ROWS: List[Tuple[str, Optional[str], str]] = [
    (o.id, o.variant or None, o.full_name) for o in registry.OPERATIONS]


def check_all(wf: Workflow) -> List[Tuple[str, Optional[str], str, StageCheck]]:
    return [(sid, v, label, check_stage(wf, sid, v)) for sid, v, label in STATUS_ROWS]


def describe_lineage(wf: Workflow, state_id: Optional[str], start_at: Optional[str] = None,
                     skip: Sequence[str] = ("upload",)) -> str:
    """'Raw → Band-pass → F-K' for the data a state holds. `start_at` trims the
    chain to begin at that stage (e.g. the stream's base)."""
    if state_id is None:
        return "—"
    chain = wf.graph.lineage(state_id)
    if start_at:
        idx = next((i for i, s in enumerate(chain) if s.op == start_at), None)
        if idx is not None:
            chain = chain[idx:]
    return " → ".join(SHORT.get(s.op, s.op) for s in chain if s.op not in skip) or "—"


def describe_stream(wf: Workflow, kind: str) -> str:
    head = wf.current(kind)
    if head is None:
        return "— (not available yet)"
    return describe_lineage(wf, head.id, start_at=STREAM_BASE.get(kind))


# What each kind of data is called in the "current processing state" summary.
KIND_LABELS: Dict[str, str] = {t.id: t.label for t in registry.DATA_TYPES if t.summarised}


def current_processing_state(wf: Workflow) -> List[Dict[str, object]]:
    """What the data IS right now — one row per kind of data that exists, naming the
    state that is current and what has been applied to produce it.

    This is deliberately not the history (what was done, in what order — that is the
    action log) and not the recommended workflow (what is usually done). It is derived
    from the states and their current pointers only.

    Each row: kind, data (display name), state (id), name ("Band-pass (S3)"),
    applied ("Raw → AGC → Band-pass"), n_states (how many states of this kind exist),
    stale (why it is out of date, or ""), released (arrays are not in memory)."""
    g = wf.graph
    stale = stale_info(wf)
    rows: List[Dict[str, object]] = []
    for kind, label in KIND_LABELS.items():
        cur = g.current(kind)
        if cur is None:
            continue
        rows.append({
            "kind": kind, "data": label, "state": cur.id, "name": state_name(cur),
            "applied": describe_lineage(wf, cur.id), "n_states": len(g.of_kind(kind)),
            "stale": stale[cur.id][0].detail if cur.id in stale else "",
            "released": not cur.materialized})
    return rows


def advice_pairs() -> List[Tuple[str, str]]:
    """(earlier, later) pairs implied by every `Advice` — used by tests to
    assert the recommendations agree with the recommended order."""
    return registry.advice_pairs()
