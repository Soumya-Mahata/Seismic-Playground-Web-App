"""
modules/stage_ops.py — Executors on data states (pure; no Streamlit)
=====================================================================
Every processing operation is one call of the same generic `execute()`:

    INPUT STATE(S) ──▶ OPERATION + PARAMETERS ──▶ OUTPUT STATE
         S2              Band-pass(8–60 Hz)            S3

`execute` (1) asks the dependency engine which input states the operation
would read and whether it may run (⛔ raises `DependencyError`; ⚠ never
raises), (2) fingerprints the operation + parameters + inputs and *re-uses*
an existing state if the identical computation was already done, (3) runs the
unchanged processing maths on the input payloads, and (4) records the result
as a NEW state in the graph. Nothing is ever overwritten, so operations can be
repeated, applied in any order, applied to an earlier state, undone and redone.

Where the arrays live
---------------------
  store["graph"]        the `ProcessingGraph` (records only — no arrays)
  store["containers"]   {root state id: {gather id: gather dict}}. One container per
                        gather set (a `shot_gathers` or `cdp_sort` state is the *root*
                        and owns the native arrays: traces, offsets, ids …). A derived
                        state adds its own arrays to the same gather dicts under keys
                        suffixed `@<state id>` ("traces@S3"), so a new state costs one
                        new array and never a copy of its input.
  store["blobs"]        {state id: {name: array/object}} for states that are not gather
                        sets: stacked sections, migrated images, velocity fields …
  store[<legacy keys>]  `stack_matrix`, `Vfield_1`, `gathers` … are *reference aliases* to
                        the payload of the current/latest state, refreshed by
                        `sync_aliases` (no copies). Read them; never write them.

Memory: `release_payload` drops a state's arrays (the record stays);
`materialize` re-computes them from the recorded operation and parameters —
which is also what makes a session reproducible (`replay_recipe`).

`store` is any mutable mapping with the keys of
`pipeline.PROCESSING_STORE_DEFAULTS`: `st.session_state` in the app, a plain
dict (see `new_store`) everywhere else, which is how the tests run the real
pipeline without a UI.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import config as cfg
from modules import filtering, migration, nmo, preprocessing, regularization, segy_io, stacking
from modules import statics as statics_mod
from modules import velocity as vel_mod
from modules.dependencies import StageCheck, Workflow, check_stage, get_spec, state_name
from modules.pipeline import (ALIAS_KEYS, PROCESSING_STORE_DEFAULTS, ROOT_STAGES, STAGE_KIND,
                              stage_label)
from modules.state_graph import (DataState, GraphError, ProcessingGraph, array_digest, canonical)
from modules.validation import (DependencyError, ValidationError, require_gather_key,
                                validate_cdp_gathers, validate_frequency_band,
                                validate_velocity_field)

ProgressCb = Optional[Callable[[float], None]]

# Operations that change each trace's gain: amplitude-based trace detection must
# look at data from *before* the first of them.
GAIN_ALTERING_STAGES: Tuple[str, ...] = ("agc", "balance")


# ─────────────────────────────────────────────────────────────────────────
# Store access
# ─────────────────────────────────────────────────────────────────────────

def new_store(survey=None) -> Dict[str, Any]:
    """A fresh, empty data store (a plain dict; the same keys as the app's
    session state, minus its UI-only entries)."""
    store = copy.deepcopy(PROCESSING_STORE_DEFAULTS)
    store["graph"] = ProcessingGraph()
    if survey is not None:
        run_load_survey(store, survey, getattr(survey, "source_name", "survey"))
    return store


def graph(store: MutableMapping) -> ProcessingGraph:
    if store.get("graph") is None:
        store["graph"] = ProcessingGraph()
    return store["graph"]


def containers(store: MutableMapping) -> Dict[str, dict]:
    if store.get("containers") is None:
        store["containers"] = {}
    return store["containers"]


def blobs(store: MutableMapping) -> Dict[str, dict]:
    if store.get("blobs") is None:
        store["blobs"] = {}
    return store["blobs"]


def workflow(store: MutableMapping) -> Workflow:
    """The dependency engine's view of this store's graph."""
    return Workflow(graph(store))


# ---- remembered input choices (a UI convenience, never part of a state) ---------------

def _pick_key(stage_id: str, variant: Optional[str] = None) -> str:
    return f"{stage_id}:{variant}" if variant else stage_id


def get_picks(store: MutableMapping, stage_id: str, variant: Optional[str] = None) -> Dict[str, str]:
    """The input states a person chose for this stage ({role: state id})."""
    return dict((store.get("input_picks") or {}).get(_pick_key(stage_id, variant), {}))


def set_pick(store: MutableMapping, stage_id: str, role: str, state_id: Optional[str],
             variant: Optional[str] = None) -> None:
    """Remember (or, with None, forget) the input state chosen for one role."""
    picks = {k: dict(v) for k, v in (store.get("input_picks") or {}).items()}
    entry = picks.setdefault(_pick_key(stage_id, variant), {})
    if state_id:
        entry[role] = state_id
    else:
        entry.pop(role, None)
    store["input_picks"] = {k: v for k, v in picks.items() if v}


def check(store: MutableMapping, stage_id: str, variant: Optional[str] = None,
          inputs: Optional[Dict[str, str]] = None) -> StageCheck:
    """✓ / ⚠ / ⛔ for an operation against the current data states.

    With no `inputs`, the person's remembered input choices are used (and
    ignored if they no longer exist); explicit `inputs` are strict."""
    lenient = inputs is None
    chosen = get_picks(store, stage_id, variant) if lenient else dict(inputs)
    return check_stage(workflow(store), stage_id, variant, inputs=chosen, lenient=lenient)


# ---- reading a state's payload ------------------------------------------------------------------

def gathers_of(store: MutableMapping, state: Optional[DataState]) -> dict:
    """The gather container a gather-based state's arrays live in."""
    root = state.root if state is not None else None
    container = containers(store).get(root) if root else None
    if container is None:
        raise DependencyError(
            f"{state_name(state)} has no gathers in memory.", stage_id=state.op if state else "")
    return container


def io_keys(store: MutableMapping, state: DataState) -> Tuple[dict, Dict[str, str]]:
    """(gathers, {logical name: key inside each gather}) of a gather-based state."""
    return gathers_of(store, state), state.keys


def blob_of(store: MutableMapping, state: Optional[DataState], name: str, default: Any = None) -> Any:
    if state is None:
        return default
    return blobs(store).get(state.id, {}).get(name, default)


def section_of(store: MutableMapping, state: Optional[DataState]) -> Optional[np.ndarray]:
    """The (n_cdp, n_samples) section of a stack / post-stack / image / final state."""
    return blob_of(store, state, "section")


def cdp_list_of(store: MutableMapping, state: DataState) -> List[int]:
    """The CDP numbers a CDP-based state (gathers, velocity, section …) is laid out on."""
    return list(gathers_of(store, state).keys())


# ─────────────────────────────────────────────────────────────────────────
# One execution
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    """What an operation's compute function sees: its input states, the
    store's payloads, and a place to put its output."""
    store: MutableMapping
    sid: str                                  # id the output state will get
    stage_id: str
    variant: str
    inputs: Dict[str, str]                    # role -> input state id
    progress_cb: ProgressCb = None
    new_gathers: Optional[dict] = None        # set by root operations (shot_gathers, cdp_sort)
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def survey(self):
        return self.store["survey"]

    def state(self, role: str = "data") -> DataState:
        return graph(self.store)[self.inputs[role]]

    def key(self, name: str) -> str:
        """The key this state's own `name` array gets inside every gather dict."""
        return f"{name}@{self.sid}"

    def container(self, role: str = "data") -> dict:
        return gathers_of(self.store, self.state(role))

    def in_key(self, role: str = "data", name: str = "traces") -> str:
        """Key of an input state's `name` array — checked to be on every gather."""
        st = self.state(role)
        key = st.keys.get(name)
        if key is None:
            raise DependencyError(f"{state_name(st)} holds no '{name}' array.", stage_id=self.stage_id,
                                  missing=[name])
        require_gather_key(self.container(role), key, f"the {state_name(st)} output",
                           stage_label(self.stage_id))
        return key

    def off_key(self, role: str = "data") -> str:
        return self.state(role).keys.get("offsets", "offsets")

    def blob(self, role: str = "data") -> dict:
        return blobs(self.store).get(self.inputs[role], {})

    def section(self, role: str = "data") -> np.ndarray:
        sec = self.blob(role).get("section")
        if sec is None:
            st = self.state(role)
            raise DependencyError(f"The {state_name(st)} section is not available.",
                                  stage_id=self.stage_id, missing=[state_name(st)])
        return sec

    def put(self, **arrays: Any) -> None:
        """Store non-gather results (sections, fields …) under this state."""
        self.payload.update(arrays)

    def tick(self, i: int, n: int) -> None:
        if self.progress_cb and (i % 20 == 0):
            self.progress_cb((i + 1) / n)

    def done(self) -> None:
        if self.progress_cb:
            self.progress_cb(1.0)


Computed = Tuple[Dict[str, str], Dict[str, Any]]          # (state.keys, state.info)


@dataclass(frozen=True)
class OpDef:
    compute: Callable[[Run, dict], Computed]
    extras: Optional[Callable[[MutableMapping, StageCheck, dict], Dict[str, str]]] = None


OPS: Dict[str, OpDef] = {}


def _op(key: str, extras: Optional[Callable] = None):
    def deco(fn):
        OPS[key] = OpDef(fn, extras)
        return fn
    return deco


def _op_key(stage_id: str, variant: Optional[str] = None) -> str:
    return f"{stage_id}:{variant}" if variant else stage_id


def _free_owned(store: MutableMapping, root: Optional[str], sid: str) -> None:
    """Drop every array state `sid` owns."""
    if root == sid:
        containers(store).pop(sid, None)
    else:
        suffix = f"@{sid}"
        container = containers(store).get(root) if root else None
        if container:
            for g in container.values():
                for k in [k for k in g if k.endswith(suffix)]:
                    del g[k]
    blobs(store).pop(sid, None)


def execute(store: MutableMapping, stage_id: str, params: Optional[dict] = None,
            variant: Optional[str] = None, inputs: Optional[Dict[str, str]] = None,
            progress_cb: ProgressCb = None, reuse: bool = True,
            meta: Optional[dict] = None) -> DataState:
    """Run one operation and return the state it produced.

    `inputs` names input states explicitly ({'data': 'S4'}); anything not
    named is resolved by the dependency engine (the person's remembered
    choice, else the current data of the right kind). `reuse`: if a state
    with the same lineage fingerprint already exists it is re-used (made
    current) instead of being computed and stored again. `meta` is extra
    diagnostic info to record (not part of the state's identity)."""
    opdef = OPS.get(_op_key(stage_id, variant))
    if opdef is None:
        raise KeyError(f"No executor for stage '{_op_key(stage_id, variant)}'")
    g = graph(store)
    chk = check(store, stage_id, variant, inputs)
    if not chk.can_run:
        raise DependencyError(f"Cannot run {stage_label(stage_id)}. {chk.blocker_text()}",
                              stage_id=stage_id,
                              missing=[s for i in chk.blockers for s in i.stages])
    try:
        cparams = canonical(params or {})
    except TypeError as exc:
        raise ValidationError(str(exc)) from exc
    in_ids = {role: sid for role, sid in chk.inputs.items() if sid}
    if opdef.extras:
        in_ids.update(opdef.extras(store, chk, cparams))

    if reuse:
        hit = g.find(g.fingerprint_for(stage_id, variant or "", cparams, in_ids))
        if hit is not None:
            materialize(store, hit.id)
            g.select(hit.id)
            sync_aliases(store)
            store["last_run"] = {"state": hit.id, "op": stage_id, "variant": variant or "",
                                 "reused": True}
            return hit

    for src in in_ids.values():
        materialize(store, src, progress_cb=None)
    sid = g.peek_id()
    root = "self" if stage_id in ROOT_STAGES else (g[in_ids["data"]].root if "data" in in_ids else None)
    run = Run(store, sid, stage_id, variant or "", in_ids, progress_cb)
    try:
        keys, info = opdef.compute(run, cparams)
    except BaseException:
        _free_owned(store, sid if root == "self" else root, sid)
        raise
    if run.new_gathers is not None:
        containers(store)[sid] = run.new_gathers
    if run.payload:
        blobs(store)[sid] = run.payload
    notes = ("Run despite: " + " | ".join(w.message for w in chk.warnings)) if chk.warnings else ""
    state = g.add(kind=STAGE_KIND[stage_id], op=stage_id, variant=variant or "", params=cparams,
                  inputs=in_ids, root=root, keys=keys, info={**info, **(meta or {})}, notes=notes,
                  sid=sid)
    sync_aliases(store)
    store["last_run"] = {"state": state.id, "op": stage_id, "variant": variant or "", "reused": False}
    budget = float(getattr(cfg, "STATE_MEMORY_BUDGET_MB", 0) or 0)
    if budget > 0:
        enforce_budget(store, int(budget * 2 ** 20), protect=(state.id,))
    return state


# ─────────────────────────────────────────────────────────────────────────
# Memory: release / materialize
# ─────────────────────────────────────────────────────────────────────────

def _nbytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    if hasattr(obj, "__dataclass_fields__"):
        return sum(_nbytes(getattr(obj, f)) for f in obj.__dataclass_fields__)
    return 0


def payload_bytes(store: MutableMapping, sid: str) -> int:
    """Memory held by the arrays state `sid` owns (an upload counts the raw trace array)."""
    s = graph(store).get(sid)
    if s is None or not s.materialized:
        return 0
    total = _nbytes(blobs(store).get(sid))
    if s.kind == "raw":
        raw = getattr(store.get("survey"), "raw_data", None)
        return total + (int(raw.nbytes) if raw is not None else 0)
    container = containers(store).get(s.root) if s.root else None
    if container:
        if s.is_root:
            total += sum(_nbytes(v) for g in container.values() for k, v in g.items() if "@" not in k)
        else:
            suffix = f"@{sid}"
            total += sum(_nbytes(v) for g in container.values() for k, v in g.items()
                         if k.endswith(suffix))
    return total


def releasable(store: MutableMapping, sid: str) -> bool:
    """Can this state's arrays be dropped and re-computed later? Roots own the base
    gathers (and the survey owns the raw data), so they stay."""
    s = graph(store).get(sid)
    return s is not None and s.materialized and not s.is_root and s.kind != "raw"


def memory_report(store: MutableMapping) -> List[Dict[str, Any]]:
    return [{"id": s.id, "op": s.op, "variant": s.variant, "kind": s.kind,
             "bytes": payload_bytes(store, s.id), "materialized": s.materialized,
             "releasable": releasable(store, s.id)} for s in graph(store)]


def release_payload(store: MutableMapping, sid: str) -> int:
    """Free a state's arrays but keep its record. Returns the bytes freed. The
    state is re-computed from its recorded operation and parameters when it
    is next used (`materialize`)."""
    g = graph(store)
    s = g[sid]
    if not releasable(store, sid):
        raise ValidationError(f"{state_name(s)} cannot be released (it is a base set of gathers, "
                              f"the loaded data, or already released).")
    freed = payload_bytes(store, sid)
    _free_owned(store, s.root, sid)
    s.materialized = False
    g.log("release", [sid], f"released {sid} ({freed / 2 ** 20:.1f} MB)")
    sync_aliases(store)
    return freed


def enforce_budget(store: MutableMapping, budget_bytes: int, protect: Sequence[str] = ()) -> List[str]:
    """Release the least recently used states until the payloads fit the budget.
    The current state of every kind, and `protect`, are never released."""
    g = graph(store)
    keep = set(protect) | {c.id for k in ("shot", "cdp", "velocity", "nmo", "dmo", "section", "image",
                                          "final") if (c := g.current(k)) is not None}
    released: List[str] = []
    total = sum(payload_bytes(store, s.id) for s in g)
    for s in sorted(g, key=lambda x: x.touch):
        if total <= budget_bytes:
            break
        if s.id in keep or not releasable(store, s.id):
            continue
        total -= release_payload(store, s.id)
        released.append(s.id)
    return released


def materialize(store: MutableMapping, sid: str, progress_cb: ProgressCb = None) -> DataState:
    """Make sure a state's arrays are in memory, re-computing them from the
    recorded operation and parameters if they were released."""
    g = graph(store)
    s = g[sid]
    if s.materialized:
        return s
    for src in s.inputs.values():
        materialize(store, src)
    opdef = OPS[_op_key(s.op, s.variant)]
    run = Run(store, sid, s.op, s.variant, dict(s.inputs), progress_cb)
    try:
        opdef.compute(run, copy.deepcopy(s.params))
    except BaseException:
        _free_owned(store, s.root, sid)
        raise
    if run.payload:
        blobs(store)[sid] = run.payload
    s.materialized = True
    g.log("materialize", [sid], f"re-computed {sid} from its recorded operation")
    sync_aliases(store)
    return s


# ─────────────────────────────────────────────────────────────────────────
# Moving around the graph
# ─────────────────────────────────────────────────────────────────────────

def select_state(store: MutableMapping, sid: str) -> DataState:
    """Make a state the current data of its kind (a deliberate choice)."""
    materialize(store, sid)
    s = graph(store).select(sid)
    sync_aliases(store)
    return s


def undo(store: MutableMapping, kind: str) -> Optional[DataState]:
    """Step the current data of `kind` back to the state it was made from."""
    g = graph(store)
    target = g.undo_target(kind)
    if target is None:
        return None
    materialize(store, target.id)
    g.undo(kind)
    sync_aliases(store)
    return target


def redo(store: MutableMapping, kind: str) -> Optional[DataState]:
    g = graph(store)
    target = g.redo_target(kind)
    if target is None:
        return None
    materialize(store, target.id)
    g.redo(kind)
    sync_aliases(store)
    return target


def discard_state(store: MutableMapping, sid: str) -> List[str]:
    """Delete a state AND every state built on it, freeing their arrays."""
    g = graph(store)
    doomed = [(d.id, d.root) for d in reversed(g.dependents(sid))] + [(sid, g[sid].root)]
    removed = g.discard(sid)
    for state_id, root in doomed:
        _free_owned(store, root, state_id)
    gone = set(removed)
    store["input_picks"] = {k: {r: s for r, s in v.items() if s not in gone}
                            for k, v in (store.get("input_picks") or {}).items()}
    store["input_picks"] = {k: v for k, v in store["input_picks"].items() if v}
    if (store.get("last_run") or {}).get("state") in gone:
        store["last_run"] = None
    sync_aliases(store)
    return removed


def reset_store(store: MutableMapping) -> None:
    """Forget everything derived from the loaded survey."""
    store["graph"] = ProcessingGraph()
    store["containers"] = {}
    store["blobs"] = {}
    store["input_picks"] = {}
    store["manual_overrides_1"], store["manual_overrides_2"] = {}, {}
    store["excluded_cdps_1"], store["excluded_cdps_2"] = set(), set()
    store["last_run"] = None
    sync_aliases(store)


# ─────────────────────────────────────────────────────────────────────────
# Legacy aliases (references, never copies)
# ─────────────────────────────────────────────────────────────────────────

def mute_summary(kind: str, **kw) -> str:
    if kind == "flat":
        return f"flat top mute, t_top={kw['t_top'] * 1000:.0f} ms"
    if kind == "linear":
        return f"top mute (moveout), v={kw['velocity']:.0f} m/s, t0={kw['t0'] * 1000:.0f} ms"
    return (f"surgical mute, offset [{kw['x_min']:.0f},{kw['x_max']:.0f}] m, "
            f"time [{kw['t_min'] * 1000:.0f},{kw['t_max'] * 1000:.0f}] ms")


def sync_aliases(store: MutableMapping) -> None:
    """Refresh the legacy convenience keys (`stack_matrix`, `Vfield_1`,
    `gathers` …) so they point at the payload of the current/latest state of
    each stage. References only: no array is copied."""
    g, C, B = graph(store), containers(store), blobs(store)
    for k in ALIAS_KEYS:
        store[k] = copy.copy(PROCESSING_STORE_DEFAULTS[k])
    store["completed_stages"] = {s.op for s in g}

    def live(state: Optional[DataState]) -> Optional[DataState]:
        return state if state is not None and state.materialized else None

    shot = g.current("shot")
    if shot is not None:
        store["shot_gathers"] = C.get(shot.root)
    cdp = g.current("cdp")
    if cdp is not None and C.get(cdp.root) is not None:
        store["gathers"] = C[cdp.root]
        store["cdp_list"] = list(C[cdp.root].keys())

    for stage, (fk, ak, rk) in {"vel1": ("Vfield_1", "analysis_1", "vel_results_1"),
                                "vel2": ("Vfield_2", "analysis_2", "vel_results_2")}.items():
        s = live(g.latest(stage))
        if s is not None:
            b = B.get(s.id, {})
            store[fk], store[ak], store[rk] = b.get("field"), b.get("analysis"), b.get("results")
    cur_vel = live(g.current("velocity"))
    if cur_vel is not None:
        store["velocities"] = B.get(cur_vel.id, {}).get("velocities")

    s = live(g.latest("statics"))
    if s is not None:
        b = B.get(s.id, {})
        store["shot_static"], store["receiver_static"] = b.get("shot_static"), b.get("receiver_static")
        store["static_diag"] = b.get("diag")
    s = live(g.latest("data_reg"))
    if s is not None:
        store["regular_offsets"] = B.get(s.id, {}).get("regular_offsets")
    s = live(g.latest("stack"))
    if s is not None:
        b = B.get(s.id, {})
        store["stack_matrix"], store["fold_matrix"], store["naive_matrix"] = (
            b.get("section"), b.get("fold"), b.get("naive"))
    for stage, key in (("rna", "stack_rna"), ("decon_post", "stack_decon_post"), ("tvf", "stack_tvf")):
        s = live(g.latest(stage))
        if s is not None:
            store[key] = B.get(s.id, {}).get("section")
    img = live(g.current("image"))
    if img is not None:
        b = B.get(img.id, {})
        store["stack_migrated"] = b.get("section")
        if img.variant == "pstm":
            store["pstm_image"], store["pstm_fold"] = b.get("section"), b.get("fold")
    fin = live(g.current("final"))
    if fin is not None:
        store["stack_final"] = B.get(fin.id, {}).get("section")

    s = g.latest("agc")
    if s is not None:
        store["agc_config"] = {"window": s.params.get("window"), "mode": s.params.get("mode")}
    s = g.latest("mute")
    if s is not None:
        store["mute_config"] = {**s.params, "summary": mute_summary(**s.params)}
    s = g.latest("fk")
    if s is not None:
        store["fk_polygon"] = s.params.get("polygon")
        store["fk_config"] = {"mode": s.params.get("mode"), "coeff": s.params.get("coeff"),
                              "source": s.info.get("source", "custom polygon")}


# ─────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────

def replay_recipe(store: MutableMapping, sid: str, survey=None, progress_cb: ProgressCb = None
                  ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Re-create state `sid` from scratch in a NEW store, by re-running the
    recorded operations with their recorded parameters on the loaded survey.

    Returns (new_store, {old state id: new state id}). Raises `GraphError` if
    any step's lineage fingerprint differs from the original — i.e. if the
    run was not reproducible (different survey, changed maths …)."""
    recipe = graph(store).recipe(sid)
    survey = survey if survey is not None else store["survey"]
    new = new_store(survey)
    new_up = graph(new).latest("upload")
    mapping: Dict[str, str] = {}
    for step in recipe:
        if step["op"] == "upload":
            if step["fingerprint"] != new_up.fingerprint:
                raise GraphError("The survey differs from the one this processing was applied to "
                                 "(its data fingerprint is not the same), so it cannot be replayed.")
            mapping[step["id"]] = new_up.id
            continue
        roles = {n.role for n in get_spec(step["op"], step["variant"] or None).needs}
        inputs = {r: mapping[src] for r, src in step["inputs"].items() if r in roles}
        state = execute(new, step["op"], step["params"], step["variant"] or None, inputs=inputs,
                        progress_cb=progress_cb)
        if state.fingerprint != step["fingerprint"]:
            raise GraphError(f"Replaying {step['id']} ({stage_label(step['op'])}) did not reproduce the "
                             f"recorded fingerprint — the computation is not reproducible.")
        mapping[step["id"]] = state.id
    return new, mapping


# ─────────────────────────────────────────────────────────────────────────
# Data: load, shot gathers
# ─────────────────────────────────────────────────────────────────────────

def run_load_survey(store: MutableMapping, survey, source_label: str) -> DataState:
    """Install a survey as the (only) loaded data. Everything derived from
    the previous survey is discarded; the new graph starts at S0."""
    reset_store(store)
    store["survey"] = survey
    state = graph(store).add(
        kind="raw", op="upload",
        params={"source": source_label, "traces": int(survey.num_traces),
                "samples": int(survey.num_samples), "dt": float(survey.sample_interval),
                "digest": array_digest(survey.raw_data)},
        keys={"traces": "raw_data"}, info={"synthetic": bool(getattr(survey, "is_synthetic", False))})
    sync_aliases(store)
    store["last_run"] = {"state": state.id, "op": "upload", "variant": "", "reused": False}
    return state


@_op("shot_gathers")
def _c_shot_gathers(run: Run, p: dict) -> Computed:
    gathers = segy_io.build_shot_gathers(run.survey, n_min_traces=int(p["n_min_traces"]))
    run.new_gathers = gathers
    return {"traces": "traces", "offsets": "offsets"}, {"n_shots": len(gathers)}


def run_build_shot_gathers(store: MutableMapping, n_min_traces: int = 1) -> DataState:
    return execute(store, "shot_gathers", {"n_min_traces": int(n_min_traces)})


# ─────────────────────────────────────────────────────────────────────────
# Shot-domain pre-stack conditioning — each independent of the others
# ─────────────────────────────────────────────────────────────────────────

def _shot_io(run: Run) -> Tuple[dict, str, str]:
    return run.container(), run.in_key(), run.off_key()


def _shot_out(run: Run, ok: str, **extra: str) -> Dict[str, str]:
    return {"traces": run.key("traces"), "offsets": ok, **extra}


@_op("bandpass")
def _c_bandpass(run: Run, p: dict) -> Computed:
    survey = run.survey
    gathers, ik, ok = _shot_io(run)
    validate_frequency_band(p["fmin"], p["fmax"], 0.5 / survey.sample_interval)
    h = filtering.design_bandpass(p["fmin"], p["fmax"], survey.sample_interval, survey.num_samples,
                                  p["low_slope"], p["high_slope"])
    out = run.key("traces")
    for g in gathers.values():
        g[out] = filtering.apply_time_domain_conv(g[ik], h)
    return _shot_out(run, ok), {}


def run_bandpass(store: MutableMapping, fmin: float = cfg.FMIN_DEFAULT, fmax: float = cfg.FMAX_DEFAULT,
                 low_slope: float = cfg.LOW_SLOPE_DEFAULT, high_slope: float = cfg.HIGH_SLOPE_DEFAULT,
                 inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "bandpass", {"fmin": fmin, "fmax": fmax, "low_slope": low_slope,
                                       "high_slope": high_slope}, inputs=inputs)


@_op("agc")
def _c_agc(run: Run, p: dict) -> Computed:
    gathers, ik, ok = _shot_io(run)
    out = run.key("traces")
    for g in gathers.values():
        g[out] = preprocessing.agc(g[ik], window=p["window"], mode=p["mode"])
    return _shot_out(run, ok), {}


def run_agc(store: MutableMapping, window: int = cfg.AGC_WINDOW_DEFAULT, mode: str = cfg.AGC_MODE_DEFAULT,
            inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "agc", {"window": window, "mode": mode}, inputs=inputs)


@_op("despike")
def _c_despike(run: Run, p: dict) -> Computed:
    gathers, ik, ok = _shot_io(run)
    out, mask_key = run.key("traces"), run.key("spike_mask")
    total = 0
    for g in gathers.values():
        cleaned, spike_mask = preprocessing.despike_traces(g[ik], p["window"], p["threshold"])
        g[out], g[mask_key] = cleaned, spike_mask
        total += int(spike_mask.sum())
    return _shot_out(run, ok, spike_mask=mask_key), {"samples_flagged": total}


def run_despike(store: MutableMapping, window: int = 15, threshold: float = 6.0,
                inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "despike", {"window": window, "threshold": threshold}, inputs=inputs)


def kill_detection_index(chain: Sequence[str]) -> int:
    """Index (into `chain`) of the stage whose output trace-kill detection should measure.

    Energy-based dead/noisy detection is blind after a per-trace gain (AGC,
    balancing) has equalised every trace's level, so it must look at
    amplitude-preserving data: the output of the operation just BEFORE the
    first gain-altering one in the data's history. `chain` is the data's
    ancestry (stage ids), oldest first, ending with the data being edited.
    With no gain stage in it, the edited data itself is measured."""
    for i, sid in enumerate(chain):
        if sid in GAIN_ALTERING_STAGES and i > 0:
            return i - 1
    return len(chain) - 1


def kill_detection_stage(chain: Sequence[str]) -> str:
    return chain[kill_detection_index(chain)]


def _extras_trace_edit(store: MutableMapping, chk: StageCheck, params: dict) -> Dict[str, str]:
    """Trace edit reads two states: the data it edits, and the (amplitude-
    preserving) state the kill mask is measured on. Recording the second
    one makes the result independent of where in the order the edit was run."""
    data = chk.state("data")
    if data is None:
        return {}
    lineage = graph(store).lineage(data.id)
    detect = lineage[kill_detection_index([s.op for s in lineage])]
    return {"detect": detect.id} if detect.id != data.id else {}


@_op("trace_edit", extras=_extras_trace_edit)
def _c_trace_edit(run: Run, p: dict) -> Computed:
    gathers, ik, ok = _shot_io(run)
    role = "detect" if "detect" in run.inputs else "data"
    dk = run.in_key(role)
    out, flag_key = run.key("traces"), run.key("kill_flags")
    total = 0
    for g in gathers.values():
        flags = preprocessing.flag_bad_traces(g[dk], p["dead_frac"], p["noisy_mult"])
        g[out] = preprocessing.apply_trace_kill(g[ik], flags["kill"])
        g[flag_key] = flags
        total += int(flags["kill"].sum())
    return (_shot_out(run, ok, kill_flags=flag_key),
            {"traces_killed": total, "detected_on": state_name(run.state(role))})


def run_trace_kill(store: MutableMapping, dead_frac: float = 0.02, noisy_mult: float = 8.0,
                   inputs: Optional[Dict[str, str]] = None) -> DataState:
    """Kill dead/noisy traces. The mask is *measured* on amplitude-preserving
    data (upstream of any AGC / balancing in this data's history) and
    *applied* to the data actually being processed."""
    return execute(store, "trace_edit", {"dead_frac": dead_frac, "noisy_mult": noisy_mult}, inputs=inputs)


@_op("mute")
def _c_mute(run: Run, p: dict) -> Computed:
    dt = run.survey.sample_interval
    gathers, ik, ok = _shot_io(run)
    kind = p["kind"]
    out, mt_key, box_key = run.key("traces"), run.key("mute_times"), run.key("surgical_box")
    for g in gathers.values():
        data = g[ik]
        if kind == "flat":
            muted = preprocessing.apply_top_bottom_mute(data, dt, p["t_top"], taper_s=p["taper"])
            mute_times, in_box = np.full(data.shape[0], p["t_top"]), None
        elif kind == "linear":
            muted, mute_times = preprocessing.apply_linear_mute(
                data, g[ok], dt, p["velocity"], t_intercept=p["t0"], taper_s=p["taper"])
            in_box = None
        else:
            muted, in_box = preprocessing.apply_surgical_mute(
                data, g[ok], dt, t_min=p["t_min"], t_max=p["t_max"], x_min=p["x_min"], x_max=p["x_max"],
                taper_s=p["taper"])
            mute_times = np.full(data.shape[0], p["t_max"])
        g[out], g[mt_key], g[box_key] = muted, mute_times, in_box
    return _shot_out(run, ok, mute_times=mt_key, surgical_box=box_key), {"summary": mute_summary(**p)}


def run_mute(store: MutableMapping, kind: str = "flat", *, taper: float = cfg.MUTE_TAPER_DEFAULT,
             t_top: float = cfg.MUTE_TTOP_DEFAULT, velocity: float = 500.0, t0: float = 0.0,
             x_min: float = 0.0, x_max: float = 0.0, t_min: float = 0.0, t_max: float = 0.0,
             inputs: Optional[Dict[str, str]] = None) -> DataState:
    if kind not in ("flat", "linear", "surgical"):
        raise ValidationError(f"Unknown mute type '{kind}' (expected flat, linear or surgical).")
    if kind == "flat":
        params = dict(kind=kind, t_top=t_top, taper=taper)
    elif kind == "linear":
        params = dict(kind=kind, velocity=velocity, t0=t0, taper=taper)
    else:
        params = dict(kind=kind, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max, taper=taper)
    return execute(store, "mute", params, inputs=inputs)


@_op("balance")
def _c_balance(run: Run, p: dict) -> Computed:
    dt = run.survey.sample_interval
    gathers, ik, ok = _shot_io(run)
    ref_window = slice(0, int(p["ref_s"] / dt))
    out = run.key("traces")
    for g in gathers.values():
        g[out] = preprocessing.trace_normalize(g[ik], ref_window)
    return _shot_out(run, ok), {}


def run_balance(store: MutableMapping, ref_s: float = cfg.NORM_WINDOW_S_DEFAULT,
                inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "balance", {"ref_s": ref_s}, inputs=inputs)


@_op("fk")
def _c_fk(run: Run, p: dict) -> Computed:
    survey = run.survey
    gathers, ik, ok = _shot_io(run)
    polygon = p["polygon"]
    out = run.key("traces")
    items = list(gathers.values())
    for i, g in enumerate(items):
        filt, *_ = filtering.fk_filter_gather(g[ik], survey.sample_interval, survey.group_interval,
                                              polygon, polygon_mode=p["mode"],
                                              reject_coefficient=p["coeff"])
        g[out] = filt
        run.tick(i, len(items))
    run.done()
    n_vertices = (sum(len(ring) for ring in polygon)
                  if polygon and isinstance(polygon[0][0], (list, tuple)) else len(polygon))
    return _shot_out(run, ok), {"polygon_vertices": n_vertices}


def run_fk(store: MutableMapping, polygon, mode: str = cfg.FK_POLYGON_MODE_DEFAULT,
           coeff: float = cfg.FK_REJECT_COEFFICIENT_DEFAULT, source_desc: str = "custom polygon",
           progress_cb: ProgressCb = None, inputs: Optional[Dict[str, str]] = None) -> DataState:
    if polygon is None or len(polygon) < 3:
        raise ValidationError("An F-K polygon needs at least 3 vertices.")
    return execute(store, "fk", {"polygon": polygon, "mode": mode, "coeff": float(coeff)},
                   progress_cb=progress_cb, inputs=inputs, meta={"source": source_desc})


@_op("mult_supp")
def _c_mult_supp(run: Run, p: dict) -> Computed:
    gathers, ik, ok = _shot_io(run)
    out = run.key("traces")
    items = list(gathers.values())
    for i, g in enumerate(items):
        g[out] = preprocessing.predictive_decon_gather(g[ik], p["operator_length"], p["prediction_lag"],
                                                       p["pre_whitening"])
        run.tick(i, len(items))
    run.done()
    return _shot_out(run, ok), {}


def run_mult_supp(store: MutableMapping, operator_length: int = 60, prediction_lag: int = 30,
                  pre_whitening: float = cfg.DECON_PW_DEFAULT, progress_cb: ProgressCb = None,
                  inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "mult_supp", {"operator_length": operator_length,
                                        "prediction_lag": prediction_lag,
                                        "pre_whitening": pre_whitening},
                   progress_cb=progress_cb, inputs=inputs)


@_op("decon")
def _c_decon(run: Run, p: dict) -> Computed:
    gathers, ik, ok = _shot_io(run)
    out = run.key("traces")
    items = list(gathers.values())
    for i, g in enumerate(items):
        g[out] = preprocessing.spiking_decon_gather(g[ik], p["operator_length"], p["pre_whitening"])
        run.tick(i, len(items))
    run.done()
    return _shot_out(run, ok), {}


def run_decon(store: MutableMapping, operator_length: int = cfg.DECON_OPLEN_DEFAULT,
              pre_whitening: float = cfg.DECON_PW_DEFAULT, progress_cb: ProgressCb = None,
              inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "decon", {"operator_length": operator_length, "pre_whitening": pre_whitening},
                   progress_cb=progress_cb, inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# CDP sort
# ─────────────────────────────────────────────────────────────────────────

@_op("cdp_sort")
def _c_cdp_sort(run: Run, p: dict) -> Computed:
    """Group traces by CDP. Takes the current shot-domain data (whatever
    conditioning was applied, in whatever order) — or the raw survey if no
    shot gathers were built, since sorting only needs the headers."""
    survey = run.survey
    n_min = int(p["n_min_traces"])
    src = run.state()
    if src.kind == "raw":
        gathers = segy_io.build_cdp_gathers(survey, n_min_traces=n_min)
    else:
        flat = segy_io.scatter_to_flat(run.container(), run.in_key(), survey.num_traces,
                                       survey.num_samples)
        gathers = segy_io.build_cdp_gathers(survey, n_min_traces=n_min, data_override=flat)
    validate_cdp_gathers(gathers, n_min)
    run.store["n_min_traces"] = n_min
    run.new_gathers = gathers
    return {"traces": "traces", "offsets": "offsets"}, {"n_cdp": len(gathers), "source": state_name(src)}


def run_cdp_sort(store: MutableMapping, n_min_traces: int = cfg.N_MIN_TRACES_DEFAULT,
                 inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "cdp_sort", {"n_min_traces": int(n_min_traces)}, inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# CDP domain: velocity, NMO, statics, DMO, regularisation
# ─────────────────────────────────────────────────────────────────────────

_VEL_STORE_KEYS = {"vel1": ("manual_overrides_1", "excluded_cdps_1"),
                   "vel2": ("manual_overrides_2", "excluded_cdps_2")}


@_op("vel1")
@_op("vel2")
def _c_velocity(run: Run, p: dict) -> Computed:
    survey = run.survey
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    velocities = np.linspace(p["vmin"], p["vmax"], int(p["n_velocities"]))
    overrides = {int(c): np.asarray(v, dtype=float) for c, v in p["manual_overrides"].items()}
    label = f"@{run.sid}"
    Vfield, analysis, results = vel_mod.pick_velocity_field(
        gathers, list(gathers.keys()), ik, velocities, survey.sample_interval, survey.num_samples,
        n_analysis=int(p["n_analysis"]), min_abs_offset=p["min_offset"], label=label,
        window_len=int(p["window_len"]), reliability_threshold=p["reliability_threshold"],
        autopick_kwargs=dict(min_time=p["min_time"], search_half_width_ms=p["search_w"],
                             min_confidence=p["min_conf"], smooth_sigma=int(p["smooth"])),
        manual_overrides=overrides, excluded_cdps=set(int(c) for c in p["excluded_cdps"]),
        progress_cb=run.progress_cb, offsets_key=ok)
    run.put(field=Vfield, analysis=analysis, results=results, velocities=velocities)
    n_reliable = sum(r.is_reliable for r in results.values())
    keys = {"semblance": f"semblance_{label}", "vrms": f"vrms_{label}",
            "confidence": f"vrms_confidence_{label}", "curve": f"vrms_field_{label}", "offsets": ok}
    return keys, {"n_analysis": len(analysis), "reliable": f"{n_reliable}/{len(analysis)}"}


def velocity_pick_params(store: MutableMapping, stage_id: str, manual_overrides: Optional[dict] = None,
                         excluded_cdps: Optional[set] = None) -> Tuple[Dict[str, List[float]], List[int]]:
    """The person's manual picks and excluded CDPs for a velocity pass, in the form
    recorded as parameters of the state ({cdp: [velocity per sample]}, [cdp, …]).
    Comparing this with a state's recorded parameters tells whether the picks
    were edited after that state was made."""
    mo_key, ex_key = _VEL_STORE_KEYS[stage_id]
    mo = (store.get(mo_key) or {}) if manual_overrides is None else manual_overrides
    ex = (store.get(ex_key) or set()) if excluded_cdps is None else excluded_cdps
    return ({str(int(c)): [float(x) for x in np.asarray(v, dtype=float)] for c, v in sorted(mo.items())},
            sorted(int(c) for c in ex))


def run_velocity(store: MutableMapping, stage_id: str = "vel1", *, vmin: float = cfg.VELOCITY_MIN_DEFAULT,
                 vmax: float = cfg.VELOCITY_MAX_DEFAULT, n_analysis: int = cfg.N_ANALYSIS_CDPS_DEFAULT,
                 min_offset: float = cfg.MIN_ABS_OFFSET_DEFAULT,
                 window_len: int = cfg.SEMBLANCE_WINDOW_LEN_DEFAULT,
                 min_time: float = cfg.AUTOPICK_MIN_TIME_DEFAULT,
                 search_w: float = cfg.AUTOPICK_SEARCH_HALF_WIDTH_MS_DEFAULT,
                 min_conf: float = cfg.AUTOPICK_MIN_CONFIDENCE_DEFAULT,
                 smooth: int = cfg.AUTOPICK_SMOOTH_SIGMA_DEFAULT,
                 reliability_threshold: float = cfg.RELIABILITY_THRESHOLD_DEFAULT,
                 manual_overrides: Optional[dict] = None, excluded_cdps: Optional[set] = None,
                 progress_cb: ProgressCb = None, inputs: Optional[Dict[str, str]] = None) -> DataState:
    """Semblance velocity analysis. Pass 1 analyses the CDP-sorted gathers; pass 2
    analyses whatever the pre-NMO CDP gathers are by then (refined by statics /
    DMO / regularisation, or not).

    The manual picks and excluded CDPs the person has made (kept in the store
    as `manual_overrides_N` / `excluded_cdps_N`) are *parameters of the state*:
    recorded in full, so the same velocities can be reproduced."""
    if stage_id not in _VEL_STORE_KEYS:
        raise ValueError(f"run_velocity: stage must be 'vel1' or 'vel2', got {stage_id!r}")
    mo, ex = velocity_pick_params(store, stage_id, manual_overrides, excluded_cdps)
    params = {"vmin": vmin, "vmax": vmax, "n_velocities": cfg.VELOCITY_N_DEFAULT,
              "n_analysis": int(n_analysis), "min_offset": min_offset, "window_len": int(window_len),
              "min_time": min_time, "search_w": search_w, "min_conf": min_conf, "smooth": int(smooth),
              "reliability_threshold": reliability_threshold,
              "manual_overrides": mo, "excluded_cdps": ex}
    return execute(store, stage_id, params, progress_cb=progress_cb, inputs=inputs)


@_op("nmo1")
def _c_nmo1(run: Run, p: dict) -> Computed:
    dt = run.survey.sample_interval
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    vk = run.in_key("velocity", "curve")
    out, live_key = run.key("traces"), run.key("live")
    for g in gathers.values():
        g[out], g[live_key] = nmo.nmo_correction(g[ik], g[ok], g[vk], dt, stretch_mute=p["stretch_mute"])
    return {"traces": out, "live": live_key, "offsets": ok}, {}


def run_nmo1(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT,
             inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "nmo1", {"stretch_mute": stretch_mute}, inputs=inputs)


@_op("statics")
def _c_statics(run: Run, p: dict) -> Computed:
    dt = run.survey.sample_interval
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    nk, nlive = run.in_key("nmo", "traces"), run.state("nmo").keys["live"]
    shot_static, receiver_static, diag = statics_mod.solve_residual_statics(
        gathers, nk, nlive, dt, max_shift_s=p["max_shift_s"])
    out = run.key("traces")
    statics_mod.apply_residual_statics(gathers, ik, out, shot_static, receiver_static, dt)
    run.put(shot_static=shot_static, receiver_static=receiver_static, diag=diag)
    return {"traces": out, "offsets": ok}, dict(diag)


def run_statics(store: MutableMapping, max_shift_s: float = cfg.MAX_STATIC_SHIFT_S_DEFAULT,
                inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "statics", {"max_shift_s": max_shift_s}, inputs=inputs)


@_op("dmo")
def _c_dmo(run: Run, p: dict) -> Computed:
    """DMO as implemented here NMO-corrects its own input with the pass-1
    velocities first, so it needs pre-NMO gathers and velocities — it does
    not need NMO Pass 1 or residual statics to have been run."""
    dt = run.survey.sample_interval
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    vk = run.in_key("velocity", "curve")
    out, nmo_key = run.key("traces"), run.key("nmo_in")
    for g in gathers.values():
        nmo_in, _ = nmo.nmo_correction(g[ik], g[ok], g[vk], dt, stretch_mute=p["stretch_mute"])
        g[nmo_key] = nmo_in
        g[out] = nmo.dmo_correction(nmo_in, g[ok], g[vk], dt)
    return {"traces": out, "nmo_in": nmo_key, "offsets": ok}, {}


def run_dmo(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT,
            inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "dmo", {"stretch_mute": stretch_mute}, inputs=inputs)


@_op("inv_nmo")
def _c_inv_nmo(run: Run, p: dict) -> Computed:
    dt = run.survey.sample_interval
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    vk = run.in_key("velocity", "curve")
    out = run.key("traces")
    for g in gathers.values():
        g[out] = nmo.inverse_nmo_correction(g[ik], g[ok], g[vk], dt)
    return {"traces": out, "offsets": ok}, {}


def run_inverse_nmo(store: MutableMapping, inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "inv_nmo", {}, inputs=inputs)


@_op("data_reg")
def _c_data_reg(run: Run, p: dict) -> Computed:
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    out, off_out, live = run.key("traces"), run.key("offsets"), run.key("live")
    target = regularization.regularize_all_gathers(
        gathers, input_key=ik, offsets_key=ok, n_offsets=int(p["n_offsets"]),
        out_traces_key=out, out_offsets_key=off_out, out_live_key=live)
    run.put(regular_offsets=target)
    return {"traces": out, "offsets": off_out, "live": live}, {}


def run_data_reg(store: MutableMapping, n_offsets: int = 31,
                 inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "data_reg", {"n_offsets": int(n_offsets)}, inputs=inputs)


@_op("nmo_final")
def _c_nmo_final(run: Run, p: dict) -> Computed:
    """NMO on the current pre-NMO gathers with the best velocities available
    (pass 2, else pass 1)."""
    dt = run.survey.sample_interval
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    vk = run.in_key("velocity", "curve")
    reg = [s for s in graph(run.store).lineage(run.inputs["data"]) if s.op == "data_reg"]
    reg_live = reg[-1].keys["live"] if reg else None
    out, live_key = run.key("traces"), run.key("live")
    for g in gathers.values():
        traces, live = nmo.nmo_correction(g[ik], g[ok], g[vk], dt, stretch_mute=p["stretch_mute"])
        if reg_live:
            # AND with the regularisation live-mask so offsets outside a gather's real fold
            # (zero-filled by interpolation, not real data) don't inflate the stack's fold count.
            live = live & g[reg_live][:, None]
        g[out], g[live_key] = traces, live
    return ({"traces": out, "live": live_key, "offsets": ok},
            {"velocity": state_name(run.state("velocity"))})


def run_nmo_final(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT,
                  inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "nmo_final", {"stretch_mute": stretch_mute}, inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Stack and post-stack
# ─────────────────────────────────────────────────────────────────────────

@_op("stack")
def _c_stack(run: Run, p: dict) -> Computed:
    gathers = run.container()
    nmo_state = run.state()
    tk = run.in_key()
    stack_matrix, fold_matrix, naive_matrix = stacking.build_stack_section(
        gathers, list(gathers.keys()), tk, nmo_state.keys["live"])
    for g in gathers.values():            # the maths writes fixed per-gather names; two stacks would collide
        for k in ("stack_dmo", "fold", "stack_naive_mean"):
            g.pop(k, None)
    run.put(section=stack_matrix, fold=fold_matrix, naive=naive_matrix)
    return {}, {"n_cdp": len(gathers), "stacked": state_name(nmo_state)}


def run_stack(store: MutableMapping, inputs: Optional[Dict[str, str]] = None) -> DataState:
    """Stack NMO-corrected gathers (Final NMO, else NMO Pass 1). Stacking
    un-flattened gathers is a ⛔ — blocked by the dependency check, not merely
    discouraged."""
    return execute(store, "stack", {}, inputs=inputs)


@_op("rna")
def _c_rna(run: Run, p: dict) -> Computed:
    run.put(section=filtering.random_noise_attenuation(run.section(), p["spatial_window"]))
    return {}, {}


def run_rna(store: MutableMapping, spatial_window: int = cfg.RNA_SPATIAL_WINDOW_DEFAULT,
            inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "rna", {"spatial_window": spatial_window}, inputs=inputs)


@_op("decon_post")
def _c_decon_post(run: Run, p: dict) -> Computed:
    run.put(section=preprocessing.spiking_decon_gather(run.section(), p["operator_length"],
                                                       p["pre_whitening"]))
    return {}, {}


def run_decon_post(store: MutableMapping, operator_length: int = cfg.DECON_AFTER_OPLEN_DEFAULT,
                   pre_whitening: float = cfg.DECON_AFTER_PW_DEFAULT,
                   inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "decon_post", {"operator_length": operator_length,
                                         "pre_whitening": pre_whitening}, inputs=inputs)


@_op("tvf")
def _c_tvf(run: Run, p: dict) -> Computed:
    run.put(section=filtering.time_variant_filter(run.section(), run.survey.sample_interval,
                                                  p["zones"], p["taper_s"]))
    return {}, {"zones": len(p["zones"])}


def run_tvf(store: MutableMapping, inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "tvf", {"zones": cfg.TVF_ZONES_DEFAULT, "taper_s": cfg.TVF_TAPER_S_DEFAULT},
                   inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Migration (two implementations) and final scaling
# ─────────────────────────────────────────────────────────────────────────

def _migration_velocity(run: Run) -> np.ndarray:
    vstate = run.state("velocity")
    vfield = run.blob("velocity").get("field")
    validate_velocity_field(vfield, run.survey.num_samples, len(run.container("velocity")),
                            what=f"{stage_label(vstate.op)} velocity field")
    return vfield


@_op("migration:poststack")
def _c_migration_poststack(run: Run, p: dict) -> Computed:
    survey = run.survey
    section = run.section()
    cdp_list = list(run.container("velocity").keys())
    vfield = _migration_velocity(run)
    aperture_traces = max(1, int(round(p["aperture_m"] / survey.group_interval)))
    x_positions = np.array(cdp_list, dtype=float) * survey.group_interval
    run.put(section=migration.kirchhoff_migration(section, x_positions, survey.twt, vfield,
                                                  aperture_traces))
    return {}, {"domain": "post-stack", "velocity": state_name(run.state("velocity"))}


def run_migration_poststack(store: MutableMapping, aperture_m: float = cfg.APERTURE_M_DEFAULT,
                            inputs: Optional[Dict[str, str]] = None) -> DataState:
    """Post-stack Kirchhoff time migration: needs a stacked section and a velocity
    field (pass 2 if available, else pass 1)."""
    return execute(store, "migration", {"aperture_m": aperture_m}, variant="poststack", inputs=inputs)


@_op("migration:pstm")
def _c_migration_pstm(run: Run, p: dict) -> Computed:
    survey = run.survey
    gathers, ik, ok = run.container(), run.in_key(), run.off_key()
    vfield = _migration_velocity(run)
    aperture_traces = max(1, int(round(p["aperture_m"] / survey.group_interval)))
    migrated, fold = migration.prestack_kirchhoff_time_migration(
        gathers, list(gathers.keys()), ik, survey.twt, vfield, survey.group_interval, aperture_traces,
        offsets_key=ok, cdp_stride=int(p["cdp_stride"]), progress_cb=run.progress_cb)
    run.put(section=migrated, fold=fold)
    return {}, {"domain": "pre-stack (PSTM)", "velocity": state_name(run.state("velocity"))}


def run_migration_pstm(store: MutableMapping, aperture_m: float = cfg.APERTURE_M_DEFAULT,
                       cdp_stride: int = 3, progress_cb: ProgressCb = None,
                       inputs: Optional[Dict[str, str]] = None) -> DataState:
    """Pre-stack Kirchhoff time migration: needs pre-stack, offset-preserving CDP
    gathers (NOT NMO-corrected, NOT stacked) and a velocity field. It performs
    migration and stack in one pass, so no stack is required — and
    regularisation is recommended, not required."""
    return execute(store, "migration", {"aperture_m": aperture_m, "cdp_stride": int(cdp_stride)},
                   variant="pstm", progress_cb=progress_cb, inputs=inputs)


@_op("final_scale")
def _c_final_scale(run: Run, p: dict) -> Computed:
    run.put(section=migration.final_scaling(run.section(), run.survey.sample_interval,
                                            window=p["window"], t_top_mute=p["t_top_mute"]))
    return {}, {}


def run_final_scale(store: MutableMapping, window: int = cfg.FINAL_AGC_WINDOW_DEFAULT,
                    inputs: Optional[Dict[str, str]] = None) -> DataState:
    return execute(store, "final_scale", {"window": window, "t_top_mute": cfg.MUTE_TTOP_DEFAULT},
                   inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Reading results back
# ─────────────────────────────────────────────────────────────────────────

_SECTION_LABELS = {"stack": "Stack", "rna": "Post-stack (after random noise attenuation)",
                   "decon_post": "Post-stack (after deconvolution)", "tvf": "Post-stack (pre-migration)"}


def best_available_state(store: MutableMapping) -> Tuple[Optional[DataState], Optional[str]]:
    """The most advanced section-like state: final-scaled > migrated > the
    current stacked/post-stack section. It is returned even if it is out of
    date (see `dependencies.stale_info`) — pages show that warning rather
    than silently substituting an older result."""
    g = graph(store)
    for kind, label in (("final", "Final processed section"), ("image", "Migrated section")):
        s = g.current(kind)
        if s is not None and s.materialized and section_of(store, s) is not None:
            return s, label
    s = g.current("section")
    if s is not None and s.materialized and section_of(store, s) is not None:
        return s, _SECTION_LABELS.get(s.op, "Stack")
    return None, None


def best_available_section(store: MutableMapping) -> Tuple[Optional[np.ndarray], Optional[str]]:
    s, label = best_available_state(store)
    return (section_of(store, s), label) if s is not None else (None, None)
