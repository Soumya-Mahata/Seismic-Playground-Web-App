"""
modules/state_graph.py — Data states:  INPUT STATE → OPERATION + PARAMETERS → OUTPUT STATE
==========================================================================================
The application used to remember *which stages had been run*. It now
remembers *which data states exist and how each one was made*:

        S1 ──AGC(window=500)──▶ S2 ──Band-pass(8–60 Hz)──▶ S3 ──Mute(t_top=0.1)──▶ S4

Every processing operation consumes one or more existing states (its
inputs) and produces exactly one new state. States are never edited and a
re-run never overwrites anything: running AGC again simply creates another
state. That one rule is what gives order independence, repeated and
skipped operations, a lineage to query for downstream/stale detection, a
complete history, cheap undo/redo (move a pointer) and reproducibility (the
recorded operations *are* the recipe).

This module is deliberately the whole "graph framework": a dict of small
records plus a few queries. It is pure Python — no Streamlit, no arrays.
Seismic payloads live in the data store next to it (see
`modules/stage_ops.py`); a state only records *where* (`keys`, `root`).

Identity
--------
  id           "S7"    Stable, unique within a graph, never reused (even after a discard).
  fingerprint  12 hex  Content-addressed LINEAGE hash: sha1 of the operation, its complete
                       parameters and the fingerprints of its input states. The same
                       operation with the same parameters on equivalent inputs has the
                       same fingerprint in any session — which is what makes a run
                       reproducible and lets an identical re-run reuse a state instead of
                       recomputing and storing a second copy.
  version      int     1 for the first state made by an operation, 2 for the next …

"Current"
---------
`current(kind)` is the most recently produced-or-selected state of a kind
(`touch` is a logical clock). Undo/redo and "make current" only move that
pointer; no data is touched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from modules.action_log import ActionLog
from modules.canonical import array_digest, canonical, jsonable
from modules.registry import KINDS as _REGISTRY_KINDS

__all__ = ["KINDS", "GraphError", "DataState", "ProcessingGraph", "make_fingerprint",
           "array_digest", "canonical", "jsonable"]      # the last three live in modules/canonical.py

# Kinds of data a state can hold (what a downstream step can ask for) — declared in the registry.
KINDS: Tuple[str, ...] = _REGISTRY_KINDS


class GraphError(KeyError):
    """An unknown state id, or an invalid graph operation."""

    def __str__(self) -> str:                     # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


# ─────────────────────────────────────────────────────────────────────────
# Canonical parameters and fingerprints  (the helpers live in modules/canonical.py)
# ─────────────────────────────────────────────────────────────────────────

def make_fingerprint(op: str, variant: str, params: Mapping[str, Any],
                     input_fingerprints: Mapping[str, str]) -> str:
    payload = json.dumps([op, variant or "", canonical(params), sorted(input_fingerprints.items())],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────
# The state record
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DataState:
    """One immutable-in-spirit data state and the operation that made it.

    Only `touch` (the "current" clock), `materialized` (is its payload held
    in memory right now), `keys`/`info` (payload locators and diagnostics)
    and `notes` change after creation."""
    id: str
    kind: str
    op: str                                  # the operation (stage id) that produced it
    variant: str = ""
    params: Dict[str, Any] = field(default_factory=dict)       # complete, canonical, reproducible
    inputs: Dict[str, str] = field(default_factory=dict)       # role -> input state id
    fingerprint: str = ""
    version: int = 1                         # n-th state produced by (op, variant)
    seq: int = 0                             # creation order, 1-based
    timestamp: str = ""
    notes: str = ""                          # e.g. the warnings the run went ahead despite
    root: Optional[str] = None               # id of the geometry root whose gather container holds it
    keys: Dict[str, str] = field(default_factory=dict)         # payload locator inside the container
    info: Dict[str, Any] = field(default_factory=dict)         # diagnostics; NOT part of identity
    touch: int = 0
    materialized: bool = True

    @property
    def is_root(self) -> bool:
        return self.root == self.id

    def op_key(self) -> str:
        return f"{self.op}:{self.variant}" if self.variant else self.op


# ─────────────────────────────────────────────────────────────────────────
# The graph
# ─────────────────────────────────────────────────────────────────────────

class ProcessingGraph:
    """All data states of one session, in creation order, plus a pointer per
    kind. Plain Python data: it lives happily in `st.session_state`, can be
    pickled, and has no references to the arrays."""

    def __init__(self) -> None:
        self._states: Dict[str, DataState] = {}
        self._counter = 0                    # next id number; never decremented
        self._clock = 0                      # logical time for `touch`
        self._redo: Dict[str, List[str]] = {}
        # The processing HISTORY: an append-only, immutable record of what the person did, in the
        # order they did it (modules/action_log.py). It is not derived from the states — states
        # can be discarded, history cannot — and it survives `reset()` (a new dataset session).
        self.actions: ActionLog = ActionLog()
        # Technical trace of things the *system* did (re-computing a released state, freeing
        # memory to stay in budget). Not history; may be trimmed.
        self.system_events: List[Dict[str, Any]] = []

    # ---- container protocol ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self) -> Iterator[DataState]:
        return iter(sorted(self._states.values(), key=lambda s: s.seq))

    def __contains__(self, sid: object) -> bool:
        return sid in self._states

    def __getitem__(self, sid: str) -> DataState:
        try:
            return self._states[sid]
        except KeyError:
            raise GraphError(f"No data state '{sid}' (it may have been discarded).") from None

    def get(self, sid: Optional[str]) -> Optional[DataState]:
        return self._states.get(sid) if sid else None

    def states(self) -> List[DataState]:
        return list(self)

    # ---- creating states ---------------------------------------------------------
    def peek_id(self) -> str:
        """The id the next state will get (so payload keys can be named before
        the computation finishes)."""
        return f"S{self._counter}"

    def fingerprint_for(self, op: str, variant: str, params: Mapping[str, Any],
                        inputs: Mapping[str, str]) -> str:
        return make_fingerprint(op, variant, params,
                                {role: self[sid].fingerprint for role, sid in inputs.items()})

    def find(self, fingerprint: str) -> Optional[DataState]:
        """The state with this lineage fingerprint, if one exists (the most
        recently touched when there are several)."""
        hits = [s for s in self._states.values() if s.fingerprint == fingerprint]
        return max(hits, key=lambda s: s.touch) if hits else None

    def add(self, *, kind: str, op: str, variant: str = "", params: Optional[Mapping[str, Any]] = None,
            inputs: Optional[Mapping[str, str]] = None, root: Optional[str] = None,
            keys: Optional[Mapping[str, str]] = None, info: Optional[Mapping[str, Any]] = None,
            notes: str = "", sid: Optional[str] = None) -> DataState:
        if kind not in KINDS:
            raise ValueError(f"Unknown data kind '{kind}'.")
        inputs = dict(inputs or {})
        for role, src in inputs.items():
            if src not in self._states:
                raise GraphError(f"Input '{role}' of {op} is state '{src}', which does not exist.")
        expected = self.peek_id()
        if sid is not None and sid != expected:
            raise GraphError(f"State id '{sid}' was reserved but '{expected}' is next.")
        sid = expected
        params = canonical(params or {})
        self._counter += 1
        self._clock += 1
        state = DataState(
            id=sid, kind=kind, op=op, variant=variant or "", params=params, inputs=inputs,
            fingerprint=self.fingerprint_for(op, variant or "", params, inputs),
            version=1 + sum(1 for s in self._states.values() if s.op == op and s.variant == (variant or "")),
            seq=self._counter, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes, root=sid if root == "self" else root, keys=dict(keys or {}),
            info=dict(info or {}), touch=self._clock)
        self._states[sid] = state
        self._redo.pop(kind, None)           # a new state of this kind ends the redo history
        return state

    # ---- "current" pointers ----------------------------------------------------------
    def of_kind(self, kind: str) -> List[DataState]:
        """States of a kind, most recently current first."""
        return sorted((s for s in self._states.values() if s.kind == kind), key=lambda s: -s.touch)

    def of_op(self, op: str, variant: Optional[str] = None) -> List[DataState]:
        return sorted((s for s in self._states.values()
                       if s.op == op and (variant is None or s.variant == variant)),
                      key=lambda s: -s.touch)

    def current(self, kind: str) -> Optional[DataState]:
        states = self.of_kind(kind)
        return states[0] if states else None

    def latest(self, op: str, variant: Optional[str] = None) -> Optional[DataState]:
        states = self.of_op(op, variant)
        return states[0] if states else None

    def _touch(self, state: DataState) -> None:
        self._clock += 1
        state.touch = self._clock

    def select(self, sid: str) -> DataState:
        """Make a state the current one of its kind (a deliberate choice, so
        it also ends that kind's redo history)."""
        state = self[sid]
        self._touch(state)
        self._redo.pop(state.kind, None)
        return state

    # ---- undo / redo ---------------------------------------------------------------------
    def undo_target(self, kind: str) -> Optional[DataState]:
        """Where `undo(kind)` would go: the state the current one was made from,
        if that is data of the same kind (i.e. un-apply the last operation)."""
        cur = self.current(kind)
        parent = self.get(cur.inputs.get("data")) if cur else None
        return parent if parent is not None and parent.kind == kind else None

    def can_undo(self, kind: str) -> bool:
        return self.undo_target(kind) is not None

    def can_redo(self, kind: str) -> bool:
        return any(s in self._states for s in self._redo.get(kind, []))

    def undo(self, kind: str) -> Optional[DataState]:
        """Step the current pointer of `kind` back to the state it was made
        from. Nothing is deleted: the undone state stays in the graph (and
        in `redo`)."""
        cur = self.current(kind)
        target = self.undo_target(kind)
        if cur is None or target is None:
            return None
        self._redo.setdefault(kind, []).append(cur.id)
        self._touch(target)
        return target

    def redo_target(self, kind: str) -> Optional[DataState]:
        """Where `redo(kind)` would go (None if there is nothing to redo)."""
        return next((self._states[i] for i in reversed(self._redo.get(kind, [])) if i in self._states),
                    None)

    def redo(self, kind: str) -> Optional[DataState]:
        stack = self._redo.get(kind, [])
        while stack:
            sid = stack.pop()
            if sid in self._states:
                self._touch(self._states[sid])
                return self._states[sid]
        return None

    # ---- lineage ---------------------------------------------------------------------------
    def lineage(self, sid: Optional[str]) -> List[DataState]:
        """The chain of states the data in `sid` was built through (following
        each state's `data` input), oldest first, ending with `sid`."""
        out: List[DataState] = []
        seen: Set[str] = set()
        cur = self.get(sid)
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            out.append(cur)
            cur = self.get(cur.inputs.get("data"))
        return list(reversed(out))

    def ancestors(self, sid: str) -> List[DataState]:
        """Every state `sid` was built from, through any input role, in
        creation order (excluding `sid`)."""
        found: Set[str] = set()
        stack = [sid]
        while stack:
            for src in self[stack.pop()].inputs.values():
                if src in self._states and src not in found:
                    found.add(src)
                    stack.append(src)
        return [s for s in self if s.id in found]

    def dependents(self, sid: str) -> List[DataState]:
        """Every state built, directly or indirectly, from `sid` (any input
        role), in creation order."""
        self[sid]
        found: Set[str] = set()
        grew = True
        while grew:
            grew = False
            for s in self._states.values():
                if s.id in found or s.id == sid:
                    continue
                if any(src == sid or src in found for src in s.inputs.values()):
                    found.add(s.id)
                    grew = True
        return [s for s in self if s.id in found]

    def is_ancestor(self, older: str, newer: str) -> bool:
        return any(a.id == older for a in self.ancestors(newer))

    # ---- removal -------------------------------------------------------------------------------------
    def discard(self, sid: str) -> List[str]:
        """Remove a state and everything built on it. Returns the removed ids
        (dependents first, so a caller can free payloads in a safe order).
        Ids are not reused."""
        victims = [d.id for d in self.dependents(sid)] + [sid]
        victims = sorted(set(victims), key=lambda i: -self._states[i].seq)
        for v in victims:
            del self._states[v]
        for kind in list(self._redo):
            self._redo[kind] = [i for i in self._redo[kind] if i in self._states]
        return victims

    def reset(self) -> List[str]:
        """Forget every state (new data was loaded). Ids restart at S0. The action log is
        NOT cleared — it moves on to a new dataset session, so earlier entries stay."""
        removed = [s.id for s in self]
        self._states.clear()
        self._counter = 0
        self._clock = 0
        self._redo.clear()
        self.system_events.clear()
        self.actions.new_session()
        return removed

    # ---- system trace & reproducibility ---------------------------------------------------------------------
    def note_system(self, event: str, ids: Sequence[str], text: str) -> None:
        """Note something the *system* did on its own (a payload re-computed, memory freed
        to stay in budget). This is a technical trace, not the processing history."""
        self.system_events.append({"event": event, "states": list(ids), "text": text,
                                   "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        del self.system_events[:-500]

    def recipe(self, sid: str) -> List[Dict[str, Any]]:
        """The operations that produce state `sid` from the loaded data, in
        an order they can be replayed (`stage_ops.replay_recipe`). Each step
        names its inputs by the ids used *in this recipe*."""
        chain = self.ancestors(sid) + [self[sid]]
        return [{"id": s.id, "kind": s.kind, "op": s.op, "variant": s.variant, "params": s.params,
                 "inputs": dict(s.inputs), "fingerprint": s.fingerprint} for s in chain]

    def export(self) -> Dict[str, Any]:
        """A JSON-able snapshot: the data states, the current pointers, and the action log
        in chronological order (no arrays)."""
        return {
            "states": [{"id": s.id, "kind": s.kind, "op": s.op, "variant": s.variant,
                        "version": s.version, "params": s.params, "inputs": dict(s.inputs),
                        "fingerprint": s.fingerprint, "seq": s.seq, "timestamp": s.timestamp,
                        "notes": s.notes, "info": jsonable(s.info)} for s in self],
            "current": {k: (self.current(k).id if self.current(k) else None) for k in KINDS},
            "actions": self.actions.to_dicts(),
            "action_log_problems": self.actions.verify(),
        }
