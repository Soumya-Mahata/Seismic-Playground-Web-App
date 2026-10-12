"""
modules/action_log.py — Processing history as an immutable chronological action log
=====================================================================================
Three things a person can ask about their processing session are kept apart:

  CURRENT PROCESSING STATE   what the data is *now*              (`dependencies.current_processing_state`)
  RECOMMENDED WORKFLOW       what is *usually* done, in what order (`pipeline.PIPELINE_STAGES`)
  PROCESSING HISTORY         what the person actually *did*        (this module)

This module is only the third. It is an **append-only record of actions in the
exact order they were performed**. Nothing in it is ever reordered, edited,
merged or removed:

  * every execution is its own entry — running Band-pass twice is two entries,
    even if the second one found an identical, already-computed state and re-used it;
  * a failed or blocked attempt is an entry too (status "failure", with the reason);
  * undo / redo / make-current / discard / free-memory are entries (they change what
    the current state is, so the history has to say so);
  * discarding a state does not delete the entries that created or used it — a later
    "discard" entry records that it happened (`ActionLog.discarded_by`);
  * the recommended order of the seismic workflow never influences the order here.

The order is the action number `n` (1, 2, 3 …), not the clock — timestamps can tie.

Immutability
------------
`Action` is a frozen dataclass of plain immutable values (strings, numbers, tuples;
parameters and QC are stored as canonical JSON text), and `ActionLog` has no method
that changes or removes an entry. Python cannot make that impossible to bypass, so the
log is also **hash-chained**: each entry's hash covers its content and the previous
entry's hash, and `ActionLog.verify()` reports any entry that was altered, removed,
inserted or reordered.

Sessions
--------
Loading new data starts a new *dataset session*: state ids restart at S0 for the new
data, so each entry records its `session`. Earlier sessions' entries stay in the log.

Display order (newest- or oldest-first) is chosen by the reader with `view(order=…)`,
which returns a sorted *copy*; the underlying order is never touched.

No Streamlit dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from modules.canonical import jsonable

GENESIS = "0" * 16

# what kind of action an entry records
OPERATION = "operation"          # a processing operation (incl. loading data, building gathers)
NAVIGATION = ("undo", "redo", "select")      # moves the "current" pointer; no new data
MANAGEMENT = ("discard", "release")          # removes a state / frees its arrays
KINDS: Tuple[str, ...] = (OPERATION,) + NAVIGATION + MANAGEMENT

SUCCESS, FAILURE = "success", "failure"
ORDERS = ("oldest", "newest")
ORDER_LABELS = {"oldest": "Oldest first (execution order)", "newest": "Newest first"}


def _dumps(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Action:
    """One thing the person did, exactly as it happened. Immutable."""
    n: int                              # action number: 1, 2, 3 … in the order performed
    timestamp: str
    session: int                        # dataset session (a new one starts when new data is loaded)
    kind: str                           # operation | undo | redo | select | discard | release
    action: str                         # stage id for operations ("agc"), else the kind
    label: str                          # "Band-Pass Filter", "Migration (pstm)", "Undo (shot data)"
    variant: str
    params_json: str                    # complete parameters (canonical JSON); "{}" when there are none
    inputs: Tuple[Tuple[str, str], ...]  # ((role, input state id), …); for navigation: the state left
    output: Optional[str]               # the state produced / re-used / moved to; None on failure
    status: str                         # success | failure
    reused: bool                        # success, but an identical state already existed (not recomputed)
    failure_kind: str                   # "" | blocked | invalid | error
    error: str                          # why it failed
    qc_json: str                        # relevant QC summary (canonical JSON)
    warnings: Tuple[str, ...]           # the ⚠ recommendations the person went ahead despite
    affected: Tuple[str, ...]           # discard: every state removed
    detail: str                         # one human sentence
    prev_hash: str
    hash: str

    # ---- readable views of the stored values ------------------------------------
    @property
    def params(self) -> Dict[str, Any]:
        return json.loads(self.params_json)

    @property
    def qc(self) -> Dict[str, Any]:
        return json.loads(self.qc_json)

    @property
    def inputs_dict(self) -> Dict[str, str]:
        return dict(self.inputs)

    @property
    def input_state(self) -> Optional[str]:
        """The state this action was applied to (its `data` input, else the first one)."""
        d = self.inputs_dict
        return d.get("data") or (self.inputs[0][1] if self.inputs else None)

    @property
    def ok(self) -> bool:
        return self.status == SUCCESS

    @property
    def is_operation(self) -> bool:
        return self.kind == OPERATION

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "timestamp": self.timestamp, "session": self.session, "kind": self.kind,
                "action": self.action, "label": self.label, "variant": self.variant,
                "params": self.params, "inputs": self.inputs_dict, "input_state": self.input_state,
                "output_state": self.output, "status": self.status, "reused": self.reused,
                "failure_kind": self.failure_kind, "error": self.error, "qc": self.qc,
                "warnings": list(self.warnings), "affected": list(self.affected),
                "detail": self.detail, "prev_hash": self.prev_hash, "hash": self.hash}


def _hash_entry(e: Action) -> str:
    body = json.dumps(
        [e.prev_hash, e.n, e.timestamp, e.session, e.kind, e.action, e.label, e.variant, e.params_json,
         [list(p) for p in e.inputs], e.output, e.status, e.reused, e.failure_kind, e.error, e.qc_json,
         list(e.warnings), list(e.affected), e.detail], separators=(",", ":"))
    return hashlib.sha1(body.encode()).hexdigest()[:16]


class ActionLog:
    """Append-only chronological log. The only way in is `record`; there is no
    way to change, reorder or remove an entry, and `verify()` proves it."""

    def __init__(self) -> None:
        self._entries: List[Action] = []
        self.session: int = 1

    # ---- writing ---------------------------------------------------------------------
    def record(self, kind: str, action: str, *, label: str = "", variant: str = "",
               params: Optional[Mapping[str, Any]] = None, inputs: Optional[Mapping[str, str]] = None,
               output: Optional[str] = None, status: str = SUCCESS, reused: bool = False,
               failure_kind: str = "", error: str = "", qc: Optional[Mapping[str, Any]] = None,
               warnings: Sequence[str] = (), affected: Sequence[str] = (), detail: str = "") -> Action:
        if kind not in KINDS:
            raise ValueError(f"Unknown action kind '{kind}'.")
        if status not in (SUCCESS, FAILURE):
            raise ValueError(f"Unknown action status '{status}'.")
        prev = self._entries[-1].hash if self._entries else GENESIS
        entry = Action(
            n=len(self._entries) + 1, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session=self.session, kind=kind, action=action, label=label or action, variant=variant or "",
            params_json=_dumps(params or {}),
            inputs=tuple((str(r), str(s)) for r, s in (inputs or {}).items() if s),
            output=output, status=status, reused=bool(reused), failure_kind=failure_kind, error=error,
            qc_json=_dumps(qc or {}), warnings=tuple(str(w) for w in warnings),
            affected=tuple(affected), detail=detail, prev_hash=prev, hash="")
        entry = Action(**{**entry.__dict__, "hash": _hash_entry(entry)})
        self._entries.append(entry)
        return entry

    def new_session(self) -> int:
        """Start a new dataset session (new data was loaded). A session with no
        entries yet is not left behind empty."""
        if any(e.session == self.session for e in self._entries):
            self.session += 1
        return self.session

    # ---- reading -----------------------------------------------------------------------
    def entries(self) -> Tuple[Action, ...]:
        """Every entry, in the exact order the actions were performed."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Action]:
        return iter(tuple(self._entries))

    def __getitem__(self, n: int) -> Action:
        """The entry with action number `n` (1-based)."""
        if not 1 <= n <= len(self._entries):
            raise KeyError(f"No action number {n}.")
        return self._entries[n - 1]

    def operations(self, session: Optional[int] = None) -> List[Action]:
        """Processing operations only (successful or not), in execution order."""
        return [e for e in self._entries if e.is_operation and (session is None or e.session == session)]

    def last(self, session: Optional[int] = None, kind: Optional[str] = None) -> Optional[Action]:
        for e in reversed(self._entries):
            if (session is None or e.session == session) and (kind is None or e.kind == kind):
                return e
        return None

    def view(self, order: str = "oldest", *, kinds: Optional[Sequence[str]] = None,
             session: Optional[int] = None, issues_only: bool = False) -> List[Action]:
        """A *copy* of the log for display, oldest- or newest-first, optionally filtered.
        Sorting is by action number; the log itself is never reordered."""
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
        rows = [e for e in self._entries
                if (kinds is None or e.kind in kinds)
                and (session is None or e.session == session)
                and (not issues_only or e.status == FAILURE or e.warnings)]
        return sorted(rows, key=lambda e: e.n, reverse=(order == "newest"))

    def discarded_by(self, entry: Action) -> Optional[Action]:
        """The later action that removed the state this entry produced, if any."""
        if not entry.output:
            return None
        for e in self._entries:
            if (e.kind == "discard" and e.session == entry.session and e.n > entry.n
                    and entry.output in e.affected):
                return e
        return None

    def sessions(self) -> List[int]:
        return sorted({e.session for e in self._entries})

    # ---- integrity ----------------------------------------------------------------------
    def verify(self) -> List[str]:
        """Problems found when re-checking the hash chain (empty list = intact):
        an entry whose content changed, a gap or reorder in the numbering, a
        broken link between entries."""
        problems: List[str] = []
        prev = GENESIS
        for i, e in enumerate(self._entries):
            if e.n != i + 1:
                problems.append(f"position {i + 1} holds action number {e.n} (entries were removed, "
                                f"inserted or reordered)")
            if e.prev_hash != prev:
                problems.append(f"action {e.n}: the chain to the previous entry is broken")
            if _hash_entry(e) != e.hash:
                problems.append(f"action {e.n}: its content was changed after it was recorded")
            prev = e.hash
        return problems

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]


# ─────────────────────────────────────────────────────────────────────────
# Text helpers (shared by the UI, the report and the CSV export)
# ─────────────────────────────────────────────────────────────────────────

def format_params(params: Mapping[str, Any], limit: int = 90) -> str:
    """'window=500, mode=rms' — compact, for tables."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        elif isinstance(v, (list, dict)):
            v = f"[{len(v)}]"
        parts.append(f"{k}={v}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _g(x: Any) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def format_qc(qc: Mapping[str, Any], limit: int = 140) -> str:
    """'RMS 1.2 → 0.41 (×0.34) · 0 non-finite · traces_killed=12'."""
    parts: List[str] = []
    qc = dict(qc)
    rin, rout = qc.pop("rms_in", None), qc.pop("rms_out", None)
    if rout is not None:
        if rin:
            parts.append(f"RMS {_g(rin)} → {_g(rout)} (×{_g(rout / rin)})")
        else:
            parts.append(f"RMS {_g(rout)}")
    nf = qc.pop("non_finite", None)
    if nf is not None:
        parts.append("⚠ " + f"{nf} non-finite sample(s)" if nf else "all finite")
    for k, v in qc.items():
        if isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={_g(v)}")
    text = " · ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def result_text(e: Action) -> str:
    """'✓ success' / '♻ re-used S3' / '✗ blocked'."""
    if e.status == FAILURE:
        return f"✗ {e.failure_kind or 'failed'}"
    if e.reused:
        return f"♻ re-used {e.output}"
    return "✓ success"


def one_line(e: Action) -> str:
    """'#7 ✓ AGC  S3 → S4' — for the sidebar."""
    arrow = f"{e.input_state or '—'} → {e.output or '—'}" if e.is_operation or e.output else ""
    mark = "✗" if e.status == FAILURE else ("♻" if e.reused else "✓")
    return f"#{e.n} {mark} {e.label}  {arrow}".rstrip()


def report_lines(entries: Sequence[Action]) -> List[str]:
    """The log as plain text, one block per action, in the order given (callers pass the
    chronological order for files; the report always does)."""
    lines: List[str] = []
    for e in entries:
        lines.append(f"#{e.n:<3} {e.timestamp}  {result_text(e):<14} {e.label}"
                     + (f"   [dataset {e.session}]" if e.session > 1 else ""))
        if e.inputs or e.output:
            ins = ", ".join(f"{r}={s}" for r, s in e.inputs) or "—"
            lines.append(f"       input: {ins}   →   output: {e.output or '—'}")
        params = e.params
        if params:
            lines.append(f"       parameters: {format_params(params, limit=400)}")
        if e.qc:
            lines.append(f"       QC: {format_qc(e.qc, limit=400)}")
        for w in e.warnings:
            lines.append(f"       ⚠ {w}")
        if e.status == FAILURE:
            lines.append(f"       ✗ {e.failure_kind}: {e.error}")
        if e.detail and e.status != FAILURE:
            lines.append(f"       {e.detail}")
    return lines
