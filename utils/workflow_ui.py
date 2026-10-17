"""
utils/workflow_ui.py — Presentation of workflow readiness and data states
============================================================================
Requires Streamlit. Renders what `modules/dependencies` decides, using three
distinct statuses so a person can tell a recommendation from a requirement at
a glance:

    ✓  Available                              runnable, nothing to flag
    ⚠  Recommended prerequisite not performed runnable — asks "Continue?"
    ⛔ Required input missing                  not runnable; says what, and where to get it

A page shows an operation's status with `show_status()` (banner + the data
state it will read + an input chooser) and runs it with `run_button()`, which
only interposes a confirmation when there are ⚠ warnings — and never blocks on
them. `execute()` runs it and reports which state it produced.

The rest of this file shows the *data states* (the table of states, undo / redo,
"make current", free memory, discard) and — kept strictly apart from them and from
the recommended workflow — the **current processing state** and the **processing
history**, which is the immutable chronological action log of what was done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from modules import stage_ops
from modules.action_log import (FAILURE, ORDER_LABELS, ORDERS, Action, format_params, format_qc,
                                one_line, report_lines, result_text)
from modules.dependencies import (Level, StageCheck, check_all, current_processing_state,
                                  describe_lineage, describe_stream, stale_info, stale_notes, state_name)
from modules import registry
from modules.pipeline import PIPELINE_STAGES, STAGE_GROUPS, STAGE_KIND, stage_label
from modules.state_graph import DataState
from modules.validation import ValidationError
from utils import session_state as ss

LEGEND_MD = (
    "**✓ Available** — can run now.  \n"
    "**⚠ Recommended prerequisite not performed** — can still run; you'll be asked to confirm.  \n"
    "**⛔ Required input missing** — cannot run until the input it names exists.  \n"
    "**✅ done** — already run. Running a step again makes a *new* data state; nothing is overwritten."
)


def legend() -> None:
    st.caption(LEGEND_MD)


# ---- small formatting helpers ---------------------------------------------------------------------

def params_summary(params: dict, limit: int = 90) -> str:
    """'window=500, mode=rms' — compact, for tables and captions."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        elif isinstance(v, (list, dict)):
            v = f"[{len(v)}]"
        parts.append(f"{k}={v}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def input_name(chk: StageCheck, role: str = "data") -> str:
    """'Band-pass (S3)' — the state an operation reads (or read) for `role`."""
    return state_name(chk.state(role))


def input_key(chk: StageCheck, role: str = "data", name: str = "traces") -> str:
    """The key (inside each gather) of the array the operation would read for `role`."""
    return chk.state(role).keys[name]


def input_gathers(chk: StageCheck, role: str = "data") -> Tuple[dict, str, str]:
    """(gathers, trace key, offsets key) of the gather-based state `role` resolves to.
    Use these (not the current-data alias) when previewing: the person may have
    chosen an input state that lives on a different set of gathers."""
    state = chk.state(role)
    return stage_ops.gathers_of(st.session_state, state), state.keys["traces"], state.keys.get(
        "offsets", "offsets")


@dataclass
class Result:
    """What a page needs to display one finished operation: the state it
    produced, the state it consumed, and where their arrays are."""
    state: DataState
    src: Optional[DataState]
    gathers: Optional[dict]
    n_states: int = 1                 # how many states this operation has produced

    @property
    def in_key(self) -> str:
        return self.src.keys.get("traces", "") if self.src is not None else ""

    @property
    def out_key(self) -> str:
        return self.state.keys.get("traces", "")

    def key(self, name: str) -> str:
        """Key of one of this state's own arrays (e.g. 'spike_mask', 'live')."""
        return self.state.keys[name]

    @property
    def before_label(self) -> str:
        """'Before — Band-pass (S3)': what this run actually read."""
        return f"Before — {state_name(self.src)}"

    @property
    def name(self) -> str:
        return state_name(self.state)

    @property
    def params(self) -> dict:
        return self.state.params

    @property
    def info(self) -> dict:
        return self.state.info


def result(stage_id: str, variant: Optional[str] = None) -> Optional[Result]:
    """The latest state this operation produced, ready to display; None if it
    was never run — or if the arrays needed to draw it (its own, and those of
    the state it was made from, which the before/after panels show) were
    released to save memory. In that case a note with a "Re-compute" button is
    shown instead: freeing memory would be pointless if merely drawing the page
    quietly re-computed it."""
    state = ss.latest(stage_id, variant)
    if state is None:
        return None
    needed = [state] + [s for s in (ss.source(state),) if s is not None]
    released = [s for s in needed if not s.materialized]
    if released:
        names = " and ".join(f"**{state_name(s)}**" for s in released)
        st.info(f"🧊 {names}: arrays released to save memory. They are re-computed from the recorded "
                f"operation and parameters whenever they are needed (undo / redo, making a state "
                f"current, using it as an input) — or now:", icon="💾")
        if st.button("Re-compute and show", key=f"recompute_{state.id}_{stage_id}"):
            for s in released:
                stage_ops.materialize(st.session_state, s.id)
            st.rerun()
        return None
    state = ss.result(stage_id, variant)
    try:
        gathers = stage_ops.gathers_of(st.session_state, state) if state.keys else None
    except ValidationError:
        gathers = None
    return Result(state, ss.source(state), gathers, len(ss.graph().of_op(stage_id, variant)))


def result_note(res: Result) -> None:
    """One caption saying which state is shown, when the step has produced several."""
    if res.n_states > 1:
        st.caption(f"Showing **{res.name}** — the latest of {res.n_states} states this step has produced "
                   f"(use *Data states* above to look at another).")


# ---- one-shot message after an action --------------------------------------------------------

def _flash(message: str) -> None:
    st.session_state["_flash"] = message


def render_flash() -> None:
    """Show (once) the message left by the last `execute` — what state it
    produced and which state it consumed."""
    msg = st.session_state.get("_flash")
    if msg:
        st.success(msg)
        st.session_state["_flash"] = None


# ---- single-operation status -------------------------------------------------------------------------

def render_banner(chk: StageCheck) -> None:
    if chk.level == Level.BLOCKED:
        lines = "\n".join(f"- {i.message}" + (f"  \n  → *{i.fix}*" if i.fix else "")
                          for i in chk.blockers)
        st.error(f"**⛔ {Level.BLOCKED.label}**\n\n{lines}")
        if chk.warnings:
            st.caption("Also noted: " + " ".join(w.message for w in chk.warnings))
        return
    if chk.level == Level.WARNING:
        lines = "\n".join(f"- {i.message}" for i in chk.warnings)
        st.warning(f"**⚠ {Level.WARNING.label}** — you can still run this step.\n\n{lines}")
    else:
        st.caption(f"✓ {Level.AVAILABLE.label} — every required input is present.")


def _lineage_text(state: DataState) -> str:
    return describe_lineage(ss.workflow(), state.id)


def input_picker(stage_id: str, variant: Optional[str] = None) -> None:
    """Let the person apply this operation to a different existing state
    (e.g. AGC on the raw gathers instead of the band-passed ones). Only
    shown when there is a real choice."""
    chk = ss.check(stage_id, variant)
    default = stage_ops.check(st.session_state, stage_id, variant, inputs={})   # no remembered picks
    multi = {role: opts for role, opts in chk.options.items() if len(opts) > 1}
    if not multi:
        return
    with st.expander("Choose a different input state"):
        st.caption("By default an operation reads the current data. To apply it to an earlier or "
                   "alternative state instead, pick that state — the new result becomes a new state "
                   "and nothing already computed is touched.")
        for role, opts in multi.items():
            ids = [o.id for o in opts]
            default_id = default.inputs.get(role)
            effective = chk.inputs.get(role)
            # Labels are computed here, in the script run: a format function that reads the
            # session would break wherever a widget is inspected outside a script run (AppTest).
            labels = {o.id: f"{state_name(o)} — {_lineage_text(o)}" + ("  (current)" if o.id == default_id else "")
                      for o in opts}

            def _changed(_role=role, _key=f"pick_{stage_id}_{variant}_{role}", _def=default_id):
                chosen = st.session_state.get(_key)
                ss.set_pick(stage_id, _role, None if chosen == _def else chosen, variant)

            st.selectbox(f"Input for “{role}”", ids, index=ids.index(effective) if effective in ids else 0,
                         format_func=labels.get, key=f"pick_{stage_id}_{variant}_{role}", on_change=_changed)


def show_status(stage_id: str, variant: Optional[str] = None, picker: bool = True) -> StageCheck:
    """Render the operation's readiness banner, the state(s) it will read, and
    an input chooser; return the check so the caller can
    `if not chk.can_run: return` before drawing controls that need the missing input."""
    chk = ss.check(stage_id, variant)
    render_banner(chk)
    if chk.can_run:
        reads = ", ".join(f"{state_name(s)}" + (f" ({role})" if len(chk.states) > 1 else "")
                          for role, s in chk.states.items() if s is not None)
        if reads:
            st.caption(f"📥 Reads: **{reads}**")
        if picker:
            input_picker(stage_id, variant)
    return chk


def run_button(stage_id: str, label: str, *, key: str, variant: Optional[str] = None,
               type: str = "primary", disabled: bool = False, help: Optional[str] = None) -> bool:
    """A Run button that returns True when the operation should execute now.

    * ⛔ blocked   -> disabled, returns False.
    * ✓ available -> returns True on click.
    * ⚠ warnings  -> a click opens a "Continue anyway?" confirmation; the
                     operation runs only when that is confirmed. Never blocks.
    """
    chk = ss.check(stage_id, variant)
    if not chk.can_run:
        st.button(label, key=key, disabled=True,
                  help="A required input is missing — see the ⛔ message above.")
        return False
    clicked = st.button(label, key=key, type=type, disabled=disabled, help=help)
    if not chk.warnings:
        return clicked

    pending = f"_confirm_{key}"
    if clicked:
        st.session_state[pending] = True
    if st.session_state.get(pending):
        st.warning("**Continue?** " + " ".join(w.message for w in chk.warnings)
                   + "\n\nNothing prevents running it — this is only a recommendation.")
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("Continue anyway", key=f"{key}__continue", type="primary"):
            st.session_state[pending] = False
            return True
        if c2.button("Cancel", key=f"{key}__cancel"):
            st.session_state[pending] = False
            st.rerun()
    return False


def execute(fn, *args, **kwargs) -> bool:
    """Run a `stage_ops` executor against the session. A genuine missing input
    (`DependencyError`) or invalid parameter (`ValidationError`) is shown as a
    message rather than a traceback. On success the page reruns (so every
    widget re-reads the new data state) after leaving a note saying which
    state was produced; on failure this returns False."""
    try:
        state = fn(st.session_state, *args, **kwargs)
    except ValidationError as e:
        st.error(str(e))
        return False
    run = st.session_state.get("last_run") or {}
    if isinstance(state, DataState):
        src = ss.source(state)
        built_from = f" from {state_name(src)}" if src is not None else ""
        if run.get("reused"):
            _flash(f"♻ Identical to {state_name(state)} (same operation, parameters and input) — "
                   f"reusing it; nothing was recomputed.")
        else:
            _flash(f"✅ Produced **{state_name(state)}**{built_from}.")
    st.rerun()
    return True


# ---- data state --------------------------------------------------------------------------------------

def data_state_caption(stream: str, prefix: str) -> None:
    """e.g. '📍 Current shot-domain data: Raw → Band-pass → F-K (S5)'."""
    head = ss.current(stream)
    tag = f" ({head.id})" if head is not None else ""
    st.caption(f"📍 {prefix}: **{describe_stream(ss.workflow(), stream)}**{tag}")


def stale_banner(state_ids: Optional[Sequence[str]] = None) -> None:
    """⟳ notes for states built from data that has since moved on. Nothing is
    discarded — a stale state is valid data that is simply not built from the
    latest inputs."""
    for sid, msg in stale_notes(ss.workflow()).items():
        if state_ids is None or sid in state_ids:
            st.info(f"⟳ {msg}", icon="🔄")


def stale_for_ops(stage_ids: Sequence[str]) -> List[str]:
    """Ids of the stale states produced by any of these operations."""
    stale = stale_info(ss.workflow())
    return [s.id for s in ss.graph() if s.op in stage_ids and s.id in stale]


# ---- the table of states + undo / redo / current / free / discard -----------------------------------

def _state_rows(kinds: Optional[Sequence[str]]) -> List[dict]:
    g = ss.graph()
    wf = ss.workflow()
    stale = stale_info(wf)
    memory = {m["id"]: m for m in stage_ops.memory_report(st.session_state)}
    currents = {k: (g.current(k).id if g.current(k) else None) for k in {s.kind for s in g}}
    rows = []
    for s in g:
        if kinds is not None and s.kind not in kinds:
            continue
        flags = []
        if currents.get(s.kind) == s.id:
            flags.append("◀ current")
        if s.id in stale:
            flags.append("⟳ stale")
        if not s.materialized:
            flags.append("💤 released")
        rows.append({
            "state": s.id, "step": stage_label(s.op) + (f" ({s.variant})" if s.variant else "")
            + (f" v{s.version}" if s.version > 1 else ""),
            "built from": ", ".join(f"{role}={sid}" for role, sid in s.inputs.items()) or "—",
            "parameters": params_summary(s.params),
            "data": describe_lineage(wf, s.id),
            "status": "  ".join(flags),
            "MB": round(memory.get(s.id, {}).get("bytes", 0) / 2 ** 20, 1),
            "fingerprint": s.fingerprint,
        })
    return rows


def states_panel(kinds: Sequence[str], key: str, title: str = "🗂 Data states · undo · redo") -> None:
    """The data states of these kinds, with the controls that move around them.

    Nothing here recomputes anything except re-materialising a state whose
    arrays were released: undo/redo and “make current” only move a pointer;
    “discard” removes a state and what was built on it; “free memory” drops
    arrays but keeps the record."""
    g = ss.graph()
    states = [s for s in g if s.kind in kinds]
    if not states:
        return
    with st.expander(title):
        st.dataframe(pd.DataFrame(_state_rows(kinds)), hide_index=True)
        for kind in kinds:
            if not any(s.kind == kind for s in states):
                continue
            c1, c2, c3 = st.columns([1, 1, 4])
            u, r = g.undo_target(kind), g.redo_target(kind)
            if c1.button("↶ Undo", key=f"{key}_undo_{kind}", disabled=u is None,
                         help=(f"Go back to {state_name(u)}" if u else "Nothing to undo")):
                ss.undo(kind)
                st.rerun()
            if c2.button("↷ Redo", key=f"{key}_redo_{kind}", disabled=r is None,
                         help=(f"Go forward to {state_name(r)}" if r else "Nothing to redo")):
                ss.redo(kind)
                st.rerun()
            cur = g.current(kind)
            c3.caption(f"Current {kind} data: **{state_name(cur)}**" if cur else "")
        ids = [s.id for s in states]
        names = {s.id: f"{state_name(s)} — {describe_lineage(ss.workflow(), s.id)}" for s in states}
        pick = st.selectbox("State", ids, index=len(ids) - 1, format_func=names.get, key=f"{key}_pick")
        deps = [d for d in g.dependents(pick)]
        b1, b2, b3 = st.columns(3)
        if b1.button("Make current", key=f"{key}_select"):
            ss.select(pick)
            st.rerun()
        if b2.button("Free memory", key=f"{key}_release", disabled=not stage_ops.releasable(
                st.session_state, pick), help="Drop this state's arrays; it is re-computed from its "
                                              "recorded operation when needed again."):
            ss.release(pick)
            st.rerun()
        if b3.button("Discard", key=f"{key}_discard",
                     help="Delete this state and every state built on it."):
            ss.discard(pick)
            st.rerun()
        st.caption("Discarding also removes: " + (", ".join(state_name(d) for d in deps) or "nothing else")
                   + ".")


def undo_panel(stage_ids: Sequence[str], key: str) -> None:
    """The states made by these operations (kept under its old name)."""
    kinds = list(dict.fromkeys(STAGE_KIND[s] for s in stage_ids if s in STAGE_KIND))
    states_panel(kinds, key)


# ---- whole-workflow status ----------------------------------------------------------------------------

def status_rows() -> List[dict]:
    wf = ss.workflow()
    stale = stale_info(wf)
    rows = []
    for sid, variant, label, chk in check_all(wf):
        issues = [i for i in chk.issues if not (chk.done and i.kind == "repeated_operation")]
        note = ""
        blockers = [i for i in issues if i.level == Level.BLOCKED]
        warnings = [i for i in issues if i.level == Level.WARNING]
        if blockers:
            note = blockers[0].message
        elif warnings:
            note = warnings[0].message
        made = wf.graph.of_op(sid, variant)
        if made and not note and any(m.id in stale for m in made):
            m = next(m for m in made if m.id in stale)
            note = "⟳ " + stale[m.id][0].detail
        level = (Level.BLOCKED if blockers else Level.WARNING if warnings else Level.AVAILABLE)
        rows.append({"stage": sid, "variant": variant, "label": label, "level": level, "done": chk.done,
                     "note": note, "n_states": len(made),
                     "order": min((m.seq for m in made), default=None)})
    return rows


def sidebar_status() -> None:
    """Compact per-step status list for the sidebar."""
    if st.session_state["survey"] is None:
        st.caption("Load data to see which steps are available.")
        return
    for r in status_rows():
        count = f" ×{r['n_states']}" if r["n_states"] > 1 else ""
        st.caption(f"{r['level'].symbol} {r['label']}" + (" ✅" if r["done"] else "") + count)
    legend()


def sidebar_current_data() -> None:
    """Kept for callers of the old name; the sidebar shows the compact current state."""
    current_state_panel(compact=True)


def home_status() -> None:
    """Grouped status for the home page: one column per workflow phase."""
    rows = {(r["stage"], r["variant"]): r for r in status_rows()}
    cols = st.columns(len(STAGE_GROUPS))
    for col, (heading, group) in zip(cols, STAGE_GROUPS):
        with col:
            st.markdown(f"**{heading}**")
            for s in PIPELINE_STAGES:
                if s.group != group:
                    continue
                variants = [None] if s.id != "migration" else ["poststack", "pstm"]
                for v in variants:
                    r = rows[(s.id, v)]
                    tag = " ✅" if r["done"] else ""
                    count = f" ×{r['n_states']}" if r["n_states"] > 1 else ""
                    opt = " · optional" if s.optional and not r["done"] else ""
                    st.caption(f"{r['level'].symbol} {r['label']}{tag}{count}{opt}")
    legend()


# ---- the three things this app keeps apart ------------------------------------------------------------
#
#   CURRENT PROCESSING STATE   what the data is right now            (derived from the states)
#   RECOMMENDED WORKFLOW       what is usually done, in what order   (status panels above)
#   PROCESSING HISTORY         what was actually done, in what order (the immutable action log)
#
# The functions below draw the first and the third. Neither is computed from the other.

_CURRENT_ICONS = {t.id: t.icon for t in registry.DATA_TYPES if t.summarised}    # declared in the registry


def current_state_panel(compact: bool = False) -> None:
    """CURRENT PROCESSING STATE — one row per kind of data that exists: which state is current
    and what has been applied to produce it. Not a history (that is the action log) and not a
    recommendation (that is the workflow status)."""
    if st.session_state["survey"] is None:
        if not compact:
            st.caption("No data loaded yet.")
        return
    rows = current_processing_state(ss.workflow())
    if not rows:
        if not compact:
            st.caption("No data states yet.")
        return
    if compact:
        for r in rows:
            flags = (" ⟳" if r["stale"] else "") + (" 💤" if r["released"] else "")
            st.caption(f"{_CURRENT_ICONS.get(r['kind'], '•')} {r['data']}: **{r['applied']}** "
                       f"({r['state']}){flags}")
        return
    st.dataframe(pd.DataFrame([{
        "Data": r["data"], "Current state": r["name"], "What has been applied": r["applied"],
        "States of this kind": r["n_states"],
        "Status": "  ".join(f for f in (("⟳ " + r["stale"]) if r["stale"] else "",
                                         "💤 arrays released" if r["released"] else "") if f) or "✓ up to date",
    } for r in rows]), hide_index=True)


def history_order_control(key: str) -> str:
    """The explicit, user-selectable display order of the history ("oldest first" or "newest
    first"). One shared setting (`ss.history_order()`), so every history on every page agrees;
    it changes only how entries are *listed* — the log itself, and each entry's action number,
    are never touched. Returns the order in force."""
    st.session_state[key] = ss.history_order()          # one setting, however many widgets show it

    def _changed(_key=key):
        ss.set_history_order(st.session_state[_key])

    st.radio("Show history", list(ORDERS), format_func=ORDER_LABELS.get, key=key, horizontal=True,
             on_change=_changed,
             help="Only the order the list is shown in. The action numbers (#1, #2 …) are the order "
                  "the actions were performed and never change.")
    return ss.history_order()


HISTORY_FILTERS = ("All actions", "Operations only", "Failures & warnings")


def _action_rows(entries: Sequence[Action], log, with_dataset: bool) -> List[dict]:
    rows = []
    for e in entries:
        gone = log.discarded_by(e)
        output = (e.output or "—") + (f"  (discarded by #{gone.n})" if gone else "")
        problem = (f"✗ {e.error}" if e.status == FAILURE else "") or " | ".join(e.warnings)
        row = {"#": e.n, "Time": e.timestamp, "Action": e.label,
               "Parameters": format_params(e.params), "Input": e.input_state or "—", "Output": output,
               "Result": result_text(e), "QC": format_qc(e.qc) or e.detail, "Warnings / reason": problem}
        if with_dataset:
            row = {"Dataset": e.session, **row}
        rows.append(row)
    return rows


def history_table(key: str = "history", *, inspector: bool = True) -> None:
    """PROCESSING HISTORY — every action in the order it was performed (the immutable action
    log): number, time, operation, parameters, input and output state, result, QC and warnings.

    Listed oldest- or newest-first according to the one shared display setting; the filter and
    the dataset toggle only hide rows. Nothing here can change the log."""
    log = ss.action_log()
    if len(log) == 0:
        st.caption("Nothing has been done yet. Every operation you run — successful or not — is "
                   "recorded here in the order you ran it.")
        return
    order = history_order_control(f"{key}_order")
    c1, c2 = st.columns([2, 3])
    shown_filter = c1.radio("Filter", HISTORY_FILTERS, key=f"{key}_filter", horizontal=True)
    session = log.session
    if len(log.sessions()) > 1:
        earlier = len(log) - len(log.view(session=log.session))
        if c2.checkbox(f"Include earlier datasets ({earlier} action(s))", key=f"{key}_all_sessions"):
            session = None
    kinds = ("operation",) if shown_filter == "Operations only" else None
    entries = log.view(order, kinds=kinds, session=session, issues_only=shown_filter == "Failures & warnings")
    st.caption(f"{len(entries)} of {len(log.view(session=session))} action(s) shown · "
               f"{'oldest → newest' if order == 'oldest' else 'newest → oldest'}.")
    if not entries:
        st.caption("No action matches this filter.")
        return
    st.dataframe(pd.DataFrame(_action_rows(entries, log, with_dataset=session is None)), hide_index=True)
    if inspector:
        numbers = [e.n for e in entries]
        names = {e.n: one_line(e) for e in entries}
        with st.expander("Inspect one action"):
            pick = st.selectbox("Action", numbers, format_func=names.get, key=f"{key}_inspect")
            st.code("\n".join(report_lines([log[pick]])), language=None)
            st.json(log[pick].to_dict(), expanded=False)


def sidebar_history(limit: int = 8) -> None:
    """The most recent actions of the current dataset, one line each, listed in the chosen
    display order. The full log is on the History page."""
    log = ss.action_log()
    entries = log.view("oldest", session=log.session)
    if not entries:
        st.caption("Nothing has been done yet.")
        return
    order = history_order_control("history_order_sidebar")
    recent = entries[-limit:]
    for e in (recent if order == "oldest" else recent[::-1]):
        st.caption(one_line(e))
    more = len(entries) - len(recent)
    st.caption((f"Last {len(recent)} of {len(entries)} actions · " if more else "")
               + "full log: 🕘 History page")
