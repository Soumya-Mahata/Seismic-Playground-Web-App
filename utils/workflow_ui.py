"""
utils/workflow_ui.py — Presentation of workflow readiness
============================================================
Requires Streamlit. Renders what `modules/dependencies` decides, using
three distinct states so a person can tell a recommendation from a
requirement at a glance:

    ✓  Available                              runnable, nothing to flag
    ⚠  Recommended prerequisite not performed runnable — asks "Continue?"
    ⛔ Required input missing                  not runnable; says what, and where to get it

A page shows a stage's status with `show_status()` (banner + early exit
for ⛔), and runs it with `run_button()`, which only interposes a
confirmation when there are ⚠ warnings — and never blocks on them.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from modules import stage_ops
from modules.dependencies import Level, SHORT, StageCheck, check_all, describe_stream, stale_notes
from modules.pipeline import PIPELINE_STAGES, STAGE_GROUPS, stage_label
from modules.validation import ValidationError
from utils import session_state as ss

LEGEND_MD = (
    "**✓ Available** — can run now.  \n"
    "**⚠ Recommended prerequisite not performed** — can still run; you'll be asked to confirm.  \n"
    "**⛔ Required input missing** — cannot run until the input it names exists.  \n"
    "**✅ done** — already run (running it again replaces it and discards only the steps built on it)."
)


def legend() -> None:
    st.caption(LEGEND_MD)


# ---- single-stage status ------------------------------------------------

def _dependents_note(stage_id: str) -> Optional[str]:
    deps = ss.workflow().dependents(stage_id)
    if not deps:
        return None
    return ("↻ Running this again replaces its result and discards the steps built on it: "
            + ", ".join(stage_label(d) for d in deps) + ".")


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
    if chk.done:
        note = _dependents_note(chk.stage_id)
        if note:
            st.caption(note)


def show_status(stage_id: str, variant: Optional[str] = None) -> StageCheck:
    """Render the stage's readiness banner and return the check, so the
    caller can `if not chk.can_run: return` before drawing controls that
    need the missing input."""
    chk = ss.check(stage_id, variant)
    render_banner(chk)
    return chk


def run_button(stage_id: str, label: str, *, key: str, variant: Optional[str] = None,
               type: str = "primary", disabled: bool = False, help: Optional[str] = None) -> bool:
    """A Run button that returns True when the stage should execute now.

    * ⛔ blocked   -> disabled, returns False.
    * ✓ available -> returns True on click.
    * ⚠ warnings  -> a click opens a "Continue anyway?" confirmation; the
                     stage runs only when that is confirmed. Never blocks.
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
    """Run a `stage_ops` executor against the session. A genuine missing
    input (`DependencyError`) or invalid parameter (`ValidationError`)
    is shown as a message rather than a traceback. On success the page
    reruns (so every widget re-reads the new data state); on failure this
    returns False."""
    try:
        fn(st.session_state, *args, **kwargs)
    except ValidationError as e:
        st.error(str(e))
        return False
    st.rerun()
    return True


def input_key(chk: StageCheck, role: str = "data") -> str:
    """The data key a stage would read for `role` right now (e.g.
    'traces_bp' if band-pass is the head of the shot data)."""
    return stage_ops.data_keys(ss.workflow(), chk.inputs[role])[0]


def input_name(chk_or_entry, role: str = "data") -> str:
    """Short human name of the stage whose output a run read/reads."""
    src = (chk_or_entry.inputs.get(role) if isinstance(chk_or_entry, StageCheck)
           else chk_or_entry.consumed.get(role))
    return SHORT.get(src, src or "?")


# ---- data state ----------------------------------------------------------

def data_state_caption(stream: str, prefix: str) -> None:
    """e.g. '📍 Current shot-domain data: Raw → Band-pass → F-K'."""
    st.caption(f"📍 {prefix}: **{describe_stream(ss.workflow(), stream)}**")


def stale_banner(stage_ids: Optional[List[str]] = None) -> None:
    """⟳ notes for finished stages built from data that has since moved on."""
    notes = stale_notes(ss.workflow())
    for sid, msg in notes.items():
        if stage_ids is None or sid in stage_ids:
            st.info(f"⟳ {msg}", icon="🔄")


# ---- undo / reorder -----------------------------------------------------------

def undo_panel(stage_ids: List[str], key: str) -> None:
    """Discard one finished step together with everything built from it.

    This is how a person reorders: the data lineage is honest (a result
    cannot be 'un-applied' from the middle of a chain without redoing what
    came after), so to apply two steps in the other order, discard the one
    that went first — which also discards what consumed it — then run
    them in the order wanted."""
    wf = ss.workflow()
    done = [s for s in stage_ids if wf.has(s)]
    if not done:
        return
    with st.expander("↩ Undo or reorder steps"):
        st.caption("Discard a finished step and the steps built on top of it, then run them again in "
                   "the order you want. Steps that never used its output are kept.")
        order = {h.stage_id: h.seq for h in wf.history}
        done.sort(key=lambda s: order.get(s, 0))
        pick = st.selectbox("Step to discard", done, key=f"{key}_pick",
                            format_func=lambda s: f"{order.get(s, 0)}. {stage_label(s)}")
        victims = wf.dependents(pick)
        st.caption("Will also discard: " + (", ".join(stage_label(v) for v in victims) or "nothing else") + ".")
        if st.button("Discard", key=f"{key}_discard"):
            ss.invalidate_from(pick, include_self=True)
            st.rerun()


# ---- whole-workflow status ----------------------------------------------------

def status_rows() -> List[dict]:
    wf = ss.workflow()
    stale = stale_notes(wf)
    rows = []
    for sid, variant, label, chk in check_all(wf):
        note = ""
        if chk.blockers:
            note = chk.blockers[0].message
        elif chk.warnings:
            note = chk.warnings[0].message
        if sid in stale and not note:
            note = "⟳ " + stale[sid]
        rows.append({"stage": sid, "variant": variant, "label": label, "level": chk.level,
                     "done": chk.done, "note": note,
                     "order": next((h.seq for h in wf.history if h.stage_id == sid), None)})
    return rows


def sidebar_status() -> None:
    """Compact per-step status list for the sidebar."""
    rows = status_rows()
    if st.session_state["survey"] is None:
        st.caption("Load data to see which steps are available.")
        return
    for r in rows:
        st.caption(f"{r['level'].symbol} {r['label']}" + (" ✅" if r["done"] else ""))
    legend()


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
                    opt = " · optional" if s.optional and not r["done"] else ""
                    st.caption(f"{r['level'].symbol} {r['label']}{tag}{opt}")
    legend()


def history_table() -> None:
    """The processing history in *execution* order — which is not
    necessarily the recommended order."""
    wf = ss.workflow()
    if not wf.history:
        st.caption("Nothing has been run yet.")
        return
    rec_index = {s.id: i for i, s in enumerate(PIPELINE_STAGES)}
    rows = []
    prev_rec = -1
    for h in sorted(wf.history, key=lambda e: e.seq):
        out_of_order = rec_index.get(h.stage_id, 0) < prev_rec
        prev_rec = max(prev_rec, rec_index.get(h.stage_id, 0))
        rows.append({"#": h.seq, "step": h.label, "ran on": ", ".join(
                         stage_label(s) for s in [h.consumed.get("data")] if s) or "—",
                     "outside recommended order": "yes" if out_of_order else "",
                     "time": h.timestamp})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
