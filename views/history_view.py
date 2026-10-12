"""views/history_view.py — the processing history (immutable action log) and the current state."""

from __future__ import annotations

import streamlit as st

from utils import io_export, session_state as ss
from utils import workflow_ui as wf


def render():
    st.header("🕘 Processing History")
    st.caption(
        "Three different things, shown separately: **where the data is now** (current processing "
        "state), **what is usually done** (the recommended workflow — Home page), and **what you "
        "actually did, in the order you did it** (the history below)."
    )

    st.subheader("Current processing state")
    st.caption("What the data is right now — derived from the data states, not from the history.")
    wf.current_state_panel()

    st.divider()
    st.subheader("History — what was done, in order")
    log = ss.action_log()
    problems = log.verify()
    if problems:
        st.error("The action log failed its integrity check — an entry was altered after it was "
                 "recorded:\n\n" + "\n".join(f"- {p}" for p in problems))
    elif len(log):
        st.caption(f"🔒 Append-only: {len(log)} action(s), numbered in the order performed. Entries are "
                   f"never edited, reordered or removed — repeating an operation adds an entry, and "
                   f"discarding a state leaves the entry that made it. Integrity verified.")
    wf.history_table("history_page")

    if len(log):
        survey = st.session_state["survey"]
        base = survey.source_name.rsplit(".", 1)[0] if survey is not None else "seismic"
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇ History (.csv)", io_export.action_log_csv_bytes(log),
                           file_name=f"{base}_history.csv", mime="text/csv",
                           help="Every action in chronological order (action number, time, operation, "
                                "parameters, input/output state, result, QC, warnings).")
        c2.download_button("⬇ History (.json)", io_export.action_log_json_bytes(log),
                           file_name=f"{base}_history.json", mime="application/json",
                           help="The complete records, chronological, with the integrity hash chain.")
