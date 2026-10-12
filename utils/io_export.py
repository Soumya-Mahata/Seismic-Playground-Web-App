"""
utils/io_export.py — Export Helpers
=======================================
Requires no Streamlit — pure file-writing functions the `views/` layer
calls after collecting a save path from the user (Streamlit's
`st.download_button` reads bytes, so most of these return bytes/str
rather than writing to disk directly).
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules import segy_io
from modules.action_log import ActionLog, format_params, format_qc, report_lines, result_text
from modules.dependencies import Workflow, current_processing_state
from modules.pipeline import history_to_report_lines
from modules.state_graph import ProcessingGraph
import config as cfg


def section_to_csv_bytes(section: np.ndarray, cdp_list: List[int], twt: np.ndarray) -> bytes:
    df = pd.DataFrame(section.T, columns=[f"cdp_{c}" for c in cdp_list])
    df.insert(0, "twt_s", twt)
    return df.to_csv(index=False).encode("utf-8")


def section_to_npy_bytes(section: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, section)
    return buf.getvalue()


def segy_bytes_from_section(section: np.ndarray, dt: float, cdp_list: List[int],
                             group_interval: float, text_header: str = "") -> bytes:
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sgy")
    tmp.close()
    try:
        segy_io.write_segy_section(tmp.name, section, dt, np.array(cdp_list), group_interval, text_header)
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        os.unlink(tmp.name)
    return data


def horizon_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_html_bytes(fig: go.Figure) -> bytes:
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def processing_report_text(survey_info: dict, graph: ProcessingGraph, extra_notes: str = "",
                            state_id: Optional[str] = None) -> str:
    """The processing report. `graph` is the session's `ProcessingGraph`.

    Three sections that are deliberately not mixed up:
      CURRENT PROCESSING STATE  what the data is now (from the states);
      ACTION LOG                what was done, in the exact order it was done (the immutable log —
                                always chronological, whatever order the app displays it in);
      DATA STATES               the reference table of what exists, each with the state(s) it consumed.
    With `state_id`, the action log lists only the successful operations that produced that state
    or the states it was built from (still in the order performed), and the state table the same."""
    log = graph.actions
    lines = [
        "=" * 70,
        f"{cfg.APP_TITLE.upper()} — PROCESSING REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET",
        "-" * 70,
    ]
    for k, v in survey_info.items():
        lines.append(f"{k}: {v}")

    lines += ["", "CURRENT PROCESSING STATE (what the data is now)", "-" * 70]
    now = current_processing_state(Workflow(graph))
    lines += [f"{r['data']}: {r['applied']}  [{r['state']}]"
              + (f"  — out of date: {r['stale']}" if r["stale"] else "")
              + ("  — arrays released" if r["released"] else "") for r in now] or ["  (no data states)"]

    if state_id is None:
        entries = list(log.entries())
        lines += ["", f"ACTION LOG ({len(entries)} action(s), in the exact order performed — repeated "
                      f"operations are separate entries; nothing is overwritten)", "-" * 70]
        shown_graph = graph
        states_title = ("DATA STATES (reference: every state that exists, in the order produced, with "
                        "the state(s) it consumed, its operation and parameters)")
    else:
        keep = {s.id for s in graph.ancestors(state_id)} | {state_id}
        entries = [e for e in log.entries()
                   if e.is_operation and e.ok and not e.reused and e.session == log.session
                   and e.output in keep]
        lines += ["", f"ACTION LOG — the operations that produced {state_id} (successful, in the order "
                      f"performed; the full log is in the unfiltered report)", "-" * 70]
        shown_graph = ProcessingGraph.__new__(ProcessingGraph)
        shown_graph.__dict__.update(graph.__dict__)
        shown_graph._states = {k: v for k, v in graph._states.items() if k in keep}
        states_title = f"DATA STATES THAT PRODUCED {state_id} (in the order they were produced)"
    lines += report_lines(entries) or ["  (no actions recorded)"]

    lines += ["", states_title, "-" * 70]
    lines += history_to_report_lines(shown_graph) or ["  (no steps recorded)"]
    if extra_notes:
        lines += ["", "NOTES", "-" * 70, extra_notes]
    lines += ["", "=" * 70,
              f"Generated by {cfg.APP_TITLE} ({cfg.APP_TAGLINE}).",
              "See README.md 'Known limitations' for documented simplifications",
              "and engineering trade-offs made in this pipeline.",
              "=" * 70]
    return "\n".join(lines)


ACTION_LOG_COLUMNS = ["action_number", "timestamp", "dataset", "type", "operation", "parameters",
                      "input_state", "output_state", "result", "qc_summary", "warnings", "error"]


def action_log_csv_bytes(log: ActionLog, order: str = "oldest") -> bytes:
    """The action log as CSV, one row per action. Rows are in chronological order unless `order`
    says otherwise; `action_number` always identifies the true execution position."""
    rows = [{"action_number": e.n, "timestamp": e.timestamp, "dataset": e.session, "type": e.kind,
             "operation": e.label, "parameters": format_params(e.params, limit=10 ** 6),
             "input_state": e.input_state or "", "output_state": e.output or "",
             "result": result_text(e), "qc_summary": format_qc(e.qc, limit=10 ** 6) or e.detail,
             "warnings": " | ".join(e.warnings), "error": e.error}
            for e in log.view(order)]
    return pd.DataFrame(rows, columns=ACTION_LOG_COLUMNS).to_csv(index=False).encode("utf-8")


def action_log_json_bytes(log: ActionLog) -> bytes:
    """The complete action records (chronological) with their integrity hash chain."""
    return json.dumps({"actions": log.to_dicts(), "integrity_problems": log.verify()}, indent=2,
                      default=str).encode("utf-8")


def params_json_bytes(graph: ProcessingGraph) -> bytes:
    """The whole processing graph as JSON: every data state with the state(s) it
    consumed, its operation, complete parameters, lineage fingerprint and
    diagnostics, which state is current for each kind of data, and the immutable
    action log (every action in the order performed). No arrays."""
    return json.dumps(graph.export(), indent=2, default=str).encode("utf-8")


def recipe_json_bytes(graph: ProcessingGraph, state_id: str) -> bytes:
    """The minimal, replayable recipe for one state: exactly the operations (with
    parameters and which recorded state each consumed) that produce it from the
    loaded survey. `stage_ops.replay_recipe` re-runs it and checks the lineage
    fingerprints."""
    return json.dumps({"target": state_id, "steps": graph.recipe(state_id)}, indent=2,
                      default=str).encode("utf-8")
