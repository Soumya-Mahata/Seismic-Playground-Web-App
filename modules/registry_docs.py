"""
modules/registry_docs.py — documentation generated from the operation registry
===============================================================================
`docs/OPERATIONS.md` (a reference of every operation: what it reads and produces, what it
requires, what is recommended, its parameters and warnings, a workflow diagram) and
`docs/workflow_graph.json` (the same relations as plain data, for any future workflow
visualisation) are *generated* from `modules/registry.py`, never written by hand, so the
documentation cannot say something the code does not.

    python -m modules.registry_docs          # rewrite both files
    python -m modules.registry_docs --check  # exit 1 if they are out of date

`tests/test_registry.py` runs the check. No Streamlit dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

from modules import registry as R

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OPERATIONS_MD = DOCS_DIR / "OPERATIONS.md"
GRAPH_JSON = DOCS_DIR / "workflow_graph.json"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _default(p: R.ParamSpec) -> str:
    if p.default is R.REQUIRED:
        return "*required*"
    text = repr(p.default)
    return f"`{text if len(text) <= 40 else text[:37] + '…'}`"


def _limits(p: R.ParamSpec) -> str:
    parts: List[str] = []
    if p.choices:
        parts.append(" / ".join(p.choices))
    if p.min is not None:
        parts.append(f"≥ {p.min:g}")
    if p.gt is not None:
        parts.append(f"> {p.gt:g}")
    if p.max is not None:
        parts.append(f"≤ {p.max:g}")
    return ", ".join(parts)


def _kinds(op: R.Operation) -> str:
    if not op.needs:
        return "—"
    return "; ".join(f"**{role}**: {' or '.join(kinds)}" for role, kinds in op.input_types.items())


def _stage_names(stage_ids) -> str:
    return ", ".join(R.stage_label(s) for s in stage_ids) or "—"


def _node(key: str) -> str:
    return "n_" + key.replace(":", "_")


def mermaid() -> str:
    """The flow of data between operations, as a Mermaid flowchart: an operation produces a kind
    of data; another operation reads the current state of a kind (solid = preferred input, dotted =
    a fallback) or the output of a named operation (thick). Recommendations are not drawn (they are
    listed per operation) — the diagram shows what can *run*, not what is usual."""
    lines = ["flowchart LR"]
    for gid, heading in R.GROUPS:
        lines.append(f'  subgraph g_{gid}["{heading}"]')
        for o in R.OPERATIONS:
            if o.group == gid:
                shape = ('(["', '"])') if o.optional else ('["', '"]')
                lines.append(f"    {_node(o.key)}{shape[0]}{o.full_name}{shape[1]}")
        lines.append("  end")
    for t in R.DATA_TYPES:
        lines.append(f'  d_{t.id}{{{{"{t.icon} {t.label or t.id}"}}}}')
    graph = R.workflow_graph()
    for e in graph["edges"]:
        if e["type"] == "produces":
            lines.append(f"  {_node(e['from'])} --> d_{e['to'].split(':')[1]}")
        elif e["type"] == "consumes":
            arrow = "-->" if e["preferred"] else "-.->"
            lines.append(f"  d_{e['from'].split(':')[1]} {arrow} {_node(e['to'])}")
        elif e["type"] == "requires":
            lines.append(f"  {_node(e['from'])} ==> {_node(e['to'])}")
    return "\n".join(lines)


def render_operations_md() -> str:
    out: List[str] = []
    w = out.append
    w("# Operations reference")
    w("")
    w("> **Generated** from `modules/registry.py` by `python -m modules.registry_docs` — do not edit by "
      "hand; change the registry and regenerate (`tests/test_registry.py` fails if this file is out of date).")
    w("")
    w("Every operation is declared once in the registry. This page is the whole declaration, "
      "rendered. Four things are kept apart (see `docs/DEPENDENCY_MODEL.md`): **required inputs** "
      "(⛔ the operation cannot run without them), **recommendations** (⚠ a warning, never a block), the "
      "**recommended order** (the order of this page — advice only) and what has actually been "
      "done (the data states and the action log).")
    w("")
    w("## Kinds of data")
    w("")
    w("| Kind | Name | What it is |")
    w("|---|---|---|")
    for t in R.DATA_TYPES:
        w(f"| `{t.id}` | {t.icon} {_cell(t.label or '—')} | {_cell(t.description)} |")
    w("")
    w("## Operations in the recommended order")
    w("")
    w("| Step | Operation | Category | Produces | Required input | Optional |")
    w("|---|---|---|---|---|---|")
    for o in R.OPERATIONS:
        w(f"| {o.step_number} | [{_cell(o.full_name)}](#{_anchor(o)}) | {R.CATEGORIES[o.category]} | "
          f"`{o.output_type}` | {_cell(_kinds(o))} | {'yes' if o.optional else ''} |")
    w("")
    w("## Data flow")
    w("")
    w("Rounded boxes are optional operations. Solid arrow into an operation: it reads the *current* "
      "state of that kind of data; dotted: a less-preferred alternative; thick: it needs the output of "
      "that specific operation.")
    w("")
    w("```mermaid")
    w(mermaid())
    w("```")
    w("")
    w("## Each operation")
    for o in R.OPERATIONS:
        w("")
        w(f"### {o.full_name} <a id=\"{_anchor(o)}\"></a>")
        w("")
        w(f"`{o.key}` · step {o.step_number} · {R.CATEGORIES[o.category]} · "
          f"{'optional' if o.optional else 'not optional'} · in the app: *{o.location}*")
        w("")
        w(o.help)
        w("")
        w(f"- **Produces:** `{o.output_type}` ({R.DATA_TYPE_BY_ID[o.output_type].label or o.output_type})")
        if o.needs:
            for n in o.needs:
                alt = " (fallback if the first is missing)" if len(n.candidates) > 1 and n.fallback else ""
                w(f"- **Requires ({n.role}):** {n.what} — from "
                  f"{' › '.join(_cand(c) for c in n.candidates)}{alt}." + (f" {n.why}" if n.why else ""))
        else:
            w("- **Requires:** nothing.")
        w(f"- **Hard dependencies (named operations):** {_stage_names(o.hard_dependencies)}")
        for x in o.exclusions:
            w(f"- **Cannot read data that went through:** {_stage_names(x.stages)} — {x.message}")
        w(f"- **Recommended after:** {_stage_names(o.recommended_predecessors)}")
        w(f"- **Recommended before:** {_stage_names(o.recommended_successors)}")
        if o.extra_inputs:
            w(f"- **Also records the input:** {', '.join(o.extra_inputs)}")
        if o.alters_amplitudes:
            w("- **Changes each trace's gain** (amplitude-based detection must look upstream of it).")
        warns = R.possible_warnings(o.key)
        if warns:
            w("- **Warnings it can raise:**")
            for x in warns:
                w(f"    - *{x.kind.replace('_', ' ')}* — {x.text}")
        if o.params:
            w("")
            w("| Parameter | Type | Default | Unit | Limits | |")
            w("|---|---|---|---|---|---|")
            for p in o.params:
                when = f" (only for {p.when[0]} = {' / '.join(map(str, p.when[1]))})" if p.when else ""
                fixed = " *fixed by the application*" if not p.user else ""
                w(f"| `{p.name}` | {p.type} | {_default(p)} | {_cell(p.unit)} | {_cell(_limits(p))} | "
                  f"{_cell(p.help)}{when}{fixed} |")
        else:
            w("")
            w("*No parameters.*")
    w("")
    w("---")
    w("")
    w("Limits are *validity* bounds (what the maths can use), checked before a run; they are not the "
      "ranges of the sliders. Defaults come from `config.py`.")
    w("")
    return "\n".join(out)


def _anchor(o: R.Operation) -> str:
    return "op-" + o.key.replace(":", "-")


def _cand(token: str) -> str:
    if token.startswith("@"):
        t = R.DATA_TYPE_BY_ID[token[1:]]
        return f"the current {t.label or t.id}".replace("the current Loaded survey", "the loaded survey")
    return f"{R.stage_label(token)} output"


def render_graph_json() -> str:
    return json.dumps(R.workflow_graph(), indent=1, ensure_ascii=False) + "\n"


def write() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    OPERATIONS_MD.write_text(render_operations_md(), encoding="utf-8")
    GRAPH_JSON.write_text(render_graph_json(), encoding="utf-8")


def out_of_date() -> List[str]:
    stale = []
    for path, text in ((OPERATIONS_MD, render_operations_md()), (GRAPH_JSON, render_graph_json())):
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(path.name)
    return stale


if __name__ == "__main__":
    if "--check" in sys.argv:
        stale = out_of_date()
        if stale:
            print("out of date (run: python -m modules.registry_docs):", ", ".join(stale))
            sys.exit(1)
        print("docs are up to date")
    else:
        write()
        print(f"wrote {OPERATIONS_MD} and {GRAPH_JSON}")
