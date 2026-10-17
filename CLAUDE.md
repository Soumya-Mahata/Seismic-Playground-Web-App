# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run app.py
```

### Installation
```bash
pip install -r requirements.txt
```

### Testing
Requires `pytest`.
```bash
# Core numeric tests of the processing maths
pytest tests/test_pipeline.py -v

# The operation registry (single source of operation metadata) — and that the generated docs are current
pytest tests/test_registry.py -v
python -m modules.registry_docs --check      # after editing the registry: python -m modules.registry_docs

# The data-state model and the dependency engine (no seismic data needed)
pytest tests/test_state_graph.py tests/test_dependencies.py -v

# The processing history (immutable chronological action log) on its own
pytest tests/test_action_log.py -v

# The same model through the real maths, and through the UI
pytest tests/test_stage_ops.py tests/test_app_nonlinear.py -v

# Full end-to-end pipeline check against synthetic data (ground-truth checks)
python tests/manual_pipeline_check.py

# Scripted UI flow test (simulated widget interactions, every page)
python tests/test_app_flow.py
```

All of the above is the validation gate for any change: unit-level numeric
checks, the registry's consistency, the state graph and dependency rules, real-maths runs in
non-standard orders, history order == execution order, a full non-UI pipeline
run with ground-truth sanity checks, and headless `streamlit.testing.v1.AppTest` walkthroughs of every
page. Run all of them after any change, even a text-only one. With the
recommended order the results are **bit-for-bit identical** to the original
linear app — keep it that way (the processing maths in
`modules/{filtering,preprocessing,nmo,velocity,statics,regularization,stacking,migration}.py`
must not change as a side effect of workflow work).

## Architecture & Structure

### High-Level Design
The project is a Streamlit-based interactive platform for 2D land seismic
processing and interpretation. Several things that a linear pipeline fuses into
one are kept **separate** here:

| Idea | Question it answers | Lives in | Can block a run? |
|---|---|---|---|
| Operation metadata | "What is this operation — what does it read, produce, require, recommend, accept?" | `modules/registry.py` (`Operation`) — **the single source** | — |
| Recommended order | "What is the usual sequence?" | declaration order in `registry.OPERATIONS` (`pipeline.PIPELINE_STAGES` is a view) | no — display/docs only |
| Genuine dependencies | "What input must exist for this to compute?" | `registry` (`Need`, `Exclusion`), evaluated by `modules/dependencies` | **yes (⛔)** |
| Recommendations | "What is scientifically preferable?" | `registry` (`Advice`) + order/repeat/stale checks in `dependencies` | no — ⚠ and a *Continue?* prompt |
| Data states | "What data exists, and how was each piece made?" | `modules/state_graph.py` (the graph) + `modules/stage_ops.py` (payloads) | — |
| Current processing state | "What is the data *now*?" | `dependencies.current_processing_state` (derived from the states) | — |
| Processing history | "What did the person actually do, in what order?" | `modules/action_log.py` (immutable chronological action log) | — |

See the **Operation registry**, **Data-state model** and **Processing history** sections below, and
`docs/DEPENDENCY_MODEL.md` for the ⛔/⚠ rules. The last three rows are never to
be derived from one another: the history is *not* the state graph read in a
different order, the current state is *not* the last history entry, and
neither follows the recommended order.

### Folder Structure
- `app.py`: Main entry point. Handles routing, sidebar navigation, and global CSS.
- `config.py`: Centralized repository for all default processing parameters (and `STATE_MEMORY_BUDGET_MB`).
- `modules/`: **Pure computation layer**. No Streamlit dependencies. Contains the physics and math for seismic processing (NMO/DMO, migration, filtering, statics, stacking, regularisation, attributes, etc.) **and** the workflow model:
    - `state_graph.py`: `DataState`, `ProcessingGraph` — records only, no arrays (see below). Owns the `ActionLog` as `graph.actions`.
    - `action_log.py`: `Action` (frozen entry), `ActionLog` (append-only, hash-chained, `view(order, …)`), text helpers (`format_params`, `format_qc`, `report_lines`) — the processing history (see below).
    - `canonical.py`: `canonical`/`jsonable`/`array_digest`, shared by the graph (fingerprints) and the log (recorded parameters).
    - `registry.py`: **the operation registry** — one `Operation` declaration per operation (see below). Everything about an operation that is not its maths is declared here, once.
    - `registry_docs.py`: generates `docs/OPERATIONS.md` and `docs/workflow_graph.json` from the registry (`python -m modules.registry_docs [--check]`).
    - `dependencies.py`: the engine that answers ✓/⚠/⛔ from the registry's `Need`/`Advice`/`Exclusion`s, resolves which states an operation reads, and detects stale states.
    - `stage_ops.py`: `execute()` and one `run_*` wrapper per operation; owns the payloads (arrays) and the legacy aliases; binds each operation's compute / extras / QC functions to the registry.
    - `pipeline.py`: `PIPELINE_STAGES`, `STAGE_KIND`, `ROOT_STAGES`, `STAGE_GROUPS` as views of the registry; store defaults; `ALIAS_KEYS`.
- `visualization/`: **Plotly figure builders**. No Streamlit dependencies. Handles the conversion of seismic data into visual representations, including the shared `theme.py` style tokens and the Petrel-convention `ATTRIBUTE_COLORSCALE_MAP` lookup used by the Attributes page.
- `views/`: **UI layer**. One module per application page. Contains Streamlit widgets and orchestrates calls between `modules/` and `visualization/`. Views never run processing maths inline: they call `stage_ops.run_*` through `wf.execute`.
- `utils/`: Shared helpers.
    - `session_state.py`: Owns the `st.session_state` schema; a thin adapter over the graph and the engine (`graph`, `check`, `current`, `latest`, `result`, `select`, `undo`, `redo`, `discard`, `release`, `best_available_section`) plus `action_log()` and the display-order setting `history_order()` / `set_history_order()`.
    - `workflow_ui.py`: ✓ Available / ⚠ Recommended prerequisite not performed / ⛔ Required input missing banners, the "Continue?" run button (`run_button`), `execute`, the "Choose a different input state" picker, the data-states panel (table + undo/redo/make current/free memory/discard), stale banners, sidebar/home status; and the **current-state panel** (`current_state_panel`), the **history table** (`history_table`, `sidebar_history`) with its explicit oldest/newest control (`history_order_control`). Pages must use these rather than gating on `require_stage`.
    - `click_picking.py`: `numeric_point_form()` — a small (x, y) + "Add point" form used for all point-entry (horizons, faults, velocity picks, F-K polygon). Plotly/Streamlit can't deliver click-selection events on `go.Heatmap` traces, so every one of these panels uses this form instead of a literal click.
    - `io_export.py`: SEG-Y, CSV, NumPy, and the processing report (current state · action log · data states) / state-graph JSON / replayable recipe JSON / action-log CSV and JSON.
- `docs/`: `DEPENDENCY_MODEL.md` (the ⛔/⚠ rules), `OPERATIONS.md` and `workflow_graph.json` (**generated** from the registry — never edit by hand).
- `tests/`: `test_pipeline.py` (numeric unit tests of the maths), `test_registry.py` (the registry: consistency, derived views, no re-declared dependency tables, bound implementations, schema validation, generated docs current), `test_action_log.py` (the history on its own: order, immutability, tamper detection, views), `test_state_graph.py` (the graph: identity, current pointers, undo/redo, lineage, discard), `test_dependencies.py` (engine on a bare graph: ⛔/⚠/✓, stale, order independence), `test_stage_ops.py` (real maths: non-standard orders, repeat/branch/memoise/no-copy/release/undo/discard/replay, history == execution order), `test_app_nonlinear.py` (UI: ⚠ confirm / ⛔ block, states panel, input picker, History page and display order), `test_app_flow.py` (AppTest walkthrough of the recommended order), `manual_pipeline_check.py` (non-UI full run with ground-truth checks).

### Key Technical Patterns
- **Workflow order (recommended, not enforced)**: Pre-stack conditioning runs in the **shot domain** (band-pass → AGC → trace edit/kill/de-spike → mute → balance → F-K dip filter → multiple suppression → deconvolution), then **CDP Sort & Binning**, then velocity analysis pass 1 → NMO → residual statics → DMO → inverse NMO → data regularisation → velocity analysis pass 2 → final NMO → stack → post-stack processing (random noise attenuation → decon → time-variant filter) → migration (post-stack Kirchhoff, or pre-stack Kirchhoff time migration on CDP gathers before stack) → final scaling. The full ordered stage list is the declaration order in `modules/registry.OPERATIONS` (`pipeline.PIPELINE_STAGES` is a view of it; `docs/OPERATIONS.md` renders it). A stage runs whenever its **genuine** inputs exist; a skipped or out-of-order *recommended* step only produces a ⚠ "Continue?" — never a block.
- **Widgets whose default depends on upstream data** (e.g. the multiple-suppression lag) must be keyed by the input *state*, or they freeze at the value computed when the tab first drew.
- **Widget `format_func`s must not read `st.session_state`** — `AppTest` evaluates them outside a script run. Precompute a `{value: label}` dict in the script run and pass `labels.get` (see `workflow_ui.input_picker`).
- **Click-to-pick**: Uses `utils/click_picking.numeric_point_form()` to collect (x, y) picks via a form + editable table rather than literal Plotly click events, for horizons, faults, velocity picks, and the F-K accept/reject polygon (which also supports inserting a point between two existing rows and reordering via an explicit "order" column).
- **Data flow**: Traces → Shot Gathers → CDP Gathers → Stacked Section → (optionally) Migrated Section → Interpretation (well tie, horizon picking, fault interpretation, surface map, attribute analysis) → Export.
- **Colorscales**: Seismic sections use a custom diverging navy/white/maroon "Seismic" scale (`visualization/theme.SEISMIC_RWB`, since Plotly has no built-in "seismic" colorscale). Each computed attribute on the Attributes page is rendered with its own Petrel-convention colorscale via `modules/attributes.ATTRIBUTE_COLORSCALE_MAP` (e.g. Turbo for RMS Amplitude/Sweetness/Dominant Frequency, Hot for Envelope, a custom high-contrast dark-to-red scale for Variance, Greys_r for Coherence/Similarity, Plasma for Instantaneous Frequency, a cyclic scale for Instantaneous Phase, RdBu for Cosine of Phase).
- **Survey geometry**: Synthetic test data (`modules/synthetic_data.py`) exposes `dx`/`dt` controls in the Data Management page (defaults 6.25 m / 2.0 ms), with a live k-Nyquist/f-Nyquist caption, so the F-K dip filter has a sensible default wavenumber range out of the box.

## Operation registry (`modules/registry.py`)

**Every operation is declared once, in one place.** `registry.OPERATIONS` is a tuple of frozen `Operation` records — 27 entries for the 26 stages (migration has `poststack` and `pstm` variants), in the recommended order. Nothing else in the code base re-declares a stage list, a dependency, a parameter default, a label or a warning; the workflow UI, the dependency engine, parameter validation, the history labels, the generated docs and the workflow-graph data all read the registry. Pure Python, no Streamlit.

| Declared per operation | Field(s) |
|---|---|
| identity & presentation | `id`, `variant`/`variant_label`, `display_name`, `short_name`, `phrase`, `description`, `help`, `caveats`, `step_number`, `group`, `category`, `location`, `optional` |
| data types | `output_type` (a `DataType` id: `raw`, `shot`, `cdp`, `velocity`, `nmo`, `dmo`, `section`, `image`, `final`); input types = the kinds named by its `needs` (`op.input_types`) |
| required inputs / hard dependencies | `needs` (`Need`: a *kind* of data, or the output of a named operation) and `exclusions` (`Exclusion`: data that went through X cannot be read). Derived: `required_inputs`, `hard_dependencies` |
| recommended predecessors & warnings | `advice` (`Advice`: ⚠ wording, never a block). Derived: `recommended_predecessors`, `recommended_successors`, `registry.possible_warnings(key)` |
| parameter schema | `params` (`ParamSpec`: type, default from `config.py`, unit, `min` inclusive / `gt` strict *validity* bounds, `choices`, `when`, `user`) |
| validation function | `ParamSpec.check` per parameter, then the operation's own `validate` (band-pass band vs Nyquist, F-K polygon ≥ 3 vertices, velocity range) |
| processing function / QC builder | bound by `stage_ops` (`@_op`, `@_qc`) through `registry.bind`; `op.compute`, `op.extras`, `op.qc` |
| structure flags | `is_root`, `stream_base`, `alters_amplitudes`, `extra_inputs`, `executable` |

**Kept apart, as always:** the registry *declares* needs and advice; it does not evaluate them (`dependencies.py` does) and it does not hold what has been done (the state graph and the action log do). Declaration order is the recommended order — advice only.

**Derived views keep the old names** so no caller changed: `pipeline.PIPELINE_STAGES / STAGE_KIND / STAGE_INDEX / ROOT_STAGES / STAGE_GROUPS / stage_label`, `dependencies.STAGE_SPECS / get_spec / STATUS_ROWS / STREAM_BASE / STREAM_MEMBERS / SHOT_OPERATORS / SHORT / PHRASE / STAGE_LOCATION / KIND_LABELS / advice_pairs`, `dependencies.StageSpec` (= `Operation`), `state_graph.KINDS`, `stage_ops.GAIN_ALTERING_STAGES`. Do not add a second table next to them — `tests/test_registry.py` has AST guards that fail on a re-declared `Need`/`Advice`/`ParamSpec`/`Operation` or a dict literal keyed by stage ids outside `registry.py`.

**Queries:** `get(id, variant)`, `find(key)` (`"mute"`, `"migration:pstm"`), `operations(group=, category=)`, `stages()`, `variants(id)`, `stage_description/label/short_name/phrase/location`, `processed_kinds()`, `recommended_successors(id)`, `possible_warnings(key)`, `workflow_graph()` (JSON-serialisable nodes — operations and data kinds — and edges: `produces`, `consumes`, `requires`, `recommends`, `excludes`; for any future workflow visualisation), `problems()` (the consistency check — empty when the declarations agree with each other and the implementations are bound).

**Parameter validation** is `Operation.validate_params(params, survey=None, complete=True)`, called by `stage_ops.execute` after dependency checking and before the maths: unknown or missing names, `when` mismatches, type / choice / bound errors, then the operation's `validate`. A rejection raises `ValidationError`, changes nothing (no state, no id consumed) and is recorded in the history as `failure_kind="invalid"`. Bounds are **validity bounds** — what the maths can use — not the sliders' ranges; a UI slider must stay inside them.

**Generated documentation:** `docs/OPERATIONS.md` and `docs/workflow_graph.json` are written by `python -m modules.registry_docs` and checked by `tests/test_registry.py`. After any registry change, regenerate them or the test fails.

**Streamlit hot reload:** `registry.bind` is idempotent for the same function (so re-importing `stage_ops` is safe), but restart `streamlit run app.py` after editing `registry.py` — its module-level tables are built at import time.

## Data-state model

**The application tracks the data that actually exists, not a position in a pipeline.** Every operation does exactly this and nothing else:

```
INPUT STATE(S)  ──▶  OPERATION + PARAMETERS  ──▶  OUTPUT STATE
      S2                Band-pass(8–60 Hz)              S3
```

Example (ids as the app assigns them: S0 is the loaded survey, S1 the shot gathers):

```
S0 upload ─▶ S1 shot_gathers ─AGC(window=500)─▶ S2 ─Band-pass(8–60)─▶ S3 ─Mute(t_top=0.1)─▶ S4 ─CDP sort─▶ S5
                    └────────Band-pass(8–60)────────▶ S6          (a branch off S1, made later)
```

**The one rule that everything else follows from: a state is never edited and nothing is overwritten.** Running AGC again creates another state (`AGC v2`). That single rule gives order independence, repeated and skipped operations, a lineage to query for downstream/stale detection, a complete history, undo/redo as a pointer move, memoisation, and reproducibility.

### The state record (`modules/state_graph.DataState`)

| Field | Meaning |
|---|---|
| `id` | `"S7"` — stable, unique in a graph, **never reused** (not even after a discard) |
| `kind` | what the data *is*: `raw`, `shot`, `cdp`, `velocity`, `nmo`, `dmo`, `section`, `image`, `final` (`pipeline.STAGE_KIND`) |
| `op`, `variant` | the operation (stage id) that made it; `variant` for migration (`poststack`/`pstm`) |
| `params` | the **complete** parameters, canonicalised (sorted keys, numpy → Python, arrays → digest). A parameter that cannot be written down raises — what is not recordable is not reproducible |
| `inputs` | `{role: state id}` — which states it consumed (`data`; plus `velocity`, `nmo`, `detect` …) |
| `fingerprint` | 12-hex sha1 **lineage** hash of op + variant + params + the fingerprints of the inputs. Same computation on equivalent inputs ⇒ same fingerprint in any session |
| `version`, `seq`, `timestamp` | n-th state of this op / creation order / wall time |
| `notes` | the ⚠ warnings the run went ahead despite ("Run despite: …") |
| `root`, `keys` | where the arrays are (see below) — locators, not data |
| `info` | diagnostics (traces killed, polygon vertices …) — **not** part of identity |
| `touch`, `materialized` | the "current" clock; whether the arrays are in memory right now |

### The graph (`ProcessingGraph`) — deliberately small
A dict of `DataState` records plus queries; plain Python, picklable, no arrays, no framework. Streamlit's `session_state["graph"]` holds it.
- **current(kind)** — the state a downstream step reads by default: the most recently produced-*or-selected* state of that kind (`touch` is a logical clock). **latest(op)** — the newest state an operation made.
- **lineage(sid)** — the chain of states the data was built through (following `data` inputs): "what has been applied to this data?". **ancestors / dependents(sid)** — through *any* input role: "what does this depend on / what is built from it?".
- **undo(kind)** moves the current pointer to the `data` input if it is the same kind; **redo** walks back; any new state or deliberate `select` ends the redo history. Nothing is deleted or recomputed.
- **discard(sid)** removes a state and every dependent (then `stage_ops.discard_state` frees their arrays).
- **recipe(sid)**, **export()** (states, current pointers, and the action log with its integrity check). `graph.actions` is the processing history (below); `graph.system_events` holds only what the *system* did (re-materialising a state, the memory budget releasing old ones) and is not history.

### Where the arrays live (`modules/stage_ops.py`) — no input copies
```
store["graph"]        the ProcessingGraph                         (records)
store["containers"]   {root state id: {gather id: gather dict}}   (gather-based states)
store["blobs"]        {state id: {name: array/object}}            (sections, velocity fields, statics …)
store[<legacy keys>]  stack_matrix, Vfield_1, gathers, shot_gathers … = REFERENCES to the current/latest
                      state's payload, refreshed by sync_aliases(). Read-only: pages may read them, nothing writes them.
```
- A `shot_gathers` or `cdp_sort` state is a **root** and owns the native arrays (`traces`, `offsets`, ids …). A derived state adds its own arrays to the *same* gather dicts under keys suffixed with its id — `traces@S3`, `kill_flags@S5` — and records them in `state.keys`. A new state therefore costs one new array per gather and **never a copy of its input**; two states never collide on a key. Offsets are shared (`keys["offsets"] == "offsets"`) unless a state really makes new ones (data regularisation).
- Sections/images/finals/velocity fields live in `blobs[sid]`. They inherit `root` from the CDP root they came from, which is how "velocity field belongs to different gathers than this section" is detected structurally (⛔ `container_mismatch`).
- `payload_bytes`, `memory_report`, `releasable`: roots and the loaded survey cannot be released (they own the base data); everything else can.

### The compute contract (`stage_ops.execute`)
One generic path for every operation:
1. `check()` (the dependency engine) resolves `inputs` — explicit `inputs={'data': 'S4'}` > the person's remembered pick > the current state of the right kind — and raises `DependencyError` on ⛔ (⚠ never raises; it is recorded in `notes`). A blocked run changes nothing: no state, no id consumed, no array.
2. Parameters are canonicalised; any extra input roles are resolved (`extras`); `registry` **validates the parameters** against the operation's schema (a rejection is logged as `invalid` and raised, again changing nothing); the lineage **fingerprint** is computed.
3. **Memoisation:** if a state with that fingerprint exists it is re-used — made current, re-materialised if released, `last_run["reused"] = True` — instead of being computed and stored again.
4. The operation's compute function (`op.compute`, bound with `@_op`) runs the *unchanged* maths on the inputs' payloads through a `Run` helper (`run.in_key(role)`, `run.key(name)` → `name@<new id>`, `run.container()`, `run.put(...)`) and returns `(keys, info)`. If it raises, everything it wrote is freed.
5. The state is added to the graph, aliases are synced, the memory budget (`config.STATE_MEMORY_BUDGET_MB`, default 0 = off) is enforced.

### How the required capabilities map onto the model
| Requirement | How it works | Tested in |
|---|---|---|
| Operation order independence | Each op reads the *current state of its kind* (or any state named), never "the stage before it" | `test_stage_ops.py` (shot operators in a non-textbook order; both task orders), `test_dependencies.py` (all 24 orders) |
| Repeated operations | New state, `version` 2, input = the first result; ⚠ `repeated_operation` with a hint to run it on the state *before* instead | `test_repeating_an_operation…`, `…applied_to_the_state_before_instead` |
| Skipped operations | A need is a *kind* of data, not a stage; optional stages never create a requirement | `test_skipping_every_optional_stage…` |
| Branching / operating on an earlier state | `inputs={'data': 'S1'}` or the "Choose a different input state" picker; both branches coexist | `test_two_branches_from_the_same_state_coexist` |
| Downstream dependency detection | `graph.dependents(sid)` through any input role; the states panel lists what a discard would also remove | `test_the_dependents_of_a_state_are_found_through_any_input_role` |
| Stale-state detection | `dependencies.stale_info`: a state is stale when the input each *tracked* `Need` would resolve to today differs from the one it recorded — reasons `behind`, `replaced`, `rolled_back`, `diverged`, `inherited` (transitive). Nothing is discarded: stale = valid data not built from the latest inputs | `test_rerunning_velocity_keeps_the_old_states…`, `test_undoing_marks_…` |
| Processing history | The immutable action log (`graph.actions`) — *not* the graph in `seq` order; see **Processing history** below. Report, state-graph JSON, action-log CSV/JSON and per-state recipe JSON on the Export page | `test_action_log.py`, the history tests in `test_stage_ops.py`, `test_history_and_exports…` |
| Undo / redo | Pointer moves only (`stage_ops.undo/redo`, states panel); undone states stay in the graph; released states are re-computed on the way | `test_undo_and_redo_only_move_the_pointer`, UI tests |
| Reproducibility | `graph.recipe(sid)` → `stage_ops.replay_recipe(store, sid)` re-runs the recorded ops with their parameters in a **new** store and checks every lineage fingerprint (`GraphError` if the survey or the maths differ). Manual velocity picks / excluded CDPs are recorded in full as parameters; the `upload` state records source, size, dt and a data digest | `test_a_recipe_replays_to_the_identical_result` |
| Memory | no input copies; `release_payload` / `materialize` (re-compute from the record); optional LRU budget that never releases the current state of any kind | `test_release_then_materialize…`, `test_a_memory_budget…` |

## Processing history (the immutable action log)

**The history is a record of what the person did, in the exact order they did it.** It is a separate object from the data states (`modules/action_log.py`, `graph.actions`) so that nothing about the data — lineage, branching, undo, discard, the recommended order — can change what it says.

```
AGC → Band-pass → Mute → F-K filter → Band-pass again      (what the user ran)
#1 AGC · #2 Band-pass · #3 Mute · #4 F-K · #5 Band-pass    (what the history says — always)
```

### Three questions, three places — never mix them up
| Question | Answer comes from | Order |
|---|---|---|
| What **is** the data now? (*current processing state*) | `dependencies.current_processing_state(wf)` ← states + current pointers | a lineage, e.g. `Raw → AGC → Band-pass` |
| What is **usually** done? (*recommended workflow*) | the registry's declaration order (`pipeline.PIPELINE_STAGES`), status panels | the textbook order |
| What was **actually done**? (*history*) | `graph.actions` (`ActionLog`) | execution order (action number) |

After an *undo* the current state moves back and the history gains an `Undo` entry; running steps in a non-textbook order changes the history only by appending. The UI labels each (sidebar: "Current processing state", "Recommended workflow", "Processing history"; Home page and the History page likewise).

### The entry (`Action`, frozen)
`n` (action number, 1, 2, 3 … — **this is the order**, not the clock), `timestamp`, `session` (dataset number), `kind` (`operation` | `undo` | `redo` | `select` | `discard` | `release`), `action`/`label`/`variant` (which operation), `params` (complete, canonical JSON), `inputs` (role → input state id; for navigation, the state left), `output` (state produced / re-used / moved to; `None` on failure), `status` (`success`/`failure`), `reused`, `failure_kind` (`blocked` ⛔, `invalid`, `error`), `error`, `qc` (RMS in → out, non-finite samples, the operation's own scalar diagnostics, velocity range), `warnings` (the ⚠ the run went ahead despite), `affected` (discard: every state removed), `detail`, and the hash chain (`prev_hash`, `hash`).

### What is, and is not, an entry
- **Entries:** every execution of an operation — a repeat is another entry, and so is an identical re-run that found an existing state and re-used it (`reused=True`); every blocked / invalid / failed attempt (a failure never consumes a state id); loading data and building gathers (`upload`, `shot_gathers`); undo, redo, make-current that actually move a pointer; discard; the person's *Free memory*.
- **Not entries (`graph.system_events`):** the memory budget releasing old states, and re-materialising a released state. The person did not do these. No-op navigation (undo with nothing to undo) is not logged either.
- Loading a new survey starts a new **dataset session**: states restart at `S0`, the log keeps the earlier session's entries (action numbers continue), `view(session=…)` / the UI toggle separate them.

### Immutability
- `Action` is a frozen dataclass of immutable values (params and QC are stored as JSON text, inputs as a tuple of pairs); `ActionLog` has no method that edits, reorders or removes an entry — only `record` and `new_session`.
- Python cannot forbid a determined edit, so the log is **hash-chained** (each entry's SHA-1 covers its content and the previous hash). `ActionLog.verify()` reports edited, removed, inserted or reordered entries; the History page shows the result and the JSON export carries it.
- Discarding a state removes the state, **not** the entries that created or used it; the later `discard` entry names what was removed and `ActionLog.discarded_by(entry)` links them.

### Display order
`view(order="oldest"|"newest", kinds=…, session=…, issues_only=…)` returns a **sorted copy**; sorting is by action number. The user setting (`st.session_state["history_order"]`, `ss.history_order()` / `ss.set_history_order()`, default *oldest first*) is shown by `wf.history_order_control(key)` on the sidebar, Home and History pages — one shared setting; each control must set its own widget key from the setting *before* instantiating the widget. It affects only how lists are drawn: the log, the action numbers, and **every export** (report, JSON, CSV) stay chronological.

### Where entries come from
Only `modules/stage_ops.py` writes the log: `execute()` (success, reuse, blocked, invalid parameters — schema or operation validator —, compute error), `_refuse()` (argument validation in a `run_*` wrapper, via `_precheck` for the mute and F-K arguments), `run_load_survey`, `select_state`/`undo`/`redo`/`discard_state`, `release_payload(actor="user")`. A new operation therefore needs **no** logging code — if it goes through `execute()` it is recorded with its parameters, inputs, output, QC and warnings. Pages never call `ActionLog.record`.

### Invariants — do not break these
- **Never mutate a state's arrays in place** and never write into another state's keys. A compute function writes only under `run.key(...)`.
- **Never overwrite or reuse a state id**; never edit `params`/`inputs`/`fingerprint` after creation (only `touch`, `materialized`, `keys`, `info`, `notes` may change).
- **Parameters must be complete and canonical.** If a result depends on it (a polygon, manual picks, a window), it is a parameter. Diagnostics go in `info`.
- If an op's result depends on a state other than its `data` input (e.g. `trace_edit` measures its kill mask on amplitude-preserving data upstream of any AGC), record that state as an extra input role (`extras=` on `@_op`, and `extra_inputs` on its registry entry — `registry.problems()` checks they agree) so identity and staleness see it.
- **Declare an operation's metadata only in `modules/registry.py`.** No second copy of a dependency, advice pair, stage list, label, kind, parameter default or bound in a view, the engine, `stage_ops` or a doc. Parameter defaults come from `config.py` (a `run_*` wrapper and the registry's `ParamSpec` read the same constant); `tests/test_registry.py` proves each wrapper records exactly the schema's defaults.
- **Pages read via `wf.result()` / `ss.*`, never by guessing keys.** The legacy aliases exist for convenience and always reflect the current/latest state; new code should take arrays from the state (`ss.gathers(state)`, `ss.section(state)`, `ss.blob(state, name)`, `state.keys[...]`).
- `wf.result()` returns `None` (with a "Re-compute" note) while the state, or the state it was made from, is released — a page drawing itself must not silently undo a deliberate *Free memory*.
- Every state a page displays must have its *input* state's arrays too (before/after panels) — `wf.result` checks both.
- **Never edit, reorder, merge, or delete an action-log entry, and never record one outside `stage_ops`.** Do not derive the history from the state graph (or the current state from the history), and do not let the recommended order, the data lineage, or the display setting influence the order of `graph.actions`. A feature that needs "history" in another order (newest first, grouped, filtered) uses `ActionLog.view(...)`.
- `st.session_state` is the store; keep `modules/` free of Streamlit imports (`stage_ops` takes any mutable mapping, which is how the tests run the real pipeline with a plain dict from `stage_ops.new_store(survey)`).

### Adding an operation
1. Add any new parameter default to `config.py` (`<NAME>_DEFAULT`).
2. Declare an `Operation` in `modules/registry.py` at its recommended position in `OPERATIONS`: id, names, `step_number`, `group`, `category`, `output_type`, `location`, `description`/`help`; its `needs` (`Need` — genuine inputs, *kinds* of data; `track=True` for those whose supersession should make a finished state stale), `advice` (`Advice` — recommendations; pairs must agree with the declaration order and the recommended order must stay warning-free), `exclusions` if some lineage cannot be read, `params` (`ParamSpec`s with `cfg` defaults and *validity* bounds), `validate` if a rule spans parameters (put the check function in `modules/validation.py`). If it transforms a stream it joins `STREAM_MEMBERS` automatically through its `output_type`. New data kind → add a `DataType`.
3. Write the compute function with `@_op("stage_id")` in `stage_ops.py` — read via `run.in_key(...)`, write via `run.key(...)`/`run.put(...)`, return `(keys, info)` — and, if the operation has scalar diagnostics for the history, an `@_qc("stage_id")` function. Add a `run_<stage>` wrapper that builds the **complete** parameter dict from the same `cfg` constants and calls `execute`. Add any legacy alias to `sync_aliases` and `pipeline.ALIAS_KEYS`.
4. Add a view with `wf.show_status` / `wf.run_button` / `wf.execute(stage_ops.run_…)`, display with `wf.result(...)`, and `wf.states_panel([...kinds…], key)`; slider defaults come from the same `cfg` constants and slider ranges stay inside the schema's bounds. The action log needs nothing from the view: `execute()` records the run (and any refusal) itself. A `run_*` wrapper that rejects its own arguments before reaching `execute` must do so with `_refuse(...)` (or `_precheck`) so the attempt is recorded.
5. Add the operation to the `CALLS` table in `tests/test_registry.py` (the test that its wrapper records exactly the schema defaults fails otherwise), then regenerate the docs: `python -m modules.registry_docs`.
6. Run the gate. `registry.problems(require_implementations=True)` (run by the tests) fails if an operation lacks a bound compute function, its advice contradicts the recommended order, a default fails its own schema, or `extras` and `extra_inputs` disagree. Re-run the equivalence check against the previous version if any maths is touched.

## Known, deliberate design decisions
(See README.md "Design notes" for the full write-up.)
- **The registry declares; it does not decide.** Needs, advice and exclusions are data in `registry.py`; evaluating them against the current states is `dependencies.py`; what happened is the graph and the log. "Warnings" in the registry are the advice (and the generic fallback / order-inversion / repeat / stale conditions the engine can raise from it — `registry.possible_warnings`), not a free-standing list.
- **Schema bounds are validity bounds.** They are the edge of what the maths can use (an AGC window of 0 samples, a negative frequency, a zero velocity), checked before the run so the failure is a clear message and a history entry rather than a crash or silent NaN. They are not the sliders' ranges, and a bound that refuses a value the maths accepts is a bug (the F-K attenuation coefficient accepts 0.0: the maths clips it to 0–1, and the slider reaches it).
- **The history is its own object, not a view of the graph.** The state graph answers "what exists / how was it made" and is rewritten by discard and reset; the action log answers "what did the person do" and only ever grows. A reused (memoised) run adds a log entry but no state; a failed run adds a log entry but no state; a discard removes states but no entries.
- **History order is the action number.** Timestamps are recorded but never used for ordering (they can tie, and a clock can move).
- NMO uses `np.interp(tn, t0_grid, trace)` as the forward-warp convention; `inverse_nmo_correction` uses the swapped convention (`np.interp(t0, tn_grid, trace)`), verified with a direct forward-then-inverse round-trip test. Getting this right matters because `nmo_correction` runs at both velocity-analysis passes and `inverse_nmo_correction` feeds the 2nd pass.
- The velocity auto-picker is a confidence-gated local walk along the semblance ridge rather than a whole-path dynamic-programming picker — see `modules/velocity.py`.
- Post-stack random noise attenuation uses a spatial (across-CDP) median filter as a stand-in for F-X deconvolution, which needs higher-fold data than a small synthetic/demo line provides.
- Deconvolution runs as the last pre-stack-conditioning step (after AGC, trace edit, mute, balance, F-K filter, and multiple suppression) to match the requested shot-domain processing order.
- **Re-running never replaces.** The old model discarded everything downstream of a re-run; the state model keeps every state and *flags* what was built from superseded inputs. Delete deliberately with *Discard*, which also removes dependents.
- **A re-run's default input is the current data.** Re-running AGC on data that already contains AGC applies it twice (with a ⚠). To redo it with new parameters, run it on the state before it (the ⚠ names it, and the input picker offers it).
- **Velocity manual picks / excluded CDPs are parameters of the *next* velocity run** (a new state), not edits applied retroactively; the page shows "✏️ Manual picks / exclusions have changed since this velocity state was made …" when they differ from what the state recorded.
- **Stacking drops the maths' fixed per-gather by-products** (`stack_dmo`, `fold`, `stack_naive_mean`) so two stacks cannot collide; they are exactly the rows of `stack_matrix`/`fold_matrix`/`naive_matrix`, which a state keeps as blobs.
- Each shot-domain state holds one full array per gather (≈ the size of the raw data). Use *Free memory* or `STATE_MEMORY_BUDGET_MB` on long sessions; released states are re-computed from their record, bit-for-bit.
