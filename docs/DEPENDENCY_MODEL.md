# The dependency model

How Seismic Playground decides what can run, what is merely advisable, and
what has already been applied to your data.

The processing workflow used to be a strictly linear pipeline: each page
refused to open until the previous stage's "done" flag was set, and
re-running a stage threw away everything after it in the list. That fused
several different ideas into one. They are now separate:

| Idea | What it answers | Where it lives | Can it block a stage? |
|---|---|---|---|
| **Recommended order** | "What is the usual sequence?" | `modules/pipeline.PIPELINE_STAGES` | No — display and documentation only |
| **Genuine dependencies** | "What input does this computation need to exist?" | `modules/dependencies.STAGE_SPECS` → `Need` | **Yes (⛔)** |
| **Recommendations** | "What is scientifically preferable before/after this?" | `STAGE_SPECS` → `Advice`, plus the order / repeat / stale checks | No — a ⚠ and a *Continue?* prompt |
| **Data states** | "What data exists, and how was each piece made?" | `modules/state_graph.py` (`DataState`, `ProcessingGraph`) | — |
| **Current processing state** | "What is the data *now*?" | `dependencies.current_processing_state` — derived from the states and their current pointers | — |
| **Processing history** | "What did the person actually do, in what order?" | `modules/action_log.py` — an immutable, chronological action log | — |

The last three rows are the ones most easily confused. The **history** is
what was *done*, in the exact order it was done (`AGC, Band-pass, Mute, F-K,
Band-pass` stays exactly that — nothing is reordered to match the recommended
order, repeats are separate entries, failed attempts and undo/redo/discard are
entries too). The **current processing state** is what the data *is* now (e.g.
`Raw → AGC → Band-pass → Mute`, after an undo it moves back while the history
only grows). The **recommended order** is only what is *usually* done. None is
computed from another, and the app shows them in separate places.

Every operation is `INPUT STATE → OPERATION + PARAMETERS → OUTPUT STATE`; a
state is never edited or overwritten. The state model itself (identity,
fingerprints, where the arrays live, undo/redo, memory, reproducibility) is
documented in **`CLAUDE.md` → "Data-state model"**; this file covers the
rules that decide ✓ / ⚠ / ⛔ on top of it.

## What a person sees

| Symbol | Meaning | Behaviour |
|---|---|---|
| ✓ **Available** | every required input is present, nothing to flag | Run button works |
| ⚠ **Recommended prerequisite not performed** | runnable, but the usual practice was not followed | Run button asks *Continue?* with the reason; **Continue anyway** runs it, **Cancel** does nothing |
| ⛔ **Required input missing** | cannot run: an input the computation needs does not exist | the page names the missing input and *where to produce it*; the Run control is disabled/absent |
| ✅ **done** | the operation has produced at least one state | running it again makes a **new state**; the old one is kept (and marked stale if something was built from it) |

Every processing page shows its operation's status as a banner, plus **📥
Reads: …** — the data state(s) the run will consume — and a *Choose a
different input state* selector when there is a real choice. The sidebar
("Recommended workflow") and Home page show all operations at once. The current
data is printed as a chain, e.g. `📍 Current shot-domain data: Raw → Band-pass → F-K`.
After a run the page says which state it produced and from which
(`✅ Produced Band-pass (S3) from Raw (S1).`), or, if the identical
computation already existed, `♻ Identical to … — reusing it`.

Example ⚠ message (AGC with no band-pass applied):

> Band-pass filtering is commonly applied before amplitude recovery (AGC), but
> it has not been applied to the current data. AGC amplifies everything in the
> trace, including out-of-band noise, so the gain is driven by the useful
> reflection band only if the data were filtered first.

## How an operation finds its input

An operation does not read "whatever the stage before it in the list
produced". Each `Need` of a stage lists *candidates*, tried in order of
preference:

* `@shot`, `@cdp`, `@section` … — "the **current** state of that kind of data"
  (the most recently produced or selected one);
* a stage id (`vel1`, `nmo_final`) — "the latest state that stage produced".

The default input is the most preferred candidate that exists. A person (or a
test) can name any other usable state: `inputs={'data': 'S1'}`, or the input
selector on the page. The new state records exactly what it read in
`state.inputs`.

| Kind | Base | Operations that take the current state of the kind as input and produce a new state of it |
|---|---|---|
| `shot` | Build Shot Gathers | de-spike, band-pass, AGC, trace edit/kill, mute, balance, F-K, multiple suppression, deconvolution |
| `cdp` | CDP Sort | residual statics, inverse NMO, data regularisation (pre-NMO gathers) |
| `section` | Stack | random noise attenuation, post-stack deconvolution, time-variant filter |

So `Raw → AGC → Band-pass → Mute` and `Raw → Band-pass → F-K` both work: each
step reads whatever the shot data currently is, and the state records exactly
what it read.

Operations that *consume* a kind but are not members of it: velocity analysis,
NMO (pass 1 and final), DMO, stack, migration, final scaling. DMO is a side
branch: it consumes the pre-NMO gathers and velocities, and produces NMO-domain
data that only inverse NMO reads.

"Has band-pass been applied to this data?" is a **lineage query**
(`graph.lineage(state)`), not a global flag — so the answer is about the data
the step is about to read.

**Same gathers.** Gather-based states keep their arrays in the container of
their *root* (the shot-gather or CDP-sort state they descend from). Two inputs
of one operation must share a root: a velocity field picked on one CDP sort
cannot flatten the gathers of another, and a section stacked from one cannot be
migrated with the velocities of another. Resolution takes `data` first and
prefers other inputs from the same root; if only mismatching ones exist the
operation is ⛔ (`container_mismatch`) and says which gathers disagree.

## Genuine dependencies (⛔)

These are properties of the computation, not of convention.

| Stage | Needs (preferred → fallback) |
|---|---|
| Build Shot Gathers | a loaded survey: Load |
| Band-Pass Filter | shot gathers: current **shot** data |
| Amplitude Recovery (AGC) | shot gathers: current **shot** data |
| De-spike | shot gathers: current **shot** data |
| Trace Edit / Kill | shot gathers: current **shot** data *(+ records the amplitude-preserving state its mask was measured on — see below)* |
| Mute | shot gathers: current **shot** data |
| Trace Balancing | shot gathers: current **shot** data |
| F-K Dip Filter | shot gathers: current **shot** data |
| Multiple Suppression | shot gathers: current **shot** data |
| Deconvolution | shot gathers: current **shot** data |
| CDP Sort / Binning | traces to sort (a loaded survey): current **shot** data → Load |
| Velocity Analysis — Pass 1 | CDP gathers: CDP sort |
| NMO Correction — Pass 1 | CDP gathers: CDP sort<br>pass-1 stacking velocities: Velocity 1 |
| Residual Statics | CDP gathers: current **cdp** data<br>NMO-flattened gathers (NMO Pass 1): NMO 1<br>structural check (statics: ⛔ on regularised gathers) |
| DMO Correction | CDP gathers: current **cdp** data<br>pass-1 stacking velocities: Velocity 1 |
| Inverse NMO | DMO-corrected gathers: DMO<br>pass-1 stacking velocities: Velocity 1 |
| Data Regularisation | CDP gathers: current **cdp** data |
| RMS Velocity Analysis (Pass 2) | CDP gathers: current **cdp** data |
| Final NMO | CDP gathers: current **cdp** data<br>stacking velocities: Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Stack | NMO-corrected CDP gathers: Final NMO → NMO 1 *(⚠ if not the first)* |
| Random Noise Attenuation | a stacked section: current **section** data |
| Deconvolution (post-stack) | a stacked section: current **section** data |
| Time-Variant Filter | a stacked section: current **section** data |
| Migration (poststack) | a stacked section: current **section** data<br>a stacking-velocity field (the migration velocity): Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Migration (pstm) | pre-stack CDP gathers with their offsets preserved: current **cdp** data<br>a stacking-velocity field (the migration velocity): Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Final Scaling | a section to scale: Migration → current **section** data *(⚠ if not the first)* |

Notes:

* **Velocity analysis needs CDP gathers** (it scans offset-sorted gathers).
* **NMO needs a velocity function.** Final NMO uses pass-2 velocities, falling
  back to pass 1 with a ⚠.
* **Stack needs moveout-corrected gathers** (Final NMO, or NMO pass 1 with a ⚠):
  without them reflections stack out of phase.
* **Post-stack migration needs a section *and* a velocity field. Pre-stack
  migration (PSTM) needs pre-stack, offset-preserving CDP gathers *and* a
  velocity field — and no stack**, since it migrates and stacks in one pass. It
  reads the current `cdp` data, never NMO-corrected data. Data regularisation
  is a recommendation for it, not a requirement.
* **Residual statics cannot run on regularised gathers** (⛔, "geometry"):
  statics are surface-consistent — they need each trace's own shot and
  receiver, and regularised gathers hold interpolated traces on a synthetic
  offset grid. Apply statics before regularisation — or, now that every state is
  kept, run them on the gathers *before* the regularisation state
  (`inputs={'data': <that state>}`).
* **Shot-domain steps need shot gathers**; **CDP sort needs only the survey**
  (it re-groups traces using headers; with no shot gathers it sorts the raw
  traces).

## Recommendations (⚠)

Each is "X is normally applied before Y". None blocks.

| Stage | Recommended in the data it reads |
|---|---|
| Amplitude Recovery (AGC) | all of: Band-pass |
| Multiple Suppression | all of: Band-pass, AGC |
| Deconvolution | all of: Band-pass, AGC<br>any of: Mute, F-K |
| CDP Sort / Binning | any of: De-spike, Band-pass, AGC, Trace kill, Mute, Balance, F-K, Multiple suppression, Decon |
| DMO Correction | all of: Statics |
| RMS Velocity Analysis (Pass 2) | any of: Statics, Inverse NMO, Regularised |
| Deconvolution (post-stack) | all of: RNA |
| Time-Variant Filter | all of: Post-stack decon |
| Migration (poststack) | all of: RNA, Post-stack decon, TVF |
| Migration (pstm) | all of: Regularised |

Checks that produce a ⚠ (`Issue.kind`):

* `missing_recommended` — a recommended predecessor is not in the lineage of the data this step reads.
* `order_inversion` — a step normally applied *after* this one is already part of the data (e.g. band-pass after AGC).
* `fallback_input` — the input comes from a less-preferred source (stack of pass-1 NMO; pass-1 velocities for migration; un-migrated section for final scaling).
* `repeated_operation` — this operation is already in the lineage of the data it would read, so it would be applied twice. The message names the state before it, so the step can be redone with new settings on *that* instead.
* `orphaned_step` — a valid step of the same kind whose result is not in the data this step will read (arises when DMO → inverse NMO carries on from gathers that statics did not touch). Work that was merely *undone* is not an orphan.
* `stale_input` — the chosen input state is out of date (see below).

Genuine, blocking kinds: `missing_input`, `container_mismatch` (inputs from
different gathers), `invalid_input` (an explicitly named state that is not
usable or no longer exists), `geometry`.

**The advice must agree with the recommended order.** `tests/test_dependencies.py`
asserts that every (earlier, later) pair implied by an `Advice` respects
`PIPELINE_STAGES`, and that following the recommended order produces **zero**
warnings at every step.

## Re-running, repeating, branching and going back

* **Nothing is overwritten and nothing is discarded implicitly.** Re-running
  an operation, with new parameters or on another input, makes a new state
  (`AGC v2 (S7)`). The earlier state, and everything built from it, stays valid
  and in memory.
* **An identical re-run is not repeated.** Same operation + parameters +
  inputs ⇒ same lineage fingerprint ⇒ the existing state is made current again
  and nothing is recomputed or stored a second time.
* **Stale, not discarded.** A state is *stale* when an input it was built from
  has since been superseded — `behind` (that input was processed further),
  `replaced` (the same operation was re-run), `rolled_back` (current data was
  undone to before it), `diverged` (a more preferred source now exists, e.g.
  Final NMO after a stack of NMO pass 1) or `inherited` (it was built from a
  stale state). Pages show a ⟳ note, running something on a stale state gives a
  ⚠, and re-running the affected steps makes fresh states. Re-running NMO pass 1
  keeps DMO, which never read it.
* **Branching.** Pick an earlier state as the input and the result becomes a
  second branch off it; both coexist. *Make current* chooses which branch the
  next step continues from.
* **Undo / redo** move the *current* pointer of a kind of data back to the
  state it was made from (and forward again); no data is deleted or recomputed.
  A new operation ends the redo history.
* **Discard** is the only thing that deletes: it removes a state and everything
  built on it (the panel lists what that includes) and frees their arrays.
* **Free memory** drops a state's arrays but keeps its record; they are
  re-computed, bit-for-bit, from the recorded operation and parameters when
  needed.
* Loading a new survey starts a new graph.

## Kill detection and gain

Trace-kill detection is energy-based, so it is blind after AGC or balancing has
equalised trace levels. It therefore measures the data from just *before* the
first gain-altering step in the data's lineage, and applies the resulting mask
to the data actually being processed (`stage_ops.kill_detection_stage`). That
detection state is recorded as a second input of the trace-edit state
(`inputs["detect"]`), so the result — the set of killed traces — is the same
whether or not AGC ran first (tested), and the record says what it was measured on.

## Code map

| File | Role |
|---|---|
| `modules/pipeline.py` | stage registry (recommended order), `STAGE_KIND`, store defaults, `ALIAS_KEYS`, report lines |
| `modules/state_graph.py` | `DataState`, `ProcessingGraph`: ids, fingerprints, current pointers, lineage/dependents, undo/redo, discard, recipe, export — pure Python, no arrays |
| `modules/dependencies.py` | the engine: `Need`, `Advice`, `StageSpec`, `STAGE_SPECS`, `Workflow`, `check_stage`, `stale_info` — pure Python |
| `modules/stage_ops.py` | `execute()` + one compute function and `run_*` wrapper per operation; payload storage, memoisation, release/materialise, replay — pure Python, takes any mutable mapping as its store; calls the **unchanged** maths |
| `modules/validation.py` | `DependencyError` (a `ValidationError`) raised when a ⛔ stage is run anyway |
| `utils/session_state.py` | thin adapter over `st.session_state` (schema, `graph`, `check`, `current`, `result`, `undo`, `redo`, `discard` …) |
| `utils/workflow_ui.py` | banners, the *Continue?* run button, input selector, data-states panel, status panels |
| `views/*.py` | pages: call `wf.show_status` / `wf.run_button` / `wf.execute(stage_ops.run_…)` and display with `wf.result` |

The processing mathematics in `modules/{filtering,preprocessing,nmo,velocity,
statics,regularization,stacking,migration}.py` was **not changed**. With the
recommended order, the app produces results bit-for-bit identical to the
original linear version: the stack, post-stack, migrated and final sections,
both velocity fields, residual statics, regularised offsets and every per-gather
array of the shot and CDP gathers were compared with exact equality after a full
end-to-end run. (The only arrays not retained are three per-gather by-products of
stacking — `stack_dmo`, `fold`, `stack_naive_mean` — which are exact rows of the
stored stack, fold and naive matrices.)

### Adding a stage

See **`CLAUDE.md` → "Adding a stage"** for the full checklist. In short: a
`Stage` in `PIPELINE_STAGES` and its `STAGE_KIND`; a `StageSpec` in
`STAGE_SPECS` (`Need`s = genuine, `Advice` = recommended); an `@_op` compute
function and a `run_<stage>` wrapper in `stage_ops.py` that records the complete
parameters; a view using `wf.show_status` / `wf.run_button` / `wf.execute`. The
registry tests fail if a stage lacks a spec or an executor, or if its advice
contradicts the recommended order.

## Tests

| File | Covers |
|---|---|
| `tests/test_state_graph.py` | the graph with no data: ids never reused, fingerprints, current pointers, undo/redo, lineage, dependents, discard, recipe |
| `tests/test_dependencies.py` | the engine on a bare graph: golden path has zero warnings; both task example orders; every optional stage skippable; shot operators in all 24 orders; every genuine dependency (⛔); recommendation wording; repeated operations and branching; stale reasons; inputs from different gathers; advice consistent with the recommended order |
| `tests/test_stage_ops.py` | the same through the real maths, and the state model: non-standard orders, skipped stages, `DependencyError` leaves no trace, repeat / branch / memoise, no input copies, release and re-materialise bit-for-bit, memory budget, undo / redo, discard, stale detection after real runs, trace-edit `detect` input, manual velocity picks as parameters, replayable recipes, PSTM without stack/regularisation, export |
| `tests/test_app_nonlinear.py` | the app driven headlessly: ⛔ messages and absent controls, ⚠ *Continue?* / Cancel, the two task orders, a skip-everything run to a final section, which state a run produced/consumed, identical re-runs reusing a state, choosing an earlier input, stale ⟳ notes, undo / redo / discard / free memory from the page |
| `tests/test_pipeline.py`, `tests/manual_pipeline_check.py`, `tests/test_app_flow.py` | unchanged validation gate (maths and the recommended-order walkthrough) |

## Grounding and limits

* The recommended predecessors for deconvolution (a wide band-pass to remove
  random noise, coherent noise eliminated, and amplitude/absorption compensation
  so the trace is closer to stationary) follow the *Preconditioning* slide of the
  project's *4 Deconvolution* notes. *6 PSTM Migration* and *7 Stack migration*
  explain why a CMP/DMO stack is not a zero-offset section and so why migration
  before stack exists as a separate implementation; they do not enumerate input
  requirements, which here come from what each implemented algorithm actually
  reads (a stacked section for post-stack migration; pre-stack offset-preserving
  gathers for PSTM). `preprocessing.pdf` and `5 Velocity_Analysis.pdf` are
  image-only (no extractable text) and were **not** machine-read, so the
  band-pass/AGC ordering advice and the velocity-analysis advice rest on
  standard practice and the existing README rather than on those two files —
  worth checking against them.
* Recommendations are pairwise ("RNA before post-stack deconvolution",
  "post-stack deconvolution before TVF"); there is no direct "RNA before TVF"
  advice, so running TVF then RNA gives no ⚠.
* State identity includes the *parameters* (so changing a parameter and
  re-running makes a new state), but staleness is judged on *inputs*: a state
  whose input was re-run is flagged, a state that merely has older parameters
  than the form currently shows is not (velocity manual picks are the exception
  and show a ✏️ note, because they are edited outside the Run form).
* Each state keeps its own full arrays (about the size of the raw data per
  shot-domain state). `STATE_MEMORY_BUDGET_MB` in `config.py`, or *Free memory*
  on the states panel, releases old states; they come back by re-computation,
  not from disk, so a released state costs CPU time instead of memory.
* Fingerprints of large arrays (more than 10⁶ elements, e.g. the loaded
  survey) hash a strided sample plus shape and dtype, not every byte.
