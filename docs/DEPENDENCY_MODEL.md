# The dependency model

How Seismic Playground decides what can run, what is merely advisable, and
what has already been applied to your data.

The processing workflow used to be a strictly linear pipeline: each page
refused to open until the previous stage's "done" flag was set, and
re-running a stage threw away everything after it in the list. That fused
four different ideas into one. They are now separate:

| Idea | What it answers | Where it lives | Can it block a stage? |
|---|---|---|---|
| **Recommended order** | "What is the usual sequence?" | `modules/pipeline.PIPELINE_STAGES` | No — display and documentation only |
| **Genuine dependencies** | "What input does this computation need to exist?" | `modules/dependencies.STAGE_SPECS` → `Need` | **Yes (⛔)** |
| **Recommendations** | "What is scientifically preferable before/after this?" | `modules/dependencies.STAGE_SPECS` → `Advice` | No — a ⚠ and a *Continue?* prompt |
| **Current data state** | "What has been applied to the data this step would read?" | derived from the history by `WorkflowState` | — |
| **Processing history** | "What was run, in what order, on what?" | `HistoryEntry` list (`session_state["history"]`) | — |

(The last two are the same list viewed two ways: the history is the log;
the data state is what you get by following its input edges.)

## What a person sees

| Symbol | Meaning | Behaviour |
|---|---|---|
| ✓ **Available** | every required input is present, nothing to flag | Run button works |
| ⚠ **Recommended prerequisite not performed** | runnable, but the usual practice was not followed | Run button asks *Continue?* with the reason; **Continue anyway** runs it, **Cancel** does nothing |
| ⛔ **Required input missing** | cannot run: an input the computation needs does not exist | the page names the missing input and *where to produce it*; the Run control is disabled/absent |
| ✅ **done** | already run | running it again replaces it and discards only the steps built on it |

Every processing page shows its stage's status as a banner; the sidebar
("Workflow status") and Home page show all stages at once. The data state is
printed as a chain, e.g. `📍 Current shot-domain data: Raw → Band-pass → F-K`.

Example ⚠ message (AGC with no band-pass applied):

> Band-pass filtering is commonly applied before amplitude recovery (AGC), but
> it has not been applied to the current data. AGC amplifies everything in the
> trace, including out-of-band noise, so the gain is driven by the useful
> reflection band only if the data were filtered first.

## Data streams: why "the previous stage" is not a thing any more

A stage does not read "whatever the stage before it in the list produced". It
reads the **head of a data stream** — the most recent valid output of the
stages that transform that data:

| Stream | Base | Members (each takes the stream's head as input, and becomes the new head) |
|---|---|---|
| `shot` | Build Shot Gathers | de-spike, band-pass, AGC, trace edit/kill, mute, balance, F-K, multiple suppression, deconvolution |
| `cdp` | CDP Sort | residual statics, inverse NMO, data regularisation (pre-NMO gathers) |
| `section` | Stack | random noise attenuation, post-stack deconvolution, time-variant filter |

So `Raw → AGC → Band-pass → Mute` and `Raw → Band-pass → F-K` both work: each
step reads whatever the shot data currently is, and the history records
exactly what it read (`HistoryEntry.consumed`, `input_key`, `output_key`).

Stages that *consume* a stream but are not members of it: velocity analysis,
NMO (pass 1 and final), DMO, stack, migration, final scaling. DMO is a side
branch: it consumes the pre-NMO gathers and velocities, and produces
NMO-domain data that only inverse NMO reads.

"Has band-pass been applied to this data?" is an **ancestry query**
(`WorkflowState.data_ancestry`), not a global flag — so the answer is about
the data the step is about to read.

## Genuine dependencies (⛔)

These are properties of the computation, not of convention.

| Stage | Needs (preferred → fallback) |
|---|---|
| Build Shot Gathers | a loaded survey: Load |
| Band-Pass Filter | shot gathers: head of the **shot** stream |
| Amplitude Recovery (AGC) | shot gathers: head of the **shot** stream |
| De-spike | shot gathers: head of the **shot** stream |
| Trace Edit / Kill | shot gathers: head of the **shot** stream |
| Mute | shot gathers: head of the **shot** stream |
| Trace Balancing | shot gathers: head of the **shot** stream |
| F-K Dip Filter | shot gathers: head of the **shot** stream |
| Multiple Suppression | shot gathers: head of the **shot** stream |
| Deconvolution | shot gathers: head of the **shot** stream |
| CDP Sort / Binning | traces to sort (a loaded survey): head of the **shot** stream → Load |
| Velocity Analysis — Pass 1 | CDP gathers: CDP sort |
| NMO Correction — Pass 1 | CDP gathers: CDP sort<br>pass-1 stacking velocities: Velocity 1 |
| Residual Statics | CDP gathers: head of the **cdp** stream<br>NMO-flattened gathers (NMO Pass 1): NMO 1<br>structural check (statics: ⛔ on regularised gathers) |
| DMO Correction | CDP gathers: head of the **cdp** stream<br>pass-1 stacking velocities: Velocity 1 |
| Inverse NMO | DMO-corrected gathers: DMO<br>pass-1 stacking velocities: Velocity 1 |
| Data Regularisation | CDP gathers: head of the **cdp** stream |
| RMS Velocity Analysis (Pass 2) | CDP gathers: head of the **cdp** stream |
| Final NMO | CDP gathers: head of the **cdp** stream<br>stacking velocities: Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Stack | NMO-corrected CDP gathers: Final NMO → NMO 1 *(⚠ if not the first)* |
| Random Noise Attenuation | a stacked section: head of the **section** stream |
| Deconvolution (post-stack) | a stacked section: head of the **section** stream |
| Time-Variant Filter | a stacked section: head of the **section** stream |
| Migration (poststack) | a stacked section: head of the **section** stream<br>a stacking-velocity field (the migration velocity): Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Migration (pstm) | pre-stack CDP gathers with their offsets preserved: head of the **cdp** stream<br>a stacking-velocity field (the migration velocity): Velocity 2 → Velocity 1 *(⚠ if not the first)* |
| Final Scaling | a section to scale: Migration → head of the **section** stream *(⚠ if not the first)* |

Notes:

* **Velocity analysis needs CDP gathers** (it scans offset-sorted gathers).
* **NMO needs a velocity function.** Final NMO uses pass-2 velocities, falling
  back to pass 1 with a ⚠.
* **Stack needs moveout-corrected gathers** (Final NMO, or NMO pass 1 with a ⚠):
  without them reflections stack out of phase.
* **Post-stack migration needs a section *and* a velocity field. Pre-stack
  migration (PSTM) needs pre-stack, offset-preserving CDP gathers *and* a
  velocity field — and no stack**, since it migrates and stacks in one pass. It
  reads the head of the `cdp` stream, never NMO-corrected data. Data
  regularisation is a recommendation for it, not a requirement.
* **Residual statics cannot run on regularised gathers** (⛔, "geometry"):
  statics are surface-consistent — they need each trace's own shot and
  receiver, and regularised gathers hold interpolated traces on a synthetic
  offset grid. Apply statics before regularisation.
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

* `missing_recommended` — a recommended predecessor is not in the ancestry of the data this step reads.
* `order_inversion` — a step normally applied *after* this one is already part of the data (e.g. band-pass after AGC).
* `fallback_input` — the input comes from a less-preferred source (stack of pass-1 NMO; pass-1 velocities for migration; un-migrated section for final scaling).
* `orphaned_step` — a valid step in the same stream whose result is not in the data this step will read (arises when DMO → inverse NMO carries on from gathers that statics did not touch).

**The advice must agree with the recommended order.** `tests/test_dependencies.py`
asserts that every (earlier, later) pair implied by an `Advice` respects
`PIPELINE_STAGES`, and that following the recommended order produces **zero**
warnings at every step.

## Re-running and reordering

* Re-running a stage replaces its result and **discards only the stages that
  consumed its output** (transitively), freeing their data. Steps that merely
  come later in the recommended order but never read it are kept — e.g.
  re-running NMO pass 1 keeps DMO.
* A re-run stage reads the best remaining input, never its own output or a
  descendant's.
* If data moved on after a step ran (conditioning applied after CDP sort; Final
  NMO run after the stack), the step is **not** discarded — a note says it was
  built from earlier data and can be re-run (`stale_notes`).
* **Reordering** is *discard, then re-run in the order you want* (the "↩ Undo or
  reorder steps" panel). A result cannot be un-applied from the middle of a
  chain without redoing what came after, so the panel discards a step together
  with what was built on it.
* Loading a new survey invalidates everything.

## Kill detection and gain

Trace-kill detection is energy-based, so it is blind after AGC or balancing has
equalised trace levels. It therefore measures the data from just *before* the
first gain-altering step in the data's ancestry, and applies the resulting mask
to the data actually being processed (`stage_ops.kill_detection_stage`). The set
of killed traces is the same whether or not AGC ran first (tested).

## Code map

| File | Role |
|---|---|
| `modules/pipeline.py` | stage registry (recommended order), data-key maps, `HistoryEntry`, `log_step` |
| `modules/dependencies.py` | the engine: `Need`, `Advice`, `StageSpec`, `STAGE_SPECS`, `WorkflowState`, `check_stage`, `stale_notes` — pure Python |
| `modules/stage_ops.py` | one executor per stage; checks, resolves inputs from the data state, calls the **unchanged** maths, records provenance — pure Python, takes any mutable mapping as its store |
| `modules/validation.py` | `DependencyError` (a `ValidationError`) raised when a ⛔ stage is run anyway |
| `utils/session_state.py` | thin adapter over `st.session_state` (schema, `check`, `invalidate_from`, `best_available_section`) |
| `utils/workflow_ui.py` | banners, the *Continue?* run button, status panels, undo/reorder panel |
| `views/*.py` | pages: call `wf.show_status` / `wf.run_button` / `wf.execute(stage_ops.run_…)` |

The processing mathematics in `modules/{filtering,preprocessing,nmo,velocity,
statics,regularization,stacking,migration}.py` was **not changed**. With the
recommended order, the refactored app produces results bit-for-bit identical to
the previous version (stack, post-stack, migrated and final sections and both
velocity fields compared with exact equality).

### Adding a stage

1. Add a `Stage` to `PIPELINE_STAGES` at its recommended position.
2. Add a `StageSpec` in `dependencies.STAGE_SPECS`: its `Need`s (genuine inputs) and `Advice` (recommendations). If it transforms a stream, add it to `STREAM_MEMBERS`.
3. Add its data keys to `pipeline.py` (`STAGE_TRACE_KEY`, `STAGE_STORE_KEYS`, `STAGE_GATHER_KEYS`).
4. Write `stage_ops.run_<stage>` following the others (`_begin` → maths → `_record`).
5. Call it from a view with `wf.show_status` / `wf.run_button` / `wf.execute`.
6. Add tests; the registry tests fail if a stage lacks a spec or an executor, or if its advice contradicts the recommended order.

## Tests

| File | Covers |
|---|---|
| `tests/test_dependencies.py` | the engine, with no data: golden path has zero warnings; both task example orders; every optional stage skippable; shot operators in all 24 orders; every genuine dependency (⛔); recommendation wording; invalidation of dependents only; stale notes; advice consistent with the recommended order |
| `tests/test_stage_ops.py` | the same through the real maths: non-standard orders, skipped stages, `DependencyError` leaves no trace, re-run freeing data, PSTM without stack/regularisation, history/export in execution order, executors refuse exactly what the UI marks ⛔ |
| `tests/test_app_nonlinear.py` | the app driven headlessly: ⛔ messages and absent controls, ⚠ *Continue?* / Cancel, the two task orders, a skip-everything run to a final section, PSTM with no stack |
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
* Dependencies are tracked per *stage*, not per parameter set: changing a
  parameter and re-running is treated as a re-run, with the dependents discarded.
