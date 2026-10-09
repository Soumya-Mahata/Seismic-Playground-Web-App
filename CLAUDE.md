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
# Run core pipeline logic tests
pytest tests/test_pipeline.py -v

# Run full end-to-end pipeline check against synthetic data
python tests/manual_pipeline_check.py

# Run scripted UI flow test (simulated widget interactions)
python tests/test_app_flow.py
```

All three are the validation gate for any change (and run `pytest tests/test_dependencies.py tests/test_stage_ops.py tests/test_app_nonlinear.py` when touching the workflow/dependency code): unit-level numeric
checks, a full non-UI pipeline run with ground-truth sanity checks, and
a headless `streamlit.testing.v1.AppTest` walkthrough of every page.
Run all three after any change, even a text-only one.

## Architecture & Structure

### High-Level Design
The project is a Streamlit-based interactive platform for 2D land seismic
processing and interpretation. The recommended processing order
(`modules/pipeline.PIPELINE_STAGES`) is *advice*, not a gate: a stage
runs whenever its **genuine** inputs exist (`modules/dependencies.STAGE_SPECS`),
and a skipped or out-of-order *recommended* step only produces a ⚠
"Continue?" — never a block. Recommended order, genuine dependencies,
current data state and processing history are four separate things; see
`docs/DEPENDENCY_MODEL.md`.

### Folder Structure
- `app.py`: Main entry point. Handles routing, sidebar navigation, and global CSS.
- `config.py`: Centralized repository for all default processing parameters.
- `modules/`: **Pure computation layer**. No Streamlit dependencies. Contains the physics and math for seismic processing (NMO/DMO, migration, filtering, statics, stacking, regularisation, attributes, etc.).
- `visualization/`: **Plotly figure builders**. No Streamlit dependencies. Handles the conversion of seismic data into visual representations, including the shared `theme.py` style tokens and the Petrel-convention `ATTRIBUTE_COLORSCALE_MAP` lookup used by the Attributes page.
- `views/`: **UI layer**. One module per application page. Contains Streamlit widgets and orchestrates calls between `modules/` and `visualization/`.
- `utils/`: Shared helpers.
    - `session_state.py`: Owns the `st.session_state` schema; thin adapter over the dependency engine (`check`, `invalidate_from`, `best_available_section`). Re-running a stage invalidates only the stages that *consumed* its output.
    - `workflow_ui.py`: ✓ Available / ⚠ Recommended prerequisite not performed / ⛔ Required input missing banners, the "Continue?" run button (`run_button`), `execute`, status panels, undo/reorder panel. Pages must use these rather than gating on `require_stage`.
    - `click_picking.py`: `numeric_point_form()` — a small (x, y) + "Add point" form used for all point-entry (horizons, faults, velocity picks, F-K polygon). Plotly/Streamlit can't deliver click-selection events on `go.Heatmap` traces, so every one of these panels uses this form instead of a literal click.
    - `io_export.py`: Handles SEG-Y, CSV, NumPy, and processing-report export.
- `tests/`: `test_pipeline.py` (pytest numeric unit tests), `test_app_flow.py` (AppTest end-to-end UI walkthrough of the recommended order), `manual_pipeline_check.py` (non-UI full pipeline run with ground-truth checks); for the dependency model: `test_dependencies.py` (engine, no data), `test_stage_ops.py` (real maths in non-standard orders, genuine-dependency failures), `test_app_nonlinear.py` (UI: ⚠ confirm / ⛔ block flows).
- `modules/dependencies.py` + `modules/stage_ops.py`: pure (no Streamlit). `dependencies` decides ✓/⚠/⛔ and derives the data state from the history; `stage_ops` has one `run_*` executor per stage (takes `st.session_state` or any dict as its store) that calls the unchanged maths and records provenance. Views call these; they do not run maths inline.

### Key Technical Patterns
- **Workflow order (recommended, not enforced)**: Pre-stack conditioning runs in the **shot domain** (band-pass → AGC → trace edit/kill/de-spike → mute → balance → F-K dip filter → multiple suppression → deconvolution), then **CDP Sort & Binning**, then velocity analysis pass 1 → NMO → residual statics → DMO → inverse NMO → data regularisation → velocity analysis pass 2 → final NMO → stack → post-stack processing (random noise attenuation → decon → time-variant filter) → migration (post-stack Kirchhoff, or pre-stack Kirchhoff time migration on CDP gathers before stack) → final scaling. The full ordered stage list lives in `modules/pipeline.PIPELINE_STAGES`.
- **State management**: The application is driven by `st.session_state`. Each processing stage's output is cached in session state to avoid redundant computation. Every run records what it consumed (`HistoryEntry.consumed`); a stage reads the *head of its data stream* (shot / cdp / section), not "the previous stage"; re-running a stage invalidates exactly its dependents.
- **Adding a stage**: registry entry → `STAGE_SPECS` (`Need`s = genuine, `Advice` = recommended) → data keys in `pipeline.py` → `stage_ops.run_*` → view via `wf.show_status` / `wf.run_button` / `wf.execute`. `Advice` pairs must agree with `PIPELINE_STAGES` order (asserted by a test) and the recommended order must stay warning-free.
- **Widgets whose default depends on upstream data** (e.g. the multiple-suppression lag) must be keyed by the input stage, or they freeze at the value computed when the tab first drew.
- **Click-to-pick**: Uses `utils/click_picking.numeric_point_form()` to collect (x, y) picks via a form + editable table rather than literal Plotly click events, for horizons, faults, velocity picks, and the F-K accept/reject polygon (which also supports inserting a point between two existing rows and reordering via an explicit "order" column).
- **Data flow**: Traces → Shot Gathers → CDP Gathers → Stacked Section → (optionally) Migrated Section → Interpretation (well tie, horizon picking, fault interpretation, surface map, attribute analysis) → Export.
- **Colorscales**: Seismic sections use a custom diverging navy/white/maroon "Seismic" scale (`visualization/theme.SEISMIC_RWB`, since Plotly has no built-in "seismic" colorscale). Each computed attribute on the Attributes page is rendered with its own Petrel-convention colorscale via `modules/attributes.ATTRIBUTE_COLORSCALE_MAP` (e.g. Turbo for RMS Amplitude/Sweetness/Dominant Frequency, Hot for Envelope, a custom high-contrast dark-to-red scale for Variance, Greys_r for Coherence/Similarity, Plasma for Instantaneous Frequency, a cyclic scale for Instantaneous Phase, RdBu for Cosine of Phase).
- **Survey geometry**: Synthetic test data (`modules/synthetic_data.py`) exposes `dx`/`dt` controls in the Data Management page (defaults 6.25 m / 2.0 ms), with a live k-Nyquist/f-Nyquist caption, so the F-K dip filter has a sensible default wavenumber range out of the box.

## Known, deliberate design decisions
(See README.md "Design notes" for the full write-up.)
- NMO uses `np.interp(tn, t0_grid, trace)` as the forward-warp convention; `inverse_nmo_correction` uses the swapped convention (`np.interp(t0, tn_grid, trace)`), verified with a direct forward-then-inverse round-trip test. Getting this right matters because `nmo_correction` runs at both velocity-analysis passes and `inverse_nmo_correction` feeds the 2nd pass.
- The velocity auto-picker is a confidence-gated local walk along the semblance ridge rather than a whole-path dynamic-programming picker — see `modules/velocity.py`.
- Post-stack random noise attenuation uses a spatial (across-CDP) median filter as a stand-in for F-X deconvolution, which needs higher-fold data than a small synthetic/demo line provides.
- Deconvolution runs as the last pre-stack-conditioning step (after AGC, trace edit, mute, balance, F-K filter, and multiple suppression) to match the requested shot-domain processing order.
