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

## Architecture & Structure

### High-Level Design
The project is a Streamlit-based interactive platform for 2D land seismic processing. It follows a strict linear pipeline where data is transformed through sequential stages.

### Folder Structure
- `app.py`: Main entry point. Handles routing, sidebar navigation, and global CSS.
- `config.py`: Centralized repository for all default processing parameters.
- `modules/`: **Pure computation layer**. No Streamlit dependencies. Contains the physics and math for seismic processing (NMO, Migration, Filtering, etc.).
    - `pipeline.py`: `PIPELINE_STAGES` — the single ordered list of every stage (id/label/description), used by the sidebar progress widget, `utils/session_state.py`'s stage-invalidation logic, and the Processing History views. Renaming or reordering a stage happens here (and in the matching `views/*.py` tab/button), not in session state.
    - `attributes.py`: `ATTRIBUTE_REGISTRY` (the attribute functions), plus `ATTRIBUTE_COLORSCALE_MAP` — the single place each attribute's display colorscale is set (matched to Petrel/industry display convention; see the module's own comment block for the sourcing/reasoning behind each choice). Change an attribute's color here, not in `views/attributes_view.py` or `visualization/interpretation_plots.py` (both just read this map).
- `visualization/`: **Plotly figure builders**. No Streamlit dependencies. Handles the conversion of seismic data into visual representations.
- `views/`: **UI layer**. One module per application page. Contains Streamlit widgets and orchestrates calls between `modules/` and `visualization/`.
- `utils/`: Shared helpers.
    - `session_state.py`: Manages the `st.session_state` schema and the stage-invalidation logic (re-running an early stage clears subsequent processed data).
    - `click_picking.py`: `numeric_point_form()` — a small (x, y) + "Add point" form used for all point-entry (horizons, faults, velocity picks, F-K polygon). Plotly/Streamlit can't deliver click-selection events on `go.Heatmap` traces, so every one of these panels uses this form instead of a literal click — don't try to re-add `on_select="rerun"` click-selection to a Heatmap-backed panel; it was tried and removed (see the module docstring).
    - `io_export.py`: Handles SEG-Y, CSV, and NumPy export operations.

### Key Technical Patterns
- **Workflow Order**: Pre-stack conditioning is performed in the **shot domain** first, followed by **CDP Sort & Binning**, then velocity analysis (pass 1 → NMO/statics/DMO/inverse-NMO → data regularisation → pass 2 → final NMO), stacking, post-stack processing, migration, then interpretation/attributes.
- **State Management**: The application is driven by `st.session_state`. Each processing stage's output is cached in session state to avoid redundant computation.
- **Click-to-Pick**: Conceptually "click-to-pick," but implemented via the numeric-form helper in `utils/click_picking.py` (see above) rather than consuming literal Plotly click/selection events, because `go.Heatmap` traces (what every one of these panels is drawn on) don't deliver those in Streamlit.
- **Data Flow**: Traces $\rightarrow$ Shot Gathers (pre-stack conditioning) $\rightarrow$ CDP Gathers (velocity/NMO/DMO) $\rightarrow$ Stacked Section $\rightarrow$ Migrated Section $\rightarrow$ Interpretation / Attributes.
- **F-K filter polygons**: two independent "Polygon source" modes in `views/preprocessing_view.py` — the default **"Apparent-velocity wedge"** (`modules/filtering.py::apparent_velocity_wedge` / `apparent_velocity_noise_wedges`, derived from one velocity-cutoff number) and **"Custom (numeric points)"**. Both are complementary accept/reject regions of the same F-K plane, and F-K spectra are always displayed cropped to `f >= 0` (matching standard reference-figure convention) even though masking itself checks both the positive and mirrored-negative frequency halves (real-signal Hermitian symmetry).
