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
- `visualization/`: **Plotly figure builders**. No Streamlit dependencies. Handles the conversion of seismic data into visual representations.
- `views/`: **UI layer**. One module per application page. Contains Streamlit widgets and orchestrates calls between `modules/` and `visualization/`.
- `utils/`: Shared helpers.
    - `session_state.py`: Manages the `st.session_state` schema and the stage-invalidation logic (re-running an early stage clears subsequent processed data).
    - `click_picking.py`: Central mechanism for "click-to-pick" interactions (horizons, faults, velocity picks) using Plotly's `on_select` event.
    - `io_export.py`: Handles SEG-Y, CSV, and NumPy export operations.

### Key Technical Patterns
- **Workflow Order**: Pre-stack conditioning is performed in the **shot domain** first, followed by **CDP Sort & Binning**, then velocity analysis.
- **State Management**: The application is driven by `st.session_state`. Each processing stage's output is cached in session state to avoid redundant computation.
- **Click-to-Pick**: Uses a specialized helper in `utils/click_picking.py` to consume selection events from Plotly charts, preventing duplicate picks during Streamlit reruns.
- **Data Flow**: Traces $\rightarrow$ Shot Gathers $\rightarrow$ CDP Gathers $\rightarrow$ Stacked Section $\rightarrow$ Migrated Section.
