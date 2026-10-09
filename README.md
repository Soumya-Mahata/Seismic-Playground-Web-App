# 🌊 Seismic Playground

**Seismic Processing & Interpretation Platform** — an interactive Streamlit
application that walks a 2D land seismic processing workflow through a
guided, parameterized, QC'd application: load data → condition it in the shot
domain → sort into CDP gathers → pick velocities (auto, numeric-pick, or
Constant-Velocity-Stack) → NMO/DMO/statics → stack → migrate → interpret
→ export. Every manual pick (velocity control points, horizons, faults,
the F-K polygon) is entered through a small numeric (x, y) + "Add point"
form beneath a fully zoomable/pannable reference plot, backed by an
editable table for review/reorder/delete — not by clicking the plot
directly (Streamlit cannot deliver click-selection events on the
`go.Heatmap` traces these panels are drawn on; see "Known Limitations").

Single entry point (`app.py`), sidebar `st.radio` navigation, all
persistent state in `st.session_state`, and Plotly visualizations forced
to a light background so colormaps stay legible regardless of the
surrounding Streamlit theme.

**Nothing here is faked.** Every "Run" button performs the real,
documented computation on the data actually loaded — including on the
synthetic demo data, which exists only because no real SEG-Y file was
available (see "Synthetic test data" below).

---

## Features

- **Data Management** — SEG-Y upload (`segyio`), EBCDIC/binary/trace
  header inspection, optional **geometry merging** (combine a second
  survey), an optional early **brute stack + fold map** QC, then build
  shot gathers. The synthetic-survey generator also exposes **station
  spacing and sample interval** directly (default 6.25 m / 2 ms, chosen
  so the default F-K wedge below tapers gradually across the spectrum
  rather than saturating to full width near f=0), with a live preview of
  the resulting F-K domain (k/f Nyquist) — both are fully adjustable,
  and a coarser spacing still works, just with a differently-shaped
  default F-K wedge.
- **Pre-Stack Conditioning** (shot domain) — band-pass; **AGC with an
  explicit mean/RMS/median choice** (all three are computed and shown
  side by side before you pick one, and the active choice is shown
  persistently in the sidebar); **trace edit/kill + de-spiking** (run on
  the pre-AGC data, since AGC's own self-normalization would otherwise
  erase the relative-amplitude differences this step needs to detect);
  **Mute** — flat top mute, offset-dependent (moveout-following) top
  mute, or a surgical (box-region) mute, with a QC plot overlaying the
  mute boundary on the pre-mute section; trace balancing; **F-K dip
  filter** — accept or reject polygon, either the default
  apparent-velocity wedge (with its true geometric complement built for
  reject mode, so both conventions give the same result) or a custom
  polygon entered as numeric (k, f) points in an editable table
  (add/edit/delete a row, or reorder/insert a vertex between two
  existing rows via the table's **order** column); **multiple
  suppression** (predictive/gap deconvolution); spiking deconvolution —
  each with before/after and F-K spectrum QC.
- **CDP Sort & Binning** — re-groups the current shot-domain data by CDP
  (conditioned by whichever pre-stack steps you chose to apply — or the
  raw traces if none), with a fold map and a semblance preview, right
  before velocity analysis.
- **Velocity Analysis, NMO & DMO** — two-pass semblance velocity
  analysis with an automatic picker, **per-analysis-CDP reliability
  flagging**, numeric-entry picking directly below the semblance panel,
  manual table-based correction/exclusion, and a **Constant-Velocity-
  Stack (CVS)** panel as a complementary, amplitude-domain check; NMO
  (pass 1), surface-consistent residual statics, simplified DMO, inverse
  NMO, **Data Regularisation** (resamples every CDP gather onto a common
  offset grid), RMS velocity analysis (pass 2, on the regularized data),
  final NMO.
- **Stacking** — fold-normalized CDP stack with a fold map and
  adjustable display (colormap / clip / gain / polarity).
- **Post-Stack Processing** — random noise attenuation, a second
  deconvolution pass, time-variant filter.
- **Migration** — a choice of **post-stack Kirchhoff time migration** or
  **pre-stack time migration (PSTM)** — a double-square-root traveltime
  summation directly on the offset-preserving regularized CDP gathers,
  which needs pre-stack CDP gathers and a velocity field (no stack
  required; Data Regularisation is recommended, not required) — plus
  final scaling.
- **Interpretation** — a 5-stage workflow, in order: **Well-to-Seismic
  Tie** (CSV or **LAS** log upload, with curve/unit mapping, and a
  choice of a user-specified Ricker wavelet or one **statistically
  extracted from the seismic data itself**) → **Horizon Picking** →
  **Fault Interpretation** → **Surface (time) Map**. Horizons/faults are
  numeric-entry picks backed by an editable table, plus horizon
  curvature.
- **Attributes** — RMS amplitude, instantaneous amplitude (envelope) /
  phase / frequency, **cosine of phase**, **sweetness**, **variance**,
  **coherence/similarity**, dominant frequency — with attribute-along-
  horizon extraction, each displayed with a **Petrel-convention color
  scale** (see "Interpretation & Attributes" below).
- **Export** — processed SEG-Y, CSV, NumPy, horizon/fault CSV, attribute
  CSV, interactive HTML figures, and a text/JSON processing report.
- **Dependency-aware workflow** — the recommended order is *advice*, not
  a gate. Optional pre-stack steps (band-pass, AGC, de-spike, trace
  kill, mute, balance, F-K, multiple suppression, deconvolution) can be
  run in any order, or skipped; genuine inputs (CDP gathers for velocity
  analysis, velocities for NMO, NMO-corrected gathers for stacking, a
  section or pre-stack gathers plus velocity for migration) are enforced.
  Every step shows **✓ Available / ⚠ Recommended prerequisite not
  performed / ⛔ Required input missing**; a ⚠ asks *Continue?* and never
  blocks. See [`docs/DEPENDENCY_MODEL.md`](docs/DEPENDENCY_MODEL.md).
- **Processing history** — every step is logged in the order it was
  *executed*, with its parameters and which earlier outputs it consumed.
  Re-running a step discards only the steps built on its output (so the
  history can never silently go stale, but unrelated later-in-the-list
  steps are kept).

---

## Workflow

The diagram below is the **recommended** order. Nothing in the app
requires a step just because the steps above it were run — only a step's
genuine inputs gate it (see [`docs/DEPENDENCY_MODEL.md`](docs/DEPENDENCY_MODEL.md)).
Every pre-stack conditioning step, and statics, DMO, regularisation, pass-2
velocity and the post-stack steps, is optional.

```
Load SEG-Y (or synthetic demo)
  Geometry Merging (optional)
        │
        ▼
Build shot gathers ── (an optional throwaway CDP grouping is available
        │               purely for an early Brute Stack / Fold Map QC)
        ▼
Pre-Stack Conditioning (shot domain) — every step optional, any order
  Band-pass → AGC → Trace edit/kill/de-spike → Mute → Balance →
  F-K filter (apparent-velocity wedge or custom polygon) →
  Multiple suppression → Deconvolution           (QC after each step)
        │
        ▼
CDP Sort / Binning  (re-group by CDP; fold map; semblance preview)
        │
        ▼
Velocity Analysis Pass 1 (auto-pick / numeric-pick / CVS)
        │            ──► NMO 1 ──► Residual Statics ──► DMO ──► Inverse NMO
        ▼
Data Regularisation ──► Velocity Analysis Pass 2 (refined) ──► Final NMO
        │
        ▼
Stack ──► Random Noise Attenuation ──► Deconvolution ──► Time-Variant Filter
        │
        ▼
Migration — post-stack Kirchhoff, or pre-stack (PSTM) ──► Final Scaling
        │
        ▼
Interpretation (Well Tie → Horizon Picking → Fault Interpretation →
                Surface Map) + Attributes
        │
        ▼
Export (SEG-Y / CSV / NumPy / picks / figures / report)
```

---

## Supported Data

- **Input:** SEG-Y (`.sgy`, `.segy`), `.su` (read via the SEG-Y path)
- **Well tie:** CSV well log (`depth_m`, `vp_m_s`, `rho_kg_m3`), optional
  checkshot CSV (`depth_m`, `twt_s`), or a LAS well-log file
- **Output:** SEG-Y, CSV, NumPy (`.npy`), interactive HTML, text/JSON

### Synthetic test data

No real SEG-Y file was available, so `modules/synthetic_data.py`
generates a genuine (if small) 2D land survey — shots, receivers, CDP
geometry, five flat reflectors with a realistic velocity-with-depth
trend, ground roll, a point diffractor (for validating migration), and
optional realistic field-noise conditions (dead channels, powerline
harmonic noise, spike bursts) — written out as a real SEG-Y file via the
same writer used for export. A small pre-generated copy ships at
`data/sample_data/synthetic_demo_line.sgy`; the Data Management page can
also generate a fresh, larger one on demand with adjustable station
count, spacing, and sample interval. **It is labeled as synthetic
everywhere it appears in the UI and is never presented as real acquired
data.** Every processing step still performs its real computation on
this data — only the input is synthetic.

The full pipeline was validated end to end against this data's known
ground truth (see `tests/manual_pipeline_check.py`): all 5 reflectors
are recovered in the final processed section at roughly their correct
two-way time, and the point diffractor collapses under migration as
expected. See "A known, expected accuracy trade-off" below for one
honestly-documented wrinkle in exactly how tight that timing is.

---

## Installation

```bash
pip install -r requirements.txt
```

## Running locally

```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (see `.gitignore` — it already keeps
   the repo lean; the one shipped sample SEG-Y is small enough to commit
   directly).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app
   at the repo, branch, and `app.py`.
3. That's it — `requirements.txt` is all Streamlit Cloud needs. No
   Docker, no extra configuration. (There's no Dockerfile in this repo;
   it isn't needed for this deployment path, and skipping it keeps the
   repo simpler. If you do need a container for some other host, a
   standard `streamlit run app.py` entrypoint on a `python:3.11-slim`
   base with `pip install -r requirements.txt` is all it takes to build
   one.)

---

## Project Structure

```
seismic-playground/
├── app.py                     # entry point: page config, CSS, sidebar nav, routing
├── config.py                  # every default processing parameter, in one place,
│                               #   with the rationale for each noted alongside it
├── modules/                   # pure computation — no Streamlit dependency
│   ├── segy_io.py             # SEG-Y read/write, headers, shot & CDP gathers,
│   │                           #   scatter-to-flat, geometry merging
│   ├── synthetic_data.py      # synthetic survey generator (testing/demo only)
│   ├── preprocessing.py       # AGC, de-spiking, trace kill, predictive &
│   │                           #   spiking deconvolution, balancing, mute
│   ├── filtering.py           # band-pass design, F-K filter, RNA, TVF, spectra
│   ├── velocity.py            # semblance, auto-picker, Constant-Velocity-Stack analysis
│   ├── nmo.py                 # NMO / inverse NMO / DMO
│   ├── regularization.py      # CDP-gather offset regularization (for PSTM + vel. pass 2)
│   ├── statics.py             # surface-consistent residual statics
│   ├── stacking.py            # fold-normalized stacking
│   ├── migration.py           # post-stack & pre-stack (PSTM) Kirchhoff migration + final scaling
│   ├── attributes.py          # RMS, envelope, phase, cosine-of-phase, frequency,
│   │                           #   sweetness, variance, coherence + Petrel color map
│   ├── interpretation.py      # horizon/fault data model, curvature, LAS reading,
│   │                           #   statistical wavelet extraction, well-tie
│   ├── pipeline.py            # recommended-order stage registry + processing history
│   ├── dependencies.py        # dependency engine: genuine needs (⛔) vs recommendations (⚠),
│   │                           #   data streams, provenance, invalidation — pure Python
│   ├── stage_ops.py           # one executor per stage (calls the maths above, records
│   │                           #   provenance); used by the pages and by the tests
│   └── validation.py          # friendly input/parameter validation, DependencyError
├── visualization/              # Plotly figure builders — no Streamlit dependency
│   ├── theme.py                 # shared style tokens (forced-white plot background)
│   ├── controls.py              # colormap/clip/gain/polarity display controls
│   ├── seismic_plots.py         # sections, gathers, wiggle/VA, VD+wiggle,
│   │                             #   F-K spectrum, before/after, fold map
│   ├── spectra_plots.py, velocity_plots.py, interpretation_plots.py
├── views/                      # Streamlit UI — one module per app page
│   ├── home.py, data_management.py, preprocessing_view.py,
│   ├── cdp_sort_view.py, velocity_nmo_view.py, stacking_view.py,
│   ├── poststack_view.py, migration_view.py, interpretation_view.py,
│   ├── attributes_view.py, export_view.py
├── utils/
│   ├── session_state.py        # session-state schema + thin adapter over the dependency engine
│   ├── workflow_ui.py          # ✓/⚠/⛔ banners, "Continue?" run button, status panels, undo/reorder
│   ├── click_picking.py        # shared numeric-entry picking helper (velocity, horizon,
│   │                             #   fault, F-K polygon all use this one mechanism)
│   ├── io_export.py            # SEG-Y/CSV/NumPy/HTML/report export helpers
│   └── formatting.py
├── data/sample_data/            # shipped synthetic demo SEG-Y
├── docs/DEPENDENCY_MODEL.md     # the dependency model, in full
├── tests/                       # see "Testing" below
├── requirements.txt, .gitignore, .streamlit/config.toml
```

### Architecture notes

- **Why `views/` and not `pages/`:** Streamlit auto-generates multipage
  navigation from a literal `pages/` folder next to `app.py`. This app
  uses a single entry point with a sidebar `st.radio` instead, so a
  `pages/` folder here would silently create a *second*, conflicting
  navigation UI. `views/` holds the same kind of page-level modules
  without triggering that behavior.
- **No Docker.** The deployment path this app is built for (run
  locally → push to GitHub → deploy via Streamlit Community Cloud) never
  touches a container — Streamlit Cloud builds straight from
  `requirements.txt`. Adding a Dockerfile would be one more file to keep
  in sync with no benefit for that path, so it's left out; see
  "Deploying to Streamlit Community Cloud" above for the one-line note
  on building one yourself if you ever need it for a different host.
- **No hard-coded app-wide dark theme.** The app works in both light
  and dark Streamlit themes — `app.py`'s CSS only touches buttons/tabs
  (readable either way) and leaves the theme itself to the person's
  Streamlit setting. Plotly figures are separately forced to a
  consistent white background in `visualization/theme.py`, since a plot
  that inherits the app's dark background loses most colormaps' contrast.
- **Recommended order ≠ required inputs ≠ data state ≠ history.** These
  used to be one hard-coded chain (`require_stage` + "stage N done clears
  every later stage"). They are now separate: `pipeline.PIPELINE_STAGES`
  is only the recommended order; `dependencies.STAGE_SPECS` declares each
  stage's genuine `Need`s and its `Advice`; the data state and the
  history are derived from one provenance log (each run records what it
  consumed). Pages call `stage_ops.run_*`, which refuse (`DependencyError`)
  only when a genuine input is missing. Full write-up:
  [`docs/DEPENDENCY_MODEL.md`](docs/DEPENDENCY_MODEL.md).
- **Shot domain first, CDP domain second.** `gathers` (CDP-indexed) and
  `shot_gathers` (shot-indexed) are both dicts of per-gather dicts that
  accumulate keys as data flows through the pipeline (`traces` →
  `traces_bp` → `traces_agc` → …). Pre-stack conditioning runs entirely
  on `shot_gathers`; CDP sort/binning (`modules/segy_io.scatter_to_flat`
  + `build_cdp_gathers(..., data_override=...)`) re-groups the
  *conditioned* samples by CDP afterward, using the survey's own headers
  for grouping — nothing about the geometry changes, only which samples
  live in which gather.
- **Point-picking, one mechanism, every panel.** `utils/click_picking.py`
  provides `numeric_point_form()` — a small (x, y) + "Add point" control
  used identically for velocity control points, horizon picks, fault
  picks, and the F-K polygon. A literal click on the chart was tried
  first via `st.plotly_chart(..., on_select="rerun")`, but Streamlit
  does not deliver selection events for `go.Heatmap` traces (only
  scatter/bar/histogram), which is what every one of these panels is
  drawn on — so a click there would silently never fire, and turning
  `on_select` on also swaps the chart's drag mode from zoom to
  box-select, breaking normal zoom/pan too. Leaving `on_select` off
  keeps native zoom/pan; the numeric form plus an editable table
  (supporting add/edit/delete and, for the F-K polygon, reorder/insert
  a vertex between existing rows) replaces the click.
- **SEG-Y library:** `segyio` handles both reading and writing SEG-Y —
  headers, geometry, and trace data all through one dependency.

---

## Design notes (read this)

A few implementation choices are worth calling out because they weren't
the obvious first attempt — each was arrived at by testing against
known, synthetic ground truth and correcting what the numbers showed:

### 1. NMO and inverse-NMO use opposite interpolation conventions, deliberately

For each trace at offset x, the recorded two-way time at zero-offset
time t0 is tn(t0) = sqrt(t0² + x²/vrms(t0)²). Forward NMO maps the
*recorded* amplitude at tn back to t0, which means building an
interpolant from (t0_grid, trace_values) pairs and evaluating it at the
query points tn — `np.interp(tn, t0_grid, trace)`. Verified empirically
on a synthetic trace with a single reflection event at a known time:
the superficially-similar swapped-argument form,
`np.interp(t0, tn, trace)`, places the corrected event at the **wrong**
output time (0.754 s instead of the true 0.500 s), while the correct
form recovers 0.500 s exactly.

Because `nmo_correction` and `inverse_nmo_correction` are meant to be
exact inverses of each other, a direct round-trip test (apply forward
NMO, then inverse NMO, and check the original trace comes back) was
used to confirm `inverse_nmo_correction` needs the *opposite*
convention from the forward function — reusing the same convention for
both is self-consistent for forward warping but actually compounds the
same warp a second time on the way back, rather than undoing it. Full
derivation and the round-trip test are in `modules/nmo.py`'s module
docstring and `tests/test_pipeline.py::test_inverse_nmo_undoes_nmo`.
`dmo_correction` is unrelated to this — it doesn't claim to invert
anything, and consistently uses the forward-style convention.

### 2. The automatic velocity picker uses a confidence-gated local walk

An earlier dynamic-programming approach that maximized cumulative
semblance across the whole time record (minus a jump penalty)
systematically under-tracked deep, isolated coherence peaks (a 12%
velocity error at 2.2 s/3400 m/s in testing) and let one bad edge-CDP
pick corrupt a large stretch of the velocity field via flat
extrapolation. `modules/velocity.py` instead uses a confidence-gated
local walk seeded from each panel's strongest peak, plus a
lateral-consistency outlier check — recovers the true velocity to ~1%
and degrades gracefully at low fold instead of collapsing. Full
derivation and before/after numbers are in the module docstring.

### 3. De-spiking needs an isolation check, not just an outlier threshold

An outlier-threshold-only version of `despike_traces` (flagging any
sample whose residual from a local median exceeded a threshold times
the local MAD-based noise estimate) flagged ~2% of *all* samples on
**clean** synthetic data with no injected noise — every genuine, sharp
reflection peak in an otherwise-quiet local window reads as an extreme
outlier once the local noise estimate collapses toward zero, which it
does in a quiet window regardless of the threshold. Fixed by adding an
isolation check: a sample is only flagged if it's also much larger than
the *average* of its immediate neighbours' own residuals — true
burst/glitch spikes are 1-2 sample transients with small neighbours; a
real wavelet peak's neighbours ramp up and down smoothly at comparable
amplitude. This dropped the false-positive rate on clean data to
<0.05% while still catching an injected single-sample spike in testing.
See `modules/preprocessing.py`'s `despike_traces` docstring.

### 4. A known, expected accuracy trade-off from decon-last ordering

Pre-stack conditioning runs deconvolution *last* (after band-pass, AGC,
trace edit, mute, balance, F-K, and multiple suppression), matching
standard shot-domain processing practice. Testing the final processed
section against the synthetic data's known reflector times shows some
timing scatter (roughly 15-20 ms) that traces back to a well-understood,
expected mismatch: applying a *minimum-phase* deconvolution model to
the *zero-phase* synthetic Ricker wavelets used here produces a similar
timing shift regardless of where in the chain deconvolution runs — this
is a property of that mismatch, not an ordering-induced bug. See
`tests/manual_pipeline_check.py` for the reproducible comparison.

### 5. Recommendations warn; only missing inputs block

A step that is normally done earlier but wasn't (AGC without band-pass,
CDP sort without any conditioning, DMO before statics …) gets a ⚠ with
the reason and a *Continue?* prompt; running steps in a non-standard order
is allowed and recorded. Things that are genuinely impossible are ⛔
— e.g. stacking un-flattened gathers, NMO with no velocity function, or
residual statics on regularised gathers (which no longer carry each
trace's own shot/receiver). Re-ordering two finished steps is "discard
one, then run them in the order you want": a result cannot be un-applied
from the middle of a chain without redoing what came after. Because
every executor reads the *current head* of its data stream rather than a
fixed predecessor, a step's input is whatever was actually applied last —
and trace-kill detection deliberately measures amplitude-preserving data
from before any AGC/balancing so its result does not depend on that
order. With the recommended order the results are bit-for-bit identical
to the previous (linear) version.

### Other documented simplifications

- **DMO** is a common-offset time-stretch approximation, not a full
  log-stretch/Kirchhoff DMO operator.
- **No field statics** — only data-driven, surface-consistent residual
  statics are computed (no weathering-velocity/uphole-time data
  available to compute a field static from).
- **Random noise attenuation** is a spatial median filter across CDPs,
  standing in for F-X deconvolution (which needs higher fold than a
  small teaching/demo line typically has, to avoid its own artifacts).
- **Migration** is 2-D time migration only (post-stack Kirchhoff or
  pre-stack PSTM) — no depth migration, no 3-D.
- **Geometry merging** combines two already-geometry-complete surveys
  without ID collisions; it does not reconcile coordinates from external
  SPS navigation files the way a full industry geometry merge would.

---

## Interpretation & Attributes

The Interpretation and Attributes pages follow a 5-stage workflow in
order — **Well-to-Seismic Tie → Horizon Picking → Fault Interpretation →
Surface (Time) Map → Attribute Analysis**:

- **Well-to-Seismic Tie** accepts a CSV well log or a **LAS** file (read
  via `lasio`, with curve/unit mapping so common LAS naming variations
  are handled rather than assuming fixed curve names), builds a
  reflectivity series from the sonic/density logs, and convolves it with
  either a user-specified Ricker wavelet or a wavelet **statistically
  extracted from the seismic data itself** (`extract_statistical_wavelet`
  in `modules/interpretation.py`), then reports a correlation
  coefficient against the real seismic trace at the well location.
- **Horizon/fault picking** is numeric-entry on the section, backed by
  an editable table for precise review/correction — never an automatic
  tracker, since a fabricated auto-tracker would risk presenting
  invented picks as real interpretation.
- **Horizon curvature** is the 2-D analogue (second derivative of the
  picked line) of the 3D surface-curvature attribute.
- **Surface (Time) Map** grids a picked horizon's time values along the
  line into a continuous time-structure profile (the 2D-line analogue of
  an areal time/depth structure map), with an optional time-to-depth
  conversion from an average-velocity function.
- **Attributes**: RMS amplitude, instantaneous amplitude (envelope),
  instantaneous phase, cosine of phase, instantaneous frequency,
  sweetness (amplitude/frequency hybrid), variance (discontinuity),
  coherence/similarity, and dominant frequency — the instantaneous-
  attribute family follows the standard complex-trace definitions
  (Taner, Koehler & Sheriff, 1979). Each attribute is displayed with a
  **Petrel-convention color scale** (`ATTRIBUTE_COLORSCALE_MAP` in
  `modules/attributes.py`):
  - RMS amplitude & sweetness: a warm-cool sequential ramp (cool = low,
    warm = high).
  - Variance: a high-contrast scale — a dark, low-contrast background
    with a sharp red/yellow spike reserved for the highest
    (fault-like/discontinuous) values.
  - Instantaneous amplitude (envelope): a "hot"-style black → red/orange
    → yellow → white ramp.
  - Instantaneous phase: a cyclic scale that wraps cleanly at the
    ±180° boundary.
  - Cosine of phase: a diverging red/white/blue scale centered on zero.
  - Instantaneous frequency: a sequential dark-purple → orange → yellow
    ramp.
  - Coherence/similarity: plain grayscale, with continuous reflectivity
    shown bright and discontinuities shown dark — the inverse-scaled
    counterpart of variance.
  - Dominant frequency: a full-spectrum rainbow, read as a coarse
    attenuation/tuning indicator.

---

## Example Usage

1. Open the app, go to **Data Management**, and click "Generate
   synthetic survey" (or upload your own SEG-Y). Optionally merge in a
   second survey, or run the optional brute-stack QC. Then **build shot
   gathers**.
2. Work through **Pre-Stack Conditioning** tab by tab (band-pass → AGC →
   trace de-spike → trace kill → mute → balance → F-K filter → multiple
   suppression → deconvolution — all optional, and in any order; a step
   done out of the usual order shows a ⚠ and asks you to confirm) — each
   has a "Run" button and before/after QC. For the F-K filter, try the custom (numeric points)
   polygon option and zoom into the spectrum to read off a few (k, f)
   coordinates to enter.
3. **CDP Sort & Binning** — re-group into CDP gathers, check the fold
   map, preview a semblance panel.
4. In **Velocity, NMO & DMO**, run Pass 1, review the reliability chart,
   zoom into the semblance panel and **enter a (velocity, time) point**
   below it to add a manual pick (or use the CVS panel as a second
   opinion), then continue through NMO → Statics → DMO → Inverse NMO →
   Data Regularisation → RMS Velocity Analysis (Pass 2) → Final NMO. Only
   Pass 1 velocities and one NMO pass are needed to stack; the rest are
   refinements you can skip.
5. **Stack**, then **Post-Stack Processing**, then **Migration** — pick
   post-stack or pre-stack (PSTM) migration, and adjust the display
   controls (colormap/clip/gain/polarity/trace-balanced display) on the
   final section.
6. In **Interpretation**, do the Well-to-Seismic Tie first (CSV or LAS
   log, Ricker or statistically-extracted wavelet), then zoom into the
   section and enter (CDP, time) points to pick a horizon and a fault;
   check the horizon's curvature; generate a time/depth structure
   profile. In **Attributes**, compute sweetness or coherence and
   extract it along your horizon.
7. Download whatever you need from **Export**.

---

## Testing

- `tests/manual_pipeline_check.py` — runs the full processing chain
  (shot-domain conditioning → CDP sort → velocity/NMO/DMO → stack →
  post-stack → migration) against the synthetic survey directly (no UI)
  and checks the final section recovers every known reflector at
  roughly its correct two-way time.
- `tests/test_app_flow.py` — a scripted `streamlit.testing.v1.AppTest`
  walkthrough that drives the *actual app* through every stage via
  simulated widget interactions and fails on any uncaught exception.
  This is also how several real bugs were caught during development
  (see below).
- `tests/test_pipeline.py` — a pytest suite (22 tests) covering the
  documented fixes above plus shot/CDP gather round-tripping, geometry
  merging, trace edit/de-spiking, predictive deconvolution, CVS, the
  attributes, and horizon curvature.
- `tests/test_dependencies.py` — the dependency engine with no data and no
  UI (80 tests): the recommended order gives zero warnings at every step;
  the two non-standard orders from the design brief; every optional stage
  skippable; shot-domain steps in all permutations; every genuine
  dependency failing as ⛔; invalidation of dependents only; stale notes;
  recommendations consistent with the recommended order.
- `tests/test_stage_ops.py` — the same behaviours through the real
  processing maths (45 tests): non-standard orders, skipped stages,
  `DependencyError` leaving no trace, re-runs freeing data, pre-stack
  migration with no stack, execution-order history/exports, and
  executors refusing exactly what the UI marks ⛔.
- `tests/test_app_nonlinear.py` — the app driven headlessly (13 tests): ⛔
  messages and absent controls, ⚠ *Continue?* / *Cancel*, non-standard
  orders, a skip-everything run to a final section.

Bugs found and fixed via this testing, beyond the NMO/picker/de-spike
items documented above: `pick_velocity_field` crashed outright on a
small-aperture survey where no CDP met the default minimum-offset
requirement (now falls back gracefully); several Plotly charts collided
on auto-generated element IDs when rendered in the same script run (now
explicitly keyed); a `st.data_editor` cell-edit path raised a pandas
dtype error when a fractional value was typed into an integer-typed
column (fixed by building that column as float64 from the start).

Run (requires `pip install pytest` in addition to `requirements.txt`):
```bash
pytest tests/test_pipeline.py -v
python tests/manual_pipeline_check.py
python tests/test_app_flow.py
# dependency model:
pytest tests/test_dependencies.py tests/test_stage_ops.py tests/test_app_nonlinear.py -v
```

---

## Known Limitations

- DMO is a simplified time-stretch approximation.
- No field statics (no uphole/weathering-velocity data available).
- Random noise attenuation is a spatial median filter, not F-X decon.
- Migration offers both post-stack Kirchhoff time migration and
  pre-stack Kirchhoff time migration (PSTM); both are still 2-D time
  migration (no depth migration, no 3-D).
- Point-picking (velocity, horizon, fault, F-K polygon) is entered via a
  small numeric (x, y) + "Add point" form under a fully zoomable/
  pannable reference plot, rather than by clicking the plot directly:
  Streamlit does not deliver point-selection events for `go.Heatmap`
  traces (only scatter/bar/histogram do), so every one of these
  Heatmap-backed panels could never have supported real click-to-pick in
  the first place — this was confirmed, not assumed, during development
  (see `utils/click_picking.py`'s module docstring). The editable table
  next to every picker remains the full-precision way to review/correct
  entries, and for the F-K polygon also supports reordering vertices via
  an explicit **order** column (plain Streamlit has no drag-and-drop row
  reordering).
- Re-ordering finished steps is discard-then-re-run (a result cannot be
  un-applied from the middle of a chain). Dependencies are tracked per
  stage, not per parameter set. Recommendations are pairwise (e.g. there
  is no direct "RNA before TVF" advice, only RNA→decon and decon→TVF).
- Pre-stack migration on un-regularised gathers runs, but sums over
  irregular offsets; Data Regularisation is recommended before it.
- Geometry merging combines surveys without ID collisions but doesn't
  reconcile external SPS navigation coordinates.
- Designed and tested against a single 2D line; 3D survey geometry is
  out of scope.
- Large surveys: processing loops per-trace/per-gather in NumPy — fast
  enough for a teaching/demo-sized line but not optimized for a full
  production 3D volume.

## Recommended Future Improvements

- A full log-stretch or Kirchhoff DMO operator.
- F-X deconvolution as an alternative random-noise-attenuation method.
- Pre-stack *depth* migration (time migration is covered; depth is not).
- Automatic (assisted) horizon tracking seeded from manual picks.
- 3D survey support (inline/crossline geometry, 3D migration aperture).
- A full SPS-navigation-based geometry merge.
- An RGB spectral-decomposition attribute (three frequency bands blended
  into one color image) as a complement to the single-attribute displays.
