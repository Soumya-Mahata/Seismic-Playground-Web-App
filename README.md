# 🌊 Seismic Playground

**Seismic Processing & Interpretation Platform** — an interactive Streamlit
application that turns a 2D land seismic processing workflow — the kind
you'd otherwise run cell by cell in a notebook — into a guided,
parameterized, QC'd application: load data → condition it in the shot
domain → sort into CDP gathers → pick velocities (auto, click, or
Constant-Velocity-Stack) → NMO/DMO/statics → stack → migrate → interpret
→ export. Click-to-pick is the primary interaction for every manual
pick (velocity control points, horizons, faults, the F-K reject
polygon) — no typing coordinates required.

Built as a sibling application to an existing Petrophysics Streamlit
platform, reusing its architectural pattern (single-file entry point,
`st.session_state`-driven workflow, Plotly visualizations forced to a
light background for theme-independent legibility) rather than its code.

---

## Overview

The processing algorithms and parameters are ported from a reference
notebook (`Seismic_Processing_CORRECTED.ipynb`) covering raw SEG-Y
through Kirchhoff-migrated, scaled output. The **stage ordering** —
pre-stack conditioning in the shot domain, CDP sort/binning afterward,
right before velocity analysis — follows a standard industry processing
flow (cross-checked against a conventional marine processing-flow
diagram and an industrial-training report's documented workflow;
citations below) rather than the notebook's own CDP-gather-first
convenience ordering. The interpretation layer (horizon/fault picking,
attributes, well tie) and several individual steps (trace edit/kill,
multiple suppression, Constant-Velocity-Stack analysis) are not in the
source notebook at all and were added from those same references — see
"Where each piece came from" below.

**Nothing here is faked.** Every "Run" button performs the real,
documented computation on the data actually loaded — including on the
synthetic demo data, which exists only because no real SEG-Y file was
available (see "Synthetic test data" below).

---

## Where each piece came from

| Source | Used for |
|---|---|
| `Seismic_Processing_CORRECTED.ipynb` (reference notebook) | The core algorithms: AGC, spiking deconvolution, band-pass design, F-K filtering, semblance velocity analysis, NMO/DMO/inverse-NMO, residual statics, fold-normalized stacking, Kirchhoff migration |
| `Final_task.ipynb` (companion notebook) | Header inspection & wiggle/variable-area display conventions |
| A conventional marine seismic processing-flow diagram | Overall stage ordering: shot-domain pre-stack conditioning *before* CDP sort/binning; an early brute-stack/fold-map QC step; naming conventions (trace kill/edit, gain recovery, F-K dip filter, multiple suppression) |
| An industrial-training report (2D/3D land processing + interpretation) | The industry noise-attenuation sequence (de-spike → trace kill → ground-roll/F-K → multiple attenuation → deconvolution); predictive/gap deconvolution for multiple suppression; Constant-Velocity-Stack (CVS) analysis; the attribute taxonomy and formulas (RMS amplitude, variance, sweetness, coherence, instantaneous phase/frequency); post-stack time migration (PoSTM) terminology |
| Standard geophysical practice (no single source) | Instantaneous-attribute definitions (Taner, Koehler & Sheriff, 1979); the approximate well-tie method (reflectivity → wavelet convolution → correlation) |

---

## Features

- **Data Management** — SEG-Y upload (`segyio`), EBCDIC/binary/trace
  header inspection, optional **geometry merging** (combine a second
  survey), an optional early **brute stack + fold map** QC, then build
  shot gathers.
- **Pre-Stack Conditioning** (shot domain) — band-pass, AGC, **trace
  edit/kill + de-spiking**, top mute, trace balancing, F-K dip filter
  (**auto polygon or click-drawn**), **multiple suppression**
  (predictive/gap deconvolution), spiking deconvolution — each with
  before/after and spectrum QC.
- **CDP Sort & Binning** — re-groups the now-conditioned data by CDP,
  with a fold map and a semblance preview, right before velocity
  analysis (matching standard practice).
- **Velocity Analysis, NMO & DMO** — two-pass semblance velocity
  analysis with an automatic picker, **per-analysis-CDP reliability
  flagging**, **click-to-pick directly on the semblance panel**, manual
  table-based correction/exclusion, and a **Constant-Velocity-Stack
  (CVS)** panel as a complementary, amplitude-domain check; NMO (both
  passes), surface-consistent residual statics, simplified DMO.
- **Stacking** — fold-normalized CDP stack with a fold map and
  adjustable display (colormap / clip / gain / polarity).
- **Post-Stack Processing** — random noise attenuation, a second
  deconvolution pass, time-variant filter.
- **Migration** — post-stack Kirchhoff time migration (PoSTM) + final
  scaling.
- **Interpretation** — **click-to-pick** horizons and faults directly on
  the section (backed by an editable table for precise review), horizon
  curvature, and an approximate well-to-seismic tie.
- **Attributes** — RMS amplitude, instantaneous amplitude/phase/
  frequency, **sweetness**, **variance**, **coherence/similarity**,
  dominant frequency — with attribute-along-horizon extraction.
- **Export** — processed SEG-Y, CSV, NumPy, horizon/fault CSV, attribute
  CSV, interactive HTML figures, and a text/JSON processing report.
- **Processing history** — every stage's parameters are logged;
  re-running an earlier stage automatically invalidates every stage
  downstream, so the history can never silently go stale.

---

## Workflow

```
Load SEG-Y (or synthetic demo)
  Geometry Merging (optional)
        │
        ▼
Build shot gathers ── (an optional throwaway CDP grouping is available
        │               purely for an early Brute Stack / Fold Map QC)
        ▼
Pre-Stack Conditioning (shot domain)
  Band-pass → AGC → Trace edit/kill/de-spike → Mute → Balance →
  F-K filter (auto or click-drawn polygon) → Multiple suppression →
  Deconvolution                                          (QC after each)
        │
        ▼
CDP Sort / Binning  (re-group by CDP; fold map; semblance preview)
        │
        ▼
Velocity Analysis Pass 1 (auto-pick / click-pick / CVS)
        │            ──► NMO 1 ──► Residual Statics ──► DMO ──► Inverse NMO
        ▼
Velocity Analysis Pass 2 (refined) ──► Final NMO
        │
        ▼
Stack ──► Random Noise Attenuation ──► Deconvolution ──► Time-Variant Filter
        │
        ▼
Kirchhoff Migration (post-stack time) ──► Final Scaling = Processed Section
        │
        ▼
Interpretation (horizons / faults / curvature / well tie) + Attributes
        │
        ▼
Export (SEG-Y / CSV / NumPy / picks / figures / report)
```

---

## Supported Data

- **Input:** SEG-Y (`.sgy`, `.segy`), `.su` (read via the SEG-Y path)
- **Well tie:** CSV well log (`depth_m`, `vp_m_s`, `rho_kg_m3`), optional
  checkshot CSV (`depth_m`, `twt_s`)
- **Output:** SEG-Y, CSV, NumPy (`.npy`), interactive HTML, text/JSON

### Synthetic test data

No real SEG-Y file was available, so `modules/synthetic_data.py`
generates a genuine (if small) 2D land survey — shots, receivers, CDP
geometry, five flat reflectors with a realistic velocity-with-depth
trend, ground roll, a point diffractor (for validating migration), and
ambient noise — written out as a real SEG-Y file via the same writer
used for export. A small pre-generated copy ships at
`data/sample_data/synthetic_demo_line.sgy`; the Data Management page can
also generate a fresh, larger one on demand. **It is labeled as
synthetic everywhere it appears in the UI and is never presented as
real acquired data.** Every processing step still performs its real
computation on this data — only the input is synthetic.

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
├── config.py                  # every default parameter, in one place, with its
│                               #   source (which notebook step / reference) noted
├── modules/                   # pure computation — no Streamlit dependency
│   ├── segy_io.py             # SEG-Y read/write, headers, shot & CDP gathers,
│   │                           #   scatter-to-flat, geometry merging
│   ├── synthetic_data.py      # synthetic survey generator (testing/demo only)
│   ├── preprocessing.py       # AGC, de-spiking, trace kill, predictive &
│   │                           #   spiking deconvolution, balancing, mute
│   ├── filtering.py           # band-pass design, F-K filter, RNA, TVF, spectra
│   ├── velocity.py            # semblance, the redesigned auto-picker (see
│   │                           #   below), Constant-Velocity-Stack analysis
│   ├── nmo.py                 # NMO / inverse NMO / DMO (bug-fixed, see below)
│   ├── statics.py             # surface-consistent residual statics
│   ├── stacking.py            # fold-normalized stacking
│   ├── migration.py           # Kirchhoff migration + final scaling
│   ├── attributes.py          # RMS, envelope, phase, frequency, sweetness,
│   │                           #   variance, coherence (interpretation layer)
│   ├── interpretation.py      # horizon/fault data model, curvature, well-tie
│   ├── pipeline.py            # the ordered stage registry + processing history
│   └── validation.py          # friendly input/parameter validation
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
│   ├── session_state.py        # session-state schema + stage-invalidation logic
│   ├── click_picking.py        # shared click-to-pick helper (velocity, horizon,
│   │                             #   fault, F-K polygon all use this one mechanism)
│   ├── io_export.py            # SEG-Y/CSV/NumPy/HTML/report export helpers
│   └── formatting.py
├── data/sample_data/            # shipped synthetic demo SEG-Y
├── tests/                       # see "Testing" below
├── requirements.txt, .gitignore, .streamlit/config.toml
```

### Architecture notes

- **Why `views/` and not `pages/`:** Streamlit auto-generates multipage
  navigation from a literal `pages/` folder next to `app.py`. The
  reference Petrophysics app uses a single entry point with a sidebar
  `st.radio` instead, so a `pages/` folder here would silently create a
  *second*, conflicting navigation UI. `views/` holds the same kind of
  page-level modules the brief's proposed structure called `pages/`,
  without triggering that behavior.
- **No Docker.** The deployment path this app is built for (run
  locally → push to GitHub → deploy via Streamlit Community Cloud) never
  touches a container — Streamlit Cloud builds straight from
  `requirements.txt`. Adding a Dockerfile would be one more file to keep
  in sync with no benefit for that path, so it's left out; see
  "Deploying to Streamlit Community Cloud" above for the one-line note
  on building one yourself if you ever need it for a different host.
- **No hard-coded app-wide dark theme.** The reference app's CSS forces
  a fixed dark background; this app's own explicit requirement is to
  work in both light and dark Streamlit themes, so `app.py`'s CSS only
  touches buttons/tabs (readable either way) and leaves the theme itself
  to the person's Streamlit setting. Plotly figures are separately
  forced to a consistent white background in `visualization/theme.py`.
- **Shot domain first, CDP domain second.** `gathers` (CDP-indexed) and
  `shot_gathers` (shot-indexed) are both dicts of per-gather dicts that
  accumulate keys as data flows through the pipeline (`traces` →
  `traces_bp` → `traces_agc` → …), mirroring the source notebook's own
  pattern. Pre-stack conditioning runs entirely on `shot_gathers`; CDP
  sort/binning (`modules/segy_io.scatter_to_flat` +
  `build_cdp_gathers(..., data_override=...)`) re-groups the
  *conditioned* samples by CDP afterward, using the survey's own headers
  for grouping — nothing about the geometry changes, only which samples
  live in which gather.
- **Point-picking, one mechanism, four uses.** `utils/click_picking.py`
  provides `numeric_point_form()` — a small (x, y) + "Add point" control
  used identically for velocity control points, horizon picks, fault
  picks, and the F-K reject polygon. Earlier versions tried
  `st.plotly_chart(..., on_select="rerun")` for genuine click-to-pick,
  but Streamlit does not deliver selection events for `go.Heatmap`
  traces (only scatter/bar/histogram), which is what every one of these
  panels is drawn on — so a click there silently never fired, and
  turning `on_select` on also swapped the chart's drag mode from zoom to
  box-select, breaking normal zoom/pan too. Removing `on_select`
  restored native zoom/pan; the numeric form replaces the click.
- **SEG-Y library:** the source notebooks use `obspy` for trace data and
  `segyio` for headers. This app uses `segyio` alone for both reading
  and writing — it covers everything needed here, and dropping `obspy`
  removes a much heavier dependency without changing any processing
  math (see `modules/segy_io.py` docstring).

---

## Deviations from the source notebook (read this)

Per the priority order in the project brief (notebook > architecture
reference > PDFs > assumptions), the notebook's algorithms were
preserved as closely as possible — **except** where testing turned up a
genuine, reproducible defect, or where the requested stage ordering
(shot-domain-first) itself changes how a step interacts with its
neighbors. Both categories are documented in code comments at the point
of the change; summarized here:

### 1. The two NMO functions had swapped interpolation conventions (fixed)

The notebook's NMO function calls
`np.interp(t0[valid], tn[valid], traces[ix, valid])`. Verified
empirically on a synthetic trace with a single reflection event at a
known time: this exact formula maps the corrected event to the
**wrong** output time (0.754 s instead of the true 0.500 s).

The natural follow-up question — since `nmo_correction` and
`inverse_nmo_correction` are meant to be exact inverses of each other —
was whether fixing one was enough. A direct round-trip test (forward
NMO then inverse NMO should return the original trace) showed it
wasn't: the notebook's `inverse_nmo_correction` uses the *same*
convention as the (corrected) forward function, which is self-consistent
for forward warping but doesn't actually invert it. The actual inverse
of the corrected forward formula turns out to be exactly the convention
the notebook's (buggy) `nmo_correction` used. In short: **the notebook's
two NMO functions each implement the other one's stated purpose.**
`modules/nmo.py` now has `nmo_correction` using the forward convention
and `inverse_nmo_correction` using the swapped one, verified with a
direct round-trip test (`tests/test_pipeline.py::test_inverse_nmo_undoes_nmo`).
`dmo_correction` was left unchanged — it doesn't claim to invert
anything. Full derivation is in `modules/nmo.py`'s module docstring.

### 2. The automatic velocity picker was redesigned (fixed)

The notebook picks velocities with a dynamic-programming path that
maximizes cumulative semblance across the *whole* time record minus a
jump penalty. Tested against synthetic gathers with known reflector
velocities, this systematically under-tracked deep, isolated coherence
peaks (a 12% velocity error at 2.2 s/3400 m/s in testing) and let one
bad edge-CDP pick corrupt a large stretch of the velocity field via flat
extrapolation. `modules/velocity.py` replaces it with a confidence-gated
local walk seeded from the panel's strongest peak, plus a lateral-
consistency outlier check — recovers the true velocity to ~1% and
degrades gracefully at low fold instead of collapsing. Full derivation
and before/after numbers are in the module docstring.

### 3. De-spiking needed an isolation check, not just an outlier threshold (fixed during development)

The first version of `despike_traces` flagged any sample whose residual
from a local median exceeded a threshold times the local (MAD-based)
noise estimate. Tested against **clean** synthetic data with no injected
noise, this flagged ~2% of *all* samples — every genuine, sharp
reflection peak in an otherwise-quiet local window reads as an extreme
outlier once the local noise estimate collapses toward zero, which it
does in a quiet window regardless of the threshold (raising the
threshold from 6 to 60 barely changed the false-positive rate — the
underlying ratio is closer to "zero or huge" than a smooth continuum).
Fixed by adding an isolation check: a sample is only flagged if it's
also much larger than the *average* of its immediate neighbours' own
residuals — true burst/glitch spikes are 1-2 sample transients with
small neighbours; a real wavelet peak's neighbours ramp up and down
smoothly at comparable amplitude. This dropped the false-positive rate
on clean data to <0.05% while still catching an injected single-sample
spike in testing. See `modules/preprocessing.py`'s `despike_traces`
docstring.

### 4. A known, expected accuracy trade-off from decon-last ordering

The requested stage order runs deconvolution *last* in pre-stack
conditioning (after band-pass, AGC, trace edit, mute, balance, F-K, and
multiple suppression) rather than right after AGC, which is where the
very first version of this app ran it. Testing the final processed
section against the synthetic data's known reflector times shows
somewhat larger timing scatter with the new ordering (roughly 15-20 ms
versus under 10 ms before) rather than the previous run's tighter
figures. Isolating the cause: applying spiking deconvolution to a single
trace shows a similar-sized timing shift whether it's the first thing
done after AGC or the last thing done after every other conditioning
step — this is a property of applying a *minimum-phase* deconvolution
model to *zero-phase* synthetic Ricker wavelets (a well-known real
mismatch, not specific to this app), not an ordering-induced bug. It's
mentioned here because the change in the reported numbers is real and
was found through the same testing process as the fixes above; unlike
those, this one didn't get "fixed" because there's nothing incorrect to
fix — it's an expected numerical property, and the requested stage
ordering (matching standard industry practice) was kept as requested.
See `tests/manual_pipeline_check.py` for the reproducible comparison.

### Other documented simplifications (carried over from the notebook, unchanged)

- **DMO** is a common-offset time-stretch approximation, not a full
  log-stretch/Kirchhoff DMO operator.
- **No field statics** — only data-driven, surface-consistent residual
  statics are computed (no weathering-velocity/uphole-time data
  available to compute a field static from).
- **Random noise attenuation** is a spatial median filter across CDPs,
  standing in for F-X deconvolution (which needs higher fold than a
  small teaching/demo line typically has, to avoid its own artifacts).
- **Migration is post-stack (PoSTM) only** — no pre-stack time or depth
  migration.
- **Geometry merging** combines two already-geometry-complete surveys
  without ID collisions; it does not reconcile coordinates from external
  SPS navigation files the way a full industry geometry merge would.

---

## Interpretation & Attributes — not from the processing notebook

The source processing notebooks stop at migration; there is no
horizon/fault picking, attribute computation, or well-tie code to port.
The Interpretation and Attributes pages are new, built directly against
standard geophysical practice and the industrial-training report's
attribute section:

- Horizon/fault picking is **click-to-pick on the section**, backed by
  an editable table for precise review/correction — never an automatic
  tracker, since there is no source basis for one and a fabricated
  auto-tracker would risk presenting invented picks as real
  interpretation.
- **Horizon curvature** is the 2-D analogue (second derivative of the
  picked line) of the 3D surface-curvature attribute.
- Attributes follow the industrial-training report's definitions
  directly: RMS amplitude, variance (discontinuity), sweetness
  (amplitude/frequency hybrid), coherence/similarity, plus the standard
  instantaneous-attribute family (envelope, phase, frequency — Taner,
  Koehler & Sheriff, 1979).
- The well tie is explicitly labeled **approximate**: it needs a
  sonic/velocity + density log (CSV) and, optionally, a checkshot; the
  wavelet is a user-specified Ricker, not one statistically extracted
  from the seismic — a real refinement, noted under "Future
  improvements."

---

## Example Usage

1. Open the app, go to **Data Management**, and click "Generate
   synthetic survey" (or upload your own SEG-Y). Optionally merge in a
   second survey, or run the optional brute-stack QC. Then **build shot
   gathers**.
2. Work through **Pre-Stack Conditioning** tab by tab (band-pass → AGC →
   trace de-spike → trace kill → mute → balance → F-K filter → multiple
   suppression → deconvolution) — each has a "Run" button and
   before/after QC. For the F-K filter, try "Build my own (numeric
   points)" and zoom into the spectrum to read off a few (k, f)
   coordinates to enter as a custom reject polygon.
3. **CDP Sort & Binning** — re-group into CDP gathers, check the fold
   map, preview a semblance panel.
4. In **Velocity, NMO & DMO**, run Pass 1, review the reliability chart,
   zoom into the semblance panel and **enter a (velocity, time) point**
   below it to add a manual pick (or use the CVS panel as a second
   opinion), then continue through NMO → Statics → DMO → Inverse NMO →
   Data Regularisation → RMS Velocity Analysis (Pass 2) → Final NMO.
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
  merging, trace edit/de-spiking, predictive deconvolution, CVS, the new
  attributes, and horizon curvature.

Bugs found and fixed via this testing, beyond the NMO/picker/de-spike
items documented above: `pick_velocity_field` crashed outright on a
small-aperture survey where no CDP met the default minimum-offset
requirement (now falls back gracefully); several Plotly charts collided
on auto-generated element IDs when rendered in the same script run (now
explicitly keyed); Streamlit's `use_container_width` argument was
replaced with the current default (`width="stretch"`) after a
deprecation warning showed up during testing.

Run (requires `pip install pytest` in addition to `requirements.txt`):
```bash
pytest tests/test_pipeline.py -v
python tests/manual_pipeline_check.py
python tests/test_app_flow.py
```

---

## Known Limitations

- DMO is a simplified time-stretch approximation.
- No field statics (no uphole/weathering-velocity data available).
- Random noise attenuation is a spatial median filter, not F-X decon.
- Migration offers both post-stack Kirchhoff time migration and a
  pre-stack Kirchhoff time migration (PSTM) that sums directly from
  offset-preserved, regularized CDP gathers; both are still 2-D time
  migration (no depth migration, no 3-D).
- The well tie supports both a user-chosen Ricker wavelet and a
  statistically extracted wavelet (zero-phase, from the seismic data
  itself).
- Point-picking (velocity, horizon, fault, F-K polygon) is entered via a
  small numeric (x, y) + "Add point" form under a fully zoomable/
  pannable reference plot, rather than by clicking the plot directly:
  Streamlit does not deliver point-selection events for `go.Heatmap`
  traces (only scatter/bar/histogram do), so every one of these
  Heatmap-backed panels could never have supported real click-to-pick in
  the first place — this was confirmed, not assumed, during development
  (see `utils/click_picking.py`'s module docstring). The editable table
  next to every picker remains the full-precision way to review/correct
  entries either way.
- Geometry merging combines surveys without ID collisions but doesn't
  reconcile external SPS navigation coordinates.
- Designed and tested against a single 2D line; 3D survey geometry is
  out of scope.
- Large surveys: processing loops per-trace/per-gather in NumPy — fast
  enough for a teaching/demo-sized line but not optimized for a full
  production 3D volume.

## Recommended Future Improvements

- Statistical wavelet extraction from the seismic for the well tie.
- A full log-stretch or Kirchhoff DMO operator.
- F-X deconvolution as an alternative random-noise-attenuation method.
- Prestack time/depth migration.
- Automatic (assisted) horizon tracking seeded from manual picks.
- 3D survey support (inline/crossline geometry, 3D migration aperture).
- A full SPS-navigation-based geometry merge.
