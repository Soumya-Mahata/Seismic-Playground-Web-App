"""
modules/stage_ops.py — Stage executors (pure; no Streamlit)
================================================================
One function per processing stage. Each one:

  1. asks the dependency engine whether the stage can run
     (`dependencies.check_stage`) and raises `DependencyError` if a
     GENUINE input is missing — recommendations never raise;
  2. resolves its input from the *current data state* (the head of its
     data stream), so a stage reads whatever was actually applied last,
     not a hard-coded predecessor's key;
  3. calls the unchanged processing maths (modules/filtering, nmo, ...);
  4. records the run in the processing history together with which
     upstream outputs it consumed, invalidating exactly its dependents.

The Streamlit views call these, and so do the tests — which is what lets
tests run the *real* pipeline in non-standard orders without a UI.

`store` is any mutable mapping with the keys of
`pipeline.PROCESSING_STORE_DEFAULTS`: `st.session_state` in the app, a
plain dict (see `new_store`) everywhere else.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import config as cfg
from modules import filtering, migration, nmo, preprocessing, regularization, segy_io, stacking
from modules import statics as statics_mod
from modules import velocity as vel_mod
from modules.dependencies import (StageCheck, WorkflowState, check_stage)
from modules.pipeline import (DEFAULT_OFFSETS_KEY, NMO_PRODUCT_KEYS, PROCESSING_STORE_DEFAULTS,
                              SECTION_KEY, STAGE_GATHER_KEYS, STAGE_OFFSETS_KEY, STAGE_STORE_KEYS,
                              STAGE_TRACE_KEY, VELOCITY_FIELD_KEY, VELOCITY_GATHER_KEY,
                              HistoryEntry, stage_label)
from modules.validation import (DependencyError, ValidationError, require_gather_key,
                                validate_cdp_gathers, validate_frequency_band,
                                validate_velocity_field)

ProgressCb = Optional[Callable[[float], None]]

# Stages whose effect is a per-trace gain change: amplitude-based trace
# detection must look at data from *before* the first of them.
GAIN_ALTERING_STAGES: Tuple[str, ...] = ("agc", "balance")


# ─────────────────────────────────────────────────────────────────────────
# Store, state, bookkeeping
# ─────────────────────────────────────────────────────────────────────────

def new_store(survey=None) -> Dict[str, Any]:
    """A fresh, empty data store (a plain dict; same keys as the app's
    session state, minus its UI-only entries)."""
    store = copy.deepcopy(PROCESSING_STORE_DEFAULTS)
    if survey is not None:
        run_load_survey(store, survey, getattr(survey, "source_name", "survey"))
    return store


def workflow(store: MutableMapping) -> WorkflowState:
    """The current workflow state — a live view over `store['history']`."""
    if store.get("history") is None:
        store["history"] = []
    return WorkflowState(store["history"])


def check(store: MutableMapping, stage_id: str, variant: Optional[str] = None) -> StageCheck:
    return check_stage(workflow(store), stage_id, variant)


def clear_stage_data(store: MutableMapping, stage_id: str) -> None:
    """Free everything a (now invalid) stage produced."""
    for key in STAGE_STORE_KEYS.get(stage_id, []):
        store[key] = None
    container_key, keys = STAGE_GATHER_KEYS.get(stage_id, (None, ()))
    if container_key and store.get(container_key):
        for g in store[container_key].values():
            for k in keys:
                g.pop(k, None)


def invalidate_stage(store: MutableMapping, stage_id: str, include_self: bool = True) -> List[str]:
    """Drop a stage (optionally) and every stage built on it, freeing
    their data. Returns the stage ids removed."""
    state = workflow(store)
    removed = state.invalidate(stage_id, include_self=include_self)
    for sid in removed:
        clear_stage_data(store, sid)
    store["completed_stages"] = state.completed()
    return removed


@dataclass
class _Run:
    store: MutableMapping
    state: WorkflowState
    check: StageCheck
    stage_id: str
    variant: Optional[str]

    def src(self, role: str = "data") -> Optional[str]:
        return self.check.inputs.get(role)


def _begin(store: MutableMapping, stage_id: str, variant: Optional[str] = None) -> _Run:
    state = workflow(store)
    chk = check_stage(state, stage_id, variant)
    if not chk.can_run:
        missing = [s for i in chk.blockers for s in i.stages]
        raise DependencyError(f"Cannot run {stage_label(stage_id)}. {chk.blocker_text()}",
                              stage_id=stage_id, missing=missing)
    return _Run(store, state, chk, stage_id, variant)


def _begin_raw(store: MutableMapping) -> _Run:
    """A run record for the root stage (`upload`), which has no inputs."""
    state = workflow(store)
    return _Run(store, state, check_stage(state, "upload"), "upload", None)


def _record(run: _Run, params: Optional[dict] = None, input_key: str = "", output_key: str = "",
            offsets_key: str = "") -> HistoryEntry:
    notes = ""
    if run.check.warnings:
        notes = "Run despite: " + " | ".join(w.message for w in run.check.warnings)
    consumed = {role: sid for role, sid in run.check.inputs.items() if sid}
    removed = run.state.record(run.stage_id, consumed=consumed, params=params, notes=notes,
                               input_key=input_key, output_key=output_key, offsets_key=offsets_key,
                               variant=run.variant or "")
    for sid in removed:
        clear_stage_data(run.store, sid)
    run.store["completed_stages"] = run.state.completed()
    return run.state.entry(run.stage_id)


def data_keys(state: WorkflowState, stage_id: str) -> Tuple[str, str]:
    """(trace key, offsets key) of the data a stage's output holds."""
    e = state.entry(stage_id)
    trace_key = (e.output_key if e and e.output_key else STAGE_TRACE_KEY.get(stage_id, ""))
    offsets_key = (e.offsets_key if e and e.offsets_key
                   else STAGE_OFFSETS_KEY.get(stage_id, DEFAULT_OFFSETS_KEY))
    return trace_key, offsets_key


def kill_detection_stage(chain: Sequence[str]) -> str:
    """Which stage's output trace-kill detection should measure.

    Energy-based dead/noisy detection is blind after a per-trace gain
    (AGC, balancing) has equalised every trace's level, so it must look at
    amplitude-preserving data: the output of the stage just BEFORE the
    first gain-altering stage in the data's history. `chain` is the data's
    ancestry, oldest first, ending with the stage being edited. With no
    gain stage in it, the edited data itself is measured."""
    for i, sid in enumerate(chain):
        if sid in GAIN_ALTERING_STAGES and i > 0:
            return chain[i - 1]
    return chain[-1]


def _shot_io(run: _Run) -> Tuple[dict, str]:
    shot_gathers = run.store["shot_gathers"]
    in_key, _ = data_keys(run.state, run.src("data"))
    require_gather_key(shot_gathers, in_key, f"the {stage_label(run.src('data'))} output",
                       stage_label(run.stage_id))
    return shot_gathers, in_key


def _progress(items_len: int, i: int, cb: ProgressCb) -> None:
    if cb and (i % 20 == 0):
        cb((i + 1) / items_len)


# ─────────────────────────────────────────────────────────────────────────
# Data: load, shot gathers
# ─────────────────────────────────────────────────────────────────────────

def run_load_survey(store: MutableMapping, survey, source_label: str) -> HistoryEntry:
    """Install a survey as the current data. Everything derived from the
    previous survey (all stages) is invalidated."""
    invalidate_stage(store, "upload", include_self=True)
    store["survey"] = survey
    run = _begin_raw(store)
    return _record(run, params={"source": source_label, "traces": survey.num_traces},
                   output_key="raw_data")


def run_build_shot_gathers(store: MutableMapping, n_min_traces: int = 1) -> HistoryEntry:
    run = _begin(store, "shot_gathers")
    shot_gathers = segy_io.build_shot_gathers(store["survey"], n_min_traces=n_min_traces)
    store["shot_gathers"] = shot_gathers
    return _record(run, params={"n_shots": len(shot_gathers)}, output_key="traces",
                   offsets_key=DEFAULT_OFFSETS_KEY)


# ─────────────────────────────────────────────────────────────────────────
# Shot-domain pre-stack conditioning — each independent of the others
# ─────────────────────────────────────────────────────────────────────────

def run_bandpass(store: MutableMapping, fmin: float = cfg.FMIN_DEFAULT, fmax: float = cfg.FMAX_DEFAULT,
                 low_slope: float = cfg.LOW_SLOPE_DEFAULT, high_slope: float = cfg.HIGH_SLOPE_DEFAULT
                 ) -> HistoryEntry:
    run = _begin(store, "bandpass")
    survey = store["survey"]
    shot_gathers, in_key = _shot_io(run)
    validate_frequency_band(fmin, fmax, 0.5 / survey.sample_interval)
    h = filtering.design_bandpass(fmin, fmax, survey.sample_interval, survey.num_samples,
                                  low_slope, high_slope)
    for g in shot_gathers.values():
        g["traces_bp"] = filtering.apply_time_domain_conv(g[in_key], h)
    return _record(run, {"fmin_hz": fmin, "fmax_hz": fmax}, in_key, "traces_bp", DEFAULT_OFFSETS_KEY)


def run_agc(store: MutableMapping, window: int = cfg.AGC_WINDOW_DEFAULT,
            mode: str = cfg.AGC_MODE_DEFAULT) -> HistoryEntry:
    run = _begin(store, "agc")
    shot_gathers, in_key = _shot_io(run)
    for g in shot_gathers.values():
        for m in ("mean", "rms", "median"):
            g.pop(f"traces_agc_{m}", None)
        g["traces_agc"] = preprocessing.agc(g[in_key], window=window, mode=mode)
    store["agc_config"] = {"window": window, "mode": mode}
    return _record(run, {"window": window, "mode": mode}, in_key, "traces_agc", DEFAULT_OFFSETS_KEY)


def run_despike(store: MutableMapping, window: int = 15, threshold: float = 6.0) -> HistoryEntry:
    run = _begin(store, "despike")
    shot_gathers, in_key = _shot_io(run)
    total = 0
    for g in shot_gathers.values():
        cleaned, spike_mask = preprocessing.despike_traces(g[in_key], window, threshold)
        g["traces_despiked"] = cleaned
        g["_spike_mask"] = spike_mask
        total += int(spike_mask.sum())
    return _record(run, {"window": window, "threshold": threshold, "samples_flagged": total},
                   in_key, "traces_despiked", DEFAULT_OFFSETS_KEY)


def run_trace_kill(store: MutableMapping, dead_frac: float = 0.02, noisy_mult: float = 8.0
                   ) -> HistoryEntry:
    """Kill dead/noisy traces. The kill mask is *measured* on
    amplitude-preserving data (upstream of any AGC / balancing in this
    data's history — see `kill_detection_stage`) and *applied* to the
    data actually being processed."""
    run = _begin(store, "trace_edit")
    shot_gathers, in_key = _shot_io(run)
    chain = run.state.data_ancestry(run.src("data"))
    det_stage = kill_detection_stage(chain)
    det_key, _ = data_keys(run.state, det_stage)
    require_gather_key(shot_gathers, det_key, f"the {stage_label(det_stage)} output",
                       stage_label("trace_edit"))
    total = 0
    for g in shot_gathers.values():
        flags = preprocessing.flag_bad_traces(g[det_key], dead_frac, noisy_mult)
        g["traces_edited"] = preprocessing.apply_trace_kill(g[in_key], flags["kill"])
        g["_kill_flags"] = flags
        total += int(flags["kill"].sum())
    return _record(run, {"traces_killed": total, "dead_frac": dead_frac, "noisy_mult": noisy_mult,
                         "detected_on": stage_label(det_stage)},
                   in_key, "traces_edited", DEFAULT_OFFSETS_KEY)


def mute_summary(kind: str, **kw) -> str:
    if kind == "flat":
        return f"flat top mute, t_top={kw['t_top'] * 1000:.0f} ms"
    if kind == "linear":
        return f"top mute (moveout), v={kw['velocity']:.0f} m/s, t0={kw['t0'] * 1000:.0f} ms"
    return (f"surgical mute, offset [{kw['x_min']:.0f},{kw['x_max']:.0f}] m, "
            f"time [{kw['t_min'] * 1000:.0f},{kw['t_max'] * 1000:.0f}] ms")


def run_mute(store: MutableMapping, kind: str = "flat", *, taper: float = cfg.MUTE_TAPER_DEFAULT,
             t_top: float = cfg.MUTE_TTOP_DEFAULT, velocity: float = 500.0, t0: float = 0.0,
             x_min: float = 0.0, x_max: float = 0.0, t_min: float = 0.0, t_max: float = 0.0
             ) -> HistoryEntry:
    if kind not in ("flat", "linear", "surgical"):
        raise ValidationError(f"Unknown mute type '{kind}' (expected flat, linear or surgical).")
    run = _begin(store, "mute")
    dt = store["survey"].sample_interval
    shot_gathers, in_key = _shot_io(run)
    kwargs: Dict[str, Any]
    if kind == "flat":
        kwargs = dict(kind=kind, t_top=t_top, taper=taper)
    elif kind == "linear":
        kwargs = dict(kind=kind, velocity=velocity, t0=t0, taper=taper)
    else:
        kwargs = dict(kind=kind, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max, taper=taper)
    for g in shot_gathers.values():
        data = g[in_key]
        if kind == "flat":
            muted = preprocessing.apply_top_bottom_mute(data, dt, t_top, taper_s=taper)
            mute_times, in_box = np.full(data.shape[0], t_top), None
        elif kind == "linear":
            muted, mute_times = preprocessing.apply_linear_mute(
                data, g["offsets"], dt, velocity, t_intercept=t0, taper_s=taper)
            in_box = None
        else:
            muted, in_box = preprocessing.apply_surgical_mute(
                data, g["offsets"], dt, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max,
                taper_s=taper)
            mute_times = np.full(data.shape[0], t_max)
        g["traces_muted"], g["_mute_times"], g["_surgical_box"] = muted, mute_times, in_box
    summary = mute_summary(**kwargs)
    store["mute_config"] = {**kwargs, "summary": summary}
    return _record(run, {"type": kind, **{k: v for k, v in kwargs.items() if k != "kind"}},
                   in_key, "traces_muted", DEFAULT_OFFSETS_KEY)


def run_balance(store: MutableMapping, ref_s: float = cfg.NORM_WINDOW_S_DEFAULT) -> HistoryEntry:
    run = _begin(store, "balance")
    dt = store["survey"].sample_interval
    shot_gathers, in_key = _shot_io(run)
    ref_window = slice(0, int(ref_s / dt))
    for g in shot_gathers.values():
        g["traces_balanced"] = preprocessing.trace_normalize(g[in_key], ref_window)
    return _record(run, {"ref_window_s": ref_s}, in_key, "traces_balanced", DEFAULT_OFFSETS_KEY)


def run_fk(store: MutableMapping, polygon, mode: str = cfg.FK_POLYGON_MODE_DEFAULT,
           coeff: float = cfg.FK_REJECT_COEFFICIENT_DEFAULT, source_desc: str = "custom polygon",
           progress_cb: ProgressCb = None) -> HistoryEntry:
    if polygon is None or len(polygon) < 3:
        raise ValidationError("An F-K polygon needs at least 3 vertices.")
    run = _begin(store, "fk")
    survey = store["survey"]
    shot_gathers, in_key = _shot_io(run)
    store["fk_polygon"] = list(polygon)
    store["fk_config"] = {"mode": mode, "coeff": float(coeff), "source": source_desc}
    items = list(shot_gathers.values())
    for i, g in enumerate(items):
        filt, *_ = filtering.fk_filter_gather(g[in_key], survey.sample_interval, survey.group_interval,
                                              polygon, polygon_mode=mode, reject_coefficient=coeff)
        g["traces_fk_filtered"] = filt
        _progress(len(items), i, progress_cb)
    if progress_cb:
        progress_cb(1.0)
    n_vertices = (sum(len(ring) for ring in polygon)
                  if polygon and isinstance(polygon[0][0], (list, tuple)) else len(polygon))
    return _record(run, {"polygon_vertices": n_vertices, "mode": mode, "coefficient": coeff,
                         "source": source_desc}, in_key, "traces_fk_filtered", DEFAULT_OFFSETS_KEY)


def run_mult_supp(store: MutableMapping, operator_length: int = 60, prediction_lag: int = 30,
                  pre_whitening: float = cfg.DECON_PW_DEFAULT, progress_cb: ProgressCb = None
                  ) -> HistoryEntry:
    run = _begin(store, "mult_supp")
    shot_gathers, in_key = _shot_io(run)
    items = list(shot_gathers.values())
    for i, g in enumerate(items):
        g["traces_demultiple"] = preprocessing.predictive_decon_gather(
            g[in_key], operator_length, prediction_lag, pre_whitening)
        _progress(len(items), i, progress_cb)
    if progress_cb:
        progress_cb(1.0)
    return _record(run, {"operator_length": operator_length, "prediction_lag_samples": prediction_lag},
                   in_key, "traces_demultiple", DEFAULT_OFFSETS_KEY)


def run_decon(store: MutableMapping, operator_length: int = cfg.DECON_OPLEN_DEFAULT,
              pre_whitening: float = cfg.DECON_PW_DEFAULT, progress_cb: ProgressCb = None
              ) -> HistoryEntry:
    run = _begin(store, "decon")
    shot_gathers, in_key = _shot_io(run)
    items = list(shot_gathers.values())
    for i, g in enumerate(items):
        g["traces_decon"] = preprocessing.spiking_decon_gather(g[in_key], operator_length, pre_whitening)
        _progress(len(items), i, progress_cb)
    if progress_cb:
        progress_cb(1.0)
    return _record(run, {"operator_length": operator_length, "pre_whitening": pre_whitening},
                   in_key, "traces_decon", DEFAULT_OFFSETS_KEY)


# ─────────────────────────────────────────────────────────────────────────
# CDP sort
# ─────────────────────────────────────────────────────────────────────────

def run_cdp_sort(store: MutableMapping, n_min_traces: int = cfg.N_MIN_TRACES_DEFAULT) -> HistoryEntry:
    """Group traces by CDP. Takes the current shot-domain data (whatever
    conditioning was applied, in whatever order) — or the raw survey if
    no shot gathers were built, since sorting only needs the headers."""
    run = _begin(store, "cdp_sort")
    survey = store["survey"]
    src = run.src("data")
    if src == "upload":
        in_key = "raw_data"
        gathers = segy_io.build_cdp_gathers(survey, n_min_traces=n_min_traces)
    else:
        in_key, _ = data_keys(run.state, src)
        require_gather_key(store["shot_gathers"], in_key, f"the {stage_label(src)} output", "CDP Sort")
        flat = segy_io.scatter_to_flat(store["shot_gathers"], in_key, survey.num_traces,
                                       survey.num_samples)
        gathers = segy_io.build_cdp_gathers(survey, n_min_traces=n_min_traces, data_override=flat)
    validate_cdp_gathers(gathers, n_min_traces)
    store["n_min_traces"] = n_min_traces
    store["gathers"] = gathers
    store["cdp_list"] = list(gathers.keys())
    return _record(run, {"n_min_traces": n_min_traces, "n_cdp": len(gathers)}, in_key, "traces",
                   DEFAULT_OFFSETS_KEY)


# ─────────────────────────────────────────────────────────────────────────
# CDP domain: velocity, NMO, statics, DMO, regularisation
# ─────────────────────────────────────────────────────────────────────────

_VEL_STATE_KEYS = {"vel1": ("Vfield_1", "analysis_1", "vel_results_1", "manual_overrides_1",
                            "excluded_cdps_1"),
                   "vel2": ("Vfield_2", "analysis_2", "vel_results_2", "manual_overrides_2",
                            "excluded_cdps_2")}


def _cdp_io(run: _Run, role: str = "data") -> Tuple[dict, list, str, str]:
    gathers, cdp_list = run.store["gathers"], run.store["cdp_list"]
    in_key, off_key = data_keys(run.state, run.src(role))
    require_gather_key(gathers, in_key, f"the {stage_label(run.src(role))} output",
                       stage_label(run.stage_id))
    return gathers, cdp_list, in_key, off_key


def run_velocity(store: MutableMapping, stage_id: str = "vel1", *, vmin: float = cfg.VELOCITY_MIN_DEFAULT,
                 vmax: float = cfg.VELOCITY_MAX_DEFAULT, n_analysis: int = cfg.N_ANALYSIS_CDPS_DEFAULT,
                 min_offset: float = cfg.MIN_ABS_OFFSET_DEFAULT,
                 window_len: int = cfg.SEMBLANCE_WINDOW_LEN_DEFAULT,
                 min_time: float = cfg.AUTOPICK_MIN_TIME_DEFAULT,
                 search_w: float = cfg.AUTOPICK_SEARCH_HALF_WIDTH_MS_DEFAULT,
                 min_conf: float = cfg.AUTOPICK_MIN_CONFIDENCE_DEFAULT,
                 smooth: int = cfg.AUTOPICK_SMOOTH_SIGMA_DEFAULT,
                 reliability_threshold: float = cfg.RELIABILITY_THRESHOLD_DEFAULT,
                 progress_cb: ProgressCb = None) -> HistoryEntry:
    """Semblance velocity analysis on the CDP gathers. Pass 1 analyses the
    CDP-sorted gathers; pass 2 analyses whatever the pre-NMO gathers are
    by then (refined by statics / DMO / regularisation, or not)."""
    if stage_id not in _VEL_STATE_KEYS:
        raise ValueError(f"run_velocity: stage must be 'vel1' or 'vel2', got {stage_id!r}")
    run = _begin(store, stage_id)
    survey = store["survey"]
    gathers, cdp_list, in_key, off_key = _cdp_io(run)
    field_key, analysis_key, results_key, overrides_key, excluded_key = _VEL_STATE_KEYS[stage_id]
    velocities = np.linspace(vmin, vmax, cfg.VELOCITY_N_DEFAULT)
    Vfield, analysis, results = vel_mod.pick_velocity_field(
        gathers, cdp_list, in_key, velocities, survey.sample_interval, survey.num_samples,
        n_analysis=int(n_analysis), min_abs_offset=min_offset, label=stage_id,
        window_len=int(window_len), reliability_threshold=reliability_threshold,
        autopick_kwargs=dict(min_time=min_time, search_half_width_ms=search_w, min_confidence=min_conf,
                             smooth_sigma=int(smooth)),
        manual_overrides=store[overrides_key], excluded_cdps=store[excluded_key],
        progress_cb=progress_cb, offsets_key=off_key)
    store["velocities"] = velocities
    store[field_key], store[analysis_key], store[results_key] = Vfield, analysis, results
    n_reliable = sum(r.is_reliable for r in results.values())
    return _record(run, {"n_analysis": len(analysis), "reliable": f"{n_reliable}/{len(analysis)}"},
                   in_key, "", off_key)


def run_nmo1(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT) -> HistoryEntry:
    run = _begin(store, "nmo1")
    survey = store["survey"]
    gathers, _, in_key, off_key = _cdp_io(run)
    vkey = VELOCITY_GATHER_KEY[run.src("velocity")]
    require_gather_key(gathers, vkey, "the pass-1 velocity curves", "NMO Pass 1")
    for g in gathers.values():
        out, live = nmo.nmo_correction(g[in_key], g[off_key], g[vkey], survey.sample_interval,
                                       stretch_mute=stretch_mute)
        g["traces_nmo1"], g["nmo1_live_mask"] = out, live
    return _record(run, {"stretch_mute": stretch_mute}, in_key, "traces_nmo1", off_key)


def run_statics(store: MutableMapping, max_shift_s: float = cfg.MAX_STATIC_SHIFT_S_DEFAULT
                ) -> HistoryEntry:
    run = _begin(store, "statics")
    dt = store["survey"].sample_interval
    gathers, _, in_key, off_key = _cdp_io(run)
    require_gather_key(gathers, "traces_nmo1", "the NMO Pass 1 output", "Residual Statics")
    shot_static, receiver_static, diag = statics_mod.solve_residual_statics(
        gathers, "traces_nmo1", "nmo1_live_mask", dt, max_shift_s=max_shift_s)
    statics_mod.apply_residual_statics(gathers, in_key, "traces_statics", shot_static,
                                       receiver_static, dt)
    store["shot_static"], store["receiver_static"], store["static_diag"] = (
        shot_static, receiver_static, diag)
    return _record(run, dict(diag), in_key, "traces_statics", off_key)


def run_dmo(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT) -> HistoryEntry:
    """DMO as implemented here NMO-corrects its own input with the pass-1
    velocities first, so it needs pre-NMO gathers and velocities — it does
    not need NMO Pass 1 or residual statics to have been run."""
    run = _begin(store, "dmo")
    dt = store["survey"].sample_interval
    gathers, _, in_key, off_key = _cdp_io(run)
    vkey = VELOCITY_GATHER_KEY[run.src("velocity")]
    require_gather_key(gathers, vkey, "the pass-1 velocity curves", "DMO")
    for g in gathers.values():
        nmo_in, _ = nmo.nmo_correction(g[in_key], g[off_key], g[vkey], dt, stretch_mute=stretch_mute)
        g["traces_nmo1_statics"] = nmo_in
        g["traces_dmo"] = nmo.dmo_correction(nmo_in, g[off_key], g[vkey], dt)
    return _record(run, {}, in_key, "traces_dmo", off_key)


def run_inverse_nmo(store: MutableMapping) -> HistoryEntry:
    run = _begin(store, "inv_nmo")
    dt = store["survey"].sample_interval
    gathers, _, in_key, off_key = _cdp_io(run)       # in_key = traces_dmo
    vkey = VELOCITY_GATHER_KEY[run.src("velocity")]
    for g in gathers.values():
        g["traces_inv_nmo"] = nmo.inverse_nmo_correction(g[in_key], g[off_key], g[vkey], dt)
    return _record(run, {}, in_key, "traces_inv_nmo", off_key)


def run_data_reg(store: MutableMapping, n_offsets: int = 31) -> HistoryEntry:
    run = _begin(store, "data_reg")
    gathers, _, in_key, off_key = _cdp_io(run)
    store["regular_offsets"] = regularization.regularize_all_gathers(
        gathers, input_key=in_key, offsets_key=off_key, n_offsets=int(n_offsets))
    return _record(run, {"n_offsets": int(n_offsets)}, in_key, "traces_regularized",
                   "offsets_regularized")


def run_nmo_final(store: MutableMapping, stretch_mute: float = cfg.NMO_STRETCH_MUTE_DEFAULT
                  ) -> HistoryEntry:
    """NMO on the current pre-NMO gathers with the best velocities
    available (pass 2, else pass 1)."""
    run = _begin(store, "nmo_final")
    dt = store["survey"].sample_interval
    gathers, _, in_key, off_key = _cdp_io(run)
    vel_stage = run.src("velocity")
    vkey = VELOCITY_GATHER_KEY[vel_stage]
    require_gather_key(gathers, vkey, f"the {stage_label(vel_stage)} velocity curves", "Final NMO")
    regularised = "data_reg" in run.state.data_ancestry(run.src("data"))
    for g in gathers.values():
        out, live = nmo.nmo_correction(g[in_key], g[off_key], g[vkey], dt, stretch_mute=stretch_mute)
        if regularised:
            # AND with the regularisation live-mask so offsets outside a gather's real fold
            # (zero-filled by interpolation, not real data) don't inflate the stack's fold count.
            live = live & g["live_regularized"][:, None]
        g["traces_nmo_final"], g["nmo_final_live_mask"] = out, live
    return _record(run, {"stretch_mute": stretch_mute, "velocity": stage_label(vel_stage)},
                   in_key, "traces_nmo_final", off_key)


# ─────────────────────────────────────────────────────────────────────────
# Stack and post-stack
# ─────────────────────────────────────────────────────────────────────────

def run_stack(store: MutableMapping) -> HistoryEntry:
    """Stack the best available NMO-corrected gathers (Final NMO, else
    NMO Pass 1). Stacking un-flattened gathers is a ⛔ — it is blocked by
    the dependency check, not merely discouraged."""
    run = _begin(store, "stack")
    gathers, cdp_list = store["gathers"], store["cdp_list"]
    src = run.src("data")
    traces_key, live_key = NMO_PRODUCT_KEYS[src]
    require_gather_key(gathers, traces_key, f"the {stage_label(src)} output", "Stack")
    stack_matrix, fold_matrix, naive_matrix = stacking.build_stack_section(
        gathers, cdp_list, traces_key, live_key)
    store["stack_matrix"], store["fold_matrix"], store["naive_matrix"] = (
        stack_matrix, fold_matrix, naive_matrix)
    return _record(run, {"n_cdp": len(cdp_list), "stacked": stage_label(src)}, traces_key, "stack_matrix")


def _section_in(run: _Run) -> Tuple[np.ndarray, str]:
    src = run.src("data")
    key = SECTION_KEY[src]
    section = run.store.get(key)
    if section is None:
        raise DependencyError(f"{stage_label(src)} output is missing from the data store.",
                              stage_id=run.stage_id, missing=[stage_label(src)])
    return section, key


def run_rna(store: MutableMapping, spatial_window: int = cfg.RNA_SPATIAL_WINDOW_DEFAULT
            ) -> HistoryEntry:
    run = _begin(store, "rna")
    section, key = _section_in(run)
    store["stack_rna"] = filtering.random_noise_attenuation(section, spatial_window)
    return _record(run, {"spatial_window": spatial_window}, key, "stack_rna")


def run_decon_post(store: MutableMapping, operator_length: int = cfg.DECON_AFTER_OPLEN_DEFAULT,
                   pre_whitening: float = cfg.DECON_AFTER_PW_DEFAULT) -> HistoryEntry:
    run = _begin(store, "decon_post")
    section, key = _section_in(run)
    store["stack_decon_post"] = preprocessing.spiking_decon_gather(section, operator_length, pre_whitening)
    return _record(run, {"operator_length": operator_length, "pre_whitening": pre_whitening},
                   key, "stack_decon_post")


def run_tvf(store: MutableMapping) -> HistoryEntry:
    run = _begin(store, "tvf")
    dt = store["survey"].sample_interval
    section, key = _section_in(run)
    store["stack_tvf"] = filtering.time_variant_filter(section, dt, cfg.TVF_ZONES_DEFAULT,
                                                       cfg.TVF_TAPER_S_DEFAULT)
    return _record(run, {"zones": len(cfg.TVF_ZONES_DEFAULT)}, key, "stack_tvf")


# ─────────────────────────────────────────────────────────────────────────
# Migration (two implementations) and final scaling
# ─────────────────────────────────────────────────────────────────────────

def _migration_velocity(run: _Run) -> Tuple[np.ndarray, str]:
    vel_stage = run.src("velocity")
    vfield = run.store.get(VELOCITY_FIELD_KEY[vel_stage])
    survey, cdp_list = run.store["survey"], run.store["cdp_list"]
    validate_velocity_field(vfield, survey.num_samples, len(cdp_list),
                            what=f"{stage_label(vel_stage)} velocity field")
    return vfield, vel_stage


def run_migration_poststack(store: MutableMapping, aperture_m: float = cfg.APERTURE_M_DEFAULT
                            ) -> HistoryEntry:
    """Post-stack Kirchhoff time migration: needs a stacked section and a
    velocity field (pass 2 if available, else pass 1)."""
    run = _begin(store, "migration", "poststack")
    survey, cdp_list = store["survey"], store["cdp_list"]
    section, key = _section_in(run)
    vfield, vel_stage = _migration_velocity(run)
    aperture_traces = max(1, int(round(aperture_m / survey.group_interval)))
    x_positions = np.array(cdp_list, dtype=float) * survey.group_interval
    store["stack_migrated"] = migration.kirchhoff_migration(
        section, x_positions, survey.twt, vfield, aperture_traces)
    store["pstm_image"] = None
    store["pstm_fold"] = None
    return _record(run, {"domain": "post-stack", "aperture_m": aperture_m,
                         "velocity": stage_label(vel_stage)}, key, "stack_migrated")


def run_migration_pstm(store: MutableMapping, aperture_m: float = cfg.APERTURE_M_DEFAULT,
                       cdp_stride: int = 3, progress_cb: ProgressCb = None) -> HistoryEntry:
    """Pre-stack Kirchhoff time migration: needs pre-stack, offset-
    preserving CDP gathers (NOT NMO-corrected, NOT stacked) and a velocity
    field. It performs migration and stack in one pass, so no stack is
    required — and regularisation is recommended, not required."""
    run = _begin(store, "migration", "pstm")
    survey, cdp_list = store["survey"], store["cdp_list"]
    gathers, _, in_key, off_key = _cdp_io(run)
    vfield, vel_stage = _migration_velocity(run)
    aperture_traces = max(1, int(round(aperture_m / survey.group_interval)))
    migrated, fold = migration.prestack_kirchhoff_time_migration(
        gathers, cdp_list, in_key, survey.twt, vfield, survey.group_interval, aperture_traces,
        offsets_key=off_key, cdp_stride=cdp_stride, progress_cb=progress_cb)
    store["pstm_image"], store["pstm_fold"], store["stack_migrated"] = migrated, fold, migrated
    return _record(run, {"domain": "pre-stack (PSTM)", "aperture_m": aperture_m,
                         "cdp_stride": cdp_stride, "velocity": stage_label(vel_stage)},
                   in_key, "stack_migrated", off_key)


def run_final_scale(store: MutableMapping, window: int = cfg.FINAL_AGC_WINDOW_DEFAULT) -> HistoryEntry:
    run = _begin(store, "final_scale")
    dt = store["survey"].sample_interval
    section, key = _section_in(run)
    store["stack_final"] = migration.final_scaling(section, dt, window=window,
                                                   t_top_mute=cfg.MUTE_TTOP_DEFAULT)
    return _record(run, {"agc_window": window}, key, "stack_final")


# ─────────────────────────────────────────────────────────────────────────
# Reading results back
# ─────────────────────────────────────────────────────────────────────────

def best_available_section(store: MutableMapping) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """The most advanced valid section: final-scaled > migrated > the head
    of the post-stack chain (TVF > decon > RNA > stack). Only sections
    that are still valid in the history count, so a stale array can never
    be exported or interpreted."""
    state = workflow(store)
    if state.has("final_scale") and store.get("stack_final") is not None:
        return store["stack_final"], "Final processed section"
    if state.has("migration") and store.get("stack_migrated") is not None:
        return store["stack_migrated"], "Migrated section"
    head = state.stream_head("section")
    if head and store.get(SECTION_KEY[head]) is not None:
        label = {"stack": "Stack", "rna": "Post-stack (after random noise attenuation)",
                 "decon_post": "Post-stack (after deconvolution)",
                 "tvf": "Post-stack (pre-migration)"}[head]
        return store[SECTION_KEY[head]], label
    return None, None
