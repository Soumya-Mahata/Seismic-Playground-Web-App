"""
modules/registry.py — the operation registry: ONE declaration per processing operation
======================================================================================
Everything the application needs to *know about* an operation is declared here,
once. Nothing else defines it:

  identity & presentation   id, variant, display name, short name, the name used
                            inside a recommendation sentence, description, help
                            text, category, layout group, where it lives in the app
  data types                the kind of state it produces (`output_type`) and,
                            derived from its needs, the kinds it reads
  hard dependencies         `needs`     — an input the computation cannot run
                            without. Missing  ->  ⛔ blocked.
                            `exclusions` — data it must NOT read (statics on
                            regularised gathers).  ->  ⛔ blocked.
  recommendations           `advice`    — what is normally applied first.
                            ->  ⚠ warning, never a block.
  warnings                  derived: the advice, the fallback inputs, the order
                            inversions, plus static `caveats` (see `possible_warnings`)
  parameters                `params`    — the parameter schema: name, type, default
                            (from `config.py`), unit, bounds / choices, help
  validation                `validate`  — operation-level checks; the schema's own
                            checks run first (`Operation.validate_params`)
  processing function       `compute`   — bound by `modules/stage_ops.py`
  QC builder                `qc`        — optional extra QC for the action log, bound
                            by `stage_ops` (a generic RMS/diagnostics summary is
                            always produced)

Who reads it
------------
  modules/pipeline.py      PIPELINE_STAGES, STAGE_KIND, ROOT_STAGES, STAGE_GROUPS —
                           derived views, kept under their old names
  modules/dependencies.py  the ⛔/⚠/✓ engine: needs, advice, exclusions, names,
                           streams, status rows — all from here
  modules/stage_ops.py     `execute` (validation, executors, QC), the history labels
  modules/state_graph.py   the kinds of data a state can hold
  utils/workflow_ui.py     status rows, layout groups, icons
  modules/registry_docs.py docs/OPERATIONS.md and the workflow-graph data
  tests/test_registry.py   completeness and consistency

What is deliberately NOT here
-----------------------------
The processing maths (`modules/{filtering,preprocessing,nmo,velocity,statics,
regularization,stacking,migration}.py`) and the rules that *evaluate* needs and
advice against a graph (`modules/dependencies.py`). The registry declares; the
engine decides.

Why implementations are "bound", not imported
---------------------------------------------
A compute function needs `stage_ops.Run`, so the registry cannot import it
without a cycle. `stage_ops` attaches each function to its operation at import
time (`bind`); `Operation.compute` / `.extras` / `.qc` look the binding up.
`problems(require_implementations=True)` (run by the tests) reports any operation
left without one.

No Streamlit dependency. The registry is plain, immutable data plus queries.
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import config as cfg
from modules import validation as V
from modules.validation import ValidationError


# ─────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Need:
    """One genuine input of an operation.

    `candidates` are tried in order of preference. A token starting with
    '@' means "the current state of that kind" ('@shot', '@cdp',
    '@section'); anything else is a stage id whose latest state is the
    input. If a candidate other than the first supplies the input and
    `fallback` is set, the operation is still runnable but gets a ⚠ with that
    text.

    `track`: a finished state is *stale* when the input this need would
    resolve to today is not the one it recorded."""
    role: str
    candidates: Tuple[str, ...]
    what: str
    why: str = ""
    fix: str = ""
    fallback: str = ""
    track: bool = False


@dataclass(frozen=True)
class Advice:
    """A scientifically preferable ordering — never blocks.

    mode "all": each listed stage should be in the lineage of the data
    this operation reads (one ⚠ naming whichever are missing).
    mode "any": at least one should be."""
    stages: Tuple[str, ...]
    reason: str
    mode: str = "all"
    message: str = ""     # overrides the generated sentence


@dataclass(frozen=True)
class Exclusion:
    """Data an operation must NOT read: if the input's lineage contains any of
    `stages` the operation is ⛔ (a structural impossibility, not a preference)."""
    role: str
    stages: Tuple[str, ...]
    message: str
    fix: str = ""
    kind: str = "geometry"


REQUIRED: Any = object()          # `ParamSpec.default` of a parameter with no default


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of an operation, as recorded on the state it produces.

    `type`: "float" | "int" | "choice" | "polygon" | "list" | "dict" | "text".
    `default` comes from `config.py` wherever the value is a tunable default
    (`REQUIRED` = the caller must supply it). `min`/`max` are inclusive, `gt` is
    strict; they are *validity* bounds — what the maths can use — not the ranges of the
    UI sliders. `when=("kind", ("flat",))`: the parameter exists (is recorded and
    checked) only when another parameter has one of those values. `user=False`:
    recorded for reproducibility but fixed by the application, not a setting."""
    name: str
    type: str
    default: Any = REQUIRED
    unit: str = ""
    help: str = ""
    min: Optional[float] = None
    gt: Optional[float] = None
    max: Optional[float] = None
    choices: Tuple[str, ...] = ()
    label: str = ""
    when: Optional[Tuple[str, Tuple[Any, ...]]] = None
    user: bool = True

    def applies(self, params: Mapping[str, Any]) -> bool:
        if self.when is None:
            return True
        other, values = self.when
        return params.get(other) in values

    def check(self, value: Any, op_name: str) -> None:
        """Raise `ValidationError` (a sentence a person can act on) if `value` is not valid."""
        label = self.label or self.name
        unit = f" {self.unit}" if self.unit else ""

        def bad(rule: str):
            raise ValidationError(f"{op_name}: {label} {rule} (got {value!r}).")

        t = self.type
        if t == "choice":
            if value not in self.choices:
                raise ValidationError(f"Unknown {label} '{value}' (expected {_join(self.choices, 'or')}).")
        elif t in ("float", "int"):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                bad("must be a number" if t == "float" else "must be a whole number")
            if not math.isfinite(float(value)):
                bad("must be finite")
            if t == "int" and not (isinstance(value, numbers.Integral) or float(value).is_integer()):
                bad("must be a whole number")
            if self.min is not None and value < self.min:
                bad(f"must be at least {self.min:g}{unit}")
            if self.gt is not None and value <= self.gt:
                bad(f"must be greater than {self.gt:g}{unit}")
            if self.max is not None and value > self.max:
                bad(f"must be at most {self.max:g}{unit}")
        elif t == "polygon":        # None is left to the operation's validator, which says why
            if value is not None and (isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple))):
                bad("must be a list of vertices")
        elif t == "list":
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
                bad("must be a list")
        elif t == "dict":
            if not isinstance(value, Mapping):
                bad("must be a mapping")
        elif t == "text":
            if not isinstance(value, str):
                bad("must be text")
        else:                       # a schema typo must not pass silently
            raise ValueError(f"ParamSpec '{self.name}' has unknown type '{t}'")


def _join(items: Sequence[str], conj: str = "and") -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


# ─────────────────────────────────────────────────────────────────────────
# Data types, layout groups, categories
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataType:
    """A kind of data state (what a downstream step can ask for)."""
    id: str
    label: str            # how the "current processing state" summary names it
    icon: str
    description: str
    summarised: bool = True       # listed in the current-processing-state summary


DATA_TYPES: Tuple[DataType, ...] = (
    DataType("raw", "Loaded survey", "📂",
             "The loaded survey (SEG-Y or synthetic): traces and headers, before any processing.",
             summarised=False),
    DataType("shot", "Shot-domain data", "🔧",
             "Traces grouped by shot (field record) — the domain of pre-stack conditioning."),
    DataType("cdp", "CDP gathers", "🔀",
             "Traces grouped by common-depth-point ensemble, with their offsets — the domain of "
             "velocity analysis, NMO, residual statics, DMO and pre-stack migration."),
    DataType("velocity", "Velocity field", "🎯",
             "A stacking-velocity field V(t, CDP) from a velocity analysis."),
    DataType("nmo", "NMO-corrected gathers", "📐",
             "CDP gathers after moveout correction, with a mask of the samples kept — ready to stack."),
    DataType("dmo", "DMO output", "📐", "CDP gathers after the (simplified) dip-moveout correction."),
    DataType("section", "Stacked / post-stack section", "📚",
             "A stacked section (CDP × time), or a post-stack-processed version of it."),
    DataType("image", "Migrated image", "🗺️", "A migrated section (post-stack or pre-stack time migration)."),
    DataType("final", "Final section", "🏁", "The final-scaled section."),
)
DATA_TYPE_BY_ID: Dict[str, DataType] = {t.id: t for t in DATA_TYPES}

# (id, heading) — the four columns of the Home page.
GROUPS: Tuple[Tuple[str, str], ...] = (
    ("data", "📂 Data & Shot Gathers"),
    ("prestack", "⚙️ Pre-Stack Conditioning"),
    ("velocity", "🔀 CDP Sort → 🎯 Velocity/NMO/DMO"),
    ("stack", "📚 Stack → 🗺️ Migration"),
)

# What an operation scientifically *is* (the layout `group` above is where it is shown).
CATEGORIES: Dict[str, str] = {
    "data": "Data loading & organisation", "filtering": "Filtering", "gain": "Gain & scaling",
    "editing": "Trace & sample editing", "deconvolution": "Deconvolution", "velocity": "Velocity analysis",
    "moveout": "Moveout correction", "statics": "Statics", "regularisation": "Regularisation",
    "stacking": "Stacking", "migration": "Migration",
}


# ─────────────────────────────────────────────────────────────────────────
# The operation
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Operation:
    """Everything declared about one operation. See the module docstring."""
    # ---- identity & presentation
    id: str
    display_name: str
    short_name: str
    description: str
    step_number: str                 # position in the *recommended* workflow
    group: str
    category: str
    output_type: str                 # the kind of state this operation produces
    location: str                    # where in the app it lives
    variant: str = ""
    variant_label: str = ""
    phrase: str = ""                 # how a recommendation sentence names it ("" -> display_name)
    optional: bool = False           # nothing *needs* it, so it can always be skipped
    help: str = ""
    caveats: Tuple[str, ...] = ()
    stage_summary: str = ""          # one line for a whole stage with variants (on its first variant)
    # ---- structure
    is_root: bool = False            # starts a new set of gathers (owns a gather container)
    stream_base: bool = False        # first operation of its stream of data (shot / cdp / section)
    alters_amplitudes: bool = False  # changes each trace's gain (amplitude-based detection must look upstream)
    executable: bool = True          # run through `stage_ops.execute` with a bound compute function
                                     # (False: the survey loader, `stage_ops.run_load_survey`)
    # ---- dependencies
    needs: Tuple[Need, ...] = ()
    advice: Tuple[Advice, ...] = ()
    exclusions: Tuple[Exclusion, ...] = ()
    extra_inputs: Tuple[str, ...] = ()    # input roles recorded beyond the needs (see stage_ops extras)
    # ---- parameters & validation
    params: Tuple[ParamSpec, ...] = ()
    validate: Optional[Callable[[Mapping[str, Any], Any], None]] = None

    # ---- derived identity
    @property
    def key(self) -> str:
        return f"{self.id}:{self.variant}" if self.variant else self.id

    @property
    def full_name(self) -> str:
        return f"{self.display_name} — {self.variant_label}" if self.variant_label else self.display_name

    @property
    def phrase_text(self) -> str:
        return self.phrase or self.display_name

    # ---- derived data types and dependencies
    @property
    def input_types(self) -> Dict[str, Tuple[str, ...]]:
        """role -> the kinds of data that can fill it (from the needs)."""
        return {n.role: tuple(dict.fromkeys(_token_kind(t) for t in n.candidates)) for n in self.needs}

    @property
    def required_inputs(self) -> Tuple[str, ...]:
        return tuple(n.what for n in self.needs)

    @property
    def hard_dependencies(self) -> Tuple[str, ...]:
        """Stage ids a need names explicitly (NMO: CDP sort and pass-1 velocity analysis). An
        operation whose needs are only '@kind' tokens has none — it runs on whichever state of that
        kind is current (band-pass: any shot-domain data)."""
        return tuple(dict.fromkeys(t for n in self.needs for t in n.candidates if not t.startswith("@")))

    @property
    def recommended_predecessors(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(s for a in self.advice for s in a.stages))

    @property
    def recommended_successors(self) -> Tuple[str, ...]:
        """Stages that list this one as a recommended predecessor ("band-pass is recommended before AGC")."""
        return recommended_successors(self.id)

    # ---- parameters
    @property
    def settings(self) -> Tuple[ParamSpec, ...]:
        """The parameters a person can set (the rest are fixed by the application)."""
        return tuple(p for p in self.params if p.user)

    def default_params(self, **overrides: Any) -> Dict[str, Any]:
        """The complete parameter dict this operation records when run with its defaults
        (plus `overrides`), exactly as `stage_ops` records it. Parameters with no default
        (`REQUIRED`) appear only if supplied."""
        out: Dict[str, Any] = {}
        for p in self.params:
            if p.name in overrides:
                value = overrides[p.name]
            elif p.default is REQUIRED:
                continue
            else:
                value = copy.deepcopy(p.default)
            if p.applies({**out, **overrides}):
                out[p.name] = value
        return out

    def validate_params(self, params: Mapping[str, Any], survey: Any = None, *, complete: bool = True) -> None:
        """Check a parameter dict. Raises `ValidationError` with a message written for a person.

        The schema runs first (unknown names, missing names, types, choices, bounds), then
        the operation's own `validate` (cross-parameter and data-dependent rules, e.g. the
        band against the Nyquist frequency). With `complete=False` only the parameters
        present are checked — for a caller vetting its arguments before it has built the whole set."""
        known = {p.name for p in self.params}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValidationError(f"{self.display_name}: unknown parameter(s) {', '.join(unknown)}; "
                                  f"it takes {', '.join(sorted(known)) or 'none'}.")
        for p in self.params:
            if not p.applies(params):
                if p.name in params:
                    raise ValidationError(f"{self.display_name}: '{p.name}' does not apply when "
                                          f"{p.when[0]} = {params.get(p.when[0])!r}.")
                continue
            if p.name not in params:
                if complete and p.default is REQUIRED:
                    raise ValidationError(f"{self.display_name}: parameter '{p.name}' is required.")
                if complete:
                    raise ValidationError(f"{self.display_name}: parameter '{p.name}' is missing "
                                          f"(parameters are recorded in full).")
                continue
            p.check(params[p.name], self.display_name)
        if self.validate is not None:
            self.validate(params, survey)

    # ---- bound implementations (attached by modules/stage_ops.py)
    @property
    def compute(self) -> Optional[Callable]:
        return _BOUND.get(self.key, _Bound()).compute

    @property
    def extras(self) -> Optional[Callable]:
        return _BOUND.get(self.key, _Bound()).extras

    @property
    def qc(self) -> Optional[Callable]:
        return _BOUND.get(self.key, _Bound()).qc


@dataclass
class _Bound:
    compute: Optional[Callable] = None
    extras: Optional[Callable] = None
    qc: Optional[Callable] = None


_BOUND: Dict[str, _Bound] = {}


def bind(key: str, *, compute: Optional[Callable] = None, extras: Optional[Callable] = None,
         qc: Optional[Callable] = None) -> None:
    """Attach an implementation to a declared operation.

    Each piece can be bound to one function. Binding a *different* function to a bound slot is
    an error (two executors for one operation); binding the same function again — what happens
    when `stage_ops` is re-imported by a hot reload — replaces it."""
    if key not in BY_KEY:
        raise KeyError(f"Cannot bind to '{key}': no such operation is declared in the registry.")
    slot = _BOUND.setdefault(key, _Bound())
    for what, fn in (("compute", compute), ("extras", extras), ("qc", qc)):
        if fn is None:
            continue
        old = getattr(slot, what)
        if old is not None and (old.__module__, old.__qualname__) != (fn.__module__, fn.__qualname__):
            raise ValueError(f"'{key}' already has a {what} function bound ({old.__qualname__}).")
        setattr(slot, what, fn)


def _token_kind(token: str) -> str:
    """The data kind a need's candidate token stands for ('@shot' -> shot; 'vel1' -> velocity)."""
    return token[1:] if token.startswith("@") else STAGE_KIND[token]


# ─────────────────────────────────────────────────────────────────────────
# The declarations — in the RECOMMENDED order (that order is advice, never enforced)
# ─────────────────────────────────────────────────────────────────────────

_AT_VEL1 = "Velocity, NMO & DMO → Velocity Pass 1"
_AT_NMO1 = "Velocity, NMO & DMO → NMO Pass 1"
_AT_DMO = "Velocity, NMO & DMO → DMO Correction"

_SHOT = Need("data", ("@shot",), "shot gathers", fix="Build shot gathers (Data Management)",
             why="Pre-stack conditioning runs on shot gathers.")
_CDP_HEAD = Need("data", ("@cdp",), "CDP gathers", fix="Run CDP Sort & Binning")
_VEL_BEST_FALLBACK = ("Pass-2 velocity analysis has not been performed — the pass-1 (approximate) "
                      "velocities will be used.")
_SECTION = Need("data", ("@section",), "a stacked section", fix="Run Stacking")
_STRETCH = ParamSpec("stretch_mute", "float", cfg.NMO_STRETCH_MUTE_DEFAULT, min=0.0,
                     help="Fractional NMO-stretch limit; samples stretched by more are muted and "
                          "excluded from the fold.")


def _pw(default: float) -> ParamSpec:
    """Pre-whitening — the same parameter on four deconvolution operations."""
    return ParamSpec("pre_whitening", "float", default, min=0.0,
                     help="Fraction of the zero-lag autocorrelation added to stabilise the operator.")


def _oplen(default: int) -> ParamSpec:
    return ParamSpec("operator_length", "int", default, "samples", min=1,
                     help="Length of the deconvolution operator.")


# Both passes of the velocity analysis take exactly the same parameters.
_VELOCITY_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("vmin", "float", cfg.VELOCITY_MIN_DEFAULT, "m/s", gt=0.0, help="Lowest velocity scanned."),
    ParamSpec("vmax", "float", cfg.VELOCITY_MAX_DEFAULT, "m/s", gt=0.0, help="Highest velocity scanned."),
    ParamSpec("n_velocities", "int", cfg.VELOCITY_N_DEFAULT, min=2,
              help="Number of trial velocities (fixed by the application).", user=False),
    ParamSpec("n_analysis", "int", cfg.N_ANALYSIS_CDPS_DEFAULT, "CDPs", min=1, help="Number of CDPs analysed."),
    ParamSpec("min_offset", "float", cfg.MIN_ABS_OFFSET_DEFAULT, "m", min=0.0,
              help="Minimum far offset an analysis CDP must have."),
    ParamSpec("window_len", "int", cfg.SEMBLANCE_WINDOW_LEN_DEFAULT, "samples", min=0,
              help="Semblance half-window."),
    ParamSpec("min_time", "float", cfg.AUTOPICK_MIN_TIME_DEFAULT, "s", min=0.0,
              help="Auto-picker ignores earlier times (direct arrival)."),
    ParamSpec("search_w", "float", cfg.AUTOPICK_SEARCH_HALF_WIDTH_MS_DEFAULT, "m/s", gt=0.0,
              help="Half-width of the picker's search window around the current pick."),
    ParamSpec("min_conf", "float", cfg.AUTOPICK_MIN_CONFIDENCE_DEFAULT, min=0.0,
              help="Minimum semblance to accept a move."),
    ParamSpec("smooth", "int", cfg.AUTOPICK_SMOOTH_SIGMA_DEFAULT, "samples", min=0,
              help="Final smoothing of the pick."),
    ParamSpec("reliability_threshold", "float", cfg.RELIABILITY_THRESHOLD_DEFAULT, min=0.0,
              help="Largest relative deviation from the lateral median trend before a pick is flagged "
                   "unreliable."),
    ParamSpec("manual_overrides", "dict", {}, help="Manual picks {CDP: velocity per sample}."),
    ParamSpec("excluded_cdps", "list", [], help="CDPs left out of the analysis."),
)

_LOAD_OPS: Tuple[Operation, ...] = (
    # ── data ────────────────────────────────────────────────────────────
    Operation(
        id="upload", display_name="Load Data", short_name="Load", step_number="1", group="data",
        category="data", output_type="raw", location="Data Management → ① Load data", executable=False,
        description="Read a SEG-Y survey (or generate synthetic test data).",
        help="Loads a SEG-Y file or generates the synthetic test survey. The survey is state S0 of a "
             "session; its size, sample interval and a digest of the trace data are recorded so a "
             "recorded recipe can verify that it is replayed on the same data.",
        params=(ParamSpec("source", "text", REQUIRED, help="Where the data came from.", user=False),
                ParamSpec("traces", "int", REQUIRED, help="Number of traces.", user=False),
                ParamSpec("samples", "int", REQUIRED, help="Samples per trace.", user=False),
                ParamSpec("dt", "float", REQUIRED, "s", help="Sample interval.", user=False),
                ParamSpec("digest", "text", REQUIRED, help="Digest of the trace data.", user=False))),
    Operation(
        id="shot_gathers", display_name="Build Shot Gathers", short_name="Raw", step_number="1",
        group="data", category="data", output_type="shot", optional=True, is_root=True, stream_base=True,
        location="Data Management → ⑥ Build shot gathers",
        description="Group traces by shot (FieldRecord) — the starting point for shot-domain "
                    "pre-stack conditioning.",
        help="Groups the traces by shot (the SEG-Y FieldRecord number) so conditioning can be applied "
             "shot by shot. Optional: the CDP sort can read the raw survey directly, e.g. for data "
             "that were already conditioned.",
        needs=(Need("data", ("upload",), "a loaded survey",
                    fix="Load a SEG-Y file or generate synthetic data (Data Management)"),),
        params=(ParamSpec("n_min_traces", "int", cfg.SHOT_GATHER_MIN_TRACES_DEFAULT, "traces", min=1,
                          help="Shots with fewer traces are dropped."),)),

)

# ── shot-domain conditioning: each only needs shot gathers. These are the "shot operators":
# every operation that transforms the shot-domain stream, which is what the CDP sort's
# "no conditioning applied" advice refers to. ──────────────────────────────────────────
_CONDITIONING: Tuple[Operation, ...] = (
    Operation(
        id="bandpass", display_name="Band-Pass Filter", short_name="Band-pass", step_number="2",
        group="prestack", category="filtering", output_type="shot", optional=True,
        phrase="band-pass filtering", location="Pre-Stack Conditioning → Band-Pass",
        description="Trapezoidal frequency-domain band-pass (shot domain).",
        help="Zero-phase trapezoidal band-pass: a linear taper below the low-cut and above the "
             "high-cut, smoothed with a Hann kernel to avoid Gibbs ringing, applied to every trace by "
             "convolution in the time domain.",
        needs=(_SHOT,), validate=V.check_bandpass_params,
        params=(ParamSpec("fmin", "float", cfg.FMIN_DEFAULT, "Hz", min=0.0, help="Low-cut frequency."),
                ParamSpec("fmax", "float", cfg.FMAX_DEFAULT, "Hz", gt=0.0,
                          help="High-cut frequency; must be below the data's Nyquist frequency."),
                ParamSpec("low_slope", "float", cfg.LOW_SLOPE_DEFAULT, "Hz", min=0.0,
                          help="Width of the taper below the low-cut."),
                ParamSpec("high_slope", "float", cfg.HIGH_SLOPE_DEFAULT, "Hz", min=0.0,
                          help="Width of the taper above the high-cut."))),
    Operation(
        id="agc", display_name="Amplitude Recovery (AGC)", short_name="AGC", step_number="3",
        group="prestack", category="gain", output_type="shot", optional=True, alters_amplitudes=True,
        phrase="amplitude recovery (AGC)", location="Pre-Stack Conditioning → Amplitude Recovery",
        description="Sliding-window automatic gain control (shot domain).",
        help="Sliding-window automatic gain control: every sample is divided by a running RMS, mean or "
             "median of the trace amplitude. It is self-normalising, so it equalises amplitudes along "
             "each trace and between traces.",
        caveats=("Not amplitude-preserving: after AGC every trace sits at a similar level, so "
                 "energy-based trace-kill detection measures the data from before the AGC.",),
        needs=(_SHOT,),
        advice=(Advice(("bandpass",), "AGC amplifies everything in the trace, including out-of-band noise, "
                                      "so the gain is driven by the useful reflection band only if the "
                                      "data were filtered first."),),
        params=(ParamSpec("window", "int", cfg.AGC_WINDOW_DEFAULT, "samples", min=1,
                          help="Length of the sliding gain window."),
                ParamSpec("mode", "choice", cfg.AGC_MODE_DEFAULT, choices=("rms", "mean", "median"),
                          label="AGC mode", help="Statistic of the amplitude in the window."))),
    Operation(
        id="despike", display_name="De-spike", short_name="De-spike", step_number="4a", group="prestack",
        category="editing", output_type="shot", optional=True, phrase="de-spiking",
        location="Pre-Stack Conditioning → De-spike",
        description="Remove burst-noise spikes sample-by-sample (shot domain).",
        help="Flags samples whose amplitude exceeds a threshold times the local robust noise level and "
             "that are also isolated in time, and replaces them with the local median. The isolation "
             "test keeps genuine sharp reflection wavelets from being mistaken for spikes.",
        needs=(_SHOT,),
        params=(ParamSpec("window", "int", cfg.DESPIKE_WINDOW_DEFAULT, "samples", min=1,
                          help="Sliding-median window."),
                ParamSpec("threshold", "float", cfg.DESPIKE_THRESHOLD_DEFAULT, "× noise level", gt=0.0,
                          help="Amplitude, as a multiple of the local robust noise level, above which an "
                               "isolated sample is a spike."))),
    Operation(
        id="trace_edit", display_name="Trace Edit / Kill", short_name="Trace kill", step_number="4b",
        group="prestack", category="editing", output_type="shot", optional=True, phrase="trace edit/kill",
        location="Pre-Stack Conditioning → Trace Edit/Kill", extra_inputs=("detect",),
        description="Zero dead/abnormally noisy traces (shot domain).",
        help="Zeroes (rather than deletes, which would disturb offset and CDP bookkeeping) dead traces — "
             "energy far below the gather median — and abnormally noisy ones — energy far above it. The "
             "kill mask is measured on amplitude-preserving data (the state just before the first "
             "AGC or trace balancing in the data's history) and applied to the data being edited; the "
             "state it was measured on is recorded as the extra input 'detect'.",
        caveats=("Detection is energy-based and blind after a per-trace gain; the application measures "
                 "upstream of AGC / balancing automatically.",),
        needs=(_SHOT,),
        params=(ParamSpec("dead_frac", "float", cfg.TRACE_KILL_DEAD_FRAC_DEFAULT, "× median", min=0.0,
                          help="A trace whose energy is below this fraction of the median is dead."),
                ParamSpec("noisy_mult", "float", cfg.TRACE_KILL_NOISY_MULT_DEFAULT, "× median", min=0.0,
                          help="A trace whose energy exceeds this multiple of the median is noisy."))),
    Operation(
        id="mute", display_name="Mute", short_name="Mute", step_number="5", group="prestack",
        category="editing", output_type="shot", optional=True, phrase="muting",
        location="Pre-Stack Conditioning → Mute",
        description="Top mute (flat or offset-dependent) and surgical mute (shot domain).",
        help="Three kinds. Flat: a cosine-tapered top mute, the same time on every trace. Linear: an "
             "offset-dependent (linear-moveout) mute that zeroes each trace before t0 + |offset| / "
             "velocity, following a direct arrival or ground-roll cone. Surgical: zeroes a rectangular "
             "offset–time box inside the gather.",
        needs=(_SHOT,),
        params=(ParamSpec("kind", "choice", "flat", choices=("flat", "linear", "surgical"),
                          label="mute type", help="Which mute to apply."),
                ParamSpec("t_top", "float", cfg.MUTE_TTOP_DEFAULT, "s", min=0.0, when=("kind", ("flat",)),
                          help="Time above which every trace is muted."),
                ParamSpec("velocity", "float", cfg.MUTE_LINEAR_VELOCITY_DEFAULT, "m/s", gt=0.0,
                          when=("kind", ("linear",)), help="Apparent velocity of the muted event."),
                ParamSpec("t0", "float", 0.0, "s", when=("kind", ("linear",)),
                          help="Intercept time of the mute line."),
                ParamSpec("x_min", "float", 0.0, "m", when=("kind", ("surgical",)),
                          help="Smallest |offset| of the muted box."),
                ParamSpec("x_max", "float", 0.0, "m", when=("kind", ("surgical",)),
                          help="Largest |offset| of the muted box."),
                ParamSpec("t_min", "float", 0.0, "s", when=("kind", ("surgical",)),
                          help="Start time of the muted box."),
                ParamSpec("t_max", "float", 0.0, "s", when=("kind", ("surgical",)),
                          help="End time of the muted box."),
                ParamSpec("taper", "float", cfg.MUTE_TAPER_DEFAULT, "s", min=0.0,
                          help="Cosine taper at the edge of the mute."))),
    Operation(
        id="balance", display_name="Trace Balancing", short_name="Balance", step_number="6",
        group="prestack", category="gain", output_type="shot", optional=True, alters_amplitudes=True,
        phrase="trace balancing", location="Pre-Stack Conditioning → Trace Balancing",
        description="Scalar RMS normalization per trace (shot domain).",
        help="Divides each trace by a single RMS scalar measured in a reference window from t = 0. A "
             "per-trace scalar, distinct from the sliding-window AGC: relative amplitudes within a "
             "trace are preserved.",
        caveats=("Removes between-trace amplitude differences, so energy-based trace-kill detection "
                 "measures the data from before it.",),
        needs=(_SHOT,),
        params=(ParamSpec("ref_s", "float", cfg.NORM_WINDOW_S_DEFAULT, "s", gt=0.0,
                          help="Length of the reference window, from t = 0."),)),
    Operation(
        id="fk", display_name="F-K Dip Filter", short_name="F-K", step_number="7", group="prestack",
        category="filtering", output_type="shot", optional=True, phrase="F-K filtering",
        location="Pre-Stack Conditioning → F-K Filter", validate=V.check_fk_polygon_params,
        description="2-D FFT polygon mask — ground-roll attenuation (shot domain).",
        help="Transforms each shot gather to the F-K domain, multiplies it by a polygon mask and "
             "transforms back. Accept mode: the polygon encloses the signal and everything outside it is "
             "attenuated. Reject mode: the polygon encloses the noise. Amplitudes on the wrong side are "
             "scaled by a small coefficient rather than an exact zero.",
        caveats=("Accept and reject polygons are not interchangeable: accept needs the signal region, "
                 "reject needs the noise region.",),
        needs=(_SHOT,),
        params=(ParamSpec("polygon", "polygon", REQUIRED,
                          help="Vertices of the mask in the F-K plane (at least 3)."),
                ParamSpec("mode", "choice", cfg.FK_POLYGON_MODE_DEFAULT, choices=("accept", "reject"),
                          label="polygon mode", help="Whether the polygon encloses the signal or the noise."),
                ParamSpec("coeff", "float", cfg.FK_REJECT_COEFFICIENT_DEFAULT, min=0.0,
                          help="Multiplier on the attenuated side. A small non-zero value is recommended "
                               "(a hard zero rings); the maths clips it to 0–1."))),
    Operation(
        id="mult_supp", display_name="Multiple Suppression", short_name="Multiple suppression",
        step_number="8", group="prestack", category="deconvolution", output_type="shot", optional=True,
        phrase="multiple suppression", location="Pre-Stack Conditioning → Multiple Suppression",
        description="Predictive (gap) deconvolution — attenuates short-period reverberations/multiples.",
        help="Predictive (gap) deconvolution applied trace by trace: it predicts and removes the "
             "periodic energy beyond the prediction lag.",
        needs=(_SHOT,),
        advice=(Advice(("bandpass", "agc"), "Predictive deconvolution assumes a stationary, noise-limited "
                                            "trace; band-pass filtering and amplitude recovery are "
                                            "normally applied first."),),
        params=(_oplen(cfg.MULT_SUPP_OPLEN_DEFAULT),
                ParamSpec("prediction_lag", "int", cfg.MULT_SUPP_LAG_DEFAULT, "samples", min=1,
                          help="Gap: the lag over which the trace is predicted."),
                _pw(cfg.DECON_PW_DEFAULT))),
    Operation(
        id="decon", display_name="Deconvolution", short_name="Decon", step_number="9", group="prestack",
        category="deconvolution", output_type="shot", optional=True, phrase="deconvolution",
        location="Pre-Stack Conditioning → Deconvolution",
        description="Spiking deconvolution per trace (shot domain).",
        help="Spiking (Wiener–Hopf) deconvolution applied trace by trace, with pre-whitening for "
             "numerical stability: it compresses the wavelet towards a spike to sharpen resolution.",
        caveats=("Assumes a minimum-phase wavelet; on zero-phase data (such as the synthetic Ricker "
                 "wavelet) a per-trace timing/phase shift is expected.",),
        needs=(_SHOT,),
        advice=(Advice(("bandpass", "agc"), "Spiking deconvolution assumes a stationary, noise-limited "
                                            "trace: a wide band-pass and amplitude recovery are normally "
                                            "applied first (4 Deconvolution, 'Preconditioning')."),
                Advice(("mute", "fk"), "Coherent noise (ground roll, direct arrival) is normally removed "
                                       "before deconvolution so it is not treated as part of the wavelet.",
                       mode="any")),
        params=(_oplen(cfg.DECON_OPLEN_DEFAULT), _pw(cfg.DECON_PW_DEFAULT))),

)

_SHOT_OPERATORS: Tuple[str, ...] = tuple(o.id for o in _CONDITIONING)

OPERATIONS: Tuple[Operation, ...] = _LOAD_OPS + _CONDITIONING + (
    # ── CDP sort: only a survey is genuinely required ─────────────────────
    Operation(
        id="cdp_sort", display_name="CDP Sort / Binning", short_name="CDP sort", step_number="10",
        group="velocity", category="data", output_type="cdp", is_root=True, stream_base=True,
        phrase="CDP sort", location="CDP Sort & Binning",
        description="Re-group the traces (conditioned, if any conditioning was applied) by CDP "
                    "ensemble number.",
        help="Regroups the traces — the conditioned ones, if any shot-domain operation was applied — "
             "into common-depth-point ensembles, dropping CDPs below the minimum fold. It reads the "
             "current shot-domain data, or the raw survey if no shot gathers were built (sorting only "
             "needs the headers).",
        needs=(Need("data", ("@shot", "upload"), "traces to sort (a loaded survey)",
                    fix="Load data (Data Management)", track=True),),
        advice=(Advice(_SHOT_OPERATORS, "", mode="any",
                       message="No pre-stack conditioning has been applied, so the CDP gathers will be "
                               "built from the raw, unconditioned traces. (Fine for data that were "
                               "already processed.)"),),
        params=(ParamSpec("n_min_traces", "int", cfg.N_MIN_TRACES_DEFAULT, "traces", min=1,
                          help="Minimum fold: CDPs with fewer traces are dropped."),)),

    # ── velocity / NMO / statics / DMO / regularisation ──────────────────
    Operation(
        id="vel1", display_name="Velocity Analysis — Pass 1", short_name="Velocity 1", step_number="11",
        group="velocity", category="velocity", output_type="velocity",
        location=_AT_VEL1, validate=V.check_velocity_scan_params,
        description="Semblance velocity analysis (approximate); auto-pick, manual click-pick, or "
                    "Constant-Velocity Stack.",
        help="Semblance velocity analysis on CDP gathers chosen evenly along the line (skipping those "
             "whose far offset is too short), auto-picked by a confidence-gated local walk along the "
             "semblance ridge and interpolated laterally into a V(t, CDP) field. Manual picks and "
             "excluded CDPs are parameters of the state. Pass 1 is approximate.",
        needs=(Need("data", ("cdp_sort",), "CDP gathers", fix="Run CDP Sort & Binning",
                    why="Velocity analysis scans offset-sorted CDP gathers.", track=True),),
        params=_VELOCITY_PARAMS),
    Operation(
        id="nmo1", display_name="NMO Correction — Pass 1", short_name="NMO 1", step_number="12",
        group="velocity", category="moveout", output_type="nmo", location=_AT_NMO1,
        description="Apply pass-1 stacking velocities to the CDP-sorted gathers.",
        help="Forward NMO: flattens moveout to zero-offset time t0 with the pass-1 velocities. Samples "
             "whose NMO stretch exceeds the stretch-mute limit are muted and excluded from the fold.",
        needs=(Need("data", ("cdp_sort",), "CDP gathers", fix="Run CDP Sort & Binning", track=True),
               Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=_AT_VEL1,
                    why="NMO correction is defined by a velocity function.", track=True)),
        params=(_STRETCH,)),
    Operation(
        id="statics", display_name="Residual Statics", short_name="Statics", step_number="13",
        group="velocity", category="statics", output_type="cdp", optional=True,
        phrase="the residual-statics correction", location="Velocity, NMO & DMO → Residual Statics",
        description="Surface-consistent shot/receiver statics.",
        help="Estimates surface-consistent shot and receiver residual shifts by cross-correlating "
             "NMO-flattened traces with a pilot trace, then applies them to the pre-NMO gathers.",
        caveats=("Surface-consistent: refused on regularised gathers, whose traces no longer have a "
                 "real shot and receiver.",),
        needs=(_CDP_HEAD,
               Need("nmo", ("nmo1",), "NMO-flattened gathers (NMO Pass 1)", fix=_AT_NMO1,
                    why="Residual statics are estimated by cross-correlating NMO-flattened traces with "
                        "a pilot trace.", track=True)),
        exclusions=(Exclusion(
            "data", ("data_reg",),
            "Residual statics are surface-consistent: they need each trace's own shot and receiver. "
            "Regularised gathers hold interpolated traces on a synthetic offset grid, so that identity "
            "is gone.",
            fix="Apply residual statics before Data Regularisation (or choose gathers that were not "
                "regularised as the input)."),),
        params=(ParamSpec("max_shift_s", "float", cfg.MAX_STATIC_SHIFT_S_DEFAULT, "s", gt=0.0,
                          help="Largest shift searched, in either direction."),)),
    Operation(
        id="dmo", display_name="DMO Correction", short_name="DMO", step_number="14", group="velocity",
        category="moveout", output_type="dmo", optional=True, phrase="DMO", location=_AT_DMO,
        description="Simplified dip-moveout time-stretch.",
        help="Simplified dip-moveout: NMO-corrects its own input with the pass-1 velocities, then "
             "applies a time-stretch approximation of DMO. It needs pre-NMO gathers and velocities — "
             "not the NMO Pass 1 output or residual statics.",
        caveats=("A time-stretch stand-in, not a full common-offset DMO operator: it does not "
                 "redistribute energy spatially across CDPs.",),
        needs=(_CDP_HEAD,
               Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=_AT_VEL1,
                    why="DMO acts on NMO-corrected data, so it needs the velocities that flatten the "
                        "gathers first.", track=True)),
        advice=(Advice(("statics",), "Residual statics are normally applied before DMO so the DMO "
                                     "stretch does not smear uncorrected shifts."),),
        params=(_STRETCH,)),
    Operation(
        id="inv_nmo", display_name="Inverse NMO", short_name="Inverse NMO", step_number="15",
        group="velocity", category="moveout", output_type="cdp", optional=True, phrase="inverse NMO",
        location="Velocity, NMO & DMO → Inverse NMO",
        description="Undo pass-1 NMO on the DMO-corrected data.",
        help="Re-introduces the moveout that a forward NMO with the same velocities removes (the exact "
             "inverse of the forward correction), returning the DMO output to the pre-NMO domain for a "
             "second velocity analysis.",
        needs=(Need("data", ("dmo",), "DMO-corrected gathers", fix=_AT_DMO,
                    why="Inverse NMO undoes pass-1 NMO on the DMO output.", track=True),
               Need("velocity", ("vel1",), "pass-1 stacking velocities", fix=_AT_VEL1, track=True))),
    Operation(
        id="data_reg", display_name="Data Regularisation", short_name="Regularised", step_number="15b",
        group="velocity", category="regularisation", output_type="cdp", optional=True,
        phrase="data regularisation", location="Velocity, NMO & DMO → Data Regularisation",
        description="Interpolate each CDP gather onto a common, regularly spaced offset grid — "
                    "recommended before velocity analysis, stacking, and migration.",
        help="Interpolates every CDP gather onto one shared, regularly spaced offset axis and records "
             "which output samples are real data. Afterwards the traces no longer carry a real shot and "
             "receiver, so surface-consistent residual statics cannot follow.",
        needs=(_CDP_HEAD,),
        params=(ParamSpec("n_offsets", "int", cfg.DATA_REG_N_OFFSETS_DEFAULT, "offsets", min=2,
                          help="Number of offsets on the regular axis."),)),
    Operation(
        id="vel2", display_name="RMS Velocity Analysis (Pass 2)", short_name="Velocity 2",
        step_number="16", group="velocity", category="velocity", output_type="velocity", optional=True,
        phrase="pass-2 velocity analysis", location="Velocity, NMO & DMO → RMS Velocity Analysis (Pass 2)",
        validate=V.check_velocity_scan_params,
        description="Refined semblance velocity analysis on the refined (statics/DMO/regularised) gathers.",
        help="The same semblance analysis as pass 1, run on the current pre-NMO CDP gathers after "
             "refinement (residual statics, DMO, regularisation). Its velocities are preferred over "
             "pass 1 wherever a later step needs a stacking velocity.",
        needs=(_CDP_HEAD,),
        advice=(Advice(("statics", "inv_nmo", "data_reg"), "", mode="any",
                       message="No refinement (residual statics, DMO or regularisation) has been applied "
                               "since pass 1, so pass-2 velocity analysis will re-analyse essentially the "
                               "same data."),),
        params=_VELOCITY_PARAMS),
    Operation(
        id="nmo_final", display_name="Final NMO", short_name="Final NMO", step_number="17",
        group="velocity", category="moveout", output_type="nmo", phrase="final NMO",
        location="Velocity, NMO & DMO → Final NMO",
        description="Apply the best available stacking velocities (pass 2, else pass 1) to the current "
                    "pre-NMO gathers.",
        help="NMO of the current pre-NMO gathers with the best available velocities (pass 2, else "
             "pass 1). Where the data were regularised, the live-sample mask is combined with the "
             "regularisation mask so interpolated-only offsets do not inflate the stack fold.",
        needs=(Need("data", ("@cdp",), "CDP gathers", fix="Run CDP Sort & Binning", track=True),
               Need("velocity", ("vel2", "vel1"), "stacking velocities", fix=_AT_VEL1,
                    why="NMO correction is defined by a velocity function.", fallback=_VEL_BEST_FALLBACK,
                    track=True)),
        params=(_STRETCH,)),

    # ── stack and post-stack ───────────────────────────────────────────────
    Operation(
        id="stack", display_name="Stack", short_name="Stack", step_number="18", group="stack",
        category="stacking", output_type="section", stream_base=True, location="Stacking",
        description="Fold-normalized CDP stack of NMO-corrected gathers.",
        help="Fold-normalised stack of NMO-corrected gathers (live samples only); a naive mean is kept "
             "alongside for comparison.",
        needs=(Need("data", ("nmo_final", "nmo1"), "NMO-corrected CDP gathers",
                    fix="Apply NMO (Velocity, NMO & DMO → NMO Pass 1 or Final NMO)",
                    why="Stacking sums traces at each CDP; without moveout correction reflections stack "
                        "out of phase.",
                    fallback="Final NMO has not been applied — the pass-1 NMO output will be stacked "
                             "(approximate moveout correction, and none of the later refinements).",
                    track=True),)),
    Operation(
        id="rna", display_name="Random Noise Attenuation", short_name="RNA", step_number="19",
        group="stack", category="filtering", output_type="section", optional=True,
        phrase="random noise attenuation", location="Post-Stack Processing → Random Noise Attenuation",
        description="Spatial median filter across CDPs.",
        help="A running median across CDPs: coherent reflections correlate from trace to trace and "
             "survive, random noise does not.",
        caveats=("A documented simplification: a spatial median stands in for F-X deconvolution, which "
                 "needs higher-fold, regularly sampled data.",),
        needs=(_SECTION,),
        params=(ParamSpec("spatial_window", "int", cfg.RNA_SPATIAL_WINDOW_DEFAULT, "traces", min=1,
                          help="Length of the median filter across CDPs."),)),
    Operation(
        id="decon_post", display_name="Deconvolution (post-stack)", short_name="Post-stack decon",
        step_number="20", group="stack", category="deconvolution", output_type="section", optional=True,
        phrase="post-stack deconvolution", location="Post-Stack Processing → Deconvolution",
        description="A milder second spiking-deconvolution pass.",
        help="A second, milder spiking deconvolution applied to the stacked section.",
        needs=(_SECTION,),
        advice=(Advice(("rna",), "Noise attenuation is normally applied first so deconvolution does not "
                                 "sharpen random noise."),),
        params=(_oplen(cfg.DECON_AFTER_OPLEN_DEFAULT), _pw(cfg.DECON_AFTER_PW_DEFAULT))),
    Operation(
        id="tvf", display_name="Time-Variant Filter", short_name="TVF", step_number="21", group="stack",
        category="filtering", output_type="section", optional=True, phrase="time-variant filtering",
        location="Post-Stack Processing → Time-Variant Filter",
        description="Depth-zoned band-pass blending.",
        help="Applies a different band-pass in each time zone and blends the results with a "
             "raised-cosine taper across the zone boundaries: higher frequencies attenuate faster with "
             "depth, so the pass-band is wide at shallow times and narrower with depth.",
        needs=(_SECTION,),
        advice=(Advice(("decon_post",), "The time-variant filter normally follows deconvolution, "
                                        "removing the high-frequency noise it boosts."),),
        params=(ParamSpec("zones", "list", cfg.TVF_ZONES_DEFAULT,
                          help="(t_start, t_end, fmin, fmax) per zone (fixed by the application).",
                          user=False),
                ParamSpec("taper_s", "float", cfg.TVF_TAPER_S_DEFAULT, "s", min=0.0,
                          help="Blend length across zone boundaries (fixed by the application).",
                          user=False))),

    # ── migration (two implementations, two sets of inputs) ──────────────
    Operation(
        id="migration", variant="poststack", variant_label="post-stack", display_name="Migration",
        short_name="Migration", step_number="22", group="stack", category="migration",
        output_type="image", optional=True, phrase="migration", location="Migration",
        description="Post-stack Kirchhoff time migration of a stacked section.",
        stage_summary="Post-stack Kirchhoff time migration, or pre-stack Kirchhoff time migration (PSTM) "
                      "on CDP gathers before stack.",
        help="Kirchhoff time migration of the stacked section: each output sample sums the diffraction "
             "contributions of the neighbouring traces within the aperture along "
             "t = sqrt(t0² + (2·dx/v)²), with an obliquity weight. The migration velocity is the "
             "stacking-velocity field (pass 2, else pass 1).",
        needs=(Need("data", ("@section",), "a stacked section", fix="Run Stacking",
                    why="Post-stack migration operates on a stacked (≈ zero-offset) section.", track=True),
               Need("velocity", ("vel2", "vel1"), "a stacking-velocity field (the migration velocity)",
                    fix=_AT_VEL1, fallback=_VEL_BEST_FALLBACK.replace(
                        "will be used", "will be used as the migration velocity"), track=True)),
        advice=(Advice(("rna", "decon_post", "tvf"), "Post-stack conditioning is normally applied before "
                       "migration so noise is not migrated into the image."),),
        params=(ParamSpec("aperture_m", "float", cfg.APERTURE_M_DEFAULT, "m", gt=0.0,
                          help="Half-aperture of the migration summation."),)),
    Operation(
        id="migration", variant="pstm", variant_label="pre-stack (PSTM)", display_name="Migration",
        short_name="Migration", step_number="22", group="stack", category="migration",
        output_type="image", optional=True, phrase="migration", location="Migration",
        description="Pre-stack Kirchhoff time migration (PSTM) on CDP gathers, before stack.",
        help="Pre-stack Kirchhoff time migration directly from offset-preserving CDP gathers, using the "
             "double-square-root traveltime for a diffractor below each output point. Migration and "
             "stack happen in one pass, so no stack is required; regularisation is recommended, not "
             "required.",
        caveats=("Needs pre-stack, offset-preserving gathers: stacked or NMO-corrected data cannot be "
                 "used.",),
        needs=(Need("data", ("@cdp",), "pre-stack CDP gathers with their offsets preserved",
                    fix="Run CDP Sort & Binning", track=True,
                    why="PSTM sums pre-stack traces using their true offsets, so it cannot use stacked or "
                        "NMO-corrected data."),
               Need("velocity", ("vel2", "vel1"), "a stacking-velocity field (the migration velocity)",
                    fix=_AT_VEL1, fallback=_VEL_BEST_FALLBACK.replace(
                        "will be used", "will be used as the migration velocity"), track=True)),
        advice=(Advice(("data_reg",), "Regularised offsets give the migration a consistent offset "
                                      "sampling at every CDP."),),
        params=(ParamSpec("aperture_m", "float", cfg.APERTURE_M_DEFAULT, "m", gt=0.0,
                          help="Half-aperture of the migration summation."),
                ParamSpec("cdp_stride", "int", cfg.PSTM_CDP_STRIDE_DEFAULT, "CDPs", min=1,
                          help="Migrate every n-th output CDP (1 = every CDP, slow)."))),
    Operation(
        id="final_scale", display_name="Final Scaling", short_name="Final scaling", step_number="23",
        group="stack", category="gain", output_type="final", optional=True,
        location="Migration → Final scaling",
        description="Display AGC + re-applied top mute.",
        help="A light AGC pass that restores a balanced amplitude range after migration's obliquity "
             "weighting, followed by the top mute again: migration can pull energy back into the muted "
             "shallow zone.",
        needs=(Need("data", ("migration", "@section"), "a section to scale", fix="Run Stacking",
                    fallback="Migration has not been applied — final scaling will be applied to the "
                             "un-migrated section.", track=True),),
        params=(ParamSpec("window", "int", cfg.FINAL_AGC_WINDOW_DEFAULT, "samples", min=1,
                          help="Length of the final AGC window."),
                ParamSpec("t_top_mute", "float", cfg.MUTE_TTOP_DEFAULT, "s", min=0.0,
                          help="Top-mute time re-applied after the AGC (fixed by the application).",
                          user=False))),
)

# ─────────────────────────────────────────────────────────────────────────
# Indexes and queries (all derived from OPERATIONS)
# ─────────────────────────────────────────────────────────────────────────

BY_KEY: Dict[str, Operation] = {o.key: o for o in OPERATIONS}

# One entry per *stage* (migration's two variants share one): the first declared operation.
_STAGE_FIRST: Dict[str, Operation] = {}
for _o in OPERATIONS:
    _STAGE_FIRST.setdefault(_o.id, _o)
del _o

STAGE_INDEX: Dict[str, int] = {sid: i for i, sid in enumerate(_STAGE_FIRST)}
STAGE_KIND: Dict[str, str] = {sid: o.output_type for sid, o in _STAGE_FIRST.items()}
ROOT_STAGES: Tuple[str, ...] = tuple(sid for sid, o in _STAGE_FIRST.items() if o.is_root)
KINDS: Tuple[str, ...] = tuple(t.id for t in DATA_TYPES)
STREAM_BASE: Dict[str, str] = {o.output_type: o.id for o in OPERATIONS if o.stream_base}
STREAM_MEMBERS: Dict[str, Tuple[str, ...]] = {
    kind: tuple(sid for sid, o in _STAGE_FIRST.items() if o.output_type == kind and sid != base)
    for kind, base in STREAM_BASE.items()}
GAIN_ALTERING_STAGES: Tuple[str, ...] = tuple(sid for sid, o in _STAGE_FIRST.items() if o.alters_amplitudes)


def get(stage_id: str, variant: Optional[str] = None) -> Operation:
    """The operation `stage_id` (and `variant`, for migration). Raises `ValueError` for a stage that
    has variants but was given none, `KeyError` for an unknown one."""
    key = f"{stage_id}:{variant}" if variant else stage_id
    if key not in BY_KEY:
        if stage_id == "migration":
            raise ValueError("Migration needs a variant: 'poststack' or 'pstm'.")
        raise KeyError(f"No operation '{key}' is declared in the registry")
    return BY_KEY[key]


def find(key: str) -> Optional[Operation]:
    return BY_KEY.get(key)


def operations(*, group: Optional[str] = None, category: Optional[str] = None,
               output_type: Optional[str] = None) -> List[Operation]:
    return [o for o in OPERATIONS
            if (group is None or o.group == group) and (category is None or o.category == category)
            and (output_type is None or o.output_type == output_type)]


def stages() -> List[Operation]:
    """The first operation of every stage, in the recommended order (migration once)."""
    return list(_STAGE_FIRST.values())


def stage_ids() -> List[str]:
    return list(_STAGE_FIRST)


def variants(stage_id: str) -> List[Operation]:
    return [o for o in OPERATIONS if o.id == stage_id]


def stage_description(stage_id: str) -> str:
    """One line for the whole stage: the `stage_summary` its first variant declares, else the
    variants' own lines joined."""
    ops = variants(stage_id)
    summary = next((o.stage_summary for o in ops if o.stage_summary), "")
    return summary or " ".join(dict.fromkeys(o.description for o in ops))


def stage_label(stage_id: str) -> str:
    o = _STAGE_FIRST.get(stage_id)
    return o.display_name if o else stage_id


def short_name(stage_id: str) -> str:
    o = _STAGE_FIRST.get(stage_id)
    return o.short_name if o else stage_id


def phrase(stage_id: str) -> str:
    o = _STAGE_FIRST.get(stage_id)
    return o.phrase_text if o else stage_id


def location(stage_id: str) -> str:
    o = _STAGE_FIRST.get(stage_id)
    return o.location if o else stage_id


def processed_kinds() -> Tuple[str, ...]:
    """Every kind of data except the loaded survey."""
    return tuple(k for k in KINDS if k != "raw")


def recommended_successors(stage_id: str) -> Tuple[str, ...]:
    """Stage ids that recommend `stage_id` as a predecessor — what it is "recommended before"."""
    return tuple(dict.fromkeys(o.id for o in OPERATIONS if any(stage_id in a.stages for a in o.advice)))


def advice_pairs() -> List[Tuple[str, str]]:
    """(earlier, later) stage pairs implied by every `Advice`."""
    return [(earlier, o.id) for o in OPERATIONS for a in o.advice for earlier in a.stages]


@dataclass(frozen=True)
class WarningInfo:
    kind: str                       # missing_recommended | fallback_input | order_inversion | caveat
    text: str
    stages: Tuple[str, ...] = ()


def possible_warnings(key: str) -> List[WarningInfo]:
    """Every warning an operation can raise, as declared: the advice (a recommended predecessor
    missing), the fallback inputs, running it before something that is normally applied after it
    (order inversion), and its static caveats. Generic warnings that apply to every operation
    (the same operation applied twice; an input that is out of date) are not repeated here."""
    op = BY_KEY[key]
    out: List[WarningInfo] = []
    for a in op.advice:
        out.append(WarningInfo("missing_recommended", a.message or a.reason, tuple(a.stages)))
    for n in op.needs:
        if n.fallback:
            out.append(WarningInfo("fallback_input", n.fallback, tuple(n.candidates)))
    later = recommended_successors(op.id)
    if later:
        out.append(WarningInfo(
            "order_inversion",
            f"{op.display_name} is normally applied before {_join([phrase(s) for s in later])}; running it "
            f"on data that already went through them applies it out of order.", later))
    out.extend(WarningInfo("caveat", c) for c in op.caveats)
    return out


def workflow_graph() -> Dict[str, Any]:
    """The operations and their relations as plain data (JSON-serialisable), for drawing the
    workflow. Nodes are operations and kinds of data. Edges:
      produces    operation -> data kind it creates
      consumes    data kind -> operation that reads the current state of that kind (a `Need`)
      requires    operation -> operation a `Need` names explicitly (the latest state it made)
      recommends  earlier operation -> later one (`Advice`; never blocks)
      excludes    operation -> operation that cannot read data that went through it (`Exclusion`)
    Edges that come from one declaration carry its role / preference / reason."""
    nodes: List[Dict[str, Any]] = [
        {"id": o.key, "type": "operation", "label": o.full_name, "category": o.category, "group": o.group,
         "step": o.step_number, "optional": o.optional, "output": o.output_type} for o in OPERATIONS]
    nodes += [{"id": f"data:{t.id}", "type": "data", "label": t.label or t.id, "icon": t.icon}
              for t in DATA_TYPES]
    edges: List[Dict[str, Any]] = []

    def keys_of(stage_id: str) -> List[str]:
        return [o.key for o in variants(stage_id)]

    for o in OPERATIONS:
        edges.append({"from": o.key, "to": f"data:{o.output_type}", "type": "produces"})
        for n in o.needs:
            for i, tok in enumerate(n.candidates):
                base = {"to": o.key, "role": n.role, "preferred": i == 0, "tracked": n.track}
                if tok.startswith("@"):
                    edges.append({"from": f"data:{tok[1:]}", "type": "consumes", **base})
                else:
                    edges.extend({"from": k, "type": "requires", **base} for k in keys_of(tok))
        for a in o.advice:
            for s in a.stages:
                edges.extend({"from": k, "to": o.key, "type": "recommends", "mode": a.mode,
                              "reason": a.message or a.reason} for k in keys_of(s))
        for x in o.exclusions:
            for s in x.stages:
                edges.extend({"from": k, "to": o.key, "type": "excludes", "role": x.role} for k in keys_of(s))
    return {"recommended_order": [o.key for o in OPERATIONS], "nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────────────────
# Consistency
# ─────────────────────────────────────────────────────────────────────────

def problems(require_implementations: bool = False) -> List[str]:
    """Everything wrong with the declarations (an empty list = consistent). `require_implementations`
    also demands that `stage_ops` has bound a compute function to every operation (and an `extras`
    function to every operation that declares `extra_inputs`)."""
    out: List[str] = []
    keys = [o.key for o in OPERATIONS]
    out += [f"duplicate operation key '{k}'" for k in sorted({k for k in keys if keys.count(k) > 1})]
    stage_set = set(_STAGE_FIRST)
    kind_set = set(KINDS)
    group_ids = {g for g, _ in GROUPS}

    for sid in _STAGE_FIRST:                    # the variants of a stage are declared together
        idx = [i for i, o in enumerate(OPERATIONS) if o.id == sid]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            out.append(f"the variants of '{sid}' are not declared next to each other")
        for f in ("step_number", "group", "optional", "display_name", "category"):
            if len({getattr(o, f) for o in variants(sid)}) > 1:
                out.append(f"the variants of '{sid}' disagree on {f}")
        if len(variants(sid)) > 1 and any(not o.variant for o in variants(sid)):
            out.append(f"'{sid}' has variants, but one is declared without a variant name")

    for o in OPERATIONS:
        k = o.key
        if o.group not in group_ids:
            out.append(f"{k}: unknown group '{o.group}'")
        if o.category not in CATEGORIES:
            out.append(f"{k}: unknown category '{o.category}'")
        if o.output_type not in kind_set:
            out.append(f"{k}: unknown output type '{o.output_type}'")
        if not (o.display_name and o.short_name and o.description and o.help and o.location):
            out.append(f"{k}: display name, short name, description, help and location are all required")
        roles = [n.role for n in o.needs]
        if len(roles) != len(set(roles)):
            out.append(f"{k}: two needs share a role")
        if o.needs and "data" not in roles:
            out.append(f"{k}: needs but no 'data' role")
        for n in o.needs:
            for tok in n.candidates:
                if tok.startswith("@"):
                    if tok[1:] not in kind_set:
                        out.append(f"{k}: need '{n.role}' names unknown kind '{tok}'")
                elif tok not in stage_set:
                    out.append(f"{k}: need '{n.role}' names unknown stage '{tok}'")
        for a in o.advice:
            for s in a.stages:
                if s not in stage_set:
                    out.append(f"{k}: advice names unknown stage '{s}'")
                elif STAGE_INDEX[s] >= STAGE_INDEX[o.id]:
                    out.append(f"{k}: advice says '{s}' comes before it, but the recommended order puts "
                               f"'{s}' at or after it")
            if a.mode not in ("all", "any"):
                out.append(f"{k}: advice mode '{a.mode}' is neither 'all' nor 'any'")
            if not (a.reason or a.message):
                out.append(f"{k}: advice on {a.stages} gives no reason")
        for x in o.exclusions:
            if x.role not in roles:
                out.append(f"{k}: exclusion on role '{x.role}', which it has no need for")
            for s in x.stages:
                if s not in stage_set:
                    out.append(f"{k}: exclusion names unknown stage '{s}'")
        names = [p.name for p in o.params]
        if len(names) != len(set(names)):
            out.append(f"{k}: duplicate parameter names")
        seen: List[str] = []
        for p in o.params:
            if p.when is not None and p.when[0] not in seen:
                out.append(f"{k}: parameter '{p.name}' depends on '{p.when[0]}', which is not declared before it")
            if p.type == "choice" and not p.choices:
                out.append(f"{k}: choice parameter '{p.name}' has no choices")
            if p.min is not None and p.max is not None and p.min > p.max:
                out.append(f"{k}: parameter '{p.name}' has min > max")
            if p.default is not REQUIRED:
                try:
                    p.check(p.default, o.display_name)
                except (ValidationError, ValueError) as exc:
                    out.append(f"{k}: the default of '{p.name}' fails its own schema ({exc})")
            seen.append(p.name)
        if o.is_root and o.output_type not in ("shot", "cdp"):
            out.append(f"{k}: a root operation must produce gathers")
        if o.alters_amplitudes and o.output_type != "shot":
            out.append(f"{k}: alters_amplitudes is used for shot-domain operations only")
        if require_implementations and o.executable:
            if o.compute is None:
                out.append(f"{k}: no compute function is bound")
            if bool(o.extra_inputs) != (o.extras is not None):
                out.append(f"{k}: extra_inputs and the bound extras function disagree")

    if set(_SHOT_OPERATORS) != set(STREAM_MEMBERS.get("shot", ())):
        out.append("the conditioning operations are not exactly the members of the shot stream")
    for kind in ("shot", "cdp", "section"):
        bases = [o.id for o in OPERATIONS if o.stream_base and o.output_type == kind]
        if len(bases) != 1:
            out.append(f"stream '{kind}' must have exactly one base operation (has {bases})")
    if set(ROOT_STAGES) != {"shot_gathers", "cdp_sort"}:
        out.append(f"unexpected root stages {ROOT_STAGES}")
    for o in OPERATIONS:
        if not o.executable and (o.compute is not None or o.params and any(p.user for p in o.params)):
            out.append(f"{o.key}: declared not executable, but it has an executor or user settings")
    for key in _BOUND:
        if key not in BY_KEY:
            out.append(f"an implementation is bound to undeclared operation '{key}'")
    return out
