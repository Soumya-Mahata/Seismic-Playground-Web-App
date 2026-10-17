"""
tests/test_registry.py — the operation registry (modules/registry.py)
=====================================================================
The registry is the ONE place an operation is declared. These tests check that

  * the declarations are complete and consistent with each other,
  * everything else that used to declare the same facts now *derives* them
    (and nothing re-declares them),
  * the parameter schema agrees with what the executors actually record,
  * the registry's validation runs before any maths, and a rejection is history,
  * the generated documentation is up to date.

Run:
    pytest tests/test_registry.py -v
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from modules import dependencies as D
from modules import pipeline as P
from modules import registry as R
from modules import registry_docs
from modules import stage_ops as so
from modules import state_graph
from modules.canonical import canonical
from modules.synthetic_data import generate_synthetic_survey
from modules.validation import DependencyError, ValidationError


# ─────────────────────────────────────────────────────────────────────────
# The declarations are complete and consistent
# ─────────────────────────────────────────────────────────────────────────

def test_the_registry_is_internally_consistent():
    assert R.problems(require_implementations=True) == []


def test_there_is_one_operation_per_stage_and_per_migration_variant():
    assert len(R.OPERATIONS) == 27 and len(R.stage_ids()) == 26
    assert {o.variant for o in R.variants("migration")} == {"poststack", "pstm"}
    assert [o.key for o in R.OPERATIONS if o.id == "migration"] == ["migration:poststack", "migration:pstm"]


def test_migration_needs_a_variant_and_unknown_operations_are_refused():
    with pytest.raises(ValueError):
        R.get("migration")
    with pytest.raises(KeyError):
        R.get("not_a_stage")
    assert R.find("not_a_stage") is None


def test_every_operation_documents_itself():
    for o in R.OPERATIONS:
        assert o.display_name and o.short_name and o.description and o.help and o.location, o.key
        assert o.category in R.CATEGORIES and o.group in dict(R.GROUPS), o.key


@pytest.mark.parametrize("mutation, message", [
    (lambda o: R.Operation(**{**_fields(o), "output_type": "nonsense"}), "unknown output type"),
    (lambda o: R.Operation(**{**_fields(o), "needs": (R.Need("data", ("no_such_stage",), "x"),)}),
     "unknown stage"),
    (lambda o: R.Operation(**{**_fields(o), "needs": (R.Need("data", ("@nonsense",), "x"),)}), "unknown kind"),
    (lambda o: R.Operation(**{**_fields(o), "advice": (R.Advice(("final_scale",), "r"),)}),
     "recommended order"),
    (lambda o: R.Operation(**{**_fields(o), "advice": (R.Advice(("bandpass",), ""),)}), "gives no reason"),
])
def test_the_consistency_check_catches_a_bad_declaration(monkeypatch, mutation, message):
    """`problems()` is only worth running if it can fail: break one declaration and it must say so."""
    bad = mutation(R.get("agc"))
    ops = tuple(bad if o.key == "agc" else o for o in R.OPERATIONS)
    monkeypatch.setattr(R, "OPERATIONS", ops)
    assert any(message in p for p in R.problems()), R.problems()


def _fields(op):
    return {f: getattr(op, f) for f in op.__dataclass_fields__}


# ─────────────────────────────────────────────────────────────────────────
# The example declarations from the task
# ─────────────────────────────────────────────────────────────────────────

def test_band_pass_reads_shot_gathers_and_has_no_hard_dependency():
    bp = R.get("bandpass")
    assert bp.input_types == {"data": ("shot",)} and bp.output_type == "shot"
    assert bp.required_inputs == ("shot gathers",)
    assert bp.hard_dependencies == ()                                   # hard_dependencies: none
    assert "agc" in bp.recommended_successors                           # recommended_before: AGC
    assert bp.recommended_predecessors == ()


def test_agc_lists_band_pass_as_its_recommended_predecessor():
    assert R.get("agc").recommended_predecessors == ("bandpass",)
    assert R.get("agc").alters_amplitudes


def test_nmo_requires_a_velocity_model_and_names_its_hard_dependencies():
    nmo = R.get("nmo1")
    assert nmo.input_types == {"data": ("cdp",), "velocity": ("velocity",)}
    assert nmo.output_type == "nmo"
    assert "pass-1 stacking velocities" in nmo.required_inputs
    assert nmo.hard_dependencies == ("cdp_sort", "vel1")                # velocity analysis is a hard dependency


def test_stack_needs_moveout_corrected_gathers_and_migration_needs_its_velocity():
    assert R.get("stack").input_types == {"data": ("nmo",)}
    for v in ("poststack", "pstm"):
        assert R.get("migration", v).input_types["velocity"] == ("velocity",)
    assert R.get("migration", "poststack").input_types["data"] == ("section",)
    assert R.get("migration", "pstm").input_types["data"] == ("cdp",)


def test_residual_statics_declares_what_it_cannot_read():
    (excl,) = R.get("statics").exclusions
    assert excl.stages == ("data_reg",) and excl.role == "data"


def test_the_warnings_an_operation_can_raise_are_listed_from_its_declarations():
    kinds = {w.kind for w in R.possible_warnings("agc")}
    assert kinds == {"missing_recommended", "order_inversion", "caveat"}
    assert {w.kind for w in R.possible_warnings("nmo_final")} == {"fallback_input"}
    assert any("min" in w.text.lower() for w in R.possible_warnings("decon") if w.kind == "caveat")


# ─────────────────────────────────────────────────────────────────────────
# Everything else derives from it — and nothing re-declares it
# ─────────────────────────────────────────────────────────────────────────

def test_the_pipeline_module_is_a_view_of_the_registry():
    assert [s.id for s in P.PIPELINE_STAGES] == R.stage_ids()
    for s, o in zip(P.PIPELINE_STAGES, R.stages()):
        assert (s.label, s.step_number, s.group, s.optional) == (o.display_name, o.step_number, o.group, o.optional)
    assert P.STAGE_KIND == R.STAGE_KIND and P.ROOT_STAGES == R.ROOT_STAGES and P.STAGE_INDEX == R.STAGE_INDEX
    assert [g for _, g in P.STAGE_GROUPS] == [g for g, _ in R.GROUPS]
    assert P.stage_label("agc") == "Amplitude Recovery (AGC)" and P.stage_label("zzz") == "zzz"


def test_the_dependency_engine_reads_the_registrys_declarations():
    assert D.STAGE_SPECS == R.BY_KEY
    for key, op in R.BY_KEY.items():
        assert D.STAGE_SPECS[key] is op
        assert D.get_spec(*(key.split(":") if ":" in key else (key,))) is op
    assert D.Need is R.Need and D.Advice is R.Advice
    assert D.STREAM_BASE == R.STREAM_BASE and D.STREAM_MEMBERS == R.STREAM_MEMBERS
    assert D.SHOT_OPERATORS == R.STREAM_MEMBERS["shot"]
    assert D.advice_pairs() == R.advice_pairs()


def test_names_locations_and_labels_come_from_the_registry():
    for sid in R.stage_ids():
        assert D.SHORT[sid] == R.short_name(sid) and D.STAGE_LOCATION[sid] == R.location(sid)
    assert D.state_name(None) == "—"
    assert {k: v for k, v in D.PHRASE.items()} == {o.id: o.phrase for o in R.stages() if o.phrase}
    assert D.KIND_LABELS == {t.id: t.label for t in R.DATA_TYPES if t.summarised}
    assert [(a, b) for a, b, _ in D.STATUS_ROWS] == [(o.id, o.variant or None) for o in R.OPERATIONS]
    assert [c for _, _, c in D.STATUS_ROWS] == [o.full_name for o in R.OPERATIONS]


def test_the_kinds_of_data_a_state_can_hold_come_from_the_registry():
    assert state_graph.KINDS == R.KINDS == tuple(t.id for t in R.DATA_TYPES)
    assert set(R.STAGE_KIND.values()) <= set(R.KINDS)
    assert set(so.GAIN_ALTERING_STAGES) == {"agc", "balance"} == {o.id for o in R.OPERATIONS if o.alters_amplitudes}


def test_the_old_hand_kept_tables_are_gone():
    """The registry replaced these; nothing may bring them back as a second source of truth."""
    assert not hasattr(so, "OPS") and not hasattr(so, "OpDef")
    assert not hasattr(D, "_statics_geometry")


def _python_files():
    for sub in ("modules", "utils", "views"):
        yield from (ROOT / sub).glob("*.py")
    yield ROOT / "app.py"


def test_dependency_declarations_exist_only_in_the_registry():
    """No file but the registry constructs a Need / Advice / Exclusion / ParamSpec / Operation —
    the dependency definitions are not duplicated across files."""
    names = {"Need", "Advice", "Exclusion", "ParamSpec", "Operation"}
    offenders = []
    for path in _python_files():
        if path.name == "registry.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.func.id}(…)")
    assert offenders == []


def test_no_other_file_keeps_a_table_keyed_by_stage_id():
    """A dict literal of five or more stage ids is a hand-kept per-stage table — the shape of the
    tables (names, locations, kinds, specs) that now live in the registry."""
    stage_ids = set(R.stage_ids())
    offenders = []
    for path in _python_files():
        if path.name in ("registry.py",):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                if len([k for k in keys if k in stage_ids]) >= 5:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


# ─────────────────────────────────────────────────────────────────────────
# Bound implementations
# ─────────────────────────────────────────────────────────────────────────

def test_every_executable_operation_has_a_compute_function_and_only_the_loader_has_none():
    assert [o.key for o in R.OPERATIONS if o.compute is None] == ["upload"]
    assert [o.key for o in R.OPERATIONS if not o.executable] == ["upload"]
    assert [o.key for o in R.OPERATIONS if o.extras is not None] == ["trace_edit"]
    assert R.get("trace_edit").extra_inputs == ("detect",)
    assert [o.key for o in R.OPERATIONS if o.qc is not None] == ["vel1", "vel2"]


def test_a_different_function_cannot_be_bound_to_a_bound_operation_nor_to_an_undeclared_one():
    with pytest.raises(ValueError, match="already has a compute function"):
        R.bind("agc", compute=lambda run, p: ({}, {}))
    with pytest.raises(KeyError):
        R.bind("not_declared", compute=lambda run, p: ({}, {}))


def test_re_binding_the_same_function_after_a_module_reload_replaces_it():
    """A hot reload of `stage_ops` re-runs its decorators; that must not be an error."""
    import types
    original = R.get("agc").compute
    clone = types.FunctionType(original.__code__, original.__globals__, original.__name__)
    clone.__module__, clone.__qualname__ = original.__module__, original.__qualname__
    try:
        R.bind("agc", compute=clone)
        assert R.get("agc").compute is clone
    finally:
        R.bind("agc", compute=original)
    assert R.get("agc").compute is original and R.problems(require_implementations=True) == []


def test_the_velocity_qc_builder_adds_the_velocity_range_to_the_history(survey_store):
    so.run_build_shot_gathers(survey_store)
    so.run_cdp_sort(survey_store)
    state = so.run_velocity(survey_store, "vel1", n_analysis=3)
    qc = so.action_log(survey_store).last(kind="operation").qc
    assert "v_min" in qc and "v_max" in qc and 0 < qc["v_min"] < qc["v_max"]
    assert so.qc_summary(survey_store, state)["v_min"] == qc["v_min"]


# ─────────────────────────────────────────────────────────────────────────
# The parameter schema agrees with what the executors record
# ─────────────────────────────────────────────────────────────────────────

TRIANGLE = [(0.0, 0.0), (0.05, 100.0), (-0.05, 100.0)]

# how to call each executor with nothing but its defaults: key -> [(overrides the schema needs, call)]
CALLS = {
    "shot_gathers": [({}, lambda s: so.run_build_shot_gathers(s))],
    "bandpass": [({}, lambda s: so.run_bandpass(s))],
    "agc": [({}, lambda s: so.run_agc(s))],
    "despike": [({}, lambda s: so.run_despike(s))],
    "trace_edit": [({}, lambda s: so.run_trace_kill(s))],
    "mute": [({"kind": "flat"}, lambda s: so.run_mute(s)),
             ({"kind": "linear"}, lambda s: so.run_mute(s, "linear")),
             ({"kind": "surgical"}, lambda s: so.run_mute(s, "surgical"))],
    "balance": [({}, lambda s: so.run_balance(s))],
    "fk": [({"polygon": TRIANGLE}, lambda s: so.run_fk(s, TRIANGLE))],
    "mult_supp": [({}, lambda s: so.run_mult_supp(s))],
    "decon": [({}, lambda s: so.run_decon(s))],
    "cdp_sort": [({}, lambda s: so.run_cdp_sort(s))],
    "vel1": [({}, lambda s: so.run_velocity(s, "vel1"))],
    "nmo1": [({}, lambda s: so.run_nmo1(s))],
    "statics": [({}, lambda s: so.run_statics(s))],
    "dmo": [({}, lambda s: so.run_dmo(s))],
    "inv_nmo": [({}, lambda s: so.run_inverse_nmo(s))],
    "data_reg": [({}, lambda s: so.run_data_reg(s))],
    "vel2": [({}, lambda s: so.run_velocity(s, "vel2"))],
    "nmo_final": [({}, lambda s: so.run_nmo_final(s))],
    "stack": [({}, lambda s: so.run_stack(s))],
    "rna": [({}, lambda s: so.run_rna(s))],
    "decon_post": [({}, lambda s: so.run_decon_post(s))],
    "tvf": [({}, lambda s: so.run_tvf(s))],
    "migration:poststack": [({}, lambda s: so.run_migration_poststack(s))],
    "migration:pstm": [({}, lambda s: so.run_migration_pstm(s))],
    "final_scale": [({}, lambda s: so.run_final_scale(s))],
}


def test_the_table_of_calls_covers_every_executable_operation():
    assert set(CALLS) == {o.key for o in R.OPERATIONS if o.executable}


@pytest.mark.parametrize("key", sorted(CALLS))
def test_an_executor_called_with_defaults_records_exactly_the_schemas_defaults(monkeypatch, key):
    """The keys and values an executor records (they fix the state's fingerprint) are the schema's
    defaults — so the schema, the `run_*` signatures and `config.py` cannot drift apart."""
    captured = []

    def fake_execute(store, stage_id, params=None, variant=None, **kw):
        captured.append((stage_id, variant, canonical(params or {})))

    monkeypatch.setattr(so, "execute", fake_execute)
    op = R.find(key)
    for overrides, call in CALLS[key]:
        captured.clear()
        call(so.new_store())
        (stage_id, variant, params), = captured
        assert (stage_id, variant or None) == (op.id, op.variant or None)
        assert params == canonical(op.default_params(**overrides)), (key, overrides)


@pytest.mark.parametrize("key", sorted(CALLS))
def test_the_default_parameters_pass_their_own_validation(key):
    op = R.find(key)
    for overrides, _ in CALLS[key]:
        op.validate_params(op.default_params(**overrides), None)


def test_defaults_come_from_the_config_module():
    assert R.get("bandpass").default_params() == {
        "fmin": cfg.FMIN_DEFAULT, "fmax": cfg.FMAX_DEFAULT,
        "low_slope": cfg.LOW_SLOPE_DEFAULT, "high_slope": cfg.HIGH_SLOPE_DEFAULT}
    assert R.get("agc").default_params() == {"window": cfg.AGC_WINDOW_DEFAULT, "mode": cfg.AGC_MODE_DEFAULT}
    assert R.get("tvf").default_params()["zones"] == cfg.TVF_ZONES_DEFAULT
    assert R.get("tvf").default_params()["zones"] is not cfg.TVF_ZONES_DEFAULT      # a copy: config is never edited


def test_parameters_that_only_apply_to_one_kind_of_mute_are_recorded_only_for_it():
    mute = R.get("mute")
    assert set(mute.default_params(kind="flat")) == {"kind", "t_top", "taper"}
    assert set(mute.default_params(kind="linear")) == {"kind", "velocity", "t0", "taper"}
    assert set(mute.default_params(kind="surgical")) == {"kind", "x_min", "x_max", "t_min", "t_max", "taper"}
    with pytest.raises(ValidationError, match="does not apply"):
        mute.validate_params({"kind": "flat", "t_top": 0.1, "taper": 0.05, "velocity": 500.0})


def test_settings_are_the_parameters_a_person_can_set():
    assert [p.name for p in R.get("tvf").settings] == []                 # fixed by the application
    assert "n_velocities" not in [p.name for p in R.get("vel1").settings]
    assert [p.name for p in R.get("agc").settings] == ["window", "mode"]


# ─────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key, params, message", [
    ("agc", {"window": 250}, "missing"),
    ("agc", {"window": 250, "mode": "rms", "extra": 1}, "unknown parameter"),
    ("agc", {"window": 0, "mode": "rms"}, "at least 1"),
    ("agc", {"window": 2.5, "mode": "rms"}, "whole number"),
    ("agc", {"window": True, "mode": "rms"}, "whole number"),
    ("agc", {"window": "250", "mode": "rms"}, "whole number"),
    ("agc", {"window": 250, "mode": "bogus"}, "Unknown AGC mode 'bogus' \\(expected rms, mean or median\\)"),
    ("bandpass", {"fmin": float("nan"), "fmax": 60.0, "low_slope": 6.0, "high_slope": 12.0}, "finite"),
    ("bandpass", {"fmin": -1.0, "fmax": 60.0, "low_slope": 6.0, "high_slope": 12.0}, "at least 0"),
    ("bandpass", {"fmin": 40.0, "fmax": 20.0, "low_slope": 6.0, "high_slope": 12.0}, "must be less than"),
    ("vel1", {**R.get("vel1").default_params(), "vmin": 4000.0, "vmax": 2000.0}, "must be less than"),
    ("vel1", {**R.get("vel1").default_params(), "vmin": 0.0}, "greater than 0"),
    ("vel1", {**R.get("vel1").default_params(), "manual_overrides": [1, 2]}, "mapping"),
    ("mute", {"kind": "bogus"}, "Unknown mute type 'bogus' \\(expected flat, linear or surgical\\)"),
    ("mute", {"kind": "linear", "velocity": 0.0, "t0": 0.0, "taper": 0.05}, "greater than 0"),
    ("fk", {"polygon": None, "mode": "accept", "coeff": 0.02}, "at least 3 vertices"),
    ("fk", {"polygon": [(0, 0), (1, 1)], "mode": "accept", "coeff": 0.02}, "at least 3 vertices"),
    ("fk", {"polygon": "triangle", "mode": "accept", "coeff": 0.02}, "list of vertices"),
    ("fk", {"polygon": TRIANGLE, "mode": "accept", "coeff": -0.01}, "at least 0"),
    ("fk", {"mode": "accept", "coeff": 0.02}, "required"),
])
def test_the_schema_rejects_what_the_maths_cannot_use(key, params, message):
    with pytest.raises(ValidationError, match=message):
        R.find(key).validate_params(params)


_BOUNDED = [(o.key, p.name) for o in R.OPERATIONS for p in o.params
            if p.type in ("float", "int") and (p.min is not None or p.gt is not None)]


@pytest.mark.parametrize("key, name", _BOUNDED)
def test_a_declared_bound_is_exactly_the_edge_of_what_is_accepted(key, name):
    """`min` is inclusive and `gt` strict, for every bounded parameter: the value on the edge is
    accepted (min) or refused (gt), and the next representable step beyond it goes the other way."""
    op = R.find(key)
    p = next(q for q in op.params if q.name == name)
    step = 1 if p.type == "int" else 1e-9
    if p.min is not None:
        p.check(p.min, op.display_name)
        with pytest.raises(ValidationError):
            p.check(p.min - step, op.display_name)
    if p.gt is not None:
        with pytest.raises(ValidationError):
            p.check(p.gt, op.display_name)
        p.check(p.gt + step, op.display_name)


def test_the_fk_coefficient_bound_matches_what_the_ui_and_the_maths_accept():
    """The slider runs 0.0-0.3 and `fk_filter_gather` clips the coefficient to [0, 1]: 0.0 is a valid
    (if not recommended) value, so the schema must not refuse it."""
    fk = R.get("fk")
    fk.validate_params({"polygon": TRIANGLE, "mode": "accept", "coeff": 0.0})
    fk.validate_params({"polygon": TRIANGLE, "mode": "reject", "coeff": 0.3})


def test_a_stage_with_variants_has_one_summary_line_and_the_others_use_their_own():
    assert R.stage_description("migration") == (
        "Post-stack Kirchhoff time migration, or pre-stack Kirchhoff time migration (PSTM) "
        "on CDP gathers before stack.")
    assert R.stage_description("agc") == R.get("agc").description
    assert [s.description for s in P.PIPELINE_STAGES if s.id == "migration"] == [R.stage_description("migration")]


def test_a_partial_set_checks_only_what_it_has():
    agc = R.get("agc")
    agc.validate_params({"window": 100}, complete=False)
    with pytest.raises(ValidationError):
        agc.validate_params({"window": -1}, complete=False)
    with pytest.raises(ValidationError, match="unknown parameter"):
        agc.validate_params({"windo": 100}, complete=False)


def test_numpy_numbers_are_valid_numbers():
    R.get("agc").validate_params({"window": np.int64(250), "mode": "rms"})
    R.get("bandpass").validate_params({"fmin": np.float64(8.0), "fmax": np.float32(60.0),
                                       "low_slope": 6, "high_slope": 12})


def test_the_nyquist_check_needs_the_survey_and_uses_it():
    class Survey:                           # dt = 2 ms -> Nyquist 250 Hz
        sample_interval = 0.002
    params = {"fmin": 10.0, "fmax": 300.0, "low_slope": 6.0, "high_slope": 12.0}
    R.get("bandpass").validate_params(params, None)                      # data-independent part only
    with pytest.raises(ValidationError, match="Nyquist"):
        R.get("bandpass").validate_params(params, Survey())


# ─────────────────────────────────────────────────────────────────────────
# Validation runs before the maths, and a rejection is history
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def survey():
    return generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                     inject_field_noise=True)


@pytest.fixture
def survey_store(survey):
    return so.new_store(survey)


def _last(store):
    return so.action_log(store).last(kind="operation")


def test_an_invalid_band_is_rejected_before_any_state_is_made_and_is_recorded(survey_store):
    so.run_build_shot_gathers(survey_store)
    before = [s.id for s in so.graph(survey_store)]
    with pytest.raises(ValidationError, match="Nyquist"):
        so.run_bandpass(survey_store, fmin=10.0, fmax=400.0)             # survey dt = 2 ms
    e = _last(survey_store)
    assert (e.action, e.status, e.failure_kind) == ("bandpass", "failure", "invalid")
    assert e.params["fmax"] == 400.0 and e.output is None
    assert [s.id for s in so.graph(survey_store)] == before               # nothing was stored
    nxt = so.run_bandpass(survey_store)
    assert int(nxt.id[1:]) == int(before[-1][1:]) + 1                     # and no state id was consumed


def test_a_schema_violation_in_a_complete_call_is_recorded_as_invalid(survey_store):
    so.run_build_shot_gathers(survey_store)
    with pytest.raises(ValidationError, match="at least 1"):
        so.run_agc(survey_store, window=0)
    with pytest.raises(ValidationError, match="Unknown AGC mode"):
        so.run_agc(survey_store, mode="bogus")
    log = [(e.action, e.failure_kind) for e in so.action_log(survey_store).entries() if e.status == "failure"]
    assert log == [("agc", "invalid"), ("agc", "invalid")]
    assert not so.graph(survey_store).of_op("agc")


def test_an_unknown_parameter_is_refused_rather_than_silently_recorded(survey_store):
    so.run_build_shot_gathers(survey_store)
    with pytest.raises(ValidationError, match="unknown parameter"):
        so.execute(survey_store, "agc", {"window": 100, "mode": "rms", "windw": 5})
    with pytest.raises(ValidationError, match="missing"):
        so.execute(survey_store, "agc", {"window": 100})


def test_unknown_mute_type_and_short_fk_polygon_are_still_refused_with_their_original_messages(survey_store):
    so.run_build_shot_gathers(survey_store)
    with pytest.raises(ValidationError) as e1:
        so.run_mute(survey_store, "bogus")
    assert str(e1.value) == "Unknown mute type 'bogus' (expected flat, linear or surgical)."
    with pytest.raises(ValidationError) as e2:
        so.run_fk(survey_store, [(0, 0), (1, 1)])
    assert str(e2.value) == "An F-K polygon needs at least 3 vertices."
    recorded = [(e.action, e.failure_kind, e.params) for e in so.action_log(survey_store).entries()
                if e.status == "failure"]
    assert recorded[0] == ("mute", "invalid", {"kind": "bogus"})
    assert recorded[1][:2] == ("fk", "invalid") and recorded[1][2]["mode"] == "accept"


def test_a_dependency_failure_is_still_reported_as_blocked_not_invalid(survey_store):
    with pytest.raises(DependencyError):
        so.run_agc(survey_store)                                          # no shot gathers yet
    assert _last(survey_store).failure_kind == "blocked"


def test_a_valid_run_records_the_same_parameters_the_schema_declares(survey_store):
    so.run_build_shot_gathers(survey_store)
    state = so.run_agc(survey_store, window=100, mode="mean")
    assert state.params == R.get("agc").default_params(window=100, mode="mean")


def test_history_labels_come_from_the_registry(survey_store):
    so.run_build_shot_gathers(survey_store)
    so.run_agc(survey_store)
    so.run_cdp_sort(survey_store)
    so.run_velocity(survey_store, "vel1", n_analysis=3)
    labels = {e.action: e.label for e in so.action_log(survey_store).entries() if e.kind == "operation"}
    for action, label in labels.items():
        assert label == R.stage_label(action), (action, label)


# ─────────────────────────────────────────────────────────────────────────
# Generated documentation and workflow-graph data
# ─────────────────────────────────────────────────────────────────────────

def test_the_generated_documentation_is_up_to_date():
    assert registry_docs.out_of_date() == [], "run: python -m modules.registry_docs"


def test_every_operation_appears_in_the_generated_reference():
    text = registry_docs.render_operations_md()
    for o in R.OPERATIONS:
        assert o.full_name in text and f"`{o.key}`" in text
    for p in R.get("vel1").params:
        assert f"`{p.name}`" in text
    assert "```mermaid" in text


def test_the_workflow_graph_is_plain_serialisable_data_with_no_dangling_edges():
    import json
    graph = R.workflow_graph()
    json.dumps(graph)
    ids = {n["id"] for n in graph["nodes"]}
    assert graph["recommended_order"] == [o.key for o in R.OPERATIONS]
    for e in graph["edges"]:
        assert e["from"] in ids and e["to"] in ids, e
    kinds = {e["type"] for e in graph["edges"]}
    assert kinds == {"produces", "consumes", "requires", "recommends", "excludes"}


def test_the_graph_edges_reflect_the_declared_dependencies():
    graph = R.workflow_graph()
    edges = {(e["type"], e["from"], e["to"]) for e in graph["edges"]}
    assert ("consumes", "data:shot", "bandpass") in edges
    assert ("requires", "vel1", "nmo1") in edges and ("requires", "cdp_sort", "nmo1") in edges
    assert ("recommends", "bandpass", "agc") in edges
    assert ("excludes", "data_reg", "statics") in edges
    assert ("produces", "stack", "data:section") in edges
    assert not any(t == "requires" and f == "bandpass" for t, f, _ in edges)      # no hard dependency
