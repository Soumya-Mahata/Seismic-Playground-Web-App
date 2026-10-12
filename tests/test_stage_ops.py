"""
tests/test_stage_ops.py — data states through the real maths
================================================================
`modules/stage_ops.py` holds one executor per operation; the Streamlit pages
and these tests call the same functions. Each call does

        INPUT STATE ──▶ OPERATION + PARAMETERS ──▶ OUTPUT STATE

and records the result as a new state in the `ProcessingGraph`. A small
synthetic survey is processed in orders the old linear pipeline refused —
skipping optional stages, repeating them, applying them out of the textbook
order, branching from an earlier state — and the genuine dependencies are shown
to still hold.

    pytest tests/test_stage_ops.py -v

Layout
  * non-standard orders that must work (and their numerical meaning)
  * skipped optional stages
  * genuine dependency failures  (⛔ -> DependencyError, nothing changes)
  * the executors agree with the dependency engine
  * the state model: repeat, branch, memoise, no input copies, release /
    materialise, undo / redo, discard, stale detection, replay (reproducibility)
  * the two migration implementations
  * history / export
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from modules import filtering, preprocessing, segy_io
from modules import stage_ops as so
from modules.dependencies import Level, describe_stream, stale_info
from modules.pipeline import PIPELINE_STAGES
from modules.state_graph import GraphError
from modules.synthetic_data import generate_synthetic_survey
from modules.validation import DependencyError, ValidationError
from utils import io_export

N_MIN = cfg.N_MIN_TRACES_DEFAULT


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def survey():
    return generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                     inject_field_noise=True)


@pytest.fixture
def store(survey):
    return so.new_store(survey)


@pytest.fixture
def shots(store):
    so.run_build_shot_gathers(store)
    return store


@pytest.fixture
def cdp(shots):
    """Survey → shot gathers → CDP sort (no conditioning at all)."""
    so.run_cdp_sort(shots, N_MIN)
    return shots


@pytest.fixture
def vel1(cdp):
    so.run_velocity(cdp, "vel1", n_analysis=3)
    return cdp


def fk_polygon(survey):
    return filtering.default_fk_polygon(survey.sample_interval, survey.group_interval)


def ops(store):
    """Operations in the order they were run."""
    return [s.op for s in so.graph(store)]


def latest(store, op, variant=None):
    return so.graph(store).latest(op, variant)


def traces(store, state, nth=3, name="traces"):
    """One gather's array of a gather-based state."""
    container = so.gathers_of(store, state)
    return container[list(container)[nth]][state.keys[name]]


def finite(a):
    return a is not None and np.isfinite(a).all()


def kinds(chk):
    return [i.kind for i in chk.issues]


def footprint(store):
    """What is held in memory: the graph's states, and how many arrays each gather dict holds."""
    return ([s.id for s in so.graph(store)],
            {root: sorted((gid, len(g)) for gid, g in c.items()) for root, c in so.containers(store).items()},
            sorted(so.blobs(store)))


# ─────────────────────────────────────────────────────────────────────────
# Non-standard orders (the two examples from the task, through real maths)
# ─────────────────────────────────────────────────────────────────────────

def test_raw_shot_agc_bandpass_mute_then_cdp_sort(shots):
    """Raw → Shot Gathers → AGC → Band-pass → Mute → CDP Sort."""
    sg = latest(shots, "shot_gathers")
    agc = so.run_agc(shots)
    bp = so.run_bandpass(shots)
    mute = so.run_mute(shots, "flat", t_top=0.05)
    sort = so.run_cdp_sort(shots, N_MIN)

    assert ops(shots)[-4:] == ["agc", "bandpass", "mute", "cdp_sort"]
    assert (agc.inputs["data"], bp.inputs["data"], mute.inputs["data"], sort.inputs["data"]) == \
           (sg.id, agc.id, bp.id, mute.id)
    assert describe_stream(so.workflow(shots), "shot") == "Raw → AGC → Band-pass → Mute"
    # every output is a new array under the state's own key; nothing was written over an input
    assert agc.keys["traces"] == f"traces@{agc.id}" and bp.keys["traces"] == f"traces@{bp.id}"
    # the warnings were advice, and were kept in the record
    assert "Run despite" in agc.notes and "Band-pass" in agc.notes
    assert "Run despite" in bp.notes
    assert finite(traces(shots, mute)) and finite(traces(shots, sort))


def test_raw_shot_bandpass_fk_then_cdp_sort(shots, survey):
    """Raw → Shot Gathers → Band-pass → F-K → CDP Sort (AGC, mute, decon… all skipped)."""
    so.run_bandpass(shots)
    fk = so.run_fk(shots, fk_polygon(survey))
    sort = so.run_cdp_sort(shots, N_MIN)
    assert sort.inputs["data"] == fk.id
    assert sort.notes == ""                                        # not even a ⚠
    assert describe_stream(so.workflow(shots), "shot") == "Raw → Band-pass → F-K"


def test_executors_apply_the_same_maths_as_calling_the_modules_directly(shots, survey):
    """Band-pass then AGC through the executors == the module calls on a shot."""
    bp = so.run_bandpass(shots, fmin=cfg.FMIN_DEFAULT, fmax=cfg.FMAX_DEFAULT)
    agc = so.run_agc(shots, window=cfg.AGC_WINDOW_DEFAULT, mode=cfg.AGC_MODE_DEFAULT)
    raw = traces(shots, latest(shots, "shot_gathers"), nth=5)
    h = filtering.design_bandpass(cfg.FMIN_DEFAULT, cfg.FMAX_DEFAULT, survey.sample_interval,
                                  survey.num_samples, cfg.LOW_SLOPE_DEFAULT, cfg.HIGH_SLOPE_DEFAULT)
    expected_bp = filtering.apply_time_domain_conv(raw, h)
    expected_agc = preprocessing.agc(expected_bp, window=cfg.AGC_WINDOW_DEFAULT, mode=cfg.AGC_MODE_DEFAULT)
    np.testing.assert_array_equal(traces(shots, bp, nth=5), expected_bp)
    np.testing.assert_array_equal(traces(shots, agc, nth=5), expected_agc)


def test_processing_order_is_real_agc_then_bandpass_differs_from_bandpass_then_agc(survey):
    """The recommended order is a recommendation because it *matters*: the
    two orders give different data, and each is recorded for what it is."""
    a = so.new_store(survey); so.run_build_shot_gathers(a)
    so.run_agc(a); out_a = so.run_bandpass(a)                    # AGC → BP
    b = so.new_store(survey); so.run_build_shot_gathers(b)
    so.run_bandpass(b); out_b = so.run_agc(b)                    # BP → AGC
    assert not np.allclose(traces(a, out_a, 5), traces(b, out_b, 5))
    assert describe_stream(so.workflow(a), "shot") == "Raw → AGC → Band-pass"
    assert describe_stream(so.workflow(b), "shot") == "Raw → Band-pass → AGC"
    assert out_a.fingerprint != out_b.fingerprint                # different lineage, different identity


def test_shot_operators_chain_whatever_the_order(shots, survey):
    """Every shot operator in an arbitrary, non-textbook order: each reads the
    previous output, nothing is blocked, results stay finite."""
    order = [("balance", lambda: so.run_balance(shots)),
             ("decon", lambda: so.run_decon(shots)),
             ("mute", lambda: so.run_mute(shots, "flat", t_top=0.05)),
             ("despike", lambda: so.run_despike(shots)),
             ("mult_supp", lambda: so.run_mult_supp(shots, 30, 10)),
             ("bandpass", lambda: so.run_bandpass(shots)),
             ("fk", lambda: so.run_fk(shots, fk_polygon(survey))),
             ("agc", lambda: so.run_agc(shots)),
             ("trace_edit", lambda: so.run_trace_kill(shots))]
    previous = latest(shots, "shot_gathers")
    for stage, fn in order:
        state = fn()
        assert state.op == stage and state.inputs["data"] == previous.id, stage
        previous = state
    assert finite(traces(shots, previous))
    sort = so.run_cdp_sort(shots, N_MIN)
    assert sort.inputs["data"] == previous.id and finite(traces(shots, sort, nth=0))


def test_cdp_gathers_carry_the_conditioning_that_was_applied(shots, survey):
    """The sorted CDP traces equal the head of the shot data, re-grouped."""
    bp = so.run_bandpass(shots)
    sort = so.run_cdp_sort(shots, N_MIN)
    flat = segy_io.scatter_to_flat(so.gathers_of(shots, bp), bp.keys["traces"], survey.num_traces,
                                   survey.num_samples)
    expected = segy_io.build_cdp_gathers(survey, n_min_traces=N_MIN, data_override=flat)
    assert list(expected) == so.cdp_list_of(shots, sort)
    got = so.gathers_of(shots, sort)
    for c in list(expected)[:5]:
        np.testing.assert_array_equal(got[c]["traces"], expected[c]["traces"])


# ─────────────────────────────────────────────────────────────────────────
# Skipped optional stages
# ─────────────────────────────────────────────────────────────────────────

def test_cdp_sort_without_shot_gathers_sorts_the_raw_traces(store, survey):
    sort = so.run_cdp_sort(store, N_MIN)
    assert sort.inputs["data"] == latest(store, "upload").id
    expected = segy_io.build_cdp_gathers(survey, n_min_traces=N_MIN)
    assert list(expected) == so.cdp_list_of(store, sort)
    c = so.cdp_list_of(store, sort)[10]
    np.testing.assert_array_equal(so.gathers_of(store, sort)[c]["traces"], expected[c]["traces"])


def test_skipping_every_optional_stage_still_reaches_a_final_section(cdp):
    """CDP sort → velocity 1 → NMO 1 → stack → migration → final scaling.
    No conditioning, statics, DMO, regularisation, pass-2 velocity, or post-stack filters."""
    vel = so.run_velocity(cdp, "vel1", n_analysis=3)
    nmo1 = so.run_nmo1(cdp)
    stack = so.run_stack(cdp)
    mig = so.run_migration_poststack(cdp, aperture_m=150.0)
    fin = so.run_final_scale(cdp)
    assert finite(cdp["stack_matrix"]) and finite(cdp["stack_migrated"]) and finite(cdp["stack_final"])
    assert mig.inputs == {"data": stack.id, "velocity": vel.id}
    assert stack.inputs["data"] == nmo1.id and fin.inputs["data"] == mig.id
    for stage in ("shot_gathers", "cdp_sort", "vel1", "nmo1", "stack", "migration", "final_scale"):
        assert so.workflow(cdp).has(stage)
    for skipped in ("bandpass", "statics", "dmo", "inv_nmo", "data_reg", "vel2", "nmo_final", "rna",
                    "decon_post", "tvf"):
        assert not so.workflow(cdp).has(skipped)
    assert so.best_available_section(cdp)[1] == "Final processed section"


def test_stack_can_use_pass_one_nmo_and_says_so(vel1):
    nmo1 = so.run_nmo1(vel1)
    chk = so.check(vel1, "stack")
    assert chk.level == Level.WARNING
    stack = so.run_stack(vel1)
    assert stack.inputs["data"] == nmo1.id and "Final NMO has not been applied" in stack.notes


def test_final_nmo_needs_only_pass_one_velocity(vel1):
    vel = latest(vel1, "vel1")
    state = so.run_nmo_final(vel1)
    assert state.inputs["velocity"] == vel.id
    assert finite(traces(vel1, state, nth=0))


def test_dmo_runs_without_nmo_pass_one_or_statics(vel1):
    state = so.run_dmo(vel1)
    assert state.inputs == {"data": latest(vel1, "cdp_sort").id, "velocity": latest(vel1, "vel1").id}
    assert "statics" in state.notes.lower() and "commonly applied before DMO" in state.notes
    assert finite(traces(vel1, state, nth=0))


def test_post_stack_operators_apply_in_any_order(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    tvf = so.run_tvf(vel1)                   # TVF first…
    rna = so.run_rna(vel1)                   # …then RNA — on the TVF output, not on the stack
    dec = so.run_decon_post(vel1)
    assert (rna.inputs["data"], dec.inputs["data"]) == (tvf.id, rna.id)
    assert describe_stream(so.workflow(vel1), "section") == "Stack → TVF → RNA → Post-stack decon"
    assert so.best_available_section(vel1)[1].startswith("Post-stack")
    assert finite(vel1["stack_decon_post"])


def test_trace_kill_finds_the_same_traces_whether_or_not_agc_ran_first(survey):
    """AGC equalises trace energy, which would blind an energy-based dead/noisy
    detector. The kill step measures amplitude-preserving data, so the traces
    killed do not depend on whether AGC was applied before it."""
    a = so.new_store(survey); so.run_build_shot_gathers(a)
    kill_first = so.run_trace_kill(a)
    b = so.new_store(survey); so.run_build_shot_gathers(b)
    agc = so.run_agc(b)
    kill_after_agc = so.run_trace_kill(b)
    assert kill_first.info["traces_killed"] > 0
    assert kill_after_agc.info["traces_killed"] == kill_first.info["traces_killed"]
    # the second input role records *where the mask was measured*: upstream of the gain
    assert kill_first.inputs.keys() == {"data"}
    assert kill_after_agc.inputs == {"data": agc.id, "detect": agc.inputs["data"]}
    assert "AGC" not in kill_after_agc.info["detected_on"]
    for gid in list(so.gathers_of(a, kill_first))[:6]:
        fa = so.gathers_of(a, kill_first)[gid][kill_first.keys["kill_flags"]]["kill"]
        fb = so.gathers_of(b, kill_after_agc)[gid][kill_after_agc.keys["kill_flags"]]["kill"]
        np.testing.assert_array_equal(fa, fb)


def test_kill_detection_source_rule():
    f = so.kill_detection_stage
    assert f(["shot_gathers", "trace_edit"]) == "trace_edit"                       # no gain: measure the data itself
    assert f(["shot_gathers", "agc", "trace_edit"]) == "shot_gathers"
    assert f(["shot_gathers", "bandpass", "agc", "trace_edit"]) == "bandpass"
    assert f(["shot_gathers", "bandpass", "balance", "agc", "trace_edit"]) == "bandpass"
    assert f(["shot_gathers", "agc"]) == "shot_gathers"


# ─────────────────────────────────────────────────────────────────────────
# Genuine dependency failures: ⛔ -> DependencyError, and nothing changes
# ─────────────────────────────────────────────────────────────────────────

def assert_blocked(store, fn, *args, **kwargs):
    """The executor refuses with a DependencyError and leaves no trace: no
    state, no id consumed, no array stored."""
    before, next_id = footprint(store), so.graph(store).peek_id()
    with pytest.raises(DependencyError) as exc:
        fn(store, *args, **kwargs)
    assert footprint(store) == before and so.graph(store).peek_id() == next_id
    return exc.value


def test_velocity_analysis_without_cdp_gathers_is_refused(shots):
    err = assert_blocked(shots, so.run_velocity, "vel1")
    assert err.stage_id == "vel1" and "CDP gathers" in str(err) and "CDP Sort" in str(err)
    assert shots["Vfield_1"] is None


def test_nmo_without_velocity_is_refused(cdp):
    for fn in (so.run_nmo1, so.run_nmo_final, so.run_dmo):
        err = assert_blocked(cdp, fn)
        assert "velocities" in str(err), fn.__name__


def test_stack_without_moveout_correction_is_refused(vel1):
    err = assert_blocked(vel1, so.run_stack)
    assert err.stage_id == "stack" and "NMO" in str(err)
    assert vel1["stack_matrix"] is None


def test_stack_blocked_message_names_where_to_fix_it(vel1):
    chk = so.check(vel1, "stack")
    assert chk.level == Level.BLOCKED
    assert "Velocity, NMO & DMO" in chk.blockers[0].fix


def test_residual_statics_need_nmo_pass_one(vel1):
    err = assert_blocked(vel1, so.run_statics)
    assert "NMO" in str(err)


def test_residual_statics_are_refused_on_regularised_gathers(vel1):
    so.run_nmo1(vel1)
    reg = so.run_data_reg(vel1)
    err = assert_blocked(vel1, so.run_statics)
    assert "shot and receiver" in str(err)
    # …but statics can still be applied to the gathers *before* regularisation (branch from
    # an earlier state), and regularisation then follows it
    pre_reg = reg.inputs["data"]
    statics = so.run_statics(vel1, inputs={"data": pre_reg})
    assert statics.inputs["data"] == pre_reg
    assert so.run_data_reg(vel1).inputs["data"] == statics.id


def test_inverse_nmo_needs_dmo_output(vel1):
    assert_blocked(vel1, so.run_inverse_nmo)


def test_post_stack_steps_need_a_stack(vel1):
    so.run_nmo1(vel1)
    for fn in (so.run_rna, so.run_decon_post, so.run_tvf, so.run_final_scale):
        assert_blocked(vel1, fn)
    assert_blocked(vel1, so.run_migration_poststack)


def test_post_stack_migration_needs_a_stack_but_not_post_stack_filters(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    state = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert "Run despite" in state.notes and "random noise" in state.notes.lower()
    assert finite(vel1["stack_migrated"])


def test_prestack_migration_needs_gathers_and_a_velocity_field(store):
    assert_blocked(store, so.run_migration_pstm, aperture_m=150.0)           # no CDP gathers
    so.run_build_shot_gathers(store)
    so.run_cdp_sort(store, N_MIN)
    err = assert_blocked(store, so.run_migration_pstm, aperture_m=150.0)     # no velocity
    assert "velocity" in str(err)


def test_shot_domain_steps_need_shot_gathers(store):
    for fn in (so.run_bandpass, so.run_agc, so.run_despike, so.run_trace_kill, so.run_balance,
               so.run_decon, so.run_mult_supp):
        assert_blocked(store, fn)
    assert_blocked(store, so.run_mute, "flat")


def test_an_explicit_input_of_the_wrong_kind_is_refused(vel1):
    """Naming an input state does not bypass the rules: a velocity state is not gathers."""
    vel = latest(vel1, "vel1")
    assert_blocked(vel1, so.run_bandpass, inputs={"data": vel.id})
    assert_blocked(vel1, so.run_nmo1, inputs={"velocity": latest(vel1, "cdp_sort").id})


def test_an_unknown_input_state_is_refused(shots):
    assert_blocked(shots, so.run_agc, inputs={"data": "S999"})


def test_non_reproducible_parameters_are_rejected_and_nothing_is_stored(shots):
    """A parameter that cannot be written down cannot be replayed, so it is refused up front."""
    before = footprint(shots)
    with pytest.raises(ValidationError):
        so.execute(shots, "agc", {"window": object(), "mode": "rms"})
    assert footprint(shots) == before


def test_a_failing_computation_leaves_no_state_and_no_arrays(shots):
    before = footprint(shots)
    with pytest.raises(Exception):
        so.run_bandpass(shots, fmin=500.0, fmax=20.0)                 # not a valid band
    assert footprint(shots) == before


# ─────────────────────────────────────────────────────────────────────────
# The executors agree with the status the UI shows
# ─────────────────────────────────────────────────────────────────────────

def _executors(survey):
    poly = fk_polygon(survey)
    return {
        ("shot_gathers", None): lambda s: so.run_build_shot_gathers(s),
        ("bandpass", None): so.run_bandpass, ("agc", None): so.run_agc,
        ("despike", None): so.run_despike, ("trace_edit", None): so.run_trace_kill,
        ("mute", None): lambda s: so.run_mute(s, "flat"), ("balance", None): so.run_balance,
        ("fk", None): lambda s: so.run_fk(s, poly), ("mult_supp", None): so.run_mult_supp,
        ("decon", None): so.run_decon, ("cdp_sort", None): lambda s: so.run_cdp_sort(s, N_MIN),
        ("vel1", None): lambda s: so.run_velocity(s, "vel1", n_analysis=3),
        ("nmo1", None): so.run_nmo1, ("statics", None): so.run_statics, ("dmo", None): so.run_dmo,
        ("inv_nmo", None): so.run_inverse_nmo, ("data_reg", None): so.run_data_reg,
        ("vel2", None): lambda s: so.run_velocity(s, "vel2", n_analysis=3),
        ("nmo_final", None): so.run_nmo_final, ("stack", None): so.run_stack,
        ("rna", None): so.run_rna, ("decon_post", None): so.run_decon_post, ("tvf", None): so.run_tvf,
        ("migration", "poststack"): lambda s: so.run_migration_poststack(s, aperture_m=150.0),
        ("migration", "pstm"): lambda s: so.run_migration_pstm(s, aperture_m=150.0, cdp_stride=10),
        ("final_scale", None): so.run_final_scale,
    }


@pytest.mark.parametrize("state_name", ["survey_only", "cdp_sorted", "velocity_1"])
def test_every_blocked_stage_is_refused_by_its_executor(survey, state_name):
    """⛔ in the UI  <=>  the executor raises DependencyError. (Checked on three
    states; stages that are not blocked are not run here, only the refusals.)"""
    store = so.new_store(survey)
    if state_name != "survey_only":
        so.run_cdp_sort(store, N_MIN)
    if state_name == "velocity_1":
        so.run_velocity(store, "vel1", n_analysis=3)
    n_blocked = 0
    for (stage, variant), fn in _executors(survey).items():
        chk = so.check(store, stage, variant)
        if not chk.can_run:
            n_blocked += 1
            assert_blocked(store, fn)
    assert n_blocked >= 5


def test_ui_status_matches_for_a_mid_workflow_state(vel1):
    levels = {sid: so.check(vel1, sid, v).level for sid, v in
              [("agc", None), ("nmo1", None), ("stack", None), ("rna", None), ("statics", None)]}
    assert levels["agc"] == Level.WARNING          # shot gathers exist; band-pass advised
    assert levels["nmo1"] == Level.AVAILABLE
    assert levels["stack"] == Level.BLOCKED
    assert levels["rna"] == Level.BLOCKED
    assert levels["statics"] == Level.BLOCKED


def test_every_stage_in_the_registry_has_an_executor(survey):
    execs = {sid for sid, _ in _executors(survey)}
    assert {s.id for s in PIPELINE_STAGES} - {"upload"} == execs


# ═════════════════════════════════════════════════════════════════════════
# THE STATE MODEL
# ═════════════════════════════════════════════════════════════════════════

# ---- every operation records what it consumed and produced -------------------------------

def test_every_state_records_its_input_operation_and_parameters(shots):
    sg = latest(shots, "shot_gathers")
    bp = so.run_bandpass(shots, fmin=7.0, fmax=55.0)
    assert (bp.op, bp.kind, bp.inputs) == ("bandpass", "shot", {"data": sg.id})
    assert bp.params == {"fmin": 7.0, "fmax": 55.0, "low_slope": cfg.LOW_SLOPE_DEFAULT,
                         "high_slope": cfg.HIGH_SLOPE_DEFAULT}
    assert bp.root == sg.id and bp.version == 1 and bp.seq > sg.seq and bp.timestamp
    assert len(bp.fingerprint) == 12 and bp.fingerprint != sg.fingerprint


def test_state_ids_are_stable_unique_and_never_reused(shots):
    a = so.run_agc(shots)
    b = so.run_bandpass(shots)
    ids = [s.id for s in so.graph(shots)]
    assert len(ids) == len(set(ids)) and ids[:2] == ["S0", "S1"]
    so.discard_state(shots, b.id)
    c = so.run_mute(shots, "flat")
    assert c.id not in (a.id, b.id) and int(c.id[1:]) > int(b.id[1:])


# ---- repeated operations ----------------------------------------------------------------------------

def test_repeating_an_operation_makes_a_new_version_on_top_of_the_first(shots):
    first = so.run_agc(shots, window=500)
    chk = so.check(shots, "agc")
    assert "repeated_operation" in kinds(chk) and chk.can_run          # a ⚠, never a block
    second = so.run_agc(shots, window=250)
    assert (first.version, second.version) == (1, 2)
    assert second.inputs["data"] == first.id and second.id != first.id
    assert "Run despite" in second.notes
    assert describe_stream(so.workflow(shots), "shot") == "Raw → AGC → AGC"
    # both results are kept, each under its own key, and they differ
    assert first.keys["traces"] != second.keys["traces"]
    assert not np.array_equal(traces(shots, first), traces(shots, second))


def test_a_repeat_can_be_applied_to_the_state_before_instead(shots):
    """Re-doing AGC with new parameters, but on the *original* data rather than on its own output."""
    sg = latest(shots, "shot_gathers")
    first = so.run_agc(shots, window=500)
    again = so.run_agc(shots, window=250, inputs={"data": sg.id})
    assert again.inputs["data"] == sg.id and again.version == 2
    assert "repeated_operation" not in kinds(so.check(shots, "agc", inputs={"data": sg.id}))
    assert describe_stream(so.workflow(shots), "shot") == "Raw → AGC"        # one AGC in the current chain
    assert latest(shots, "agc").id == again.id and first.id in [s.id for s in so.graph(shots)]


# ---- branching: operate on an earlier state -------------------------------------------------------

def test_two_branches_from_the_same_state_coexist(shots):
    sg = latest(shots, "shot_gathers")
    a = so.run_agc(shots)
    b = so.run_bandpass(shots, inputs={"data": sg.id})               # a different branch off the raw gathers
    g = so.graph(shots)
    assert a.inputs["data"] == b.inputs["data"] == sg.id
    assert {d.id for d in g.dependents(sg.id)} == {a.id, b.id}
    assert g.current("shot").id == b.id                              # the one run last
    keep = traces(shots, a).copy()

    so.select_state(shots, a.id)                                     # go back to the AGC branch…
    mute = so.run_mute(shots, "flat")
    assert mute.inputs["data"] == a.id                               # …and continue from there
    np.testing.assert_array_equal(traces(shots, a), keep)            # the other branch's data is untouched
    assert finite(traces(shots, b))


# ---- memoisation: an identical computation is not done (or stored) twice -----------------------

def test_an_identical_rerun_reuses_the_state(vel1):
    nmo = so.run_nmo1(vel1)
    n, held = len(so.graph(vel1)), so.payload_bytes(vel1, nmo.id)
    again = so.run_nmo1(vel1)
    assert again.id == nmo.id and len(so.graph(vel1)) == n
    assert vel1["last_run"] == {"state": nmo.id, "op": "nmo1", "variant": "", "reused": True}
    assert so.payload_bytes(vel1, nmo.id) == held > 0               # not stored a second time


def test_a_changed_parameter_or_input_is_not_reused(vel1):
    a = so.run_nmo1(vel1, stretch_mute=0.3)
    b = so.run_nmo1(vel1, stretch_mute=0.5)
    assert a.id != b.id and a.fingerprint != b.fingerprint
    assert vel1["last_run"]["reused"] is False


def test_rerunning_an_identical_step_on_the_same_input_returns_to_that_state(shots):
    sg = latest(shots, "shot_gathers")
    bp = so.run_bandpass(shots)
    so.run_agc(shots)
    so.run_bandpass(shots, fmin=6.0, fmax=70.0)                      # moved on, on top of the AGC
    again = so.run_bandpass(shots, inputs={"data": sg.id})           # exactly the first band-pass
    assert again.id == bp.id
    assert so.graph(shots).current("shot").id == bp.id               # it is current data again
    assert shots["last_run"]["reused"] is True


def test_reusing_a_released_state_brings_it_back(shots):
    sg = latest(shots, "shot_gathers")
    bp = so.run_bandpass(shots)
    expected = traces(shots, bp).copy()
    so.release_payload(shots, bp.id)
    again = so.run_bandpass(shots, inputs={"data": sg.id})
    assert again.id == bp.id and bp.materialized
    np.testing.assert_array_equal(traces(shots, bp), expected)


def test_fingerprints_are_the_same_in_every_session(survey):
    """Identity is the lineage, not the session or the order the numbers were handed out."""
    a, b = so.new_store(survey), so.new_store(survey)
    so.run_build_shot_gathers(a)
    so.run_agc(a)                                                       # decoy: shifts b's ids
    so.run_build_shot_gathers(b); so.run_agc(b)
    so.run_bandpass(a); so.run_bandpass(b)
    assert [s.fingerprint for s in so.graph(a)] == [s.fingerprint for s in so.graph(b)]


# ---- no input copies ---------------------------------------------------------------------------------------

def test_a_new_state_adds_one_array_per_gather_and_copies_no_input(shots):
    sg = latest(shots, "shot_gathers")
    container = so.gathers_of(shots, sg)
    native = {gid: g["traces"] for gid, g in container.items()}
    offsets = {gid: g["offsets"] for gid, g in container.items()}
    before = so.payload_bytes(shots, sg.id)

    bp = so.run_bandpass(shots)
    agc = so.run_agc(shots)

    assert so.gathers_of(shots, bp) is container and so.gathers_of(shots, agc) is container
    assert len(so.containers(shots)) == 1                               # still one container
    for gid, g in container.items():
        assert g["traces"] is native[gid]                                # the root's arrays are the same objects
        assert g["offsets"] is offsets[gid]
        assert {k for k in g if "@" in k} == {bp.keys["traces"], agc.keys["traces"]}
    assert bp.keys["offsets"] == agc.keys["offsets"] == "offsets"       # offsets are shared, not duplicated
    assert so.payload_bytes(shots, sg.id) == before                      # the root's footprint did not grow
    assert so.payload_bytes(shots, bp.id) == so.payload_bytes(shots, agc.id) == before - sum(
        g["offsets"].nbytes + sum(v.nbytes for k, v in g.items() if isinstance(v, np.ndarray)
                                  and k not in ("traces", "offsets") and "@" not in k)
        for g in container.values())


def test_legacy_keys_are_references_not_copies(vel1):
    so.run_nmo1(vel1)
    stack = so.run_stack(vel1)
    cdp = latest(vel1, "cdp_sort")
    vel = latest(vel1, "vel1")
    assert vel1["gathers"] is so.containers(vel1)[cdp.root]
    assert vel1["stack_matrix"] is so.section_of(vel1, stack)
    assert vel1["Vfield_1"] is so.blob_of(vel1, vel, "field")
    assert vel1["completed_stages"] == so.workflow(vel1).completed()


def test_two_stacks_do_not_overwrite_each_other(vel1):
    nmo1 = so.run_nmo1(vel1)
    s1 = so.run_stack(vel1)
    final = so.run_nmo_final(vel1)
    s2 = so.run_stack(vel1)
    assert s2.inputs["data"] == final.id and s1.inputs["data"] == nmo1.id
    assert so.section_of(vel1, s1) is not so.section_of(vel1, s2)
    assert vel1["stack_matrix"] is so.section_of(vel1, s2)               # the alias follows the current one
    so.select_state(vel1, s1.id)
    assert vel1["stack_matrix"] is so.section_of(vel1, s1)


# ---- releasing memory and re-computing it ----------------------------------------------------------------

def test_release_then_materialize_gives_the_same_numbers(shots):
    bp = so.run_bandpass(shots)
    agc = so.run_agc(shots)
    expected = {gid: g[agc.keys["traces"]].copy() for gid, g in so.gathers_of(shots, agc).items()}

    freed = so.release_payload(shots, agc.id)
    assert freed > 0 and not agc.materialized and so.payload_bytes(shots, agc.id) == 0
    assert not any(agc.keys["traces"] in g for g in so.gathers_of(shots, bp).values())
    assert any(s["id"] == agc.id and not s["materialized"] for s in so.memory_report(shots))

    so.materialize(shots, agc.id)
    assert agc.materialized
    for gid, g in so.gathers_of(shots, agc).items():
        np.testing.assert_array_equal(g[agc.keys["traces"]], expected[gid])


def test_materializing_a_state_also_rebuilds_released_inputs(shots):
    bp = so.run_bandpass(shots)
    agc = so.run_agc(shots)
    expected = traces(shots, agc).copy()
    so.release_payload(shots, bp.id)
    so.release_payload(shots, agc.id)
    so.materialize(shots, agc.id)
    assert bp.materialized and agc.materialized
    np.testing.assert_array_equal(traces(shots, agc), expected)


def test_base_gathers_and_the_loaded_data_cannot_be_released(shots):
    for state in (latest(shots, "shot_gathers"), latest(shots, "upload")):
        assert not so.releasable(shots, state.id)
        with pytest.raises(ValidationError):
            so.release_payload(shots, state.id)


def test_a_memory_budget_releases_old_states_but_never_the_current_one(shots, monkeypatch):
    monkeypatch.setattr(cfg, "STATE_MEMORY_BUDGET_MB", 0.5)
    bp = so.run_bandpass(shots)
    agc = so.run_agc(shots)
    mute = so.run_mute(shots, "flat")
    assert not bp.materialized and not agc.materialized and mute.materialized
    assert finite(traces(shots, mute))
    # undo walks back onto released data: it is re-computed on the way
    back = so.undo(shots, "shot")
    assert back.id == agc.id and agc.materialized and finite(traces(shots, agc))


def test_the_budget_can_be_applied_by_hand(shots):
    bp = so.run_bandpass(shots)
    so.run_agc(shots)
    so.run_mute(shots, "flat")
    released = so.enforce_budget(shots, 1)
    assert bp.id in released and not bp.materialized


# ---- undo / redo ------------------------------------------------------------------------------------------------

def test_undo_and_redo_only_move_the_pointer(shots):
    a, b, c = so.run_bandpass(shots), so.run_agc(shots), so.run_mute(shots, "flat")
    g = so.graph(shots)
    n, keys = len(g), footprint(shots)[1]
    assert so.undo(shots, "shot").id == b.id and g.current("shot").id == b.id
    assert so.undo(shots, "shot").id == a.id
    assert so.redo(shots, "shot").id == b.id and so.redo(shots, "shot").id == c.id
    assert so.redo(shots, "shot") is None
    assert len(g) == n and footprint(shots)[1] == keys                    # nothing deleted, nothing recomputed
    kinds = [e.kind for e in g.actions.entries()]
    assert kinds.count("undo") == 2 and kinds.count("redo") == 2      # the pointer moves are logged


def test_undo_stops_at_the_base_gathers(shots):
    so.run_bandpass(shots)
    assert so.undo(shots, "shot").op == "shot_gathers"
    assert so.undo(shots, "shot") is None


def test_redoing_the_same_step_after_undo_reuses_it(shots):
    so.run_bandpass(shots)
    agc = so.run_agc(shots, window=400)
    so.undo(shots, "shot")
    n = len(so.graph(shots))
    again = so.run_agc(shots, window=400)
    assert again.id == agc.id and len(so.graph(shots)) == n


def test_a_new_operation_ends_the_redo_history(shots):
    so.run_bandpass(shots)
    so.run_agc(shots)
    so.undo(shots, "shot")
    assert so.graph(shots).can_redo("shot")
    so.run_despike(shots)
    assert not so.graph(shots).can_redo("shot") and so.redo(shots, "shot") is None


def test_undo_moves_the_legacy_aliases_for_derived_data(vel1):
    so.run_nmo1(vel1)
    s1 = so.run_stack(vel1)
    final = so.run_nmo_final(vel1)
    s2 = so.run_stack(vel1)
    assert vel1["stack_matrix"] is so.section_of(vel1, s2)
    # the section kind has no predecessor to step back to; but the current stack can be chosen
    so.select_state(vel1, s1.id)
    assert vel1["stack_matrix"] is so.section_of(vel1, s1) and final.id != s1.id


# ---- discarding -----------------------------------------------------------------------------------------------------

def test_discarding_a_state_removes_what_was_built_on_it_and_frees_the_arrays(shots):
    a, b, c = so.run_bandpass(shots), so.run_agc(shots), so.run_mute(shots, "flat")
    removed = so.discard_state(shots, b.id)
    assert set(removed) == {b.id, c.id}
    assert so.graph(shots).get(a.id) is a and so.graph(shots).get(b.id) is None
    leftovers = {k for g in so.gathers_of(shots, a).values() for k in g if k.endswith((f"@{b.id}", f"@{c.id}"))}
    assert not leftovers
    assert not so.workflow(shots).has("agc") and not so.workflow(shots).has("mute")
    assert so.workflow(shots).has("bandpass")
    # and the data can now be processed the other way round
    so.run_mute(shots, "flat")
    so.run_agc(shots)
    assert describe_stream(so.workflow(shots), "shot") == "Raw → Band-pass → Mute → AGC"


def test_discarding_forgets_remembered_input_choices(shots):
    bp = so.run_bandpass(shots)
    so.set_pick(shots, "agc", "data", bp.id)
    assert so.get_picks(shots, "agc") == {"data": bp.id}
    so.discard_state(shots, bp.id)
    assert so.get_picks(shots, "agc") == {}
    assert so.check(shots, "agc").can_run                                  # falls back to the current data


def test_a_remembered_input_choice_is_used_by_default(shots):
    sg = latest(shots, "shot_gathers")
    so.run_bandpass(shots)
    so.set_pick(shots, "agc", "data", sg.id)
    assert so.run_agc(shots).inputs["data"] == sg.id                      # not the (current) band-pass output


def test_loading_a_new_survey_resets_everything(vel1):
    so.run_nmo1(vel1)
    other = generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                      seed=7, source_name="other.sgy")
    so.run_load_survey(vel1, other, "other.sgy")
    assert [(s.id, s.op) for s in so.graph(vel1)] == [("S0", "upload")]
    assert vel1["gathers"] is None and vel1["Vfield_1"] is None and vel1["shot_gathers"] is None
    assert not so.containers(vel1) and not so.blobs(vel1)


def test_raw_survey_data_is_never_modified(survey):
    before = survey.raw_data.copy()
    store = so.new_store(survey)
    so.run_build_shot_gathers(store)
    so.run_agc(store); so.run_bandpass(store); so.run_mute(store, "flat")
    so.run_cdp_sort(store, N_MIN)
    np.testing.assert_array_equal(survey.raw_data, before)


# ---- stale detection through real runs ------------------------------------------------------------------

def test_rerunning_velocity_keeps_the_old_states_but_marks_them_stale(vel1):
    v1 = latest(vel1, "vel1")
    nmo = so.run_nmo1(vel1)
    stack = so.run_stack(vel1)
    assert not stale_info(so.workflow(vel1))                               # all consistent so far

    v2 = so.run_velocity(vel1, "vel1", n_analysis=4)                      # new parameters → a new state
    assert v2.id != v1.id and v2.version == 2
    for old in (v1, nmo, stack):                                         # nothing was discarded or freed
        assert so.graph(vel1).get(old.id) is old and old.materialized
    assert finite(traces(vel1, nmo, nth=0)) and finite(so.section_of(vel1, stack))

    stale = stale_info(so.workflow(vel1))
    assert nmo.id in stale and stale[nmo.id][0].kind == "replaced" and stale[nmo.id][0].source == v1.id
    assert stack.id in stale and stale[stack.id][0].kind == "inherited"
    assert v2.id not in stale

    # making the NMO current again, from the new velocities, clears the first level…
    nmo2 = so.run_nmo1(vel1)
    assert nmo2.inputs["velocity"] == v2.id and nmo2.id not in stale_info(so.workflow(vel1))
    # …and re-stacking clears the rest of the *current* data (the old stack stays listed as stale)
    stack2 = so.run_stack(vel1)
    assert stack2.id not in stale_info(so.workflow(vel1))


def test_the_dependents_of_a_state_are_found_through_any_input_role(vel1):
    v1 = latest(vel1, "vel1")
    nmo = so.run_nmo1(vel1)
    stack = so.run_stack(vel1)
    sort = latest(vel1, "cdp_sort")
    deps = {s.id for s in so.workflow(vel1).dependents(v1.id)}
    assert deps == {nmo.id, stack.id}                                      # the velocity is a side input of the NMO
    assert nmo.id in {s.id for s in so.workflow(vel1).dependents(sort.id)}


def test_undoing_marks_the_states_built_on_what_was_undone(shots):
    so.run_bandpass(shots)
    agc = so.run_agc(shots)
    sort = so.run_cdp_sort(shots, N_MIN)
    assert not stale_info(so.workflow(shots))
    so.undo(shots, "shot")                                               # the shot data goes back before the AGC
    assert so.graph(shots).current("shot").op == "bandpass"
    stale = stale_info(so.workflow(shots))
    assert list(stale) == [sort.id] and stale[sort.id][0].kind == "rolled_back"
    assert stale[sort.id][0].source == agc.id
    assert agc.materialized and so.graph(shots).get(agc.id) is agc       # the undone state is still there
    # carrying on in a different direction turns "rolled back" into "diverged"; redo is gone
    so.run_despike(shots)
    assert stale_info(so.workflow(shots))[sort.id][0].kind == "diverged"
    assert not so.graph(shots).can_redo("shot")
    # sorting what is current now makes a fresh, consistent state
    fresh = so.run_cdp_sort(shots, N_MIN)
    assert fresh.id != sort.id and fresh.id not in stale_info(so.workflow(shots))


# ---- trace-edit / velocity parameters that depend on the person's work -----------------------------------

def test_velocity_manual_picks_are_recorded_as_parameters_of_the_state(cdp, survey):
    c0 = so.cdp_list_of(cdp, latest(cdp, "cdp_sort"))[0]
    picks = np.full(survey.num_samples, 1800.0)
    auto = so.run_velocity(cdp, "vel1", n_analysis=3)
    cdp["manual_overrides_1"] = {c0: picks}
    manual = so.run_velocity(cdp, "vel1", n_analysis=3)
    assert manual.id != auto.id and manual.fingerprint != auto.fingerprint
    assert manual.params["manual_overrides"] == {str(c0): [1800.0] * survey.num_samples}
    assert auto.params["manual_overrides"] == {}
    # the same picks again → the identical state, not a third one
    assert so.run_velocity(cdp, "vel1", n_analysis=3).id == manual.id
    # the picks the person has *now* vs what the state was made with
    assert so.velocity_pick_params(cdp, "vel1") == (manual.params["manual_overrides"],
                                                    manual.params["excluded_cdps"])
    cdp["manual_overrides_1"] = {}
    assert so.velocity_pick_params(cdp, "vel1")[0] != manual.params["manual_overrides"]


def test_excluded_cdps_are_recorded_as_parameters_too(cdp):
    c0 = so.cdp_list_of(cdp, latest(cdp, "cdp_sort"))[0]
    state = so.run_velocity(cdp, "vel1", n_analysis=3, excluded_cdps={c0})
    assert state.params["excluded_cdps"] == [c0]


# ---- reproducibility ---------------------------------------------------------------------------------------------------

def test_a_recipe_replays_to_the_identical_result(shots, survey):
    so.run_bandpass(shots)
    so.run_agc(shots, window=300)
    so.run_mute(shots, "flat", t_top=0.05)
    so.run_cdp_sort(shots, N_MIN)
    so.run_velocity(shots, "vel1", n_analysis=3)
    so.run_nmo1(shots)
    stack = so.run_stack(shots)

    new, mapping = so.replay_recipe(shots, stack.id)
    new_stack = so.graph(new)[mapping[stack.id]]
    assert new_stack.fingerprint == stack.fingerprint
    np.testing.assert_array_equal(so.section_of(new, new_stack), so.section_of(shots, stack))
    # only what the stack needs was replayed (not the whole session)
    assert {s.op for s in so.graph(new)} >= {"upload", "shot_gathers", "agc", "bandpass", "mute",
                                              "cdp_sort", "vel1", "nmo1", "stack"}
    for old_id, new_id in mapping.items():
        assert so.graph(new)[new_id].fingerprint == so.graph(shots)[old_id].fingerprint


def test_a_recipe_only_contains_the_ancestors_of_its_target(shots):
    bp = so.run_bandpass(shots)
    so.run_agc(shots)
    so.run_despike(shots, inputs={"data": bp.id})                        # a branch that bp's recipe must not include
    recipe = so.graph(shots).recipe(bp.id)
    assert [s["op"] for s in recipe] == ["upload", "shot_gathers", "bandpass"]
    payload = json.loads(io_export.recipe_json_bytes(so.graph(shots), bp.id))
    assert payload["target"] == bp.id and [s["op"] for s in payload["steps"]] == [s["op"] for s in recipe]


def test_a_recipe_does_not_replay_on_a_different_survey(shots):
    bp = so.run_bandpass(shots)
    other = generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                      seed=11, source_name="other.sgy")
    with pytest.raises(GraphError):
        so.replay_recipe(shots, bp.id, survey=other)


# ─────────────────────────────────────────────────────────────────────────
# Migration: two implementations, two sets of inputs
# ─────────────────────────────────────────────────────────────────────────

def test_prestack_migration_runs_without_a_stack_or_regularisation(vel1):
    state = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)
    assert state.variant == "pstm" and state.kind == "image"
    assert state.inputs == {"data": latest(vel1, "cdp_sort").id, "velocity": latest(vel1, "vel1").id}
    assert "Regularised offsets" in state.notes
    assert finite(vel1["stack_migrated"]) and finite(vel1["pstm_image"])
    assert vel1["stack_matrix"] is None and not so.workflow(vel1).has("stack")
    section, label = so.best_available_section(vel1)
    assert label == "Migrated section" and section is vel1["stack_migrated"]


def test_prestack_migration_prefers_regularised_gathers_when_they_exist(vel1):
    reg = so.run_data_reg(vel1)
    vel2 = so.run_velocity(vel1, "vel2", n_analysis=3)
    state = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)
    assert state.inputs == {"data": reg.id, "velocity": vel2.id}
    assert state.notes == ""


def test_the_two_migrations_are_separate_states_and_record_which_ran(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    post = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert post.variant == "poststack" and post.info["domain"] == "post-stack"
    pre = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)      # does NOT replace it
    assert pre.variant == "pstm" and pre.params["cdp_stride"] == 10
    assert so.graph(vel1).get(post.id) is post and post.materialized
    assert so.graph(vel1).current("image").id == pre.id                      # the newest is current…
    so.select_state(vel1, post.id)                                           # …and either can be chosen
    assert vel1["stack_migrated"] is so.section_of(vel1, post)


def test_migration_velocity_is_pass_two_when_available(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    vel2 = so.run_velocity(vel1, "vel2", n_analysis=3)      # built on the CDP gathers, after the stack
    state = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert state.inputs["velocity"] == vel2.id


def test_a_velocity_field_from_other_gathers_is_refused_for_migration(vel1):
    """Velocities are laid out on one set of CDP gathers; migrating a section of another is ⛔."""
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    other = so.run_cdp_sort(vel1, N_MIN + 1)                                 # a second CDP sort (new root)
    assert other.root != latest(vel1, "vel1").root
    err = assert_blocked(vel1, so.run_migration_pstm, aperture_m=150.0, cdp_stride=10)
    assert "velocity" in str(err).lower()


def test_final_scaling_works_on_an_unmigrated_section_with_a_note(vel1):
    so.run_nmo1(vel1)
    stack = so.run_stack(vel1)
    state = so.run_final_scale(vel1)
    assert state.inputs["data"] == stack.id and "Migration has not been applied" in state.notes
    assert finite(vel1["stack_final"])
    assert so.best_available_section(vel1)[1] == "Final processed section"


def test_best_available_section_priority_and_validity(vel1):
    assert so.best_available_section(vel1) == (None, None)
    so.run_nmo1(vel1)
    stack = so.run_stack(vel1)
    assert so.best_available_section(vel1)[1] == "Stack"
    so.run_rna(vel1)
    assert "random noise" in so.best_available_section(vel1)[1]
    so.run_migration_poststack(vel1, aperture_m=150.0)
    assert so.best_available_section(vel1)[1] == "Migrated section"
    so.run_final_scale(vel1)
    assert so.best_available_section(vel1)[1] == "Final processed section"
    # discarding the stack drops everything built on it — no orphaned section can be exported
    so.discard_state(vel1, stack.id)
    assert so.best_available_section(vel1) == (None, None)
    assert vel1["stack_matrix"] is None and vel1["stack_final"] is None


# ─────────────────────────────────────────────────────────────────────────
# History and export: creation order, with the states consumed
# ─────────────────────────────────────────────────────────────────────────

def test_history_and_exports_list_steps_in_creation_order_with_inputs(shots):
    agc = so.run_agc(shots)
    bp = so.run_bandpass(shots)
    so.run_mute(shots, "flat")
    g = so.graph(shots)
    assert [s.op for s in g] == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    assert [s.seq for s in g] == sorted(s.seq for s in g)

    report = io_export.processing_report_text({"Source": "synthetic"}, g)
    log_text = report[report.index("ACTION LOG"):report.index("DATA STATES")]
    assert log_text.index("Amplitude Recovery") < log_text.index("Band-Pass Filter") < log_text.index("Mute")
    states_text = report[report.index("DATA STATES"):]
    assert f"consumed: data={agc.id}" in states_text and "fingerprint:" in states_text

    payload = json.loads(io_export.params_json_bytes(g))
    assert [s["op"] for s in payload["states"]] == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    exported_bp = payload["states"][3]
    assert exported_bp["id"] == bp.id and exported_bp["inputs"] == {"data": agc.id}
    assert exported_bp["fingerprint"] == bp.fingerprint and exported_bp["params"]["fmax"] == cfg.FMAX_DEFAULT
    assert payload["current"]["shot"] == latest(shots, "mute").id


def test_the_report_for_one_state_lists_only_what_produced_it(shots):
    bp = so.run_bandpass(shots)
    so.run_agc(shots)
    report = io_export.processing_report_text({}, so.graph(shots), state_id=bp.id)
    assert "Band-Pass Filter" in report and "Amplitude Recovery" not in report
    assert report.count("Band-Pass Filter") >= 1 and "ACTION LOG — the operations that produced" in report


# ─────────────────────────────────────────────────────────────────────────
# The processing history is an immutable, chronological action log
# ─────────────────────────────────────────────────────────────────────────
#
# History (what was done, in order) is kept apart from the current state (what the data is now)
# and from the recommended workflow (what is usually done). Pure properties of the log are in
# tests/test_action_log.py; these run real operations through it.

from modules.dependencies import current_processing_state          # noqa: E402
from modules.pipeline import STAGE_INDEX                            # noqa: E402

SHOT_RUNNERS = {
    "agc": lambda s, **kw: so.run_agc(s, **kw),
    "bandpass": lambda s, **kw: so.run_bandpass(s, **kw),
    "mute": lambda s, **kw: so.run_mute(s, "flat", **kw),
    "balance": lambda s, **kw: so.run_balance(s, **kw),
    "despike": lambda s, **kw: so.run_despike(s, **kw),
}


def alog(store):
    return so.action_log(store)


def op_entries(store):
    """The processing operations in the history, in the order recorded (failures included)."""
    return alog(store).operations()


def run_in_order(store, sequence):
    for stage in sequence:
        SHOT_RUNNERS[stage](store)


def test_the_history_is_exactly_the_order_the_user_ran_the_operations(shots, survey):
    """AGC → Band-pass → Mute → F-K → Band-pass again. Nothing is moved to the 'recommended' place."""
    so.run_agc(shots)
    so.run_bandpass(shots)
    so.run_mute(shots, "flat")
    so.run_fk(shots, fk_polygon(survey))
    so.run_bandpass(shots)

    entries = alog(shots).entries()
    assert [e.action for e in entries] == ["upload", "shot_gathers", "agc", "bandpass", "mute", "fk",
                                           "bandpass"]
    assert [e.n for e in entries] == list(range(1, 8))
    assert entries[3].label == entries[6].label and entries[3].label != entries[2].label
    performed = [e.action for e in entries][2:]
    assert performed == ["agc", "bandpass", "mute", "fk", "bandpass"]
    # …which is NOT the recommended seismic order (band-pass is recommended before AGC and mute)
    recommended = [STAGE_INDEX[a] for a in performed]
    assert recommended != sorted(recommended)
    # each entry says which state it read and which it made, and they chain in the order performed
    ops_ = [e for e in entries if e.action in {"agc", "bandpass", "mute", "fk"}]
    for before, after in zip(ops_, ops_[1:]):
        assert after.input_state == before.output
    assert [e.output for e in entries] == [s.id for s in so.graph(shots)]       # one state per action here
    assert alog(shots).verify() == []


@pytest.mark.parametrize("sequence", [
    ["mute", "agc", "bandpass", "bandpass", "balance", "agc"],      # repeats, reversed recommendation
    ["balance", "mute", "bandpass", "despike"],
    ["bandpass", "agc", "mute", "balance"],                         # the textbook order
])
def test_any_order_the_user_chooses_is_the_order_recorded(shots, sequence):
    run_in_order(shots, sequence)
    assert [e.action for e in op_entries(shots)][2:] == sequence
    assert [e.n for e in alog(shots).entries()] == list(range(1, len(sequence) + 3))
    # the display setting is only a view of that
    assert [e.action for e in alog(shots).view("oldest")][2:] == sequence
    assert [e.action for e in alog(shots).view("newest")][:-2] == sequence[::-1]
    assert [e.action for e in alog(shots).entries()][2:] == sequence       # …and did not touch the log


def test_the_same_operation_run_twice_is_two_entries_and_two_states(shots):
    a = so.run_bandpass(shots)
    b = so.run_bandpass(shots)                                              # reads a's output: a new state
    runs = [e for e in op_entries(shots) if e.action == "bandpass"]
    assert [e.output for e in runs] == [a.id, b.id] and a.id != b.id
    assert runs[1].input_state == a.id and runs[0].n < runs[1].n
    assert runs[0].output == a.id                                            # the first entry is untouched


def test_an_identical_rerun_is_still_its_own_entry_even_though_nothing_is_recomputed(shots):
    sg = latest(shots, "shot_gathers")
    first = so.run_agc(shots, inputs={"data": sg.id})
    n_states, n_entries = len(so.graph(shots)), len(alog(shots))
    again = so.run_agc(shots, inputs={"data": sg.id})                       # same op, params and input
    assert again.id == first.id and len(so.graph(shots)) == n_states        # no new state …
    assert len(alog(shots)) == n_entries + 1                                # … but a new history entry
    e1, e2 = [e for e in op_entries(shots) if e.action == "agc"]
    assert not e1.reused and e2.reused and e1.output == e2.output == first.id
    assert e1.n < e2.n and e2.status == "success"


def test_blocked_and_invalid_attempts_are_in_the_history_when_they_happened(shots):
    so.run_agc(shots)
    states_before = len(so.graph(shots))
    with pytest.raises(DependencyError):
        so.run_stack(shots)                                                 # no NMO-corrected gathers
    so.run_bandpass(shots)
    with pytest.raises(ValidationError):
        so.run_mute(shots, "bogus")
    so.run_mute(shots, "flat")

    seq = [(e.action, e.status, e.failure_kind) for e in op_entries(shots)][2:]
    assert seq == [("agc", "success", ""), ("stack", "failure", "blocked"), ("bandpass", "success", ""),
                   ("mute", "failure", "invalid"), ("mute", "success", "")]
    blocked = op_entries(shots)[3]
    assert blocked.output is None and "NMO-corrected" in blocked.error and not blocked.ok
    assert len(so.graph(shots)) == states_before + 2                        # only the two successes made states
    assert [e.n for e in alog(shots).entries()] == list(range(1, 8))        # no gaps for the failures


def test_a_failure_does_not_consume_a_state_id(shots):
    so.run_agc(shots)
    last_id = list(so.graph(shots))[-1].id
    with pytest.raises(DependencyError):
        so.run_stack(shots)
    nxt = so.run_bandpass(shots)
    assert int(nxt.id[1:]) == int(last_id[1:]) + 1


def test_the_warnings_a_run_went_ahead_despite_are_recorded_with_the_entry(shots):
    agc = so.run_agc(shots)                                                 # band-pass not applied: ⚠
    bp = so.run_bandpass(shots)                                             # AGC already applied: ⚠
    mute = so.run_mute(shots, "flat")                                       # nothing to warn about
    e_agc, e_bp, e_mute = [e for e in op_entries(shots) if e.action in ("agc", "bandpass", "mute")]
    assert e_agc.warnings and "Band-pass" in e_agc.warnings[0]
    assert e_bp.warnings and "AGC" in e_bp.warnings[0]
    assert e_mute.warnings == ()
    assert "Run despite" in agc.notes and "Run despite" in bp.notes and not mute.notes


def test_every_successful_entry_carries_input_output_parameters_and_qc(shots):
    agc = so.run_agc(shots, window=300)
    e = [x for x in op_entries(shots) if x.action == "agc"][0]
    d = e.to_dict()
    assert d["n"] == e.n and d["timestamp"] and d["label"]                    # number, time, name
    assert d["params"] == agc.params and d["params"]["window"] == 300         # complete parameters
    assert d["input_state"] == latest(shots, "shot_gathers").id and d["output_state"] == agc.id
    assert d["status"] == "success" and d["failure_kind"] == ""
    qc = d["qc"]
    assert qc["rms_in"] > 0 and qc["rms_out"] > 0 and qc["non_finite"] == 0  # QC summary
    assert d["detail"] and e.input_state == agc.inputs["data"]


def test_the_qc_summary_reports_non_finite_samples(shots):
    agc = so.run_agc(shots)
    assert so.qc_summary(shots, agc)["non_finite"] == 0
    gather = next(iter(so.gathers_of(shots, agc).values()))
    gather[agc.keys["traces"]][0, 0] = np.nan                               # corrupt one sample on purpose
    qc = so.qc_summary(shots, agc)
    assert qc["non_finite"] == 1
    from modules.action_log import format_qc
    assert "⚠ 1 non-finite sample(s)" in format_qc(qc)


def test_the_upload_and_gather_building_are_the_first_actions_of_a_dataset(shots):
    first, second = alog(shots).entries()[:2]
    assert (first.action, first.output, first.input_state) == ("upload", "S0", None)
    assert (second.action, second.input_state, second.output) == ("shot_gathers", "S0", "S1")
    assert first.qc["traces"] > 0 and first.qc["samples"] > 0 and first.qc["dt_ms"] > 0


def test_undo_redo_and_make_current_are_logged_in_sequence(shots):
    a, b = so.run_bandpass(shots), so.run_agc(shots)
    so.undo(shots, "shot")
    so.redo(shots, "shot")
    so.select_state(shots, a.id)
    tail = alog(shots).entries()[-3:]
    assert [(e.kind, e.input_state, e.output) for e in tail] == [
        ("undo", b.id, a.id), ("redo", a.id, b.id), ("select", b.id, a.id)]
    assert all(e.status == "success" and not e.is_operation for e in tail)


def test_navigation_that_changes_nothing_is_not_logged(shots):
    so.run_bandpass(shots)
    so.undo(shots, "shot")
    n = len(alog(shots))
    assert so.undo(shots, "shot") is None                                   # nothing left to undo
    assert so.redo(shots, "cdp") is None
    assert len(alog(shots)) == n


def test_discarding_a_state_keeps_the_entries_that_made_it_and_says_what_was_removed(shots):
    a, b, c = so.run_bandpass(shots), so.run_agc(shots), so.run_mute(shots, "flat")
    before = alog(shots).entries()
    removed = so.discard_state(shots, b.id)
    log = alog(shots)
    assert log.entries()[:len(before)] == before                            # nothing earlier changed or vanished
    gone = log[len(before) + 1]
    assert gone.kind == "discard" and set(gone.affected) == set(removed) == {b.id, c.id}
    assert [log.discarded_by(e) is not None for e in before if e.output in (a.id, b.id, c.id)] == [
        False, True, True]
    assert b.id not in {s.id for s in so.graph(shots)}                      # the state is gone …
    assert any(e.output == b.id for e in before)                            # … its history entry is not
    assert log.verify() == []


def test_freeing_memory_is_logged_but_the_budget_and_recomputation_are_system_events(shots):
    bp = so.run_bandpass(shots)
    agc = so.run_agc(shots)
    so.run_mute(shots, "flat")
    n = len(alog(shots))
    so.enforce_budget(shots, 1)                                             # the system frees old states
    assert len(alog(shots)) == n and so.graph(shots).system_events          # not the person's doing
    so.materialize(shots, bp.id)                                            # re-computation on demand
    assert len(alog(shots)) == n
    so.release_payload(shots, bp.id)                                        # the person pressed "Free memory"
    last = alog(shots).entries()[-1]
    assert last.kind == "release" and last.input_state == bp.id and len(alog(shots)) == n + 1
    assert last.qc["freed_mb"] >= 0 and agc.id != bp.id


def test_the_history_is_not_the_current_state_and_is_not_the_recommended_workflow(shots):
    so.run_agc(shots)
    so.run_bandpass(shots)
    so.undo(shots, "shot")                                                  # current data goes back to AGC
    wf = so.workflow(shots)
    rows = {r["kind"]: r for r in current_processing_state(wf)}
    assert rows["shot"]["applied"] == "Raw → AGC"                           # where the data IS
    performed = [e.action for e in op_entries(shots)][2:]
    assert performed == ["agc", "bandpass"]                                 # what was DONE (band-pass included)
    assert alog(shots).entries()[-1].kind == "undo"                         # and the undo that moved us back
    # the recommendation is a third, independent thing: it puts band-pass first
    assert STAGE_INDEX["bandpass"] < STAGE_INDEX["agc"]


def test_the_current_state_follows_the_pointer_while_the_history_only_grows(shots):
    run_in_order(shots, ["bandpass", "agc", "mute"])
    sizes = [len(alog(shots))]
    for move in (lambda: so.undo(shots, "shot"), lambda: so.undo(shots, "shot"),
                 lambda: so.redo(shots, "shot")):
        move()
        sizes.append(len(alog(shots)))
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)         # strictly growing
    now = {r["kind"]: r["applied"] for r in current_processing_state(so.workflow(shots))}
    assert now["shot"] == "Raw → Band-pass → AGC"


def test_loading_a_new_survey_starts_a_new_dataset_but_keeps_the_earlier_history(shots, survey):
    run_in_order(shots, ["agc", "bandpass"])
    earlier = alog(shots).entries()
    other = generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                      seed=7, source_name="other.sgy")
    so.run_load_survey(shots, other, "other.sgy")
    log = alog(shots)
    assert log.entries()[:len(earlier)] == earlier                          # the old dataset's history survives
    assert [(s.id, s.op) for s in so.graph(shots)] == [("S0", "upload")]    # while its states are gone
    assert log.entries()[-1].action == "upload" and log.entries()[-1].session == 2
    assert [e.n for e in log.entries()] == list(range(1, len(earlier) + 2))
    assert log.view(session=log.session)[0].n == len(earlier) + 1
    so.run_build_shot_gathers(shots)
    so.run_agc(shots)
    assert [e.action for e in log.view(session=2)] == ["upload", "shot_gathers", "agc"]
    assert log.verify() == []


def test_resetting_the_store_does_not_erase_the_log(shots):
    run_in_order(shots, ["agc"])
    n = len(alog(shots))
    so.reset_store(shots)
    assert len(alog(shots)) == n and alog(shots).verify() == []


def test_a_replay_in_a_new_store_logs_its_own_history_in_recipe_order(shots):
    run_in_order(shots, ["bandpass", "agc", "mute"])
    cdp_state = so.run_cdp_sort(shots, N_MIN)
    original = alog(shots).entries()
    new, _ = so.replay_recipe(shots, cdp_state.id)
    replayed = [e.action for e in alog(new).entries() if e.is_operation]
    assert replayed == [s["op"] for s in so.graph(shots).recipe(cdp_state.id)]
    assert replayed[-3:] == ["agc", "mute", "cdp_sort"] and alog(new).verify() == []
    assert alog(shots).entries() == original                                # the original log is untouched


def test_history_order_follows_execution_not_state_ids_or_lineage(shots):
    """Branching off an earlier state: the entry for the branch comes when it was RUN."""
    sg = latest(shots, "shot_gathers")
    a = so.run_agc(shots)
    b = so.run_bandpass(shots)                                              # on top of AGC
    c = so.run_bandpass(shots, inputs={"data": sg.id})                      # branch off the base gathers, later
    e_b, e_c = [e for e in op_entries(shots) if e.action == "bandpass"]
    assert e_b.n < e_c.n and (e_b.input_state, e_c.input_state) == (a.id, sg.id)
    assert [e.output for e in op_entries(shots)][2:] == [a.id, b.id, c.id]


def test_the_report_lists_the_action_log_chronologically_even_when_states_are_branched(shots):
    sg = latest(shots, "shot_gathers")
    so.run_agc(shots)
    so.run_bandpass(shots, inputs={"data": sg.id})
    so.run_mute(shots, "flat")
    so.run_bandpass(shots)
    report = io_export.processing_report_text({"Source": "synthetic"}, so.graph(shots))
    head, rest = report.index("CURRENT PROCESSING STATE"), report.index("ACTION LOG")
    assert head < rest < report.index("DATA STATES")                       # three sections, kept apart
    log_text = report[rest:report.index("DATA STATES")]
    marks = [log_text.index(f"#{e.n} ") for e in alog(shots).entries()]
    assert marks == sorted(marks) and log_text.count("Band-Pass Filter") == 2
    assert "input: data=" in log_text and "parameters:" in log_text and "QC: RMS" in log_text


def test_the_json_export_contains_the_action_log_in_execution_order(shots):
    run_in_order(shots, ["agc", "bandpass", "bandpass"])
    payload = json.loads(io_export.params_json_bytes(so.graph(shots)))
    assert [a["action"] for a in payload["actions"]][2:] == ["agc", "bandpass", "bandpass"]
    assert [a["n"] for a in payload["actions"]] == list(range(1, 6))
    assert payload["action_log_problems"] == []
    assert payload["actions"][3]["output_state"] != payload["actions"][4]["output_state"]
    assert "events" not in payload                                          # the old event list is gone


def test_the_csv_export_matches_the_log_row_for_row(shots):
    import csv
    import io
    run_in_order(shots, ["mute", "agc"])
    rows = list(csv.DictReader(io.StringIO(io_export.action_log_csv_bytes(alog(shots)).decode())))
    assert [r["operation"] for r in rows] == [e.label for e in alog(shots).entries()]
    assert [int(r["action_number"]) for r in rows] == [e.n for e in alog(shots).entries()]
    assert rows[-1]["input_state"] and rows[-1]["output_state"] and rows[-1]["result"] == "✓ success"
