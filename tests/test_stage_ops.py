"""
tests/test_stage_ops.py — non-linear processing through the real maths
==========================================================================
`modules/stage_ops.py` holds one executor per stage; the Streamlit pages
and these tests call the same functions. Here a small synthetic survey is
processed in orders the old linear pipeline refused — skipping optional
stages, applying them out of the textbook order — and the genuine
dependencies are shown to still hold.

    pytest tests/test_stage_ops.py -v

Layout
  * non-standard orders that must work (and their numerical meaning)
  * skipped optional stages
  * genuine dependency failures  (⛔ -> DependencyError, nothing changes)
  * re-running a stage: invalidation, freed data
  * the two migration implementations
  * history / export in execution order
  * the executors agree with the dependency engine
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from modules import filtering, preprocessing, segy_io
from modules import stage_ops as so
from modules.dependencies import Level, describe_stream
from modules.pipeline import PIPELINE_STAGES
from modules.synthetic_data import generate_synthetic_survey
from modules.validation import DependencyError
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
    so.run_cdp_sort(shots)
    return shots


@pytest.fixture
def vel1(cdp):
    so.run_velocity(cdp, "vel1", n_analysis=3)
    return cdp


def fk_polygon(survey):
    return filtering.default_fk_polygon(survey.sample_interval, survey.group_interval)


def hist_ids(store):
    return [h.stage_id for h in sorted(store["history"], key=lambda e: e.seq)]


def finite(a):
    return a is not None and np.isfinite(a).all()


# ─────────────────────────────────────────────────────────────────────────
# Non-standard orders (the two examples from the task, through real maths)
# ─────────────────────────────────────────────────────────────────────────

def test_raw_shot_agc_bandpass_mute_then_cdp_sort(shots, survey):
    """Raw → Shot Gathers → AGC → Band-pass → Mute → CDP Sort."""
    agc = so.run_agc(shots)
    bp = so.run_bandpass(shots)
    mute = so.run_mute(shots, "flat", t_top=0.05)
    sort = so.run_cdp_sort(shots, N_MIN)

    assert hist_ids(shots)[-4:] == ["agc", "bandpass", "mute", "cdp_sort"]
    assert (agc.consumed["data"], bp.consumed["data"], mute.consumed["data"], sort.consumed["data"]) == \
           ("shot_gathers", "agc", "bandpass", "mute")
    assert describe_stream(so.workflow(shots), "shot") == "Raw → AGC → Band-pass → Mute"
    # band-pass ran on the AGC output (not on raw), and mute on the band-passed data
    assert (agc.output_key, bp.input_key, bp.output_key, mute.input_key) == \
           ("traces_agc", "traces_agc", "traces_bp", "traces_bp")
    # the warnings were advice, and were kept in the history
    assert "Run despite" in agc.notes and "Band-pass" in agc.notes
    assert "Run despite" in bp.notes
    assert shots["gathers"] and finite(next(iter(shots["gathers"].values()))["traces"])


def test_raw_shot_bandpass_fk_then_cdp_sort(shots, survey):
    """Raw → Shot Gathers → Band-pass → F-K → CDP Sort (AGC, mute, decon… all skipped)."""
    so.run_bandpass(shots)
    so.run_fk(shots, fk_polygon(survey))
    sort = so.run_cdp_sort(shots, N_MIN)
    assert sort.consumed["data"] == "fk" and sort.input_key == "traces_fk_filtered"
    assert sort.notes == ""                                        # not even a ⚠
    assert describe_stream(so.workflow(shots), "shot") == "Raw → Band-pass → F-K"


def test_executors_apply_the_same_maths_as_calling_the_modules_directly(shots, survey):
    """Band-pass then AGC through the executors == the module calls on a shot."""
    so.run_bandpass(shots, fmin=cfg.FMIN_DEFAULT, fmax=cfg.FMAX_DEFAULT)
    so.run_agc(shots, window=cfg.AGC_WINDOW_DEFAULT, mode=cfg.AGC_MODE_DEFAULT)
    sid = list(shots["shot_gathers"])[5]
    raw = shots["shot_gathers"][sid]["traces"]
    h = filtering.design_bandpass(cfg.FMIN_DEFAULT, cfg.FMAX_DEFAULT, survey.sample_interval,
                                  survey.num_samples, cfg.LOW_SLOPE_DEFAULT, cfg.HIGH_SLOPE_DEFAULT)
    expected_bp = filtering.apply_time_domain_conv(raw, h)
    expected_agc = preprocessing.agc(expected_bp, window=cfg.AGC_WINDOW_DEFAULT, mode=cfg.AGC_MODE_DEFAULT)
    np.testing.assert_array_equal(shots["shot_gathers"][sid]["traces_bp"], expected_bp)
    np.testing.assert_array_equal(shots["shot_gathers"][sid]["traces_agc"], expected_agc)


def test_processing_order_is_real_agc_then_bandpass_differs_from_bandpass_then_agc(survey):
    """The recommended order is a recommendation because it *matters*: the
    two orders give different data, and each is recorded for what it is."""
    a = so.new_store(survey); so.run_build_shot_gathers(a)
    so.run_agc(a); so.run_bandpass(a)                      # AGC → BP
    b = so.new_store(survey); so.run_build_shot_gathers(b)
    so.run_bandpass(b); so.run_agc(b)                      # BP → AGC
    sid = list(a["shot_gathers"])[5]
    out_a = a["shot_gathers"][sid]["traces_bp"]
    out_b = b["shot_gathers"][sid]["traces_agc"]
    assert not np.allclose(out_a, out_b)
    assert describe_stream(so.workflow(a), "shot") == "Raw → AGC → Band-pass"
    assert describe_stream(so.workflow(b), "shot") == "Raw → Band-pass → AGC"


def test_shot_operators_chain_whatever_the_order(shots, survey):
    """Every shot operator in an arbitrary, non-textbook order: each reads the
    previous output, nothing is blocked, results stay finite."""
    order = [("balance", lambda s: so.run_balance(s)),
             ("decon", lambda s: so.run_decon(s)),
             ("mute", lambda s: so.run_mute(s, "flat", t_top=0.05)),
             ("despike", lambda s: so.run_despike(s)),
             ("mult_supp", lambda s: so.run_mult_supp(s, 30, 10)),
             ("bandpass", lambda s: so.run_bandpass(s)),
             ("fk", lambda s: so.run_fk(shots, fk_polygon(survey))),
             ("agc", lambda s: so.run_agc(s)),
             ("trace_edit", lambda s: so.run_trace_kill(s))]
    previous = "shot_gathers"
    for stage, fn in order:
        entry = fn(shots)
        assert entry.consumed["data"] == previous, stage
        previous = stage
    last = shots["shot_gathers"][list(shots["shot_gathers"])[3]]
    assert finite(last["traces_edited"])
    so.run_cdp_sort(shots, N_MIN)
    assert finite(next(iter(shots["gathers"].values()))["traces"])


def test_cdp_gathers_carry_the_conditioning_that_was_applied(shots, survey):
    """The sorted CDP traces equal the head of the shot data, re-grouped."""
    so.run_bandpass(shots)
    so.run_cdp_sort(shots, N_MIN)
    flat = segy_io.scatter_to_flat(shots["shot_gathers"], "traces_bp", survey.num_traces, survey.num_samples)
    expected = segy_io.build_cdp_gathers(survey, n_min_traces=N_MIN, data_override=flat)
    assert list(expected) == shots["cdp_list"]
    for c in list(expected)[:5]:
        np.testing.assert_array_equal(shots["gathers"][c]["traces"], expected[c]["traces"])


# ─────────────────────────────────────────────────────────────────────────
# Skipped optional stages
# ─────────────────────────────────────────────────────────────────────────

def test_cdp_sort_without_shot_gathers_sorts_the_raw_traces(store, survey):
    entry = so.run_cdp_sort(store, N_MIN)
    assert entry.consumed["data"] == "upload"
    expected = segy_io.build_cdp_gathers(survey, n_min_traces=N_MIN)
    assert list(expected) == store["cdp_list"]
    c = store["cdp_list"][10]
    np.testing.assert_array_equal(store["gathers"][c]["traces"], expected[c]["traces"])


def test_skipping_every_optional_stage_still_reaches_a_final_section(cdp):
    """CDP sort → velocity 1 → NMO 1 → stack → migration → final scaling.
    No conditioning, statics, DMO, regularisation, pass-2 velocity, or post-stack filters."""
    so.run_velocity(cdp, "vel1", n_analysis=3)
    so.run_nmo1(cdp)
    so.run_stack(cdp)
    mig = so.run_migration_poststack(cdp, aperture_m=150.0)
    fin = so.run_final_scale(cdp)
    assert finite(cdp["stack_matrix"]) and finite(cdp["stack_migrated"]) and finite(cdp["stack_final"])
    assert mig.consumed == {"data": "stack", "velocity": "vel1"}
    assert fin.consumed["data"] == "migration"
    for stage in ("shot_gathers", "cdp_sort", "vel1", "nmo1", "stack", "migration", "final_scale"):
        assert so.workflow(cdp).has(stage)
    for skipped in ("bandpass", "statics", "dmo", "inv_nmo", "data_reg", "vel2", "nmo_final", "rna",
                    "decon_post", "tvf"):
        assert not so.workflow(cdp).has(skipped)
    assert so.best_available_section(cdp)[1] == "Final processed section"


def test_stack_can_use_pass_one_nmo_and_says_so(vel1):
    so.run_nmo1(vel1)
    chk = so.check(vel1, "stack")
    assert chk.level == Level.WARNING
    entry = so.run_stack(vel1)
    assert entry.consumed["data"] == "nmo1" and "Final NMO has not been applied" in entry.notes


def test_final_nmo_needs_only_pass_one_velocity(vel1):
    entry = so.run_nmo_final(vel1)
    assert entry.consumed["velocity"] == "vel1"
    assert finite(next(iter(vel1["gathers"].values()))["traces_nmo_final"])


def test_dmo_runs_without_nmo_pass_one_or_statics(vel1):
    entry = so.run_dmo(vel1)
    assert entry.consumed == {"data": "cdp_sort", "velocity": "vel1"}
    assert "statics" in entry.notes.lower() and "commonly applied before DMO" in entry.notes
    assert finite(next(iter(vel1["gathers"].values()))["traces_dmo"])


def test_post_stack_operators_apply_in_any_order(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    so.run_tvf(vel1)                         # TVF first…
    rna = so.run_rna(vel1)                   # …then RNA — on the TVF output, not on the stack
    dec = so.run_decon_post(vel1)
    assert (rna.consumed["data"], dec.consumed["data"]) == ("tvf", "rna")
    assert rna.input_key == "stack_tvf" and dec.input_key == "stack_rna"
    assert describe_stream(so.workflow(vel1), "section") == "Stack → TVF → RNA → Post-stack decon"
    assert so.best_available_section(vel1)[1].startswith("Post-stack")
    assert finite(vel1["stack_decon_post"])


def test_trace_kill_finds_the_same_traces_whether_or_not_agc_ran_first(survey):
    """AGC equalises trace energy, which would blind an energy-based dead/noisy
    detector. The kill step measures amplitude-preserving data, so the number
    of traces killed does not depend on whether AGC was applied before it."""
    a = so.new_store(survey); so.run_build_shot_gathers(a)
    kill_first = so.run_trace_kill(a)
    b = so.new_store(survey); so.run_build_shot_gathers(b)
    so.run_agc(b)
    kill_after_agc = so.run_trace_kill(b)
    assert kill_first.params["traces_killed"] > 0
    assert kill_after_agc.params["traces_killed"] == kill_first.params["traces_killed"]
    assert kill_after_agc.params["detected_on"] != "AGC"        # measured upstream of the gain


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
    """The executor refuses with a DependencyError and leaves no trace."""
    history_before = [(h.stage_id, h.seq) for h in store["history"]]
    with pytest.raises(DependencyError) as exc:
        fn(store, *args, **kwargs)
    assert [(h.stage_id, h.seq) for h in store["history"]] == history_before
    return exc.value


def test_velocity_analysis_without_cdp_gathers_is_refused(shots):
    err = assert_blocked(shots, so.run_velocity, "vel1")
    assert err.stage_id == "vel1" and "CDP gathers" in str(err) and "CDP Sort" in str(err)
    assert shots["Vfield_1"] is None


def test_nmo_without_velocity_is_refused(cdp):
    for fn in (so.run_nmo1, so.run_nmo_final, so.run_dmo):
        err = assert_blocked(cdp, fn)
        assert "velocities" in str(err), fn.__name__
    assert cdp["gathers"][cdp["cdp_list"][0]].get("traces_nmo1") is None


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
    so.run_data_reg(vel1)
    err = assert_blocked(vel1, so.run_statics)
    assert "shot and receiver" in str(err)
    # …but statics before regularisation is fine, and regularisation can follow
    so.invalidate_stage(vel1, "data_reg")
    so.run_statics(vel1)
    assert so.run_data_reg(vel1).consumed["data"] == "statics"


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
    entry = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert "Run despite" in entry.notes and "random noise" in entry.notes.lower()
    assert finite(vel1["stack_migrated"])


def test_prestack_migration_needs_gathers_and_a_velocity_field(shots, store):
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


def test_blocked_stage_consumes_nothing_and_stores_nothing(cdp):
    before = {k for k, v in cdp.items() if v is not None}
    assert_blocked(cdp, so.run_stack)
    assert {k for k, v in cdp.items() if v is not None} == before


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


# ─────────────────────────────────────────────────────────────────────────
# Re-running: invalidate exactly the dependents, free exactly their data
# ─────────────────────────────────────────────────────────────────────────

def test_rerunning_velocity_discards_nmo_and_stack_and_frees_their_data(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    g = vel1["gathers"][vel1["cdp_list"][0]]
    assert "traces_nmo1" in g and vel1["stack_matrix"] is not None
    so.run_velocity(vel1, "vel1", n_analysis=3)
    assert not so.workflow(vel1).has("nmo1") and not so.workflow(vel1).has("stack")
    assert "traces_nmo1" not in g
    assert vel1["stack_matrix"] is None and vel1["fold_matrix"] is None
    assert so.workflow(vel1).has("cdp_sort") and so.workflow(vel1).has("vel1")
    assert vel1["completed_stages"] == so.workflow(vel1).completed()


def test_rerunning_nmo_pass_one_keeps_dmo_which_did_not_use_it(vel1):
    so.run_dmo(vel1)
    so.run_nmo1(vel1)
    g = vel1["gathers"][vel1["cdp_list"][0]]
    so.run_nmo1(vel1)                      # again
    assert so.workflow(vel1).has("dmo") and "traces_dmo" in g


def test_rerunning_a_shot_step_discards_the_cdp_gathers_built_from_it(shots):
    so.run_bandpass(shots)
    so.run_agc(shots)
    so.run_cdp_sort(shots, N_MIN)
    so.run_bandpass(shots, fmin=6.0, fmax=70.0)
    assert not so.workflow(shots).has("agc") and not so.workflow(shots).has("cdp_sort")
    assert shots["gathers"] is None and shots["cdp_list"] in (None, [])
    sg = next(iter(shots["shot_gathers"].values()))
    assert "traces_agc" not in sg and "traces_bp" in sg


def test_a_rerun_step_reads_the_best_remaining_input_not_its_own_output(shots):
    so.run_bandpass(shots)
    so.run_agc(shots)
    # re-running AGC must read the band-pass output, not (its own) AGC output
    entry = so.run_agc(shots, window=300)
    assert entry.consumed["data"] == "bandpass" and entry.input_key == "traces_bp"


def test_discarding_a_step_also_discards_what_was_built_on_it(shots):
    so.run_bandpass(shots)
    so.run_agc(shots)
    so.run_mute(shots, "flat")
    removed = so.invalidate_stage(shots, "agc")
    assert removed == ["agc", "mute"]
    assert so.workflow(shots).has("bandpass")
    # the reorder: now apply them the other way round
    so.run_mute(shots, "flat")
    so.run_agc(shots)
    assert describe_stream(so.workflow(shots), "shot") == "Raw → Band-pass → Mute → AGC"


def test_loading_a_new_survey_resets_everything(vel1, survey):
    so.run_nmo1(vel1)
    other = generate_synthetic_survey(n_stations=60, shot_interval_stations=1, channels_per_side=15,
                                      seed=7, source_name="other.sgy")
    so.run_load_survey(vel1, other, "other.sgy")
    assert hist_ids(vel1) == ["upload"]
    assert vel1["gathers"] is None and vel1["Vfield_1"] is None and vel1["shot_gathers"] is None


def test_raw_survey_data_is_never_modified(survey):
    before = survey.raw_data.copy()
    store = so.new_store(survey)
    so.run_build_shot_gathers(store)
    so.run_agc(store); so.run_bandpass(store); so.run_mute(store, "flat")
    so.run_cdp_sort(store, N_MIN)
    np.testing.assert_array_equal(survey.raw_data, before)


# ─────────────────────────────────────────────────────────────────────────
# Migration: two implementations, two sets of inputs
# ─────────────────────────────────────────────────────────────────────────

def test_prestack_migration_runs_without_a_stack_or_regularisation(vel1):
    entry = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)
    assert entry.variant == "pstm"
    assert entry.consumed == {"data": "cdp_sort", "velocity": "vel1"}
    assert entry.input_key == "traces" and "Regularised offsets" in entry.notes
    assert finite(vel1["stack_migrated"]) and finite(vel1["pstm_image"])
    assert vel1["stack_matrix"] is None and not so.workflow(vel1).has("stack")
    section, label = so.best_available_section(vel1)
    assert label == "Migrated section" and section is vel1["stack_migrated"]


def test_prestack_migration_prefers_regularised_gathers_when_they_exist(vel1):
    so.run_data_reg(vel1)
    so.run_velocity(vel1, "vel2", n_analysis=3)
    entry = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)
    assert entry.consumed == {"data": "data_reg", "velocity": "vel2"}
    assert entry.input_key == "traces_regularized" and entry.offsets_key == "offsets_regularized"
    assert entry.notes == ""


def test_the_two_migrations_record_which_implementation_ran(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    post = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert post.variant == "poststack" and post.params["domain"] == "post-stack"
    pre = so.run_migration_pstm(vel1, aperture_m=150.0, cdp_stride=10)    # replaces it
    assert so.workflow(vel1).entry("migration").variant == "pstm" and pre.params["cdp_stride"] == 10
    assert hist_ids(vel1).count("migration") == 1


def test_migration_velocity_is_pass_two_when_available(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    so.run_velocity(vel1, "vel2", n_analysis=3)      # built on the CDP gathers, after the stack
    entry = so.run_migration_poststack(vel1, aperture_m=150.0)
    assert entry.consumed["velocity"] == "vel2"


def test_final_scaling_works_on_an_unmigrated_section_with_a_note(vel1):
    so.run_nmo1(vel1)
    so.run_stack(vel1)
    entry = so.run_final_scale(vel1)
    assert entry.consumed["data"] == "stack" and "Migration has not been applied" in entry.notes
    assert finite(vel1["stack_final"])
    assert so.best_available_section(vel1)[1] == "Final processed section"


def test_best_available_section_priority_and_validity(vel1):
    assert so.best_available_section(vel1) == (None, None)
    so.run_nmo1(vel1); so.run_stack(vel1)
    assert so.best_available_section(vel1)[1] == "Stack"
    so.run_rna(vel1)
    assert "random noise" in so.best_available_section(vel1)[1]
    so.run_migration_poststack(vel1, aperture_m=150.0)
    assert so.best_available_section(vel1)[1] == "Migrated section"
    so.run_final_scale(vel1)
    assert so.best_available_section(vel1)[1] == "Final processed section"
    # discarding the stack drops everything built on it — no stale section can be exported
    so.invalidate_stage(vel1, "stack")
    assert so.best_available_section(vel1) == (None, None)


# ─────────────────────────────────────────────────────────────────────────
# History and export: execution order, with lineage
# ─────────────────────────────────────────────────────────────────────────

def test_history_and_exports_list_steps_in_execution_order_with_inputs(shots):
    so.run_agc(shots)
    so.run_bandpass(shots)
    so.run_mute(shots, "flat")
    entries = sorted(shots["history"], key=lambda e: e.seq)
    assert [e.stage_id for e in entries] == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    assert [e.seq for e in entries] == sorted(e.seq for e in entries)

    report = io_export.processing_report_text({"Source": "synthetic"}, shots["history"])
    assert report.index("Amplitude Recovery") < report.index("Band-Pass Filter") < report.index("Mute")
    assert "inputs: data←" in report

    import json
    payload = json.loads(io_export.params_json_bytes(shots["history"]))
    assert [p["stage"] for p in payload] == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    bp = payload[3]
    assert bp["order"] == 4 and bp["consumed"] == {"data": "agc"} and bp["input"] == "traces_agc"
    assert bp["output"] == "traces_bp"


def test_every_stage_in_the_registry_has_an_executor(survey):
    execs = {sid for sid, _ in _executors(survey)}
    assert {s.id for s in PIPELINE_STAGES} - {"upload"} == execs
