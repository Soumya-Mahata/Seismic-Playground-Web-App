"""
tests/test_dependencies.py — the dependency model on data states (pure; no data, no UI)
==========================================================================================
The workflow engine keeps these apart: the *recommended order*, *genuine
dependencies*, the *data states* that exist (INPUT STATE → OPERATION +
PARAMETERS → OUTPUT STATE) and the *processing history*. These tests pin
down each of them.

    pytest tests/test_dependencies.py -v

A tiny helper (`run`) plays the part of an executor: it asks the engine
whether an operation can run, then adds the new state to a `ProcessingGraph`
exactly as `stage_ops.execute` does. No seismic maths is involved, so every
rule is checked in milliseconds. (`tests/test_stage_ops.py` repeats the
important orders through the real maths.)
"""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.dependencies import (STAGE_SPECS, STATUS_ROWS, STREAM_BASE, STREAM_MEMBERS, SHOT_OPERATORS,
                                  Level, Workflow, advice_pairs, check_all, check_stage, describe_stream,
                                  get_spec, resolve_inputs, stale_info, stale_notes, state_name)
from modules.pipeline import PIPELINE_STAGES, ROOT_STAGES, STAGE_INDEX, STAGE_KIND
from modules.state_graph import ProcessingGraph


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def new_graph(*stages):
    """A graph with the survey loaded (S0), plus any `stages` run in order."""
    g = ProcessingGraph()
    g.add(kind="raw", op="upload", params={"source": "test"})
    for s in stages:
        run(g, *((s,) if isinstance(s, str) else s))
    return g


def run(g, stage_id, variant=None, *, inputs=None, params=None, allow_warnings=True):
    """Check, assert runnable, and add the output state — like `stage_ops.execute`."""
    chk = check_stage(Workflow(g), stage_id, variant, inputs=inputs)
    assert chk.can_run, f"{stage_id} blocked: {chk.blocker_text()}"
    if not allow_warnings:
        assert not chk.warnings, f"{stage_id}: unexpected warnings: {[w.message for w in chk.warnings]}"
    in_ids = {r: s for r, s in chk.inputs.items() if s}
    root = "self" if stage_id in ROOT_STAGES else (g[in_ids["data"]].root if "data" in in_ids else None)
    g.add(kind=STAGE_KIND[stage_id], op=stage_id, variant=variant or "", params=params or {},
          inputs=in_ids, root=root)
    return chk


def wf(g):
    return Workflow(g)


def chain(g, kind):
    """Operations applied to the kind's current data, from its base onward."""
    ops = [s.op for s in g.lineage(g.current(kind).id)]
    return ops[ops.index(STREAM_BASE[kind]):]


def sid(g, op, variant=None):
    """Id of the most recent state an operation produced."""
    return g.latest(op, variant).id


def kinds(chk):
    return [i.kind for i in chk.issues]


def blocked(g, stage_id, variant=None):
    chk = check_stage(wf(g), stage_id, variant)
    assert chk.level == Level.BLOCKED and not chk.can_run, f"{stage_id} should be ⛔"
    return chk


def check(g, stage_id, variant=None, **kw):
    return check_stage(wf(g), stage_id, variant, **kw)


# the standard recommended path, as (stage, variant) pairs
STANDARD = [(s.id, "poststack" if s.id == "migration" else None)
            for s in PIPELINE_STAGES if s.id != "upload"]


# ─────────────────────────────────────────────────────────────────────────
# Registry consistency: the recommended order is advice, and the advice
# agrees with it
# ─────────────────────────────────────────────────────────────────────────

def test_every_registered_stage_has_a_dependency_spec():
    for s in PIPELINE_STAGES:
        if s.id == "migration":
            assert get_spec("migration", "poststack") and get_spec("migration", "pstm")
        else:
            assert get_spec(s.id) is not None, s.id


def test_every_stage_has_a_data_kind_and_roots_are_registered():
    for s in PIPELINE_STAGES:
        assert s.id in STAGE_KIND, s.id
    assert set(ROOT_STAGES) <= set(STAGE_INDEX)


def test_migration_requires_a_variant():
    with pytest.raises(ValueError):
        get_spec("migration")


def test_status_rows_cover_both_migration_implementations():
    rows = {(sid_, v) for sid_, v, _ in STATUS_ROWS}
    assert ("migration", "poststack") in rows and ("migration", "pstm") in rows
    assert len(STATUS_ROWS) == len(PIPELINE_STAGES) + 1


def test_recommendations_agree_with_the_recommended_order():
    """Every 'X is normally applied before Y' advice must match the order of
    PIPELINE_STAGES — the two descriptions of 'recommended' must not drift."""
    pairs = advice_pairs()
    assert pairs
    for earlier, later in pairs:
        assert STAGE_INDEX[earlier] < STAGE_INDEX[later], (earlier, later)


def test_stream_members_are_registered_stages():
    for members in STREAM_MEMBERS.values():
        for m in members:
            assert m in STAGE_INDEX


# ─────────────────────────────────────────────────────────────────────────
# The standard order is still perfectly clean
# ─────────────────────────────────────────────────────────────────────────

def test_standard_order_has_no_warnings_at_any_step():
    g = new_graph()
    for stage, variant in STANDARD:
        run(g, stage, variant, allow_warnings=False)
    assert stale_notes(wf(g)) == {}
    assert {s.op for s in g} == {s.id for s in PIPELINE_STAGES}
    assert len(g) == len(PIPELINE_STAGES)            # one state per step: S0 … S{n-1}


def test_standard_order_pstm_variant_is_also_clean():
    stages = [s for s in STANDARD if s[0] not in ("stack", "rna", "decon_post", "tvf", "migration",
                                                  "final_scale")]
    g = new_graph()
    for stage, variant in stages:
        run(g, stage, variant, allow_warnings=False)
    chk = check(g, "migration", "pstm")
    assert chk.level == Level.AVAILABLE, [w.message for w in chk.warnings]


# ─────────────────────────────────────────────────────────────────────────
# The two orders from the task are valid; skipped optional stages are free
# ─────────────────────────────────────────────────────────────────────────

def test_raw_agc_bandpass_mute_is_valid():
    g = new_graph("shot_gathers")
    agc = run(g, "agc")
    # advice only: band-pass has not been applied — a ⚠, never a ⛔
    assert agc.level == Level.WARNING
    assert "missing_recommended" in kinds(agc)
    bp = run(g, "bandpass")           # applied *after* AGC — still allowed
    assert bp.level == Level.WARNING and "order_inversion" in kinds(bp)
    mute = run(g, "mute")
    assert mute.can_run
    assert describe_stream(wf(g), "shot") == "Raw → AGC → Band-pass → Mute"
    run(g, "cdp_sort")


def test_the_user_example_states_s0_to_s3():
    """S0 raw → AGC(A) → S1 → Band-pass(B) → S2 → Mute(C) → S3 (S0 = shot gathers here)."""
    g = new_graph("shot_gathers")
    run(g, "agc", params={"window": 500})
    run(g, "bandpass", params={"fmin": 8.0, "fmax": 60.0})
    run(g, "mute", params={"t_top": 0.1})
    shot = g.get(sid(g, "shot_gathers"))
    agc, bp, mute = (g.latest(o) for o in ("agc", "bandpass", "mute"))
    assert (agc.inputs, bp.inputs, mute.inputs) == (
        {"data": shot.id}, {"data": agc.id}, {"data": bp.id})
    assert (agc.params, bp.params, mute.params) == (
        {"window": 500}, {"fmax": 60.0, "fmin": 8.0}, {"t_top": 0.1})
    assert g.current("shot").id == mute.id


def test_raw_bandpass_fk_cdpsort_is_valid_and_clean():
    g = new_graph("shot_gathers")
    run(g, "bandpass", allow_warnings=False)
    run(g, "fk", allow_warnings=False)
    run(g, "cdp_sort", allow_warnings=False)
    assert describe_stream(wf(g), "shot") == "Raw → Band-pass → F-K"
    assert g.latest("cdp_sort").inputs["data"] == sid(g, "fk")


def test_no_optional_stage_is_forced_before_cdp_sort():
    """Raw → shot gathers → CDP sort needs none of the nine conditioning steps."""
    g = new_graph("shot_gathers")
    chk = run(g, "cdp_sort")
    assert chk.can_run
    assert kinds(chk) == ["missing_recommended"]       # an advisory note, not a block
    assert "No pre-stack conditioning" in chk.warnings[0].message


def test_cdp_sort_needs_only_the_survey_not_shot_gathers():
    g = new_graph()
    chk = check(g, "cdp_sort")
    assert chk.can_run and chk.inputs["data"] == sid(g, "upload")


def test_every_shot_operator_runs_with_only_shot_gathers():
    """No shot-domain operator depends on any other shot-domain operator."""
    for op in SHOT_OPERATORS:
        g = new_graph("shot_gathers")
        assert check(g, op).can_run, op
        assert check(g, op).blockers == []


@pytest.mark.parametrize("order", list(itertools.permutations(["bandpass", "agc", "mute", "fk"])))
def test_shot_operators_run_in_any_order(order):
    g = new_graph("shot_gathers")
    for op in order:
        run(g, op)                       # never blocked, whatever the order
    assert chain(g, "shot") == ["shot_gathers", *order]
    run(g, "cdp_sort")


def test_each_operator_reads_the_current_data_not_a_fixed_predecessor():
    g = new_graph("shot_gathers", "mute", "bandpass")
    assert g.latest("mute").inputs["data"] == sid(g, "shot_gathers")
    assert g.latest("bandpass").inputs["data"] == sid(g, "mute")
    assert g.current("shot").id == sid(g, "bandpass")


def test_skipping_every_optional_refinement_reaches_a_stack():
    """CDP sort → velocity 1 → NMO 1 → stack → post-stack migration: no
    statics, DMO, regularisation, pass-2 velocity, or post-stack filters."""
    g = new_graph("cdp_sort", "vel1", "nmo1")
    stack = run(g, "stack")
    assert stack.inputs["data"] == sid(g, "nmo1")
    mig = run(g, "migration", "poststack")
    assert mig.can_run
    assert {"fallback_input", "missing_recommended"} <= set(kinds(mig))   # ⚠ only


def test_final_nmo_does_not_require_pass_two_velocity():
    g = new_graph("cdp_sort", "vel1")
    chk = check(g, "nmo_final")
    assert chk.can_run and chk.inputs["velocity"] == sid(g, "vel1")
    assert kinds(chk) == ["fallback_input"]


def test_dmo_and_statics_are_independent_branches():
    """DMO needs velocities, not NMO-1 or statics; statics needs NMO-1, not DMO."""
    g = new_graph("cdp_sort", "vel1")
    dmo = check(g, "dmo")
    assert dmo.can_run and "missing_recommended" in kinds(dmo)   # statics advised, not required
    g = new_graph("cdp_sort", "vel1", "nmo1")
    assert check(g, "statics").can_run
    assert check(g, "dmo").can_run


def test_pass_two_velocity_does_not_require_regularisation():
    g = new_graph("cdp_sort", "vel1")
    chk = check(g, "vel2")
    assert chk.can_run
    assert [i.kind for i in chk.issues] == ["missing_recommended"]


def test_post_stack_operators_are_independent_and_reorderable():
    for order in itertools.permutations(["rna", "decon_post", "tvf"]):
        g = new_graph("cdp_sort", "vel1", "nmo1", "stack")
        for op in order:
            run(g, op)
        assert chain(g, "section") == ["stack", *order]
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack")
    assert check(g, "tvf").can_run          # tvf without decon_post: ⚠, not ⛔


# ─────────────────────────────────────────────────────────────────────────
# Recommendations produce ⚠ — with the wording the task asked for
# ─────────────────────────────────────────────────────────────────────────

def test_agc_without_bandpass_warns_in_plain_words():
    chk = check(new_graph("shot_gathers"), "agc")
    assert chk.level == Level.WARNING and chk.can_run
    msg = chk.warnings[0].message
    assert msg.startswith("Band-pass filtering is commonly applied before")
    assert "has not been applied to the current data" in msg


def test_bandpass_then_agc_is_clean():
    g = new_graph("shot_gathers", "bandpass")
    assert check(g, "agc").level == Level.AVAILABLE


def test_decon_expects_coherent_noise_removed_but_does_not_require_it():
    g = new_graph("shot_gathers", "bandpass", "agc")
    chk = check(g, "decon")
    assert chk.can_run and chk.level == Level.WARNING
    assert any("ground roll" in w.message or "Coherent noise" in w.message for w in chk.warnings)
    g2 = new_graph("shot_gathers", "bandpass", "agc", "mute")
    assert check(g2, "decon").level == Level.AVAILABLE      # either one satisfies it
    g3 = new_graph("shot_gathers", "bandpass", "agc", "fk")
    assert check(g3, "decon").level == Level.AVAILABLE


def test_running_a_step_after_what_normally_follows_it_is_flagged():
    g = new_graph("shot_gathers", "decon")
    chk = check(g, "mute")
    assert chk.can_run and "order_inversion" in kinds(chk)
    assert "already part of the current data" in chk.warnings[0].message


def test_orphaned_step_is_flagged_when_a_branch_is_left_out_of_the_data():
    """Statics were solved from the CDP gathers; DMO → inverse NMO then
    carried on from the *un-corrected* gathers. Regularising now would read
    the inverse-NMO data, which does not contain the statics."""
    g = new_graph("cdp_sort", "vel1", "dmo", "nmo1", "statics", "inv_nmo")
    assert g.latest("inv_nmo").inputs["data"] == sid(g, "dmo")
    chk = check(g, "data_reg")
    assert chk.can_run
    orphan = [i for i in chk.warnings if i.kind == "orphaned_step"]
    assert orphan and "Residual Statics" in orphan[0].message


# ─────────────────────────────────────────────────────────────────────────
# Genuine dependencies are preserved (⛔)
# ─────────────────────────────────────────────────────────────────────────

def test_nothing_runs_without_data():
    g = ProcessingGraph()
    for stage in ("shot_gathers", "cdp_sort", "bandpass", "vel1", "stack", "rna"):
        chk = blocked(g, stage)
        assert chk.blockers[0].kind == "missing_input"


def test_shot_domain_steps_need_shot_gathers():
    g = new_graph()                         # survey only
    chk = blocked(g, "bandpass")
    assert "shot gathers" in chk.blockers[0].message
    assert "Build shot gathers" in chk.blockers[0].fix


def test_velocity_analysis_needs_cdp_gathers():
    g = new_graph("shot_gathers", "bandpass")
    chk = blocked(g, "vel1")
    assert "CDP gathers" in chk.blockers[0].message
    assert "CDP Sort" in chk.blockers[0].fix


def test_nmo_needs_velocity_information():
    g = new_graph("cdp_sort")
    for stage in ("nmo1", "nmo_final", "dmo"):
        chk = blocked(g, stage)
        assert "velocities" in chk.blockers[0].message


def test_stack_needs_moveout_corrected_gathers():
    g = new_graph("cdp_sort", "vel1")        # velocities exist, NMO has not been applied
    chk = blocked(g, "stack")
    assert "NMO-corrected" in chk.blockers[0].message
    assert "out of phase" in chk.blockers[0].message
    assert "NMO" in chk.blockers[0].fix


def test_stack_after_pass_one_nmo_warns_but_runs():
    g = new_graph("cdp_sort", "vel1", "nmo1")
    chk = check(g, "stack")
    assert chk.level == Level.WARNING and chk.can_run
    assert chk.inputs["data"] == sid(g, "nmo1")
    assert "Final NMO has not been applied" in chk.warnings[0].message


def test_stack_prefers_final_nmo_when_present():
    g = new_graph("cdp_sort", "vel1", "nmo1", "nmo_final")
    chk = check(g, "stack")
    assert chk.level == Level.AVAILABLE and chk.inputs["data"] == sid(g, "nmo_final")


def test_nmo_dependent_stages_need_their_own_inputs():
    g = new_graph("cdp_sort", "vel1")
    blocked(g, "statics")                    # needs NMO pass 1
    blocked(g, "inv_nmo")                    # needs DMO output


def test_residual_statics_cannot_run_on_regularised_gathers():
    g = new_graph("cdp_sort", "vel1", "nmo1", "data_reg")
    chk = blocked(g, "statics")
    assert chk.blockers[0].kind == "geometry"
    assert "shot and receiver" in chk.blockers[0].message


def test_statics_before_regularisation_is_fine():
    g = new_graph("cdp_sort", "vel1", "nmo1", "statics", "data_reg")
    assert g.latest("data_reg").inputs["data"] == sid(g, "statics")


def test_post_stack_steps_need_a_stack():
    g = new_graph("cdp_sort", "vel1", "nmo1")
    for stage in ("rna", "decon_post", "tvf"):
        chk = blocked(g, stage)
        assert "stacked section" in chk.blockers[0].message


def test_post_stack_migration_needs_a_section_and_a_velocity_field():
    g = new_graph("cdp_sort")
    chk = blocked(g, "migration", "poststack")
    msgs = " ".join(b.message for b in chk.blockers)
    assert "stacked section" in msgs and "velocity" in msgs
    g = new_graph("cdp_sort", "vel1")        # velocity, still no section
    chk = blocked(g, "migration", "poststack")
    assert len(chk.blockers) == 1 and "stacked section" in chk.blockers[0].message


def test_prestack_migration_needs_gathers_and_velocity_but_not_a_stack():
    g = new_graph("shot_gathers")
    blocked(g, "migration", "pstm")                             # no CDP gathers
    g = new_graph("cdp_sort")
    chk = blocked(g, "migration", "pstm")                       # gathers, no velocity
    assert "velocity" in chk.blockers[0].message
    g = new_graph("cdp_sort", "vel1")
    chk = check(g, "migration", "pstm")
    assert chk.can_run and not chk.blockers          # no stack, no NMO, no regularisation needed
    assert set(kinds(chk)) == {"fallback_input", "missing_recommended"}      # ⚠ only
    assert any("Regularised offsets" in w.message for w in chk.warnings)


def test_prestack_migration_reads_offset_preserving_gathers_not_nmo_output():
    g = new_graph("cdp_sort", "vel1", "nmo1", "nmo_final")
    chk = check(g, "migration", "pstm")
    assert chk.inputs["data"] == sid(g, "cdp_sort")          # NMO output is not 'cdp'-kind data
    g = new_graph("cdp_sort", "vel1", "data_reg")
    assert check(g, "migration", "pstm").inputs["data"] == sid(g, "data_reg")
    g = new_graph("cdp_sort", "vel1", "data_reg", "vel2")
    chk = check(g, "migration", "pstm")
    assert chk.inputs["data"] == sid(g, "data_reg") and chk.level == Level.AVAILABLE


def test_migration_velocity_prefers_pass_two():
    g = new_graph("cdp_sort", "vel1", "data_reg", "vel2")
    chk = check(g, "migration", "pstm")
    assert chk.inputs["velocity"] == sid(g, "vel2") and chk.level == Level.AVAILABLE
    g = new_graph("cdp_sort", "vel1", "data_reg")
    chk = check(g, "migration", "pstm")
    assert chk.inputs["velocity"] == sid(g, "vel1")
    assert "fallback_input" in kinds(chk)


def test_final_scaling_needs_some_section():
    g = new_graph("cdp_sort", "vel1", "nmo1")
    blocked(g, "final_scale")
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack")
    chk = check(g, "final_scale")
    assert chk.can_run and chk.warnings and "Migration has not been applied" in chk.warnings[0].message
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack", ("migration", "poststack"))
    chk = check(g, "final_scale")
    assert chk.level == Level.AVAILABLE and chk.inputs["data"] == sid(g, "migration")


def test_blocked_beats_warning_in_the_level():
    chk = check(new_graph("cdp_sort", "vel1"), "stack")
    assert chk.level == Level.BLOCKED
    assert Level.AVAILABLE < Level.WARNING < Level.BLOCKED
    assert (Level.AVAILABLE.symbol, Level.WARNING.symbol, Level.BLOCKED.symbol) == ("✓", "⚠", "⛔")


def test_status_panel_uses_all_three_states():
    g = new_graph("shot_gathers", "cdp_sort")
    levels = {chk.level for _, _, _, chk in check_all(wf(g))}
    assert levels == {Level.AVAILABLE, Level.WARNING, Level.BLOCKED}


# ─────────────────────────────────────────────────────────────────────────
# Data states: what each operation consumed, repeated operations, branching
# ─────────────────────────────────────────────────────────────────────────

def test_history_is_in_execution_order_not_recommended_order():
    g = new_graph("shot_gathers", "agc", "bandpass", "mute")
    assert [s.op for s in g] == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    assert STAGE_INDEX["bandpass"] < STAGE_INDEX["agc"]          # recommended order differs


def test_each_state_records_what_it_consumed():
    g = new_graph("shot_gathers", "agc", "bandpass")
    assert g.latest("agc").inputs == {"data": sid(g, "shot_gathers")}
    assert g.latest("bandpass").inputs == {"data": sid(g, "agc")}


def test_multi_input_operations_record_every_input_role():
    g = new_graph("cdp_sort", "vel1", "nmo1")
    assert g.latest("nmo1").inputs == {"data": sid(g, "cdp_sort"), "velocity": sid(g, "vel1")}


def test_gather_states_share_the_root_of_the_gathers_they_live_on():
    g = new_graph("shot_gathers", "bandpass", "cdp_sort", "vel1", "nmo1", "stack")
    shot_root, cdp_root = sid(g, "shot_gathers"), sid(g, "cdp_sort")
    assert g.latest("bandpass").root == shot_root
    assert {g.latest(o).root for o in ("cdp_sort", "vel1", "nmo1", "stack")} == {cdp_root}


def test_repeating_an_operation_makes_a_second_state_and_warns():
    g = new_graph("shot_gathers", "bandpass", "agc")
    chk = check(g, "agc")
    assert "repeated_operation" in kinds(chk)
    assert "applied twice" in chk.warnings[-1].message or "applied twice" in " ".join(
        w.message for w in chk.warnings)
    run(g, "agc")                                              # allowed
    assert [s.version for s in g.of_op("agc")][::-1] == [1, 2]
    assert len(g.of_op("agc")) == 2 and g.current("shot").id == g.latest("agc").id
    # the repeat names the state to go back to if a redo with other settings was meant
    assert state_name(g.get(g.of_op("agc")[1].inputs["data"])) in chk.warnings[-1].message \
        or "run it on the state before it" in " ".join(w.message for w in chk.warnings)


def test_applying_an_operation_to_an_earlier_state_branches_without_a_repeat_warning():
    g = new_graph("shot_gathers", "bandpass", "agc")
    shot = sid(g, "shot_gathers")
    chk = check(g, "agc", inputs={"data": shot})
    assert chk.inputs["data"] == shot and "repeated_operation" not in kinds(chk)
    run(g, "agc", inputs={"data": shot}, params={"window": 100})
    assert g.current("shot").inputs["data"] == shot            # a branch off the raw gathers
    assert len(g.of_op("agc")) == 2 and len(g.of_op("bandpass")) == 1   # nothing was replaced
    assert {d.op for d in g.dependents(shot)} == {"bandpass", "agc"}


def test_options_list_every_state_an_input_could_be():
    g = new_graph("shot_gathers", "bandpass", "agc")
    opts = [s.op for s in check(g, "mute").options["data"]]
    assert set(opts) == {"shot_gathers", "bandpass", "agc"} and opts[0] == "agc"   # current first


def test_explicit_input_must_be_a_usable_state():
    g = new_graph("shot_gathers", "bandpass", "cdp_sort")
    chk = check(g, "agc", inputs={"data": sid(g, "cdp_sort")})        # a CDP state is not shot data
    assert chk.level == Level.BLOCKED and chk.blockers[0].kind == "invalid_input"
    chk = check(g, "agc", inputs={"data": "S99"})
    assert chk.blockers[0].kind == "invalid_input" and "no longer exists" in chk.blockers[0].message
    # a remembered UI choice that has gone stale is ignored, not an error
    lenient = check_stage(wf(g), "agc", None, inputs={"data": "S99"}, lenient=True)
    assert lenient.can_run and lenient.inputs["data"] == sid(g, "bandpass")


def test_resolve_inputs_matches_the_check():
    g = new_graph("cdp_sort", "vel1")
    assert resolve_inputs(wf(g), "nmo1") == {"data": sid(g, "cdp_sort"), "velocity": sid(g, "vel1")}
    assert resolve_inputs(wf(new_graph("cdp_sort")), "nmo1")["velocity"] is None


def test_inputs_from_different_gathers_cannot_be_combined():
    """A velocity field picked on one CDP sort cannot flatten the gathers of another."""
    g = new_graph("cdp_sort", "vel1")
    run(g, "cdp_sort", params={"n_min_traces": 5})                    # a second sort: new gathers
    sort_a = g.of_op("cdp_sort")[-1].id
    chk = blocked(g, "nmo1")
    assert chk.blockers[0].kind == "container_mismatch"
    assert "different gathers" in chk.blockers[0].message
    # applying it to the first sort (explicitly) is fine
    ok = check(g, "nmo1", inputs={"data": sort_a})
    assert ok.can_run and ok.inputs["velocity"] == sid(g, "vel1")


def test_undone_work_is_not_reported_as_an_orphaned_step():
    g = new_graph("shot_gathers", "bandpass", "agc")
    g.undo("shot")                                       # current = band-pass; AGC is redo-able
    assert "orphaned_step" not in kinds(check(g, "cdp_sort"))
    run(g, "decon", inputs={"data": sid(g, "shot_gathers")})    # a different branch: AGC/band-pass left out
    assert "orphaned_step" in kinds(check(g, "cdp_sort"))


def test_nothing_is_overwritten_by_rerunning_or_branching():
    g = new_graph("cdp_sort", "vel1", "nmo1", "dmo", "inv_nmo")
    before = [s.id for s in g]
    run(g, "nmo1")
    assert [s.id for s in g][: len(before)] == before            # old states are all still there
    assert len(g.of_op("nmo1")) == 2


# ─────────────────────────────────────────────────────────────────────────
# Stale-state detection (a state built from inputs that have since moved on)
# ─────────────────────────────────────────────────────────────────────────

def test_rerunning_nmo1_leaves_dmo_alone_but_flags_what_was_built_from_the_old_nmo1():
    """DMO reads the CDP gathers and velocities; it does not read NMO-1. So re-running
    NMO-1 does not affect DMO/inverse NMO — but residual statics, solved from the old
    NMO-1, are now stale."""
    g = new_graph("cdp_sort", "vel1", "nmo1", "dmo", "inv_nmo")
    run(g, "statics")                                        # solved from nmo1 v1
    old_nmo = sid(g, "nmo1")
    run(g, "nmo1")                                           # nmo1 v2
    stale = stale_info(wf(g))
    assert g.latest("dmo").id not in stale and g.latest("inv_nmo").id not in stale
    reasons = stale[sid(g, "statics")]
    assert reasons[0].kind == "replaced" and reasons[0].source == old_nmo


def test_rerunning_velocity_flags_everything_built_from_it():
    g = new_graph("cdp_sort", "vel1", "nmo1", "dmo", "inv_nmo", "stack")
    old = [sid(g, o) for o in ("nmo1", "dmo", "inv_nmo", "stack")]
    run(g, "vel1")
    stale = stale_info(wf(g))
    assert set(old) <= set(stale)                            # all flagged ...
    assert all(s.id in g for s in map(g.get, old))           # ... none discarded
    assert stale[sid(g, "nmo1") if False else old[0]][0].kind == "replaced"
    assert stale[old[3]][0].kind == "inherited"              # the stack inherits it from NMO


def test_stale_reason_rolled_back():
    g = new_graph("shot_gathers", "bandpass", "agc", "cdp_sort")
    assert stale_notes(wf(g)) == {}
    g.undo("shot")                                           # current shot data: before the AGC
    reasons = stale_info(wf(g))[sid(g, "cdp_sort")]
    assert reasons[0].kind == "rolled_back"


def test_stale_note_when_conditioning_is_applied_after_the_cdp_sort():
    g = new_graph("shot_gathers", "cdp_sort")
    assert stale_notes(wf(g)) == {}
    run(g, "bandpass")
    notes = stale_notes(wf(g))
    assert sid(g, "cdp_sort") in notes and "band-pass" in notes[sid(g, "cdp_sort")].lower()
    assert stale_info(wf(g))[sid(g, "cdp_sort")][0].kind == "behind"
    assert "cdp_sort" in {s.op for s in g}                   # nothing was thrown away


def test_stale_note_when_final_nmo_arrives_after_the_stack():
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack", "nmo_final")
    notes = stale_notes(wf(g))
    assert sid(g, "stack") in notes and "final nmo" in notes[sid(g, "stack")].lower()
    assert stale_info(wf(g))[sid(g, "stack")][0].kind == "diverged"


def test_an_operation_reading_a_stale_state_is_warned():
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack", "nmo_final")
    # the person asks for RNA on the (now out-of-date) stack: allowed, with a ⚠ naming why
    chk = check(g, "rna")
    assert chk.can_run and "stale_input" in kinds(chk)


def test_no_stale_notes_for_a_consistent_chain():
    g = new_graph("shot_gathers", "bandpass", "agc", "cdp_sort", "vel1", "nmo1", "nmo_final", "stack")
    assert stale_notes(wf(g)) == {}


def test_a_stale_state_is_still_usable_data():
    g = new_graph("cdp_sort", "vel1", "nmo1", "stack")
    run(g, "vel1")
    assert sid(g, "stack") in stale_info(wf(g))
    chk = check(g, "rna")                                    # reads the stale stack: ⚠, not ⛔
    assert chk.can_run and chk.level == Level.WARNING


# ─────────────────────────────────────────────────────────────────────────
# Describing the data state
# ─────────────────────────────────────────────────────────────────────────

def test_describe_stream_shows_what_has_been_applied():
    assert describe_stream(wf(new_graph()), "shot").startswith("—")
    g = new_graph("shot_gathers")
    assert describe_stream(wf(g), "shot") == "Raw"
    run(g, "bandpass"); run(g, "fk")
    assert describe_stream(wf(g), "shot") == "Raw → Band-pass → F-K"
    g.undo("shot")
    assert describe_stream(wf(g), "shot") == "Raw → Band-pass"      # undo moves the current data


def test_state_name_includes_the_stable_id_and_version():
    g = new_graph("shot_gathers", "agc")
    run(g, "agc")
    first, second = g.of_op("agc")[1], g.of_op("agc")[0]
    assert state_name(first) == f"AGC ({first.id})" and state_name(second) == f"AGC v2 ({second.id})"
    assert state_name(None) == "—"


def test_every_stage_spec_names_only_known_stages():
    known = set(STAGE_INDEX) | {"upload"}
    for key, spec in STAGE_SPECS.items():
        for need in spec.needs:
            for cand in need.candidates:
                assert cand.startswith("@") or cand in known, (key, cand)
        for adv in spec.advice:
            assert set(adv.stages) <= known, (key, adv.stages)
