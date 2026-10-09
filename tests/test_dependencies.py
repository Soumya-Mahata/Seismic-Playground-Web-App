"""
tests/test_dependencies.py — the dependency model (pure; no data, no UI)
===========================================================================
The workflow engine keeps four things apart that used to be one linear
chain: the *recommended order*, *genuine dependencies*, the *current data
state* and the *processing history*. These tests pin down each of them.

    pytest tests/test_dependencies.py -v

A tiny helper (`run`) plays the part of a stage executor: it asks the
engine whether a stage can run, then records it exactly as
`stage_ops._record` does. No seismic maths is involved, so every rule is
checked in milliseconds. (`tests/test_stage_ops.py` repeats the important
orders through the real maths.)
"""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.dependencies import (STAGE_SPECS, STATUS_ROWS, STREAM_BASE, STREAM_MEMBERS, SHOT_OPERATORS, Level,
                                  WorkflowState, advice_pairs, check_all, check_stage,
                                  describe_stream, get_spec, stale_notes)
from modules.pipeline import PIPELINE_STAGES, STAGE_INDEX


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def new_state(*stages):
    """A state with the survey loaded, plus any `stages` run in order."""
    state = WorkflowState()
    state.record("upload")
    for s in stages:
        run(state, *((s,) if isinstance(s, str) else s))
    return state


def run(state, stage_id, variant=None, *, allow_warnings=True):
    """Check, assert runnable, and record — like a `stage_ops` executor."""
    chk = check_stage(state, stage_id, variant)
    assert chk.can_run, f"{stage_id} blocked: {chk.blocker_text()}"
    if not allow_warnings:
        assert not chk.warnings, f"{stage_id}: unexpected warnings: {[w.message for w in chk.warnings]}"
    state.record(stage_id, consumed={r: s for r, s in chk.inputs.items() if s},
                 variant=variant or "")
    return chk


def chain(state, stream):
    """Stages applied to the stream's current data, from its base onward."""
    anc = state.data_ancestry(state.stream_head(stream))
    return anc[anc.index(STREAM_BASE[stream]):]


def kinds(chk):
    return [i.kind for i in chk.issues]


def blocked(state, stage_id, variant=None):
    chk = check_stage(state, stage_id, variant)
    assert chk.level == Level.BLOCKED and not chk.can_run, f"{stage_id} should be ⛔"
    return chk


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


def test_migration_requires_a_variant():
    with pytest.raises(ValueError):
        get_spec("migration")


def test_status_rows_cover_both_migration_implementations():
    rows = {(sid, v) for sid, v, _ in STATUS_ROWS}
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
    state = new_state()
    for stage, variant in STANDARD:
        run(state, stage, variant, allow_warnings=False)
    assert stale_notes(state) == {}
    # every stage in the workflow is now ✓ or at worst advisory about re-running
    assert {h.stage_id for h in state.history} == {s.id for s in PIPELINE_STAGES}


def test_standard_order_pstm_variant_is_also_clean():
    stages = [s for s in STANDARD if s[0] not in ("stack", "rna", "decon_post", "tvf", "migration",
                                                  "final_scale")]
    state = new_state()
    for stage, variant in stages:
        run(state, stage, variant, allow_warnings=False)
    chk = check_stage(state, "migration", "pstm")
    assert chk.level == Level.AVAILABLE, [w.message for w in chk.warnings]


# ─────────────────────────────────────────────────────────────────────────
# The two orders from the task are valid; skipped optional stages are free
# ─────────────────────────────────────────────────────────────────────────

def test_raw_agc_bandpass_mute_is_valid():
    state = new_state("shot_gathers")
    agc = run(state, "agc")
    # advice only: band-pass has not been applied — a ⚠, never a ⛔
    assert agc.level == Level.WARNING
    assert "missing_recommended" in kinds(agc)
    bp = run(state, "bandpass")           # applied *after* AGC — still allowed
    assert bp.level == Level.WARNING and "order_inversion" in kinds(bp)
    mute = run(state, "mute")
    assert mute.can_run
    assert describe_stream(state, "shot") == "Raw → AGC → Band-pass → Mute"
    run(state, "cdp_sort")


def test_raw_bandpass_fk_cdpsort_is_valid_and_clean():
    state = new_state("shot_gathers")
    run(state, "bandpass", allow_warnings=False)
    run(state, "fk", allow_warnings=False)
    run(state, "cdp_sort", allow_warnings=False)
    assert describe_stream(state, "shot") == "Raw → Band-pass → F-K"
    assert state.entry("cdp_sort").consumed["data"] == "fk"


def test_no_optional_stage_is_forced_before_cdp_sort():
    """Raw → shot gathers → CDP sort needs none of the nine conditioning steps."""
    state = new_state("shot_gathers")
    chk = run(state, "cdp_sort")
    assert chk.can_run
    assert kinds(chk) == ["missing_recommended"]       # an advisory note, not a block
    assert "No pre-stack conditioning" in chk.warnings[0].message


def test_cdp_sort_needs_only_the_survey_not_shot_gathers():
    state = new_state()
    chk = check_stage(state, "cdp_sort")
    assert chk.can_run and chk.inputs["data"] == "upload"


def test_every_shot_operator_runs_with_only_shot_gathers():
    """No shot-domain operator depends on any other shot-domain operator."""
    for op in SHOT_OPERATORS:
        state = new_state("shot_gathers")
        assert check_stage(state, op).can_run, op
        assert [b for b in check_stage(state, op).blockers] == []


@pytest.mark.parametrize("order", list(itertools.permutations(["bandpass", "agc", "mute", "fk"])))
def test_shot_operators_run_in_any_order(order):
    state = new_state("shot_gathers")
    for op in order:
        run(state, op)                       # never blocked, whatever the order
    assert chain(state, "shot") == ["shot_gathers", *order]
    run(state, "cdp_sort")


def test_each_operator_reads_the_current_head_not_a_fixed_predecessor():
    state = new_state("shot_gathers", "mute", "bandpass")
    assert state.entry("mute").consumed["data"] == "shot_gathers"
    assert state.entry("bandpass").consumed["data"] == "mute"
    assert state.stream_head("shot") == "bandpass"


def test_skipping_every_optional_refinement_reaches_a_stack():
    """CDP sort → velocity 1 → NMO 1 → stack → post-stack migration: no
    statics, DMO, regularisation, pass-2 velocity, or post-stack filters."""
    state = new_state("cdp_sort", "vel1", "nmo1")
    stack = run(state, "stack")
    assert stack.inputs["data"] == "nmo1"
    mig = run(state, "migration", "poststack")
    assert mig.can_run
    assert {"fallback_input", "missing_recommended"} <= set(kinds(mig))   # ⚠ only


def test_final_nmo_does_not_require_pass_two_velocity():
    state = new_state("cdp_sort", "vel1")
    chk = check_stage(state, "nmo_final")
    assert chk.can_run and chk.inputs["velocity"] == "vel1"
    assert kinds(chk) == ["fallback_input"]


def test_dmo_and_statics_are_independent_branches():
    """DMO needs velocities, not NMO-1 or statics; statics needs NMO-1, not DMO."""
    state = new_state("cdp_sort", "vel1")
    dmo = check_stage(state, "dmo")
    assert dmo.can_run and "missing_recommended" in kinds(dmo)   # statics advised, not required
    state = new_state("cdp_sort", "vel1", "nmo1")
    assert check_stage(state, "statics").can_run
    assert check_stage(state, "dmo").can_run


def test_pass_two_velocity_does_not_require_regularisation():
    state = new_state("cdp_sort", "vel1")
    chk = check_stage(state, "vel2")
    assert chk.can_run
    assert [i.kind for i in chk.issues] == ["missing_recommended"]


def test_post_stack_operators_are_independent_and_reorderable():
    for order in itertools.permutations(["rna", "decon_post", "tvf"]):
        state = new_state("cdp_sort", "vel1", "nmo1", "stack")
        for op in order:
            run(state, op)
        assert chain(state, "section") == ["stack", *order]
    state = new_state("cdp_sort", "vel1", "nmo1", "stack")
    assert check_stage(state, "tvf").can_run          # tvf without decon_post: ⚠, not ⛔


# ─────────────────────────────────────────────────────────────────────────
# Recommendations produce ⚠ — with the wording the task asked for
# ─────────────────────────────────────────────────────────────────────────

def test_agc_without_bandpass_warns_in_plain_words():
    chk = check_stage(new_state("shot_gathers"), "agc")
    assert chk.level == Level.WARNING and chk.can_run
    msg = chk.warnings[0].message
    assert msg.startswith("Band-pass filtering is commonly applied before")
    assert "has not been applied to the current data" in msg


def test_bandpass_then_agc_is_clean():
    state = new_state("shot_gathers", "bandpass")
    assert check_stage(state, "agc").level == Level.AVAILABLE


def test_decon_expects_coherent_noise_removed_but_does_not_require_it():
    state = new_state("shot_gathers", "bandpass", "agc")
    chk = check_stage(state, "decon")
    assert chk.can_run and chk.level == Level.WARNING
    assert any("ground roll" in w.message or "Coherent noise" in w.message for w in chk.warnings)
    state2 = new_state("shot_gathers", "bandpass", "agc", "mute")
    assert check_stage(state2, "decon").level == Level.AVAILABLE      # either one satisfies it
    state3 = new_state("shot_gathers", "bandpass", "agc", "fk")
    assert check_stage(state3, "decon").level == Level.AVAILABLE


def test_running_a_step_after_what_normally_follows_it_is_flagged():
    state = new_state("shot_gathers", "decon")
    chk = check_stage(state, "mute")
    assert chk.can_run and "order_inversion" in kinds(chk)
    assert "already part of the current data" in chk.warnings[0].message


def test_orphaned_step_is_flagged_when_a_branch_is_left_out_of_the_data():
    """Statics were solved from the CDP gathers; DMO → inverse NMO then
    carried on from the *un-corrected* gathers. Regularising now would read
    the inverse-NMO data, which does not contain the statics."""
    state = new_state("cdp_sort", "vel1", "dmo", "nmo1", "statics", "inv_nmo")
    assert state.entry("inv_nmo").consumed["data"] == "dmo"
    chk = check_stage(state, "data_reg")
    assert chk.can_run
    orphan = [i for i in chk.warnings if i.kind == "orphaned_step"]
    assert orphan and "Residual Statics" in orphan[0].message


# ─────────────────────────────────────────────────────────────────────────
# Genuine dependencies are preserved (⛔)
# ─────────────────────────────────────────────────────────────────────────

def test_nothing_runs_without_data():
    state = WorkflowState()
    for stage in ("shot_gathers", "cdp_sort", "bandpass", "vel1", "stack", "rna"):
        chk = blocked(state, stage)
        assert chk.blockers[0].kind == "missing_input"


def test_shot_domain_steps_need_shot_gathers():
    state = new_state()                         # survey only
    chk = blocked(state, "bandpass")
    assert "shot gathers" in chk.blockers[0].message
    assert "Build shot gathers" in chk.blockers[0].fix


def test_velocity_analysis_needs_cdp_gathers():
    state = new_state("shot_gathers", "bandpass")
    chk = blocked(state, "vel1")
    assert "CDP gathers" in chk.blockers[0].message
    assert "CDP Sort" in chk.blockers[0].fix


def test_nmo_needs_velocity_information():
    state = new_state("cdp_sort")
    chk = blocked(state, "nmo1")
    assert "velocities" in chk.blockers[0].message
    chk = blocked(state, "nmo_final")
    assert "velocities" in chk.blockers[0].message
    chk = blocked(state, "dmo")
    assert "velocities" in chk.blockers[0].message


def test_stack_needs_moveout_corrected_gathers():
    state = new_state("cdp_sort", "vel1")        # velocities exist, NMO has not been applied
    chk = blocked(state, "stack")
    assert "NMO-corrected" in chk.blockers[0].message
    assert "out of phase" in chk.blockers[0].message
    assert "NMO" in chk.blockers[0].fix


def test_stack_after_pass_one_nmo_warns_but_runs():
    state = new_state("cdp_sort", "vel1", "nmo1")
    chk = check_stage(state, "stack")
    assert chk.level == Level.WARNING and chk.can_run
    assert chk.inputs["data"] == "nmo1"
    assert "Final NMO has not been applied" in chk.warnings[0].message


def test_stack_prefers_final_nmo_when_present():
    state = new_state("cdp_sort", "vel1", "nmo1", "nmo_final")
    chk = check_stage(state, "stack")
    assert chk.level == Level.AVAILABLE and chk.inputs["data"] == "nmo_final"


def test_nmo_dependent_stages_need_their_own_inputs():
    state = new_state("cdp_sort", "vel1")
    blocked(state, "statics")                    # needs NMO pass 1
    blocked(state, "inv_nmo")                    # needs DMO output


def test_residual_statics_cannot_run_on_regularised_gathers():
    state = new_state("cdp_sort", "vel1", "nmo1", "data_reg")
    chk = blocked(state, "statics")
    assert chk.blockers[0].kind == "geometry"
    assert "shot and receiver" in chk.blockers[0].message


def test_statics_before_regularisation_is_fine():
    state = new_state("cdp_sort", "vel1", "nmo1", "statics", "data_reg")
    assert state.entry("data_reg").consumed["data"] == "statics"


def test_post_stack_steps_need_a_stack():
    state = new_state("cdp_sort", "vel1", "nmo1")
    for stage in ("rna", "decon_post", "tvf"):
        chk = blocked(state, stage)
        assert "stacked section" in chk.blockers[0].message


def test_post_stack_migration_needs_a_section_and_a_velocity_field():
    state = new_state("cdp_sort")
    chk = blocked(state, "migration", "poststack")
    msgs = " ".join(b.message for b in chk.blockers)
    assert "stacked section" in msgs and "velocity" in msgs
    state = new_state("cdp_sort", "vel1")        # velocity, still no section
    chk = blocked(state, "migration", "poststack")
    assert len(chk.blockers) == 1 and "stacked section" in chk.blockers[0].message


def test_prestack_migration_needs_gathers_and_velocity_but_not_a_stack():
    state = new_state("shot_gathers")
    blocked(state, "migration", "pstm")                             # no CDP gathers
    state = new_state("cdp_sort")
    chk = blocked(state, "migration", "pstm")                       # gathers, no velocity
    assert "velocity" in chk.blockers[0].message
    state = new_state("cdp_sort", "vel1")
    chk = check_stage(state, "migration", "pstm")
    assert chk.can_run and not chk.blockers          # no stack, no NMO, no regularisation needed
    assert set(kinds(chk)) == {"fallback_input", "missing_recommended"}      # ⚠ only
    assert any("Regularised offsets" in w.message for w in chk.warnings)


def test_prestack_migration_reads_offset_preserving_gathers_not_nmo_output():
    state = new_state("cdp_sort", "vel1", "nmo1", "nmo_final")
    chk = check_stage(state, "migration", "pstm")
    assert chk.inputs["data"] == "cdp_sort"          # NMO is not part of the pre-NMO stream
    state = new_state("cdp_sort", "vel1", "data_reg")
    assert check_stage(state, "migration", "pstm").inputs["data"] == "data_reg"
    state = new_state("cdp_sort", "vel1", "data_reg", "vel2")
    chk = check_stage(state, "migration", "pstm")
    assert chk.inputs["data"] == "data_reg" and chk.level == Level.AVAILABLE


def test_migration_velocity_prefers_pass_two():
    state = new_state("cdp_sort", "vel1", "data_reg", "vel2")
    chk = check_stage(state, "migration", "pstm")
    assert chk.inputs["velocity"] == "vel2" and chk.level == Level.AVAILABLE
    state = new_state("cdp_sort", "vel1", "data_reg")
    chk = check_stage(state, "migration", "pstm")
    assert chk.inputs["velocity"] == "vel1"
    assert "fallback_input" in kinds(chk)


def test_final_scaling_needs_some_section():
    state = new_state("cdp_sort", "vel1", "nmo1")
    blocked(state, "final_scale")
    state = new_state("cdp_sort", "vel1", "nmo1", "stack")
    chk = check_stage(state, "final_scale")
    assert chk.can_run and chk.warnings and "Migration has not been applied" in chk.warnings[0].message
    state = new_state("cdp_sort", "vel1", "nmo1", "stack", ("migration", "poststack"))
    chk = check_stage(state, "final_scale")
    assert chk.level == Level.AVAILABLE and chk.inputs["data"] == "migration"


def test_blocked_beats_warning_in_the_level():
    chk = check_stage(new_state("cdp_sort", "vel1"), "stack")
    assert chk.level == Level.BLOCKED
    assert Level.AVAILABLE < Level.WARNING < Level.BLOCKED
    assert (Level.AVAILABLE.symbol, Level.WARNING.symbol, Level.BLOCKED.symbol) == ("✓", "⚠", "⛔")


def test_status_panel_uses_all_three_states():
    state = new_state("shot_gathers", "cdp_sort")
    levels = {chk.level for _, _, _, chk in check_all(state)}
    assert levels == {Level.AVAILABLE, Level.WARNING, Level.BLOCKED}


# ─────────────────────────────────────────────────────────────────────────
# Data state, history and invalidation
# ─────────────────────────────────────────────────────────────────────────

def test_history_is_in_execution_order_not_recommended_order():
    state = new_state("shot_gathers", "agc", "bandpass", "mute")
    ordered = [h.stage_id for h in sorted(state.history, key=lambda e: e.seq)]
    assert ordered == ["upload", "shot_gathers", "agc", "bandpass", "mute"]
    assert STAGE_INDEX["bandpass"] < STAGE_INDEX["agc"]          # recommended order differs


def test_history_records_what_each_step_consumed():
    state = new_state("shot_gathers", "agc", "bandpass")
    assert state.entry("agc").consumed == {"data": "shot_gathers"}
    assert state.entry("bandpass").consumed == {"data": "agc"}


def test_rerunning_a_step_invalidates_only_what_was_built_on_it():
    """DMO reads the CDP gathers and velocities; it does not read NMO-1.
    So re-running NMO-1 must keep DMO and inverse NMO."""
    state = new_state("cdp_sort", "vel1", "nmo1", "dmo", "inv_nmo")
    removed = state.record("nmo1", consumed={"data": "cdp_sort", "velocity": "vel1"})
    assert removed == []
    assert state.has("dmo") and state.has("inv_nmo")


def test_rerunning_velocity_invalidates_everything_that_used_it():
    state = new_state("cdp_sort", "vel1", "nmo1", "dmo", "inv_nmo", "stack")
    removed = state.record("vel1", consumed={"data": "cdp_sort"})
    assert set(removed) == {"nmo1", "dmo", "inv_nmo", "stack"}
    assert state.has("cdp_sort") and state.has("vel1")


def test_rerunning_a_shot_step_invalidates_its_consumers_and_the_cdp_sort():
    state = new_state("shot_gathers", "bandpass", "agc", "cdp_sort", "vel1")
    removed = state.record("bandpass", consumed={"data": "shot_gathers"})
    assert set(removed) == {"agc", "cdp_sort", "vel1"}


def test_a_step_that_nothing_consumed_can_be_rerun_without_side_effects():
    state = new_state("shot_gathers", "agc", "bandpass")
    removed = state.record("bandpass", consumed={"data": "agc"})
    assert removed == [] and state.has("agc")


def test_steps_later_in_the_recommended_order_are_kept_unless_dependent():
    """DMO and inverse NMO come after NMO-1 in the recommended order, but DMO
    reads the pre-NMO gathers: re-running NMO-1 leaves them alone. (The old
    linear model discarded everything 'after' the re-run stage.)"""
    state = new_state("cdp_sort", "vel1", "dmo", "inv_nmo", "nmo1")
    assert STAGE_INDEX["nmo1"] < STAGE_INDEX["dmo"] < STAGE_INDEX["inv_nmo"]
    assert state.record("nmo1", consumed={"data": "cdp_sort", "velocity": "vel1"}) == []
    assert state.has("dmo") and state.has("inv_nmo")


def test_a_rerun_stage_reads_the_best_remaining_head_not_its_own_output():
    state = new_state("shot_gathers", "bandpass", "agc", "mute")
    chk = check_stage(state, "agc")              # re-run AGC: it must not read mute (a descendant)
    assert chk.inputs["data"] == "bandpass"
    chk = check_stage(state, "bandpass")
    assert chk.inputs["data"] == "shot_gathers"


def test_invalidate_returns_ids_so_data_can_be_freed():
    state = new_state("shot_gathers", "bandpass", "agc", "cdp_sort")
    removed = state.invalidate("bandpass", include_self=True)
    assert removed[0] == "bandpass" and set(removed) == {"bandpass", "agc", "cdp_sort"}
    assert state.completed() == {"upload", "shot_gathers"}


def test_loading_new_data_invalidates_everything():
    state = new_state("shot_gathers", "bandpass", "cdp_sort", "vel1")
    removed = state.invalidate("upload", include_self=True)
    assert state.completed() == set() and "vel1" in removed


# ─────────────────────────────────────────────────────────────────────────
# Stale notes (data moved on after a step ran — informational, not an error)
# ─────────────────────────────────────────────────────────────────────────

def test_stale_note_when_conditioning_is_applied_after_the_cdp_sort():
    state = new_state("shot_gathers", "cdp_sort")
    assert stale_notes(state) == {}
    run(state, "bandpass")
    notes = stale_notes(state)
    assert "cdp_sort" in notes and "band-pass" in notes["cdp_sort"].lower()
    assert state.has("cdp_sort")                  # nothing was thrown away


def test_stale_note_when_final_nmo_arrives_after_the_stack():
    state = new_state("cdp_sort", "vel1", "nmo1", "stack", "nmo_final")
    notes = stale_notes(state)
    assert "stack" in notes and "final nmo" in notes["stack"].lower()


def test_no_stale_notes_for_a_consistent_chain():
    state = new_state("shot_gathers", "bandpass", "agc", "cdp_sort", "vel1", "nmo1", "nmo_final",
                      "stack")
    assert stale_notes(state) == {}


# ─────────────────────────────────────────────────────────────────────────
# Describing the data state
# ─────────────────────────────────────────────────────────────────────────

def test_describe_stream_shows_what_has_been_applied():
    assert describe_stream(new_state(), "shot").startswith("—")
    state = new_state("shot_gathers")
    assert describe_stream(state, "shot") == "Raw"
    run(state, "bandpass"); run(state, "fk")
    assert describe_stream(state, "shot") == "Raw → Band-pass → F-K"


def test_every_stage_spec_names_only_known_stages():
    known = set(STAGE_INDEX) | {"upload"}
    for key, spec in STAGE_SPECS.items():
        for need in spec.needs:
            for cand in need.candidates:
                assert cand.startswith("@") or cand in known, (key, cand)
        for adv in spec.advice:
            assert set(adv.stages) <= known, (key, adv.stages)
