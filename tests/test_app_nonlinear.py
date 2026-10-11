"""
tests/test_app_nonlinear.py — the data-state UI, driven headlessly
=====================================================================
Where `tests/test_app_flow.py` walks the recommended order end to end, this
drives the *Streamlit app itself* through orders the old linear gates
refused, and checks the three states a person sees:

    ✓ Available  /  ⚠ Recommended prerequisite not performed  /  ⛔ Required input missing

and what the app does with the data states behind them: which state each run
consumed and produced, re-runs that make a new state (or reuse an identical
one), stale states, undo / redo and choosing an earlier input.

    pytest tests/test_app_nonlinear.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PRESTACK = "⚙️ Pre-Stack Conditioning"
CDP = "🔀 CDP Sort & Binning"
VEL = "🎯 Velocity, NMO & DMO"
STACK = "📚 Stacking"
POST = "🧹 Post-Stack Processing"
MIG = "🗺️ Migration"
EXPORT = "💾 Export"


def goto(at, page):
    at.radio(key="active_page").set_value(page).run()
    assert not at.exception, [str(e) for e in at.exception]


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception, (key, [str(e) for e in at.exception])


def has_button(at, key):
    return any(b.key == key for b in at.button)


def sort_cdp(at):
    """Press 'Sort into CDP gathers'; confirm if a recommendation is pending."""
    [b for b in at.button if "Sort into CDP gathers" in (b.label or "")][0].click().run()
    assert not at.exception, [str(e) for e in at.exception]
    if has_button(at, "run_cdp_sort__continue"):
        click(at, "run_cdp_sort__continue")


def texts(at):
    """Everything the page says in warnings / errors / infos / captions."""
    out = ([e.value for e in at.error] + [w.value for w in at.warning]
           + [i.value for i in at.info] + [c.value for c in at.caption])
    return "\n".join(out)


def graph(at):
    return at.session_state["graph"]


def history(at):
    """Operations in the order they were run (the order the states were created)."""
    return [s.op for s in graph(at)]


def last_state(at):
    return list(graph(at))[-1]


def states(at, op):
    return [s for s in graph(at) if s.op == op]


@pytest.fixture
def at():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    goto(app, "📂 Data Management")
    ni = app.number_input
    ni[0].set_value(60); ni[1].set_value(1); ni[2].set_value(15)
    app.run()
    [b for b in app.button if "Generate synthetic" in (b.label or "")][0].click().run()
    assert app.session_state["survey"] is not None
    return app


@pytest.fixture
def at_shots(at):
    [b for b in at.button if "Build shot gathers" in (b.label or "")][0].click().run()
    assert at.session_state["shot_gathers"] is not None
    return at


# ─────────────────────────────────────────────────────────────────────────
# ⛔ — what a person sees when a genuine input is missing
# ─────────────────────────────────────────────────────────────────────────

def test_pages_say_what_is_missing_when_nothing_has_been_run(at):
    goto(at, PRESTACK)
    assert "Required input missing" in texts(at) and "shot gathers" in texts(at)
    assert not has_button(at, "run_bp")                 # no control that could not work
    goto(at, VEL)
    assert "CDP gathers" in texts(at)
    goto(at, STACK)
    assert "NMO-corrected" in texts(at)
    goto(at, POST)
    assert "stacked section" in texts(at)
    goto(at, MIG)
    assert "Required input missing" in texts(at)


def test_stack_is_blocked_until_moveout_is_corrected(at_shots):
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    goto(at_shots, VEL)
    click(at_shots, "run_vel1")
    # velocities exist, NMO has not been applied: the stack page refuses, and says how to fix it
    goto(at_shots, STACK)
    assert "Required input missing" in texts(at_shots)
    assert "NMO-corrected" in texts(at_shots)
    assert not [b for b in at_shots.button if "Run stack" in (b.label or "")]
    assert at_shots.session_state["stack_matrix"] is None
    assert "stack" not in history(at_shots)


def test_nmo_tab_needs_velocity_but_velocity_tab_only_needs_cdp_gathers(at_shots):
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    goto(at_shots, VEL)
    assert has_button(at_shots, "run_vel1")             # velocity analysis: ✓
    assert not has_button(at_shots, "run_nmo1")         # NMO: ⛔ no velocity yet
    assert "velocities" in texts(at_shots)
    click(at_shots, "run_vel1")
    assert has_button(at_shots, "run_nmo1")             # now runnable — and nothing else was needed


# ─────────────────────────────────────────────────────────────────────────
# ⚠ — recommendations ask "Continue?", never block
# ─────────────────────────────────────────────────────────────────────────

def test_agc_before_bandpass_asks_to_continue_then_runs(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_agc")
    # not run yet: the recommendation is shown and a confirmation is waiting
    assert "agc" not in history(at_shots)
    assert any("Continue?" in w.value for w in at_shots.warning)
    assert "Band-pass filtering is commonly applied before" in texts(at_shots)
    assert "has not been applied to the current data" in texts(at_shots)
    click(at_shots, "run_agc__continue")
    assert "agc" in history(at_shots)
    assert "Run despite" in last_state(at_shots).notes


def test_cancelling_the_confirmation_runs_nothing(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_agc")
    click(at_shots, "run_agc__cancel")
    assert "agc" not in history(at_shots)
    assert not any("Continue?" in w.value for w in at_shots.warning)


def test_task_order_agc_bandpass_mute_then_cdp_sort(at_shots):
    """Raw → Shot Gathers → AGC → Band-pass → Mute (→ CDP Sort)."""
    goto(at_shots, PRESTACK)
    click(at_shots, "run_agc"); click(at_shots, "run_agc__continue")           # ⚠ band-pass not applied
    click(at_shots, "run_bp"); click(at_shots, "run_bp__continue")             # ⚠ normally before AGC
    click(at_shots, "run_mute")                                                # ✓ (no confirmation)
    assert history(at_shots)[-3:] == ["agc", "bandpass", "mute"]
    agc, bp, mute = (states(at_shots, op)[-1] for op in ("agc", "bandpass", "mute"))
    assert bp.inputs == {"data": agc.id} and mute.inputs == {"data": bp.id}    # each read the previous output
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    assert at_shots.session_state["gathers"] is not None
    assert history(at_shots)[-1] == "cdp_sort" and last_state(at_shots).inputs == {"data": mute.id}


def test_task_order_bandpass_fk_then_cdp_sort_needs_no_confirmation(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    click(at_shots, "run_fk")
    goto(at_shots, CDP)
    btn = [b for b in at_shots.button if "Sort into CDP gathers" in (b.label or "")][0]
    btn.click().run()
    assert not at_shots.exception
    assert history(at_shots)[-3:] == ["bandpass", "fk", "cdp_sort"]      # ran straight away


def test_cdp_sort_with_no_conditioning_is_a_warning_not_a_block(at_shots):
    goto(at_shots, CDP)
    assert "No pre-stack conditioning" in texts(at_shots)
    sort_cdp(at_shots)
    assert at_shots.session_state["gathers"] is not None


# ─────────────────────────────────────────────────────────────────────────
# A whole skip-everything path, through the UI
# ─────────────────────────────────────────────────────────────────────────

def test_minimal_path_from_survey_to_final_section(at):
    """Survey → CDP sort (no shot gathers, no conditioning) → velocity 1 → NMO 1 →
    stack → migration → final scaling → export. Only genuine inputs are produced."""
    goto(at, CDP)
    sort_cdp(at)
    goto(at, VEL)
    click(at, "run_vel1")
    click(at, "run_nmo1")
    goto(at, STACK)
    [b for b in at.button if "Run stack" in (b.label or "")][0].click().run()
    assert "Final NMO has not been applied" in texts(at)             # ⚠, with the choice to go on
    click(at, "run_stack__continue")
    assert at.session_state["stack_matrix"] is not None
    goto(at, MIG)
    [b for b in at.button if "Run Kirchhoff migration" in (b.label or "")][0].click().run()
    click(at, "run_postmig__continue")
    assert at.session_state["stack_migrated"] is not None
    [b for b in at.button if "Run final scaling" in (b.label or "")][0].click().run()
    assert at.session_state["stack_final"] is not None
    goto(at, EXPORT)
    for skipped in ("shot_gathers", "bandpass", "statics", "dmo", "data_reg", "vel2", "nmo_final",
                    "rna", "decon_post", "tvf"):
        assert skipped not in history(at)
    # the whole line of states is there, each built from the one before
    chain = [s.op for s in graph(at).lineage(states(at, "final_scale")[-1].id)]
    assert chain == ["upload", "cdp_sort", "nmo1", "stack", "migration", "final_scale"]
    assert states(at, "final_scale")[-1].inputs == {"data": states(at, "migration")[-1].id}


def test_post_stack_steps_are_independent_in_the_ui(at):
    goto(at, CDP)
    sort_cdp(at)
    goto(at, VEL)
    click(at, "run_vel1"); click(at, "run_nmo1")
    goto(at, STACK)
    [b for b in at.button if "Run stack" in (b.label or "")][0].click().run()
    click(at, "run_stack__continue")
    goto(at, POST)
    click(at, "run_tvf"); click(at, "run_tvf__continue")           # TVF first (⚠ decon not applied)
    click(at, "run_rna")                                           # then RNA, on the TVF output (✓)
    assert history(at)[-2:] == ["tvf", "rna"]
    assert last_state(at).inputs == {"data": states(at, "tvf")[-1].id}


def test_prestack_migration_is_available_without_a_stack(at):
    goto(at, CDP)
    sort_cdp(at)
    goto(at, VEL)
    click(at, "run_vel1")
    goto(at, MIG)
    at.radio(key="mig_domain").set_value("Pre-stack (PSTM, on CDP gathers)").run()
    assert not at.exception
    at.slider(key="mig_aperture").set_value(150.0).run()
    at.slider(key="pstm_stride").set_value(10).run()
    assert has_button(at, "run_pstm")                              # no stack, no regularisation required
    click(at, "run_pstm")
    click(at, "run_pstm__continue")                                # ⚠ regularisation recommended
    assert at.session_state["stack_migrated"] is not None
    assert "stack" not in history(at)


# ─────────────────────────────────────────────────────────────────────────
# Data states: what each run consumed and produced
# ─────────────────────────────────────────────────────────────────────────

def test_a_run_says_which_state_it_produced_and_from_which(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    bp = states(at_shots, "bandpass")[-1]
    shot = states(at_shots, "shot_gathers")[-1]
    assert bp.inputs == {"data": shot.id}
    assert any(f"Produced **Band-pass ({bp.id})**" in s.value and shot.id in s.value for s in at_shots.success)
    # …and the page says what it will read next
    assert f"({bp.id})" in texts(at_shots) and "Reads" in texts(at_shots)


def test_an_identical_rerun_reuses_the_state_instead_of_computing_another(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    n = len(graph(at_shots))
    # the data is now band-passed; the same band-pass on the same input is only reachable by choosing
    # that input, and then it is recognised as already done
    sg = states(at_shots, "shot_gathers")[-1]
    at_shots.selectbox(key="pick_bandpass_None_data").set_value(sg.id).run()
    click(at_shots, "run_bp")
    assert len(graph(at_shots)) == n and len(states(at_shots, "bandpass")) == 1
    assert at_shots.session_state["last_run"]["reused"] is True
    assert any("Identical" in s.value for s in at_shots.success)


def test_choosing_an_earlier_input_state_branches_instead_of_stacking_on_top(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    sg = states(at_shots, "shot_gathers")[-1]
    bp = states(at_shots, "bandpass")[-1]
    # AGC on the *raw* gathers although a band-passed state exists
    at_shots.selectbox(key="pick_agc_None_data").set_value(sg.id).run()
    assert not at_shots.exception
    click(at_shots, "run_agc")
    click(at_shots, "run_agc__continue")                     # ⚠ band-pass is normally applied first
    agc = states(at_shots, "agc")[-1]
    assert agc.inputs == {"data": sg.id} and bp.materialized
    assert {d.id for d in graph(at_shots).dependents(sg.id)} == {bp.id, agc.id}   # two branches of S1


def test_repeating_an_operation_is_a_warning_and_a_new_version(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_agc"); click(at_shots, "run_agc__continue")
    click(at_shots, "run_agc")                                # again, on its own output
    assert any("Continue?" in w.value for w in at_shots.warning)
    click(at_shots, "run_agc__continue")
    agc1, agc2 = states(at_shots, "agc")
    assert agc2.version == 2 and agc2.inputs == {"data": agc1.id}


# ─────────────────────────────────────────────────────────────────────────
# Re-running keeps old states; what is built on superseded data is flagged stale
# ─────────────────────────────────────────────────────────────────────────

def test_rerunning_velocity_with_new_parameters_keeps_the_old_states_and_flags_them(at_shots):
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    goto(at_shots, VEL)
    at_shots.number_input(key="vel1_nanalysis").set_value(3).run()
    click(at_shots, "run_vel1")
    click(at_shots, "run_nmo1")
    assert len(states(at_shots, "vel1")) == 1 and "⟳" not in texts(at_shots)
    at_shots.number_input(key="vel1_nanalysis").set_value(4).run()
    click(at_shots, "run_vel1")
    v1, v2 = states(at_shots, "vel1")
    nmo = states(at_shots, "nmo1")[0]
    assert v2.version == 2 and nmo.materialized               # the NMO of the first velocities is still there…
    assert nmo.inputs["velocity"] == v1.id
    assert "⟳" in texts(at_shots) and "re-run" in texts(at_shots)      # …and the page says it is out of date
    click(at_shots, "run_nmo1")                               # NMO with the new velocities → a second state
    assert states(at_shots, "nmo1")[-1].inputs["velocity"] == v2.id


def test_rerunning_velocity_unchanged_does_not_make_another_state(at_shots):
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    goto(at_shots, VEL)
    at_shots.number_input(key="vel1_nanalysis").set_value(3).run()
    click(at_shots, "run_vel1")
    click(at_shots, "run_vel1")
    assert len(states(at_shots, "vel1")) == 1
    assert at_shots.session_state["last_run"]["reused"] is True


# ─────────────────────────────────────────────────────────────────────────
# Undo / redo / choose / discard from the page
# ─────────────────────────────────────────────────────────────────────────

def test_undo_and_redo_move_between_states_without_recomputing(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    click(at_shots, "run_agc")
    bp, agc = states(at_shots, "bandpass")[-1], states(at_shots, "agc")[-1]
    n = len(graph(at_shots))
    assert graph(at_shots).current("shot").id == agc.id
    click(at_shots, "states_shot_undo_shot")
    assert graph(at_shots).current("shot").id == bp.id and len(graph(at_shots)) == n
    click(at_shots, "states_shot_redo_shot")
    assert graph(at_shots).current("shot").id == agc.id and len(graph(at_shots)) == n


def test_the_next_step_reads_the_state_that_undo_went_back_to(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    click(at_shots, "run_agc")
    click(at_shots, "states_shot_undo_shot")                  # back to the band-passed data
    click(at_shots, "run_mute")
    bp = states(at_shots, "bandpass")[-1]
    assert last_state(at_shots).op == "mute" and last_state(at_shots).inputs == {"data": bp.id}


def test_discarding_a_state_from_the_page_removes_its_dependents(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    click(at_shots, "run_agc")
    click(at_shots, "run_mute")
    bp = states(at_shots, "bandpass")[-1]
    agc = states(at_shots, "agc")[-1]
    at_shots.selectbox(key="states_shot_pick").set_value(agc.id).run()
    click(at_shots, "states_shot_discard")
    assert history(at_shots)[-1] == "bandpass" and "agc" not in history(at_shots) and "mute" not in history(at_shots)
    assert graph(at_shots).current("shot").id == bp.id


def test_freeing_memory_from_the_page_keeps_the_record_and_it_comes_back(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    click(at_shots, "run_agc")
    bp = states(at_shots, "bandpass")[-1]
    at_shots.selectbox(key="states_shot_pick").set_value(bp.id).run()
    click(at_shots, "states_shot_release")
    assert not graph(at_shots)[bp.id].materialized                    # stays released while the page is drawn
    assert "released to save memory" in texts(at_shots)
    assert graph(at_shots)[bp.id].keys["traces"]                      # the record is intact
    click(at_shots, f"recompute_{bp.id}_bandpass")                    # …and can be brought back from the page
    assert graph(at_shots)[bp.id].materialized
    # undo onto a released state re-computes it on the way
    at_shots.selectbox(key="states_shot_pick").set_value(bp.id).run()
    click(at_shots, "states_shot_release")
    assert not graph(at_shots)[bp.id].materialized
    click(at_shots, "states_shot_undo_shot")
    assert graph(at_shots)[bp.id].materialized and graph(at_shots).current("shot").id == bp.id


# ─────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────

def test_home_page_and_sidebar_show_status_symbols(at_shots):
    goto(at_shots, "🏠 Home")
    shown = "\n".join(c.value for c in at_shots.caption)
    assert "✓" in shown and "⛔" in shown


def test_the_export_page_offers_the_state_history_and_a_recipe(at_shots):
    goto(at_shots, PRESTACK)
    click(at_shots, "run_bp")
    goto(at_shots, EXPORT)
    assert not at_shots.exception
    shown = texts(at_shots) + "\n".join(m.value for m in at_shots.markdown)
    assert "S" in shown and "Band-pass" in shown
