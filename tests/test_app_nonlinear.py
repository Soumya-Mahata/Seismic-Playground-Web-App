"""
tests/test_app_nonlinear.py — the dependency-aware UI, driven headlessly
============================================================================
Where `tests/test_app_flow.py` walks the recommended order end to end, this
drives the *Streamlit app itself* through orders the old linear gates
refused, and checks the three states a person sees:

    ✓ Available  /  ⚠ Recommended prerequisite not performed  /  ⛔ Required input missing

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
    """Everything the page says in warnings / errors / captions."""
    out = [e.value for e in at.error] + [w.value for w in at.warning] + [c.value for c in at.caption]
    return "\n".join(out)


def history(at):
    return [h.stage_id for h in sorted(at.session_state["history"], key=lambda e: e.seq)]


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
    assert "Run despite" in at_shots.session_state["history"][-1].notes


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
    assert at_shots.session_state["history"][-2].consumed == {"data": "agc"}
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    assert at_shots.session_state["gathers"] is not None
    assert history(at_shots)[-1] == "cdp_sort"


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
    assert at.session_state["history"][-1].consumed == {"data": "tvf"}


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
# Re-running discards only what was built on it
# ─────────────────────────────────────────────────────────────────────────

def test_rerunning_velocity_discards_downstream_steps_in_the_ui(at_shots):
    goto(at_shots, CDP)
    sort_cdp(at_shots)
    goto(at_shots, VEL)
    click(at_shots, "run_vel1")
    click(at_shots, "run_nmo1")
    assert "nmo1" in history(at_shots)
    click(at_shots, "run_vel1")
    assert "nmo1" not in history(at_shots) and "vel1" in history(at_shots)


def test_home_page_and_sidebar_show_status_symbols(at_shots):
    goto(at_shots, "🏠 Home")
    shown = "\n".join(c.value for c in at_shots.caption)
    assert "✓" in shown and "⛔" in shown
