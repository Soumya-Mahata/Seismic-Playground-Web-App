"""
End-to-end AppTest walkthrough: generate synthetic data, build shot
gathers, run through every processing stage via simulated widget
interactions (pre-stack conditioning in the shot domain, CDP sort,
velocity/NMO/DMO, stacking, post-stack, migration, interpretation,
attributes, export), and check for exceptions at each step. Not a unit
test suite — a scripted smoke test of the real interactive flow.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest


def check(at, label):
    if at.exception:
        print(f"\n❌ EXCEPTION after: {label}")
        for e in at.exception:
            print(e)
        raise SystemExit(1)
    print(f"✅ {label}")


at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
at.run()
check(at, "initial load")

# ---- Data Management: generate synthetic data (small, for a fast test) ----
at.radio(key="active_page").set_value("📂 Data Management").run()
check(at, "nav to Data Management")

demo_inputs = at.number_input
demo_inputs[0].set_value(60)   # stations
demo_inputs[1].set_value(1)    # shot interval (kept dense enough to clear the default min-fold)
demo_inputs[2].set_value(15)   # channels per side
at.run()
check(at, "set synthetic params")

gen_btn = [b for b in at.button if "Generate synthetic" in (b.label or "")]
assert gen_btn, "generate button not found"
gen_btn[0].click().run()
check(at, "generate synthetic survey")
assert at.session_state["survey"] is not None, "survey not loaded"
print(f"   survey: {at.session_state['survey'].num_traces} traces")

shot_btn = [b for b in at.button if "Build shot gathers" in (b.label or "")]
assert shot_btn, "build shot gathers button not found"
shot_btn[0].click().run()
check(at, "build shot gathers")
assert at.session_state["shot_gathers"] is not None
print(f"   shot gathers: {len(at.session_state['shot_gathers'])} shots")

# ---- Pre-Stack Conditioning (shot domain): all 8 steps, in order ----
at.radio(key="active_page").set_value("⚙️ Pre-Stack Conditioning").run()
check(at, "nav to Pre-Stack Conditioning")

for run_key, label in [
    ("run_bp", "Band-pass"), ("run_agc", "AGC"), ("run_despike", "De-spike"),
    ("run_kill", "Trace kill"), ("run_mute", "Mute"), ("run_bal", "Balance"),
    ("run_fk", "F-K filter (default polygon)"),
    ("run_ms", "Multiple suppression"), ("run_decon", "Deconvolution"),
]:
    btn = at.button(key=run_key)
    btn.click().run()
    check(at, f"run {label}")

# ---- CDP Sort & Binning ----
at.radio(key="active_page").set_value("🔀 CDP Sort & Binning").run()
check(at, "nav to CDP Sort & Binning")
sort_btn = [b for b in at.button if "Sort into CDP gathers" in (b.label or "")]
assert sort_btn
sort_btn[0].click().run()
check(at, "CDP sort/binning")
assert at.session_state["gathers"] is not None
print(f"   CDP gathers: {len(at.session_state['cdp_list'])} CDPs")

# ---- Velocity / NMO / DMO ----
at.radio(key="active_page").set_value("🎯 Velocity, NMO & DMO").run()
check(at, "nav to Velocity/NMO/DMO")

at.button(key="run_vel1").click().run()
check(at, "velocity pass 1")
assert at.session_state["Vfield_1"] is not None

at.button(key="run_nmo1").click().run()
check(at, "NMO pass 1")

at.button(key="run_statics").click().run()
check(at, "residual statics")

at.button(key="run_dmo").click().run()
check(at, "DMO")

at.button(key="run_invnmo").click().run()
check(at, "inverse NMO")

at.button(key="run_datareg").click().run()
check(at, "data regularisation")

at.button(key="run_vel2").click().run()
check(at, "RMS velocity analysis (pass 2)")
assert at.session_state["Vfield_2"] is not None

at.button(key="run_nmofinal").click().run()
check(at, "final NMO")

# ---- Stacking ----
at.radio(key="active_page").set_value("📚 Stacking").run()
check(at, "nav to Stacking")
run_stack = [b for b in at.button if "Run stack" in (b.label or "")]
run_stack[0].click().run()
check(at, "stack")
assert at.session_state["stack_matrix"] is not None

# ---- Post-Stack Processing ----
at.radio(key="active_page").set_value("🧹 Post-Stack Processing").run()
check(at, "nav to Post-Stack Processing")
at.button(key="run_rna").click().run()
check(at, "RNA")
at.button(key="run_decon2").click().run()
check(at, "post-stack decon")
at.button(key="run_tvf").click().run()
check(at, "TVF")

# ---- Migration ----
at.radio(key="active_page").set_value("🗺️ Migration").run()
check(at, "nav to Migration")
mig_btn = [b for b in at.button if "Run Kirchhoff migration" in (b.label or "")]
mig_btn[0].click().run()
check(at, "migration")
assert at.session_state["stack_migrated"] is not None
scale_btn = [b for b in at.button if "Run final scaling" in (b.label or "")]
scale_btn[0].click().run()
check(at, "final scaling")
assert at.session_state["stack_final"] is not None

# ---- Interpretation ----
at.radio(key="active_page").set_value("🧭 Interpretation").run()
check(at, "nav to Interpretation")

# ---- Attributes ----
at.radio(key="active_page").set_value("🧬 Attributes").run()
check(at, "nav to Attributes")
compute_btn = [b for b in at.button if "Compute attribute" in (b.label or "")]
compute_btn[0].click().run()
check(at, "compute attribute")

# ---- Export ----
at.radio(key="active_page").set_value("💾 Export").run()
check(at, "nav to Export")

print("\n🎉 FULL END-TO-END APP WALKTHROUGH SUCCEEDED — no exceptions at any stage")
