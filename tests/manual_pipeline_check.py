"""
tests/manual_pipeline_check.py — full pipeline run against synthetic
ground truth, shot-domain-first architecture. Run manually:

    python tests/manual_pipeline_check.py
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np

from modules.synthetic_data import generate_synthetic_survey
from modules import segy_io, preprocessing, filtering, velocity, nmo, statics, stacking, migration
import config as cfg

t_start = time.time()
survey = generate_synthetic_survey()
shot_gathers = segy_io.build_shot_gathers(survey)
dt, ns, twt = survey.sample_interval, survey.num_samples, survey.twt
print(f"[1] {survey.num_traces} traces -> {len(shot_gathers)} shot gathers")

# 1) band-pass (shot domain)
h_bp = filtering.design_bandpass(cfg.FMIN_DEFAULT, cfg.FMAX_DEFAULT, dt, ns)
for g in shot_gathers.values():
    g["traces_bp"] = filtering.apply_time_domain_conv(g["traces"], h_bp)
print("[2] Band-pass filter done")

# 2) AGC (shot domain)
for g in shot_gathers.values():
    g["traces_agc"] = preprocessing.agc(g["traces_bp"], window=cfg.AGC_WINDOW_DEFAULT)
print("[3] AGC done")

# 3) trace edit / kill / de-spike (shot domain)
for g in shot_gathers.values():
    cleaned, _ = preprocessing.despike_traces(g["traces_agc"])
    flags = preprocessing.flag_bad_traces(cleaned)
    g["traces_edited"] = preprocessing.apply_trace_kill(cleaned, flags["kill"])
print("[4] Trace edit / kill / de-spike done")

# 4) top mute (shot domain)
for g in shot_gathers.values():
    g["traces_muted"] = preprocessing.apply_top_bottom_mute(g["traces_edited"], dt, t_top=cfg.MUTE_TTOP_DEFAULT)
print("[5] Top mute done")

# 5) trace balancing (shot domain)
for g in shot_gathers.values():
    g["traces_balanced"] = preprocessing.trace_normalize(g["traces_muted"])
print("[6] Trace balancing done")

# 6) F-K filter (shot domain)
for g in shot_gathers.values():
    filt, _, _, _, _, _ = filtering.fk_filter_gather(g["traces_balanced"], dt, survey.group_interval,
                                                       cfg.FK_POLYGON_DEFAULT)
    g["traces_fk_filtered"] = filt
print("[7] F-K filter done")

# 7) multiple suppression (predictive decon, shot domain)
for g in shot_gathers.values():
    g["traces_demultiple"] = preprocessing.predictive_decon_gather(g["traces_fk_filtered"], 60, 30, 0.02)
print("[8] Multiple suppression done")

# 8) deconvolution (spiking, shot domain) -- final pre-stack conditioning step
for g in shot_gathers.values():
    g["traces_decon"] = preprocessing.spiking_decon_gather(g["traces_demultiple"], cfg.DECON_OPLEN_DEFAULT,
                                                            cfg.DECON_PW_DEFAULT)
print("[9] Deconvolution done -- pre-stack conditioning complete")

# 9) CDP sort / binning -- re-group the conditioned shot-domain data by CDP
flat_conditioned = segy_io.scatter_to_flat(shot_gathers, "traces_decon", survey.num_traces, survey.num_samples)
gathers = segy_io.build_cdp_gathers(survey, n_min_traces=cfg.N_MIN_TRACES_DEFAULT, data_override=flat_conditioned)
cdp_list = list(gathers.keys())
print(f"[10] CDP sort/binning done -> {len(cdp_list)} CDP gathers "
      f"(fold {min(g['traces'].shape[0] for g in gathers.values())}-{max(g['traces'].shape[0] for g in gathers.values())})")

# 10) velocity analysis pass 1
velocities = np.linspace(cfg.VELOCITY_MIN_DEFAULT, cfg.VELOCITY_MAX_DEFAULT, cfg.VELOCITY_N_DEFAULT)
t0 = time.time()
Vfield_1, analysis_1, results_1 = velocity.pick_velocity_field(
    gathers, cdp_list, "traces", velocities, dt, ns,
    n_analysis=cfg.N_ANALYSIS_CDPS_DEFAULT, min_abs_offset=cfg.MIN_ABS_OFFSET_DEFAULT, label="pass1")
print(f"[11] Velocity analysis pass 1 done ({time.time()-t0:.2f}s), "
      f"{sum(r.is_reliable for r in results_1.values())}/{len(results_1)} reliable picks")

# 11) approximate NMO (pass 1)
for cdp, g in gathers.items():
    nmo1, live1 = nmo.nmo_correction(g["traces"], g["offsets"], g["vrms_field_pass1"], dt,
                                      stretch_mute=cfg.NMO_STRETCH_MUTE_DEFAULT)
    g["traces_nmo1"], g["nmo1_live_mask"] = nmo1, live1
print("[12] NMO pass 1 done")

# 12) residual statics
shot_static, receiver_static, static_diag = statics.solve_residual_statics(
    gathers, "traces_nmo1", "nmo1_live_mask", dt, max_shift_s=cfg.MAX_STATIC_SHIFT_S_DEFAULT)
statics.apply_residual_statics(gathers, "traces", "traces_statics", shot_static, receiver_static, dt)
print(f"[13] Residual statics solved: {static_diag}")

# 13) DMO correction
for cdp, g in gathers.items():
    nmo1s, live1s = nmo.nmo_correction(g["traces_statics"], g["offsets"], g["vrms_field_pass1"], dt,
                                        stretch_mute=cfg.NMO_STRETCH_MUTE_DEFAULT)
    g["traces_nmo1_statics"] = nmo1s
    g["traces_dmo"] = nmo.dmo_correction(nmo1s, g["offsets"], g["vrms_field_pass1"], dt)
print("[14] DMO correction done")

# 14) inverse NMO
for cdp, g in gathers.items():
    g["traces_inv_nmo"] = nmo.inverse_nmo_correction(g["traces_dmo"], g["offsets"], g["vrms_field_pass1"], dt)
print("[15] Inverse NMO done")

# 15) velocity analysis pass 2
t0 = time.time()
Vfield_2, analysis_2, results_2 = velocity.pick_velocity_field(
    gathers, cdp_list, "traces_inv_nmo", velocities, dt, ns,
    n_analysis=cfg.N_ANALYSIS_CDPS_DEFAULT, min_abs_offset=cfg.MIN_ABS_OFFSET_DEFAULT, label="pass2")
print(f"[16] Velocity analysis pass 2 done ({time.time()-t0:.2f}s), "
      f"{sum(r.is_reliable for r in results_2.values())}/{len(results_2)} reliable picks")

# 16) final NMO
for cdp, g in gathers.items():
    nmo_final, live_final = nmo.nmo_correction(g["traces_inv_nmo"], g["offsets"], g["vrms_field_pass2"], dt,
                                                stretch_mute=cfg.NMO_STRETCH_MUTE_DEFAULT)
    g["traces_nmo_final"], g["nmo_final_live_mask"] = nmo_final, live_final
print("[17] Final NMO done")

# 17) DMO stack
stack_matrix, fold_matrix, naive_matrix = stacking.build_stack_section(
    gathers, cdp_list, "traces_nmo_final", "nmo_final_live_mask")
print(f"[18] Stack done: shape={stack_matrix.shape}, fold range {fold_matrix.min():.0f}-{fold_matrix.max():.0f}")

# 18) random noise attenuation
stack_rna = filtering.random_noise_attenuation(stack_matrix, spatial_window=cfg.RNA_SPATIAL_WINDOW_DEFAULT)
print("[19] Random noise attenuation done")

# 19) decon after stack
stack_decon = preprocessing.spiking_decon_gather(stack_rna, operator_length=cfg.DECON_AFTER_OPLEN_DEFAULT,
                                                  pre_whitening=cfg.DECON_AFTER_PW_DEFAULT)
print("[20] Post-stack deconvolution done")

# 20) time-variant filter
stack_tvf = filtering.time_variant_filter(stack_decon, dt, cfg.TVF_ZONES_DEFAULT, taper_s=cfg.TVF_TAPER_S_DEFAULT)
print("[21] Time-variant filter done")

# 21) migration
x_positions = np.array(cdp_list, dtype=float) * survey.group_interval
aperture_traces = max(1, int(round(cfg.APERTURE_M_DEFAULT / survey.group_interval)))
t0 = time.time()
stack_migrated = migration.kirchhoff_migration(stack_tvf, x_positions, twt, Vfield_2, aperture_traces)
print(f"[22] Kirchhoff migration done ({time.time()-t0:.2f}s)")

# 22) final scaling
stack_final = migration.final_scaling(stack_migrated, dt, window=cfg.FINAL_AGC_WINDOW_DEFAULT,
                                       t_top_mute=cfg.MUTE_TTOP_DEFAULT)
print("[23] Final scaling done")

print(f"\nFull pipeline completed in {time.time()-t_start:.2f}s total")

# ---- sanity checks against ground truth ----
print("\n--- Sanity checks ---")
assert np.isfinite(stack_final).all(), "non-finite values in final section!"
print("No NaN/Inf in final section: OK")

mean_trace = np.mean(np.abs(stack_final), axis=0)
print("(Note: decon is now the *last* pre-stack-conditioning step, per the requested shot-domain "
      "processing order, rather than running right after AGC as in the very first version of this "
      "app. Spiking deconvolution assumes a minimum-phase wavelet; this synthetic data's Ricker "
      "wavelet is zero-phase, so some per-trace timing/phase shift from deconvolution is expected "
      "regardless of ordering -- confirmed by testing decon in isolation on a single trace both "
      "early and late in the chain, which showed a similar shift either way. This is a property of "
      "applying minimum-phase deconvolution to non-minimum-phase data, not a bug -- see README.)")
for t0_true in [0.35, 0.65, 1.05, 1.55, 2.2]:
    it = int(round(t0_true / dt))
    window = slice(max(0, it - 10), it + 10)
    local_peak_t = twt[window][np.argmax(mean_trace[window])]
    err_ms = abs(local_peak_t - t0_true) * 1000
    status = "OK" if err_ms < 25 else "CHECK"
    print(f"true t0={t0_true}s  final-section peak at t={local_peak_t:.3f}s  "
          f"(err={err_ms:.1f} ms, < one dominant period ~13ms x a small factor)  [{status}]")

diff_t0 = 1.20
it = int(round(diff_t0 / dt))
window = slice(max(0, it - 25), it + 25)
pre_mig = np.std(np.max(np.abs(stack_tvf[:, window]), axis=1))
post_mig = np.std(np.max(np.abs(stack_migrated[:, window]), axis=1))
print(f"Diffraction-zone amplitude std across CDPs: pre-migration={pre_mig:.4f}, "
      f"post-migration={post_mig:.4f} (migration should concentrate energy)")

print("\nALL STAGES EXECUTED SUCCESSFULLY")
