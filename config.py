"""
config.py — Application-wide constants and default processing parameters
==========================================================================
No Streamlit dependency. Central place for defaults so every view/module
uses the same numbers, with the rationale for each documented alongside
it (see the comment on each block).
"""

APP_TITLE = "Seismic Playground"
APP_TAGLINE = "An Interactive Seismic Data Processing & Interpretation Platform"
APP_ICON = "🌊"
APP_VERSION = "1.0.0"

# ── Minimum traces to keep a CDP gather (drops near-edge, low-fold CDPs) ────
N_MIN_TRACES_DEFAULT = 10

# ── 4) Amplitude recovery (AGC) ──────────────────────────────────────────
AGC_WINDOW_DEFAULT = 250          # samples
AGC_MODE_DEFAULT = "rms"          # rms | mean | median

# ── 5) Deconvolution before stack (spiking / Wiener-Hopf) ───────────────
DECON_OPLEN_DEFAULT = 30          # samples
DECON_PW_DEFAULT = 0.01           # pre-whitening fraction (1%)

# ── 6) Band-pass filter ──────────────────────────────────────────────────
FMIN_DEFAULT = 12.0                # Hz
FMAX_DEFAULT = 75.0                # Hz
LOW_SLOPE_DEFAULT = 6.0            # Hz taper below FMIN
HIGH_SLOPE_DEFAULT = 12.0          # Hz taper above FMAX
SMOOTH_WIDTH_HZ_DEFAULT = 2.0      # Hann-kernel smoothing width

# ── 7) Trace balancing ───────────────────────────────────────────────────
NORM_WINDOW_S_DEFAULT = 1.5        # seconds, from t=0

# ── 8) Top mute ───────────────────────────────────────────────────────────
MUTE_TTOP_DEFAULT = 0.05           # seconds
MUTE_TAPER_DEFAULT = 0.05          # seconds

# ── 9) F-K dip filter ─────────────────────────────────────────────────────
# Ground roll/direct-wave/refraction noise has a characteristic LOW
# apparent velocity compared to primary reflections, and an event of
# apparent velocity V traces the straight line |k| = f / V in the f-k
# domain — so rather than an arbitrary hand-picked polygon, the default
# separation boundary here is generated directly from a single,
# physically meaningful apparent-velocity cutoff via
# `modules.filtering.apparent_velocity_wedge`. Events slower than this
# cutoff (ground roll, direct wave, low-velocity refractions) fall
# outside the resulting central wedge; primary reflections (higher
# apparent velocity) fall inside it.
FK_VELOCITY_CUTOFF_DEFAULT = 1800.0    # m/s
# Chosen with a safety margin below the shallowest default reflector's
# far-offset apparent velocity (~2077 m/s at t0=0.35s, v_nmo=1900 m/s,
# x=1500m, via v_apparent(x) = v_nmo^2 * t(x) / x) so that reflector's
# far-offset flank is not softened by the accept-wedge taper.
# "Accept" (encloses the signal, i.e. the wedge above) is the preferred
# convention in practice, since it's easier to enclose the signal zone
# than to trace the noise region's own boundary; "reject" (encloses the
# noise, i.e. everything outside the wedge) is the alternative, giving
# identical results for the same boundary.
FK_POLYGON_MODE_DEFAULT = "accept"
# Amplitudes on the "wrong" side of the boundary are multiplied by a small
# coefficient rather than an exact zero — multiplying by a constant in the
# frequency domain scales the time-domain signal by the inverse of that
# constant, and the inverse of zero is infinite, so an exact-zero cutoff
# would blow up the filtered trace rather than just attenuating it.
FK_REJECT_COEFFICIENT_DEFAULT = 0.02

# ── 10/15) Velocity analysis (semblance) ─────────────────────────────────
VELOCITY_MIN_DEFAULT = 1500.0      # m/s
VELOCITY_MAX_DEFAULT = 4500.0      # m/s
VELOCITY_N_DEFAULT = 60
SEMBLANCE_WINDOW_LEN_DEFAULT = 5   # half-window, samples
N_ANALYSIS_CDPS_DEFAULT = 9
MIN_ABS_OFFSET_DEFAULT = 700.0     # m, minimum far offset required at an analysis CDP

# Auto-picker (confidence-gated local walk — see modules/velocity.py
# docstring for the rationale behind this over a whole-path DP picker)
AUTOPICK_MIN_TIME_DEFAULT = 0.12          # s, ignore very early (direct-arrival) time
AUTOPICK_SEARCH_HALF_WIDTH_MS_DEFAULT = 600.0  # m/s, local search window around the current pick
AUTOPICK_MIN_CONFIDENCE_DEFAULT = 0.12    # minimum semblance to accept a move
AUTOPICK_SMOOTH_SIGMA_DEFAULT = 8         # samples, final smoothing
RELIABILITY_THRESHOLD_DEFAULT = 0.15      # max tolerated relative deviation from the
                                           # lateral median trend before a pick is flagged

# ── 11/16) NMO ────────────────────────────────────────────────────────────
NMO_STRETCH_MUTE_DEFAULT = 0.30    # fractional stretch cutoff

# ── 12) Residual statics ─────────────────────────────────────────────────
MAX_STATIC_SHIFT_S_DEFAULT = 0.02  # s, +/- search window

# ── 18) Post-stack random noise attenuation ──────────────────────────────
RNA_SPATIAL_WINDOW_DEFAULT = 3     # traces (spatial median filter)

# ── 19) Deconvolution after stack ────────────────────────────────────────
DECON_AFTER_OPLEN_DEFAULT = 16
DECON_AFTER_PW_DEFAULT = 0.05

# ── 20) Time-variant filter ───────────────────────────────────────────────
TVF_ZONES_DEFAULT = [
    (0.0, 1.0, 15.0, 90.0),
    (1.0, 2.0, 10.0, 65.0),
    (2.0, 3.01, 8.0, 45.0),
]
TVF_TAPER_S_DEFAULT = 0.15

# ── 21) Kirchhoff migration ───────────────────────────────────────────────
APERTURE_M_DEFAULT = 900.0         # m, half-aperture

# ── 22) Final scaling ─────────────────────────────────────────────────────
FINAL_AGC_WINDOW_DEFAULT = 300     # samples

# ── Display ────────────────────────────────────────────────────────────
SEISMIC_CMAP = "Seismic"           # high-contrast navy/white/maroon diverging scale (see visualization/theme.py)
PCLIP_DEFAULT = 98.0               # percentile amplitude clip for display

# ── Performance guardrails ───────────────────────────────────────────────
MAX_TRACES_FULL_PROCESS = 20000    # above this, the UI nudges the user toward a windowed subset
LARGE_DATASET_WARN_MB = 200
