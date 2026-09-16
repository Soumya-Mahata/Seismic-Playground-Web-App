"""
modules/migration.py — Steps 21 & 22 of the reference pipeline
==================================================================
Post-stack (zero-offset / exploding-reflector) Kirchhoff time migration
and the final display-gain scaling pass.

Ported directly from `Seismic_Processing_CORRECTED.ipynb`
(`kirchhoff_migration`). No change to the math. Vectorized across time
per contributing trace (the diffraction travel-time curve is computed
for all output times at once), so a full section migrates in well under
a second for a typical small teaching/demo line.
"""

from __future__ import annotations

import numpy as np

from modules.preprocessing import agc, mute_mask


def kirchhoff_migration(section: np.ndarray, x_positions: np.ndarray, t: np.ndarray,
                         velocity_field_2d: np.ndarray, aperture_traces: int) -> np.ndarray:
    """Sum each output trace's diffraction contributions from every
    neighbouring trace within `aperture_traces`, using the diffraction
    travel-time curve t_travel = sqrt(t^2 + (2*dx/v)^2) and an
    obliquity/amplitude weight of t / t_travel.

    `velocity_field_2d`: (n_samples, n_cdp) migration-velocity field
    (typically the 2nd-pass refined stacking-velocity field).
    """
    n_cdp, nt = section.shape
    migrated = np.zeros_like(section)
    for i0 in range(n_cdp):
        lo, hi = max(0, i0 - aperture_traces), min(n_cdp, i0 + aperture_traces + 1)
        v_col = velocity_field_2d[:, i0]
        accum = np.zeros(nt)
        for xi in range(lo, hi):
            dxi = x_positions[xi] - x_positions[i0]
            t_travel = np.sqrt(t ** 2 + (2.0 * dxi / v_col) ** 2)
            valid = t_travel <= t[-1]
            contrib = np.interp(t_travel, t, section[xi, :])
            weight = np.where(valid, t / np.maximum(t_travel, 1e-6), 0.0)
            accum += contrib * weight
        migrated[i0, :] = accum
    return migrated


def final_scaling(migrated: np.ndarray, dt: float, window: int = 300,
                   t_top_mute: float = 0.05, taper_s: float = 0.05) -> np.ndarray:
    """Step 22: a light final AGC pass restores a balanced, presentation-
    ready amplitude range after migration's obliquity weighting (which
    otherwise leaves shallow energy visually dominant), then re-applies
    the top mute — migration's diffraction summation can pull energy
    from neighbouring live traces back into the originally-muted shallow
    zone, so standard practice is to clean this up post-migration."""
    scaled = agc(migrated, window=window, mode="rms")
    mask = mute_mask(migrated.shape[1], dt, t_top=t_top_mute, taper_s=taper_s)
    return scaled * mask[np.newaxis, :]
