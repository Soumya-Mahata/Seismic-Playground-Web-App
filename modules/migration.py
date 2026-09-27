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


def prestack_kirchhoff_time_migration(gathers: dict, cdp_list: list, input_key: str, twt: np.ndarray,
                                       velocity_field_2d: np.ndarray, group_interval: float,
                                       aperture_traces: int, offsets_key: str = "offsets",
                                       cdp_stride: int = 1, progress_cb=None) -> tuple:
    """Pre-Stack (Kirchhoff) Time Migration — PSTM, report Sec 6.13.3:
    "preserves all offset information ... standard workflow" — vs. the
    post-stack `kirchhoff_migration` above, which only ever sees an
    already-stacked (approximately zero-offset) trace per CDP.

    Migrates directly from offset-preserving CDP gathers using the
    double-square-root (DSR) prestack traveltime for a diffractor at
    depth z = V*t0/2 below output point x0, source at (midpoint - h) and
    receiver at (midpoint + h):

        t(xm, h; x0, t0) = [ sqrt(z^2 + (xm - h - x0)^2)
                            + sqrt(z^2 + (xm + h - x0)^2) ] / V

    summed over every live trace of every gather within the lateral
    aperture of each output CDP, then fold-normalized — i.e. this
    performs the migration-and-stack in one pass, operating on
    individual prestack traces rather than on precomputed common-offset
    volumes, which keeps it compatible with this app's per-CDP-gather
    data structure. `cdp_stride` migrates only every Nth output CDP
    (the rest are left at 0) to keep runtime manageable on larger lines;
    this is a documented simplification, not a numerical shortcut inside
    the summation itself.

    Returns (migrated_image, fold_image), both (n_cdp, n_samples).
    """
    n_cdp = len(cdp_list)
    nt = len(twt)
    x_positions = np.array(cdp_list, dtype=float) * group_interval
    migrated = np.zeros((n_cdp, nt))
    fold = np.zeros((n_cdp, nt))
    out_indices = list(range(0, n_cdp, max(1, cdp_stride)))
    for count, i0 in enumerate(out_indices):
        x0 = x_positions[i0]
        v_col = np.maximum(velocity_field_2d[:, i0], 50.0)
        z_col = v_col * twt / 2.0
        lo, hi = max(0, i0 - aperture_traces), min(n_cdp, i0 + aperture_traces + 1)
        accum = np.zeros(nt)
        foldc = np.zeros(nt)
        for j in range(lo, hi):
            g = gathers[cdp_list[j]]
            traces = g[input_key]
            offs = np.asarray(g[offsets_key], dtype=float)
            xm = x_positions[j]
            dxm = xm - x0
            for k in range(traces.shape[0]):
                h = offs[k] / 2.0
                d1 = np.sqrt(z_col ** 2 + (dxm - h) ** 2)
                d2 = np.sqrt(z_col ** 2 + (dxm + h) ** 2)
                t_travel = (d1 + d2) / v_col
                valid = t_travel <= twt[-1]
                contrib = np.interp(t_travel, twt, traces[k, :])
                weight = np.where(valid, twt / np.maximum(t_travel, 1e-6), 0.0)
                accum += contrib * weight
                foldc += (weight > 0).astype(float)
        migrated[i0, :] = accum
        fold[i0, :] = foldc
        if progress_cb:
            progress_cb((count + 1) / len(out_indices))
    with np.errstate(divide="ignore", invalid="ignore"):
        migrated_norm = np.where(fold > 0, migrated / np.maximum(fold, 1.0), 0.0)
    if cdp_stride > 1:
        # fill the skipped output CDPs by linear interpolation from the migrated ones,
        # so the display isn't full of literal zero stripes
        migrated_idx = np.array(out_indices)
        for it in range(nt):
            migrated_norm[:, it] = np.interp(np.arange(n_cdp), migrated_idx, migrated_norm[migrated_idx, it])
    return migrated_norm, fold


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
