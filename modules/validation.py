"""
modules/validation.py — Input Validation & Friendly Diagnostics
===================================================================
No Streamlit dependency. Every function returns (ok: bool, message: str)
or raises a `ValidationError` with a message written for an end user
(not a stack trace) — callers in the `views/` layer decide how to show
it (st.error, st.warning, etc).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class ValidationError(Exception):
    """Raised for problems that should stop a processing step with a
    clear, friendly message rather than an unhandled traceback."""


ALLOWED_SEGY_EXTENSIONS = {".sgy", ".segy", ".su"}


def validate_upload_extension(filename: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_SEGY_EXTENSIONS:
        raise ValidationError(
            f"'{filename}' doesn't look like a SEG-Y file (expected one of "
            f"{sorted(ALLOWED_SEGY_EXTENSIONS)}). If this is a renamed SEG-Y "
            f"file it should still open — but header parsing may fail if the "
            f"file isn't actually SEG-Y."
        )


def validate_survey(num_traces: int, num_samples: int, sample_interval: float) -> None:
    if num_traces <= 0:
        raise ValidationError("The file contains zero traces — is this a valid SEG-Y file?")
    if num_samples <= 1:
        raise ValidationError(f"Only {num_samples} sample(s) per trace were found — "
                               f"check the binary header's sample-interval field.")
    if not (1e-5 <= sample_interval <= 0.05):
        raise ValidationError(
            f"Sample interval read as {sample_interval * 1000:.3f} ms, which is outside a "
            f"plausible range (0.01-50 ms). The binary header may be non-standard; try "
            f"re-exporting the SEG-Y with an explicit sample interval."
        )


def validate_cdp_gathers(gathers: dict, n_min_traces: int) -> None:
    if len(gathers) == 0:
        raise ValidationError(
            f"No CDP gathers reached the minimum fold ({n_min_traces} traces). "
            f"Either the CDP header field is empty/constant (common in files exported "
            f"without geometry applied) or try lowering the minimum-fold threshold."
        )


def validate_parameter_range(value: float, name: str, lo: float, hi: float, unit: str = "") -> None:
    if not (lo <= value <= hi):
        raise ValidationError(f"{name} = {value}{unit} is outside the valid range [{lo}, {hi}]{unit}.")


def validate_frequency_band(fmin: float, fmax: float, nyquist: float) -> None:
    if fmin >= fmax:
        raise ValidationError(f"Low-cut frequency ({fmin} Hz) must be less than high-cut ({fmax} Hz).")
    if fmax >= nyquist:
        raise ValidationError(
            f"High-cut frequency ({fmax} Hz) must be below the Nyquist frequency "
            f"({nyquist:.1f} Hz implied by this data's sample interval)."
        )


def check_finite(data: np.ndarray, stage_name: str) -> Optional[str]:
    """Return a warning string (not raised — this is informational) if a
    processing stage produced any non-finite values."""
    n_bad = int(np.sum(~np.isfinite(data)))
    if n_bad > 0:
        return (f"{stage_name} produced {n_bad} non-finite sample(s) "
                f"({100 * n_bad / data.size:.3f}% of the section) — these were "
                f"replaced with 0 before continuing.")
    return None


def sanitize(data: np.ndarray) -> np.ndarray:
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
