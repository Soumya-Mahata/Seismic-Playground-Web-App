"""utils/formatting.py — small, no-dependency display helpers."""
from __future__ import annotations

import numpy as np


def fmt_hz(v: float) -> str:
    return f"{v:.1f} Hz"


def fmt_ms(v_s: float) -> str:
    return f"{v_s * 1000:.1f} ms"


def fmt_m(v: float) -> str:
    return f"{v:,.0f} m"


def fmt_ms_per_s(v: float) -> str:
    return f"{v:,.0f} m/s"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def safe_minmax(arr: np.ndarray):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())
