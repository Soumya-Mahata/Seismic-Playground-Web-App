"""
modules/canonical.py — one spelling for parameters, and cheap array identity
================================================================================
Shared by the state graph (fingerprints) and the action log (recorded parameters).
No dependencies on the rest of the app, so both can import it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

_ARRAY_FULL_HASH_LIMIT = 1_000_000     # elements; above this a strided sample is hashed


# ─────────────────────────────────────────────────────────────────────────
# Canonical parameters and fingerprints
# ─────────────────────────────────────────────────────────────────────────

def array_digest(a: np.ndarray) -> str:
    """Cheap, stable identity of an array: shape, dtype and its bytes (or a
    strided sample of them when it is large)."""
    a = np.asarray(a)
    flat = np.ascontiguousarray(a).ravel()
    if flat.size > _ARRAY_FULL_HASH_LIMIT:
        flat = flat[:: max(1, flat.size // 65536)]
    h = hashlib.sha1()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(np.ascontiguousarray(flat).tobytes())
    return h.hexdigest()[:16]


def canonical(value: Any) -> Any:
    """Reduce parameters to plain JSON-able Python with a single spelling
    (sorted keys, tuples → lists, numpy scalars → Python scalars, arrays →
    digests). Raises TypeError for anything that cannot be reproduced from
    its recorded form — a parameter that is not canonical is not
    reproducible, so it must not be recorded silently."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(value, np.ndarray):
        return {"__ndarray__": array_digest(value), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    raise TypeError(f"Parameter of type {type(value).__name__} cannot be recorded reproducibly: {value!r}")


def jsonable(value: Any) -> Any:
    """Like `canonical`, but never raises: unknown objects and large arrays
    are summarised. For diagnostics (`info`) and reports, which are not
    part of a state's identity."""
    try:
        return canonical(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [jsonable(v) for v in value]
        return repr(value)[:120]
