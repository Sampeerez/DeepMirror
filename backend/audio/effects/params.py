"""Tolerant parameter parsing for effect settings sent by the client."""

from __future__ import annotations

import numpy as np


def clip_float(value: object, lo: float, hi: float, default: float) -> float:
    """``value`` as a float clamped to ``[lo, hi]``, or ``default`` when unusable."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(number) else float(np.clip(number, lo, hi))


def clip_int(value: object, lo: int, hi: int, default: int) -> int:
    try:
        return int(np.clip(int(value), lo, hi))  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return default
