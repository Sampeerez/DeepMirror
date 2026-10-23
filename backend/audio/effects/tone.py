"""Tone shaping: RBJ shelving EQ."""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sps

from backend.audio.effects.params import clip_float as _clipf

logger = logging.getLogger(__name__)


def _rbj_shelf(sample_rate: int, f0: float, gain_db: float, kind: str) -> np.ndarray:
    """RBJ audio-EQ-cookbook shelving biquad, returned as one SOS row."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * min(f0 / sample_rate, 0.45)
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)  # shelf slope S = 1
    two_sqrt_a_alpha = 2.0 * np.sqrt(A) * alpha

    if kind == "low":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_a_alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


class EqEffect:
    """Low shelf (250 Hz) + high shelf (4 kHz) tone control."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._gains = None
        self._sos = None
        self._zi = None

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        low = _clipf(params.get("low"), -15.0, 15.0, 0.0)
        high = _clipf(params.get("high"), -15.0, 15.0, 0.0)
        if (low, high) != self._gains:
            self._sos = np.vstack(
                [
                    _rbj_shelf(self.sr, 250.0, low, "low"),
                    _rbj_shelf(self.sr, 4000.0, high, "high"),
                ]
            )
            self._zi = np.zeros((self._sos.shape[0], 2))
            self._gains = (low, high)
        y, self._zi = sps.sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)
