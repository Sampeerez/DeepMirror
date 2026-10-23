"""Modulation effects."""

from __future__ import annotations

import logging

import numpy as np

from backend.audio.effects.params import clip_float as _clipf

logger = logging.getLogger(__name__)


class ChorusEffect:
    """LFO-modulated fractional delay mixed with the dry signal."""

    MAX_DELAY_S = 0.040

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._phase = 0.0
        self._hist = np.zeros(int(self.MAX_DELAY_S * sample_rate) + 2, dtype=np.float32)

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        rate = _clipf(params.get("rate"), 0.05, 5.0, 0.8)
        depth = _clipf(params.get("depth"), 0.0, 1.0, 0.5)
        mix = _clipf(params.get("mix"), 0.0, 1.0, 0.5)
        n = len(x)

        base = 0.020 * self.sr
        span = 0.012 * self.sr * depth
        phases = self._phase + 2.0 * np.pi * rate * np.arange(n) / self.sr
        self._phase = float((self._phase + 2.0 * np.pi * rate * n / self.sr) % (2.0 * np.pi))
        delays = base + span * np.sin(phases)

        full = np.concatenate([self._hist, x.astype(np.float32)])
        positions = np.arange(len(full), dtype=np.float64)
        read_pos = len(self._hist) + np.arange(n, dtype=np.float64) - delays
        wet = np.interp(read_pos, positions, full)

        keep = int(self.MAX_DELAY_S * self.sr) + 2
        self._hist = full[-keep:]
        return ((1.0 - mix) * x + mix * wet).astype(np.float32)
