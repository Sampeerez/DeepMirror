"""Time-based effects: feedback combs, all-passes, echo and reverb."""

from __future__ import annotations

import logging

import numpy as np

from backend.audio.effects.params import clip_float as _clipf

logger = logging.getLogger(__name__)


class _FeedbackComb:
    """Feedback comb filter: y[n] = x[n - D] + fb * y[n - D].

    Processed block-wise in slices no longer than the delay so the recursion
    only ever references samples already stored in the history buffers.
    """

    def __init__(self, delay_samples: int) -> None:
        self.delay = max(1, int(delay_samples))
        self._x_hist = np.zeros(self.delay, dtype=np.float32)
        self._y_hist = np.zeros(self.delay, dtype=np.float32)

    def process(self, x: np.ndarray, feedback: float) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float32)
        pos = 0
        while pos < len(x):
            blk = min(self.delay, len(x) - pos)
            xb = x[pos : pos + blk]
            yb = self._x_hist[:blk] + feedback * self._y_hist[:blk]
            out[pos : pos + blk] = yb
            self._x_hist = np.concatenate([self._x_hist[blk:], xb]).astype(np.float32)
            self._y_hist = np.concatenate([self._y_hist[blk:], yb]).astype(np.float32)
            pos += blk
        return out


class _Allpass:
    """Schroeder allpass: y[n] = -g*x[n] + x[n - D] + g*y[n - D]."""

    def __init__(self, delay_samples: int, g: float = 0.7) -> None:
        self.delay = max(1, int(delay_samples))
        self.g = g
        self._x_hist = np.zeros(self.delay, dtype=np.float32)
        self._y_hist = np.zeros(self.delay, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float32)
        pos = 0
        while pos < len(x):
            blk = min(self.delay, len(x) - pos)
            xb = x[pos : pos + blk]
            yb = -self.g * xb + self._x_hist[:blk] + self.g * self._y_hist[:blk]
            out[pos : pos + blk] = yb
            self._x_hist = np.concatenate([self._x_hist[blk:], xb]).astype(np.float32)
            self._y_hist = np.concatenate([self._y_hist[blk:], yb]).astype(np.float32)
            pos += blk
        return out


class EchoEffect:
    """Feedback delay line."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._time = None
        self._comb: _FeedbackComb | None = None

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        time_s = _clipf(params.get("time"), 0.03, 1.5, 0.25)
        feedback = _clipf(params.get("feedback"), 0.0, 0.9, 0.35)
        mix = _clipf(params.get("mix"), 0.0, 1.0, 0.35)
        if self._comb is None or self._time != time_s:
            self._comb = _FeedbackComb(int(time_s * self.sr))
            self._time = time_s
        wet = self._comb.process(x, feedback)
        return ((1.0 - mix) * x + mix * wet).astype(np.float32)


class ReverbEffect:
    """Schroeder reverb: four parallel feedback combs into two series allpasses."""

    COMB_DELAYS_S = (0.0297, 0.0371, 0.0411, 0.0437)

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._size = None
        self._combs: list[_FeedbackComb] = []
        self._allpasses: list[_Allpass] = []

    def _build(self, size: float) -> None:
        scale = 0.6 + 1.6 * size
        self._combs = [_FeedbackComb(int(t * scale * self.sr)) for t in self.COMB_DELAYS_S]
        self._allpasses = [
            _Allpass(int(0.0050 * self.sr), 0.7),
            _Allpass(int(0.0017 * self.sr), 0.7),
        ]
        self._size = size

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        size = _clipf(params.get("size"), 0.0, 1.0, 0.5)
        mix = _clipf(params.get("mix"), 0.0, 1.0, 0.35)
        if self._size != size or not self._combs:
            self._build(size)

        base_fb = min(0.70 + 0.22 * size, 0.9)
        wet = np.zeros(len(x), dtype=np.float32)
        for i, comb in enumerate(self._combs):
            wet += comb.process(x, min(base_fb + i * 0.012, 0.92))
        wet /= len(self._combs)
        for allpass in self._allpasses:
            wet = allpass.process(wet)
        return ((1.0 - mix) * x + mix * wet).astype(np.float32)
