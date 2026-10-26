"""Voice-character effects: robot, whisper, telephone, distortion, bitcrush."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import signal as sps

from backend.audio.effects.params import clip_float as _clipf
from backend.audio.effects.params import clip_int as _clipi

logger = logging.getLogger(__name__)


class RobotEffect:
    """Ring modulation against a sine carrier, classic metallic robot voice."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._phase = 0.0

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        freq = _clipf(params.get("freq"), 10.0, 400.0, 90.0)
        n = len(x)
        phases = self._phase + 2.0 * np.pi * freq * np.arange(n) / self.sr
        self._phase = float((self._phase + 2.0 * np.pi * freq * n / self.sr) % (2.0 * np.pi))
        return (x * np.sin(phases)).astype(np.float32)


class WhisperEffect:
    """Replaces the excitation with envelope-shaped noise for a breathy voice."""

    BLOCK = 128

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._env = 0.0
        self._rng = np.random.default_rng(0x5EED)

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        mix = _clipf(params.get("mix"), 0.0, 1.0, 1.0)
        n = len(x)
        n_blocks = math.ceil(n / self.BLOCK)
        padded = np.zeros(n_blocks * self.BLOCK, dtype=np.float32)
        padded[:n] = x
        rms = np.sqrt(np.mean(padded.reshape(n_blocks, self.BLOCK) ** 2, axis=1))

        env = np.empty(n_blocks, dtype=np.float32)
        e = self._env
        for i in range(n_blocks):
            e = 0.72 * e + 0.28 * rms[i]
            env[i] = e
        self._env = float(e)

        env_full = np.repeat(env, self.BLOCK)[:n]
        noise = self._rng.standard_normal(n).astype(np.float32) * env_full * 1.6
        return ((1.0 - mix) * x + mix * noise).astype(np.float32)


class TelephoneEffect:
    """Narrow 300–3400 Hz bandpass with light saturation, landline character."""

    def __init__(self, sample_rate: int) -> None:
        nyq = sample_rate / 2.0
        lo = min(300.0 / nyq, 0.4)
        hi = min(3400.0 / nyq, 0.95)
        self._sos: np.ndarray = np.asarray(sps.butter(4, [lo, hi], btype="band", output="sos"))
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        y, self._zi = sps.sosfilt(self._sos, x, zi=self._zi)
        return (np.tanh(y * 1.6) * 0.92).astype(np.float32)


class DistortionEffect:
    """Soft-clipping tanh drive."""

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        drive = _clipf(params.get("drive"), 0.0, 1.0, 0.4)
        gain = 1.0 + drive * 24.0
        y = np.tanh(x * gain)
        return (y * (0.9 / max(np.tanh(gain * 0.25), 1e-3)) * 0.35 + y * 0.45).astype(np.float32)


class BitcrushEffect:
    """Bit-depth reduction + sample-rate decimation (sample & hold)."""

    def __init__(self) -> None:
        self._hold = 0.0
        self._count = 0

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        bits = _clipi(params.get("bits"), 2, 16, 8)
        down = _clipi(params.get("down"), 1, 32, 4)
        n = len(x)
        y = x

        if down > 1:
            idx = np.arange(n) + self._count
            keep = (idx % down) == 0
            pos = np.where(keep, np.arange(n), -1)
            pos = np.maximum.accumulate(pos)
            y = np.where(pos >= 0, x[np.maximum(pos, 0)], np.float32(self._hold))
            self._count = int((self._count + n) % down)
            self._hold = float(y[-1]) if n > 0 else self._hold

        q = float(2 ** (bits - 1))
        return (np.round(y * q) / q).astype(np.float32)
