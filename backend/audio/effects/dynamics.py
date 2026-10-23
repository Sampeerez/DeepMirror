"""Dynamics: compressor, noise gate, output gain and peak limiter."""

from __future__ import annotations

import logging
import math

import numpy as np

from backend.audio.effects.params import clip_float as _clipf

logger = logging.getLogger(__name__)


class CompressorEffect:
    """Block-based RMS compressor with attack/release smoothing and auto make-up."""

    BLOCK = 64

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._env_db = -80.0
        self._attack = math.exp(-self.BLOCK / (sample_rate * 0.006))
        self._release = math.exp(-self.BLOCK / (sample_rate * 0.090))

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        threshold = _clipf(params.get("threshold"), -60.0, 0.0, -20.0)
        ratio = _clipf(params.get("ratio"), 1.0, 20.0, 3.0)
        n = len(x)
        n_blocks = math.ceil(n / self.BLOCK)
        padded = np.zeros(n_blocks * self.BLOCK, dtype=np.float32)
        padded[:n] = x
        rms_db = 20.0 * np.log10(
            np.sqrt(np.mean(padded.reshape(n_blocks, self.BLOCK) ** 2, axis=1)) + 1e-9
        )

        gains_db = np.empty(n_blocks, dtype=np.float32)
        env = self._env_db
        for i in range(n_blocks):
            coeff = self._attack if rms_db[i] > env else self._release
            env = coeff * env + (1.0 - coeff) * rms_db[i]
            over = env - threshold
            gains_db[i] = -over * (1.0 - 1.0 / ratio) if over > 0 else 0.0
        self._env_db = float(env)

        makeup_db = -threshold * (1.0 - 1.0 / ratio) * 0.4
        gains = 10.0 ** ((gains_db + makeup_db) / 20.0)
        return (x * np.repeat(gains, self.BLOCK)[:n]).astype(np.float32)


class GateEffect:
    """Noise gate with hysteresis and smoothed gain transitions."""

    BLOCK = 64

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._gain = 1.0
        self._open = True

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        threshold = _clipf(params.get("threshold"), -80.0, -10.0, -45.0)
        n = len(x)
        n_blocks = math.ceil(n / self.BLOCK)
        padded = np.zeros(n_blocks * self.BLOCK, dtype=np.float32)
        padded[:n] = x
        rms_db = 20.0 * np.log10(
            np.sqrt(np.mean(padded.reshape(n_blocks, self.BLOCK) ** 2, axis=1)) + 1e-9
        )

        gains = np.empty(n_blocks, dtype=np.float32)
        gain = self._gain
        is_open = self._open
        for i in range(n_blocks):
            if is_open and rms_db[i] < threshold - 6.0:
                is_open = False
            elif not is_open and rms_db[i] > threshold:
                is_open = True
            target = 1.0 if is_open else 0.03
            gain = 0.82 * gain + 0.18 * target
            gains[i] = gain
        self._gain = float(gain)
        self._open = is_open

        return (x * np.repeat(gains, self.BLOCK)[:n]).astype(np.float32)


class GainEffect:
    """Simple output trim."""

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        db = _clipf(params.get("db"), -24.0, 24.0, 0.0)
        return (x * (10.0 ** (db / 20.0))).astype(np.float32)


class LimiterEffect:
    """Block-based peak limiter: instant attack, smooth release.

    Each 64-sample block gets a gain no larger than ``threshold / peak`` of
    that block, so the output never exceeds the threshold. Gain recovers with
    an exponential release between blocks, which keeps the gain changes far
    below audible zipper rates. State carries across chunks.
    """

    BLOCK = 64

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._gain = 1.0
        # ~60 ms release time constant, expressed per block.
        self._release = math.exp(-self.BLOCK / (sample_rate * 0.060))

    def process(self, x: np.ndarray, params: dict) -> np.ndarray:
        threshold_db = _clipf(params.get("threshold"), -12.0, 0.0, -1.0)
        threshold = 10.0 ** (threshold_db / 20.0)
        n = len(x)
        n_blocks = math.ceil(n / self.BLOCK)
        padded = np.zeros(n_blocks * self.BLOCK, dtype=np.float32)
        padded[:n] = x
        peaks = np.max(np.abs(padded.reshape(n_blocks, self.BLOCK)), axis=1)
        ceilings = np.minimum(1.0, threshold / np.maximum(peaks, 1e-9))

        gains = np.empty(n_blocks, dtype=np.float32)
        gain = self._gain
        release = self._release
        for i in range(n_blocks):
            recovered = 1.0 - (1.0 - gain) * release
            gain = min(recovered, float(ceilings[i]))
            gains[i] = gain
        self._gain = float(gain)

        return (x * np.repeat(gains, self.BLOCK)[:n]).astype(np.float32)


# Post-chain processing order. The gate runs pre-model (see pre_process).
