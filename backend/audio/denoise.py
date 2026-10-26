"""Spectral-gating noise reduction, used when DeepFilterNet is not available.

The noise spectrum is learnt from non-speech frames only; bins are attenuated
by a Wiener-style gain with a floor, smoothed across bins and frames to avoid
musical noise.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

OVER_SUBTRACTION = 1.5
# While someone speaks the reduction is gentle: soft word endings and quiet
# syllables sit close to the noise floor, and cutting them sounds like a gate.
SPEECH_OVER_SUBTRACTION = 1.0
SPEECH_MAX_REDUCTION_DB = 6.0
LEARN_RATE = 0.15            # weight of the newest non-speech frames in the noise estimate
MIN_NOISE_FRAMES = 8         # about 0.2 s of non-speech before reducing anything


class SpectralDenoiser:
    def __init__(self, rate: int, reduction_db: float = 12.0) -> None:
        self.rate = int(rate)
        self.n_fft = 1 << max(8, round(np.log2(0.021 * self.rate)))   # ~21 ms: 1024 at 48 kHz
        self.hop = self.n_fft // 4
        self.window = np.asarray(signal.get_window("hann", self.n_fft))
        self.reduction_db = reduction_db
        self.reset()

    def reset(self) -> None:
        self.noise_power: np.ndarray | None = None
        self._noise_frames = 0

    @property
    def ready(self) -> bool:
        return self.noise_power is not None and self._noise_frames >= MIN_NOISE_FRAMES and self.reduction_db > 0

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        _, _, spectrum = signal.stft(audio, nperseg=self.n_fft, noverlap=self.n_fft - self.hop,
                                     window=self.window, boundary="even", padded=True)  # type: ignore[arg-type]
        return spectrum

    def learn(self, audio: np.ndarray, speech_gain: np.ndarray) -> None:
        """Update the noise estimate from the parts of ``audio`` where ``speech_gain`` is ~0."""
        if len(audio) < self.n_fft:
            return
        spectrum = self._stft(audio.astype(np.float64))
        power = np.abs(spectrum) ** 2
        # Speech gain at each frame centre; only clearly non-speech frames teach the noise.
        centres = np.clip(np.arange(power.shape[1]) * self.hop, 0, len(speech_gain) - 1)
        quiet = speech_gain[centres] < 0.02
        # Skip frames at the buffer edges, where the STFT padding distorts the spectrum.
        quiet[:2] = quiet[-2:] = False
        if not quiet.any():
            return
        estimate = power[:, quiet].mean(axis=1)
        if self.noise_power is None:
            self.noise_power = estimate
        else:
            weight = min(1.0, LEARN_RATE * quiet.sum() / 8)
            self.noise_power = (1 - weight) * self.noise_power + weight * estimate
        self._noise_frames += int(quiet.sum())

    def apply(self, audio: np.ndarray, speech_gain: np.ndarray | None = None) -> np.ndarray:
        """``audio`` with the learnt noise attenuated by up to ``reduction_db``.

        Where ``speech_gain`` marks speech, at most ``SPEECH_MAX_REDUCTION_DB``.
        """
        if not self.ready or len(audio) < self.n_fft:
            return audio
        assert self.noise_power is not None
        source = audio.astype(np.float64)
        spectrum = self._stft(source)
        power = np.abs(spectrum) ** 2
        frames = power.shape[1]
        speech = np.zeros(frames, bool)
        if speech_gain is not None and len(speech_gain):
            centres = np.clip(np.arange(frames) * self.hop, 0, len(speech_gain) - 1)
            speech = speech_gain[centres] > 0.5
        floor = np.where(speech, 10 ** (-min(self.reduction_db, SPEECH_MAX_REDUCTION_DB) / 20),
                         10 ** (-self.reduction_db / 20))
        over = np.where(speech, SPEECH_OVER_SUBTRACTION, OVER_SUBTRACTION)
        gain = 1.0 - over[None, :] * self.noise_power[:, None] / np.maximum(power, 1e-20)
        gain = np.clip(gain, floor[None, :], 1.0)
        gain = ndimage.uniform_filter(gain, size=3, mode="nearest")   # tame musical noise (3x3)
        _, cleaned = signal.istft(spectrum * gain, nperseg=self.n_fft, noverlap=self.n_fft - self.hop,
                                  window=self.window, boundary=True)  # type: ignore[arg-type]
        cleaned = cleaned[: len(source)]
        if len(cleaned) < len(source):
            cleaned = np.pad(cleaned, (0, len(source) - len(cleaned)))
        return cleaned.astype(np.float32)

    def reduction_now_db(self, audio: np.ndarray, cleaned: np.ndarray) -> float:
        """How much quieter the cleaned signal is (diagnostics)."""
        before = float(np.mean(np.square(audio, dtype=np.float64)))
        after = float(np.mean(np.square(cleaned, dtype=np.float64)))
        if before <= 1e-14 or after <= 1e-14:
            return 0.0
        return round(10 * np.log10(before / after), 1)
