"""Keep only the voice of a training recording.

Videos and streams come with music, effects and room noise under the voice, and
the voice model learns all of it. Each recording goes through:

1. Mel-Band RoFormer vocal separation (music and effects out), used only when
   there is a real background, so a clean recording is not touched;
2. DeepFilterNet3 for the noise that is left;
3. Silero VAD: everything that is not speech is silenced, so the slicing that
   follows (cuts at pauses) drops it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.audio.resampling import resample

logger = logging.getLogger(__name__)

# A background quieter than this under the voice is left alone (separation would only add artifacts).
CLEAN_BACKGROUND_DB = -35.0


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


@dataclass
class Isolation:
    audio: np.ndarray
    background_db: float | None     # background level under the voice before cleaning (None: not measured)
    separated: bool
    speech_share: float             # share of the recording kept as speech


class VoiceIsolator:
    """Holds the models while a job prepares its recordings."""

    def __init__(self, separator_path: Path | None, denoiser_path: Path | None, vad_path: Path | None,
                 device) -> None:
        self.separator = None
        if separator_path is not None and separator_path.is_file():
            from backend.training.roformer import load_separator

            self.separator = load_separator(str(separator_path), device)
        self.denoiser_path = denoiser_path if denoiser_path is not None and denoiser_path.is_file() else None
        self.vad_path = vad_path

    def close(self) -> None:
        self.separator = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:   # pragma: no cover
            pass

    def __call__(self, audio: np.ndarray, rate: int, progress: Callable[[float], None] | None = None) -> Isolation:
        background_db, separated = None, False
        voice = np.asarray(audio, np.float32)
        if self.separator is not None:
            from backend.training.roformer import SAMPLE_RATE, separate_vocals

            mix = resample(voice, rate, SAMPLE_RATE).astype(np.float32)
            vocals = separate_vocals(self.separator, mix, progress)
            background_db = _db(mix - vocals) - _db(vocals)
            if background_db > CLEAN_BACKGROUND_DB:
                voice, separated = resample(vocals, SAMPLE_RATE, rate).astype(np.float32), True
        if self.denoiser_path is not None:
            from backend.audio.neural_denoise import denoise_offline

            voice = resample(denoise_offline(self.denoiser_path, resample(voice, rate, 48000).astype(np.float32)),
                             48000, rate).astype(np.float32)[: len(voice)]
        gain = self._speech_gain(voice, rate)
        speech_share = float(np.mean(gain > 0.5)) if len(gain) else 0.0
        return Isolation((voice * gain).astype(np.float32), background_db, separated, speech_share)

    def _speech_gain(self, voice: np.ndarray, rate: int) -> np.ndarray:
        from backend.audio.voice_activity import RATE, SpeechTimeline, load_vad

        if self.vad_path is None or not self.vad_path.is_file():
            return np.ones(len(voice), np.float32)
        detector = load_vad(self.vad_path, lambda: -60.0)
        timeline = SpeechTimeline(detector, history_seconds=len(voice) / rate + 10)
        audio16 = resample(voice, rate, RATE).astype(np.float32)
        for start in range(0, len(audio16), RATE * 4):
            timeline.feed(audio16[start:start + RATE * 4])
        return timeline.gain(0, len(voice), rate)
