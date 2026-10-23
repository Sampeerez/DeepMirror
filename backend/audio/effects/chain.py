"""Per-stream effect chain: noise gate before the model, effect rack after it.

Every effect keeps its own state (delay lines, filter memories, oscillator
phases) so a continuous stream can be processed chunk by chunk without
boundary artifacts. A chain belongs to one stream and is not thread-safe.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.audio.effects.character import (
    BitcrushEffect,
    DistortionEffect,
    RobotEffect,
    TelephoneEffect,
    WhisperEffect,
)
from backend.audio.effects.dynamics import CompressorEffect, GainEffect, GateEffect, LimiterEffect
from backend.audio.effects.modulation import ChorusEffect
from backend.audio.effects.pitch import FormantShifter, PitchShifter
from backend.audio.effects.space import EchoEffect, ReverbEffect
from backend.audio.effects.tone import EqEffect

logger = logging.getLogger(__name__)


POST_CHAIN_ORDER = (
    "robot",
    "whisper",
    "distortion",
    "bitcrush",
    "telephone",
    "chorus",
    "echo",
    "reverb",
    "eq",
    "compressor",
    "gain",
    # Last, so no later stage can push peaks back over the ceiling.
    "limiter",
)

KNOWN_EFFECTS = (*POST_CHAIN_ORDER, "gate")


class EffectsChain:
    """A stateful per-stream effect chain.

    Layout: ``pre_process`` (noise gate) runs on the raw mic signal before any
    model, ``apply_pitch`` provides model-free pitch shifting, and
    ``post_process`` runs the formant shifter plus the ordered effect rack on
    the (converted) signal.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sr = int(sample_rate)
        self._pitch = PitchShifter(self.sr)
        self._formant = FormantShifter(self.sr)
        self._effects = {
            "robot": RobotEffect(self.sr),
            "whisper": WhisperEffect(self.sr),
            "distortion": DistortionEffect(),
            "bitcrush": BitcrushEffect(),
            "telephone": TelephoneEffect(self.sr),
            "chorus": ChorusEffect(self.sr),
            "echo": EchoEffect(self.sr),
            "reverb": ReverbEffect(self.sr),
            "eq": EqEffect(self.sr),
            "compressor": CompressorEffect(self.sr),
            "gain": GainEffect(),
            "limiter": LimiterEffect(self.sr),
            "gate": GateEffect(self.sr),
        }

    @staticmethod
    def _enabled(effects: dict, key: str) -> dict | None:
        entry = effects.get(key) if isinstance(effects, dict) else None
        if isinstance(entry, dict) and entry.get("enabled"):
            return entry
        return None

    @staticmethod
    def count_active(effects: dict, formant_shift: float = 0.0) -> int:
        count = 1 if abs(formant_shift) > 1e-3 else 0
        if not isinstance(effects, dict):
            return count
        for key in KNOWN_EFFECTS:
            entry = effects.get(key)
            if isinstance(entry, dict) and entry.get("enabled"):
                count += 1
        return count

    def pre_process(self, audio: np.ndarray, effects: dict) -> np.ndarray:
        """Input-side conditioning applied before model inference."""
        params = self._enabled(effects, "gate")
        if params is None or len(audio) == 0:
            return audio
        try:
            return self._effects["gate"].process(audio, params)
        except Exception:
            logger.exception("Noise gate failed — passing signal through")
            return audio

    def apply_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Model-free pitch shift used in DSP passthrough mode."""
        try:
            return self._pitch.process(audio, semitones)
        except Exception:
            logger.exception("Pitch shifter failed — passing signal through")
            return audio

    def post_process(
        self, audio: np.ndarray, effects: dict, formant_shift: float = 0.0
    ) -> np.ndarray:
        """Output-side chain: formant shift then the ordered effect rack."""
        if len(audio) == 0:
            return audio
        y = audio

        if abs(formant_shift) > 1e-3:
            try:
                y = self._formant.process(y, formant_shift)
            except Exception:
                logger.exception("Formant shifter failed — skipping")

        if isinstance(effects, dict):
            for key in POST_CHAIN_ORDER:
                params = self._enabled(effects, key)
                if params is None:
                    continue
                try:
                    y = self._effects[key].process(y, params)
                except Exception:
                    logger.exception("Effect %r failed — skipping", key)

        return np.clip(y, -1.0, 1.0).astype(np.float32)
