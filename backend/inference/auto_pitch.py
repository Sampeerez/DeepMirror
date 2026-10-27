"""Automatic pitch shift: move the speaker's register onto the voice model's.

Voices trained in the studio store the F0 statistics of their dataset
(``f0_stats`` in the ``.pth``). The live stream tracks the speaker's median F0
over the last few seconds of voiced speech and shifts by the interval between
the two medians, so a lower voice driving a higher model (or the reverse)
lands in the register the model was trained on, while the speaker's own
intonation contour is kept.

The shift only moves at pauses (never inside a word) and only when it drifts
clearly, so it does not wobble with the natural pitch movement of speech.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

FRAMES_PER_SECOND = 100
MIN_VOICED_SECONDS = 1.0          # speech needed before the first estimate
STEP = 0.5                        # applied shifts are multiples of half a semitone
HYSTERESIS = 1.0                  # semitones of drift needed to move an applied shift
MAX_SHIFT = 24.0


@dataclass(frozen=True, slots=True)
class F0Stats:
    """Voiced F0 distribution of a voice, in Hz."""

    median_hz: float
    p10_hz: float
    p90_hz: float

    def to_dict(self) -> dict[str, float]:
        return {key: round(value, 2) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: object) -> F0Stats | None:
        if not isinstance(value, dict):
            return None
        try:
            stats = cls(float(value["median_hz"]), float(value["p10_hz"]), float(value["p90_hz"]))
        except (KeyError, TypeError, ValueError):
            return None
        return stats if 40.0 <= stats.median_hz <= 1200.0 else None


def compute_f0_stats(tracks: Iterable[np.ndarray]) -> F0Stats | None:
    """Statistics of the voiced frames (F0 > 0) of one or more pitch tracks."""
    voiced = [np.asarray(track, dtype=np.float64) for track in tracks]
    values = np.concatenate([track[track > 0] for track in voiced]) if voiced else np.zeros(0)
    if len(values) < FRAMES_PER_SECOND // 2:
        return None
    # Percentiles in log-frequency, where pitch is perceived linearly.
    p10, median, p90 = np.exp(np.percentile(np.log(values), [10, 50, 90]))
    return F0Stats(float(median), float(p10), float(p90))


def semitones_between(source_hz: float, target_hz: float) -> float:
    return 12.0 * math.log2(target_hz / source_hz)


def _quantise(semitones: float) -> float:
    return float(np.clip(round(semitones / STEP) * STEP, -MAX_SHIFT, MAX_SHIFT))


# --- The speaker -------------------------------------------------------------
PROFILE_SECONDS = 30.0            # voiced speech the profile keeps (a moving picture of the speaker)
CALIBRATED_SECONDS = 10.0         # voiced speech after which the profile counts as calibrated
DEFAULT_EXPRESSIVENESS = 1.25
EXPRESSIVENESS_RANGE = (0.5, 2.0)
AUTO_EXPRESSIVENESS_RANGE = (0.8, 1.8)


class SpeakerProfile:
    """The speaker's pitch, learnt from the first seconds of speech and kept current.

    It can start from known statistics (``seed``), so a new session sounds
    right from the first word; the speaker's own speech takes over as soon as
    there is a second of it, unless ``fixed`` (a recorded voice profile, which
    a shout or a laugh must not move).
    """

    def __init__(self, seed: F0Stats | None = None, fixed: bool = False) -> None:
        self.seed = seed
        self.fixed = fixed and seed is not None
        self._log_f0: deque[float] = deque(maxlen=int(PROFILE_SECONDS * FRAMES_PER_SECOND))
        self.voiced_frames = 0

    def observe(self, f0_hz: np.ndarray) -> bool:
        """Add newly heard F0 frames (Hz, 0 = unvoiced); True if they contain no voice (a pause)."""
        f0 = np.asarray(f0_hz, dtype=np.float64)
        voiced = f0[f0 > 0]
        self._log_f0.extend(np.log(voiced).tolist())
        self.voiced_frames += len(voiced)
        return len(voiced) == 0

    @property
    def stats(self) -> F0Stats | None:
        if self.fixed or len(self._log_f0) < MIN_VOICED_SECONDS * FRAMES_PER_SECOND:
            return self.seed
        values = np.exp(np.fromiter(self._log_f0, dtype=np.float64))
        return compute_f0_stats([values])

    @property
    def calibration(self) -> float:
        """0..1: how much of the calibration speech has been heard (1 with a saved profile)."""
        if self.seed is not None and (self.fixed or self.voiced_frames == 0):
            return 1.0
        return min(1.0, self.voiced_frames / (CALIBRATED_SECONDS * FRAMES_PER_SECOND))

    def describe(self) -> dict | None:
        stats = self.stats
        if stats is None:
            return None
        return {**stats.to_dict(), "calibration": round(self.calibration, 2), "from_profile": self.fixed,
                "range_semitones": round(semitones_between(stats.p10_hz, stats.p90_hz) / 2, 1)}


def auto_expressiveness(target: F0Stats, speaker: F0Stats) -> float:
    """Pitch-movement scale that gives the speaker's melody the target voice's range."""
    speaker_range = math.log(speaker.p90_hz / speaker.p10_hz)
    if speaker_range < 1e-3:
        return 1.0
    ratio = math.log(target.p90_hz / target.p10_hz) / speaker_range
    return float(np.clip(ratio, *AUTO_EXPRESSIVENESS_RANGE))


class AutoPitch:
    """Decides the shift that moves the speaker's register onto ``target``'s."""

    def __init__(self, target: F0Stats, profile: SpeakerProfile, initial_shift: float | None = None) -> None:
        self.target = target
        self.profile = profile
        self.shift: float | None = initial_shift

    @property
    def speaker_median_hz(self) -> float | None:
        stats = self.profile.stats
        return stats.median_hz if stats is not None else None

    def _ideal_shift(self) -> float | None:
        median = self.speaker_median_hz
        return None if median is None else semitones_between(median, self.target.median_hz)

    def update(self, pause: bool) -> None:
        """Call after the profile heard new frames; ``pause`` when they held no voice."""
        if pause:
            self.on_pause()
        elif self.shift is None:
            ideal = self._ideal_shift()
            if ideal is not None:  # first estimate as soon as there is enough speech
                self.shift = _quantise(ideal)

    def observe(self, f0_hz: np.ndarray) -> None:
        """Feed newly heard F0 frames to the profile and update (standalone use)."""
        self.update(self.profile.observe(f0_hz))

    def on_pause(self) -> None:
        """Between words: re-centre the shift if the speaker's register clearly moved."""
        ideal = self._ideal_shift()
        if ideal is None:
            return
        if self.shift is None or abs(ideal - self.shift) >= HYSTERESIS:
            self.shift = _quantise(ideal)
