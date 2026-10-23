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
TRACK_SECONDS = 8.0               # how much recent voiced speech the median covers
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


class AutoPitch:
    """Tracks the speaker's register and decides the shift towards ``target``."""

    def __init__(self, target: F0Stats, initial_shift: float | None = None) -> None:
        self.target = target
        self._log_f0: deque[float] = deque(maxlen=int(TRACK_SECONDS * FRAMES_PER_SECOND))
        self.shift: float | None = initial_shift

    @property
    def speaker_median_hz(self) -> float | None:
        if len(self._log_f0) < MIN_VOICED_SECONDS * FRAMES_PER_SECOND:
            return None
        return float(math.exp(float(np.median(np.fromiter(self._log_f0, dtype=np.float64)))))

    def _ideal_shift(self) -> float | None:
        median = self.speaker_median_hz
        return None if median is None else semitones_between(median, self.target.median_hz)

    def observe(self, f0_hz: np.ndarray) -> None:
        """Feed newly heard F0 frames (Hz, 0 = unvoiced) at 100 frames/s."""
        f0 = np.asarray(f0_hz, dtype=np.float64)
        voiced = f0[f0 > 0]
        self._log_f0.extend(np.log(voiced).tolist())
        if len(voiced) == 0:
            self.on_pause()
        elif self.shift is None:
            ideal = self._ideal_shift()
            if ideal is not None:  # first estimate as soon as there is enough speech
                self.shift = _quantise(ideal)

    def on_pause(self) -> None:
        """Between words: re-centre the shift if the speaker's register clearly moved."""
        ideal = self._ideal_shift()
        if ideal is None:
            return
        if self.shift is None or abs(ideal - self.shift) >= HYSTERESIS:
            self.shift = _quantise(ideal)
