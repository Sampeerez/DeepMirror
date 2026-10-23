"""Output level: the converted voice as loud as the speaker, without clipping.

RVC voices come out at their own level: some models are a few dB louder or
quieter than others, and ``rms_mix_rate`` only partly follows the input. The
matcher measures the speaker's voice and the converted voice over the same
speech samples and slowly moves a gain so the two match. It only learns while
someone speaks (silence says nothing about level), moves with a time constant
of about half a second (no pumping inside words), and is limited to +-18 dB.
"""

from __future__ import annotations

import math

import numpy as np

TIME_CONSTANT_SECONDS = 0.5
MAX_GAIN_DB = 18.0
MIN_SPEECH_SECONDS = 0.05
CEILING = 10 ** (-0.5 / 20)   # -0.5 dBFS
KNEE = 10 ** (-3 / 20)        # transparent below -3 dBFS


class LoudnessMatcher:
    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        self.reset()

    def reset(self) -> None:
        self.gain_db = 0.0
        self._learnt = False

    def update(self, reference: np.ndarray, converted: np.ndarray, speech_gain: np.ndarray) -> float:
        """Learn from one chunk; returns the gain (linear) to apply to ``converted``.

        Only 10 ms frames where the speaker's voice really has energy count
        (inside speech, within 30 dB of the chunk's loudest frame, above the
        noise): the edges of a speech segment are mostly silence on one side
        and model output on the other, and would mislead the estimate.
        """
        frame = max(1, self.rate // 100)
        count = min(len(reference), len(converted), len(speech_gain)) // frame
        if count:
            shape = (count, frame)
            ref = np.mean(np.square(reference[: count * frame].reshape(shape), dtype=np.float64), axis=1)
            out = np.mean(np.square(converted[: count * frame].reshape(shape), dtype=np.float64), axis=1)
            inside = speech_gain[: count * frame].reshape(shape).mean(axis=1) > 0.5
            ref_db = 10 * np.log10(np.maximum(ref, 1e-14))
            valid = inside & (ref_db > -75.0)
            if valid.any():
                valid &= ref_db > ref_db[valid].max() - 30.0
            if valid.sum() * frame >= MIN_SPEECH_SECONDS * self.rate and out[valid].sum() > 1e-14:
                target = float(np.clip(10 * math.log10(ref[valid].sum() / out[valid].sum()),
                                       -MAX_GAIN_DB, MAX_GAIN_DB))
                if not self._learnt:
                    self.gain_db, self._learnt = target, True   # first real speech: go straight there
                else:
                    weight = 1 - math.exp(-len(converted) / (TIME_CONSTANT_SECONDS * self.rate))
                    self.gain_db += weight * (target - self.gain_db)
        return 10 ** (self.gain_db / 20)


def soft_limit(audio: np.ndarray) -> np.ndarray:
    """Transparent below -3 dBFS, then a smooth knee that never exceeds -0.5 dBFS."""
    magnitude = np.abs(audio)
    over = magnitude > KNEE
    if not over.any():
        return audio
    out = audio.copy()
    span = CEILING - KNEE
    excess = (magnitude[over] - KNEE) / span
    out[over] = np.sign(audio[over]) * (KNEE + span * np.tanh(excess))
    return out.astype(np.float32, copy=False)
