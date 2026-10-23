"""High-quality sample-rate conversion (libsoxr).

FFT resampling (``scipy.signal.resample``) treats every buffer as one period
of a periodic signal, which smears the end of each streaming window into its
start. soxr is a proper band-limited polyphase resampler and returns
delay-compensated output, so a resampled window stays aligned with its input.
"""

from __future__ import annotations

import numpy as np
import soxr


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a complete mono buffer; returns float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if from_rate == to_rate or audio.size == 0:
        return audio
    return soxr.resample(audio, from_rate, to_rate, quality="HQ").astype(np.float32, copy=False)

