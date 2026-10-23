"""Realtime voice conversion over a rolling window, joined with SOLA.

Every live chunk re-renders a window of recent audio and emits one chunk of
output from it::

    |<-------------- history (16 kHz) ------------------------------>|
    |  lookback (past context)  | emitted + crossfade + search | lookahead |
                                                                 ^ newest input

* **Lookback** gives HuBERT, the pitch tracker and the synthesizer's
  attention real left context, which stabilises timbre and intonation.
* **Lookahead** means every emitted sample was rendered with real audio to
  its right, too. Model edges (reflect padding, attention and convolution
  boundary effects, F0 tracker edge frames) fall inside the lookahead and are
  discarded. It costs exactly ``lookahead`` of extra latency.
* **SOLA** (synchronised overlap-add): the tail held back from the previous
  render is matched against the new render within a small search range and
  the two are crossfaded at the best-correlated offset, so consecutive
  windows join without clicks or comb-filter "phasing".

The window is rendered by an injected callable, which keeps this module
independent from the model code and easy to test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from backend.audio.resampling import resample

MODEL_RATE = 16000
# Fraction of the best correlation traded for staying near the centre offset.
_CENTRE_BIAS = 0.05
FRAME = 160  # model frame hop at 16 kHz

# Renders a 16 kHz window; returns (audio, sample_rate) aligned to the window start.
WindowRenderer = Callable[[np.ndarray], tuple[np.ndarray, int]]


@dataclass(frozen=True, slots=True)
class StreamTiming:
    lookback_seconds: float = 1.0
    lookahead_seconds: float = 0.15
    crossfade_seconds: float = 0.05
    search_seconds: float = 0.012

    def algorithmic_delay(self) -> float:
        """Latency this stage adds on top of chunk buffering and compute time."""
        return self.lookahead_seconds + self.crossfade_seconds + self.search_seconds / 2


def _sola_offset(reference: np.ndarray, candidate: np.ndarray, search: int) -> int:
    """Offset in ``[0, search]`` where ``candidate`` best matches ``reference``.

    Normalised cross-correlation with a mild preference for the centre of the
    range, so the alignment cannot creep to an edge over many chunks and lose
    its ability to correct in that direction. Silence has no phase to match
    and keeps the nominal (centre) alignment.
    """
    centre = search // 2
    length = len(reference)
    reference = reference.astype(np.float64)
    if search <= 0 or length == 0 or float(reference @ reference) < 1e-10 * length:
        return centre
    region = candidate[: length + search].astype(np.float64)
    numerator = np.correlate(region, reference, mode="valid")[: search + 1]
    energy = np.convolve(region * region, np.ones(length), mode="valid")[: search + 1]
    scores = numerator / np.sqrt(energy + 1e-12)
    distance = np.abs(np.arange(len(scores)) - centre) / max(centre, 1)
    scores = scores - _CENTRE_BIAS * np.abs(scores).max() * distance
    return int(np.argmax(scores))


class RealtimeVoiceStream:
    """Converts fixed-size live chunks into output chunks of the same size."""

    def __init__(self, client_rate: int, render: WindowRenderer, timing: StreamTiming | None = None) -> None:
        self.client_rate = int(client_rate)
        self.render = render
        self.timing = timing or StreamTiming()
        self._crossfade = max(1, round(self.timing.crossfade_seconds * self.client_rate))
        self._search = max(0, round(self.timing.search_seconds * self.client_rate))
        self._fade = max(1, round(0.01 * self.client_rate))
        self.reset()

    def reset(self) -> None:
        """Forget all context (new model, stream restart, wake from silence)."""
        # Kept at the client rate and resampled whole on every chunk: a
        # streaming resampler emits in bursts, which would make the window's
        # end (and therefore the output timing) jitter by tens of ms.
        self._history = np.zeros(0, dtype=np.float32)
        self._tail: np.ndarray | None = None
        self._resuming = False

    def _history_length(self, chunk_length: int) -> int:
        t = self.timing
        seconds = (t.lookback_seconds + t.lookahead_seconds + t.crossfade_seconds + t.search_seconds
                   + chunk_length / self.client_rate)
        return math.ceil(seconds * self.client_rate)

    def _push(self, chunk: np.ndarray) -> int:
        length = self._history_length(len(chunk))
        if len(self._history) < length:
            self._history = np.concatenate([np.zeros(length - len(self._history), np.float32), self._history])
        self._history = np.concatenate([self._history, np.asarray(chunk, dtype=np.float32)])[-length:]
        return length

    def idle(self, chunk: np.ndarray) -> np.ndarray:
        """Keep listening without rendering (silence saver asleep); return silence.

        The history keeps advancing, so the first render after waking has the
        real audio before the onset as context and the attack is not lost.
        """
        if len(chunk):
            self._push(chunk)
        self._tail = None
        self._resuming = True
        return np.zeros(len(chunk), dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed ``len(chunk)`` new samples; return exactly as many converted samples."""
        n = len(chunk)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        length = self._push(chunk)

        model_input = resample(self._history, self.client_rate, MODEL_RATE)
        dropped = len(model_input) % FRAME  # whole model frames; trim the oldest samples
        rendered, rate = self.render(model_input[dropped:])
        rendered = resample(rendered, rate, self.client_rate)

        crossfade = min(self._crossfade, n)
        span = n + crossfade + self._search
        # Rendered sample 0 is history time ``dropped / MODEL_RATE``.
        end_seconds = length / self.client_rate - self.timing.lookahead_seconds - dropped / MODEL_RATE
        end = round(end_seconds * self.client_rate)
        if len(rendered) < end:
            rendered = np.pad(rendered, (0, end - len(rendered)))
        region = rendered[max(0, end - span):end]
        if len(region) < span:
            region = np.pad(region, (span - len(region), 0))

        if self._tail is None or len(self._tail) != crossfade:
            offset = self._search // 2
            output = region[offset:offset + n].copy()
            if self._resuming:  # fade in from the silence emitted while asleep
                ramp = min(self._fade, n)
                output[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        else:
            offset = _sola_offset(self._tail, region, self._search)
            output = region[offset:offset + n].copy()
            phase = np.linspace(0.0, np.pi / 2, crossfade, dtype=np.float32)
            fade_in = np.sin(phase) ** 2
            output[:crossfade] = self._tail * (1.0 - fade_in) + output[:crossfade] * fade_in
        self._tail = region[offset + n:offset + n + crossfade].copy()
        self._resuming = False
        return output.astype(np.float32, copy=False)
