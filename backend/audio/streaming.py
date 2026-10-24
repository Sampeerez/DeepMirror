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

Around the model, every chunk also goes through:

* **Voice detection** (Silero VAD): only speech is converted. Room noise,
  breaths and clicks never go through the voice model.
* **Noise reduction**: the model hears the voice with the room noise removed
  (spectral gating learnt from the non-speech parts), which keeps it
  articulate.
* **Ambience**: the room is mixed back *unconverted* and time-aligned with
  the voice: the real input during pauses and the removed noise under
  speech, so the result is the converted voice in your room, not a gate.
* **Loudness matching** to the speaker's own voice, then a soft limiter.

The window is rendered by an injected callable, which keeps this module
independent from the model code and easy to test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from backend.audio.denoise import SpectralDenoiser
from backend.audio.loudness import LoudnessMatcher, soft_limit
from backend.audio.resampling import resample
from backend.audio.voice_activity import LevelVad, SileroVad, SpeechTimeline

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


# The converted voice may not be louder than the speaker's own voice (at the
# matched level) by more than this, frame by frame. Where the input is almost
# silent (the end of a quiet word, the gap in "no... no") a model can invent
# vowels out of noise; this keeps them as quiet as what was actually said.
# While the detector hears speech the margin allows the model's own dynamics;
# in the pre-roll and hang-over around it, nothing louder than the input.
CEILING_MARGIN_DB = 8.0
EDGE_MARGIN_DB = 0.0
CEILING_FRAME_SECONDS = 0.02


def envelope_ceiling(converted: np.ndarray, reference: np.ndarray, rate: int,
                     heard: np.ndarray | None = None) -> np.ndarray:
    """``converted`` attenuated wherever its 20 ms envelope exceeds ``reference``'s by the margin."""
    frame = max(1, round(CEILING_FRAME_SECONDS * rate))
    count = min(len(converted), len(reference)) // frame
    if count < 3:
        return converted
    shape = (count, frame)
    have = np.sqrt(np.mean(np.square(converted[: count * frame].reshape(shape), dtype=np.float64), axis=1))
    want = np.sqrt(np.mean(np.square(reference[: count * frame].reshape(shape), dtype=np.float64), axis=1))
    # Neighbouring frames count too: the model's timing differs from the input by a few ms.
    padded = np.pad(want, 1, mode="edge")
    want = np.maximum(want, np.maximum(padded[:-2], padded[2:]))
    margin = np.full(count, CEILING_MARGIN_DB)
    if heard is not None and len(heard) >= count * frame:
        margin = np.where(heard[: count * frame].reshape(shape).any(axis=1), CEILING_MARGIN_DB, EDGE_MARGIN_DB)
    limit = want * 10 ** (margin / 20) + 1e-7
    gains = np.minimum(1.0, limit / np.maximum(have, 1e-12))
    if gains.min() >= 0.999:
        return converted
    centres = (np.arange(count) + 0.5) * frame
    curve = np.interp(np.arange(len(converted)), centres, gains).astype(np.float32)
    return (converted * curve).astype(np.float32)


# Below this distance from the room's noise floor the input holds no voice, so
# whatever the model renders there (the tail of a phrase, a breath next to
# speech) is invented noise and is faded out.
SNR_GATE_DB = 6.0
SNR_GATE_WIDTH_DB = 6.0


def above_room(dry: np.ndarray, rate: int, noise_db: float) -> np.ndarray:
    """0..1 per sample: how clearly the input stands out from the room noise (smooth, 20 ms frames)."""
    frame = max(1, round(CEILING_FRAME_SECONDS * rate))
    count = len(dry) // frame
    if count < 3:
        return np.ones(len(dry), np.float32)
    power = np.mean(np.square(dry[: count * frame].reshape(count, frame), dtype=np.float64), axis=1)
    level = 10 * np.log10(np.maximum(power, 1e-14))
    gains = np.clip((level - (noise_db + SNR_GATE_DB)) / SNR_GATE_WIDTH_DB, 0.0, 1.0)
    padded = np.pad(gains, 1, mode="edge")
    gains = np.maximum(gains, np.maximum(padded[:-2], padded[2:]))   # never clip an onset or ending frame
    centres = (np.arange(count) + 0.5) * frame
    return np.interp(np.arange(len(dry)), centres, gains).astype(np.float32)


def snr_gate(converted: np.ndarray, dry: np.ndarray, rate: int, noise_db: float | None) -> np.ndarray:
    """Fade ``converted`` where ``dry`` (the input) is within SNR_GATE_DB of the room noise."""
    if noise_db is None:
        return converted
    curve = above_room(dry[: len(converted)], rate, noise_db)
    if len(curve) < len(converted):
        curve = np.pad(curve, (0, len(converted) - len(curve)), mode="edge")
    return (converted * curve).astype(np.float32)


@dataclass(slots=True)
class StreamCleanup:
    """What happens around the model (all adjustable live)."""

    isolate_voice: bool = True      # convert speech only; the rest is ambience
    noise_reduction_db: float = 12.0
    ambient_db: float = -60.0       # steady room noise mixed back (never the raw input); <= -60 is off
    match_loudness: bool = True

    @property
    def ambient_gain(self) -> float:
        return 0.0 if self.ambient_db <= -60 else 10 ** (self.ambient_db / 20)


@dataclass(slots=True)
class StreamStats:
    speech_probability: float = 0.0
    noise_reduction_db: float = 0.0
    loudness_gain_db: float = 0.0
    neural_vad: bool = False


class RealtimeVoiceStream:
    """Converts fixed-size live chunks into output chunks of the same size."""

    def __init__(self, client_rate: int, render: WindowRenderer, timing: StreamTiming | None = None, *,
                 vad: SileroVad | LevelVad | None = None, cleanup: StreamCleanup | None = None) -> None:
        self.client_rate = int(client_rate)
        self.render = render
        self.timing = timing or StreamTiming()
        # Without a detector everything is "speech" and nothing is added around the model.
        self.cleanup = cleanup or StreamCleanup(isolate_voice=vad is not None, noise_reduction_db=0.0,
                                                ambient_db=-60.0, match_loudness=False)
        self.speech = SpeechTimeline(vad) if vad is not None else None
        self.denoiser = SpectralDenoiser(self.client_rate, self.cleanup.noise_reduction_db)
        self.loudness = LoudnessMatcher(self.client_rate)
        self.stats = StreamStats(neural_vad=bool(self.speech and self.speech.neural))
        self._crossfade = max(1, round(self.timing.crossfade_seconds * self.client_rate))
        self._search = max(0, round(self.timing.search_seconds * self.client_rate))
        self._fade = max(1, round(0.01 * self.client_rate))
        self.reset()

    def reset(self) -> None:
        """Forget all context (new model, stream restart)."""
        # Kept at the client rate and resampled whole on every chunk: a
        # streaming resampler emits in bursts, which would make the window's
        # end (and therefore the output timing) jitter by tens of ms.
        self._history = np.zeros(0, dtype=np.float32)
        self._total = 0                       # client samples heard so far
        self._fed16 = 0                       # of which fed to the voice detector (at 16 kHz)
        self._tail: np.ndarray | None = None
        self._ambient_tail: np.ndarray | None = None
        self._resuming = False
        self._chunk = 0
        self._speech_gain = np.zeros(0, np.float32)
        if self.speech is not None:
            self.speech.reset()
        self.denoiser.reset()
        self.loudness.reset()

    def set_cleanup(self, cleanup: StreamCleanup) -> None:
        self.cleanup = cleanup
        self.denoiser.reduction_db = cleanup.noise_reduction_db

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
        self._total += len(chunk)
        self._chunk = len(chunk)
        return length

    # --- Listening ----------------------------------------------------------

    def listen(self, chunk: np.ndarray) -> bool:
        """Add a chunk to the history and the voice detector; True if it contains speech."""
        self._push(chunk)
        if self.speech is None:
            self._speech_gain = np.ones(len(self._history), np.float32)
            return True
        fed_target = round(self._total * MODEL_RATE / self.client_rate)
        fresh = fed_target - self._fed16
        if fresh > 0:
            tail = self._history[-(math.ceil(fresh * self.client_rate / MODEL_RATE) + 64):]
            audio16 = resample(tail, self.client_rate, MODEL_RATE)[-fresh:]
            self.speech.feed(audio16)
            self._fed16 = fed_target
        start = self._total - len(self._history)
        self._speech_gain = self.speech.gain(start, len(self._history), self.client_rate)
        self.stats.speech_probability = round(self.speech.last_probability(), 2)
        if self.cleanup.noise_reduction_db > 0:
            # Learn the room from the newest audio that is certainly not speech (STFT context around it).
            recent = min(len(self._history), len(chunk) + 2 * self.denoiser.n_fft + round(0.35 * self.client_rate))
            noise_only = self.speech.noise_only(self._total - recent, recent, self.client_rate)
            self.denoiser.learn(self._history[-recent:], np.where(noise_only, 0.0, 1.0).astype(np.float32))
        return self.speech.speech_in_last(len(chunk) / self.client_rate)

    def _emitted_end(self) -> int:
        """History index where the emitted region ends (``lookahead`` before the newest sample)."""
        return len(self._history) - round(self.timing.lookahead_seconds * self.client_rate)

    def _ambience(self, dry: np.ndarray, clean: np.ndarray | None) -> np.ndarray:
        """The steady room noise only (what the reducer removed), never the raw input.

        Breaths, clicks and your own voice are not steady noise, so they never
        reach the output this way, not even when the detector misses a word.
        """
        level = self.cleanup.ambient_gain
        noise_db = self.speech.noise_db if self.speech is not None else None
        if level <= 0 or clean is None or noise_db is None:
            return np.zeros_like(dry)
        # Only where the input is nothing but room: anything louder (voice, a breath, a
        # click) silences the ambience, so a sound the detector missed cannot leak through it.
        only_room = 1.0 - above_room(dry, self.client_rate, noise_db)
        return (level * (dry - clean) * only_room).astype(np.float32)

    def _join_ambience(self, ambience: np.ndarray, n: int, crossfade: int) -> np.ndarray:
        # Always the nominal (centre) offset: consecutive spans are contiguous in time.
        centre = self._search // 2
        out = ambience[centre:centre + n].copy()
        if self._ambient_tail is not None and len(self._ambient_tail) == crossfade:
            fade_in = np.sin(np.linspace(0.0, np.pi / 2, crossfade, dtype=np.float32)) ** 2
            out[:crossfade] = self._ambient_tail * (1.0 - fade_in) + out[:crossfade] * fade_in
        self._ambient_tail = ambience[centre + n:centre + n + crossfade].copy()
        return out

    # --- Output -------------------------------------------------------------

    def ambient(self) -> np.ndarray:
        """Output for the chunk just heard, without running the model (asleep): ambience only."""
        n = self._chunk
        crossfade = min(self._crossfade, n)
        span = n + crossfade + self._search
        end = self._emitted_end()
        self._tail = None
        self._resuming = True
        if self.cleanup.ambient_gain <= 0 or self.cleanup.noise_reduction_db <= 0:
            self._ambient_tail = None
            return np.zeros(n, dtype=np.float32)
        clean_history = self.denoiser.apply(self._history, self._speech_gain)

        def span_of(values: np.ndarray) -> np.ndarray:
            part = values[max(0, end - span):end]
            return np.pad(part, (span - len(part), 0)) if len(part) < span else part

        clean = span_of(clean_history) if clean_history is not self._history else None
        out = self._join_ambience(self._ambience(span_of(self._history), clean), n, crossfade)
        return soft_limit(out.astype(np.float32, copy=False))

    def convert(self, fade_out: bool = False) -> np.ndarray:
        """Output for the chunk just heard, rendered by the model."""
        n = self._chunk
        length = len(self._history)
        cleanup = self.cleanup
        clean_history = (self.denoiser.apply(self._history, self._speech_gain)
                         if cleanup.noise_reduction_db > 0 else None)
        if clean_history is self._history:   # nothing learnt yet
            clean_history = None
        model_source = clean_history if clean_history is not None else self._history

        model_input = resample(model_source, self.client_rate, MODEL_RATE)
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

        # The same span of the input: for the speech gain, the ambience and the level.
        h_end = end + round(dropped / MODEL_RATE * self.client_rate)
        h_start = h_end - span

        def input_span(values: np.ndarray) -> np.ndarray:
            part = values[max(0, h_start):max(0, h_end)]
            return np.pad(part, (span - len(part), 0)) if len(part) < span else part

        dry = input_span(self._history)
        clean = input_span(clean_history) if clean_history is not None else None
        speech = input_span(self._speech_gain) if cleanup.isolate_voice else np.ones(span, np.float32)
        heard = (self.speech.core(self._total - len(self._history) + max(0, h_start), span, self.client_rate)
                 if self.speech is not None and cleanup.isolate_voice else None)
        self.stats.noise_reduction_db = self.denoiser.reduction_now_db(dry, clean) if clean is not None else 0.0
        reference = clean if clean is not None else dry
        level = self.loudness.update(reference, region, speech)   # learnt even when not applied
        if cleanup.match_loudness:
            region = region * level
            self.stats.loudness_gain_db = round(self.loudness.gain_db, 1)
            region = envelope_ceiling(region, reference, self.client_rate, heard)
        else:
            region = envelope_ceiling(region, reference / max(level, 1e-6), self.client_rate, heard)
        if cleanup.isolate_voice and self.speech is not None:
            region = snr_gate(region, dry, self.client_rate, self.speech.noise_db)
        voice = (region * speech).astype(np.float32)

        if self._tail is None or len(self._tail) != crossfade:
            offset = self._search // 2
            output = voice[offset:offset + n].copy()
            if self._resuming:  # fade in from the ambience emitted while asleep
                ramp = min(self._fade, n)
                output[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        else:
            offset = _sola_offset(self._tail, voice, self._search)
            output = voice[offset:offset + n].copy()
            phase = np.linspace(0.0, np.pi / 2, crossfade, dtype=np.float32)
            fade_in = np.sin(phase) ** 2
            output[:crossfade] = self._tail * (1.0 - fade_in) + output[:crossfade] * fade_in
        self._tail = voice[offset + n:offset + n + crossfade].copy()
        self._resuming = False
        if fade_out:  # last render before sleeping: the voice fades out, the room carries on
            output *= np.linspace(1.0, 0.0, n, dtype=np.float32)

        output = output + self._join_ambience(self._ambience(dry, clean), n, crossfade)
        return soft_limit(output.astype(np.float32, copy=False))

    # --- One-call helpers ---------------------------------------------------

    def idle(self, chunk: np.ndarray) -> np.ndarray:
        """Keep listening without rendering (silence saver asleep); return the ambience.

        The history keeps advancing, so the first render after waking has the
        real audio before the onset as context and the attack is not lost.
        """
        if len(chunk) == 0:
            return np.zeros(0, dtype=np.float32)
        self.listen(chunk)
        return self.ambient()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed ``len(chunk)`` new samples; return exactly as many converted samples."""
        if len(chunk) == 0:
            return np.zeros(0, dtype=np.float32)
        self.listen(chunk)
        return self.convert()
