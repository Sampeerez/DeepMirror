"""Which parts of the input are someone speaking.

Silero VAD separates speech from room noise, breaths and clicks at any level;
a level detector stands in until its weights are downloaded. Frames that are
clearly pitched and above the room count as voice too, so a held vowel or a
rolled r is never cut.

``SpeechTimeline`` turns the per-frame decisions into a smooth 0..1 gain with
hysteresis, a minimum of confident frames (a breath is not speech), pre-roll
for attacks and hang-over for word endings.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

RATE = 16000
FRAME = 512                 # Silero's frame at 16 kHz (32 ms)
CONTEXT = 64                # samples of the previous frame it expects in front
ENTER, LEAVE = 0.5, 0.35
# A segment is speech only with this many confident frames: breaths and clicks
# make the detector waver around 0.5 for a frame or two, speech sits near 1.
CONFIDENT, MIN_CONFIDENT_FRAMES = 0.7, 3
PRE_ROLL_SECONDS = 0.08
HANG_OVER_SECONDS = 0.3
RAMP_SECONDS = 0.02
HISTORY_SECONDS = 8.0       # decisions kept (more than any stream window)
# Laughs, humming and "mmm" are voice too, but a speech detector may not call
# them speech. A frame that is clearly periodic (pitched) and well above the
# room counts as voice; breaths and mouth clicks are not periodic and stay out.
VOCAL_PERIODICITY = 0.45   # voice reads 0.7+, a rolled r 0.45-0.6, noise and breaths below 0.2
VOCAL_ABOVE_NOISE_DB = 12.0
VOCAL_PROBABILITY = 0.8
MIN_LAG, MAX_LAG = RATE // 600, RATE // 70
ENVELOPE_SAMPLES = RATE // 200
VOICE_BAND_HZ = 1000
MIN_LOW_SHARE = 0.3        # a voice has most of its energy below 1 kHz; key clicks do not
# Room noise is learnt only from frames that are clearly not voice.
NOISE_MAX_PERIODICITY = 0.35


def vocal_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 512-sample frame: (normalised autocorrelation peak in the voice pitch range, level dB).

    Periodicity is measured below 1 kHz, where a voice's fundamental and first
    harmonics are and a key's ringing is not, and with the loudness flattened
    (5 ms envelope), so the taps of a rolled r do not hide that it is voice.
    """
    x = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sum(x * x, axis=1)
    level = 10 * np.log10(np.maximum(energy / frames.shape[1], 1e-12))
    size = frames.shape[1]
    low = np.fft.rfft(x, axis=1)
    low[:, int(VOICE_BAND_HZ * size / RATE) + 1:] = 0
    x = np.fft.irfft(low, n=size, axis=1)
    low_share = np.sum(x * x, axis=1) / np.maximum(energy, 1e-12)
    envelope = np.sqrt(ndimage.uniform_filter1d(x * x, ENVELOPE_SAMPLES, axis=1, mode="nearest"))
    x = x / (envelope + 1e-3 * np.sqrt(energy / frames.shape[1] + 1e-12)[:, None] + 1e-12)
    energy = np.sum(x * x, axis=1)
    spectrum = np.fft.rfft(x, n=2 * frames.shape[1], axis=1)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, : MAX_LAG + 1]
    peak = autocorrelation[:, MIN_LAG:MAX_LAG + 1].max(axis=1) / np.maximum(energy, 1e-12)
    return np.where(low_share >= MIN_LOW_SHARE, peak, 0.0), level


@cache
def _session(model_path: str):
    """One ONNX Runtime session per process; each stream keeps its own recurrent state."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


class SileroVad:
    """Streaming Silero VAD: feed 16 kHz audio, get one speech probability per 32 ms."""

    neural = True

    def __init__(self, model_path: Path) -> None:
        self._session = _session(str(model_path))
        self._rate = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros((1, CONTEXT), np.float32)
        self._pending = np.zeros(0, np.float32)

    def feed(self, audio16: np.ndarray) -> np.ndarray:
        audio = np.concatenate([self._pending, np.asarray(audio16, np.float32)])
        count = len(audio) // FRAME
        self._pending = audio[count * FRAME:]
        probabilities = np.empty(count, np.float32)
        for k in range(count):
            frame = np.concatenate([self._context, audio[None, k * FRAME:(k + 1) * FRAME]], axis=1)
            output, state = self._session.run(None, {"input": frame, "state": self._state, "sr": self._rate})
            self._state = np.asarray(state, np.float32)
            self._context = frame[:, -CONTEXT:]
            probabilities[k] = float(np.asarray(output).reshape(-1)[0])
        return probabilities


class LevelVad:
    """Stand-in when Silero is unavailable: a frame is speech when it is loud enough."""

    neural = False

    def __init__(self, threshold_db: Callable[[], float]) -> None:
        self.threshold_db = threshold_db
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, np.float32)

    def feed(self, audio16: np.ndarray) -> np.ndarray:
        audio = np.concatenate([self._pending, np.asarray(audio16, np.float32)])
        count = len(audio) // FRAME
        self._pending = audio[count * FRAME:]
        if count == 0:
            return np.zeros(0, np.float32)
        frames = audio[: count * FRAME].reshape(count, FRAME).astype(np.float64)
        # Loudest 10 ms (160 samples) of each frame, like the chunk level meter.
        power = np.square(frames[:, :480]).reshape(count, 3, 160).mean(axis=2).max(axis=1)
        level = 10 * np.log10(np.maximum(power, 1e-12))
        return (level >= self.threshold_db()).astype(np.float32)


def load_vad(model_path: Path | None, threshold_db: Callable[[], float]) -> SileroVad | LevelVad:
    if model_path is not None and model_path.is_file():
        try:
            return SileroVad(model_path)
        except Exception:
            logger.exception("Silero VAD failed to load; using the level detector")
    return LevelVad(threshold_db)


class SpeechTimeline:
    """Speech decisions on the absolute 16 kHz timeline of one stream."""

    def __init__(self, detector: SileroVad | LevelVad, history_seconds: float = HISTORY_SECONDS) -> None:
        self.detector = detector
        self.history_seconds = history_seconds
        self.reset()

    @property
    def neural(self) -> bool:
        return bool(self.detector.neural)

    def reset(self) -> None:
        self.detector.reset()
        self._probabilities = np.zeros(0, np.float32)
        self._first_frame = 0          # absolute index of _probabilities[0]
        self._fed = 0                  # 16 kHz samples fed so far
        self._pending = np.zeros(0, np.float32)
        self._noise_db = -70.0         # room level, learnt from non-speech frames
        self._noise_heard = False

    @property
    def frames_known(self) -> int:
        return self._first_frame + len(self._probabilities)

    def feed(self, audio16: np.ndarray) -> np.ndarray:
        """Add newly heard 16 kHz audio; returns the probabilities of the frames it completed."""
        fresh = self.detector.feed(audio16)
        self._fed += len(audio16)
        audio = np.concatenate([self._pending, np.asarray(audio16, np.float32)])
        count = len(fresh)
        if count and len(audio) >= count * FRAME:
            periodic, level = vocal_frames(audio[: count * FRAME].reshape(count, FRAME).astype(np.float64))
            # A held vowel can look like "not speech" to the detector; it is never the room.
            quiet = (fresh < 0.2) & (periodic < NOISE_MAX_PERIODICITY)
            if quiet.any():
                # Minimum tracking: the room is the quietest the input gets. Follow it down
                # quickly, up only slowly, so a long breath or hum is never taken for the room.
                heard = float(np.percentile(level[quiet], 20))
                if not self._noise_heard:
                    self._noise_db = heard
                elif heard < self._noise_db:
                    self._noise_db = 0.5 * self._noise_db + 0.5 * heard
                else:   # at most 0.1 dB per frame (3 dB/s) upwards
                    self._noise_db += min(0.02 * (heard - self._noise_db), 0.1)
                self._noise_heard = True
            vocal = (periodic >= VOCAL_PERIODICITY) & (level >= self._noise_db + VOCAL_ABOVE_NOISE_DB)
            fresh = np.where(vocal, np.maximum(fresh, VOCAL_PROBABILITY), fresh).astype(np.float32)
        self._pending = audio[count * FRAME:]
        self._probabilities = np.concatenate([self._probabilities, fresh])
        keep = int(self.history_seconds * RATE / FRAME)
        if len(self._probabilities) > keep:
            drop = len(self._probabilities) - keep
            self._probabilities = self._probabilities[drop:]
            self._first_frame += drop
        return fresh

    @property
    def noise_db(self) -> float | None:
        """The room's level (dBFS) learnt from non-speech frames, once some were heard."""
        return self._noise_db if self._noise_heard else None

    def last_probability(self) -> float:
        return float(self._probabilities[-1]) if len(self._probabilities) else 0.0

    def _speech_frames(self) -> np.ndarray:
        """Hysteresis, pre-roll and hang-over over the known frames (bool per frame)."""
        probabilities = self._probabilities
        state = np.zeros(len(probabilities), bool)
        speaking = False
        start = 0
        for i, p in enumerate(probabilities):
            now = p >= ENTER or (speaking and p >= LEAVE)
            if now and not speaking:
                start = i
            if speaking and not now:
                self._accept(state, probabilities, start, i)
            speaking = now
        if speaking:
            self._accept(state, probabilities, start, len(probabilities))
        if not state.any():
            return state
        frame_seconds = FRAME / RATE
        pre, hang = math.ceil(PRE_ROLL_SECONDS / frame_seconds), math.ceil(HANG_OVER_SECONDS / frame_seconds)
        kernel = np.ones(pre + hang + 1)
        # Frame i is speech if any speech frame lies in [i - hang, i + pre].
        widened = np.convolve(state.astype(float), kernel, mode="full")[pre:pre + len(state)] > 0
        return widened

    @staticmethod
    def _accept(state: np.ndarray, probabilities: np.ndarray, start: int, end: int) -> None:
        if int(np.count_nonzero(probabilities[start:end] >= CONFIDENT)) >= MIN_CONFIDENT_FRAMES:
            state[start:end] = True

    def gain(self, start: int, length: int, rate: int) -> np.ndarray:
        """Speech gain (0..1) for ``length`` samples at ``rate`` starting at absolute sample ``start``.

        Frames not heard yet repeat the newest decision.
        """
        if length <= 0:
            return np.zeros(0, np.float32)
        speech = self._speech_frames()
        times = (start + np.arange(length)) * (RATE / rate)          # in 16 kHz samples
        frames = np.floor(times / FRAME).astype(np.int64) - self._first_frame
        if len(speech) == 0:
            return np.zeros(length, np.float32)
        hard = speech[np.clip(frames, 0, len(speech) - 1)].astype(np.float32)
        ramp = max(1, round(RAMP_SECONDS * rate))
        if ramp > 1 and 0 < hard.sum() < length:
            window = np.sin(np.linspace(0, np.pi, 2 * ramp + 1)) ** 2
            # Pad with the edge values so the span's borders are not faded.
            padded = np.pad(hard, (ramp, ramp), mode="edge")
            hard = (np.convolve(padded, window / window.sum(), mode="same")[ramp:-ramp]).astype(np.float32)
        return np.clip(hard, 0.0, 1.0)

    def core(self, start: int, length: int, rate: int) -> np.ndarray:
        """Samples where the detector itself hears speech (no pre-roll or hang-over widening)."""
        if length <= 0 or len(self._probabilities) == 0:
            return np.zeros(max(length, 0), bool)
        heard = self._probabilities >= LEAVE
        frames = np.floor((start + np.arange(length)) * (RATE / rate) / FRAME).astype(np.int64) - self._first_frame
        return heard[np.clip(frames, 0, len(heard) - 1)]

    def noise_only(self, start: int, length: int, rate: int) -> np.ndarray:
        """Samples that are certainly not speech (bool): safe to learn the room noise from.

        Stricter than ``gain() == 0``: the detector must be clearly below its
        thresholds, away from any (even rejected) speech-like wobble, and old
        enough that no future frame can still turn it into speech.
        """
        if length <= 0 or len(self._probabilities) == 0:
            return np.zeros(max(length, 0), bool)
        unsure = self._probabilities >= 0.2
        margin = math.ceil(HANG_OVER_SECONDS * RATE / FRAME)
        unsure = np.convolve(unsure.astype(float), np.ones(2 * margin + 1), mode="same") > 0
        settled = np.arange(len(unsure)) < len(unsure) - margin   # the newest frames may still change
        quiet = ~unsure & settled
        frames = np.floor((start + np.arange(length)) * (RATE / rate) / FRAME).astype(np.int64) - self._first_frame
        known = (frames >= 0) & (frames < len(quiet))
        return known & quiet[np.clip(frames, 0, len(quiet) - 1)]

    def speech_in_last(self, seconds: float) -> bool:
        """Whether the detector heard speech in the newest ``seconds`` (before pre-roll widening)."""
        frames = max(1, math.ceil(seconds * RATE / FRAME))
        recent = self._probabilities[-frames:]
        return bool(len(recent) and recent.max() >= ENTER)
