"""Neural noise suppression of the model's input: DeepFilterNet3, streaming.

RVC turns whatever reaches it into voice-like sound, so anything that is not
speech must be gone before the model hears it. A spectral gate only removes
steady noise; DeepFilterNet3 (Schröter et al., MIT/Apache-2.0), the reference
open real-time suppressor, also removes keyboard clicks, breaths, traffic and
other changing noise while leaving the voice intact. Measured on synthetic
material: room noise -33 dB, keyboard clicks -71 dB, breaths -98 dB, clean
speech changed by -32 dB (inaudible), at under 1 ms of CPU per 10 ms of audio.

The model is the official DeepFilterNet3 checkpoint exported here (see
backend.audio.deepfilter) to one ONNX graph that takes one 10 ms hop (480
samples at 48 kHz) plus its recurrent state and returns the enhanced hop,
30 ms late. Every chunk is filtered exactly once, and the
output is kept on the same timeline as the raw input history.

DeepFilterNet removes everything below about -50 dBFS, quiet speech
included, so a slow automatic gain brings the input to a normal speaking
level first and the same gain is taken off the output afterwards. Noise is
raised with the voice, so what counts as noise does not change.

Very expressive voice (a high laugh, a shout, some singing) can be taken
for noise. Where the input is clearly pitched, well above the room, and the
filter removed most of it, the original is put back (``restore_voice``):
noise, clicks and breaths are not pitched, so they stay out.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np

from backend.audio.deepfilter.network import initial_state
from backend.audio.resampling import resample
from backend.audio.voice_activity import FRAME as VAD_FRAME
from backend.audio.voice_activity import RATE as VAD_RATE
from backend.audio.voice_activity import VOCAL_ABOVE_NOISE_DB, VOCAL_PERIODICITY, vocal_frames

RATE = 48000
HOP = 480
DELAY = 1440              # output lags input by 30 ms: half a window plus two hops of lookahead
UNLIMITED_DB = 100.0      # attenuation limit meaning "remove all the noise"
TARGET_DB = -30.0         # quieter speech peaks are raised to this level before the filter
MAX_GAIN_DB = 40.0
HOLD_HOPS = 100           # the loudest recent hop is held for 1 s, so the gain stays put while talking
RELEASE_DB = 0.03         # then the level falls 3 dB/s
GAIN_STEP_DB = 0.05       # and the gain rises at most 5 dB/s
VOICE_CUT_DB = 6.0        # a pitched frame losing this much was taken for noise
RESTORE_RAMP_SECONDS = 0.02


def restore_voice(dry: np.ndarray, clean: np.ndarray, rate: int, noise_db: float | None) -> np.ndarray:
    """``clean`` with the original put back where the filter removed clearly pitched voice."""
    count = min(len(dry), len(clean)) * VAD_RATE // rate // VAD_FRAME
    if count == 0 or noise_db is None:
        return clean
    dry16 = resample(dry, rate, VAD_RATE)[: count * VAD_FRAME].astype(np.float64).reshape(count, VAD_FRAME)
    clean16 = resample(clean, rate, VAD_RATE)[: count * VAD_FRAME].astype(np.float64).reshape(count, VAD_FRAME)
    periodic, level = vocal_frames(dry16)
    kept = 10 * np.log10(np.maximum(np.mean(clean16 ** 2, axis=1), 1e-12))
    lost = (periodic >= VOCAL_PERIODICITY) & (level >= noise_db + VOCAL_ABOVE_NOISE_DB) & (kept < level - VOICE_CUT_DB)
    if not lost.any():
        return clean
    frame = VAD_FRAME * rate / VAD_RATE
    weight = lost[np.minimum((np.arange(len(clean)) / frame).astype(np.int64), count - 1)].astype(np.float32)
    ramp = max(1, round(RESTORE_RAMP_SECONDS * rate))
    window = np.hanning(2 * ramp + 1).astype(np.float32)
    weight = np.convolve(np.pad(weight, ramp, mode="edge"), window / window.sum(), mode="valid")[: len(clean)]
    n = min(len(dry), len(clean))
    out = clean.copy()
    out[:n] += weight[:n] * (dry[:n] - clean[:n])
    return out


@cache
def _session(model_path: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Two threads cut its time by a fifth where cores are to spare (the model runs on the GPU or other cores).
    options.intra_op_num_threads = 2 if (os.cpu_count() or 1) >= 8 else 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def load_neural_denoiser(model_path: Path | None, client_rate: int) -> NeuralDenoiser | None:
    if model_path is None or not model_path.is_file():
        return None
    try:
        return NeuralDenoiser(model_path, client_rate)
    except Exception:  # a broken download must never stop the stream
        import logging

        logging.getLogger(__name__).exception("DeepFilterNet failed to load; using the spectral gate")
        return None


class NeuralDenoiser:
    """Streaming DeepFilterNet3 with its output aligned to the input timeline."""

    def __init__(self, model_path: Path, client_rate: int) -> None:
        self._session = _session(str(model_path))
        self.client_rate = int(client_rate)
        self.attenuation_db = UNLIMITED_DB
        self.reset()

    def reset(self) -> None:
        self._state = initial_state()
        self._pending = np.zeros(0, np.float32)       # 48 kHz input not yet a whole hop
        self._aligned = np.zeros(0, np.float32)       # cleaned audio, aligned with the input (recent part)
        self._fed = 0                                 # 48 kHz input samples fed
        self._produced = 0                            # 48 kHz samples the filter has output
        self._level_db = TARGET_DB - MAX_GAIN_DB      # recent peak level of the input (quiet at first)
        self._hold = 0
        self._gain_db = MAX_GAIN_DB
        self._gains = [MAX_GAIN_DB] * (DELAY // HOP)          # gains of the hops still inside the filter

    def _run(self, audio48: np.ndarray) -> np.ndarray:
        data = np.concatenate([self._pending, audio48.astype(np.float32, copy=False)])
        hops = len(data) // HOP
        self._pending = data[hops * HOP:]
        if hops == 0:
            return np.zeros(0, np.float32)
        out = np.empty(hops * HOP, np.float32)
        limit = np.array(self.attenuation_db, np.float32)
        state = self._state
        for k in range(hops):
            frame = data[k * HOP:(k + 1) * HOP]
            rms_db = 10 * np.log10(float(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)
            if rms_db >= self._level_db:
                self._level_db, self._hold = rms_db, HOLD_HOPS
            elif self._hold > 0:
                self._hold -= 1
            else:
                self._level_db = max(rms_db, self._level_db - RELEASE_DB)
            wanted = float(np.clip(TARGET_DB - self._level_db, 0.0, MAX_GAIN_DB))
            self._gain_db = min(wanted, self._gain_db + GAIN_STEP_DB)   # down at once, up slowly
            self._gains.append(self._gain_db)
            enhanced, new_state, _ = self._session.run(
                None, {"input_frame": frame * np.float32(10 ** (self._gain_db / 20)), "states": state,
                       "atten_lim_db": limit})
            # The hop coming out now went in DELAY samples ago, with that hop's gain.
            out[k * HOP:(k + 1) * HOP] = np.asarray(enhanced, np.float32) * np.float32(10 ** (-self._gains.pop(0) / 20))
            state = np.asarray(new_state, np.float32)
        self._state = state
        return out

    def feed(self, chunk: np.ndarray) -> None:
        """Filter newly heard client-rate audio (call once per chunk, in order)."""
        audio48 = chunk if self.client_rate == RATE else resample(chunk, self.client_rate, RATE)
        self._fed += len(audio48)
        produced = self._run(np.asarray(audio48, np.float32))
        # Output sample j corresponds to input sample j - DELAY: drop the first DELAY once.
        drop = max(0, min(len(produced), DELAY - self._produced))
        self._produced += len(produced)
        self._aligned = np.concatenate([self._aligned, produced[drop:]])[-RATE * 8:]

    def history(self, length: int) -> np.ndarray:
        """The cleaned version of the last ``length`` client-rate input samples.

        The newest ~30 ms (the filter's delay plus a partial hop) are not out
        yet; they lie inside the stream's lookahead, which is never emitted,
        and are left silent.
        """
        ready = max(0, self._produced - DELAY)          # input samples that have cleaned output
        missing = self._fed - ready
        cleaned = np.concatenate([self._aligned, np.zeros(missing, np.float32)])
        if self.client_rate != RATE:
            cleaned = resample(cleaned, RATE, self.client_rate)
        if len(cleaned) >= length:
            return cleaned[-length:].astype(np.float32, copy=False)
        return np.concatenate([np.zeros(length - len(cleaned), np.float32), cleaned])
