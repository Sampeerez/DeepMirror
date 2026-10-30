"""Realtime audio WebSocket: ``/ws/audio``.

Protocol:
    1. The client sends a JSON config: ``{"sample_rate": int, "chunk_size": int, ...}``.
    2. It then streams binary frames (see ``backend.audio.protocol``) and may
       send JSON settings updates at any time.
    3. Every frame is answered with a converted frame of the same length; the
       header's reserved field carries the server time in 1/100 ms.
    4. About once a second the server sends a JSON ``status`` message.

The browser does not apply the same-origin policy to WebSockets, so the
Origin header is checked before the handshake (cross-site hijacking guard).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.audio.effects import EffectsChain
from backend.audio.neural_denoise import load_neural_denoiser
from backend.audio.protocol import decode_frame, encode_frame
from backend.audio.streaming import RealtimeVoiceStream, StreamCleanup, StreamTiming
from backend.audio.voice_activity import load_vad
from backend.config import Settings
from backend.inference.auto_pitch import AutoPitch, SpeakerProfile, auto_expressiveness
from backend.inference.pipeline import ConversionSettings, VoiceConverter
from backend.model_manager import ModelManager
from backend.security import is_origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

STATUS_INTERVAL_SECONDS = 1.0
# Overloaded: over the last 3 s, more than a quarter of the chunks used over 95% of their own duration.
OVERLOAD_WINDOW_SECONDS = 3.0
OVERLOAD_MIN_CHUNKS = 8
OVERLOAD_LOAD = 0.95
OVERLOAD_SHARE = 0.25
SAMPLE_RATE_RANGE = (8000, 192000)
CHUNK_SIZE_RANGE = (128, 65536)
SILENCE_THRESHOLD_RANGE = (-80.0, -20.0)
LOOKBACK_RANGE = (0.2, 3.0)
LOOKAHEAD_RANGE = (0.0, 0.5)
NOISE_REDUCTION_RANGE = (0.0, 60.0)   # 60 = remove all the noise
AMBIENT_RANGE = (-60.0, 0.0)


def bounded_number(value: Any, cast: type, low: float, high: float) -> Any:
    """``value`` cast and range-checked, or ``None`` when invalid (never clamps transport values)."""
    if isinstance(value, bool):
        return None
    try:
        number = cast(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(number, float) and not math.isfinite(number):
        return None
    return number if low <= number <= high else None


LEVEL_FRAME_SECONDS = 0.01


def chunk_level_db(audio: np.ndarray, sample_rate: int) -> float:
    """Loudest 10 ms RMS in the chunk, in dBFS.

    A whole-chunk RMS dilutes a short, quiet syllable with the silence around
    it; the loudest short frame does not.
    """
    if audio.size == 0:
        return -math.inf
    frame = max(1, int(sample_rate * LEVEL_FRAME_SECONDS))
    count = max(1, len(audio) // frame)
    frames = np.square(audio[: count * frame] if len(audio) >= frame else audio, dtype=np.float64)
    power = float(frames.reshape(count, -1).mean(axis=1).max())
    return 10.0 * math.log10(power) if power > 0 else -math.inf


@dataclass(slots=True)
class FrameResult:
    audio: np.ndarray
    mode: str          # rvc | dsp | bypass | error
    model_ms: float
    dsp_ms: float


class StreamSession:
    """All per-connection state: settings, effect chain, realtime converter, stats."""

    def __init__(self, manager: ModelManager, settings: Settings) -> None:
        self.manager = manager
        self.config = settings
        chunk_size, lookback, lookahead = manager.stream_defaults()
        self.sample_rate = settings.SAMPLE_RATE
        self.chunk_size = chunk_size
        self.conversion = manager.default_conversion()
        self.formant_shift = 0.0
        self._pending_f0: str | None = None   # pitch tracker loading in the background
        # "auto": follow the speaker's register onto the voice's (needs voice F0 stats).
        self.pitch_mode = "auto"
        self._auto_pitch: AutoPitch | None = None
        # The speaker's own pitch: the recorded voice profile, or learnt from the first seconds of speech.
        self.profile = SpeakerProfile()
        self._profile_revision = -1
        self._auto_pitch_voice: str | None = None
        self.effects: dict = {}
        self.bypass = False
        self.silence_saver = settings.SILENCE_SAVER
        self.silence_threshold_db = settings.SILENCE_THRESHOLD_DB
        # Around the model: voice isolation, noise reduction, ambience, loudness matching.
        self.cleanup = StreamCleanup(isolate_voice=settings.VOICE_ISOLATION,
                                     noise_reduction_db=settings.NOISE_REDUCTION_DB,
                                     ambient_db=settings.AMBIENT_DB, match_loudness=settings.MATCH_LOUDNESS)
        self.timing = StreamTiming(
            lookback_seconds=lookback,
            lookahead_seconds=lookahead,
            crossfade_seconds=settings.STREAM_CROSSFADE_SECONDS,
            search_seconds=settings.STREAM_SEARCH_SECONDS,
        )
        self.chain = EffectsChain(self.sample_rate)
        self._stream: RealtimeVoiceStream | None = None
        self._stream_converter: VoiceConverter | None = None
        self._silence_seconds = 0.0
        self.sleeping = False
        self.processing_error: str | None = None
        self._logged_failure: str | None = None
        # Stream health, written to the log so a stutter leaves a trace.
        self._slow_chunks = 0
        self._last_health_log = 0.0
        # (time, took / budget) of recent chunks: sustained lateness asks the client for a longer chunk.
        self._recent_load: deque[tuple[float, float]] = deque()
        self.listen_ms = 0.0
        self._client_health: dict[str, int] = {}
        # Test samples: the raw microphone and the converted output, saved while a take records.
        self._sample: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        # Stats surfaced in status messages.
        self.latency_ms = self.model_ms = self.dsp_ms = 0.0
        self.input_level_db = -math.inf
        self.mode = "dsp"
        self.eligible_frames = self.inference_frames = 0

    # --- Settings -----------------------------------------------------------

    def apply(self, payload: dict) -> None:
        """Merge a config or settings message; invalid fields are logged and ignored."""
        previous_f0 = self.conversion.f0_method
        self.conversion = self.conversion.with_payload(payload)
        formant = bounded_number(payload.get("formant_shift"), float, -24.0, 24.0)
        if formant is not None:
            self.formant_shift = formant
        if isinstance(payload.get("effects"), dict):
            self.effects = payload["effects"]
        if payload.get("pitch_mode") in ("auto", "manual"):
            self.pitch_mode = payload["pitch_mode"]
        if payload.get("recalibrate") is True:
            self._profile_revision = -1     # back to the recorded profile, or learn again without one
        if isinstance(payload.get("bypass"), bool):
            self.bypass = payload["bypass"]
        if isinstance(payload.get("client_health"), dict):
            self._report_client_health(payload["client_health"])
        if payload.get("capture_sample") is True and self._sample is None:
            self._sample = ([], [])
        elif payload.get("capture_sample") is False:
            self.save_sample()
        if isinstance(payload.get("silence_saver"), bool):
            self.silence_saver = payload["silence_saver"]
            self._silence_seconds = 0.0
            self.sleeping = self.sleeping and self.silence_saver
        threshold = bounded_number(payload.get("silence_threshold_db"), float, *SILENCE_THRESHOLD_RANGE)
        if threshold is not None:
            self.silence_threshold_db = threshold
        self._apply_cleanup(payload)
        self._preload_new_f0(previous_f0)
        self._apply_transport(payload)
        self._apply_timing(payload)

    def _preload_new_f0(self, previous: str) -> None:
        """A newly chosen pitch tracker loads in the background; until then the old one keeps running."""
        method = self.conversion.f0_method
        if method == previous:
            return
        self.manager.note_f0_method(method)
        if self.manager.converter is None or self.manager.pitch_ready(method):
            self._pending_f0 = None
            return
        self.conversion = replace(self.conversion, f0_method=previous)
        self._pending_f0 = method

        def load() -> None:
            try:
                self.manager.prepare_pitch(method)
            except Exception:
                logger.exception("Could not load the %s pitch tracker", method)
                self._pending_f0 = None
                return
            if self._pending_f0 == method:
                self.conversion = replace(self.conversion, f0_method=method)
                self._pending_f0 = None

        threading.Thread(target=load, name=f"load-{method}", daemon=True).start()

    def _apply_cleanup(self, payload: dict) -> None:
        cleanup = self.cleanup
        changes: dict[str, Any] = {}
        for key in ("isolate_voice", "match_loudness"):
            if isinstance(payload.get(key), bool):
                changes[key] = payload[key]
        reduction = bounded_number(payload.get("noise_reduction_db"), float, *NOISE_REDUCTION_RANGE)
        if reduction is not None:
            changes["noise_reduction_db"] = reduction
        ambient = bounded_number(payload.get("ambient_db"), float, *AMBIENT_RANGE)
        if ambient is not None:
            changes["ambient_db"] = ambient
        if changes:
            self.cleanup = replace(cleanup, **changes)
            if self._stream is not None:
                self._stream.set_cleanup(self._stream_cleanup())

    def _stream_cleanup(self) -> StreamCleanup:
        """The cleanup the stream can honour: voice isolation needs the neural detector."""
        stream = self._stream
        neural = stream is not None and stream.speech is not None and stream.speech.neural
        return replace(self.cleanup, isolate_voice=self.cleanup.isolate_voice and neural)

    def _apply_transport(self, payload: dict) -> None:
        for key, (low, high) in (("sample_rate", SAMPLE_RATE_RANGE), ("chunk_size", CHUNK_SIZE_RANGE)):
            if key not in payload:
                continue
            value = bounded_number(payload[key], int, low, high)
            if value is None:
                logger.warning("Ignoring invalid %s %r (expected %d-%d)", key, payload[key], low, high)
                continue
            if key == "chunk_size" and value != self.chunk_size and self._recent_load:
                logger.info("Stream chunk changed to %d samples (%.0f ms)",
                            value, value * 1000.0 / self.sample_rate)
            setattr(self, key, value)
            self._stream = None  # window geometry depends on both
            self._recent_load.clear()
        if self.chain.sr != self.sample_rate:
            self.chain = EffectsChain(self.sample_rate)

    def _apply_timing(self, payload: dict) -> None:
        lookback = bounded_number(payload.get("lookback_seconds"), float, *LOOKBACK_RANGE)
        lookahead = bounded_number(payload.get("lookahead_seconds"), float, *LOOKAHEAD_RANGE)
        if lookback is None and lookahead is None:
            return
        self.timing = StreamTiming(
            lookback_seconds=self.timing.lookback_seconds if lookback is None else lookback,
            lookahead_seconds=self.timing.lookahead_seconds if lookahead is None else lookahead,
            crossfade_seconds=self.timing.crossfade_seconds,
            search_seconds=self.timing.search_seconds,
        )
        self._stream = None

    # --- Processing (runs in a worker thread) -------------------------------

    def _sync_profile(self) -> None:
        """Follow the recorded voice profile and voice registers when they change."""
        profiles = self.manager.profiles
        if profiles.revision == self._profile_revision:
            return
        self._profile_revision = profiles.revision
        recorded = profiles.stats()
        self.profile = SpeakerProfile(recorded, fixed=True) if recorded is not None else SpeakerProfile()
        self._auto_pitch = None

    def _auto_pitch_for(self, converter: VoiceConverter) -> AutoPitch | None:
        """The register tracker for the active voice, when automatic pitch applies."""
        self._sync_profile()
        stats = converter.voice.f0_stats
        if self.pitch_mode != "auto" or stats is None:
            return None
        if (self._auto_pitch is None or self._auto_pitch_voice != converter.voice.name
                or self._auto_pitch.target is not stats):
            # With a recorded profile the shift is known at once; otherwise start where the last stream ended.
            initial = None if self.profile.fixed else self.manager.remembered_pitch_shift(converter.voice.name)
            self._auto_pitch = AutoPitch(stats, self.profile, initial)
            self._auto_pitch_voice = converter.voice.name
        return self._auto_pitch

    def _render_settings(self, converter: VoiceConverter) -> ConversionSettings:
        settings = self.conversion
        self._sync_profile()
        speaker = self.profile.stats
        if speaker is not None:
            settings = replace(settings, pitch_center_hz=speaker.median_hz)
        auto = self._auto_pitch_for(converter)
        if auto is not None:
            if auto.shift is not None:
                settings = replace(settings, pitch_shift=auto.shift)
            if speaker is not None:   # the target voice's own melodic range
                settings = replace(settings, pitch_expressiveness=auto_expressiveness(auto.target, speaker))
        return settings

    @property
    def expressiveness_applied(self) -> float:
        converter = self.manager.converter
        if converter is None:
            return self.conversion.pitch_expressiveness
        return round(self._render_settings(converter).pitch_expressiveness, 2)

    @property
    def auto_pitch_shift(self) -> float | None:
        converter = self.manager.converter
        auto = self._auto_pitch_for(converter) if converter is not None else None
        return auto.shift if auto is not None else None

    def _realtime_stream(self, converter: VoiceConverter) -> RealtimeVoiceStream:
        if self._stream is None or self._stream_converter is not converter:
            config = self.config

            timing = self.timing
            emitted = (self.chunk_size / self.sample_rate + timing.lookahead_seconds
                       + timing.crossfade_seconds + timing.search_seconds)
            # The look-ahead is context only: it is not vocoded, except a margin for the decoder's
            # receptive field next to the emitted audio (measured: the emitted samples stay within
            # 4e-5 of a full decode).
            skip = max(0.0, timing.lookahead_seconds - config.STREAM_DECODE_MARGIN_SECONDS)

            # F0 frames of the newest chunk: the window ends ``lookahead`` after them.
            new_frames = max(1, round(self.chunk_size / self.sample_rate * 100))
            ahead_frames = round(timing.lookahead_seconds * 100)

            def observe_f0(f0: np.ndarray) -> None:
                end = len(f0) - ahead_frames
                self._sync_profile()
                pause = self.profile.observe(f0[max(0, end - new_frames):max(0, end)])
                auto = self._auto_pitch_for(converter)
                if auto is None:
                    return
                auto.update(pause)
                if auto.shift is not None:
                    self.manager.remember_pitch_shift(converter.voice.name, auto.shift)

            def render(window: np.ndarray) -> tuple[np.ndarray, int]:
                audio = converter.convert(window, self._render_settings(converter),
                                          pad_seconds=config.STREAM_PAD_SECONDS, split_long_input=False,
                                          rms_frame_seconds=config.STREAM_RMS_FRAME_SECONDS,
                                          tail_seconds=emitted + config.STREAM_DECODE_MARGIN_SECONDS,
                                          skip_seconds=skip, on_f0=observe_f0)
                return audio, converter.output_rate

            vad = load_vad(self.manager.vad_path(), lambda: self.silence_threshold_db)
            neural = (load_neural_denoiser(self.manager.denoiser_path(), self.sample_rate)
                      if self.manager.neural_denoise_fits(self.chunk_size, self.sample_rate) else None)
            stream = RealtimeVoiceStream(self.sample_rate, render, self.timing, vad=vad, neural=neural)
            self._stream = stream
            stream.set_cleanup(self._stream_cleanup())
            self._stream_converter = converter
        return self._stream

    def _sleep_after_seconds(self) -> float:
        """Silence needed before sleeping: the hold plus everything still in flight.

        Output lags input by the lookahead, the SOLA tail and one chunk, so
        sleeping any earlier would cut the end of the last word.
        """
        return (self.config.SILENCE_HOLD_SECONDS + self.timing.algorithmic_delay()
                + self.chunk_size / self.sample_rate)

    def _update_silence(self, audio: np.ndarray, speech: bool) -> bool:
        """True when inference may sleep: no speech for the hold plus everything in flight."""
        if not self.silence_saver:
            self._silence_seconds = 0.0
            return False
        if speech:
            self._silence_seconds = 0.0
            return False
        self._silence_seconds += len(audio) / self.sample_rate
        return self._silence_seconds >= self._sleep_after_seconds()

    def _report_client_health(self, health: dict) -> None:
        """Log new playback gaps and lost frames the browser reports."""
        labels = {"underruns": "playback gaps", "dropped": "frames dropped before sending",
                  "timed_out": "frames that came back too late"}
        news = []
        for key, label in labels.items():
            value = health.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > self._client_health.get(key, 0):
                news.append(f"{value - self._client_health.get(key, 0)} {label}")
                self._client_health[key] = value
        if news:
            logger.warning("Audio stuttered in the browser: %s (model %.0f ms, budget %.0f ms per chunk)",
                           ", ".join(news), self.model_ms, self.chunk_size * 1000.0 / self.sample_rate)

    def note_timing(self, elapsed_ms: float) -> None:
        """Count chunks that took longer than their own duration; log them at most every 5 s."""
        budget = self.chunk_size * 1000.0 / self.sample_rate
        now = time.monotonic()
        self._recent_load.append((now, elapsed_ms / budget))
        while self._recent_load and now - self._recent_load[0][0] > OVERLOAD_WINDOW_SECONDS:
            self._recent_load.popleft()
        if elapsed_ms <= budget:
            return
        self._slow_chunks += 1
        if now - self._last_health_log >= 5.0:
            logger.warning("%d chunk(s) took longer than real time (last %.0f ms for %.0f ms of audio; %s)",
                           self._slow_chunks, elapsed_ms, budget, self._stage_text())
            self._slow_chunks = 0
            self._last_health_log = now

    def _stage_text(self) -> str:
        """Where the time of the last chunk went, for the log."""
        converter = self.manager.converter
        parts = [f"noise and voice detection {self.listen_ms:.0f} ms", f"model {self.model_ms:.0f} ms"]
        if converter is not None:
            parts += [f"{name} {value:.0f} ms" for name, value in converter.timings.items()]
        parts.append(f"rest {max(self.dsp_ms - self.listen_ms, 0.0):.0f} ms")
        return ", ".join(parts)

    @property
    def overloaded(self) -> bool:
        """Most recent chunks barely made it, or did not: the stream needs a longer chunk."""
        loads = [load for _, load in self._recent_load]
        if len(loads) < OVERLOAD_MIN_CHUNKS or self.bypass or self.sleeping:
            return False
        return sum(load > OVERLOAD_LOAD for load in loads) / len(loads) > OVERLOAD_SHARE

    def save_sample(self) -> Path | None:
        """Write the captured microphone and output to data/samples (``<time>-mic.wav``, ``-out.wav``)."""
        sample, self._sample = self._sample, None
        if not sample or not sample[0]:
            return None
        import soundfile as sf

        folder = self.config.DATA_DIR / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / time.strftime("%Y%m%d-%H%M%S")
        sf.write(f"{stem}-mic.wav", np.concatenate(sample[0]), self.sample_rate, subtype="FLOAT")
        sf.write(f"{stem}-out.wav", np.concatenate(sample[1]), self.sample_rate, subtype="FLOAT")
        logger.info("Saved test sample %s-mic.wav / -out.wav", stem.name)
        return stem

    def process(self, audio: np.ndarray) -> FrameResult:
        """Convert one chunk. With ``bypass`` the input is returned untouched (true A/B)."""
        result = self._process(audio)
        if self._sample is not None:
            self._sample[0].append(np.array(audio, np.float32))
            self._sample[1].append(np.array(result.audio, np.float32))
        return result

    def _process(self, audio: np.ndarray) -> FrameResult:
        self.processing_error = None
        if self.bypass:
            self.sleeping = False
            return FrameResult(audio, "bypass", 0.0, 0.0)

        started = time.perf_counter()
        processed = self.chain.pre_process(audio, self.effects)
        self.input_level_db = chunk_level_db(processed, self.sample_rate)
        converter = self.manager.converter
        model_ms = 0.0
        mode = "dsp"

        if converter is None:
            self.sleeping = False
            processed = self.chain.apply_pitch(processed, self.conversion.pitch_shift)
        else:
            mode = "rvc"
            self.eligible_frames += 1
            arbiter = self.manager.arbiter
            arbiter.chunk_arrived()
            stream = self._realtime_stream(converter)

            listen_started = time.perf_counter()
            speech = stream.listen(processed)
            self.listen_ms = (time.perf_counter() - listen_started) * 1000.0
            asleep = self._update_silence(processed, speech)
            if asleep and self.sleeping:
                # Keep listening (the next onset gets real context); the room still plays.
                processed = stream.ambient()
            else:
                model_started = time.perf_counter()
                try:
                    # Last render before sleeping: the voice fades out instead of stepping to zero.
                    with arbiter.voice():
                        rendered = stream.convert(fade_out=asleep)
                except Exception as exc:
                    return self._failure(exc, audio, model_started)
                if asleep and self._auto_pitch is not None:
                    self._auto_pitch.on_pause()
                processed = rendered
                self.sleeping = asleep
                model_ms = (time.perf_counter() - model_started) * 1000.0
                self.inference_frames += 1
                self._logged_failure = None

        processed = self.chain.post_process(processed, self.effects, self.formant_shift)
        total_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(processed, mode, model_ms, max(total_ms - model_ms, 0.0))

    def _failure(self, exc: Exception, audio: np.ndarray, started: float) -> FrameResult:
        failure = type(exc).__name__
        if self._logged_failure != failure:
            logger.exception("Model inference failed; muting the stream")
        self._logged_failure = failure
        # Only the exception type reaches the client: messages can contain paths.
        self.processing_error = (f"Model inference failed ({failure}). Output is muted. "
                                 "Check the server log, then reload the model.")
        self.sleeping = False
        self.chain = EffectsChain(self.sample_rate)  # drop effect tails of the failed audio
        self._stream = None
        return FrameResult(np.zeros_like(audio), "error", (time.perf_counter() - started) * 1000.0, 0.0)

    # --- Status -------------------------------------------------------------

    def _cleanup_status(self) -> dict:
        stream = self._stream
        stats = stream.stats if stream is not None else None
        return {
            "isolate_voice": self.cleanup.isolate_voice,
            "noise_reduction_db": self.cleanup.noise_reduction_db,
            "ambient_db": self.cleanup.ambient_db,
            "match_loudness": self.cleanup.match_loudness,
            "voice_detector": "neural" if (stats and stats.neural_vad) or (stream is None and self.manager.vad_path())
            else "level",
            "speech_probability": stats.speech_probability if stats else 0.0,
            "denoiser": stream.denoiser_kind if stream is not None else None,
            "noise_reduced_db": stats.noise_reduction_db if stats else 0.0,
            "loudness_gain_db": stats.loudness_gain_db if stats else 0.0,
        }

    def status(self) -> dict:
        converter = self.manager.converter
        active = self.manager.active_model_name
        mode = "bypass" if self.bypass else (self.mode if active else "dsp")
        duty = round(self.inference_frames * 100.0 / self.eligible_frames, 1) if self.eligible_frames else 0.0
        message = {
            "type": "status",
            "latency_ms": round(self.latency_ms, 2),
            "model_ms": round(self.model_ms, 2),
            "dsp_ms": round(self.dsp_ms, 2),
            "overloaded": self.overloaded,
            "mode": mode,
            "processing_error": self.processing_error,
            "bypass": self.bypass,
            "active_model": active,
            "loading_model": (self.manager.loading or {}).get("name"),
            "loading_stage": (self.manager.loading or {}).get("stage"),
            "f0_pending": self._pending_f0,
            "silence_saver": self.silence_saver,
            "silence_threshold_db": self.silence_threshold_db,
            "inference_sleeping": bool(active and not self.bypass and self.sleeping),
            "inference_duty_percent": duty if active and not self.bypass else 0.0,
            "effects_active": 0 if self.bypass else EffectsChain.count_active(self.effects, self.formant_shift),
            "algorithmic_delay_ms": round(self.timing.algorithmic_delay() * 1000.0, 1),
            "pitch_mode": self.pitch_mode,
            "auto_pitch_available": bool(converter is not None and converter.voice.f0_stats is not None),
            "auto_pitch_shift": self.auto_pitch_shift,
            "speaker_profile": self.profile.describe(),
            "expressiveness": self.expressiveness_applied,
            "input_level_db": None if math.isinf(self.input_level_db) else round(self.input_level_db, 1),
            **self._cleanup_status(),
        }
        self.eligible_frames = self.inference_frames = 0
        return message


def _raise_priority() -> None:
    """Windows: run above normal priority, so a browser or a game in the background delays chunks less."""
    if sys.platform != "win32":
        return
    import ctypes

    above_normal = 0x8000
    kernel32 = ctypes.windll.kernel32
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), above_normal):
        logger.debug("Could not raise the process priority")


async def _read_config(websocket: WebSocket, session: StreamSession) -> bool:
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
    except TimeoutError:
        await websocket.close(code=1008, reason="Config timeout")
        return False
    except Exception:
        return False
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.close(code=1003, reason="Invalid JSON config")
        return False
    if not isinstance(config, dict):
        await websocket.close(code=1003, reason="Config must be a JSON object")
        return False
    if bounded_number(config.get("sample_rate", session.sample_rate), int, *SAMPLE_RATE_RANGE) is None:
        await websocket.close(code=1003, reason="sample_rate must be 8000-192000")
        return False
    if bounded_number(config.get("chunk_size", session.chunk_size), int, *CHUNK_SIZE_RANGE) is None:
        await websocket.close(code=1003, reason="chunk_size must be 128-65536")
        return False
    session.apply(config)
    return True


async def _send_status(websocket: WebSocket, session: StreamSession) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json(session.status())


async def _handle_frame(websocket: WebSocket, session: StreamSession, data: bytes) -> None:
    started = time.perf_counter()
    try:
        audio, sequence, _ = decode_frame(data)
    except ValueError as exc:
        logger.warning("Invalid audio frame: %s", exc)
        return
    result = await asyncio.to_thread(session.process, audio)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    session.latency_ms, session.model_ms, session.dsp_ms, session.mode = (
        elapsed_ms, result.model_ms, result.dsp_ms, result.mode)
    session.note_timing(elapsed_ms)
    await websocket.send_bytes(encode_frame(result.audio, sequence, min(int(elapsed_ms * 100), 0xFFFFFFFF)))
    if session.processing_error is not None and result.mode == "error":
        await _send_status(websocket, session)


@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    settings: Settings = websocket.app.state.settings
    origin = websocket.headers.get("origin")
    if not is_origin_allowed(origin, websocket.headers.get("host"), settings.CORS_ORIGINS,
                             settings.ALLOW_ANY_ORIGIN):
        logger.warning("Rejected WebSocket handshake from origin %r", origin)
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    session = StreamSession(websocket.app.state.model_manager, settings)
    if not await _read_config(websocket, session):
        return
    logger.info("Audio stream started: %d Hz, %d-sample chunks", session.sample_rate, session.chunk_size)
    _raise_priority()

    last_status = time.perf_counter()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                await _handle_frame(websocket, session, message["bytes"])
                if time.perf_counter() - last_status >= STATUS_INTERVAL_SECONDS:
                    await _send_status(websocket, session)
                    last_status = time.perf_counter()
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid JSON message")
                    continue
                if isinstance(payload, dict):
                    session.apply(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Audio stream error")
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="Internal server error")
    session.save_sample()
    logger.info("Audio stream closed")
