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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.audio.effects import EffectsChain
from backend.audio.protocol import decode_frame, encode_frame
from backend.audio.streaming import RealtimeVoiceStream, StreamTiming
from backend.config import Settings
from backend.inference.pipeline import VoiceConverter
from backend.model_manager import ModelManager
from backend.security import is_origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

STATUS_INTERVAL_SECONDS = 1.0
SAMPLE_RATE_RANGE = (8000, 192000)
CHUNK_SIZE_RANGE = (128, 65536)
SILENCE_THRESHOLD_RANGE = (-80.0, -20.0)
LOOKBACK_RANGE = (0.2, 3.0)
LOOKAHEAD_RANGE = (0.0, 0.5)


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


def chunk_rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -math.inf
    rms = math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0 else -math.inf


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
        self.effects: dict = {}
        self.bypass = False
        self.silence_saver = settings.SILENCE_SAVER
        self.silence_threshold_db = settings.SILENCE_THRESHOLD_DB
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
        # Stats surfaced in status messages.
        self.latency_ms = self.model_ms = self.dsp_ms = 0.0
        self.mode = "dsp"
        self.eligible_frames = self.inference_frames = 0

    # --- Settings -----------------------------------------------------------

    def apply(self, payload: dict) -> None:
        """Merge a config or settings message; invalid fields are logged and ignored."""
        self.conversion = self.conversion.with_payload(payload)
        formant = bounded_number(payload.get("formant_shift"), float, -24.0, 24.0)
        if formant is not None:
            self.formant_shift = formant
        if isinstance(payload.get("effects"), dict):
            self.effects = payload["effects"]
        if isinstance(payload.get("bypass"), bool):
            self.bypass = payload["bypass"]
        if isinstance(payload.get("silence_saver"), bool):
            self.silence_saver = payload["silence_saver"]
            self._silence_seconds = 0.0
            self.sleeping = self.sleeping and self.silence_saver
        threshold = bounded_number(payload.get("silence_threshold_db"), float, *SILENCE_THRESHOLD_RANGE)
        if threshold is not None:
            self.silence_threshold_db = threshold
        self._apply_transport(payload)
        self._apply_timing(payload)

    def _apply_transport(self, payload: dict) -> None:
        for key, (low, high) in (("sample_rate", SAMPLE_RATE_RANGE), ("chunk_size", CHUNK_SIZE_RANGE)):
            if key not in payload:
                continue
            value = bounded_number(payload[key], int, low, high)
            if value is None:
                logger.warning("Ignoring invalid %s %r (expected %d-%d)", key, payload[key], low, high)
                continue
            setattr(self, key, value)
            self._stream = None  # window geometry depends on both
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

    def _realtime_stream(self, converter: VoiceConverter) -> RealtimeVoiceStream:
        if self._stream is None or self._stream_converter is not converter:
            config = self.config

            timing = self.timing
            emitted = (self.chunk_size / self.sample_rate + timing.lookahead_seconds
                       + timing.crossfade_seconds + timing.search_seconds)

            def render(window: np.ndarray) -> tuple[np.ndarray, int]:
                audio = converter.convert(window, self.conversion, pad_seconds=config.STREAM_PAD_SECONDS,
                                          split_long_input=False,
                                          rms_frame_seconds=config.STREAM_RMS_FRAME_SECONDS,
                                          tail_seconds=emitted + config.STREAM_DECODE_MARGIN_SECONDS)
                return audio, converter.output_rate

            self._stream = RealtimeVoiceStream(self.sample_rate, render, self.timing)
            self._stream_converter = converter
        return self._stream

    def _update_silence(self, audio: np.ndarray) -> bool:
        if not self.silence_saver:
            self._silence_seconds = 0.0
            return False
        if chunk_rms_db(audio) <= self.silence_threshold_db:
            self._silence_seconds += len(audio) / self.sample_rate
        else:
            self._silence_seconds = 0.0
        return self._silence_seconds >= self.config.SILENCE_HOLD_SECONDS

    def process(self, audio: np.ndarray) -> FrameResult:
        """Convert one chunk. With ``bypass`` the input is returned untouched (true A/B)."""
        self.processing_error = None
        if self.bypass:
            self.sleeping = False
            return FrameResult(audio, "bypass", 0.0, 0.0)

        started = time.perf_counter()
        processed = self.chain.pre_process(audio, self.effects)
        converter = self.manager.converter
        model_ms = 0.0
        mode = "dsp"

        if converter is None:
            self.sleeping = False
            processed = self.chain.apply_pitch(processed, self.conversion.pitch_shift)
        else:
            mode = "rvc"
            self.eligible_frames += 1
            if self._update_silence(processed):
                if not self.sleeping and self._stream is not None:
                    self._stream.reset()  # resume from fresh context after the pause
                self.sleeping = True
                processed = np.zeros_like(processed)
            else:
                self.sleeping = False
                model_started = time.perf_counter()
                try:
                    processed = self._realtime_stream(converter).process(processed)
                except Exception as exc:
                    return self._failure(exc, audio, model_started)
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

    def status(self) -> dict:
        active = self.manager.active_model_name
        mode = "bypass" if self.bypass else (self.mode if active else "dsp")
        duty = round(self.inference_frames * 100.0 / self.eligible_frames, 1) if self.eligible_frames else 0.0
        message = {
            "type": "status",
            "latency_ms": round(self.latency_ms, 2),
            "model_ms": round(self.model_ms, 2),
            "dsp_ms": round(self.dsp_ms, 2),
            "mode": mode,
            "processing_error": self.processing_error,
            "bypass": self.bypass,
            "active_model": active,
            "silence_saver": self.silence_saver,
            "silence_threshold_db": self.silence_threshold_db,
            "inference_sleeping": bool(active and not self.bypass and self.sleeping),
            "inference_duty_percent": duty if active and not self.bypass else 0.0,
            "effects_active": 0 if self.bypass else EffectsChain.count_active(self.effects, self.formant_shift),
            "algorithmic_delay_ms": round(self.timing.algorithmic_delay() * 1000.0, 1),
        }
        self.eligible_frames = self.inference_frames = 0
        return message


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
    logger.info("Audio stream closed")
