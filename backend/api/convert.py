"""Offline file conversion (``/api/convert``): one upload in, one rendered WAV out.

Files are converted in a single pass with RVC's full padding and quiet-point
splitting (the best quality the model can give), then run through the same
effect chain as the live stream.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from backend.audio.effects import EffectsChain
from backend.audio.resampling import resample
from backend.config import Settings
from backend.inference.auto_pitch import compute_f0_stats, semitones_between
from backend.inference.pipeline import ConversionSettings, VoiceConverter
from backend.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/convert", tags=["convert"])


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 at the file's own rate (WAV, FLAC, OGG, MP3, ...)."""
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the audio file: {exc}") from exc
    return audio.mean(axis=1).astype(np.float32), int(rate)


def encode_wav(audio: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def auto_pitch_shift(converter: VoiceConverter, audio16: np.ndarray, conversion: ConversionSettings) -> float | None:
    """Semitones from the file's median voiced F0 to the voice's, or None if unknown."""
    target = converter.voice.f0_stats
    if target is None:
        return None
    source = compute_f0_stats([converter.estimate_f0(audio16, conversion)])
    if source is None:
        return None
    return float(np.clip(round(semitones_between(source.median_hz, target.median_hz) * 2) / 2, -24.0, 24.0))


def render(audio: np.ndarray, rate: int, manager: ModelManager, use_model: bool, conversion: ConversionSettings,
           formant_shift: float, effects: dict, auto_pitch: bool = False) -> np.ndarray:
    chain = EffectsChain(rate)
    processed = chain.pre_process(audio, effects)
    converter = manager.converter if use_model else None
    if converter is not None:
        if auto_pitch:
            shift = auto_pitch_shift(converter, resample(processed, rate, 16000), conversion)
            if shift is not None:
                conversion = conversion.with_payload({"pitch_shift": shift})
        converted = converter.convert(resample(processed, rate, 16000), conversion)
        processed = resample(converted, converter.output_rate, rate)
        processed = np.pad(processed, (0, max(0, len(audio) - len(processed))))[: len(audio)]
    elif abs(conversion.pitch_shift) > 1e-3:
        processed = chain.apply_pitch(processed, conversion.pitch_shift)
    return chain.post_process(processed, effects, formant_shift)


@router.post("/")
async def convert_file(
    request: Request,
    file: UploadFile,
    pitch_shift: float = Form(0.0),
    # "auto" moves the file's register onto the voice's (voices trained here); "manual" uses pitch_shift.
    pitch_mode: str = Form("manual"),
    formant_shift: float = Form(0.0),
    f0_method: str | None = Form(None),
    effects: str = Form("{}"),
    # Text, not bool: FastAPI reads an omitted bool form field as False (HTML checkbox
    # semantics), and omitting it must mean "use the active model".
    use_model: str | None = Form(None),
    index_rate: float | None = Form(None),
    filter_radius: int | None = Form(None),
    rms_mix_rate: float | None = Form(None),
    protect: float | None = Form(None),
    crepe_hop_length: int | None = Form(None),
    pitch_expressiveness: float | None = Form(None),
    pitch_correction: float | None = Form(None),
) -> Response:
    """Convert an uploaded file and return the rendered WAV (omitted controls use the server defaults)."""
    settings: Settings = request.app.state.settings
    manager: ModelManager = request.app.state.model_manager
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_BYTES // 2**20} MB)")
    try:
        effects_payload = json.loads(effects) if effects else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid effects JSON") from exc
    if not isinstance(effects_payload, dict):
        effects_payload = {}

    try:
        audio, rate = await asyncio.to_thread(decode_audio, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="Audio file contains no samples")
    if len(audio) > settings.MAX_CONVERT_SECONDS * rate:
        raise HTTPException(status_code=413, detail=f"Audio too long (max {settings.MAX_CONVERT_SECONDS // 60} minutes)")

    conversion = manager.default_conversion().with_payload({
        "pitch_shift": pitch_shift, "f0_method": f0_method, "index_rate": index_rate,
        "filter_radius": filter_radius, "rms_mix_rate": rms_mix_rate, "protect": protect,
        "crepe_hop_length": crepe_hop_length, "pitch_expressiveness": pitch_expressiveness,
        "pitch_correction": pitch_correction,
    })
    formant = float(np.clip(formant_shift, -24.0, 24.0))
    apply_model = use_model is None or use_model.strip().lower() not in {"false", "0", "off", "no"}
    try:
        rendered = await asyncio.to_thread(render, audio, rate, manager, apply_model, conversion, formant,
                                           effects_payload, pitch_mode.strip().lower() == "auto")
        wav = await asyncio.to_thread(encode_wav, rendered, rate)
    except Exception as exc:
        logger.exception("File conversion failed")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {type(exc).__name__}") from exc

    out_name = f"{Path(file.filename).stem}_converted.wav"
    # ``inline``: the studio fetches this and offers the download itself;
    # ``attachment`` invites download managers to hijack the response.
    disposition = f"inline; filename=\"converted.wav\"; filename*=UTF-8''{quote(out_name, safe='')}"
    return Response(content=wav, media_type="audio/wav", headers={"Content-Disposition": disposition})
