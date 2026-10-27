"""The speaker's voice profile and per-voice calibration (``/api/profile``)."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from backend.api.convert import decode_audio
from backend.audio.resampling import resample
from backend.model_manager import ModelManager, ModelNotFoundError
from backend.voice_profile import MAX_SECONDS, RATE, ProfileError, analyse, calibration, reference_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/profile", tags=["profile"])

LANGUAGES = {"es", "en"}


def _manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def _read_audio(request: Request, file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > request.app.state.settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The recording is too large.")
    try:
        audio, rate = await asyncio.to_thread(decode_audio, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(audio) > rate * MAX_SECONDS:
        raise HTTPException(status_code=400, detail=f"Use at most {MAX_SECONDS // 60} minutes of audio.")
    return resample(audio, rate, RATE).astype(np.float32)


def _summary(manager: ModelManager) -> dict:
    """The profile and how it calibrates the active voice."""
    profile = manager.profiles.load()
    speaker = manager.profiles.stats()
    active = manager.active_model()
    voice = None
    if active is not None:
        details = active.get("details") or {}
        converter = manager.converter
        target = converter.voice.f0_stats if converter is not None else None
        voice = {"name": active["name"], "uses_pitch": bool(details.get("f0", True)),
                 "register_source": details.get("f0_source"),
                 "calibration": calibration(speaker, target) if speaker is not None else None,
                 "target": target.to_dict() if target is not None else None}
    return {"profile": profile, "voice": voice}


@router.get("/")
async def get_profile(request: Request) -> dict:
    return _summary(_manager(request))


@router.post("/")
async def save_profile(request: Request, file: UploadFile, language: str = Form("es")) -> dict:
    manager = _manager(request)
    audio = await _read_audio(request, file)
    try:
        f0 = await asyncio.to_thread(manager.estimate_f0, audio)
        analysis = analyse(audio, f0)
    except ProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:   # the pitch tracker is still downloading
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    manager.profiles.save(audio, analysis, language if language in LANGUAGES else "es")
    return _summary(manager)


@router.delete("/")
async def delete_profile(request: Request) -> dict:
    manager = _manager(request)
    manager.profiles.delete()
    return _summary(manager)


@router.post("/reference/{name}")
async def save_reference(request: Request, name: str, file: UploadFile) -> dict:
    """A clip of the original voice of a downloaded model: its register."""
    manager = _manager(request)
    audio = await _read_audio(request, file)
    try:
        stats, voiced = reference_stats(await asyncio.to_thread(manager.estimate_f0, audio))
        manager.set_reference(name, stats, voiced)
    except ProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _summary(manager)


@router.delete("/reference/{name}")
async def delete_reference(request: Request, name: str) -> dict:
    manager = _manager(request)
    try:
        manager.set_reference(name, None)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _summary(manager)
