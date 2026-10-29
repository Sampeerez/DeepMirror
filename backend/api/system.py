"""Health and runtime configuration (``/health``, ``/api/config``)."""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, Request

from backend.assets import AssetStore
from backend.config import Settings
from backend.model_manager import ModelManager

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/api/virtual-mic")
async def virtual_mic_status(request: Request) -> dict:
    return await asyncio.to_thread(request.app.state.virtual_mic.status)


@router.post("/api/virtual-mic/install")
async def install_virtual_mic(request: Request) -> dict:
    """Download the official VB-CABLE package and open its installer (Linux: create the devices)."""
    return await asyncio.to_thread(request.app.state.virtual_mic.install)


@router.get("/api/virtual-camera")
async def virtual_camera_status(request: Request) -> dict:
    return await asyncio.to_thread(request.app.state.virtual_camera.status)


@router.get("/api/calls")
async def calls_status(request: Request) -> dict:
    """What calls need: the virtual microphone and the virtual camera."""
    mic, camera = await asyncio.gather(asyncio.to_thread(request.app.state.virtual_mic.status),
                                       asyncio.to_thread(request.app.state.virtual_camera.status))
    return {"microphone": mic, "camera": camera}


@router.post("/api/calls/setup")
async def setup_calls(request: Request) -> dict:
    """Install whatever is missing, the microphone first, then the camera (one prompt each)."""
    mic, camera = request.app.state.virtual_mic, request.app.state.virtual_camera

    def run() -> None:
        if mic.status()["supported"] and not mic.status()["installed"]:
            mic.install(wait=True)
        if camera.status()["supported"] and not camera.status()["installed"]:
            camera.install(wait=True)

    threading.Thread(target=run, name="calls-setup", daemon=True).start()
    return await calls_status(request)


@router.get("/api/config")
async def runtime_config(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    manager: ModelManager = request.app.state.model_manager
    assets: AssetStore = request.app.state.assets
    defaults = manager.default_conversion()
    chunk_size, lookback, lookahead = manager.stream_defaults()
    return {
        "version": request.app.version,
        "sample_rate": settings.SAMPLE_RATE,
        "chunk_size": chunk_size,
        "silence_saver": settings.SILENCE_SAVER,
        "silence_threshold_db": settings.SILENCE_THRESHOLD_DB,
        "stream": {
            "lookback_seconds": lookback,
            "lookahead_seconds": lookahead,
            "crossfade_seconds": settings.STREAM_CROSSFADE_SECONDS,
            "algorithmic_delay_ms": round(
                (lookahead + settings.STREAM_CROSSFADE_SECONDS + settings.STREAM_SEARCH_SECONDS / 2) * 1000, 1),
        },
        "device": manager.compute.describe(),
        "assets": assets.status(),
        "f0_methods": manager.pitch_methods(),
        "default_f0_method": defaults.f0_method,
        "defaults": {
            "index_rate": defaults.index_rate,
            "filter_radius": defaults.filter_radius,
            "rms_mix_rate": defaults.rms_mix_rate,
            "protect": defaults.protect,
            "crepe_hop_length": defaults.crepe_hop_length,
        },
    }
