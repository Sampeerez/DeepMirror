"""Voice model library endpoints (``/api/models``).

Loading a voice runs checkpoint loading and a warm-up render: seconds of
work that would stall every other request and the live audio stream if it
ran on the event loop, so it runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile

from backend.inference.checkpoints import CheckpointError
from backend.model_manager import ModelManager, ModelNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


def _manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("/")
async def list_models(request: Request) -> list[dict]:
    return await asyncio.to_thread(_manager(request).list_models)


@router.post("/upload")
async def upload_model(request: Request, file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    try:
        return await _manager(request).upload(file.filename, file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{name}")
async def delete_model(request: Request, name: str) -> dict:
    try:
        await asyncio.to_thread(_manager(request).delete, name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted", "name": name}


@router.post("/{name}/activate")
async def activate_model(request: Request, name: str) -> dict:
    try:
        return await asyncio.to_thread(_manager(request).activate, name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (CheckpointError, RuntimeError) as exc:
        logger.error("Could not activate %s: %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/deactivate")
async def deactivate_model(request: Request) -> dict:
    await asyncio.to_thread(_manager(request).deactivate)
    return {"status": "deactivated"}


@router.get("/active")
async def active_model(request: Request) -> dict:
    return _manager(request).active_model() or {"active": False}
