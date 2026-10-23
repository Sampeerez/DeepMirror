"""Preset endpoints (``/api/presets``)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.presets import PresetStore

router = APIRouter(prefix="/api/presets", tags=["presets"])


class PresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    emoji: str = Field(default="⭐", max_length=4)
    settings: dict


def _store(request: Request) -> PresetStore:
    return request.app.state.preset_store


@router.get("/")
async def list_presets(request: Request) -> dict:
    store = _store(request)
    return {"builtin": store.list_builtin(), "user": store.list_user()}


@router.post("/")
async def save_preset(request: Request, payload: PresetPayload) -> dict:
    try:
        return _store(request).save(payload.name, payload.settings, payload.emoji)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{preset_id}")
async def delete_preset(request: Request, preset_id: str) -> dict:
    try:
        _store(request).delete(preset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_id}") from exc
    return {"status": "deleted", "id": preset_id}
