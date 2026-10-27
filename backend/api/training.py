"""Voice training endpoints (``/api/training``)."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from backend.model_manager import ModelManager
from backend.training.hparams import (
    BASES,
    CONFIGS,
    DEFAULT_BASE,
    DEFAULT_EPOCHS,
    MAX_AUDIO_SECONDS,
    MAX_EPOCHS,
    MIN_AUDIO_SECONDS,
    RECOMMENDED_AUDIO_SECONDS,
)
from backend.training.jobs import TrainingError, TrainingJobs
from backend.training.plan import DEFAULT_QUALITY, QUALITY

router = APIRouter(prefix="/api/training", tags=["training"])


def _jobs(request: Request) -> TrainingJobs:
    return request.app.state.training


@router.get("/info")
async def training_info(request: Request) -> dict:
    manager: ModelManager = request.app.state.model_manager
    return {
        "sample_rates": sorted(CONFIGS),
        "default_sample_rate": 40000,
        "default_epochs": DEFAULT_EPOCHS,
        "bases": sorted(BASES),
        "default_base": DEFAULT_BASE,
        "qualities": list(QUALITY),
        "default_quality": DEFAULT_QUALITY,
        "max_epochs": MAX_EPOCHS,
        "min_audio_seconds": MIN_AUDIO_SECONDS,
        "recommended_audio_seconds": list(RECOMMENDED_AUDIO_SECONDS),
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "device": manager.compute.describe(),
    }


@router.get("/jobs")
async def list_jobs(request: Request) -> list[dict]:
    return await asyncio.to_thread(_jobs(request).list)


@router.post("/jobs")
async def create_job(
    request: Request,
    name: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    sample_rate: Annotated[int, Form()] = 40000,
    # Omitted (or 0): planned automatically from the audio length and the GPU.
    epochs: Annotated[int | None, Form()] = None,
    batch_size: Annotated[int | None, Form()] = None,
    save_every: Annotated[int, Form()] = 25,
    overwrite: Annotated[str | None, Form()] = None,
    base: Annotated[str | None, Form()] = None,
    quality: Annotated[str | None, Form()] = None,
    isolate: Annotated[str | None, Form()] = None,
) -> dict:
    if batch_size is not None and not 1 <= batch_size <= 32:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 32.")
    try:
        return await _jobs(request).create(name, files, sample_rate, epochs or None, batch_size,
                                           max(1, min(save_every, MAX_EPOCHS)),
                                           (overwrite or "").lower() in {"true", "1", "on", "yes"},
                                           base if base in BASES else DEFAULT_BASE,
                                           quality if quality in QUALITY else DEFAULT_QUALITY,
                                           (isolate or "true").lower() not in {"false", "0", "off", "no"})
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ResumePayload(BaseModel):
    epochs: int | None = Field(default=None, ge=1, le=MAX_EPOCHS)


def _call(action, *args):
    try:
        return action(*args)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    return _call(_jobs(request).describe, job_id)


@router.post("/jobs/{job_id}/stop")
async def stop_job(request: Request, job_id: str) -> dict:
    return _call(_jobs(request).stop, job_id)


@router.post("/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str, payload: ResumePayload | None = None) -> dict:
    return _call(_jobs(request).resume, job_id, payload.epochs if payload else None)


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    await asyncio.to_thread(_call, _jobs(request).delete, job_id)
    return {"status": "deleted", "id": job_id}
