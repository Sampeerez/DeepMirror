"""Server log for the studio's Logs page: ``GET /api/logs`` and the live ``/ws/logs``."""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from backend.logbook import logbook
from backend.security import is_origin_allowed

router = APIRouter(tags=["logs"])


@router.get("/api/logs")
async def read_logs(after: int = 0, level: str | None = None, limit: int = 1000) -> dict:
    lines = logbook.since(after, level, max(1, min(limit, 3000)))
    return {"lines": [line.to_dict() for line in lines], "last_id": logbook.last_id}


@router.websocket("/ws/logs")
async def live_logs(websocket: WebSocket) -> None:
    """Pushes new log lines as they are written. The client may send ``{"level": "WARNING"}``."""
    settings = websocket.app.state.settings
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host"),
                             settings.CORS_ORIGINS, settings.ALLOW_ANY_ORIGIN):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await websocket.accept()
    state = {"after": logbook.last_id, "level": None}

    async def listen() -> None:
        while True:
            message = await websocket.receive_text()
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                payload = json.loads(message)
                if isinstance(payload.get("level"), str) or payload.get("level") is None:
                    state["level"] = payload.get("level")

    listener = asyncio.create_task(listen())
    try:
        while not listener.done():
            if await asyncio.to_thread(logbook.wait, state["after"], 1.0):
                lines = logbook.since(state["after"], state["level"], 500)
                state["after"] = logbook.last_id
                if lines:
                    await websocket.send_text(json.dumps({"lines": [line.to_dict() for line in lines]}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await listener


@router.get("/api/training/jobs/{job_id}/log")
async def training_log(request: Request, job_id: str, lines: int = 400) -> dict:
    jobs = request.app.state.training
    try:
        return {"lines": await asyncio.to_thread(jobs.log_tail, job_id, max(1, min(lines, 5000)))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
