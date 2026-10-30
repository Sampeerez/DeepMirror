"""The face clone: models, faces, settings (``/api/video``) and the live camera (``/ws/video``)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.security import is_origin_allowed
from backend.video.library import MAX_PHOTOS, FaceNotFoundError
from backend.video.service import FaceService
from backend.video.worker import frame_header, split_frame

logger = logging.getLogger(__name__)
router = APIRouter(tags=["video"])

MAX_PHOTO_BYTES = 15 * 2**20
MAX_FRAME_BYTES = 4 * 2**20
CAMERA_SIZES = {(1280, 720), (960, 540), (640, 480), (1920, 1080)}


def _service(request: Request | WebSocket) -> FaceService:
    return request.app.state.faces


@router.get("/api/video/")
async def video_status(request: Request) -> dict:
    return _service(request).status()


@router.post("/api/video/prepare")
async def prepare(request: Request) -> dict:
    """Download the face models (once, about 1.8 GB)."""
    service = _service(request)
    service.prepare_models()
    return service.status()


@router.patch("/api/video/settings")
async def update_settings(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Send a JSON object.")
    return {"settings": _service(request).update_settings(payload)}


@router.post("/api/video/faces")
async def add_face(request: Request, files: list[UploadFile], name: str = Form("")) -> dict:
    service = _service(request)
    if not service.models_ready:
        raise HTTPException(status_code=409, detail="Download the face models first.")
    if not files or len(files) > MAX_PHOTOS:
        raise HTTPException(status_code=400, detail=f"Send 1 to {MAX_PHOTOS} photos.")
    photos = []
    for file in files:
        data = await file.read()
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(status_code=413, detail=f"{file.filename}: photo too large (max 15 MB).")
        photos.append(data)
    try:
        face = await asyncio.to_thread(service.add_face, name or (files[0].filename or "Face").rsplit(".", 1)[0],
                                       photos)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if service.library.active() is None:   # the first face is worn right away
        service.activate(face["id"])
    return {"face": face, **service.status()}


@router.get("/api/video/faces/{face_id}/thumbnail")
async def thumbnail(request: Request, face_id: str) -> FileResponse:
    try:
        return FileResponse(_service(request).library.thumbnail(face_id), media_type="image/jpeg")
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Face not found") from exc


@router.post("/api/video/faces/{face_id}/activate")
async def activate(request: Request, face_id: str) -> dict:
    service = _service(request)
    try:
        await asyncio.to_thread(service.activate, None if face_id == "none" else face_id)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Face not found") from exc
    return service.status()


@router.post("/api/video/faces/{face_id}/head3d")
async def build_head(request: Request, face_id: str) -> dict:
    """Build (or rebuild) the face's 3D head in the background."""
    service = _service(request)
    try:
        service.build_head(face_id)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Face not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return service.status()


@router.delete("/api/video/faces/{face_id}")
async def delete_face(request: Request, face_id: str) -> dict:
    service = _service(request)
    try:
        await asyncio.to_thread(service.delete, face_id)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Face not found") from exc
    return service.status()


def _camera_config(payload: dict) -> dict | None:
    """``{"on", "size", "fps", "delay_ms"}`` from a client message, or None if malformed."""
    size = payload.get("size")
    if not isinstance(size, list) or len(size) != 2 or tuple(size) not in CAMERA_SIZES:
        size = [1280, 720]
    delay = payload.get("delay_ms", 0)
    if not isinstance(delay, int | float) or isinstance(delay, bool):
        return None
    return {"on": bool(payload.get("on")), "size": size, "fps": 30, "delay_ms": float(min(max(delay, 0), 3000))}


@router.websocket("/ws/video")
async def video_websocket(websocket: WebSocket) -> None:
    """Camera frames in (8-byte capture time in ms, then JPEG), cloned frames out in the same format."""
    settings = websocket.app.state.settings
    origin = websocket.headers.get("origin")
    if not is_origin_allowed(origin, websocket.headers.get("host"), settings.CORS_ORIGINS,
                             settings.ALLOW_ANY_ORIGIN):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await websocket.accept()
    service = _service(websocket)
    try:
        worker = await asyncio.to_thread(service.worker)
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue(maxsize=8)

    def on_message(message: tuple) -> None:
        def put() -> None:
            if outbox.full():   # the page fell behind: drop the oldest, fresh frames matter
                with contextlib.suppress(asyncio.QueueEmpty):
                    outbox.get_nowait()
            outbox.put_nowait(message)

        if message[0] in ("frame", "stats", "camera_error", "ready"):
            loop.call_soon_threadsafe(put)

    unsubscribe = worker.subscribe(on_message)
    worker.send("reset")
    if worker.info is not None:
        await websocket.send_json({"type": "ready", **worker.info})

    async def pump() -> None:
        while True:
            message = await outbox.get()
            kind = message[0]
            if kind == "frame":
                await websocket.send_bytes(frame_header(message[1]) + message[2])
            elif kind == "stats":
                await websocket.send_json({"type": "stats", **message[1]})
            elif kind == "camera_error":
                await websocket.send_json({"type": "camera_error", "message": message[1]})
            elif kind == "ready":
                await websocket.send_json({"type": "ready", **message[1]})

    sender = asyncio.create_task(pump())
    # Frames go to the face process from a thread of their own: a pipe write that waits (the
    # face process busy for a moment) must never hold up the event loop, which also serves the
    # live voice. Only the newest few wait; older ones are dropped.
    frames: queue.Queue = queue.Queue(maxsize=3)

    def forward() -> None:
        while True:
            item = frames.get()
            if item is None:
                return
            worker.send("frame", *item)

    forwarder = threading.Thread(target=forward, name="video-forward", daemon=True)
    forwarder.start()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                if len(data) > MAX_FRAME_BYTES:
                    continue
                try:
                    t_ms, jpeg = split_frame(data)
                except ValueError:
                    continue
                if frames.full():
                    with contextlib.suppress(queue.Empty):
                        frames.get_nowait()
                frames.put_nowait((t_ms, jpeg))
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("camera"), dict):
                    config = _camera_config(payload["camera"])
                    if config is not None:
                        worker.send("camera", config)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(queue.Full):
            frames.put_nowait(None)
        sender.cancel()
        unsubscribe()
        worker.send("camera", {"on": False, "size": [1280, 720]})
        worker.send("reset")
