"""Close the studio when its page is closed.

The studio runs without a console window, so closing the browser tab is the
only way a user ends it. Every open page holds a ``/ws/presence`` socket; when
none is left for ``grace`` seconds (long enough for a reload) the server shuts
down cleanly. If no page ever connects (the browser failed to open), it shuts
down after ``first_contact`` seconds instead of lingering invisibly.

A training run is a separate process and is left to finish on its own.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.security import is_origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

GRACE_SECONDS = 15.0
FIRST_CONTACT_SECONDS = 180.0


class Lifecycle:
    def __init__(self, enabled: bool, on_exit: Callable[[], None] | None = None, *,
                 grace: float = GRACE_SECONDS, first_contact: float = FIRST_CONTACT_SECONDS) -> None:
        self.enabled = enabled and on_exit is not None
        self.on_exit = on_exit
        self.grace = grace
        self.first_contact = first_contact
        self.pages = 0
        self.ever_connected = False
        self.closed_by_idle = False
        self._started = time.monotonic()
        self._empty_since = time.monotonic()
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self.pages += 1
            self.ever_connected = True

    def disconnect(self) -> None:
        with self._lock:
            self.pages = max(0, self.pages - 1)
            if self.pages == 0:
                self._empty_since = time.monotonic()

    def should_exit(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self.enabled or self.pages > 0:
                return False
            if self.ever_connected:
                return now - self._empty_since >= self.grace
            return now - self._started >= self.first_contact

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.should_exit():
                self.closed_by_idle = True
                logger.info("The studio page was closed; shutting down")
                assert self.on_exit is not None
                self.on_exit()
                return


@router.websocket("/ws/presence")
async def presence(websocket: WebSocket) -> None:
    """Held open by every studio page; closing the last one ends the server (see ``Lifecycle``)."""
    settings = websocket.app.state.settings
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host"),
                             settings.CORS_ORIGINS, settings.ALLOW_ANY_ORIGIN):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    lifecycle: Lifecycle = websocket.app.state.lifecycle
    await websocket.accept()
    lifecycle.connect()
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        lifecycle.disconnect()
