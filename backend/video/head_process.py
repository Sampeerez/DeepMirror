"""The whole head in a process of its own.

A head render is hundreds of small PyTorch operations, and between them Python holds the GIL.
In the face process that starved the threads that receive, track and compose camera frames
(the video fell from 30 to ~11 fps whenever a head was rendering). Here the head has its own
interpreter: the face process sends a frame and the face's five points about once a second,
and gets the finished head layer back. The GPU is still shared through the same arbiter (shared
memory), so the voice keeps its priority.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run(conn: Connection, arbiter_name: str, face_dir: str, gpu: bool) -> None:
    """Process entry: render heads on request until the pipe closes."""
    import torch

    torch.set_num_threads(2)
    from backend.compute_arbiter import GpuArbiter
    from backend.video.engine import FaceEngine, FaceModels, Tracked, VideoSettings

    arbiter = GpuArbiter(arbiter_name)
    models = FaceModels(Path(face_dir), gpu)
    engine = FaceEngine(models, arbiter, VideoSettings(mode="head"))
    try:
        # The networks load on the first render (only whole-head file conversions need them).
        conn.send(("ready",))
        while True:
            message = conn.recv()
            kind = message[0]
            if kind == "stop":
                break
            if kind == "reset":
                engine.reset_head()
            elif kind == "portrait":
                engine.set_portrait(message[2])
            elif kind == "render":
                _, request, t, frame, points, wait = message
                layer = None
                try:
                    layer = engine.head_layer(frame, Tracked(None, points), t, wait=wait)
                except Exception:
                    logger.exception("Head render failed")
                conn.send(("layer", request, None if layer is None else
                           (layer.t, layer.image, layer.alpha, layer.face, layer.pose)))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        arbiter.close()


class RemoteHead:
    """The face process's side: ``render`` asks the head process and waits for the layer."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        self._portrait: int | None = None
        self._request = 0
        self.ready = False

    def _send(self, *message) -> None:
        self.conn.send(message)

    def reset(self) -> None:
        with self._lock, contextlib.suppress(OSError, EOFError, BrokenPipeError):
            self._send("reset")

    def render(self, frame: np.ndarray, points: np.ndarray, t: float, portrait: np.ndarray, wait: float):
        """The head layer for ``frame`` (a HeadLayer), or None when the GPU stayed busy."""
        from backend.video.engine import HeadLayer

        with self._lock:
            try:
                if not self.ready:
                    if not self.conn.poll(0):
                        return None
                    self.ready = self.conn.recv()[0] == "ready"
                if self._portrait != id(portrait):
                    self._send("portrait", id(portrait), portrait)
                    self._portrait = id(portrait)
                self._request += 1
                self._send("render", self._request, t, frame, points, wait)
                # The render spans several GPU gaps; ``wait`` bounds each wait without progress.
                while self.conn.poll(wait + 30.0):
                    kind, request, layer = self.conn.recv()
                    if kind == "layer" and request == self._request:
                        return None if layer is None else HeadLayer(*layer)
            except (OSError, EOFError, BrokenPipeError):
                logger.warning("The head process is gone")
            return None
