"""The face clone's own process, and the server's handle on it.

Everything that touches the face models runs here: ONNX Runtime holds
Python's GIL while it loads a model (0.7 s for a swapper), and tracking and
blending run thirty times a second; in their own process none of it can hold
up the live voice. Frames travel as JPEG over a pipe; the GPU is shared with
the voice through the GpuArbiter in shared memory.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import struct
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

JPEG_QUALITY = 88
STATS_SECONDS = 1.0


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    import cv2

    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return data.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    import cv2

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Not an image")
    return frame


# --- Inside the face process --------------------------------------------------


class _VirtualCamera:
    """Feeds the cloned frames to a virtual camera, each shown ``delay`` after its capture (in step with the voice)."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        import pyvirtualcam

        self.camera = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR)
        self.device = self.camera.device
        self.size = (width, height)
        self.delay = 0.0
        self._queue: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="virtual-camera", daemon=True)
        self._thread.start()

    def push(self, due: float, frame: np.ndarray) -> None:
        import cv2

        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size)
        with self._lock:
            self._queue.append((due, frame))
            self._queue = self._queue[-90:]

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                while self._queue and self._queue[0][0] + self.delay <= now:
                    last = self._queue.pop(0)[1]
            if last is not None:
                self.camera.send(last)
            self.camera.sleep_until_next_frame()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.camera.close()


def run(conn: Connection, arbiter_name: str, face_dir: str, cache_path: str, gpu: bool) -> None:
    """Process entry: serve frames and commands until told to stop or the server goes away."""
    import torch  # noqa: F401  (CUDA and cuDNN for ONNX Runtime)

    from backend.compute_arbiter import GpuArbiter
    from backend.video.engine import FaceEngine, FaceModels, VideoSettings
    from backend.video.live import LiveFaceSession

    arbiter = GpuArbiter(arbiter_name)
    models = FaceModels(Path(face_dir), gpu)
    engine = FaceEngine(models, arbiter, VideoSettings())
    live: LiveFaceSession | None = None
    camera: _VirtualCamera | None = None
    offset: float | None = None     # server monotonic - client capture time (s), smallest seen
    last_stats = 0.0
    last_plan = 0.0

    def send(*message) -> None:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            conn.send(message)

    def log(level: int, text: str) -> None:
        send("log", level, text)

    try:
        models.measure(Path(cache_path), arbiter)
        send("ready", {"swap_ms": models.swap_ms, "available": models.available()})
        while True:
            if not conn.poll(0.004):
                message = None
            else:
                message = conn.recv()
            if message is not None:
                kind = message[0]
                if kind == "stop":
                    break
                if kind == "frame":
                    _, t_ms, data = message
                    try:
                        frame = decode_image(data)
                    except ValueError:
                        continue
                    if live is None:
                        live = LiveFaceSession(engine)
                        engine.plan()
                        last_plan = time.monotonic()
                    t = t_ms / 1000.0
                    arrival = time.monotonic()
                    offset = arrival - t if offset is None else min(offset, arrival - t)
                    live.submit(t, frame)
                elif kind == "identity":
                    engine.set_identity(message[1])
                elif kind == "settings":
                    before = engine.settings.swapper
                    engine.settings.apply(message[1])
                    if engine.settings.swapper != before:
                        engine.plan()
                elif kind == "enroll":
                    _, request, images = message
                    try:
                        embedding, notes = models.source_embedding([decode_image(i) for i in images])
                        send("enrolled", request, embedding, notes)
                    except Exception as exc:
                        send("enrolled", request, None, str(exc))
                elif kind == "camera":
                    config = message[1]
                    if camera is not None and (not config.get("on") or camera.size != tuple(config["size"])):
                        camera.close()
                        camera = None
                    if config.get("on") and camera is None:
                        try:
                            width, height = config["size"]
                            camera = _VirtualCamera(int(width), int(height), int(config.get("fps", 30)))
                            log(logging.INFO, f"Virtual camera on: {camera.device}")
                        except Exception as exc:
                            send("camera_error", str(exc))
                    if camera is not None:
                        camera.delay = max(0.0, float(config.get("delay_ms", 0)) / 1000)
                elif kind == "reset":
                    if live is not None:
                        live.close()
                    live, offset = None, None
            if live is not None:
                for t, frame in live.ready():
                    send("frame", t * 1000.0, encode_jpeg(frame))
                    if camera is not None and offset is not None:
                        camera.push(t + offset, frame)
                now = time.monotonic()
                if now - last_plan > 10.0:   # the voice started or stopped: maybe another swapper fits better
                    engine.plan()
                    last_plan = now
                if now - last_stats > STATS_SECONDS:
                    send("stats", {**live.stats(), "camera": camera.device if camera else None})
                    last_stats = now
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if live is not None:
            live.close()
        if camera is not None:
            camera.close()
        arbiter.close()


# --- In the server --------------------------------------------------------------


class FaceWorker:
    """Starts the face process on first use and passes messages both ways."""

    def __init__(self, arbiter_name: str, face_dir: Path, cache_path: Path, gpu: bool) -> None:
        self._args = (arbiter_name, str(face_dir), str(cache_path), gpu)
        self._process: BaseProcess | None = None
        self._conn: Any = None   # the pipe's server end
        self._send_lock = threading.Lock()
        self._listeners: list[Callable[[tuple], None]] = []
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._next_request = 0
        self.info: dict | None = None
        self._ready = threading.Event()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.running:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._ready.clear()
        process = context.Process(target=run, args=(child, *self._args), name="voxclone-face", daemon=True)
        process.start()
        self._process = process
        child.close()
        self._conn = parent
        threading.Thread(target=self._read, name="face-worker-reader", daemon=True).start()
        logger.info("Face clone process started")

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        if self._process is None:
            return
        self.send("stop")
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.kill()
        self._process = None
        logger.info("Face clone process stopped")

    def send(self, *message) -> None:
        if self._conn is None:
            return
        with self._send_lock, contextlib.suppress(BrokenPipeError, OSError):
            self._conn.send(message)

    def subscribe(self, listener: Callable[[tuple], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def enroll(self, images: list[bytes], timeout: float = 60.0) -> tuple[np.ndarray | None, object]:
        """ArcFace identity of the photos (computed in the face process)."""
        self.start()
        self._next_request += 1
        request = self._next_request
        done: tuple[threading.Event, list] = (threading.Event(), [])
        self._pending[request] = done
        self.send("enroll", request, images)
        if not done[0].wait(timeout):
            self._pending.pop(request, None)
            raise TimeoutError("The face process did not answer")
        return done[1][0], done[1][1]

    def _read(self) -> None:
        conn = self._conn
        while conn is not None:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "ready":
                self.info = message[1]
                self._ready.set()
            elif kind == "enrolled":
                waiter = self._pending.pop(message[1], None)
                if waiter is not None:
                    waiter[1].extend(message[2:])
                    waiter[0].set()
            elif kind == "log":
                logger.log(message[1], message[2])
            for listener in list(self._listeners):
                try:
                    listener(message)
                except Exception:
                    logger.exception("Face worker listener failed")
        logger.info("Face clone process ended")


def frame_header(t_ms: float) -> bytes:
    return struct.pack("<d", t_ms)


def split_frame(data: bytes) -> tuple[float, bytes]:
    if len(data) < 9:
        raise ValueError("Frame too short")
    return struct.unpack_from("<d", data)[0], data[8:]
