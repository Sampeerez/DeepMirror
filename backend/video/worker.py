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
import queue
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import replace
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


def _read_image(path: str | None) -> np.ndarray | None:
    """A photo from the face library (None when there is none or it cannot be read)."""
    if not path:
        return None
    import cv2

    data = np.fromfile(path, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return image


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


def run(conn: Connection, arbiter_name: str, face_dir: str, cache_path: str, gpu: bool,
        head_conn: Connection | None = None, start_mode: str = "hair") -> None:
    """Process entry: serve frames and commands until told to stop or the server goes away."""
    import cv2
    import torch

    from backend.process_guard import exit_with_parent, lower_priority
    from backend.video.runtime import cpu_threads

    exit_with_parent()
    lower_priority()

    # A share of the cores (a quarter, 2 to 4) for one face at camera rate; the rest stay with the voice.
    cv2.setNumThreads(max(2, cpu_threads()))
    torch.set_num_threads(1)

    from backend.compute_arbiter import GpuArbiter
    from backend.video.engine import FaceEngine, FaceModels, VideoSettings
    from backend.video.live import LiveFaceSession

    arbiter = GpuArbiter(arbiter_name)
    models = FaceModels(Path(face_dir), gpu)
    engine = FaceEngine(models, arbiter, VideoSettings())
    if head_conn is not None:
        from backend.video.head_process import RemoteHead

        engine.remote = RemoteHead(head_conn)
    from backend.video.avatar import HEAD_FILE, HeadDriver, LamInstall, LiveAvatarSession

    install = LamInstall.default()
    live: LiveFaceSession | LiveAvatarSession | None = None
    # The 3D head (Clone: 3D head): wanted, built for this face, and its driver process.
    want_3d = start_mode == "3d"
    head_3d: Path | None = None
    driver: HeadDriver | None = None
    driver_started = [-1e9]
    camera: _VirtualCamera | None = None
    offset: float | None = None     # server monotonic - client capture time (s), smallest seen
    last_stats = 0.0
    last_plan = 0.0
    send_lock = threading.Lock()
    stopping = threading.Event()
    # Frames in: decoded and tracked on their own thread, so reading messages, tracking and
    # blending/encoding overlap (OpenCV and ONNX Runtime release the GIL). Only the newest few
    # wait: if tracking falls behind, the oldest frame is dropped, never the newest.
    incoming: queue.Queue = queue.Queue(maxsize=3)
    dropped = [0]   # frames tracking could not keep up with

    def send(*message) -> None:
        with send_lock, contextlib.suppress(BrokenPipeError, EOFError, OSError):
            conn.send(message)

    def track_frames() -> None:
        while not stopping.is_set():
            try:
                session, t, data = incoming.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                session.submit(t, decode_image(data))
            except ValueError:
                continue
            except Exception:
                logger.exception("Face tracking failed")

    planning = threading.Event()

    def replan() -> None:
        try:
            engine.plan(running=True)
        except Exception:
            logger.exception("Choosing the face swapper failed")
        finally:
            planning.clear()

    def emit_frames() -> None:
        nonlocal last_stats, last_plan
        while not stopping.is_set():
            session = live
            if session is None:
                time.sleep(0.01)
                continue
            frames = session.ready()
            for t, frame in frames:
                send("frame", t * 1000.0, encode_jpeg(frame))
                if camera is not None and offset is not None:
                    camera.push(t + offset, frame)
            now = time.monotonic()
            if now - last_plan > 10.0 and not planning.is_set() and not isinstance(session, LiveAvatarSession):
                # The voice started or stopped: maybe another swapper fits better. On its own
                # thread: loading and warming a swapper waits for GPU gaps, and the frames must
                # keep going out meanwhile.
                planning.set()
                threading.Thread(target=replan, name="face-plan", daemon=True).start()
                last_plan = now
            if now - last_stats > STATS_SECONDS:
                send("stats", {**session.stats(), "camera": camera.device if camera else None, "dropped": dropped[0]})
                last_stats = now
            if not frames:
                time.sleep(0.003)

    threading.Thread(target=track_frames, name="face-track", daemon=True).start()
    threading.Thread(target=emit_frames, name="face-emit", daemon=True).start()

    def log(level: int, text: str) -> None:
        send("log", level, text)

    def avatar_ready() -> bool:
        """Whether frames go through the 3D head now (else the face swap carries on)."""
        nonlocal driver
        if not want_3d or head_3d is None or not install.available():
            return False
        if driver is None or not driver.running:
            if driver is not None and time.monotonic() - driver_started[0] < 30.0:
                return False   # it just ended: not again at once
            driver_started[0] = time.monotonic()
            driver = HeadDriver(install, arbiter_name, lambda seq, layer: None,
                                log_path=Path(cache_path).parent.parent / "logs" / "head3d-drive.log")
            log(logging.INFO, "3D head process started")
        return driver.ready and driver.use(head_3d)

    def session_for_frame():
        """The live session for the next frame: with a built 3D head, the 3D head (the camera as it is
        for the few seconds its process takes to start: no face swap loaded meanwhile); else the swap."""
        nonlocal live
        use_avatar = want_3d and head_3d is not None and install.available()
        avatar_ready()   # starts the head process and loads the head as needed
        if live is not None and isinstance(live, LiveAvatarSession) != use_avatar:
            live.close()
            live = None
        if live is None:
            live = LiveAvatarSession(driver, engine) if use_avatar and driver is not None else LiveFaceSession(engine)
        return live

    class _Forward(logging.Handler):
        """This process's own log lines go to the server's log."""

        def emit(self, record: logging.LogRecord) -> None:
            with contextlib.suppress(Exception):
                send("log", record.levelno, self.format(record))

    forward = _Forward(logging.INFO)
    forward.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger("backend").addHandler(forward)
    logging.getLogger("backend").setLevel(logging.INFO)

    def _convert(job: str, source: str, target: str, identity: np.ndarray | None = None,
                 portrait: str | None = None, clone: str | None = None) -> None:
        """A whole video file, with its own engine (its own tracking) next to the live one."""
        from backend.video.offline import swap_video

        # Same swapper as the live engine (they share the loaded models); every frame swapped, so no real mouth.
        swapper = engine._swapper_id or engine.settings.swapper
        mode = clone if clone in ("hair", "face", "head") else engine.settings.mode
        offline = FaceEngine(models, arbiter, replace(engine.settings, mouth="off", swapper=swapper, mode=mode))
        # The face chosen for this file, else the one worn live.
        offline.set_identity(identity if identity is not None else engine.identity)
        offline.set_portrait(_read_image(portrait) if portrait else engine.portrait)
        offline.remote = engine.remote
        offline.hair_photos = ([str(path) for path in sorted(Path(portrait).parent.glob("photo_*.jpg"))]
                               if portrait else engine.hair_photos)
        try:
            frames = swap_video(offline, Path(source), Path(target), lambda done: send("convert_progress", job, done))
            send("convert_done", job, None, frames)
        except Exception as exc:
            logger.exception("Video conversion failed")
            send("convert_done", job, f"{type(exc).__name__}: {exc}", 0)

    try:
        models.measure(Path(cache_path), arbiter)
        # Loaded now, not on the first frame: a swapper takes a second or two to load and warm up.
        if models.available() and not (want_3d and install.available()):   # the 3D head needs no swapper
            engine.plan()
            last_plan = time.monotonic()
        send("ready", {"swap_ms": models.swap_ms, "available": models.available()})
        while True:
            if not conn.poll(0.05):
                continue
            message = conn.recv()
            if message is not None:
                kind = message[0]
                if kind == "stop":
                    break
                if kind == "frame":
                    _, t_ms, data = message
                    session = session_for_frame()
                    t = t_ms / 1000.0
                    arrival = time.monotonic()
                    offset = arrival - t if offset is None else min(offset, arrival - t)
                    if incoming.full():
                        with contextlib.suppress(queue.Empty):
                            incoming.get_nowait()
                            dropped[0] += 1
                    incoming.put_nowait((session, t, data))
                elif kind == "identity":
                    engine.set_identity(message[1])
                    engine.set_portrait(_read_image(message[2]) if len(message) > 2 else None)
                    engine.hair_photos = list(message[3]) if len(message) > 3 and message[3] else []
                    head = Path(message[2]).parent / HEAD_FILE if len(message) > 2 and message[2] else None
                    head_3d = head if head is not None and head.is_file() else None
                    avatar_ready()   # with the 3D head chosen: its process starts and loads it now
                elif kind == "settings":
                    before = engine.settings.swapper
                    want_3d = message[1].get("mode") == "3d"
                    avatar_ready()
                    # Until the 3D head is ready (or for a face without one): the face and the photo's hair.
                    engine.settings.apply({**message[1], "mode": "hair"} if want_3d else message[1])
                    if engine.settings.swapper != before and not want_3d:
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
                            log(logging.WARNING, f"Virtual camera unavailable: {exc}")
                            missing = "not installed" in str(exc).lower() or "no camera" in str(exc).lower()
                            send("camera_error", "Virtual camera not installed: Settings › Calls › Set up"
                                 if missing else f"Virtual camera: {exc}")
                    if camera is not None:
                        camera.delay = max(0.0, float(config.get("delay_ms", 0)) / 1000)
                elif kind == "convert":
                    _, job, source, target, *rest = message
                    threading.Thread(target=_convert, args=(job, source, target, *rest[:3]),
                                     name="face-convert",
                                     daemon=True).start()
                elif kind == "reset":
                    previous, live, offset = live, None, None
                    if previous is not None:
                        previous.close()
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        stopping.set()
        if live is not None:
            live.close()
        if camera is not None:
            camera.close()
        if driver is not None:
            driver.close()
        arbiter.close()


# --- In the server --------------------------------------------------------------


class FaceWorker:
    """Starts the face process on first use and passes messages both ways."""

    def __init__(self, arbiter_name: str, face_dir: Path, cache_path: Path, gpu: bool) -> None:
        self._args = (arbiter_name, str(face_dir), str(cache_path), gpu)
        self.start_mode = "hair"   # the clone mode at start (the 3D head loads no face swapper)
        self._head_process: BaseProcess | None = None
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
        # The whole head renders in its own process, wired straight to the face process.
        from backend.video import head_process

        head_face, head_side = context.Pipe()
        arbiter_name, face_dir, _cache, gpu = self._args
        head = context.Process(target=head_process.run, args=(head_side, arbiter_name, face_dir, gpu),
                               name="deepmirror-head", daemon=True)
        head.start()
        self._head_process = head
        head_side.close()
        process = context.Process(target=run, args=(child, *self._args, head_face, self.start_mode),
                                  name="deepmirror-face",
                                  daemon=True)
        process.start()
        self._process = process
        child.close()
        head_face.close()
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
        head, self._head_process = self._head_process, None
        if head is not None:   # it ends by itself once the face process's pipe closes
            head.join(timeout=3)
            if head.is_alive():
                head.kill()
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

    def convert_video(self, source: Path, target: Path, on_progress: Callable[[float], None],
                      identity: np.ndarray | None = None, portrait: Path | None = None, clone: str | None = None,
                      timeout: float = 6 * 3600) -> int:
        """Swap ``identity`` (default: the face worn live) into every frame of ``source``; returns the frame count."""
        self.start()
        if not self.wait_ready(300):
            raise RuntimeError("The face process did not start")
        self._next_request += 1
        job = f"convert-{self._next_request}"
        done = threading.Event()
        result: list = []

        def listen(message: tuple) -> None:
            if message[0] == "convert_progress" and message[1] == job:
                on_progress(message[2])
            elif message[0] == "convert_done" and message[1] == job:
                result.extend(message[2:])
                done.set()

        unsubscribe = self.subscribe(listen)
        try:
            self.send("convert", job, str(source), str(target), identity, str(portrait) if portrait else None, clone)
            if not done.wait(timeout):
                raise TimeoutError("The face process did not finish the video")
        finally:
            unsubscribe()
        if result[0]:
            raise RuntimeError(result[0])
        return int(result[1])

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
