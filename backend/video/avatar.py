"""The 3D head: a face's head built once from its photos (LAM), driven live by the camera.

LAM needs its own Python environment (PyTorch 2.3 + CUDA 11.8 and four compiled CUDA libraries,
see ``scripts/setup_head3d.py``) in ``models/lam``. The studio talks to it through two scripts that
run there: ``head3d/build.py`` (once per face, in the background) and ``head3d/drive.py`` (live:
camera frame in, head layer out). Without that install the 3D head is simply not offered.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

from backend.config import ROOT

logger = logging.getLogger(__name__)

HEAD_FILE = "head3d.pt"
PREVIEW_FILE = "head3d.jpg"
ERROR_FILE = "head3d.error"
SCRIPTS = Path(__file__).resolve().parent / "head3d"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True, slots=True)
class LamInstall:
    root: Path

    @classmethod
    def default(cls) -> LamInstall:
        return cls(Path(os.environ.get("OVC_LAM_DIR") or ROOT / "models" / "lam"))

    @property
    def python(self) -> Path:
        return self.root / "env" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    @property
    def source(self) -> Path:
        return self.root / "src"

    @property
    def landmarker(self) -> Path:
        return self.root / "mp" / "face_landmarker.task"

    def available(self) -> bool:
        return (self.python.is_file() and self.landmarker.is_file()
                and (self.source / "model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors").is_file())

    def build_environment(self) -> dict[str, str]:
        """The environment the build runs in: LAM's face tracker compiles a CUDA plugin on first use
        (nvdiffrast), so on Windows it needs the same compiler set-up as the install."""
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        scripts = str(self.python.parent)
        if sys.platform == "win32":
            vcvars = _vcvars()
            if vcvars is not None:
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    out = subprocess.run(f'cmd /s /c ""{vcvars}" >nul 2>nul && set"', capture_output=True,
                                         text=True, timeout=60, creationflags=NO_WINDOW).stdout
                    for line in out.splitlines():
                        name, equals, value = line.partition("=")
                        if equals:
                            env[name] = value
            cuda = _cuda_home()
            if cuda is not None:
                env.update(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), DISTUTILS_USE_SDK="1")
                scripts = f"{cuda / 'bin'}{os.pathsep}{scripts}"
        env["PATH"] = f"{scripts}{os.pathsep}{env.get('PATH', '')}"
        return env


def _vcvars() -> Path | None:
    """Visual Studio 2019's x64 build environment (the newest compiler CUDA 11.8 accepts)."""
    base = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "2019"
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        candidate = base / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.is_file():
            return candidate
    return None


def _cuda_home() -> Path | None:
    if os.environ.get("CUDA_PATH_V11_8"):
        return Path(os.environ["CUDA_PATH_V11_8"])
    candidate = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.8"
    return candidate if candidate.is_dir() else None


# --- Building (in the server) ----------------------------------------------------


class HeadBuilder:
    """Builds faces' 3D heads one at a time, in the background, never while the voice is live
    (a build takes the whole GPU for ~20 s)."""

    def __init__(self, faces_root: Path, parser: Callable[[], Path], voice_live: Callable[[], bool],
                 install: LamInstall | None = None, log_dir: Path | None = None,
                 on_built: Callable[[str], None] | None = None) -> None:
        self.faces_root = Path(faces_root)
        self.install = install or LamInstall.default()
        self._parser = parser
        self._voice_live = voice_live
        self._log_dir = log_dir
        self._on_built = on_built
        self._queue: queue.Queue[str] = queue.Queue()
        self._waiting: set[str] = set()
        self._progress: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._closed = False

    @property
    def available(self) -> bool:
        return self.install.available()

    def status(self, face_id: str) -> dict:
        """``{"state": ready | building | queued | failed | missing | unavailable, ...}``"""
        folder = self.faces_root / face_id
        with self._lock:
            if face_id in self._progress:
                done, stage = self._progress[face_id]
                return {"state": "building", "progress": done, "stage": stage}
            if face_id in self._waiting:
                return {"state": "queued"}
        if (folder / HEAD_FILE).is_file():
            return {"state": "ready"}
        if not self.available:
            return {"state": "unavailable"}
        if (folder / ERROR_FILE).is_file():
            return {"state": "failed", "error": (folder / ERROR_FILE).read_text(encoding="utf-8", errors="replace")[-300:]}
        return {"state": "missing"}

    def request(self, face_id: str) -> None:
        if not self.available:
            raise RuntimeError("The 3D head is not installed (scripts/setup_head3d.py)")
        with self._lock:
            if face_id in self._waiting or face_id in self._progress:
                return
            self._waiting.add(face_id)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="head3d-build", daemon=True)
                self._thread.start()
        self._queue.put(face_id)

    def head_path(self, face_id: str) -> Path | None:
        path = self.faces_root / face_id / HEAD_FILE
        return path if path.is_file() else None

    def close(self) -> None:
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()

    def _run(self) -> None:
        while not self._closed:
            try:
                face_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                with self._lock:
                    if not self._waiting:
                        return
                continue
            while self._voice_live() and not self._closed:
                time.sleep(1.0)
            with self._lock:
                self._waiting.discard(face_id)
                self._progress[face_id] = (0.0, "starting")
            try:
                self._build(face_id)
                if self._on_built is not None:
                    self._on_built(face_id)
            except Exception as exc:
                logger.exception("Building the 3D head failed")
                with contextlib.suppress(OSError):
                    (self.faces_root / face_id / ERROR_FILE).write_text(str(exc), encoding="utf-8")
            finally:
                with self._lock:
                    self._progress.pop(face_id, None)

    def _build(self, face_id: str) -> None:
        folder = self.faces_root / face_id
        if not folder.is_dir():
            return
        (folder / ERROR_FILE).unlink(missing_ok=True)
        started = time.monotonic()
        tail: deque[str] = deque(maxlen=40)
        log = None
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log = open(self._log_dir / "head3d-build.log", "w", encoding="utf-8")  # noqa: SIM115
        try:
            self._process = subprocess.Popen(
                [str(self.install.python), "-u", str(SCRIPTS / "build.py"), str(folder.resolve()), str(self._parser().resolve()),
                 str(self.install.landmarker.resolve())],
                cwd=self.install.source, env=self.install.build_environment(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if log is not None:
                    log.write(line)
                tail.append(line.rstrip())
                if line.startswith("progress "):
                    _, done, stage = [*line.rstrip().split(" ", 2), ""][:3]
                    with self._lock:
                        self._progress[face_id] = (float(done), stage)
            code = self._process.wait()
        finally:
            self._process = None
            if log is not None:
                log.close()
        if code != 0 or not (folder / HEAD_FILE).is_file():
            reason = next((line for line in reversed(tail) if line and not line.startswith(" ")), f"exit {code}")
            if "out of memory" in "\n".join(tail).lower():
                reason = "Not enough GPU memory: stop the camera and try again"
            raise RuntimeError(reason)
        logger.info("3D head of %s built in %.0f s", face_id, time.monotonic() - started)


# --- Live (in the face process) --------------------------------------------------


@dataclass(slots=True)
class Layer:
    x: int
    y: int
    image: np.ndarray        # h x w x 4, BGRA, straight alpha


class HeadDriver:
    """The drive process (LAM's environment): frames in, head layers out, asynchronously."""

    def __init__(self, install: LamInstall, arbiter_name: str, on_layer: Callable[[int, Layer | None], None],
                 log_path: Path | None = None) -> None:
        self.install = install
        self.on_layer = on_layer
        self.loaded: str | None = None     # the head file loaded in the process
        self.requested: str | None = None
        self.error: str | None = None
        self.timings: dict = {}
        self._ready = threading.Event()
        self._send_lock = threading.Lock()
        self._conn = None
        key = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=key)
        host, port = self._listener.address
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env["DEEPMIRROR_HEAD3D"] = f"{host}:{port}:{key.hex()}"
        from backend import compute_arbiter

        log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 (the child has its own handle once started)
        try:
            self._process = subprocess.Popen(
                [str(install.python), "-u", str(SCRIPTS / "drive.py"), str(Path(compute_arbiter.__file__)),
                 arbiter_name, str(install.landmarker)],
                cwd=install.source, env=env, stdout=log or subprocess.DEVNULL, stderr=log or subprocess.DEVNULL,
                creationflags=NO_WINDOW)
        finally:
            if log is not None:
                log.close()
        threading.Thread(target=self._serve, name="head3d-driver", daemon=True).start()

    @property
    def running(self) -> bool:
        return self._process.poll() is None

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def use(self, path: Path) -> bool:
        """Whether the head at ``path`` is loaded; asks for it (once) if not."""
        if self.loaded == str(path):
            return True
        if self.requested != str(path):
            self.requested, self.loaded, self.error = str(path), None, None
            self.load(path)
        return False

    def _serve(self) -> None:
        try:
            self._conn = self._listener.accept()
            while True:
                message = self._conn.recv()
                kind = message[0]
                if kind == "ready":
                    self._ready.set()
                elif kind == "layer":
                    _, seq, layer, self.timings = message
                    if layer is not None:
                        x, y, width, height, data = layer
                        layer = Layer(x, y, np.frombuffer(data, np.uint8).reshape(height, width, 4))
                    self.on_layer(seq, layer)
                elif kind == "loaded":
                    self.loaded, self.error = (message[1], None) if message[2] is None else (None, message[2])
                    if message[2]:
                        logger.warning("The 3D head did not load: %s", message[2])
                elif kind == "log":
                    logger.log(message[1], message[2])
        except (EOFError, OSError):
            pass
        finally:
            self._ready.set()

    def send(self, *message) -> bool:
        if self._conn is None:
            return False
        with self._send_lock:
            try:
                self._conn.send(message)
                return True
            except (OSError, EOFError):
                return False

    def load(self, path: Path | None) -> None:
        self.send("head", str(path) if path else None)

    def submit(self, seq: int, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        return self.send("frame", seq, width, height, np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.send("stop")
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(3)
        if self._process.poll() is None:
            self._process.kill()
        with contextlib.suppress(OSError):
            self._listener.close()


def paste(frame: np.ndarray, layer: Layer) -> np.ndarray:
    """``frame`` with the head layer over it (a copy)."""
    import cv2

    out = frame.copy()
    height, width = layer.image.shape[:2]
    roi = out[layer.y:layer.y + height, layer.x:layer.x + width]
    image = layer.image[:roi.shape[0], :roi.shape[1]]
    alpha = image[..., 3].astype(np.float32) * (1 / 255)
    roi[:] = cv2.blendLinear(np.ascontiguousarray(image[..., :3]), roi, alpha, 1 - alpha)
    return out


@dataclass(slots=True)
class _Record:
    t: float
    frame: np.ndarray
    seq: int | None          # None: not sent (the driver was busy); it gets the latest head
    arrived: float


class LiveAvatarSession:
    """Live frames through the 3D head: the same interface as ``LiveFaceSession``.

    At most ``in_flight`` frames are with the driver at once; a frame that arrives meanwhile gets the
    newest head layer (a frame old). Frames leave in order, each once its layer is back or after
    ``max_wait``.
    """

    def __init__(self, driver: HeadDriver, max_wait: float = 0.3, in_flight: int = 2,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.driver = driver
        self.max_wait = max_wait
        self.in_flight = in_flight
        self.clock = clock
        self._records: deque[_Record] = deque()
        self._layers: dict[int, Layer | None] = {}
        self._latest: tuple[float, Layer] | None = None
        self._next_seq = 0
        self._expired = -1          # the newest sequence number that has left
        self._lock = threading.Lock()
        self.frames_in = self.frames_out = 0
        self._rate_since = (clock(), 0, 0)
        self.compose_ms = 0.0
        driver.on_layer = self._on_layer

    def _on_layer(self, seq: int, layer: Layer | None) -> None:
        with self._lock:
            if seq > self._expired:   # not a frame that already left without it
                self._layers[seq] = layer

    def _pending(self) -> int:
        return sum(1 for r in self._records if r.seq is not None and r.seq not in self._layers)

    def submit(self, t: float, frame: np.ndarray) -> None:
        with self._lock:
            seq = None
            if self._pending() < self.in_flight:
                seq = self._next_seq
                self._next_seq += 1
            self._records.append(_Record(t, frame, seq, self.clock()))
            self.frames_in += 1
        if seq is not None and not self.driver.submit(seq, frame):
            self._on_layer(seq, None)

    def ready(self) -> list[tuple[float, np.ndarray]]:
        out = []
        while True:
            with self._lock:
                if not self._records:
                    break
                record = self._records[0]
                if (record.seq is not None and record.seq not in self._layers
                        and self.clock() - record.arrived < self.max_wait):
                    break
                self._records.popleft()
                layer = None
                if record.seq is not None:
                    layer = self._layers.pop(record.seq, None)
                    self._expired = record.seq
                    for late in [seq for seq in self._layers if seq < record.seq]:
                        del self._layers[late]
                if layer is not None:
                    self._latest = (record.t, layer)
                elif self._latest is not None and abs(record.t - self._latest[0]) < 0.2:
                    layer = self._latest[1]
            started = time.perf_counter()
            out.append((record.t, paste(record.frame, layer) if layer is not None else record.frame))
            self.compose_ms = 0.9 * self.compose_ms + 0.1 * (time.perf_counter() - started) * 1000
            self.frames_out += 1
        return out

    def stats(self) -> dict:
        now = self.clock()
        span = max(now - self._rate_since[0], 1e-3)
        fps_in = (self.frames_in - self._rate_since[1]) / span
        fps_out = (self.frames_out - self._rate_since[2]) / span
        self._rate_since = (now, self.frames_in, self.frames_out)
        timings = self.driver.timings or {}
        return {"mode": "3d", "fps_in": round(fps_in, 1), "fps_out": round(fps_out, 1),
                "track_ms": round(float(timings.get("track_ms", 0.0)), 1),
                "render_ms": round(float(timings.get("render_ms", 0.0)), 1),
                "compose_ms": round(self.compose_ms, 1), "face": self._latest is not None}

    def close(self) -> None:
        with self._lock:
            self._records.clear()
            self._layers.clear()
