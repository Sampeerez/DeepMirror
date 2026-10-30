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

    @property
    def marker(self) -> Path:
        """Written by scripts/setup_head3d.py once the install has been checked to run on this GPU."""
        return self.root / "installed.json"

    def available(self) -> bool:
        return self.marker.is_file() and self.python.is_file()

    @property
    def plugins(self) -> Path:
        """Prebuilt CUDA plugins loaded without a compiler (nvdiffrast's, for LAM's face tracker)."""
        return self.root / "plugins"

    def run_environment(self) -> dict[str, str]:
        """The environment LAM's scripts run in: the studio's, without its virtual environment."""
        env = {name: value for name, value in os.environ.items() if name.upper() != "VIRTUAL_ENV"}
        env["DEEPMIRROR_LAM_PLUGINS"] = str(self.plugins)
        # LAM marks its image encoder for torch.compile, which needs Triton (not on Windows): eager.
        env["TORCHDYNAMO_DISABLE"] = "1"
        return env

    def compiler_environment(self) -> dict[str, str]:
        """For compiling LAM's CUDA libraries (maintainers, see scripts/build_head3d_wheels.py):
        Visual Studio 2019's x64 tools and CUDA 11.8 on the path, and LAM's environment's scripts (ninja)."""
        env = self.run_environment()
        extra = [str(self.python.parent)]
        if sys.platform == "win32":
            vcvars = _vcvars()
            if vcvars is not None:
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    out = subprocess.run(f'cmd /s /c ""{vcvars}" >nul 2>nul && set"', capture_output=True,
                                         text=True, timeout=60, creationflags=NO_WINDOW).stdout
                    merge_environment(env, (line.partition("=")[::2] for line in out.splitlines() if "=" in line))
            cuda = _cuda_home()
            if cuda is not None:
                merge_environment(env, [("CUDA_HOME", str(cuda)), ("CUDA_PATH", str(cuda)), ("DISTUTILS_USE_SDK", "1")])
                extra.insert(0, str(cuda / "bin"))
        path = next((value for name, value in env.items() if name.upper() == "PATH"), "")
        merge_environment(env, [("PATH", os.pathsep.join([*extra, path]))])
        return env


def merge_environment(env: dict[str, str], updates) -> None:
    """Set ``updates`` (name, value pairs) in ``env``, names compared without case as Windows does:
    ``Path`` from cmd and ``PATH`` from elsewhere are one variable, never two."""
    for name, value in updates:
        for existing in [key for key in env if key.upper() == name.upper()]:
            del env[existing]
        env[name] = value


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
                 on_built: Callable[[str], None] | None = None, arbiter_name: str | None = None) -> None:
        self.faces_root = Path(faces_root)
        self.install = install or LamInstall.default()
        self._parser = parser
        self._voice_live = voice_live
        self._log_dir = log_dir
        self._on_built = on_built
        self._arbiter_name = arbiter_name
        self._steps: dict[str, int] = {}
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
            return {"state": "ready", "updated": int((folder / HEAD_FILE).stat().st_mtime)}
        if not self.available:
            return {"state": "unavailable"}
        if (folder / ERROR_FILE).is_file():
            return {"state": "failed", "error": (folder / ERROR_FILE).read_text(encoding="utf-8", errors="replace")[-300:]}
        return {"state": "missing"}

    def request(self, face_id: str, steps: int | None = None) -> None:
        """Build (or rebuild) a face's 3D head; ``steps``: training steps on its photos (more: finer)."""
        if steps is not None:
            self._steps[face_id] = steps
        self._request(face_id)

    def _request(self, face_id: str) -> None:
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
        parser = self._parser().resolve()
        env = self.install.run_environment()
        if face_id in self._steps:
            env["DEEPMIRROR_TRAIN_STEPS"] = str(self._steps.pop(face_id))
        if self._arbiter_name:
            from backend import compute_arbiter

            env["DEEPMIRROR_ARBITER_FILE"] = str(Path(compute_arbiter.__file__))
            env["DEEPMIRROR_ARBITER_NAME"] = self._arbiter_name
        try:
            self._process = subprocess.Popen(
                [str(self.install.python), "-u", str(SCRIPTS / "build.py"), str(folder.resolve()), str(parser),
                 str(self.install.landmarker.resolve()), str(parser.parent / "gpen_bfr_256.onnx")],
                cwd=self.install.source, env=env, stdout=subprocess.PIPE,
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
    points: np.ndarray | None = None   # the face's five points in the frame (from the head process)


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
        env = install.run_environment()
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
        path = Path(path).resolve()
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
                        x, y, width, height, data, five = layer
                        layer = Layer(x, y, np.frombuffer(data, np.uint8).reshape(height, width, 4),
                                      np.asarray(five, np.float32))
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
        self.send("head", str(Path(path).resolve()) if path else None)

    def submit(self, seq: int, t: float, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        return self.send("frame", seq, float(t), width, height, np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.send("stop")
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(3)
        if self._process.poll() is None:
            self._process.kill()
        with contextlib.suppress(OSError):
            self._listener.close()
