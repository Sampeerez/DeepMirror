"""The studio's side of PersonaLive (experimental: diffusion, the whole head generated, hair included;
16 GB or more, academic use only). It installs itself like the 3D head (scripts/setup_personalive.py)
and runs as a process of its own; head_engines measures it on this GPU before anything relies on it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path

import cv2
import numpy as np

from backend.config import ROOT
from backend.video.avatar import NO_WINDOW, LamInstall
from backend.video.faces import alignment
from backend.video.head import HEAD_TEMPLATE, OUTPUT
from backend.video.head3d_install import Head3DInstaller

logger = logging.getLogger(__name__)

SIZE = 512             # the crops it takes and gives
RATE = 16.0            # frames per second it is given: what it was made for
WAIT = 0.6             # how long a frame may wait for its generated head (they come in fours)
KEEPS_UP = 8.0         # generated heads per second from which frames wait for their own
DRIVER = Path(__file__).with_name("drive.py")


@dataclass(frozen=True)
class PersonaLiveInstall(LamInstall):
    @classmethod
    def default(cls) -> PersonaLiveInstall:
        return cls(Path(os.environ.get("OVC_PERSONALIVE_DIR") or ROOT / "models" / "personalive"))


class PersonaLiveInstaller(Head3DInstaller):
    SCRIPT = "setup_personalive.py"
    MIN_GPU_MB = 16000
    MIN_FREE_GB = 30
    OFF_VARIABLE = "OVC_PERSONALIVE"
    TITLE = "PersonaLive"

    def __init__(self, data_dir: Path, on_ready=None) -> None:
        super().__init__(data_dir, PersonaLiveInstall.default(), on_ready)
        self.status_path = Path(data_dir) / "personalive-install.json"
        self.log_path = Path(data_dir) / "logs" / "personalive-install.log"


class PersonaLiveHead:
    """In the face process, in place of RemoteHead: ``render`` hands in a frame and gives back the
    newest generated head ready, for a frame a little older (HeadLayer.t says which). The engine
    cuts it out as it does LivePortrait's: both give the head crop of ``HEAD_TEMPLATE``."""

    wait = WAIT
    keeps_up = KEEPS_UP

    def __init__(self, install: PersonaLiveInstall, arbiter_name: str, engine, log_path: Path | None = None) -> None:
        self.engine = engine
        self.error: str | None = None
        self.measured: float | None = None
        self._ready = threading.Event()
        self._portrait_ready = threading.Event()
        self._measure_done = threading.Event()
        self._conn = None
        self._send_lock = threading.Lock()
        self._portrait: int | None = None
        self._next_seq = 0
        self._last_sent = -1e9
        self._sent: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # seq: (frame, points)
        self._done: deque[tuple[int, float, np.ndarray]] = deque(maxlen=16)
        self._lock = threading.Lock()
        key = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=key)
        host, port = self._listener.address
        env = install.run_environment()
        env["DEEPMIRROR_PERSONALIVE"] = f"{host}:{port}:{key.hex()}"
        from backend import compute_arbiter

        log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 (the child has its own handle once started)
        try:
            self._process = subprocess.Popen(
                [str(install.python), "-u", str(DRIVER), str(Path(compute_arbiter.__file__)), arbiter_name],
                cwd=install.source, env=env, stdout=log or subprocess.DEVNULL, stderr=log or subprocess.DEVNULL,
                creationflags=NO_WINDOW)
        finally:
            if log is not None:
                log.close()
        threading.Thread(target=self._serve, name="personalive-driver", daemon=True).start()

    def _serve(self) -> None:
        try:
            self._conn = self._listener.accept()
            while True:
                self.handle(self._conn.recv())
        except (EOFError, OSError):
            self.error = "PersonaLive's process ended"
            logger.warning(self.error)
        finally:
            self._ready.set()
            self._measure_done.set()

    def handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "ready":
            self._ready.set()
        elif kind == "portrait_ready":
            self._portrait_ready.set()
        elif kind == "frames":
            with self._lock:
                self._done.extend(message[1])
        elif kind == "measured":
            self.measured = message[1]
            self._measure_done.set()

    def _send(self, *message) -> bool:
        if self._conn is None:
            return False
        with self._send_lock:
            try:
                self._conn.send(message)
                return True
            except (OSError, EOFError):
                return False

    def _use_portrait(self, portrait: np.ndarray) -> None:
        if self._portrait == id(portrait):
            return
        found = self.engine.portrait_points(portrait)
        if found is None:
            return
        crop = cv2.warpAffine(portrait, alignment(found, HEAD_TEMPLATE, SIZE), (SIZE, SIZE), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE)
        self._portrait_ready.clear()
        if self._send("portrait", crop):
            self._portrait = id(portrait)
            with self._lock:
                self._sent.clear()
                self._done.clear()

    def reset(self) -> None:
        self._portrait = None

    def measure(self, portrait: np.ndarray, timeout: float = 300.0) -> float | None:
        """One generated frame on this GPU, in ms, with the face's own photo (None: it cannot run)."""
        if not self._ready.wait(timeout) or self.error:
            return None
        self._use_portrait(portrait)
        if not self._portrait_ready.wait(timeout):
            return None
        self._measure_done.clear()
        self._send("measure")
        self._measure_done.wait(timeout)
        return self.measured

    def render(self, frame: np.ndarray, points: np.ndarray, t: float, portrait: np.ndarray, wait: float):
        if self._conn is None or not self._ready.is_set():
            return None
        self._use_portrait(portrait)
        if self._portrait_ready.is_set() and t - self._last_sent >= 1.0 / RATE - 1e-3:
            self._last_sent = t
            seq, self._next_seq = self._next_seq, self._next_seq + 1
            crop = cv2.warpAffine(frame, alignment(points, HEAD_TEMPLATE, SIZE), (SIZE, SIZE), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_REPLICATE)
            with self._lock:
                self._sent[seq] = (frame, points)
            self._send("frame", seq, t, crop)
        with self._lock:
            if not self._done:
                return None
            seq, done_t, image = self._done.popleft()
            sent = self._sent.pop(seq, None)
            for old in [s for s in self._sent if s < seq]:
                del self._sent[old]
        if sent is None:
            return None
        return self.engine._layer(sent[0], sent[1], cv2.resize(image, (OUTPUT, OUTPUT)), done_t, wait)

    def close(self) -> None:
        self._send("stop")
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(3)
        if self._process.poll() is None:
            self._process.kill()
        with contextlib.suppress(OSError):
            self._listener.close()
