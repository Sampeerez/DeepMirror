"""The 3D head installs itself: on a computer with an NVIDIA GPU of 6 GB or more, the first start
runs scripts/setup_head3d.py in the background (about 7 GB of downloads, nothing compiled) and
reports its progress; Face + hair is used meanwhile. ``OVC_HEAD3D=off`` turns this off.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from backend.config import ROOT
from backend.video.avatar import NO_WINDOW, LamInstall

logger = logging.getLogger(__name__)

MIN_GPU_MB = 6000
MIN_FREE_GB = 12        # downloads and the unpacked install side by side, at the peak
BELOW_NORMAL = 0x4000   # Windows priority class: the voice comes first, also while installing


def nvidia_memory_mb() -> int:
    """The largest NVIDIA GPU's memory, in MB (0 without one)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return 0
    try:
        out = subprocess.run([smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, creationflags=NO_WINDOW).stdout
        return max((int(float(line)) for line in out.split() if line.strip()), default=0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


class Head3DInstaller:
    def __init__(self, data_dir: Path, install: LamInstall | None = None,
                 on_ready: Callable[[], None] | None = None) -> None:
        self.install = install or LamInstall.default()
        self.status_path = Path(data_dir) / "head3d-install.json"
        self.log_path = Path(data_dir) / "logs" / "head3d-install.log"
        self.on_ready = on_ready
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._reason: str | None = None

    def why_not(self) -> str | None:
        """Why this computer does not get the 3D head installed by itself (None: it does)."""
        if os.environ.get("OVC_HEAD3D", "").strip().lower() in ("0", "off", "false", "no"):
            return "Turned off (OVC_HEAD3D)"
        if sys.platform != "win32":
            return "Installed by itself on Windows only (see scripts/setup_head3d.py)"
        memory = nvidia_memory_mb()
        if memory < MIN_GPU_MB:
            return "Needs an NVIDIA GPU with 6 GB or more" if memory else "Needs an NVIDIA GPU"
        self.install.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.install.root).free / 2**30
        if free < MIN_FREE_GB:
            return f"Needs {MIN_FREE_GB} GB free on the disk ({free:.0f} GB now)"
        return None

    def start_if_needed(self) -> None:
        """At server start: install in the background unless installed or unsuitable."""
        if self.install.available():
            return
        self._reason = self.why_not()
        if self._reason is None:
            self.start()
        else:
            logger.info("3D head not installed: %s", self._reason)

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 (the child has its own handle)
            try:
                env = {k: v for k, v in os.environ.items() if k.upper() != "VIRTUAL_ENV"}
                self._process = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "scripts" / "setup_head3d.py"), "--status", str(self.status_path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                    creationflags=NO_WINDOW | (BELOW_NORMAL if sys.platform == "win32" else 0))
            finally:
                log.close()
        logger.info("Installing the 3D head in the background (log: %s)", self.log_path)
        threading.Thread(target=self._wait, args=(self._process,), name="head3d-install", daemon=True).start()

    def _wait(self, process: subprocess.Popen) -> None:
        code = process.wait()
        if code == 0 and self.install.available():
            logger.info("3D head installed")
            if self.on_ready is not None:
                self.on_ready()
        else:
            logger.warning("Installing the 3D head failed (exit %s): see %s", code, self.log_path)

    def status(self) -> dict:
        """``{"state": ready | installing | failed | unavailable, "progress", "step", "error"}``"""
        if self.install.available():
            return {"state": "ready"}
        running = self._process is not None and self._process.poll() is None
        try:
            stored = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        if running:
            return {"state": "installing", "progress": stored.get("progress", 0.0), "step": stored.get("step", "")}
        if stored.get("state") == "failed":
            return {"state": "failed", "error": stored.get("error")}
        return {"state": "unavailable", "error": self._reason}

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()   # downloads resume on the next start
