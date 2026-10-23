"""Training jobs as seen by the server: create, start, observe, stop, resume, delete.

Each job lives in ``data/training/<id>/``::

    job.json      what to train (name, sample rate, epochs, ...)
    status.json   progress written by the worker process
    uploads/      the source audio
    dataset.npz   prepared segments and features (reused on resume)
    checkpoints/  generator/discriminator state for resuming
    worker.log    the worker's output

Only one job trains at a time: it needs the whole GPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from backend.config import ROOT, Settings
from backend.model_manager import MODEL_EXTENSIONS, sanitize_filename
from backend.training.dataset import audio_duration
from backend.training.hparams import CONFIGS, MAX_AUDIO_SECONDS, MAX_EPOCHS, MIN_AUDIO_SECONDS

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}
MAX_UPLOAD_BYTES = 2 * 1024**3
TERMINAL_STATES = {"done", "stopped", "error", "interrupted"}
# Workers refresh status.json at least every 20 s; older than this means dead.
HEARTBEAT_TIMEOUT_SECONDS = 120
_JOB_ID = re.compile(r"[0-9a-f]{12}")


class TrainingError(ValueError):
    """A user-correctable problem with a training request."""


class TrainingJobs:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.DATA_DIR / "training"
        self.root.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._mark_interrupted()

    # --- Paths & state ------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id or ""):
            raise KeyError(job_id)
        directory = self.root / job_id
        if not (directory / "job.json").is_file():
            raise KeyError(job_id)
        return directory

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(data), encoding="utf-8")
        partial.replace(path)

    @staticmethod
    def _heartbeat_alive(status: dict) -> bool:
        return (status.get("state") not in TERMINAL_STATES
                and time.time() - float(status.get("updated_at") or 0) < HEARTBEAT_TIMEOUT_SECONDS)

    def _mark_interrupted(self) -> None:
        """Jobs whose worker died with the previous server can be resumed.

        A worker is its own process and may outlive a server restart; one that
        is still sending heartbeats is left alone and keeps counting as running.
        """
        for directory in self.root.iterdir():
            status = self._read(directory / "status.json")
            if status and status.get("state") not in TERMINAL_STATES and not self._heartbeat_alive(status):
                status.update(state="interrupted", message="Training stopped unexpectedly; continue it to resume.")
                self._write(directory / "status.json", status)

    def _running_id(self) -> str | None:
        for job_id, process in list(self._processes.items()):
            if process.poll() is None:
                return job_id
            self._processes.pop(job_id)
            self._settle(job_id, process.returncode)
        for directory in self.root.iterdir():  # a worker started by an earlier server run
            if directory.name not in self._processes and self._heartbeat_alive(
                    self._read(directory / "status.json")):
                return directory.name
        return None

    def _settle(self, job_id: str, returncode: int | None) -> None:
        """A worker that died without reporting a final state is marked as failed."""
        path = self.root / job_id / "status.json"
        status = self._read(path)
        if status.get("state") not in TERMINAL_STATES:
            log = (self.root / job_id / "worker.log")
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] if log.is_file() else []
            status.update(state="error", error=tail[0] if tail else f"Worker exited with code {returncode}")
            self._write(path, status)

    def describe(self, job_id: str) -> dict:
        directory = self._job_dir(job_id)
        return {"id": job_id, **self._read(directory / "job.json"), **self._read(directory / "status.json")}

    def list(self) -> list[dict]:
        with self._lock:
            self._running_id()
        jobs = []
        for directory in self.root.iterdir():
            if (directory / "job.json").is_file():
                jobs.append(self.describe(directory.name))
        return sorted(jobs, key=lambda job: job.get("created_at", 0), reverse=True)

    # --- Lifecycle ----------------------------------------------------------

    async def create(self, name: str, files: list[UploadFile], sample_rate: int, epochs: int | None,
                     batch_size: int | None, save_every: int, overwrite: bool) -> dict:
        model_name = Path(sanitize_filename(name)).stem
        if not model_name:
            raise TrainingError("Give the voice a name.")
        if sample_rate not in CONFIGS:
            raise TrainingError(f"Sample rate must be one of {sorted(CONFIGS)}.")
        if epochs is not None and not 1 <= epochs <= MAX_EPOCHS:  # None: planned from the audio and GPU
            raise TrainingError(f"Epochs must be between 1 and {MAX_EPOCHS}.")
        existing = [self.settings.MODELS_DIR / f"{model_name}{ext}" for ext in (*MODEL_EXTENSIONS, ".index")]
        if not overwrite and any(path.exists() for path in existing):
            raise TrainingError(f"A voice called '{model_name}' already exists. Pick another name.")
        if not files:
            raise TrainingError("Upload at least one audio file.")
        with self._lock:
            if self._running_id():
                raise TrainingError("Another voice is training. Wait for it or stop it first.")

        job_id = uuid.uuid4().hex[:12]
        directory = self.root / job_id
        uploads = directory / "uploads"
        uploads.mkdir(parents=True)
        try:
            total_seconds = 0.0
            for number, upload in enumerate(files):
                extension = Path(upload.filename or "").suffix.lower()
                if extension not in AUDIO_EXTENSIONS:
                    raise TrainingError(f"'{upload.filename}' is not a supported audio file.")
                target = uploads / f"{number:03d}{extension}"
                written = 0
                with target.open("wb") as out:
                    while block := await upload.read(1024 * 1024):
                        written += len(block)
                        if written > MAX_UPLOAD_BYTES:
                            raise TrainingError("Each audio file must be under 2 GB.")
                        out.write(block)
                try:
                    total_seconds += audio_duration(target)
                except Exception as exc:
                    raise TrainingError(f"Could not read '{upload.filename}': {exc}") from exc
            if total_seconds < MIN_AUDIO_SECONDS:
                raise TrainingError(f"Upload at least {MIN_AUDIO_SECONDS // 60} minute of speech "
                                    f"(got {total_seconds:.0f} s); 5 minutes trains well, 10-30 is ideal.")
            if total_seconds > MAX_AUDIO_SECONDS:
                raise TrainingError(f"Use at most {MAX_AUDIO_SECONDS // 60} minutes of audio.")
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

        spec = {"name": model_name, "sample_rate": sample_rate, "epochs": epochs, "batch_size": batch_size,
                "save_every": save_every, "created_at": time.time(), "audio_seconds": round(total_seconds, 1),
                "files": [upload.filename for upload in files],
                # The worker is a separate process: hand it this server's locations.
                "models_dir": str(self.settings.MODELS_DIR), "assets_dir": str(self.settings.ASSETS_DIR),
                "device": self.settings.DEVICE}
        self._write(directory / "job.json", spec)
        self._write(directory / "status.json", {"state": "queued", "stage": "assets", "progress": 0.0,
                                                "message": "Starting"})
        self._start(job_id)
        return self.describe(job_id)

    def _start(self, job_id: str) -> None:
        directory = self._job_dir(job_id)
        (directory / "stop").unlink(missing_ok=True)
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
        log = (directory / "worker.log").open("ab")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._processes[job_id] = subprocess.Popen(
            [sys.executable, "-m", "backend.training.worker", str(directory)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
        logger.info("Started training job %s", job_id)

    def stop(self, job_id: str) -> dict:
        directory = self._job_dir(job_id)
        (directory / "stop").touch()
        return self.describe(job_id)

    def resume(self, job_id: str, epochs: int | None) -> dict:
        directory = self._job_dir(job_id)
        with self._lock:
            if self._running_id():
                raise TrainingError("Another voice is training. Wait for it or stop it first.")
            spec = self._read(directory / "job.json")
            status = self._read(directory / "status.json")
            if epochs is not None:
                if not 1 <= epochs <= MAX_EPOCHS:
                    raise TrainingError(f"Epochs must be between 1 and {MAX_EPOCHS}.")
                if epochs <= int(status.get("epoch") or 0):
                    raise TrainingError("To continue, choose more epochs than already trained.")
                spec["epochs"] = epochs
                self._write(directory / "job.json", spec)
            status.update(state="queued", error=None, message="Resuming")
            self._write(directory / "status.json", status)
            self._start(job_id)
        return self.describe(job_id)

    def delete(self, job_id: str) -> None:
        directory = self._job_dir(job_id)
        with self._lock:
            if self._running_id() == job_id:
                raise TrainingError("Stop the training before deleting it.")
        shutil.rmtree(directory)

    def close(self) -> None:
        """Ask running workers to save and stop when the server shuts down."""
        for job_id, process in self._processes.items():
            if process.poll() is None:
                (self.root / job_id / "stop").touch()
