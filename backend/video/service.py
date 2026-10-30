"""The face clone in the server: its models, the face library, settings and the face process."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from backend.assets import AssetStore
from backend.compute_arbiter import GpuArbiter
from backend.video.library import FaceLibrary
from backend.video.swap import SWAPPERS
from backend.video.worker import FaceWorker, decode_image, encode_jpeg

logger = logging.getLogger(__name__)

SETTING_KEYS = ("mode", "swapper", "enhance", "mouth", "occlusion")
DEFAULT_SETTINGS = {"mode": "hair", "swapper": "auto", "enhance": "auto", "mouth": "auto", "occlusion": True}


class FaceService:
    def __init__(self, assets: AssetStore, arbiter: GpuArbiter, data_dir: Path, gpu: Callable[[], bool]) -> None:
        self.assets = assets
        self.arbiter = arbiter
        self.data_dir = Path(data_dir)
        self.gpu = gpu
        self.library = FaceLibrary(self.data_dir)
        self._settings_path = self.data_dir / "video_settings.json"
        self.settings = self._load_settings()
        self._worker: FaceWorker | None = None
        self._lock = threading.Lock()
        self.last_stats: dict | None = None

    # --- Settings ---------------------------------------------------------

    def _load_settings(self) -> dict:
        try:
            stored = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        settings = {**DEFAULT_SETTINGS, **{k: v for k, v in stored.items() if k in SETTING_KEYS}}
        if settings["mode"] not in ("hair", "face"):   # the whole head is for files only now
            settings["mode"] = "hair"
        return settings

    def update_settings(self, payload: dict) -> dict:
        settings = dict(self.settings)
        if payload.get("mode") in ("hair", "face"):
            settings["mode"] = payload["mode"]
        if payload.get("swapper") == "auto" or payload.get("swapper") in SWAPPERS:
            settings["swapper"] = payload["swapper"]
        for name in ("enhance", "mouth"):
            if payload.get(name) in ("auto", "on", "off"):
                settings[name] = payload[name]
        if isinstance(payload.get("occlusion"), bool):
            settings["occlusion"] = payload["occlusion"]
        self.settings = settings
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(settings), encoding="utf-8")
        if self._worker is not None:
            self._worker.send("settings", settings)
        return settings

    # --- Models -----------------------------------------------------------

    @property
    def models_ready(self) -> bool:
        return self.assets.group_ready("video")

    def prepare_models(self) -> None:
        """Download the face models in the background (about 1.8 GB, once)."""
        self.assets.download_missing_in_background("video")

    # --- Face process -----------------------------------------------------

    def worker(self) -> FaceWorker:
        """The running face process (started on first use, with the active face and the settings)."""
        if not self.models_ready:
            raise RuntimeError("The face models are not downloaded yet")
        with self._lock:
            if self._worker is None:
                self._worker = FaceWorker(self.arbiter.name, self.assets.folder("video"),
                                          self.data_dir / "cache" / "face_swap_ms.json", self.gpu())
                self._worker.subscribe(self._on_message)
            if not self._worker.running:
                self._worker.start()
                self._worker.send("settings", self.settings)
                self._send_identity()
            return self._worker

    def _on_message(self, message: tuple) -> None:
        if message[0] == "stats":
            self.last_stats = message[1]

    def _send_identity(self) -> None:
        if self._worker is None:
            return
        face_id = self.library.active()
        if face_id is None:
            self._worker.send("identity", None, None)
            return
        portrait = self.library.portrait(face_id)
        self._worker.send("identity", self.library.identity(face_id), str(portrait) if portrait else None,
                          [str(path) for path in self.library.photos(face_id)])

    def close(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    # --- Faces ------------------------------------------------------------

    def add_face(self, name: str, photos: list[bytes]) -> dict:
        """Learn a face from one or more photos; the identity is averaged over every photo with a face."""
        worker = self.worker()
        if not worker.wait_ready(300):
            raise RuntimeError("The face process did not start")
        embedding, notes = worker.enroll(photos)
        if embedding is None:
            raise ValueError(str(notes) if notes else "No face found in the photos")
        assert isinstance(notes, list)
        best = max(range(len(notes)), key=lambda i: notes[i].get("size", 0) if notes[i].get("face") else -1)
        return self.library.add(name, photos, np.asarray(embedding), notes, _thumbnail(photos[best]))

    def activate(self, face_id: str | None) -> None:
        self.library.set_active(face_id)
        self._send_identity()

    def delete(self, face_id: str) -> None:
        self.library.delete(face_id)
        self._send_identity()

    def _cached_swap_ms(self) -> dict | None:
        """Swapper timings measured on an earlier run (per GPU), so the settings show them before the camera starts."""
        try:
            return _merge(json.loads((self.data_dir / "cache" / "face_swap_ms.json").read_text(encoding="utf-8"))) or None
        except (OSError, ValueError):
            return None

    def status(self) -> dict:
        return {
            "models_ready": self.models_ready,
            "assets": self.assets.status("video"),
            "faces": self.library.list(),
            "active_face": self.library.active(),
            "settings": self.settings,
            "swappers": [{"id": spec.id, "label": spec.label} for spec in SWAPPERS.values()],
            "running": self._worker is not None and self._worker.running,
            "swap_ms": ((self._worker.info or {}).get("swap_ms") if self._worker else None) or self._cached_swap_ms(),
            "stats": self.last_stats,
        }


def _merge(tables: dict) -> dict:
    merged: dict = {}
    for table in tables.values():
        if isinstance(table, dict):
            merged.update(table)
    return merged


def _thumbnail(photo: bytes) -> bytes:
    import cv2

    image = decode_image(photo)
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return encode_jpeg(cv2.resize(image[top:top + side, left:left + side], (160, 160)), 85)
