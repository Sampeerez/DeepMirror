"""The faces to wear: one folder per face in data/faces, with its photos, identity and a thumbnail."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

_ID = re.compile(r"^[0-9a-f]{12}$")
MAX_PHOTOS = 12


class FaceNotFoundError(KeyError):
    pass


class FaceLibrary:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "faces"
        self._lock = threading.Lock()

    def _folder(self, face_id: str) -> Path:
        if not _ID.match(face_id or "") or not (self.root / face_id / "face.json").is_file():
            raise FaceNotFoundError(face_id)
        return self.root / face_id

    def list(self) -> list[dict]:
        faces = []
        if self.root.is_dir():
            for folder in sorted(self.root.iterdir()):
                if _ID.match(folder.name) and (folder / "face.json").is_file():
                    try:
                        faces.append(json.loads((folder / "face.json").read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        continue
        return sorted(faces, key=lambda face: face.get("created_at", 0), reverse=True)

    def add(self, name: str, photos: list[bytes], embedding: np.ndarray, notes: list[dict],
            thumbnail: bytes) -> dict:
        face_id = uuid.uuid4().hex[:12]
        folder = self.root / face_id
        folder.mkdir(parents=True)
        for index, photo in enumerate(photos[:MAX_PHOTOS]):
            (folder / f"photo_{index}.jpg").write_bytes(photo)
        np.save(folder / "identity.npy", embedding.astype(np.float32))
        (folder / "thumb.jpg").write_bytes(thumbnail)
        usable = sum(1 for note in notes if note.get("face"))
        meta = {"id": face_id, "name": name.strip()[:60] or "Face", "photos": usable, "notes": notes,
                "created_at": time.time()}
        partial = folder / "face.json.part"
        partial.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        partial.replace(folder / "face.json")
        return meta

    def identity(self, face_id: str) -> np.ndarray:
        return np.load(self._folder(face_id) / "identity.npy")

    def thumbnail(self, face_id: str) -> Path:
        return self._folder(face_id) / "thumb.jpg"

    def delete(self, face_id: str) -> None:
        with self._lock:
            shutil.rmtree(self._folder(face_id))
            if self.active() == face_id:
                self.set_active(None)

    def active(self) -> str | None:
        try:
            face_id = json.loads((self.root / "active.json").read_text(encoding="utf-8")).get("face")
        except (OSError, ValueError):
            return None
        try:
            self._folder(face_id)
        except FaceNotFoundError:
            return None
        return face_id

    def set_active(self, face_id: str | None) -> None:
        if face_id is not None:
            self._folder(face_id)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "active.json").write_text(json.dumps({"face": face_id}), encoding="utf-8")
