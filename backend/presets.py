"""Voice presets: the built-in sound colours plus user presets stored as JSON.

A preset restores a sound in one click: the effect rack and, in user presets,
pitch/formant and optionally the RVC controls. Optional fields are kept only when present, so
built-in presets (which describe a character, not a pitch tracker) never
override a user's detector or retrieval settings.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

from backend.inference.pipeline import CONTROL_RANGES, clamped_number
from backend.inference.pitch import PITCH_METHOD_BY_ID

logger = logging.getLogger(__name__)

MAX_USER_PRESETS = 100
OPTIONAL_CONTROLS = ("index_rate", "filter_radius", "rms_mix_rate", "protect", "crepe_hop_length")

# Built-in presets only colour the sound: they never touch the pitch or the
# formant, which each voice calibrates to the speaker.
BUILTIN_PRESETS = [
    {
        "id": "clean",
        "name": "Natural",
        "description": "No effects: the voice as converted",
        "settings": {"effects": {}},
    },
    {
        "id": "radio",
        "name": "Radio",
        "description": "Broadcast warmth and compression",
        "settings": {
            "effects": {
                "eq": {"enabled": True, "low": 3, "high": 2},
                "compressor": {"enabled": True, "threshold": -26, "ratio": 5},
                "gain": {"enabled": True, "db": 2},
            },
        },
    },
    {
        "id": "telephone",
        "name": "Phone",
        "description": "Narrowband landline call",
        "settings": {
            "effects": {
                "telephone": {"enabled": True},
                "compressor": {"enabled": True, "threshold": -24, "ratio": 4},
            },
        },
    },
    {
        "id": "walkie",
        "name": "Walkie-talkie",
        "description": "Crackly two-way radio",
        "settings": {
            "effects": {
                "telephone": {"enabled": True},
                "distortion": {"enabled": True, "drive": 0.3},
                "bitcrush": {"enabled": True, "bits": 10, "down": 2},
                "compressor": {"enabled": True, "threshold": -24, "ratio": 6},
            },
        },
    },
    {
        "id": "megaphone",
        "name": "Megaphone",
        "description": "Distorted PA horn",
        "settings": {
            "effects": {
                "telephone": {"enabled": True},
                "distortion": {"enabled": True, "drive": 0.55},
                "gain": {"enabled": True, "db": 3},
            },
        },
    },
    {
        "id": "robot",
        "name": "Robot",
        "description": "Ring-mod metal voice with bit reduction",
        "settings": {
            "effects": {
                "robot": {"enabled": True, "freq": 90},
                "bitcrush": {"enabled": True, "bits": 10, "down": 2},
            },
        },
    },
    {
        "id": "stadium",
        "name": "Hall",
        "description": "Big announcement reverb",
        "settings": {
            "effects": {
                "echo": {"enabled": True, "time": 0.6, "feedback": 0.4, "mix": 0.4},
                "reverb": {"enabled": True, "size": 1.0, "mix": 0.5},
                "compressor": {"enabled": True, "threshold": -20, "ratio": 3},
            },
        },
    },
    {
        "id": "cave",
        "name": "Cave",
        "description": "Long dark echoes",
        "settings": {
            "effects": {
                "reverb": {"enabled": True, "size": 0.85, "mix": 0.45},
                "echo": {"enabled": True, "time": 0.35, "feedback": 0.5, "mix": 0.35},
                "eq": {"enabled": True, "low": 2, "high": -4},
            },
        },
    },
]


def normalize_settings(settings: object) -> dict:
    """The stored ``settings`` payload: always pitch/formant/effects, optional RVC controls if valid."""
    if not isinstance(settings, dict):
        settings = {}
    normalized = {
        "pitch_shift": clamped_number(settings.get("pitch_shift"), float, -24.0, 24.0) or 0.0,
        "formant_shift": clamped_number(settings.get("formant_shift"), float, -24.0, 24.0) or 0.0,
        "effects": settings.get("effects") if isinstance(settings.get("effects"), dict) else {},
    }
    method = settings.get("f0_method")
    if isinstance(method, str) and method.strip().lower() in PITCH_METHOD_BY_ID:
        normalized["f0_method"] = method.strip().lower()
    for name in OPTIONAL_CONTROLS:
        value = clamped_number(settings.get(name), *CONTROL_RANGES[name])
        if value is not None:
            normalized[name] = value
    return normalized


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return slug[:48] or "preset"


class PresetStore:
    """Thread-safe JSON-file-backed store for user presets."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._presets: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._presets = {
                    p["id"]: p
                    for p in data
                    if isinstance(p, dict) and "id" in p and "settings" in p
                }
            logger.info("Loaded %d user preset(s) from %s", len(self._presets), self._path)
        except Exception:
            logger.exception("Failed to load user presets from %s", self._path)

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(list(self._presets.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def list_builtin(self) -> list[dict]:
        return [{**p, "builtin": True} for p in BUILTIN_PRESETS]

    def list_user(self) -> list[dict]:
        with self._lock:
            return [{**p, "builtin": False} for p in self._presets.values()]

    def save(self, name: str, settings: dict) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Preset name is empty")
        if len(name) > 40:
            raise ValueError("Preset name is too long (max 40 characters)")
        if not isinstance(settings, dict):
            raise ValueError("Preset settings must be an object")

        preset_id = f"user-{_slugify(name)}"
        preset = {
            "id": preset_id,
            "name": name,
            "description": "Custom preset",
            # Stores the full Voice Lab state, not just pitch/formant/effects,
            # so applying a preset restores everything the user tuned.
            "settings": normalize_settings(settings),
        }

        with self._lock:
            if preset_id not in self._presets and len(self._presets) >= MAX_USER_PRESETS:
                raise ValueError(f"Preset limit reached ({MAX_USER_PRESETS})")
            self._presets[preset_id] = preset
            self._flush_locked()
        logger.info("Saved user preset: %s", preset_id)
        return {**preset, "builtin": False}

    def delete(self, preset_id: str) -> None:
        with self._lock:
            if preset_id not in self._presets:
                raise KeyError(preset_id)
            del self._presets[preset_id]
            self._flush_locked()
        logger.info("Deleted user preset: %s", preset_id)
