"""The speaker's voice profile and the register of each voice model.

"My voice" records the user reading a few varied lines once. Its pitch
statistics (median, range), level and microphone noise are kept on the server
in ``data/profile/`` and replace the few seconds the live stream used to learn
at every start, so the automatic pitch shift and melodic range are right from
the first word and the same in every session.

Every voice needs its own register to calibrate against. Voices trained here
carry it (``f0_stats``). Downloaded voices do not, and it cannot be read
reliably from the weights or the index, so the user can give a short clip of
the original voice; its statistics are stored in ``data/calibration/``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.inference.auto_pitch import F0Stats, auto_expressiveness, compute_f0_stats, semitones_between

logger = logging.getLogger(__name__)

RATE = 16000
FRAMES_PER_SECOND = 100
MIN_VOICED_SECONDS = 8.0        # less than this is not enough to know a voice
GOOD_VOICED_SECONDS = 30.0
MAX_SECONDS = 600
SHIFT_STEP = 0.5
MAX_SHIFT = 24.0


class ProfileError(ValueError):
    pass


def _register(median_hz: float) -> str:
    if median_hz < 115:
        return "deep"
    if median_hz < 160:
        return "low"
    if median_hz < 215:
        return "middle"
    return "high"


def analyse(audio16: np.ndarray, f0: np.ndarray) -> dict:
    """Pitch, range, level and noise of a 16 kHz recording and its pitch track (100 frames/s)."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0[f0 > 0]
    voiced_seconds = len(voiced) / FRAMES_PER_SECOND
    if voiced_seconds < MIN_VOICED_SECONDS:
        raise ProfileError(f"Only {voiced_seconds:.0f} s of voice were heard; read at least "
                           f"{MIN_VOICED_SECONDS:.0f} s (all the lines is best).")
    stats = compute_f0_stats([voiced])
    assert stats is not None
    p5, p95 = np.exp(np.percentile(np.log(voiced), [5, 95]))
    hop = RATE // FRAMES_PER_SECOND
    frames = len(audio16) // hop
    energy = np.square(audio16[: frames * hop].astype(np.float64)).reshape(frames, hop).mean(axis=1)
    level = 10 * np.log10(energy + 1e-12)
    n = min(frames, len(f0))
    speech_level = float(np.median(level[:n][f0[:n] > 0]))
    noise_floor = float(np.percentile(level, 10))
    warnings = []
    if voiced_seconds < GOOD_VOICED_SECONDS:
        warnings.append("Short recording: reading every line gives a steadier profile.")
    if speech_level < -42:
        warnings.append("Your voice was quiet: move closer to the microphone or raise its level.")
    if speech_level - noise_floor < 25:
        warnings.append("There was a lot of background noise.")
    return {
        **stats.to_dict(),
        "p5_hz": round(float(p5), 2), "p95_hz": round(float(p95), 2),
        "range_semitones": round(semitones_between(stats.p10_hz, stats.p90_hz), 1),
        "register": _register(stats.median_hz),
        "voiced_seconds": round(voiced_seconds, 1),
        "seconds": round(len(audio16) / RATE, 1),
        "speech_level_db": round(speech_level, 1),
        "noise_floor_db": round(noise_floor, 1),
        "warnings": warnings,
    }


def calibration(speaker: F0Stats, target: F0Stats | None) -> dict | None:
    """Shift and melodic range that move ``speaker`` onto ``target``'s register."""
    if target is None:
        return None
    shift = semitones_between(speaker.median_hz, target.median_hz)
    return {"pitch_shift": float(np.clip(round(shift / SHIFT_STEP) * SHIFT_STEP, -MAX_SHIFT, MAX_SHIFT)),
            "expressiveness": round(auto_expressiveness(target, speaker), 2),
            "target": target.to_dict()}


class VoiceProfileStore:
    """``data/profile/`` (the speaker) and ``data/calibration/`` (registers of downloaded voices)."""

    def __init__(self, data_dir: Path) -> None:
        self.profile_dir = Path(data_dir) / "profile"
        self.calibration_dir = Path(data_dir) / "calibration"
        self._lock = threading.Lock()
        self.revision = 0          # bumps on every change, so live streams pick it up

    # --- The speaker ----------------------------------------------------------

    @property
    def _profile_path(self) -> Path:
        return self.profile_dir / "profile.json"

    def load(self) -> dict | None:
        try:
            return json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def stats(self) -> F0Stats | None:
        profile = self.load()
        return F0Stats.from_dict(profile) if profile else None

    def save(self, audio16: np.ndarray, analysis: dict, language: str) -> dict:
        profile = {**analysis, "language": language, "created_at": time.time()}
        with self._lock:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            # The reading is kept, so the profile can be measured again by a better analysis later.
            sf.write(str(self.profile_dir / "reading.wav"), np.clip(audio16, -1, 1), RATE, subtype="PCM_16")
            _write_json(self._profile_path, profile)
            self.revision += 1
        logger.info("Saved the voice profile: %.0f Hz median, %.1f semitones of range, %.0f s of voice",
                    analysis["median_hz"], analysis["range_semitones"], analysis["voiced_seconds"])
        return profile

    def delete(self) -> None:
        with self._lock:
            for name in ("profile.json", "reading.wav"):
                (self.profile_dir / name).unlink(missing_ok=True)
            self.revision += 1

    # --- Registers of downloaded voices ---------------------------------------------

    def _reference_path(self, model_name: str) -> Path:
        return self.calibration_dir / f"{Path(model_name).stem}.json"

    def reference(self, model_name: str) -> F0Stats | None:
        try:
            data = json.loads(self._reference_path(model_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return F0Stats.from_dict(data.get("f0_stats"))

    def save_reference(self, model_name: str, stats: F0Stats, voiced_seconds: float) -> None:
        with self._lock:
            self.calibration_dir.mkdir(parents=True, exist_ok=True)
            _write_json(self._reference_path(model_name),
                        {"f0_stats": stats.to_dict(), "voiced_seconds": round(voiced_seconds, 1),
                         "created_at": time.time()})
            self.revision += 1
        logger.info("Saved the register of %s: %.0f Hz median", model_name, stats.median_hz)

    def delete_reference(self, model_name: str) -> None:
        with self._lock:
            self._reference_path(model_name).unlink(missing_ok=True)
            self.revision += 1


def reference_stats(f0: np.ndarray) -> tuple[F0Stats, float]:
    """Register of a clip of the original voice."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_seconds = float(np.sum(f0 > 0)) / FRAMES_PER_SECOND
    stats = compute_f0_stats([f0])
    if stats is None or voiced_seconds < MIN_VOICED_SECONDS / 2:
        raise ProfileError(f"Only {voiced_seconds:.0f} s of voice were found in the clip; use one with at least "
                           f"{MIN_VOICED_SECONDS / 2:.0f} s of that voice speaking.")
    return stats, voiced_seconds


def _write_json(path: Path, data: dict) -> None:
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
    partial.replace(path)
