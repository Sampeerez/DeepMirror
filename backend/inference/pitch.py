"""Pitch (F0) tracking: method catalogue, availability and extraction.

Every tracker returns F0 in Hz at 100 frames/s (one frame per 160 samples of
16 kHz audio), aligned with the HuBERT/synthesizer frame grid. ``to_rvc_pitch``
then applies the pitch shift and quantises to the 255 coarse pitch bins the
synthesizer's pitch embedding expects.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
HOP = 160
F0_MIN = 50.0
F0_MAX = 1100.0
DEFAULT_F0_METHOD = "pm"
_MEL_MIN = 1127.0 * math.log(1.0 + F0_MIN / 700.0)
_MEL_MAX = 1127.0 * math.log(1.0 + F0_MAX / 700.0)


@dataclass(frozen=True, slots=True)
class PitchMethod:
    id: str
    label: str
    description: str
    section: str
    realtime: bool = True
    offline: bool = True
    dependency: str | None = None   # importable module that must be installed
    needs_rmvpe: bool = False       # needs the downloaded RMVPE weights
    components: tuple[str, ...] = ()


PITCH_METHODS = (
    PitchMethod("pm", "PM", "Lowest latency CPU tracking", "Classic", dependency="parselmouth"),
    PitchMethod("harvest", "Harvest", "Stable CPU tracking", "Classic", dependency="pyworld"),
    PitchMethod("dio", "DIO", "Lightweight CPU tracking", "Classic", dependency="pyworld"),
    PitchMethod("crepe", "Crepe Full", "Accurate neural tracking", "Neural", dependency="torchcrepe"),
    PitchMethod("crepe-tiny", "Crepe Tiny", "Lower-load neural tracking", "Neural", dependency="torchcrepe"),
    PitchMethod("mangio-crepe", "Mangio-Crepe", "Crepe with adjustable hop length", "Neural",
                dependency="torchcrepe"),
    PitchMethod("mangio-crepe-tiny", "Mangio-Crepe Tiny", "Light Crepe with adjustable hop", "Neural",
                dependency="torchcrepe"),
    PitchMethod("rmvpe", "RMVPE", "Recommended all-round neural tracking", "Neural", needs_rmvpe=True),
    PitchMethod("fcpe", "FCPE", "Fast neural tracking", "Neural", dependency="torchfcpe"),
    PitchMethod("hybrid[crepe+rmvpe]", "Hybrid: Crepe + RMVPE", "Median of two estimators", "Hybrid",
                realtime=False, components=("crepe", "rmvpe")),
    PitchMethod("hybrid[crepe+fcpe]", "Hybrid: Crepe + FCPE", "Median of two estimators", "Hybrid",
                realtime=False, components=("crepe", "fcpe")),
    PitchMethod("hybrid[rmvpe+fcpe]", "Hybrid: RMVPE + FCPE", "Median of two estimators", "Hybrid",
                realtime=False, components=("rmvpe", "fcpe")),
    PitchMethod("hybrid[crepe+rmvpe+fcpe]", "Hybrid: Crepe + RMVPE + FCPE", "Median of three estimators",
                "Hybrid", realtime=False, components=("crepe", "rmvpe", "fcpe")),
)
PITCH_METHOD_BY_ID = {method.id: method for method in PITCH_METHODS}
_ALIASES = {"crepe_tiny": "crepe-tiny", "mangio_crepe": "mangio-crepe", "mangio_crepe_tiny": "mangio-crepe-tiny"}


def normalize_method(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("F0 method must be a string")
    method = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if method not in PITCH_METHOD_BY_ID:
        raise ValueError(f"Unsupported F0 method: {value}")
    return method


@cache
def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def method_capabilities(rmvpe_path: Path) -> list[dict]:
    """Every method with ``available``/``reason``, for the UI and validation."""
    reasons: dict[str, str | None] = {}
    for method in PITCH_METHODS:
        reason = None
        if method.dependency and not _module_available(method.dependency):
            reason = f"Install the {'classic-f0' if method.dependency == 'pyworld' else 'neural-f0'} extra"
        elif method.needs_rmvpe and not rmvpe_path.is_file():
            reason = "RMVPE weights are not downloaded yet"
        reasons[method.id] = reason
    for method in PITCH_METHODS:
        if method.components:
            missing = [PITCH_METHOD_BY_ID[c].label for c in method.components if reasons[c]]
            reasons[method.id] = f"Requires {', '.join(missing)}" if missing else None
    return [{**asdict(method), "components": list(method.components),
             "available": reasons[method.id] is None, "reason": reasons[method.id]} for method in PITCH_METHODS]


def fit_frames(values: np.ndarray, frames: int) -> np.ndarray:
    """Centre-pad or truncate an F0 track to ``frames`` (upstream RVC alignment)."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(values) >= frames:
        return values[:frames]
    before = (frames - len(values) + 1) // 2
    return np.pad(values, (before, frames - len(values) - before))


def suppress_octave_spikes(f0: np.ndarray) -> np.ndarray:
    """Replace isolated single-frame octave errors with their neighbours' pitch.

    A frame is corrected only when both neighbours are voiced, agree within
    ~4 semitones, and the frame sits more than ~11 semitones away from them;
    genuine glides and voiced/unvoiced edges are left alone.
    """
    if len(f0) < 3:
        return f0
    prev, mid, nxt = f0[:-2], f0[1:-1], f0[2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = np.sqrt(prev * nxt)
        neighbours_agree = np.abs(np.log2(prev / nxt)) < 1.0 / 3.0
        deviation = np.abs(np.log2(mid / reference))
    spike = (prev > 0) & (nxt > 0) & (mid > 0) & neighbours_agree & (deviation > 0.9)
    if not spike.any():
        return f0
    fixed = f0.copy()
    fixed[1:-1][spike] = reference[spike]
    return fixed


def to_rvc_pitch(f0_hz: np.ndarray, semitones: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift by ``semitones`` and return ``(coarse_bins int64, f0_hz float32)``."""
    f0 = np.asarray(f0_hz, dtype=np.float64) * 2.0 ** (float(semitones) / 12.0)
    mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - _MEL_MIN) * 254.0 / (_MEL_MAX - _MEL_MIN) + 1.0
    mel = np.clip(mel, 1.0, 255.0)
    return np.rint(mel).astype(np.int64), f0.astype(np.float32)


class PitchExtractor:
    """Runs one pitch tracker on 16 kHz audio; heavy models load on first use."""

    def __init__(self, device: torch.device, half: bool, rmvpe_path: Path) -> None:
        self.device = device
        self.half = half
        self.rmvpe_path = rmvpe_path
        self._rmvpe = None
        self._fcpe = None
        self._ready: set[str] = {"pm", "harvest", "dio"}   # no model to load
        self._load_lock = threading.Lock()

    def close(self) -> None:
        self._rmvpe = None
        self._fcpe = None
        self._ready = {"pm", "harvest", "dio"}

    def ready(self, method: str) -> bool:
        """Whether ``method`` runs without first loading a model (and its first-call setup)."""
        info = PITCH_METHOD_BY_ID.get(method)
        if info is None:
            return False
        if info.components:
            return all(self.ready(component) for component in info.components)
        return method in self._ready

    def prepare(self, method: str) -> None:
        """Load ``method``'s model and run it once, so its first live use is not slow."""
        if self.ready(method):
            return
        with self._load_lock:
            if not self.ready(method):
                t = np.arange(8000) / SAMPLE_RATE
                probe = (0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
                self.estimate(method, probe, len(probe) // HOP)

    def estimate(self, method: str, audio: np.ndarray, frames: int, *, filter_radius: int = 3,
                 crepe_hop_length: int = 160, rmvpe_threshold: float = 0.03) -> np.ndarray:
        """F0 in Hz, exactly ``frames`` long."""
        info = PITCH_METHOD_BY_ID[method]
        if info.components:
            tracks = [self.estimate(c, audio, frames, filter_radius=filter_radius,
                                    crepe_hop_length=crepe_hop_length, rmvpe_threshold=rmvpe_threshold)
                      for c in info.components]
            stack = np.stack([np.where(t > 0, t, np.nan) for t in tracks])
            with np.errstate(all="ignore"):
                return np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
        trackers: dict[str, Callable[[], np.ndarray]] = {
            "pm": lambda: self._pm(audio, frames),
            "harvest": lambda: self._world(audio, frames, filter_radius, harvest=True),
            "dio": lambda: self._world(audio, frames, filter_radius, harvest=False),
            "rmvpe": lambda: self._rmvpe_model().infer(audio, threshold=rmvpe_threshold),
            "fcpe": lambda: self._fcpe_track(audio, frames),
        }
        if method in trackers:
            f0 = trackers[method]()
        else:
            f0 = self._crepe(audio, frames, method, crepe_hop_length)
        self._ready.add(method)
        return suppress_octave_spikes(fit_frames(f0, frames))

    @staticmethod
    def _pm(audio: np.ndarray, frames: int) -> np.ndarray:
        import parselmouth

        pitch = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE).to_pitch_ac(
            time_step=HOP / SAMPLE_RATE, voicing_threshold=0.6, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        return fit_frames(pitch.selected_array["frequency"], frames)

    @staticmethod
    def _world(audio: np.ndarray, frames: int, filter_radius: int, *, harvest: bool) -> np.ndarray:
        pyworld: Any = importlib.import_module("pyworld")

        source = audio.astype(np.float64)
        track = pyworld.harvest if harvest else pyworld.dio
        options = {"f0_floor": F0_MIN, "f0_ceil": F0_MAX, "frame_period": 1000.0 * HOP / SAMPLE_RATE}
        if not harvest:
            options["channels_in_octave"] = 2
        rough, times = track(source, SAMPLE_RATE, **options)
        f0 = pyworld.stonemask(source, rough, times, SAMPLE_RATE)
        if filter_radius > 2 and len(f0) >= 3:
            f0 = signal.medfilt(f0, 3)
        return f0

    def _rmvpe_model(self):
        if self._rmvpe is None:
            from backend.inference.rmvpe import Rmvpe

            if not self.rmvpe_path.is_file():
                raise RuntimeError("RMVPE weights are not downloaded yet")
            self._rmvpe = Rmvpe(self.rmvpe_path, self.device, self.half)
        return self._rmvpe

    def _crepe(self, audio: np.ndarray, frames: int, method: str, hop_length: int) -> np.ndarray:
        import torchcrepe

        mangio = method.startswith("mangio-")
        source = audio.astype(np.float32).copy()
        if mangio:
            scale = float(np.quantile(np.abs(source), 0.999)) if source.size else 0.0
            if scale > 1e-7:
                source /= scale
        hop = int(np.clip(hop_length, 64, 512)) if mangio else HOP
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                torch.from_numpy(source)[None], SAMPLE_RATE, hop, F0_MIN, F0_MAX,
                "tiny" if method.endswith("tiny") else "full", batch_size=512, device=str(self.device),
                return_periodicity=True)
            periodicity = torchcrepe.filter.median(periodicity, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[periodicity < 0.1] = 0
        values = f0[0].float().cpu().numpy()
        if hop != HOP and len(values) > 1:
            values = np.interp(np.arange(frames) * float(HOP), np.arange(len(values)) * float(hop), values,
                               left=0.0, right=0.0)
        return values

    def _fcpe_track(self, audio: np.ndarray, frames: int) -> np.ndarray:
        import torchfcpe

        if self._fcpe is None:
            self._fcpe = torchfcpe.spawn_bundled_infer_model(device=str(self.device))
        with torch.no_grad():
            f0 = self._fcpe.infer(
                torch.from_numpy(audio.astype(np.float32))[None, :, None].to(self.device),
                sr=SAMPLE_RATE, decoder_mode="local_argmax", threshold=0.006, f0_min=F0_MIN, f0_max=F0_MAX,
                interp_uv=False, output_interp_target_length=frames)
        return f0.squeeze().float().cpu().numpy()
