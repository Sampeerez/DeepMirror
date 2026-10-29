"""The face clone, frame by frame.

* Faces are found on every frame (SCRFD on the CPU) and their five points are
  smoothed with a One-Euro filter: no jitter when still, no lag when moving.
* The swapper (the expensive part, on the GPU) runs on *keyframes*, as often
  as the GPU allows next to the voice. Its result is kept in the aligned,
  pose-free crop space.
* Every frame is composed from the two keyframes around it (cross-faded by
  time) placed with *that frame's* own points, so head movement is exact at
  the camera's rate and expressions move smoothly between keyframes. The
  video is delayed to match the voice anyway, so the next keyframe is usually
  already there.
* Optionally the speaker's own mouth stays (every syllable, at camera rate).
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.compute_arbiter import GpuArbiter
from backend.video.blend import Occluder, box_mask, mouth_mask, paste
from backend.video.faces import Face, FaceDetector, FaceLandmarker, FaceRecognizer, alignment, main_face
from backend.video.swap import SWAPPERS, FaceEnhancer, FaceSwapper, SwapperSpec, match_colour

logger = logging.getLogger(__name__)


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) on an array, with time in seconds."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 14.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None

    def __call__(self, x: np.ndarray, t: float, size: float = 1.0) -> np.ndarray:
        """Filtered ``x`` at time ``t``; ``size`` (the face's, in px) makes speed relative: faces per second."""
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        assert self._dx is not None
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        # Speed in face sizes per second: the same filter for a close or a far face.
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx) / max(size, 1e-6)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x.copy()


@dataclass(slots=True)
class Key:
    """A swapped face, kept in the aligned crop space of its swapper."""

    t: float
    spec: SwapperSpec
    face: np.ndarray            # float32 crop, BGR
    mask: np.ndarray            # float32 0..1


@dataclass(slots=True)
class VideoSettings:
    swapper: str = "auto"       # "auto" or a SWAPPERS id
    enhance: str = "auto"       # auto | on | off
    mouth: str = "auto"         # auto | on | off: keep the speaker's own mouth
    occlusion: bool = True      # hands, hair, glasses in front of the face stay in front

    def apply(self, payload: dict) -> None:
        if payload.get("swapper") == "auto" or payload.get("swapper") in SWAPPERS:
            self.swapper = payload["swapper"]
        for name in ("enhance", "mouth"):
            if payload.get(name) in ("auto", "on", "off"):
                setattr(self, name, payload[name])
        if isinstance(payload.get("occlusion"), bool):
            self.occlusion = payload["occlusion"]


class FaceModels:
    """Loads the face models on first use and shares them between sessions."""

    def __init__(self, directory: Path, gpu: bool) -> None:
        self.directory = directory
        self.gpu = gpu
        self._swappers: dict[str, FaceSwapper] = {}
        self._detector: FaceDetector | None = None
        self._landmarker: FaceLandmarker | None = None
        self._recognizer: FaceRecognizer | None = None
        self._enhancer: FaceEnhancer | None = None
        self._occluder: Occluder | None = None
        self.swap_ms: dict[str, float] = {}     # measured, per swapper

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector(self.directory / "scrfd_2.5g.onnx")
        return self._detector

    @property
    def landmarker(self) -> FaceLandmarker:
        if self._landmarker is None:
            self._landmarker = FaceLandmarker(self.directory / "fan_68_5.onnx")
        return self._landmarker

    @property
    def recognizer(self) -> FaceRecognizer:
        if self._recognizer is None:
            self._recognizer = FaceRecognizer(self.directory / "arcface_w600k_r50.onnx", self.gpu)
        return self._recognizer

    @property
    def enhancer(self) -> FaceEnhancer:
        if self._enhancer is None:
            self._enhancer = FaceEnhancer(self.directory / "gpen_bfr_256.onnx", self.gpu)
        return self._enhancer

    @property
    def occluder(self) -> Occluder:
        if self._occluder is None:
            self._occluder = Occluder(self.directory / "xseg_1.onnx", self.gpu)
        return self._occluder

    def keep_only(self, swapper_id: str) -> None:
        """Free the swappers not in use (each holds hundreds of MB of GPU memory)."""
        for sid in [s for s in self._swappers if s != swapper_id]:
            del self._swappers[sid]

    def measure(self, cache: Path | None = None, arbiter: GpuArbiter | None = None) -> dict[str, float]:
        """Time every available swapper once (cached per GPU), then free them: the choice needs the numbers."""
        import json

        from backend.video.runtime import gpu_providers

        key = f"{gpu_providers()[:1]}"
        try:
            import torch

            if torch.cuda.is_available():
                key += torch.cuda.get_device_name(0)
        except Exception:
            pass
        stored: dict = {}
        if cache is not None and cache.is_file():
            try:
                stored = json.loads(cache.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = {}
        known = stored.get(key, {})
        for sid in self.available():
            if sid in known:
                self.swap_ms[sid] = float(known[sid])
            elif sid not in self.swap_ms:
                self.swapper(sid, arbiter)
                self.keep_only("")
        if cache is not None:
            stored[key] = {**known, **self.swap_ms}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(stored, indent=1), encoding="utf-8")
        return dict(self.swap_ms)

    def available(self) -> list[str]:
        return [sid for sid, spec in SWAPPERS.items() if (self.directory / spec.file).is_file()]

    def swapper(self, swapper_id: str, arbiter: GpuArbiter | None = None) -> FaceSwapper:
        """The swapper, loaded and warmed up on first use (with ``arbiter``: each warm-up run in a voice gap)."""
        if swapper_id not in self._swappers:
            swapper = FaceSwapper(SWAPPERS[swapper_id], self.directory, self.gpu)
            # Warm-up: the first runs pick the CUDA kernels; time the steady state.
            frame = np.full((256, 256, 3), 128, np.uint8)
            points = np.array([[96, 110], [160, 110], [128, 145], [100, 180], [156, 180]], np.float32)
            identity = swapper.identity(np.ones(512, np.float32) / math.sqrt(512))
            times = []
            for _ in range(4):
                guess = self.swap_ms.get(swapper_id, 60.0) * (3 if not times else 1)
                with arbiter.video(guess, timeout=10.0) if arbiter else contextlib.nullcontext():
                    started = time.perf_counter()
                    swapper.swap(frame, points, identity)
                    times.append((time.perf_counter() - started) * 1000)
            self.swap_ms[swapper_id] = round(min(times[1:]), 1)
            logger.info("Face swapper %s ready: %.0f ms per face", swapper_id, self.swap_ms[swapper_id])
            self._swappers[swapper_id] = swapper
        return self._swappers[swapper_id]

    def source_embedding(self, images: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
        """The identity to wear, averaged over several photos (steadier), and a note on each photo."""
        vectors, notes = [], []
        for image in images:
            faces = self.detector.detect(image)
            face = main_face(faces)
            if face is None:
                notes.append({"face": False})
                continue
            vectors.append(self.recognizer.embedding(image, face.points))
            sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            notes.append({"face": True, "size": round(face.size), "faces": len(faces), "sharpness": round(sharpness)})
        if not vectors:
            raise ValueError("No face found in the photos")
        mean = np.mean(vectors, axis=0)
        return (mean / np.linalg.norm(mean)).astype(np.float32), notes


def keys_per_second(cost_ms: float, gap: tuple[float, float] | None) -> float:
    """Keyframes a swapper of ``cost_ms`` can make: all the GPU, or whole swaps per gap between voice renders.

    ``gap`` is ``(gap ms, voice period ms)``; None when the voice is not streaming.
    """
    if cost_ms <= 0:
        return 0.0
    if gap is None:
        return 1000.0 / cost_ms
    free_ms, period_ms = gap
    return (free_ms // (cost_ms + 6.0)) * 1000.0 / max(period_ms, 1.0)


def choose_swapper(swap_ms: dict[str, float], available: list[str], setting: str,
                   gap: tuple[float, float] | None) -> str:
    """The best swapper that still fits the GPU time the voice leaves (at least one swap per gap).

    Every model that fits makes about as many keyframes per gap, so the best one wins; without
    the voice, the best one that gives a fluid rate (MIN_KEYS_PER_SECOND).
    """
    if setting != "auto" and setting in available:
        return setting
    ranked = sorted(available, key=lambda sid: -SWAPPERS[sid].quality)
    needed = MIN_KEYS_PER_SECOND if gap is None else 1.0
    for sid in ranked:
        cost = swap_ms.get(sid)
        if cost and keys_per_second(cost, gap) >= needed:
            return sid
    return min(ranked, key=lambda sid: swap_ms.get(sid, 1e9))


# Without the voice, the automatic choice aims for this many keyframes a second.
MIN_KEYS_PER_SECOND = 12.0


@dataclass(slots=True)
class Tracked:
    face: Face | None
    points: np.ndarray | None


@dataclass
class FaceEngine:
    models: FaceModels
    arbiter: GpuArbiter
    settings: VideoSettings = field(default_factory=VideoSettings)
    identity: np.ndarray | None = None       # ArcFace embedding of the face to wear
    _filter: OneEuro = field(default_factory=OneEuro)
    _last: Face | None = None
    _missed: int = 0
    _swapper_id: str | None = None
    _conditioning: np.ndarray | None = None
    _swap_ms: float = 0.0     # measured while running (the GPU is shared, so it can exceed the warm-up time)

    def set_identity(self, embedding: np.ndarray | None) -> None:
        self.identity = embedding
        self._conditioning = None

    # --- Every frame (CPU) -------------------------------------------------

    def track(self, frame: np.ndarray, t: float) -> Tracked:
        """The face to follow and its smoothed points; a missed detection keeps the last face briefly."""
        detector = self.models.detector
        face = main_face(detector.detect_near(frame, self._last), self._last) if self._last is not None else None
        if face is None:
            face = main_face(detector.detect(frame), self._last)
        if face is None:
            self._missed += 1
            if self._last is not None and self._missed <= 4:
                return Tracked(self._last, self._last.points)
            self._last = None
            self._filter.reset()
            return Tracked(None, None)
        self._missed = 0
        smoothed = self._filter(face.points.reshape(-1), t, face.size).reshape(5, 2)
        face.points = smoothed.astype(np.float32)
        self._last = face
        return Tracked(face, face.points)

    # --- Keyframes (GPU) ---------------------------------------------------

    def plan(self) -> str:
        """Pick the swapper for the GPU time the voice leaves (at start and now and then, never per frame)."""
        sid = choose_swapper(self.models.swap_ms, self.models.available(), self.settings.swapper,
                             self.arbiter.gap())
        if sid != self._swapper_id:
            self.models.keep_only(sid)
            self.models.swapper(sid, self.arbiter)   # warms up in voice gaps, never over a chunk
            self._swapper_id, self._conditioning, self._swap_ms = sid, None, 0.0
            logger.info("Face clone uses %s", SWAPPERS[sid].label)
        return sid

    def swapper_for(self) -> FaceSwapper:
        return self.models.swapper(self._swapper_id or self.plan())

    def keyframe(self, frame: np.ndarray, tracked: Tracked, t: float, keys_per_second: float) -> Key | None:
        """Swap the face of this frame; None when the GPU had no room in time (the voice comes first)."""
        if self.identity is None or tracked.face is None or tracked.points is None:
            return None
        swapper = self.swapper_for()
        if self._conditioning is None:
            self._conditioning = swapper.identity(self.identity)
        expected = max(self.models.swap_ms.get(swapper.spec.id, 60.0), self._swap_ms)
        with self.arbiter.video(expected) as granted:
            if not granted:
                return None
            started = time.perf_counter()
            swapped, original, _, model_mask = swapper.swap(frame, tracked.points, self._conditioning)
            took = (time.perf_counter() - started) * 1000
        self._swap_ms = took if self._swap_ms == 0 else 0.8 * self._swap_ms + 0.2 * took
        size = swapper.spec.size
        mask = box_mask(size, 0.3).copy()
        if model_mask is not None:   # HyperSwap's own face mask: hands and hair in front stay
            mask = np.minimum(mask, cv2.GaussianBlur(model_mask, (0, 0), size / 128))
        elif self.settings.occlusion:
            with self.arbiter.video(30.0, timeout=0.0) as granted:   # only when the GPU has room
                if granted:
                    mask = np.minimum(mask, self.models.occluder.mask(original))
        swapped = match_colour(swapped, original, mask)
        enhance = self.settings.enhance == "on" or (
            self.settings.enhance == "auto" and tracked.face.size >= 220 and keys_per_second >= 14)
        if enhance:
            with self.arbiter.video(35.0, timeout=0.02) as granted:
                if granted:
                    swapped = self.models.enhancer.enhance(swapped)
        return Key(t, swapper.spec, swapped.astype(np.float32), mask)

    # --- Composition (CPU) -------------------------------------------------

    def compose(self, frame: np.ndarray, tracked: Tracked, t: float, before: Key | None, after: Key | None,
                keys_per_second: float) -> np.ndarray:
        """``frame`` wearing the face, interpolated between the keyframes around ``t``."""
        if tracked.points is None:
            return frame
        keys = [k for k in (before, after) if k is not None]
        if not keys:
            return frame
        if len(keys) == 2 and keys[0].spec.id == keys[1].spec.id and keys[1].t > keys[0].t:
            a = float(np.clip((t - keys[0].t) / (keys[1].t - keys[0].t), 0, 1))
            face = keys[0].face * (1 - a) + keys[1].face * a
            mask = keys[0].mask * (1 - a) + keys[1].mask * a
            spec = keys[0].spec
        else:
            nearest = min(keys, key=lambda k: abs(k.t - t))
            face, mask, spec = nearest.face, nearest.mask, nearest.spec
        matrix = alignment(tracked.points, spec.template, spec.size)
        mouth = self.settings.mouth == "on" or (self.settings.mouth == "auto" and keys_per_second < 12)
        if mouth:
            if tracked.face is not None and tracked.face.landmarks is None:
                tracked.face.landmarks = self.models.landmarker.landmarks(tracked.points)
            if tracked.face is not None and tracked.face.landmarks is not None:
                live = cv2.warpAffine(frame, matrix, (spec.size, spec.size), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                lips = mouth_mask(tracked.face.landmarks, matrix, spec.size)[:, :, None]
                face = face * (1 - lips) + live * lips
        return paste(frame, np.clip(face, 0, 255).astype(np.uint8), mask, matrix)
