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
* Whole head (the default): the photo's head (hair, beard, ears, outline),
  rendered in the speaker's pose a few times a second (``head.py``), is placed
  with every frame's own points; the swap on top keeps eyes, brows, nose and
  mouth live. Masks follow the parsed face (BiSeNet), never a soft box, so there
  is no halo.
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
from backend.video import head as heads
from backend.video import parser as parsing
from backend.video.blend import Occluder, box_mask, mouth_mask, paste
from backend.video.faces import Face, FaceDetector, FaceLandmarker, FaceRecognizer, alignment, main_face
from backend.video.parser import FaceParser, feather, region
from backend.video.swap import SWAPPERS, FaceEnhancer, FaceSwapper, SwapperSpec, match_colour

logger = logging.getLogger(__name__)

# A head render goes on only in gaps with room for a few of its stages: with the voice talking
# non-stop on a busy GPU, the last head stays (moving with the speaker) until a pause.
HEAD_MIN_ROOM_MS = 60.0


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
class HeadLayer:
    """The worn head (with the speaker's uncovered hair filled in), in the head crop of its frame."""

    t: float
    image: np.ndarray           # float32 OUTPUT x OUTPUT x 3, BGR
    alpha: np.ndarray           # float32 0..1
    face: np.ndarray            # float32 0..1: the worn face's skin, eyes, nose, mouth


@dataclass(slots=True)
class VideoSettings:
    mode: str = "head"          # head: the whole head (hair, beard, outline); face: the inside of the face
    swapper: str = "auto"       # "auto" or a SWAPPERS id
    enhance: str = "auto"       # auto | on | off
    mouth: str = "auto"         # auto | on | off: keep the speaker's own mouth
    occlusion: bool = True      # hands, hair, glasses in front of the face stay in front

    def apply(self, payload: dict) -> None:
        if payload.get("mode") in ("head", "face"):
            self.mode = payload["mode"]
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
        self._parser: FaceParser | None = None
        self._heads: heads.HeadModels | None = None
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

    @property
    def parser(self) -> FaceParser | None:
        path = self.directory / "bisenet_resnet_34.onnx"
        if self._parser is None and path.is_file():
            self._parser = FaceParser(path, self.gpu)
        return self._parser

    @property
    def heads_ready(self) -> bool:
        return heads.ready(self.directory)

    def head_models(self) -> heads.HeadModels | None:
        """LivePortrait, loaded the first time the whole head is used."""
        if self._heads is None and self.heads_ready:
            self._heads = heads.HeadModels(self.directory, self.gpu)
        return self._heads

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
    portrait: np.ndarray | None = None       # the photo the whole head comes from
    _head: heads.HeadRenderer | None = None
    _head_for: int | None = None             # id() of the portrait the renderer was prepared with
    _stage_ms: list[float] = field(default_factory=list)   # measured cost of each head render stage
    _parse_ms: float = 45.0   # one face parse on the GPU (measured as it runs)

    def set_identity(self, embedding: np.ndarray | None) -> None:
        self.identity = embedding
        self._conditioning = None

    def set_portrait(self, image: np.ndarray | None) -> None:
        self.portrait = image

    @property
    def wants_head(self) -> bool:
        return (self.settings.mode == "head" and self.portrait is not None and self.identity is not None
                and self.models.heads_ready)

    def reset_head(self) -> None:
        """The next frame becomes the rest pose the head moves from (a new session)."""
        if self._head is not None:
            self._head.reset()

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

    def plan(self, running: bool = False) -> str:
        """Pick the swapper for the GPU time the voice leaves (at start and now and then, never per frame).

        While the camera runs (``running``), loading another swapper stalls the face process for
        seconds, so it is only worth it when the current one cannot swap at all next to the voice
        and the other one can; a setting the user changed always applies.
        """
        gap = self.arbiter.gap()
        sid = choose_swapper(self.models.swap_ms, self.models.available(), self.settings.swapper, gap)
        current = self._swapper_id
        if running and current and sid != current and self.settings.swapper == "auto":
            fits = lambda swapper: keys_per_second(self.models.swap_ms.get(swapper, 0.0), gap) >= 1.0  # noqa: E731
            if fits(current) or not fits(sid):
                return current
        if sid != self._swapper_id:
            self.models.keep_only(sid)
            self.models.swapper(sid, self.arbiter)   # warms up in voice gaps, never over a chunk
            self._swapper_id, self._conditioning, self._swap_ms = sid, None, 0.0
            logger.info("Face clone uses %s", SWAPPERS[sid].label)
        return sid

    def swapper_for(self) -> FaceSwapper:
        return self.models.swapper(self._swapper_id or self.plan())

    def keyframe(self, frame: np.ndarray, tracked: Tracked, t: float, keys_per_second: float,
                 wait: float = 0.5) -> Key | None:
        """Swap the face of this frame; None when the GPU had no room in time (the voice comes first)."""
        if self.identity is None or tracked.face is None or tracked.points is None:
            return None
        swapper = self.swapper_for()
        if self._conditioning is None:
            self._conditioning = swapper.identity(self.identity)
        expected = max(self.models.swap_ms.get(swapper.spec.id, 60.0), self._swap_ms)
        with self.arbiter.video(expected, timeout=wait) as granted:
            if not granted:
                return None
            started = time.perf_counter()
            swapped, original, _, model_mask = swapper.swap(frame, tracked.points, self._conditioning)
            took = (time.perf_counter() - started) * 1000
        self._swap_ms = took if self._swap_ms == 0 else 0.8 * self._swap_ms + 0.2 * took
        size = swapper.spec.size
        head = self.wants_head
        mask = self._face_mask(tracked, alignment(tracked.points, swapper.spec.template, size), size, head)
        if model_mask is not None:   # HyperSwap's own face mask: hands and hair in front stay
            mask = np.minimum(mask, cv2.GaussianBlur(model_mask, (0, 0), size / 128))
        elif self.settings.occlusion:
            with self.arbiter.video(30.0, timeout=0.0) as granted:   # only when the GPU has room
                if granted:
                    mask = np.minimum(mask, self.models.occluder.mask(original))
        if not head:   # the whole head brings its own skin: the face is matched to it when composed
            swapped = match_colour(swapped, original, mask)
        enhance = self.settings.enhance == "on" or (
            self.settings.enhance == "auto" and tracked.face.size >= 220 and keys_per_second >= 14)
        if enhance:
            with self.arbiter.video(35.0, timeout=0.02) as granted:
                if granted:
                    # Mostly the enhanced face, a little of the swap's own texture (no plastic look).
                    swapped = cv2.addWeighted(self.models.enhancer.enhance(swapped), 0.75, swapped, 0.25, 0)
        return Key(t, swapper.spec, swapped.astype(np.float32), mask)

    def _face_mask(self, tracked: Tracked, matrix: np.ndarray, size: int, head: bool) -> np.ndarray:
        """Where the swap goes, in its crop, from the 68 landmarks (CPU, no GPU time): the whole
        face up to the forehead, or with the whole head only brows, eyes, nose and mouth. Pulled
        inside the outline and faded, so no ring shows around it.
        """
        face = tracked.face
        if face is not None and face.landmarks is None and tracked.points is not None:
            face.landmarks = self.models.landmarker.landmarks(tracked.points)
        if face is None or face.landmarks is None:
            return box_mask(size, 0.3).copy()
        points = cv2.transform(face.landmarks.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
        eyes = np.linalg.norm(points[36:42].mean(0) - points[42:48].mean(0)) or 1.0
        mask = np.zeros((size, size), np.float32)
        if head:
            features = points[17:68]
            cv2.fillConvexPoly(mask, cv2.convexHull(features.astype(np.int32)), 1.0)
            grow = max(3, int(eyes * 0.25)) | 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
            return feather(mask, erode=0.0, soft=0.07)
        # The jaw line and the brows raised by about a third of the eye distance (the forehead).
        brows = points[17:27] - np.array([0.0, eyes * 0.45], np.float32)
        outline = np.vstack([points[0:17], brows[::-1]])
        cv2.fillConvexPoly(mask, cv2.convexHull(outline.astype(np.int32)), 1.0)
        return feather(mask, erode=0.025, soft=0.06)

    # --- The whole head (GPU, in stages) -----------------------------------

    def _prepare_head(self) -> heads.HeadRenderer | None:
        if self.portrait is None:
            return None
        if self._head is None:
            models = self.models.head_models()
            if models is None:
                return None
            self._head = heads.HeadRenderer(models)
        if self._head_for != id(self.portrait):
            found = main_face(self.models.detector.detect(self.portrait))
            if found is None:
                logger.warning("No face found in the photo for the whole head")
                return None
            with self.arbiter.video(60.0, timeout=5.0) as granted:
                if not granted:
                    return None
                self._head.set_source(self.portrait, found.points)
            self._head_for = id(self.portrait)
        return self._head

    def head_layer(self, frame: np.ndarray, tracked: Tracked, t: float, wait: float = 2.0) -> HeadLayer | None:
        """Render the worn head in this frame's pose, a stage per GPU gap; None if the GPU stayed busy."""
        if tracked.points is None:
            return None
        renderer = self._prepare_head()
        if renderer is None:
            return None
        points = tracked.points.copy()
        steps = list(renderer.stages(frame, points))
        index = 0
        deadline = time.monotonic() + wait
        while index < len(steps):
            expected = self._expected_stage(index)
            # Short attempts, so the face swap (and the frames) are never held up behind the head;
            # only gaps with room for a few stages are worth starting in.
            if self.arbiter.room_ms() < max(expected, HEAD_MIN_ROOM_MS):
                if time.monotonic() > deadline:
                    return None
                time.sleep(0.01)
                continue
            with self.arbiter.video(expected, timeout=0.02) as granted:
                if not granted:
                    continue
                # As many stages as fit before the voice's next render (or all, with no voice).
                while index < len(steps):
                    started = time.perf_counter()
                    steps[index]()
                    self._measured_stage(index, (time.perf_counter() - started) * 1000)
                    index += 1
                    if index >= len(steps) or self.arbiter.room_ms() < self._expected_stage(index):
                        break
        if renderer.result is None:
            return None
        return self._layer(frame, points, renderer.result, t, wait)

    def _expected_stage(self, index: int) -> float:
        return self._stage_ms[index] if index < len(self._stage_ms) else 35.0

    def _measured_stage(self, index: int, took: float) -> None:
        if index < len(self._stage_ms):
            self._stage_ms[index] = 0.8 * self._stage_ms[index] + 0.2 * took
        else:
            self._stage_ms.append(took)

    def _layer(self, frame: np.ndarray, points: np.ndarray, rendered: np.ndarray, t: float,
               wait: float) -> HeadLayer | None:
        """Cut the rendered head out (hair, ears, neck fading out) and fill in the speaker's own
        hair that the new head does not cover, from the background around it."""
        size = heads.OUTPUT
        matrix = alignment(points, heads.HEAD_TEMPLATE, size)
        plate = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        parser = self.models.parser
        labels_new = labels_own = None
        if parser is not None:
            with self.arbiter.video(self._parse_ms, timeout=wait) as granted:
                if granted:
                    started = time.perf_counter()
                    labels_new = parser.labels(rendered)
                    self._parse_ms = 0.8 * self._parse_ms + 0.2 * (time.perf_counter() - started) * 1000
            with self.arbiter.video(self._parse_ms, timeout=wait) as granted:
                if granted:
                    labels_own = parser.labels(plate)
        if labels_new is None:
            return None
        neck = (parsing.NECK,)
        worn = region(labels_new, parsing.HEAD + neck)
        # The neck fades out towards the bottom of the crop, into the speaker's own.
        fade = np.clip((np.arange(size, dtype=np.float32) / size - 0.80) / 0.15, 0, 1)[:, None]
        worn_neck = region(labels_new, neck) * fade
        alpha = feather(np.clip(worn - worn_neck, 0, 1), erode=0.004, soft=0.02)
        fill = np.zeros((size, size), np.float32)
        if labels_own is not None:
            own = region(labels_own, parsing.HEAD)
            grow = max(3, int(size * 0.03)) | 1
            covered = cv2.dilate(worn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
            uncovered = np.clip(own - covered, 0, 1)
            if uncovered.sum() > size:
                small = cv2.resize(plate, (size // 2, size // 2), interpolation=cv2.INTER_AREA)
                hole = cv2.resize(cv2.dilate(uncovered, np.ones((5, 5), np.uint8)), (size // 2, size // 2))
                filled = cv2.inpaint(small, (hole > 0.5).astype(np.uint8), 6, cv2.INPAINT_TELEA)
                filled = cv2.resize(filled, (size, size), interpolation=cv2.INTER_LINEAR)
                fill = feather(cv2.dilate(uncovered, np.ones((7, 7), np.uint8)), erode=0.0, soft=0.02)
                plate = (plate * (1 - fill[:, :, None]) + filled * fill[:, :, None]).astype(np.uint8)
        # The photo's own lighting moves half way towards the room's (the head stays the photo's).
        rendered = _light_like(rendered, plate, worn)
        image = rendered.astype(np.float32) * alpha[:, :, None] + plate.astype(np.float32) * (1 - alpha[:, :, None])
        face = feather(region(labels_new, parsing.FACE), erode=0.01, soft=0.03)
        return HeadLayer(t, image, np.maximum(alpha, fill), face)


    # --- Composition (CPU) -------------------------------------------------

    def compose(self, frame: np.ndarray, tracked: Tracked, t: float, before: Key | None, after: Key | None,
                keys_per_second: float, head: HeadLayer | None = None) -> np.ndarray:
        """``frame`` wearing the face (and the head, when given), between the keyframes around ``t``."""
        if tracked.points is None:
            return frame
        if head is not None and self.settings.mode == "head":
            matrix = alignment(tracked.points, heads.HEAD_TEMPLATE, heads.OUTPUT)
            frame = paste(frame, np.clip(head.image, 0, 255).astype(np.uint8), head.alpha, matrix)
        else:
            head = None
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
        face = np.clip(face, 0, 255).astype(np.uint8)
        if head is not None:
            # Only inside the worn face, coloured like the worn head's skin around it.
            worn = cv2.warpAffine(head.face, matrix @ np.vstack([cv2.invertAffineTransform(
                alignment(tracked.points, heads.HEAD_TEMPLATE, heads.OUTPUT)), [0, 0, 1]]),
                (spec.size, spec.size), flags=cv2.INTER_LINEAR, borderValue=0.0)
            mask = mask * worn
            under = cv2.warpAffine(frame, matrix, (spec.size, spec.size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            face = match_colour(face, under, mask)
        return paste(frame, face, mask, matrix)


def _light_like(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``image``'s brightness moved half way towards ``reference``'s (inside ``mask``); colours kept."""
    weight = mask > 0.5
    if weight.sum() < 64:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = 0.5 * (ref[..., 0][weight].mean() - lab[..., 0][weight].mean())
    lab[..., 0] = np.clip(lab[..., 0] + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
