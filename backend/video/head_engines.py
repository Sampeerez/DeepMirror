"""Which engine clones the head on this computer, best first: measured on its own GPU, never assumed.

personalive  diffusion: face and hair generated together (the most natural, the most expensive);
head         LivePortrait, a render per frame: face and hair warped together from the photo;
3d           the face swap dressed by a 3D head with hair physics;
hair         the face swap and the photo's hair.

"auto" takes the best one that fits; while it runs, ``EngineWatch`` goes down a level if the video
cannot keep up or the voice starts arriving late. The voice always comes first. An engine the
speaker chose by hand stays, whatever it costs.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AUTO = "auto"
LEVELS = ("personalive", "head", "3d", "hair")   # best first
HEAD_LIVE_MS = 40.0          # LivePortrait: 25 renders a second, the frames between take the nearest
PERSONALIVE_LIVE_MS = 83.0   # per frame: 12 fps, the least that still looks like motion

WATCH_SECONDS = 15.0         # how long a problem must last before going down a level
VIDEO_SHARE = 0.7            # of the camera's frame rate, below which the video does not keep up
LATE_SHARE = 0.15            # of the voice's chunks: more than this over real time is too many
CHUNKS_PER_SECOND = 5.0      # about, for that share
MEASURE_RENDERS = 5
FP16_FASTER = 0.9            # half precision must take less than this share of the time...
FP16_SAME = 3.0              # ...and differ by no more than this on average (0..255)


@dataclass(frozen=True, slots=True)
class Choice:
    engine: str
    reason: str


def fits(engine: str, measured: dict) -> tuple[bool, str]:
    """Whether ``engine``, installed, can run live on this GPU, and why."""
    if engine == "head":
        ms = measured.get("head_ms")
        if ms is None:
            return False, "not measured yet"
        return ms <= HEAD_LIVE_MS, f"{ms:.0f} ms per frame (fits at {HEAD_LIVE_MS:.0f})"
    if engine == "personalive":
        ms = measured.get("personalive_ms")
        if ms is None:
            return False, "not measured yet"
        return ms <= PERSONALIVE_LIVE_MS, f"{ms:.0f} ms per frame (fits at {PERSONALIVE_LIVE_MS:.0f})"
    return True, ""


def choose(wanted: str, available: set[str], measured: dict, dropped: set[str] | frozenset[str] = frozenset()) -> Choice:
    """The engine for ``wanted`` (a level or ``auto``), among those ``available`` on this computer."""
    if wanted != AUTO:
        if wanted in available or wanted in ("hair", "face"):
            return Choice(wanted, "chosen")
        return Choice("hair", f"{wanted} is not ready")
    notes = []
    for engine in LEVELS:
        if engine in dropped:
            notes.append(f"{engine}: could not keep up")
        elif engine not in available and engine != "hair":
            continue
        else:
            ok, why = fits(engine, measured)
            if ok:
                return Choice(engine, "; ".join([*notes, why] if why else notes) or "best that fits")
            notes.append(f"{engine}: {why}")
    return Choice("hair", "; ".join(notes))


def next_level(engine: str) -> str | None:
    """The level below ``engine`` (None at the bottom)."""
    if engine not in LEVELS or engine == LEVELS[-1]:
        return None
    return LEVELS[LEVELS.index(engine) + 1]


class EngineWatch:
    """Fed with the live numbers: says when the engine has been too much for WATCH_SECONDS. ``reason``
    is why. A camera that sends few frames, or no voice, is nobody's fault."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self.reason = ""
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self._late: deque[tuple[float, float]] = deque()

    def update(self, fps_in: float, fps_out: float, voice_late: float) -> bool:
        """``voice_late``: the count of voice chunks over real time so far. True: go down a level."""
        now = self.clock()
        self._late.append((now, voice_late))
        while now - self._late[0][0] > WATCH_SECONDS:
            self._late.popleft()
        span = now - self._late[0][0]
        late = self._late[-1][1] - self._late[0][1]
        too_late = span > WATCH_SECONDS / 2 and late > LATE_SHARE * CHUNKS_PER_SECOND * span
        too_slow = fps_in >= 10.0 and fps_out < VIDEO_SHARE * fps_in
        if too_late:
            # Already counted over the whole window: no need to wait it out again.
            self.reason = f"the voice ran late ({late:.0f} chunks in {span:.0f} s)"
            return True
        if not too_slow:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since < WATCH_SECONDS:
            return False
        self.reason = f"the video fell to {fps_out:.0f} of {fps_in:.0f} fps"
        return True


# --- Measured on this GPU, once -----------------------------------------------------------------


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def load_measured(cache: Path, gpu: str | None = None) -> dict:
    try:
        stored = json.loads(Path(cache).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    known = stored.get(gpu or _gpu_name())
    return known if isinstance(known, dict) else {}


def store_measured(cache: Path, values: dict, gpu: str | None = None) -> None:
    cache = Path(cache)
    try:
        stored = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    name = gpu or _gpu_name()
    stored[name] = {**(stored.get(name) or {}), **values}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(stored, indent=1), encoding="utf-8")


def measure_head(engine) -> dict | None:
    """``{"head_ms", "head_fp16"}``: one LivePortrait frame on this GPU, in ms (the render, finding its
    face and the cut-out's two parses), on a made-up picture: the time does not depend on what it
    shows. Half precision stays only if it is faster here and gives the same picture (on a GTX 1660
    it is slower and gives NaN). Its models are freed afterwards."""
    from backend.video import head as heads

    models = engine.models.head_models(engine.arbiter)
    parser = engine.models.parser
    if models is None:
        return None
    picture = np.random.default_rng(0).integers(0, 255, (heads.OUTPUT, heads.OUTPUT, 3), dtype=np.uint8)
    picture = cv2.GaussianBlur(picture, (0, 0), 3.0)
    from backend.video.faces import TEMPLATES

    points = TEMPLATES[heads.HEAD_TEMPLATE] * heads.OUTPUT
    detector = engine.models.detector
    renderer = heads.HeadRenderer(models)
    with engine.arbiter.video(60.0, timeout=10.0):
        renderer.set_source(picture, points)

    def timed() -> tuple[float, np.ndarray | None]:
        times, rendered = [], None
        for _ in range(MEASURE_RENDERS):
            with engine.arbiter.video(60.0, timeout=10.0):
                started = time.perf_counter()
                rendered = renderer.render(picture, points)
                if parser is not None and rendered is not None:
                    detector.detect(rendered)
                    parser.labels(rendered)
                    parser.labels(rendered)
                times.append((time.perf_counter() - started) * 1000.0)
        return statistics.median(times[1:]), rendered   # the first one sets things up

    whole, reference = timed()
    fp16 = False
    half = whole
    if models.device.type == "cuda" and reference is not None:
        models.fp16 = True
        try:
            half, rendered = timed()
            fp16 = same_picture(reference, rendered) and half < whole * FP16_FASTER
        except RuntimeError:
            logger.exception("The whole head in half precision failed")
        models.fp16 = False
    engine.models.free_heads()
    return {"head_ms": half if fp16 else whole, "head_fp16": fp16}


def same_picture(reference: np.ndarray, other: np.ndarray | None) -> bool:
    """Whether half precision gave what full precision gave (NaNs come out as flat colour)."""
    return other is not None and float(np.abs(reference.astype(np.float32) - other).mean()) <= FP16_SAME
