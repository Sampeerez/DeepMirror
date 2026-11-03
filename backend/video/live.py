"""Realtime face clone: camera frames in, cloned frames out, in order and on time.

Frames are tracked as they arrive. A worker thread swaps keyframes whenever
the GPU has room. A frame is released once a keyframe at or after its time
exists (so it is interpolated, not extrapolated), or when it has waited
``max_wait`` (then the nearest keyframe is used). The output is delayed
anyway to stay in sync with the converted voice, so the wait is not seen.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from backend.video.engine import FaceEngine, HeadLayer, Key, Tracked

logger = logging.getLogger(__name__)

MAX_KEYS_PER_SECOND = 30.0
KEEP_KEYS_SECONDS = 2.0
HEAD3D_IN_FLIGHT = 2        # frames with the 3D head's process at once (a frame between gets the newest; more makes the voice late)
HEAD3D_WAIT = 0.15          # how long a frame waits for its own 3D head render
FRINGE_SHARE = 0.02         # of the head crop: hair over the face from here on counts as a fringe
HEAD3D_REUSE = 0.35         # a render this much older still dresses a frame whose own did not come
HEAD_WAIT = 0.15            # how long a frame waits for its own whole-head render...
HEAD_KEEPS_UP = 15.0        # ...on a GPU rendering at least this many per second (else, the newest)
HEAD_LAYERS = 8             # recent whole-head renders kept: a frame takes the nearest in time
BAR_DARK = 24               # brightness (0..255) below which a line of the frame may be a black bar...
BAR_STRAY = 0.03            # ...with this share of it brighter at the most (an app's small logo on the bar)
BAR_LEAST = 0.02            # of the frame's side: anything thinner is not a bar
PICTURE_LEAST = 0.15        # of the frame's side: less than this left is a dark room, not a picture in bars
PICTURE_EVERY = 15          # frames between looks at where the picture is
PICTURE_SURE = 3            # looks in a row before the picture is taken to have changed
PICTURE_SAME = 0.01         # of the frame's side: closer than this is the same picture


def picture_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Where the camera's real picture is in ``frame`` when it comes inside black bars, at any side
    (a phone upright in a landscape frame, a wide film in a 4:3 one): ``(x0, y0, x1, y1)``, or None
    when the whole frame is the picture."""
    height, width = frame.shape[:2]
    lit = frame[::4, ::4].max(2) > BAR_DARK
    sides = []
    for shares, side in ((lit.mean(0), width), (lit.mean(1), height)):
        lines = np.nonzero(shares > BAR_STRAY)[0]
        if not len(lines):
            return None
        low, high = int(lines[0]) * 4, min(side, (int(lines[-1]) + 1) * 4)
        low, high = (low if low >= BAR_LEAST * side else 0), (high if side - high >= BAR_LEAST * side else side)
        if high - low < PICTURE_LEAST * side:
            return None
        sides.append((low, high))
    (x0, x1), (y0, y1) = sides
    return None if (x0, y0, x1, y1) == (0, 0, width, height) else (x0, y0, x1, y1)


class Picture:
    """The camera's real picture within its frames, steady: everything downstream works on it alone,
    so nothing is painted on the bars, their black never gets into a blur and the head's camera is
    centred on what the lens saw. Taken at once on the first frame, and changed only when several
    looks in a row agree (a dark moment must not move it)."""

    def __init__(self) -> None:
        self.box: tuple[int, int, int, int] | None = None
        self._frames = 0
        self._wanted: tuple[int, int, int, int] | None = None
        self._agreed = 0

    def update(self, frame: np.ndarray) -> bool:
        """Looks at ``frame`` now and then; whether the picture has just moved."""
        self._frames += 1
        if self._frames % PICTURE_EVERY != 1:
            return False
        found = picture_box(frame)
        if self._same(found, self.box, frame.shape):
            self._agreed = 0
            return False
        self._agreed = self._agreed + 1 if self._same(found, self._wanted, frame.shape) else 1
        self._wanted = found
        if self._frames > 1 and self._agreed < PICTURE_SURE:
            return False
        self.box, self._agreed = found, 0
        return self._frames > 1

    @staticmethod
    def _same(one, other, shape: tuple[int, ...]) -> bool:
        if one is None or other is None:
            return one is other
        limit = PICTURE_SAME * max(shape[:2])
        return all(abs(a - b) <= limit for a, b in zip(one, other, strict=True))


@dataclass(slots=True)
class _Record:
    t: float                 # capture time (seconds, client clock)
    frame: np.ndarray
    tracked: Tracked
    arrived: float           # server monotonic time
    seq: int | None = None   # its 3D head render, when one was asked for
    whole: np.ndarray | None = None   # the frame with its bars, when ``frame`` is the picture inside them
    box: tuple[int, int, int, int] | None = None


class LiveFaceSession:
    """``head3d`` (an avatar.HeadDriver): the 3D head renders every frame too, and its hair, ears
    and outline dress the swapped face (the swap gives the face itself: the speaker's own
    expressions and mouth, with the worn identity)."""

    def __init__(self, engine: FaceEngine, max_wait: float = 0.3, head3d=None) -> None:
        self.engine = engine
        self.head3d = head3d
        self._layers3d: dict[int, Any] = {}   # avatar.Layer (or None: no face there)
        self._latest3d: tuple[float, Any, Any] | None = None   # (time, layer, the five points then)
        self._next_seq = 0
        self._picture = Picture()
        self.layers_own = self.layers_moved = self.layers_none = 0   # how frames were dressed by the 3D head
        self._fringe: bool | None = None
        self._done_seq = -1
        engine.hair_from_3d = head3d is not None
        if head3d is not None:
            head3d.on_layer = self._on_layer3d
        self.max_wait = max_wait
        self._records: deque[_Record] = deque()
        self._keys: deque[Key] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._key_times: deque[float] = deque(maxlen=30)
        self._heads_done: deque[HeadLayer] = deque(maxlen=HEAD_LAYERS)
        self._head_rendered = -1e9   # the capture time of the newest frame sent to render
        self._head_times: deque[float] = deque(maxlen=10)
        self.frames_in = self.frames_out = self.keys_made = self.keys_skipped = 0
        self._rate_since = (time.monotonic(), 0, 0)   # for the frames in and out per second in stats
        self.track_ms = self.compose_ms = 0.0
        engine.reset_head()
        self._worker = threading.Thread(target=self._keyframes, name="face-keyframes", daemon=True)
        self._worker.start()
        # The whole head renders on its own thread, in the GPU gaps the swaps leave.
        self._head_worker = threading.Thread(target=self._heads, name="face-head", daemon=True)
        self._head_worker.start()

    # --- Stats ------------------------------------------------------------

    @property
    def keys_per_second(self) -> float:
        times = list(self._key_times)
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-3)

    @property
    def heads_per_second(self) -> float:
        times = list(self._head_times)
        if len(times) < 2 or time.monotonic() - times[-1] > 5.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-3)

    def stats(self) -> dict:
        swapper = self.engine._swapper_id
        now = time.monotonic()
        span = max(now - self._rate_since[0], 1e-3)
        fps_in = (self.frames_in - self._rate_since[1]) / span
        fps_out = (self.frames_out - self._rate_since[2]) / span
        self._rate_since = (now, self.frames_in, self.frames_out)
        timings3d = (self.head3d.timings or {}) if self.head3d is not None else {}
        return {"keys_per_second": round(self.keys_per_second, 1), "swapper": swapper,
                "fps_in": round(fps_in, 1), "fps_out": round(fps_out, 1),
                "mode": "3d" if self.head3d is not None else "head" if self.engine.wants_head else "face",
                "render_ms": round(float(timings3d.get("render_ms", 0.0)), 1) if timings3d else None,
                "render_size": int(timings3d.get("render_size", 0)) if timings3d else None,
                "head_fit": timings3d.get("head_fit"),
                "heads_per_second": round(self.heads_per_second, 2),
                "swap_ms": self.engine.models.swap_ms.get(swapper or "", None),
                "track_ms": round(self.track_ms, 1), "compose_ms": round(self.compose_ms, 1),
                "layers": (self.layers_own, self.layers_moved, self.layers_none), "picture": self._picture.box,
                "face": self._records[-1].tracked.face is not None if self._records else None,
                "gpu_share": round(self.engine.arbiter.free_share(), 2)}

    # --- Input ------------------------------------------------------------

    def _on_layer3d(self, seq: int, layer) -> None:
        with self._condition:
            if seq > self._done_seq:
                self._layers3d[seq] = layer
                self._condition.notify_all()

    def submit(self, t: float, frame: np.ndarray) -> None:
        if self._picture.update(frame):
            self.engine.new_picture()
        whole, box = None, self._picture.box
        if box is not None:
            whole, frame = frame, frame[box[1]:box[3], box[0]:box[2]].copy()
        seq = None
        if self.head3d is not None:   # first: it renders while this frame is tracked here
            with self._condition:
                waiting = sum(1 for r in self._records if r.seq is not None and r.seq not in self._layers3d)
                if waiting < HEAD3D_IN_FLIGHT:
                    seq, self._next_seq = self._next_seq, self._next_seq + 1
            if seq is not None and not self.head3d.submit(seq, t, frame):
                seq = None
        started = time.perf_counter()
        tracked = self.engine.track(frame, t)
        self.track_ms = 0.9 * self.track_ms + 0.1 * (time.perf_counter() - started) * 1000
        with self._condition:
            self._records.append(_Record(t, frame, tracked, time.monotonic(), seq, whole, box))
            self.frames_in += 1
            self._condition.notify_all()

    def _tell_fringe(self) -> None:
        """The head process comes down to the eyelids only for a speaker with a fringe."""
        own = self.engine._own
        if self.head3d is None or own is None:
            return
        fringe = own.fringe is not None and float(own.fringe.mean()) > FRINGE_SHARE
        if fringe != self._fringe:
            self._fringe = fringe
            self.head3d.send("fringe", fringe)

    def _layer3d(self, record: _Record):
        """The 3D head's layer for this frame (or the newest one, if close in time)."""
        layer = None
        if record.seq is not None:
            layer = self._layers3d.pop(record.seq, None)
            self._done_seq = max(self._done_seq, record.seq)
            for late in [seq for seq in self._layers3d if seq < record.seq]:
                del self._layers3d[late]
        if layer is not None:
            self._latest3d = (record.t, layer, record.tracked.points)
            self.layers_own += 1
        elif self._latest3d is not None and abs(record.t - self._latest3d[0]) < HEAD3D_REUSE:
            # The newest render, moved from where the face was then to where it is now: the hair
            # stays on the head, not a frame behind it.
            layer = self._moved(self._latest3d[1], self._latest3d[2], record.tracked.points)
            self.layers_moved += 1
        else:
            self.layers_none += 1
        return layer

    @staticmethod
    def _moved(layer, before, now):
        if before is None or now is None:
            return layer
        matrix, _ = cv2.estimateAffinePartial2D(before.astype(np.float32), now.astype(np.float32), method=cv2.LMEDS)
        if matrix is None:
            return layer
        height, width = layer.image.shape[:2]
        corners = cv2.transform(np.array([[[layer.x, layer.y], [layer.x + width, layer.y], [layer.x, layer.y + height],
                                           [layer.x + width, layer.y + height]]], np.float32), matrix)[0]
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        local = matrix.copy()
        local[:, 2] += matrix[:, :2] @ np.array([layer.x, layer.y], np.float64) - (x0, y0)
        image = cv2.warpAffine(layer.image, local, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
        face = layer.face
        if face is not None:
            small = local.copy()
            small[:, 2] /= 4
            face = cv2.warpAffine(face, small, (max(1, image.shape[1] // 4), max(1, image.shape[0] // 4)),
                                  flags=cv2.INTER_NEAREST)
        return replace(layer, x=int(x0), y=int(y0), image=image, face=face)

    # --- Keyframes (worker thread) ----------------------------------------

    def _candidate(self) -> _Record | None:
        last = self._keys[-1].t if self._keys else -1e9
        for record in reversed(self._records):   # the newest frame with a face
            if record.tracked.face is not None and record.t >= last + 1 / MAX_KEYS_PER_SECOND:
                return record
        return None

    def _keyframes(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (self.engine.identity is None or self._candidate() is None):
                    self._condition.wait(0.1)
                if self._closed:
                    return
                record = self._candidate()
            if record is None:
                continue
            try:
                key = self.engine.keyframe(record.frame, record.tracked, record.t, self.keys_per_second)
            except Exception:
                logger.exception("Face swap failed")
                time.sleep(0.2)
                continue
            with self._condition:
                if key is None:
                    self.keys_skipped += 1
                    continue
                self._keys.append(key)
                while self._keys and self._keys[0].t < key.t - KEEP_KEYS_SECONDS:
                    self._keys.popleft()
                self._key_times.append(time.monotonic())
                self.keys_made += 1
                self._condition.notify_all()

    # --- The whole head (its own thread) ------------------------------------

    def _heads(self) -> None:
        while not self._closed:
            if not self.engine.wants_head:
                time.sleep(0.2)
                continue
            with self._condition:
                record = next((r for r in reversed(self._records) if r.tracked.face is not None), None)
            if record is None or record.t <= self._head_rendered:   # nothing new: not the same frame again
                time.sleep(0.003)
                continue
            self._head_rendered = record.t
            try:
                layer = self.engine.head_layer(record.frame, record.tracked, record.t, wait=1.0)
            except Exception:
                logger.exception("Head render failed")
                time.sleep(1.0)
                continue
            if layer is not None:
                with self._condition:
                    self._heads_done.append(layer)
                    self._head_times.append(time.monotonic())
                    self._condition.notify_all()

    # --- Output -----------------------------------------------------------

    def _head_for(self, t: float) -> HeadLayer | None:
        return min(self._heads_done, key=lambda layer: abs(layer.t - t), default=None)

    def _waits_for_head(self, record: _Record, waiting: float) -> bool:
        """A GPU that renders the head per frame: the frame waits for its own, the output is delayed
        for the voice anyway. On a slower one it takes the newest there is."""
        remote = self.engine.remote
        return (self.engine.wants_head and record.tracked.face is not None
                and waiting < getattr(remote, "wait", HEAD_WAIT)
                and self.heads_per_second >= getattr(remote, "keeps_up", HEAD_KEEPS_UP)
                and all(layer.t < record.t for layer in self._heads_done))

    def _around(self, t: float) -> tuple[Key | None, Key | None]:
        before = after = None
        for key in self._keys:
            if key.t <= t:
                before = key
            elif after is None:
                after = key
        return before, after

    def ready(self) -> list[tuple[float, np.ndarray]]:
        """Frames that can go out now, oldest first: ``(capture time, cloned frame)``."""
        out = []
        while True:
            with self._condition:
                if not self._records:
                    break
                record = self._records[0]
                latest = self._keys[-1].t if self._keys else -1e9
                waiting = time.monotonic() - record.arrived
                if (self.engine.identity is not None and record.tracked.face is not None
                        and record.t > latest and waiting < self.max_wait):
                    break
                if record.seq is not None and record.seq not in self._layers3d and waiting < HEAD3D_WAIT:
                    break
                if self._waits_for_head(record, waiting):
                    break
                self._records.popleft()
                before, after = self._around(record.t)
                layer3d = self._layer3d(record) if self.head3d is not None else None
                head = self._head_for(record.t)
                self._tell_fringe()
            started = time.perf_counter()
            if self.engine.identity is None:
                frame = record.frame
            else:
                frame = self.engine.compose(record.frame, record.tracked, record.t, before, after,
                                            self.keys_per_second, head=head, layer3d=layer3d)
            if record.whole is not None and record.box is not None:   # back between its bars
                x0, y0, x1, y1 = record.box
                whole = record.whole.copy()
                whole[y0:y1, x0:x1] = frame
                frame = whole
            self.compose_ms = 0.9 * self.compose_ms + 0.1 * (time.perf_counter() - started) * 1000
            self.frames_out += 1
            out.append((record.t, frame))
        return out

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2)
        self._head_worker.join(timeout=2)
