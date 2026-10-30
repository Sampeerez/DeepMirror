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
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.video.engine import FaceEngine, HeadLayer, Key, Tracked

logger = logging.getLogger(__name__)

MAX_KEYS_PER_SECOND = 30.0
KEEP_KEYS_SECONDS = 2.0
HEAD3D_IN_FLIGHT = 2        # frames with the 3D head's process at once (a frame between gets the newest)
HEAD3D_WAIT = 0.15          # how long a frame waits for its own 3D head render
HEAD3D_REUSE = 0.2          # a render this much older still dresses a frame whose own did not come


@dataclass(slots=True)
class _Record:
    t: float                 # capture time (seconds, client clock)
    frame: np.ndarray
    tracked: Tracked
    arrived: float           # server monotonic time
    seq: int | None = None   # its 3D head render, when one was asked for


class LiveFaceSession:
    """``head3d`` (an avatar.HeadDriver): the 3D head renders every frame too, and its hair, ears
    and outline dress the swapped face (the swap gives the face itself: the speaker's own
    expressions and mouth, with the worn identity)."""

    def __init__(self, engine: FaceEngine, max_wait: float = 0.3, head3d=None) -> None:
        self.engine = engine
        self.head3d = head3d
        self._layers3d: dict[int, Any] = {}   # avatar.Layer (or None: no face there)
        self._latest3d: tuple[float, Any] | None = None
        self._next_seq = 0
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
        self._head: HeadLayer | None = None
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
                "heads_per_second": round(self.heads_per_second, 2),
                "swap_ms": self.engine.models.swap_ms.get(swapper or "", None),
                "track_ms": round(self.track_ms, 1), "compose_ms": round(self.compose_ms, 1),
                "face": self._records[-1].tracked.face is not None if self._records else None,
                "gpu_share": round(self.engine.arbiter.free_share(), 2)}

    # --- Input ------------------------------------------------------------

    def _on_layer3d(self, seq: int, layer) -> None:
        with self._condition:
            if seq > self._done_seq:
                self._layers3d[seq] = layer
                self._condition.notify_all()

    def submit(self, t: float, frame: np.ndarray) -> None:
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
            self._records.append(_Record(t, frame, tracked, time.monotonic(), seq))
            self.frames_in += 1
            self._condition.notify_all()

    def _layer3d(self, record: _Record):
        """The 3D head's layer for this frame (or the newest one, if close in time)."""
        layer = None
        if record.seq is not None:
            layer = self._layers3d.pop(record.seq, None)
            self._done_seq = max(self._done_seq, record.seq)
            for late in [seq for seq in self._layers3d if seq < record.seq]:
                del self._layers3d[late]
        if layer is not None:
            self._latest3d = (record.t, layer)
        elif self._latest3d is not None and abs(record.t - self._latest3d[0]) < HEAD3D_REUSE:
            layer = self._latest3d[1]
        return layer

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
            if record is None:
                time.sleep(0.03)
                continue
            try:
                layer = self.engine.head_layer(record.frame, record.tracked, record.t, wait=1.0)
            except Exception:
                logger.exception("Head render failed")
                time.sleep(1.0)
                continue
            if layer is not None:
                self._head = layer
                self._head_times.append(time.monotonic())

    # --- Output -----------------------------------------------------------

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
                self._records.popleft()
                before, after = self._around(record.t)
                layer3d = self._layer3d(record) if self.head3d is not None else None
            started = time.perf_counter()
            if self.engine.identity is None:
                frame = record.frame
            else:
                frame = self.engine.compose(record.frame, record.tracked, record.t, before, after,
                                            self.keys_per_second, head=self._head, layer3d=layer3d)
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
