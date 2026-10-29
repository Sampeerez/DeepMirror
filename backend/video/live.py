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

import numpy as np

from backend.video.engine import FaceEngine, Key, Tracked

logger = logging.getLogger(__name__)

MAX_KEYS_PER_SECOND = 30.0
KEEP_KEYS_SECONDS = 2.0


@dataclass(slots=True)
class _Record:
    t: float                 # capture time (seconds, client clock)
    frame: np.ndarray
    tracked: Tracked
    arrived: float           # server monotonic time


class LiveFaceSession:
    def __init__(self, engine: FaceEngine, max_wait: float = 0.3) -> None:
        self.engine = engine
        self.max_wait = max_wait
        self._records: deque[_Record] = deque()
        self._keys: deque[Key] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._key_times: deque[float] = deque(maxlen=30)
        self.frames_in = self.frames_out = self.keys_made = self.keys_skipped = 0
        self.track_ms = self.compose_ms = 0.0
        self._worker = threading.Thread(target=self._keyframes, name="face-keyframes", daemon=True)
        self._worker.start()

    # --- Stats ------------------------------------------------------------

    @property
    def keys_per_second(self) -> float:
        times = list(self._key_times)
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-3)

    def stats(self) -> dict:
        swapper = self.engine._swapper_id
        return {"keys_per_second": round(self.keys_per_second, 1), "swapper": swapper,
                "swap_ms": self.engine.models.swap_ms.get(swapper or "", None),
                "track_ms": round(self.track_ms, 1), "compose_ms": round(self.compose_ms, 1),
                "face": self._records[-1].tracked.face is not None if self._records else None,
                "gpu_share": round(self.engine.arbiter.free_share(), 2)}

    # --- Input ------------------------------------------------------------

    def submit(self, t: float, frame: np.ndarray) -> None:
        started = time.perf_counter()
        tracked = self.engine.track(frame, t)
        self.track_ms = 0.9 * self.track_ms + 0.1 * (time.perf_counter() - started) * 1000
        with self._condition:
            self._records.append(_Record(t, frame, tracked, time.monotonic()))
            self.frames_in += 1
            self._condition.notify_all()

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
                self._records.popleft()
                before, after = self._around(record.t)
            started = time.perf_counter()
            if self.engine.identity is None:
                frame = record.frame
            else:
                frame = self.engine.compose(record.frame, record.tracked, record.t, before, after,
                                            self.keys_per_second)
            self.compose_ms = 0.9 * self.compose_ms + 0.1 * (time.perf_counter() - started) * 1000
            self.frames_out += 1
            out.append((record.t, frame))
        return out

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2)
