"""Who uses the GPU when: the voice first, the face in the gaps.

The live voice receives one chunk every ~171 ms; after the CPU work on it
(noise filter, voice detection) it renders on the GPU. The face clone runs in
its own process (so it can never hold up the voice's Python thread) and swaps
faces in between: a swap starts only if it ends before the next voice render
is expected (arrival cadence + the usual delay from arrival to render), using
the swap's own measured duration. With no voice streaming, the face has the
GPU to itself.

The state lives in a few shared-memory doubles: the server process writes the
voice side, the face process the face side. ``time.monotonic`` is the same
clock in both processes.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory

import numpy as np

VOICE_IDLE_SECONDS = 1.0   # no chunk for this long: the voice is not streaming
SAFETY_SECONDS = 0.006     # margin before the next render is due
POLL_SECONDS = 0.001        # the voice waiting for a face job to end (it must start at once)
VIDEO_POLL_SECONDS = 0.003  # a face job waiting for a gap (every wake-up takes the face process's GIL)

LONG_JOB_MS = 20.0          # a face swap; shorter jobs (a 3D head render) let a waiting one go first
LONG_WAIT_SECONDS = 0.06    # once a long job has waited this long, short ones wait for it
SHORT_WAIT_SECONDS = 0.03   # and once a short one has waited this long, the next long one waits for it

# Slots in the shared array.
_ARRIVAL, _RENDER_END, _PERIOD, _DELAY, _RENDER_MS, _RENDERING, _VIDEO_BUSY, _LONG_WAITING, _SHORT_WAITING = range(9)
_SLOTS = 9


def _smooth(current: float, value: float, weight: float = 0.2) -> float:
    return value if current == 0 else (1 - weight) * current + weight * value


class GpuArbiter:
    """Create it in the server (``name=None``); the face process attaches with its ``name``."""

    def __init__(self, name: str | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock   # injectable for tests; both processes use time.monotonic
        if name is None:
            self._shm = shared_memory.SharedMemory(create=True, size=_SLOTS * 8)
            self._owner = True
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            self._owner = False
        self._state = np.ndarray((_SLOTS,), np.float64, buffer=self._shm.buf)
        # One video job at a time in this process (the face swap and the head render share the gaps).
        self._video_lock = threading.Lock()
        if self._owner:
            self._state[:] = 0.0

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        self._state = np.zeros(_SLOTS)
        self._shm.close()
        if self._owner:
            with contextlib.suppress(FileNotFoundError):
                self._shm.unlink()

    # Readable stats.
    @property
    def period(self) -> float:
        return float(self._state[_PERIOD])

    @property
    def render_delay(self) -> float:
        return float(self._state[_DELAY])

    @property
    def render_ms(self) -> float:
        return float(self._state[_RENDER_MS])

    def _streaming(self, now: float) -> bool:
        arrival = self._state[_ARRIVAL]
        return arrival > 0 and now - arrival < VOICE_IDLE_SECONDS

    # --- Voice side (server process) ---------------------------------------

    def chunk_arrived(self) -> None:
        """Every live voice chunk, rendered or not (the silence saver skips renders, not chunks)."""
        s, now = self._state, self.clock()
        if self._streaming(now):
            s[_PERIOD] = _smooth(s[_PERIOD], now - s[_ARRIVAL])
        s[_ARRIVAL] = now

    @contextlib.contextmanager
    def voice(self) -> Iterator[None]:
        """Around a voice render: waits only for a face swap already running (one never starts now)."""
        s = self._state
        now = self.clock()
        if self._streaming(now):
            s[_DELAY] = _smooth(s[_DELAY], now - s[_ARRIVAL])
        s[_RENDERING] = 1.0
        while s[_VIDEO_BUSY]:   # at most one swap, which was planned to end before now
            time.sleep(POLL_SECONDS)
        started = self.clock()
        try:
            yield
        finally:
            end = self.clock()
            s[_RENDER_MS] = _smooth(s[_RENDER_MS], (end - started) * 1000)
            s[_RENDER_END] = end
            s[_RENDERING] = 0.0

    # --- Face side (face process) ------------------------------------------

    def gap(self) -> tuple[float, float] | None:
        """``(ms free between voice renders, ms between chunks)``, or None when the voice is not streaming."""
        if not self._streaming(self.clock()) or self.period <= 0:
            return None
        period_ms = self.period * 1000
        return max(0.0, period_ms - self.render_ms - SAFETY_SECONDS * 1000), period_ms

    def free_share(self) -> float:
        """Fraction of the GPU the face can count on (1.0 when the voice is not streaming)."""
        if not self._streaming(self.clock()) or self.period <= 0:
            return 1.0
        return max(0.0, 1.0 - self.render_ms / 1000 / self.period)

    def _fits(self, seconds: float, now: float) -> bool:
        return self._room(now) >= seconds

    def room_ms(self) -> float:
        """GPU time left before the next voice render (safety margin taken off); inf when no voice."""
        return self._room(self.clock()) * 1000

    def _room(self, now: float) -> float:
        s = self._state
        if s[_RENDERING]:
            return 0.0
        if not self._streaming(now) or s[_PERIOD] <= 0:
            return float("inf")
        period, render = s[_PERIOD], s[_RENDER_MS] / 1000
        next_render = s[_ARRIVAL] + s[_DELAY]
        # This chunk's render is done, or it was skipped (the silence saver): the next chunk's counts.
        late = now > next_render + render + SAFETY_SECONDS
        if s[_RENDER_END] >= s[_ARRIVAL] or late:
            next_render += period
        while next_render + period < now:   # chunks overdue (the client paused): plan from now
            next_render += period
        if next_render <= now:              # a render is due right now
            return 0.0
        return max(0.0, next_render - now - SAFETY_SECONDS)

    @contextlib.contextmanager
    def video(self, expected_ms: float, timeout: float = 0.5) -> Iterator[bool]:
        """Around one video job (a face swap, a 3D head render); yields False (holding nothing) when
        no gap opened within ``timeout``. One job at a time across processes (the face process and
        the 3D head's), so neither slows the other into missing the voice's gaps."""
        deadline = self.clock() + timeout
        granted = False
        long = expected_ms >= LONG_JOB_MS
        s = self._state
        while True:
            now = self.clock()
            waiting_slot = _LONG_WAITING if long else _SHORT_WAITING
            if not s[waiting_slot]:
                s[waiting_slot] = now
            # The process's video lock is taken only once the job fits, never while waiting:
            # a long job waiting for a big gap does not hold up a short one that fits now, until
            # it has waited a while (then short ones wait for it: a swap is never starved).
            other = _SHORT_WAITING if long else _LONG_WAITING
            waited = now - s[other] if s[other] > 0 else 0.0
            mine = now - s[waiting_slot]
            # The other kind of job has waited long enough, and longer than this one: it goes first
            # (only one of the two ever yields; 1 s: its process is gone).
            starving = (SHORT_WAIT_SECONDS if long else LONG_WAIT_SECONDS) < waited < 1.0 and waited > mine
            if (not s[_VIDEO_BUSY] and not starving and self._fits(expected_ms / 1000, now)
                    and self._video_lock.acquire(blocking=False)):
                if not s[_VIDEO_BUSY] and self._fits(expected_ms / 1000, self.clock()):
                    self._state[_VIDEO_BUSY] = 1.0
                    s[waiting_slot] = 0.0
                    # The voice may have started in between: then step back.
                    if not self._state[_RENDERING]:
                        granted = True
                        break
                    self._state[_VIDEO_BUSY] = 0.0
                self._video_lock.release()
            if now >= deadline:
                s[_LONG_WAITING if long else _SHORT_WAITING] = 0.0
                break
            time.sleep(VIDEO_POLL_SECONDS)
        try:
            yield granted
        finally:
            if granted:
                self._state[_VIDEO_BUSY] = 0.0
                self._video_lock.release()
