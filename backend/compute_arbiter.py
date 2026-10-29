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
import time
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory

import numpy as np

VOICE_IDLE_SECONDS = 1.0   # no chunk for this long: the voice is not streaming
SAFETY_SECONDS = 0.006     # margin before the next render is due
POLL_SECONDS = 0.001

# Slots in the shared array.
_ARRIVAL, _RENDER_END, _PERIOD, _DELAY, _RENDER_MS, _RENDERING, _VIDEO_BUSY = range(7)
_SLOTS = 8


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
        s = self._state
        if s[_RENDERING]:
            return False
        if not self._streaming(now) or s[_PERIOD] <= 0:
            return True
        period, render = s[_PERIOD], s[_RENDER_MS] / 1000
        next_render = s[_ARRIVAL] + s[_DELAY]
        # This chunk's render is done, or it was skipped (the silence saver): the next chunk's counts.
        late = now > next_render + render + SAFETY_SECONDS
        if s[_RENDER_END] >= s[_ARRIVAL] or late:
            next_render += period
        while next_render + period < now:   # chunks overdue (the client paused): plan from now
            next_render += period
        if next_render <= now:              # a render is due right now
            return False
        return next_render - now >= seconds + SAFETY_SECONDS

    @contextlib.contextmanager
    def video(self, expected_ms: float, timeout: float = 0.5) -> Iterator[bool]:
        """Around one face swap; yields False (holding nothing) when no gap opened within ``timeout``."""
        deadline = self.clock() + timeout
        granted = False
        while True:
            now = self.clock()
            if self._fits(expected_ms / 1000, now):
                self._state[_VIDEO_BUSY] = 1.0
                # The voice may have started in between: then step back.
                if not self._state[_RENDERING]:
                    granted = True
                    break
                self._state[_VIDEO_BUSY] = 0.0
            if now >= deadline:
                break
            time.sleep(POLL_SECONDS)
        try:
            yield granted
        finally:
            if granted:
                self._state[_VIDEO_BUSY] = 0.0
