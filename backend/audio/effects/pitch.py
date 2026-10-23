"""Model-free pitch shifting and spectral-envelope formant shifting."""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sps

logger = logging.getLogger(__name__)


class PitchShifter:
    """Dual-tap delay-line pitch shifter (Doppler/harmonizer style).

    Two read taps sweep through a short ring buffer at a rate of
    ``1 - 2^(semitones/12)`` with equal-power sine crossfading, which keeps
    output continuous across chunks. Used for the model-free DSP mode.
    """

    WINDOW_SECONDS = 0.032

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self.win = max(64, int(self.WINDOW_SECONDS * sample_rate))
        self._hist = np.zeros(self.win + 2, dtype=np.float32)
        self._phase = 0.0  # tap delay in samples, [0, win)

    def process(self, x: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 1e-3 or len(x) == 0:
            self._push_history(x)
            return x

        ratio = 2.0 ** (semitones / 12.0)
        n = len(x)
        rate = 1.0 - ratio
        idx = np.arange(n, dtype=np.float64)

        d1 = (self._phase + rate * idx) % self.win
        self._phase = float((self._phase + rate * n) % self.win)
        d2 = (d1 + self.win / 2.0) % self.win
        g1 = np.sin(np.pi * d1 / self.win)
        g2 = np.sin(np.pi * d2 / self.win)

        full = np.concatenate([self._hist, x.astype(np.float32)])
        positions = np.arange(len(full), dtype=np.float64)
        write_pos = len(self._hist) + idx
        tap1 = np.interp(write_pos - d1, positions, full)
        tap2 = np.interp(write_pos - d2, positions, full)

        self._hist = full[-(self.win + 2) :]
        return (g1 * tap1 + g2 * tap2).astype(np.float32)

    def _push_history(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        full = np.concatenate([self._hist, x.astype(np.float32)])
        self._hist = full[-(self.win + 2) :]


class FormantShifter:
    """STFT cepstral-envelope formant shifter with streaming overlap-add.

    Warps the spectral envelope by ``2^(semitones/12)`` while keeping the
    harmonic structure (pitch) intact. Adds FFT - HOP samples of latency.
    """

    FFT = 1024
    HOP = 256

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self.window = sps.get_window("hann", self.FFT).astype(np.float32)
        # Hann with 75% overlap sums (analysis*synthesis window) to 1.5.
        self._ola_norm = 1.5
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._ola = np.zeros(self.FFT, dtype=np.float32)
        self._out_fifo = np.zeros(0, dtype=np.float32)
        self._lifter = max(20, int(sample_rate / 1000))

    def process(self, x: np.ndarray, semitones: float) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x
        shift = 2.0 ** (semitones / 12.0)
        self._in_buf = np.concatenate([self._in_buf, x.astype(np.float32)])

        while len(self._in_buf) >= self.FFT:
            frame = self._in_buf[: self.FFT]
            self._in_buf = self._in_buf[self.HOP :]
            processed = self._process_frame(frame, shift)
            self._ola[: self.FFT] += processed
            self._out_fifo = np.concatenate([self._out_fifo, self._ola[: self.HOP] / self._ola_norm])
            self._ola = np.concatenate([self._ola[self.HOP :], np.zeros(self.HOP, dtype=np.float32)])

        if len(self._out_fifo) >= n:
            out = self._out_fifo[:n]
            self._out_fifo = self._out_fifo[n:]
        else:
            # Warm-up: pad the initial latency gap with silence.
            pad = np.zeros(n - len(self._out_fifo), dtype=np.float32)
            out = np.concatenate([pad, self._out_fifo])
            self._out_fifo = np.zeros(0, dtype=np.float32)
        return out.astype(np.float32)

    def _process_frame(self, frame: np.ndarray, shift: float) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window)
        mag = np.abs(spec)
        phase = np.angle(spec)

        log_mag = np.log(mag + 1e-9)
        ceps = np.fft.irfft(log_mag)
        ceps_lift = ceps.copy()
        ceps_lift[self._lifter : -self._lifter] = 0.0
        envelope = np.fft.rfft(ceps_lift).real

        bins = np.arange(len(mag), dtype=np.float64)
        warped = np.interp(bins / shift, bins, envelope, left=envelope[0], right=envelope[-1])

        new_mag = np.exp(log_mag - envelope + warped)
        new_spec = new_mag * np.exp(1j * phase)
        return (np.fft.irfft(new_spec) * self.window).astype(np.float32)
