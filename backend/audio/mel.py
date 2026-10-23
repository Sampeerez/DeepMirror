"""Spectrogram and mel filterbank helpers (equivalent to ``librosa.filters.mel``)."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def _hz_to_mel(freq: np.ndarray | float, htk: bool) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # Slaney: linear below 1 kHz, logarithmic above.
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(freq >= min_log_hz, min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep,
                    freq / f_sp)


def _mel_to_hz(mels: np.ndarray | float, htk: bool) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), f_sp * mels)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None,
                   htk: bool = False) -> np.ndarray:
    """Triangular mel filters with Slaney area normalisation, shape ``[n_mels, n_fft//2+1]``."""
    fmax = sr / 2.0 if fmax is None else fmax
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_freqs = _mel_to_hz(np.linspace(_hz_to_mel(fmin, htk), _hz_to_mel(fmax, htk), n_mels + 2), htk)
    widths = np.diff(mel_freqs)
    ramps = mel_freqs[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_freqs[2:n_mels + 2] - mel_freqs[:n_mels]))[:, None]
    return weights.astype(np.float32)


def linear_spectrogram(audio: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Magnitude STFT ``[batch, n_fft//2+1, frames]`` with VITS/RVC framing (``frames = samples // hop``)."""
    pad = (n_fft - hop) // 2
    audio = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(win, device=audio.device, dtype=audio.dtype)
    spectrum = torch.stft(audio, n_fft, hop_length=hop, win_length=win, window=window, center=False,
                          normalized=False, onesided=True, return_complex=True)
    return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-6)


def log_mel(spectrogram: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(torch.matmul(basis, spectrogram), min=1e-5))
