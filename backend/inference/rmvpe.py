"""RMVPE pitch estimator (deep U-Net + BiGRU over a log-mel spectrogram).

Ported from RVC's ``rmvpe.py`` (MIT). The mel filterbank comes from
``backend.audio.mel`` (HTK scale, as ``librosa.filters.mel(htk=True)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from backend.audio.mel import mel_filterbank

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 128
N_CLASSES = 360
_CENTS = np.pad(20 * np.arange(N_CLASSES) + 1997.3794084376191, (4, 4))


class _ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + (self.shortcut(x) if self.shortcut is not None else x)


class _ResEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] | None,
                 n_blocks: int) -> None:
        super().__init__()
        self.conv = nn.ModuleList([_ConvBlockRes(in_channels, out_channels)]
                                  + [_ConvBlockRes(out_channels, out_channels) for _ in range(n_blocks - 1)])
        self.pool = nn.AvgPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for block in self.conv:
            x = block(x)
        return (x, self.pool(x)) if self.pool is not None else x


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, n_encoders: int, kernel_size: tuple[int, int], n_blocks: int,
                 out_channels: int = 16) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(_ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks))
            in_channels, out_channels = out_channels, out_channels * 2
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class _Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ResEncoderBlock(in_channels, out_channels, None, n_blocks)]
                                    + [_ResEncoderBlock(out_channels, out_channels, None, n_blocks)
                                       for _ in range(n_inters - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ResDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], n_blocks: int) -> None:
        super().__init__()
        output_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01), nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([_ConvBlockRes(out_channels * 2, out_channels)]
                                   + [_ConvBlockRes(out_channels, out_channels) for _ in range(n_blocks - 1)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.conv1(x), skip), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride: tuple[int, int], n_blocks: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(_ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks))
            in_channels //= 2

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - index])
        return x


class _DeepUnet(nn.Module):
    def __init__(self, kernel_size: tuple[int, int], n_blocks: int, en_de_layers: int = 5,
                 inter_layers: int = 4) -> None:
        super().__init__()
        self.encoder = _Encoder(1, en_de_layers, kernel_size, n_blocks)
        self.intermediate = _Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel,
                                          inter_layers, n_blocks)
        self.decoder = _Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class _BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True,
                          bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self, n_blocks: int = 4, n_gru: int = 1, kernel_size: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.unet = _DeepUnet(kernel_size, n_blocks)
        self.cnn = nn.Conv2d(16, 3, 3, padding=1)
        self.fc = nn.Sequential(_BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASSES), nn.Dropout(0.25),
                                nn.Sigmoid())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1)))
        return self.fc(x.transpose(1, 2).flatten(-2))


class Rmvpe:
    """Frame-level F0 (100 frames/s) from 16 kHz audio."""

    def __init__(self, weights: Path, device: torch.device, half: bool) -> None:
        self.device = device
        self.half = half
        model = E2E()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model = model.eval().to(device)
        self.model = model.half() if half else model.float()
        self.mel_basis = torch.from_numpy(mel_filterbank(SAMPLE_RATE, 1024, N_MELS, 30, 8000, htk=True)).to(device)
        self.window = torch.hann_window(1024, device=device)

    def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(audio, n_fft=1024, hop_length=HOP_LENGTH, win_length=1024, window=self.window,
                              center=True, return_complex=True)
        mel = torch.matmul(self.mel_basis, spectrum.abs())
        if self.half:
            mel = mel.half()
        return torch.log(torch.clamp(mel, min=1e-5))

    @torch.no_grad()
    def salience(self, audio: np.ndarray) -> np.ndarray:
        """Per-frame 360-bin pitch posteriors, shape ``[frames, 360]``."""
        mel = self._log_mel(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)[None])
        frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((frames - 1) // 32 + 1) - frames))
        hidden = self.model(mel.half() if self.half else mel.float())[:, :frames]
        return hidden.squeeze(0).float().cpu().numpy()

    def infer(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        return decode_salience(self.salience(audio), threshold)


def decode_salience(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Local weighted average of cents around the peak bin; 0 Hz where unvoiced."""
    center = np.argmax(salience, axis=1) + 4
    padded = np.pad(salience, ((0, 0), (4, 4)))
    offsets = np.arange(-4, 5)
    columns = center[:, None] + offsets[None, :]
    window = np.take_along_axis(padded, columns, axis=1)
    cents = (window * _CENTS[columns]).sum(axis=1) / window.sum(axis=1)
    cents[padded.max(axis=1) <= threshold] = 0
    f0 = 10 * (2 ** (cents / 1200))
    f0[f0 == 10] = 0
    return f0
