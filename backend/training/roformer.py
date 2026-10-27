"""Mel-Band RoFormer vocal separation (Wang et al., 2023) in plain PyTorch.

Runs the "vocals" model by Kimberley Jensen (MIT, KimberleyJSN/melbandroformer)
to keep only the voice of training recordings. Module and parameter names match
the reference implementation (lucidrains/BS-RoFormer, ZFTurbo
Music-Source-Separation-Training, both MIT) so the official checkpoint loads
unchanged; the mel band layout and the rotary embedding are written here, so
librosa, einops and rotary-embedding-torch are not needed.
"""

from __future__ import annotations

import itertools
import math
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SAMPLE_RATE = 44100
N_FFT = 2048
HOP = 441
DIM, DEPTH, HEADS, DIM_HEAD, NUM_BANDS = 384, 6, 8, 64, 60
CHUNK = 352800          # 8 s per pass
OVERLAP = 2             # each sample is covered by two passes


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """Slaney mel scale (librosa's default)."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log = f >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    hz = (200.0 / 3) * m
    log = m >= 15.0
    return np.where(log, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), hz)


def mel_band_mask(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = NUM_BANDS) -> np.ndarray:
    """Which STFT bins belong to each mel band (triangular filters, as librosa.filters.mel)."""
    freqs = np.linspace(0, sample_rate / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - freqs[None, :]
    weights = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights[0, 0] = 1.0
    weights[-1, -1] = 1.0
    return weights > 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    """Interleaved rotary position embedding (rotary-embedding-torch layout)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)

    def rotate(self, t: Tensor) -> Tensor:
        positions = torch.arange(t.shape[-2], device=t.device, dtype=self.freqs.dtype)
        angles = (positions[:, None] * self.freqs[None, :]).repeat_interleave(2, dim=-1)
        pairs = t.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return t * angles.cos() + rotated * angles.sin()


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_embed: RotaryEmbedding) -> None:
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.rotary_embed.rotate(q), self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * self.to_gates(x).permute(0, 2, 1).unsqueeze(-1).sigmoid()
        return self.to_out(out.permute(0, 2, 1, 3).reshape(b, n, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Identity(),
                                 nn.Linear(dim * mult, dim), nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim: int, rotary_embed: RotaryEmbedding) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, HEADS, DIM_HEAD, rotary_embed),
                                                    FeedForward(dim)])])
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            attn, ff = cast(nn.ModuleList, layer)
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def _mlp(dim_in: int, dim_out: int, dim_hidden: int, depth: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(dims)):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class _BandSplit(nn.Module):
    def __init__(self, dims_in: tuple[int, ...]) -> None:
        super().__init__()
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, DIM)) for d in dims_in])


class _MaskEstimator(nn.Module):
    def __init__(self, dims_in: tuple[int, ...]) -> None:
        super().__init__()
        self.to_freqs = nn.ModuleList([nn.Sequential(_mlp(DIM, d * 2, DIM * 4, 2), nn.GLU(dim=-1)) for d in dims_in])


class MelBandRoformer(nn.Module):
    """Stereo, one stem (vocals), as in the official configuration."""

    freq_indices: Tensor
    num_bands_per_freq: Tensor
    window: Tensor

    def __init__(self) -> None:
        super().__init__()
        time_rotary, freq_rotary = RotaryEmbedding(DIM_HEAD), RotaryEmbedding(DIM_HEAD)
        self.layers = nn.ModuleList([nn.ModuleList([Transformer(DIM, time_rotary), Transformer(DIM, freq_rotary)])
                                     for _ in range(DEPTH)])
        mask = torch.from_numpy(mel_band_mask())                     # [bands, freqs]
        freq_indices = torch.arange(mask.shape[1]).repeat(NUM_BANDS, 1)[mask]
        freq_indices = (freq_indices[:, None] * 2 + torch.arange(2)).flatten()   # stereo: (f, s) interleaved
        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("num_bands_per_freq", mask.sum(0), persistent=False)
        dims_in = tuple(int(2 * f * 2) for f in mask.sum(1).tolist())
        self.band_split = _BandSplit(dims_in)
        self.dims_in = dims_in
        self.mask_estimators = nn.ModuleList([_MaskEstimator(dims_in)])
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    @property
    def estimator(self) -> _MaskEstimator:
        return cast(_MaskEstimator, self.mask_estimators[0])

    def forward(self, audio: Tensor) -> Tensor:
        """``audio`` [batch, 2, samples] at 44.1 kHz -> separated vocals, same shape."""
        b, channels, length = audio.shape
        spec = torch.stft(audio.reshape(b * channels, length), N_FFT, HOP, N_FFT, self.window, return_complex=True)
        spec = torch.view_as_real(spec).reshape(b, channels, *spec.shape[1:], 2)        # b s f t c
        spec = spec.permute(0, 2, 1, 3, 4).reshape(b, -1, spec.shape[3], 2)             # b (f s) t c
        x = spec[:, self.freq_indices].permute(0, 2, 1, 3).reshape(b, spec.shape[2], -1)  # b t (f c)
        x = torch.stack([net(part) for part, net in
                         zip(x.split(list(self.dims_in), dim=-1), self.band_split.to_features, strict=True)], dim=-2)
        for layer in self.layers:
            time_transformer, freq_transformer = cast(nn.ModuleList, layer)
            bands = x.shape[2]
            x = x.permute(0, 2, 1, 3).reshape(b * bands, -1, DIM)                       # (b f) t d
            x = time_transformer(x)
            x = x.reshape(b, bands, -1, DIM).permute(0, 2, 1, 3)
            frames = x.shape[1]
            x = freq_transformer(x.reshape(b * frames, bands, DIM)).reshape(b, frames, bands, DIM)
        masks = torch.cat([mlp(band) for band, mlp in
                           zip(x.unbind(dim=-2), self.estimator.to_freqs, strict=True)], dim=-1)  # b t (f c)
        masks = torch.view_as_complex(masks.reshape(b, masks.shape[1], -1, 2).permute(0, 2, 1, 3).contiguous())
        spec_c = torch.view_as_complex(spec.contiguous())                                 # b (f s) t
        summed = torch.zeros_like(spec_c).scatter_add_(
            1, self.freq_indices[None, :, None].expand(b, -1, spec_c.shape[-1]), masks.to(spec_c.dtype))
        denom = self.num_bands_per_freq.repeat_interleave(channels)[None, :, None].clamp(min=1)
        spec_c = spec_c * (summed / denom)
        spec_c = spec_c.reshape(b, -1, channels, spec_c.shape[-1]).permute(0, 2, 1, 3).reshape(b * channels, -1,
                                                                                               spec_c.shape[-1])
        spec_c[:, 0] = 0
        out = torch.istft(spec_c, N_FFT, HOP, N_FFT, self.window, length=length)
        return out.reshape(b, channels, length)


def load_separator(checkpoint: str, device: torch.device) -> MelBandRoformer:
    model = MelBandRoformer()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def separate_vocals(model: MelBandRoformer, audio: np.ndarray, progress=None) -> np.ndarray:
    """Vocals of mono 44.1 kHz ``audio`` (8 s passes, half-overlapping, Hann cross-fade)."""
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(audio, np.float32))
    length = len(x)
    step = CHUNK // OVERLAP
    padded = F.pad(x, (step, step + (-(length + 2 * step) % step) + CHUNK))
    window = torch.hann_window(CHUNK, periodic=False).clamp_min(1e-3)
    out = torch.zeros_like(padded)
    weight = torch.zeros_like(padded)
    starts = list(range(0, len(padded) - CHUNK + 1, step))
    for number, start in enumerate(starts):
        piece = padded[start:start + CHUNK].to(device)
        vocals = model(piece.expand(1, 2, -1)).mean(dim=1)[0].float().cpu()
        out[start:start + CHUNK] += vocals * window
        weight[start:start + CHUNK] += window
        if progress:
            progress((number + 1) / len(starts))
    return (out / weight.clamp_min(1e-6))[step:step + length].numpy()


def _count_parameters(model: nn.Module) -> int:   # pragma: no cover - debugging aid
    return sum(math.prod(p.shape) for p in model.parameters())
