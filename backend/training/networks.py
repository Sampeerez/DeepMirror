"""Training-time networks: the full RVC generator and its discriminators.

The generator reuses the inference :class:`Synthesizer` and adds what only
training needs: the posterior encoder (``enc_q``), the training forward pass
and weight normalisation on the same layers RVC normalises. Weight norm uses
``torch.nn.utils.parametrizations``; :func:`to_rvc_state` / :func:`from_rvc_state`
translate to and from the ``weight_g``/``weight_v`` layout of RVC checkpoints,
so the official pretrained models load and exported voices work in any RVC tool.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from backend.inference.synthesizer import (
    LRELU_SLOPE,
    CouplingLayer,
    NsfGenerator,
    Synthesizer,
    WaveNet,
    _sequence_mask,
)
from backend.training.hparams import TrainingConfig

_G_SUFFIX = ".parametrizations.weight.original0"
_V_SUFFIX = ".parametrizations.weight.original1"


def _padding(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


def _apply_weight_norm(modules: list[nn.Module]) -> None:
    for module in modules:
        weight_norm(module)


def _wavenet_layers(net: WaveNet) -> list[nn.Module]:
    return [net.cond_layer, *net.in_layers, *net.res_skip_layers]


class PosteriorEncoder(nn.Module):
    """Linear spectrogram -> latent ``z`` (training only; discarded on export)."""

    def __init__(self, spec_channels: int, out_channels: int, hidden_channels: int, gin_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, 5, 1, 16, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, spec: torch.Tensor, lengths: torch.Tensor, g: torch.Tensor):
        mask = _sequence_mask(lengths, spec.size(2)).unsqueeze(1).to(spec.dtype)
        stats = self.proj(self.enc(self.pre(spec) * mask, mask, g)) * mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * mask
        return z, mean, log_scale, mask


def slice_segments(x: torch.Tensor, starts: torch.Tensor, size: int) -> torch.Tensor:
    index = starts.view(-1, 1) + torch.arange(size, device=x.device).view(1, -1)
    if x.dim() == 2:
        return torch.gather(x, 1, index)
    return torch.gather(x, 2, index.unsqueeze(1).expand(-1, x.size(1), -1))


class TrainableSynthesizer(nn.Module):
    """RVC v2 generator with pitch, ready for GAN fine-tuning."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.synth = Synthesizer(config.synthesizer_config(), version="v2", use_pitch=True)
        self.enc_q = PosteriorEncoder(config.n_fft // 2 + 1, config.inter_channels, config.hidden_channels,
                                      config.gin_channels)
        dec = self.synth.dec
        assert isinstance(dec, NsfGenerator)
        normalised: list[nn.Module] = list(dec.ups)
        for block in dec.resblocks:
            normalised += [*getattr(block, "convs1", []), *getattr(block, "convs2", []), *getattr(block, "convs", [])]
        for flow in self.synth.flow.flows:
            if isinstance(flow, CouplingLayer):
                normalised += _wavenet_layers(flow.enc)
        normalised += _wavenet_layers(self.enc_q.enc)
        _apply_weight_norm(normalised)

    def forward(self, phone, phone_lengths, pitch, pitchf, spec, spec_lengths, speaker):
        synth = self.synth
        g = synth.emb_g(speaker).unsqueeze(-1)
        m_p, logs_p, _ = synth.enc_p(phone, pitch, phone_lengths)
        z, _, logs_q, y_mask = self.enc_q(spec, spec_lengths, g)
        z_p = synth.flow(z, y_mask, g)
        frames = self.config.segment_frames
        max_start = (spec_lengths - frames).clamp_min(0) + 1
        starts = (torch.rand(z.size(0), device=z.device) * max_start).long()
        z_slice = slice_segments(z, starts, frames)
        pitchf_slice = slice_segments(pitchf, starts, frames)
        audio = synth.dec(z_slice, pitchf_slice, g)
        return audio, starts, y_mask, (z_p, m_p, logs_p, logs_q)


class _PeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(_padding(5), 0))
            for i in range(5)
        )
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        _apply_weight_norm([*self.convs, self.conv_post])

    def forward(self, x: torch.Tensor):
        batch, channels, length = x.shape
        if length % self.period:
            pad = self.period - length % self.period
            x = F.pad(x, (0, pad), "reflect")
            length += pad
        x = x.view(batch, channels, length // self.period, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class _ScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 16, 15, 1, padding=7),
            nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
            nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)
        _apply_weight_norm([*self.convs, self.conv_post])

    def forward(self, x: torch.Tensor):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    """RVC v2 discriminator: one scale discriminator plus eight period discriminators."""

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_ScaleDiscriminator(), *(_PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11, 17, 23, 37))])

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_scores, fake_scores, real_features, fake_features = [], [], [], []
        for discriminator in self.discriminators:
            score, features = discriminator(real)
            real_scores.append(score)
            real_features.append(features)
            score, features = discriminator(fake)
            fake_scores.append(score)
            fake_features.append(features)
        return real_scores, fake_scores, real_features, fake_features


# --------------------------------------------------------------------------
# Losses (VITS / HiFi-GAN)
# --------------------------------------------------------------------------


def discriminator_loss(real_scores, fake_scores) -> torch.Tensor:
    return sum(torch.mean((1 - r.float()) ** 2) + torch.mean(f.float() ** 2)
               for r, f in zip(real_scores, fake_scores, strict=True))  # type: ignore[return-value]


def generator_loss(fake_scores) -> torch.Tensor:
    return sum(torch.mean((1 - f.float()) ** 2) for f in fake_scores)  # type: ignore[return-value]


def feature_loss(real_features, fake_features) -> torch.Tensor:
    loss = sum(torch.mean(torch.abs(r.float().detach() - f.float()))
               for real, fake in zip(real_features, fake_features, strict=True)
               for r, f in zip(real, fake, strict=True))
    return loss * 2  # type: ignore[return-value]


def kl_loss(z_p, logs_q, m_p, logs_p, mask) -> torch.Tensor:
    z_p, logs_q, m_p, logs_p, mask = (t.float() for t in (z_p, logs_q, m_p, logs_p, mask))
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * mask) / torch.sum(mask)


# --------------------------------------------------------------------------
# Checkpoint layout conversion
# --------------------------------------------------------------------------


def _generator_prefix(key: str) -> str:
    """``synth.`` holds every generator module except the posterior encoder."""
    return key if key.startswith("enc_q.") else f"synth.{key}"


def from_rvc_state(state: dict[str, torch.Tensor], *, generator: bool) -> dict[str, torch.Tensor]:
    """RVC checkpoint keys (``weight_g``/``weight_v``) -> this module's parametrised keys."""
    converted = {}
    for key, value in state.items():
        if key.endswith(".weight_g"):
            key = key[: -len(".weight_g")] + _G_SUFFIX
        elif key.endswith(".weight_v"):
            key = key[: -len(".weight_v")] + _V_SUFFIX
        converted[_generator_prefix(key) if generator else key] = value
    return converted


def to_rvc_state(state: dict[str, torch.Tensor], *, generator: bool) -> dict[str, torch.Tensor]:
    """Inverse of :func:`from_rvc_state` (keys only; values untouched)."""
    converted = {}
    for key, value in state.items():
        if generator and key.startswith("synth."):
            key = key[len("synth."):]
        if key.endswith(_G_SUFFIX):
            key = key[: -len(_G_SUFFIX)] + ".weight_g"
        elif key.endswith(_V_SUFFIX):
            key = key[: -len(_V_SUFFIX)] + ".weight_v"
        converted[key] = value
    return converted


def export_voice(generator: TrainableSynthesizer, epoch: int) -> dict:
    """Inference checkpoint in the standard RVC format (fp16 weights, no posterior encoder)."""
    state = to_rvc_state(generator.state_dict(), generator=True)
    weights = {k: v.detach().half().cpu() for k, v in state.items() if not k.startswith("enc_q.")}
    config = generator.config
    return {"weight": weights, "config": config.synthesizer_config(), "info": f"{epoch}epoch",
            "sr": config.tag, "f0": 1, "version": "v2"}
