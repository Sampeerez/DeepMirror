"""RVC synthesizers (v1/v2, with and without pitch), inference only.

Ported from RVC's ``infer_pack`` (MIT). Training-only parts (posterior
encoder, discriminators, TorchScript hooks) are dropped and weight norm is
folded into plain convolution weights at load time, so every layer here is a
regular ``nn.Conv1d``/``ConvTranspose1d``. Module and parameter names match
the original checkpoints exactly; the random number draws happen in the same
order, so a seeded run reproduces the reference implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1
SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class LayerNorm(nn.Module):
    """Channel-first layer norm (normalises dim 1 of ``[batch, channels, time]``)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# --------------------------------------------------------------------------
# Text (content) encoder: relative-position transformer
# --------------------------------------------------------------------------


class RelativeMultiHeadAttention(nn.Module):
    """Self-attention with windowed relative position embeddings (VITS style)."""

    def __init__(self, channels: int, n_heads: int, window_size: int = 10) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.emb_rel_k = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        batch, channels, length = key.shape
        query = query.view(batch, self.n_heads, self.k_channels, length).transpose(2, 3)
        key = key.view(batch, self.n_heads, self.k_channels, length).transpose(2, 3)
        value = value.view(batch, self.n_heads, self.k_channels, length).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        rel_keys = self._relative_embeddings(self.emb_rel_k, length)
        rel_logits = torch.matmul(query, rel_keys.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_to_absolute(rel_logits)
        scores = scores.masked_fill(mask == 0, -1e4)
        attention = F.softmax(scores, dim=-1)

        output = torch.matmul(attention, value)
        rel_values = self._relative_embeddings(self.emb_rel_v, length)
        output = output + torch.matmul(self._absolute_to_relative(attention), rel_values.unsqueeze(0))
        output = output.transpose(2, 3).contiguous().view(batch, channels, length)
        return self.conv_o(output)

    def _relative_embeddings(self, embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad > 0:
            embeddings = F.pad(embeddings, [0, 0, pad, pad, 0, 0])
        return embeddings[:, start: start + 2 * length - 1]

    @staticmethod
    def _relative_to_absolute(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.shape
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        flat = F.pad(x.view(batch, heads, length * 2 * length), [0, length - 1, 0, 0, 0, 0])
        return flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    @staticmethod
    def _absolute_to_relative(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.shape
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        flat = F.pad(x.view(batch, heads, length * length + length * (length - 1)), [length, 0, 0, 0, 0, 0])
        return flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]


class FeedForward(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size)

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        return F.pad(x, [(self.kernel_size - 1) // 2, self.kernel_size // 2, 0, 0, 0, 0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_1(self._same_pad(x * mask)))
        return self.conv_2(self._same_pad(x * mask)) * mask


class TransformerEncoder(nn.Module):
    def __init__(self, channels: int, filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(RelativeMultiHeadAttention(channels, n_heads) for _ in range(n_layers))
        self.norm_layers_1 = nn.ModuleList(LayerNorm(channels) for _ in range(n_layers))
        self.ffn_layers = nn.ModuleList(
            FeedForward(channels, filter_channels, kernel_size) for _ in range(n_layers)
        )
        self.norm_layers_2 = nn.ModuleList(LayerNorm(channels) for _ in range(n_layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        x = x * mask
        for attention, norm_1, ffn, norm_2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True
        ):
            x = norm_1(x + attention(x, attention_mask))
            x = norm_2(x + ffn(x, mask))
        return x * mask


class ContentEncoder(nn.Module):
    """Projects HuBERT features (+ coarse pitch) to the prior's mean/log-scale."""

    def __init__(self, feature_dim: int, out_channels: int, hidden_channels: int, filter_channels: int,
                 n_heads: int, n_layers: int, kernel_size: int, use_pitch: bool) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(feature_dim, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels) if use_pitch else None
        self.encoder = TransformerEncoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor | None,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone)
        if pitch is not None and self.emb_pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = F.leaky_relu(x * math.sqrt(self.hidden_channels), 0.1).transpose(1, -1)
        mask = _sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        stats = self.proj(self.encoder(x * mask, mask)) * mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        return mean, log_scale, mask


# --------------------------------------------------------------------------
# Normalising flow
# --------------------------------------------------------------------------


class WaveNet(nn.Module):
    """Gated dilated convolution stack used inside the coupling layers."""

    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int,
                 gin_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for index in range(n_layers):
            dilation = dilation_rate ** index
            self.in_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                            dilation=dilation, padding=_padding(kernel_size, dilation)))
            channels = 2 * hidden_channels if index < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, channels, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        g = self.cond_layer(g)
        width = self.hidden_channels
        for index, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x) + g[:, index * 2 * width:(index + 1) * 2 * width, :]
            acts = torch.tanh(x_in[:, :width]) * torch.sigmoid(x_in[:, width:])
            res_skip = res_skip_layer(acts)
            if index < self.n_layers - 1:
                x = (x + res_skip[:, :width]) * mask
                output = output + res_skip[:, width:]
            else:
                output = output + res_skip
        return output * mask


class CouplingLayer(nn.Module):
    """Mean-only affine coupling layer."""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int) -> None:
        super().__init__()
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)

    def _mean(self, x0: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.post(self.enc(self.pre(x0) * mask, mask, g)) * mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Training direction (posterior -> prior)."""
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        return torch.cat([x0, self._mean(x0, mask, g) + x1 * mask], 1)

    def inverse(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        return torch.cat([x0, (x1 - self._mean(x0, mask, g)) * mask], 1)


class Flip(nn.Module):
    """Channel reversal between coupling layers (keeps checkpoint indices aligned)."""


class ResidualCouplingFlow(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, gin_channels: int, n_flows: int = 4) -> None:
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(CouplingLayer(channels, hidden_channels, 5, 1, 3, gin_channels))
            self.flows.append(Flip())

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        for flow in self.flows:
            if isinstance(flow, CouplingLayer):
                x = flow(x, mask, g)
            else:
                x = torch.flip(x, [1])
        return x

    def inverse(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        for flow in reversed(self.flows):
            if isinstance(flow, CouplingLayer):
                x = flow.inverse(x, mask, g)
            else:
                x = torch.flip(x, [1])
        return x


# --------------------------------------------------------------------------
# HiFi-GAN vocoders
# --------------------------------------------------------------------------


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size)) for _ in dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_1, conv_2 in zip(self.convs1, self.convs2, strict=True):
            residual = conv_1(F.leaky_relu(x, LRELU_SLOPE))
            x = conv_2(F.leaky_relu(residual, LRELU_SLOPE)) + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
            for d in dilation[:2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class SineSource(nn.Module):
    """Harmonic-plus-noise excitation from a frame-rate F0 track (hn-NSF)."""

    def __init__(self, sample_rate: int, sine_amp: float = 0.1, noise_std: float = 0.003) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.l_linear = nn.Linear(1, 1)

    def forward(self, f0: torch.Tensor, upsample: int) -> torch.Tensor:
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2).float()  # [batch, frames, 1]
            rad = (f0 / self.sample_rate) % 1
            initial_phase = torch.rand(f0.shape[0], f0.shape[2], device=f0.device)
            initial_phase[:, 0] = 0
            rad[:, 0, :] = rad[:, 0, :] + initial_phase
            cumulative = torch.cumsum(rad, 1) * upsample
            cumulative = F.interpolate(cumulative.transpose(2, 1), scale_factor=float(upsample),
                                       mode="linear", align_corners=True).transpose(2, 1)
            rad = F.interpolate(rad.transpose(2, 1), scale_factor=float(upsample), mode="nearest").transpose(2, 1)
            cumulative %= 1
            wrapped = (cumulative[:, 1:, :] - cumulative[:, :-1, :]) < 0
            shift = torch.zeros_like(rad)
            shift[:, 1:, :] = wrapped * -1.0
            sine = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * torch.pi) * self.sine_amp
            voiced = F.interpolate((f0 > 0).float().transpose(2, 1), scale_factor=float(upsample),
                                   mode="nearest").transpose(2, 1)
            noise_amp = voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3
            excitation = sine * voiced + noise_amp * torch.randn_like(sine)
        excitation = excitation.to(dtype=self.l_linear.weight.dtype)
        return torch.tanh(self.l_linear(excitation))


class _HiFiGanBody(nn.Module):
    def __init__(self, initial_channel: int, resblock: str, resblock_kernel_sizes: list[int],
                 resblock_dilation_sizes: list[list[int]], upsample_rates: list[int],
                 upsample_initial_channel: int, upsample_kernel_sizes: list[int], gin_channels: int) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        block = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.stage_channels: list[int] = []
        channels = upsample_initial_channel
        for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            channels = upsample_initial_channel // (2 ** (index + 1))
            self.stage_channels.append(channels)
            self.ups.append(nn.ConvTranspose1d(upsample_initial_channel // (2 ** index), channels,
                                               kernel, rate, padding=(kernel - rate) // 2))
            for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(block(channels, kernel_size, dilation))
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def _resblocks(self, stage: int, x: torch.Tensor) -> torch.Tensor:
        blocks = list(self.resblocks)[stage * self.num_kernels:(stage + 1) * self.num_kernels]
        total = blocks[0](x)
        for block in blocks[1:]:
            total = total + block(x)
        return total / self.num_kernels

    def _finish(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class Generator(_HiFiGanBody):
    """HiFi-GAN generator for pitch-less (``nono``) models."""

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x) + self.cond(g)
        for stage, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, LRELU_SLOPE))
            x = self._resblocks(stage, x)
        return self._finish(x)


class NsfGenerator(_HiFiGanBody):
    """HiFi-GAN generator driven by a neural source-filter excitation."""

    def __init__(self, *args: object, sample_rate: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        upsample_rates: list[int] = list(args[4])  # type: ignore[arg-type]
        self.upsample = math.prod(upsample_rates)
        self.m_source = SineSource(sample_rate)
        self.noise_convs = nn.ModuleList()
        for index, channels in enumerate(self.stage_channels):
            if index + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[index + 1:])
                self.noise_convs.append(nn.Conv1d(1, channels, kernel_size=stride * 2, stride=stride,
                                                  padding=stride // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, channels, kernel_size=1))

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        source = self.m_source(f0, self.upsample).transpose(1, 2)
        x = self.conv_pre(x) + self.cond(g)
        for stage, (upsample, noise_conv) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = upsample(F.leaky_relu(x, LRELU_SLOPE)) + noise_conv(source)
            x = self._resblocks(stage, x)
        return self._finish(x)


# --------------------------------------------------------------------------
# Full synthesizer
# --------------------------------------------------------------------------


class Synthesizer(nn.Module):
    """``SynthesizerTrnMs{256,768}NSFsid[_nono]`` in a single class.

    Constructed from the checkpoint's positional ``config`` list:
    ``[spec_channels, segment_size, inter_channels, hidden_channels,
    filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock,
    resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
    upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
    gin_channels, sample_rate]``.
    """

    def __init__(self, config: list, *, version: str, use_pitch: bool) -> None:
        super().__init__()
        (_spec, _segment, inter_channels, hidden_channels, filter_channels, n_heads, n_layers,
         kernel_size, _dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
         upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
         gin_channels, sample_rate) = config[:18]
        if isinstance(sample_rate, str):
            sample_rate = SAMPLE_RATES[sample_rate]
        self.sample_rate = int(sample_rate)
        self.use_pitch = use_pitch
        self.speaker_count = int(spk_embed_dim)
        feature_dim = 256 if version == "v1" else 768
        self.enc_p = ContentEncoder(feature_dim, inter_channels, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, use_pitch)
        body = (inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels)
        self.dec: nn.Module = (NsfGenerator(*body, sample_rate=self.sample_rate) if use_pitch
                               else Generator(*body))
        self.flow = ResidualCouplingFlow(inter_channels, hidden_channels, gin_channels)
        self.emb_g = nn.Embedding(self.speaker_count, gin_channels)
        self.hop_length = math.prod(upsample_rates)

    @torch.no_grad()
    def infer(self, phone: torch.Tensor, lengths: torch.Tensor, pitch: torch.Tensor | None,
              pitchf: torch.Tensor | None, speaker: torch.Tensor) -> torch.Tensor:
        """Render waveform ``[batch, 1, samples]`` from features at 100 frames/s."""
        g = self.emb_g(speaker).unsqueeze(-1)
        mean, log_scale, mask = self.enc_p(phone, pitch if self.use_pitch else None, lengths)
        z_p = (mean + torch.exp(log_scale) * torch.randn_like(mean) * 0.66666) * mask
        z = self.flow.inverse(z_p, mask, g)
        if self.use_pitch:
            return self.dec(z * mask, pitchf, g)
        return self.dec(z * mask, g)
