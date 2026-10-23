"""HuBERT-Base content encoder in plain PyTorch.

Reproduces fairseq's ``HubertModel.extract_features`` for the RVC
``hubert_base.pt`` checkpoint (12 post-norm transformer layers, 768 dims,
50 frames/s) without fairseq. Parameter names match the checkpoint so its
state dict loads directly after weight-norm folding.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from backend.inference.checkpoints import fold_weight_norm, load_trusted_state_dict

HUBERT_SHA256 = "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96"
# (channels, kernel, stride) of the waveform feature extractor: 320x downsampling.
_CONV_LAYERS = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
_EMBED_DIM = 768
_LAYERS = 12
_HEADS = 12


class _ConvBlock(nn.Module):
    """``Conv1d -> [GroupNorm in fp32] -> GELU`` (fairseq ``conv_layers.<i>.0`` / ``.2``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, group_norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride, bias=False)
        self.norm = nn.GroupNorm(out_channels, out_channels, affine=True) if group_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            norm = self.norm
            x = F.group_norm(x.float(), norm.num_groups, norm.weight.float(), norm.bias.float(),
                             norm.eps).to(x.dtype)
        return F.gelu(x)


class _FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for index, (channels, kernel, stride) in enumerate(_CONV_LAYERS):
            blocks.append(_ConvBlock(in_channels, channels, kernel, stride, group_norm=index == 0))
            in_channels = channels
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        for block in self.conv_layers:
            x = block(x)
        return x


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)
        self.v_proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)
        self.q_proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)
        self.out_proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, _ = x.shape
        head_dim = _EMBED_DIM // _HEADS

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, _HEADS, head_dim).transpose(1, 2)

        mask = None
        if key_padding_mask is not None and key_padding_mask.any():
            mask = ~key_padding_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(heads(self.q_proj(x)), heads(self.k_proj(x)),
                                                heads(self.v_proj(x)), attn_mask=mask)
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, _EMBED_DIM))


class _EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()
        self.self_attn_layer_norm = nn.LayerNorm(_EMBED_DIM)
        self.fc1 = nn.Linear(_EMBED_DIM, 3072)
        self.fc2 = nn.Linear(3072, _EMBED_DIM)
        self.final_layer_norm = nn.LayerNorm(_EMBED_DIM)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, key_padding_mask))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class _PositionalConv(nn.Module):
    """Grouped convolutional position embedding with fairseq's ``SamePad``."""

    def __init__(self, kernel: int = 128, groups: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv1d(_EMBED_DIM, _EMBED_DIM, kernel, padding=kernel // 2, groups=groups)
        self.trim = 1 if kernel % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.trim:
            y = y[:, :, :-self.trim]
        return F.gelu(y)


class _TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_conv = _PositionalConv()
        self.layers = nn.ModuleList(_EncoderLayer() for _ in range(_LAYERS))
        self.layer_norm = nn.LayerNorm(_EMBED_DIM)

    def forward(self, x: torch.Tensor, output_layer: int) -> torch.Tensor:
        x = self.layer_norm(x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2))
        # fairseq pads the sequence to an even length and masks the pad frame.
        pad = x.size(1) % 2
        mask = None
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            mask[:, -pad:] = True
        for layer in self.layers[:output_layer]:
            x = layer(x, mask)
        return x[:, : x.size(1) - pad]


class HubertModel(nn.Module):
    """Content features for RVC: layer 9 projected to 256 dims (v1) or layer 12 (v2)."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _FeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, _EMBED_DIM)
        self.encoder = _TransformerEncoder()
        self.final_proj = nn.Linear(_EMBED_DIM, 256)

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor, output_layer: int) -> torch.Tensor:
        """``waveform`` is ``[batch, samples]`` at 16 kHz; returns ``[batch, frames, 768]``."""
        features = self.feature_extractor(waveform).transpose(1, 2)
        features = self.post_extract_proj(self.layer_norm(features))
        return self.encoder(features, output_layer)

    @torch.no_grad()
    def content(self, waveform: torch.Tensor, version: str) -> torch.Tensor:
        if version == "v1":
            return self.final_proj(self.extract_features(waveform, 9))
        return self.extract_features(waveform, 12)


_UNUSED_KEYS = {"mask_emb", "label_embs_concat"}


def _rename_fairseq_key(key: str) -> str:
    """Map fairseq ``nn.Sequential`` child indices onto this module's names."""
    parts = key.split(".")
    if parts[:2] == ["feature_extractor", "conv_layers"] and parts[3] in {"0", "2"}:
        return ".".join([*parts[:3], "conv" if parts[3] == "0" else "norm", *parts[4:]])
    if parts[:3] == ["encoder", "pos_conv", "0"]:
        return ".".join(["encoder", "pos_conv", "conv", *parts[3:]])
    return key


def load_hubert(path: Path, device: torch.device, half: bool) -> HubertModel:
    """Build HuBERT from the pinned checkpoint and move it to ``device``."""
    state = fold_weight_norm(load_trusted_state_dict(path, HUBERT_SHA256))
    model = HubertModel()
    missing, unexpected = model.load_state_dict(
        {_rename_fairseq_key(k): v for k, v in state.items() if k not in _UNUSED_KEYS}, strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"HuBERT checkpoint layout mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    model = model.eval().to(device)
    return model.half() if half else model.float()
