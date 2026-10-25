"""DeepFilterNet3 in plain PyTorch, run one 10 ms hop at a time.

A re-implementation of the network of Schröter et al. (DeepFilterNet,
github.com/Rikorose/DeepFilterNet, MIT or Apache-2.0; see LICENSE-DeepFilterNet
next to this file) for the official ``DeepFilterNet3`` checkpoint. Module and
parameter names follow the official code so its weights load unchanged; the
feature extraction and the streaming logic of the official Rust library
(libDF) are written here in torch, so no Rust extension is needed.

``StreamingDeepFilterNet.forward(frame, state, atten_lim_db)`` takes one hop
(480 samples at 48 kHz) and the flat recurrent state, and returns the
enhanced hop 1440 samples (30 ms) late, the new state and the local SNR.
Every operation is a plain tensor op (the FFTs are matrix products), so the
step exports to a single ONNX graph.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

SR = 48000
FFT = 960
HOP = 480
FREQS = FFT // 2 + 1
NB_ERB = 32
NB_DF = 96
DF_ORDER = 5
LOOKAHEAD = 2                 # conv and DF lookahead, in hops
CONV_CH = 64
HIDDEN = 256
ALPHA = 0.99                  # libDF calc_norm_alpha(48000, 480, tau=1 s)
MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)
LSNR_MIN, LSNR_MAX = -15.0, 35.0
DELAY = FFT - HOP + LOOKAHEAD * HOP   # 1440 samples


class GroupedLinear(nn.Module):
    """Block-diagonal linear layer (official GroupedLinearEinsum)."""

    def __init__(self, input_size: int, hidden_size: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, input_size // groups, hidden_size // groups))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.reshape(b, t, self.groups, -1)
        return torch.einsum("btgi,gih->btgh", x, self.weight).flatten(2, 3)


class SqueezedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int | None, num_layers: int,
                 linear_groups: int) -> None:
        super().__init__()
        self.linear_in = nn.Sequential(GroupedLinear(input_size, hidden_size, linear_groups), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_out = (nn.Sequential(GroupedLinear(hidden_size, output_size, linear_groups), nn.ReLU())
                           if output_size is not None else nn.Identity())

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        x, h = self.gru(self.linear_in(x), h)
        return self.linear_out(x), h


def conv_norm_act(cin: int, cout: int, kernel: tuple[int, int], fstride: int = 1, separable: bool = True,
                  activation: type[nn.Module] = nn.ReLU, transposed: bool = False) -> nn.Sequential:
    """Official Conv2dNormAct / ConvTranspose2dNormAct, without the causal time padding.

    The stream feeds each convolution exactly the frames it needs, so the
    padding layer is an Identity that only keeps the parameter numbering.
    """
    groups = math.gcd(cin, cout) if separable else 1
    pointwise = separable and groups > 1 and max(kernel) > 1
    fpad = kernel[1] // 2
    layers: list[nn.Module] = [nn.Identity()] if kernel[0] > 1 else []
    if transposed:
        layers.append(nn.ConvTranspose2d(cin, cout, kernel, padding=(kernel[0] - 1, fpad), output_padding=(0, fpad),
                                         stride=(1, fstride), groups=groups, bias=False))
    else:
        layers.append(nn.Conv2d(cin, cout, kernel, padding=(0, fpad), stride=(1, fstride), groups=groups, bias=False))
    if pointwise:
        layers.append(nn.Conv2d(cout, cout, 1, bias=False))
    layers += [nn.BatchNorm2d(cout), activation()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.erb_conv0 = conv_norm_act(1, CONV_CH, (3, 3))
        self.erb_conv1 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=2)
        self.erb_conv2 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=2)
        self.erb_conv3 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=1)
        self.df_conv0 = conv_norm_act(2, CONV_CH, (3, 3))
        self.df_conv1 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=2)
        emb = CONV_CH * NB_ERB // 4
        self.df_fc_emb = nn.Sequential(GroupedLinear(CONV_CH * NB_DF // 2, emb, 32), nn.ReLU())
        self.emb_gru = SqueezedGRU(emb, HIDDEN, emb, 1, 16)
        self.lsnr_fc = nn.Sequential(nn.Linear(emb, 1), nn.Sigmoid())


class ErbDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb = CONV_CH * NB_ERB // 4
        self.emb_gru = SqueezedGRU(emb, HIDDEN, emb, 2, 16)
        self.conv3p = conv_norm_act(CONV_CH, CONV_CH, (1, 1))
        self.convt3 = conv_norm_act(CONV_CH, CONV_CH, (1, 3))
        self.conv2p = conv_norm_act(CONV_CH, CONV_CH, (1, 1))
        self.convt2 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=2, transposed=True)
        self.conv1p = conv_norm_act(CONV_CH, CONV_CH, (1, 1))
        self.convt1 = conv_norm_act(CONV_CH, CONV_CH, (1, 3), fstride=2, transposed=True)
        self.conv0p = conv_norm_act(CONV_CH, CONV_CH, (1, 1))
        self.conv0_out = conv_norm_act(CONV_CH, 1, (1, 3), activation=nn.Sigmoid)


class DfDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb = CONV_CH * NB_ERB // 4
        self.df_convp = conv_norm_act(CONV_CH, DF_ORDER * 2, (5, 1))
        self.df_gru = SqueezedGRU(emb, HIDDEN, None, 2, 8)
        self.df_skip = GroupedLinear(emb, HIDDEN, 16)
        self.df_out = nn.Sequential(GroupedLinear(HIDDEN, NB_DF * DF_ORDER * 2, 16), nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())   # unused at inference


class MaskModule(nn.Module):
    erb_inv_fb: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("erb_inv_fb", torch.zeros(NB_ERB, FREQS))


# Flat recurrent state: (name, shape), in order.
STATE_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("analysis", (FFT - HOP,)),
    ("synthesis", (FFT - HOP,)),
    ("erb_norm", (NB_ERB,)),
    ("unit_norm", (NB_DF,)),
    ("erb_hist", (2, NB_ERB)),                       # last 2 ERB feature frames (erb_conv0 sees 3)
    ("spec_hist", (2 + 4, NB_DF, 2)),                # last 6 spectral feature frames (df_convp over 5 c0)
    ("noisy_hist", (DF_ORDER - 1, FREQS, 2)),        # last 4 noisy spectra (DF over frames t-4..t)
    ("h_enc", (1, HIDDEN)),
    ("h_erb", (2, HIDDEN)),
    ("h_df", (2, HIDDEN)),
)
STATE_SIZE = sum(int(np.prod(shape)) for _, shape in STATE_LAYOUT)


def initial_state() -> np.ndarray:
    parts = []
    for name, shape in STATE_LAYOUT:
        if name == "erb_norm":
            parts.append(np.linspace(*MEAN_NORM_INIT, NB_ERB))
        elif name == "unit_norm":
            parts.append(np.linspace(*UNIT_NORM_INIT, NB_DF))
        else:
            parts.append(np.zeros(int(np.prod(shape))))
    return np.concatenate(parts).astype(np.float32)


def _without_pad(block: nn.Sequential, x: Tensor) -> Tensor:
    for layer in list(block)[1:] if isinstance(block[0], nn.Identity) else block:
        x = layer(x)
    return x


class StreamingDeepFilterNet(nn.Module):
    """The whole DeepFilterNet3 stream for one 480-sample hop (see the module docstring)."""

    erb_fb: Tensor
    window: Tensor
    dft_re: Tensor
    dft_im: Tensor
    idft_re: Tensor
    idft_im: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("erb_fb", torch.zeros(FREQS, NB_ERB))
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = MaskModule()
        self.df_dec = DfDecoder()
        # Vorbis window and real DFT bases (FFT as matrix products: one ONNX graph, no FFT ops).
        n = np.arange(FFT)
        inner = np.sin(0.5 * np.pi * (n + 0.5) / (FFT // 2))
        window = np.sin(0.5 * np.pi * inner ** 2)
        k = np.arange(FREQS)
        angle = 2 * np.pi * np.outer(n, k) / FFT
        self.register_buffer("window", torch.tensor(window, dtype=torch.float32), persistent=False)
        # Forward: X = rfft(x) / FFT (libDF normalises in the analysis only).
        self.register_buffer("dft_re", torch.tensor(np.cos(angle) / FFT, dtype=torch.float32), persistent=False)
        self.register_buffer("dft_im", torch.tensor(-np.sin(angle) / FFT, dtype=torch.float32), persistent=False)
        # Inverse without normalisation (realfft): x[n] = sum_k c_k (Re X_k cos - Im X_k sin), c = 1 or 2.
        weight = np.full(FREQS, 2.0)
        weight[0] = weight[-1] = 1.0
        self.register_buffer("idft_re", torch.tensor((np.cos(angle) * weight).T, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("idft_im", torch.tensor((-np.sin(angle) * weight).T, dtype=torch.float32),
                             persistent=False)

    def _split(self, state: Tensor) -> dict[str, Tensor]:
        parts, offset = {}, 0
        for name, shape in STATE_LAYOUT:
            size = int(np.prod(shape))
            parts[name] = state[offset:offset + size].reshape(shape)
            offset += size
        return parts

    def forward(self, frame: Tensor, state: Tensor, atten_lim_db: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        s = self._split(state)
        # --- Analysis (libDF frame_analysis) -------------------------------------------------------
        buffer = torch.cat([s["analysis"], frame]) * self.window
        x_re, x_im = buffer @ self.dft_re, buffer @ self.dft_im              # [481]
        spec = torch.stack([x_re, x_im], dim=-1)                              # [481, 2]

        # --- Features (feat_erb with exponential mean norm, feat_cplx with unit norm) --------------
        erb = 10 * torch.log10((x_re * x_re + x_im * x_im) @ self.erb_fb + 1e-10)
        erb_norm = erb * (1 - ALPHA) + s["erb_norm"] * ALPHA
        feat_erb = (erb - erb_norm) / 40
        low = spec[:NB_DF]
        unit_norm = torch.sqrt((low * low).sum(-1)) * (1 - ALPHA) + s["unit_norm"] * ALPHA
        feat_spec = low / torch.sqrt(unit_norm).unsqueeze(-1)

        erb_win = torch.cat([s["erb_hist"], feat_erb[None]])                  # [3, 32]
        spec_win = torch.cat([s["spec_hist"], feat_spec[None]])               # [7, 96, 2]

        # --- Encoder -------------------------------------------------------------------------------
        enc = self.enc
        e0 = _without_pad(enc.erb_conv0, erb_win[None, None])                 # [1, 64, 1, 32]
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)                                                # [1, 64, 1, 8]
        c0_all = _without_pad(enc.df_conv0, spec_win.permute(2, 0, 1)[None])  # [1, 64, 5, 96]
        c1 = enc.df_conv1(c0_all[:, :, -1:])                                  # [1, 64, 1, 48]
        cemb = enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = e3.permute(0, 2, 3, 1).flatten(2) + cemb                        # [1, 1, 512]
        emb, h_enc = enc.emb_gru(emb, s["h_enc"].unsqueeze(1))
        lsnr = enc.lsnr_fc(emb) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN

        # --- ERB decoder: band gains ---------------------------------------------------------------
        dec = self.erb_dec
        d, h_erb = dec.emb_gru(emb, s["h_erb"].unsqueeze(1))
        d = d.reshape(1, 1, 8, CONV_CH).permute(0, 3, 1, 2)                   # [1, 64, 1, 8]
        d = dec.convt3(dec.conv3p(e3) + d)
        d = dec.convt2(dec.conv2p(e2) + d)
        d = dec.convt1(dec.conv1p(e1) + d)
        gains = dec.conv0_out(dec.conv0p(e0) + d).reshape(NB_ERB)             # [32]

        # --- DF decoder: complex filter coefficients ------------------------------------------------
        dfd = self.df_dec
        c, h_df = dfd.df_gru(emb, s["h_df"].unsqueeze(1))
        c = c + dfd.df_skip(emb)
        cp = _without_pad(dfd.df_convp, c0_all).permute(0, 2, 3, 1)           # [1, 1, 96, 10]
        coefs = (dfd.df_out(c).reshape(1, 1, NB_DF, DF_ORDER * 2) + cp).reshape(NB_DF, DF_ORDER, 2)

        # --- Enhance frame t-2 (libDF process) ------------------------------------------------------
        noisy = torch.cat([s["noisy_hist"], spec[None]])                      # frames t-4 .. t
        target = noisy[LOOKAHEAD]                                             # frame t-2
        masked = target * (gains @ self.mask.erb_inv_fb).unsqueeze(-1)
        window = noisy[:, :NB_DF].permute(1, 0, 2)                            # [96, 5, 2]
        df_re = (window[..., 0] * coefs[..., 0] - window[..., 1] * coefs[..., 1]).sum(1)
        df_im = (window[..., 0] * coefs[..., 1] + window[..., 1] * coefs[..., 0]).sum(1)
        filtered = torch.cat([torch.stack([df_re, df_im], -1), masked[NB_DF:]])
        level = lsnr.reshape(())
        # Both stages on every frame. libDF skips them by local SNR (silence below -10 dB, raw
        # speech above 30 dB); here that cost articulation in the converted voice (Pedro 0.80 ->
        # 0.77 in noise) for a quieter silence the voice detector already provides.
        out = filtered
        # Attenuation limit: mix the noisy frame back (100 dB or more means unlimited).
        lim = torch.where(atten_lim_db >= 100, torch.zeros_like(atten_lim_db), 10 ** (-atten_lim_db / 20))
        out = out * (1 - lim) + target * lim

        # --- Synthesis (libDF frame_synthesis, overlap-add) -----------------------------------------
        time = (out[:, 0] @ self.idft_re + out[:, 1] @ self.idft_im) * self.window
        enhanced = time[:HOP] + s["synthesis"]

        new_state = torch.cat([
            frame, time[HOP:], erb_norm, unit_norm, erb_win[1:].flatten(), spec_win[1:].flatten(),
            noisy[1:].flatten(), h_enc.flatten(), h_erb.flatten(), h_df.flatten(),
        ])
        return enhanced, new_state, level
