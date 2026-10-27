"""Training hyper-parameters (RVC v2 configurations) and pretrained base models."""

from __future__ import annotations

from dataclasses import dataclass

from backend.assets import PINNED_REVISION, Asset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    sample_rate: int
    tag: str                     # "40k" / "48k", as stored in RVC checkpoints
    n_fft: int
    hop: int
    win: int
    n_mels: int
    segment_samples: int         # random training crop of the waveform
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875   # per epoch
    c_mel: float = 45.0
    c_kl: float = 1.0
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    speaker_count: int = 109     # kept at the pretrained size so its embedding loads
    gin_channels: int = 256

    @property
    def segment_frames(self) -> int:
        return self.segment_samples // self.hop

    def synthesizer_config(self) -> list:
        """The positional ``config`` list stored in RVC checkpoints."""
        return [self.n_fft // 2 + 1, 32, self.inter_channels, self.hidden_channels, self.filter_channels,
                self.n_heads, self.n_layers, self.kernel_size, 0, "1", [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]], list(self.upsample_rates), 512,
                list(self.upsample_kernel_sizes), self.speaker_count, self.gin_channels, self.sample_rate]


CONFIGS = {
    40000: TrainingConfig(40000, "40k", n_fft=2048, hop=400, win=2048, n_mels=125, segment_samples=12800,
                          upsample_rates=(10, 10, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4)),
    48000: TrainingConfig(48000, "48k", n_fft=2048, hop=480, win=2048, n_mels=128, segment_samples=17280,
                          upsample_rates=(12, 10, 2, 2), upsample_kernel_sizes=(24, 20, 4, 4)),
}

_PRETRAINED = f"{PINNED_REVISION}/pretrained_v2"
_TITAN = "https://huggingface.co/blaise-tk/TITAN/resolve/cb72bb5bb05a85f07b17a33b014b694eaa9525f7/models/medium"
OFFICIAL = {
    40000: (
        Asset("pretrained-G40k", "f0G40k.pth", f"{_PRETRAINED}/f0G40k.pth",
              "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb", 73106273, False),
        Asset("pretrained-D40k", "f0D40k.pth", f"{_PRETRAINED}/f0D40k.pth",
              "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09", 142875703, False),
    ),
    48000: (
        Asset("pretrained-G48k", "f0G48k.pth", f"{_PRETRAINED}/f0G48k.pth",
              "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4", 75465569, False),
        Asset("pretrained-D48k", "f0D48k.pth", f"{_PRETRAINED}/f0D48k.pth",
              "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748", 142875703, False),
    ),
}
# TITAN Medium (blaise-tk, Apache-2.0): the official base fine-tuned on 11 h of expressive speech.
TITAN = {
    40000: (
        Asset("titan-G40k", "G-f040k-TITAN-Medium.pth", f"{_TITAN}/40k/pretrained/G-f040k-TITAN-Medium.pth",
              "217de2fd349f48856aa6a6d9d1257b07a0addccde7b69fc7b22097f1fee5924d", 438156650, False),
        Asset("titan-D40k", "D-f040k-TITAN-Medium.pth", f"{_TITAN}/40k/pretrained/D-f040k-TITAN-Medium.pth",
              "bc804da752ba9cb8e3aaa90ad102edc0a6e7f033c90db53bfcce135f100d5ad1", 857119946, False),
    ),
    48000: (
        Asset("titan-G48k", "G-f048k-TITAN-Medium.pth", f"{_TITAN}/48k/pretrained/G-f048k-TITAN-Medium.pth",
              "51a6dc93687f7e1a6051be3dec958289958c9c738d5c520e4fa956e7886ee153", 452338845, False),
        Asset("titan-D48k", "D-f048k-TITAN-Medium.pth", f"{_TITAN}/48k/pretrained/D-f048k-TITAN-Medium.pth",
              "4999a5b66aec0a9ab7e845063eeb242e47a611ab0e6260fb0b5ca44a7c5bbc44", 857126469, False),
    ),
}
BASES = {"official": OFFICIAL, "titan": TITAN}
# Mel-Band RoFormer "vocals" by Kimberley Jensen (MIT): keeps only the voice of training recordings.
SEPARATOR = Asset("separator", "MelBandRoformer.ckpt",
                  "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/"
                  "ac9b0614ab3cd7f77219e18ba494dfd93956c348/MelBandRoformer.ckpt",
                  "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e", 913106900, False)
DEFAULT_BASE = "titan"   # measured: closer timbre and steadier expressive pitch than the official base
BASE_FOLDERS = {"official": "pretrained_v2", "titan": "pretrained_titan"}

# Dataset limits. Five minutes of clean speech trains a good voice with the
# automatic plan (validation picks the best epoch); 10-30 minutes covers more
# of the speaker's pitch range and phonemes. Below a couple of minutes timbre
# and pitch range are under-sampled.
MIN_AUDIO_SECONDS = 60
RECOMMENDED_AUDIO_SECONDS = (300, 1800)
MAX_AUDIO_SECONDS = 3600
DEFAULT_EPOCHS = 300
MAX_EPOCHS = 2000
