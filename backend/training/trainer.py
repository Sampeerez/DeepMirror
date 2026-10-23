"""RVC v2 GAN fine-tuning loop.

Per step: render a random crop with the generator, update the discriminator on
real vs generated audio, then update the generator with adversarial +
feature-matching + mel (x45) + KL losses, exactly as RVC does. Precision is
chosen per device: bf16 autocast on GPUs with native bf16, fp16 with loss
scaling where FP16 is reliable, fp32 elsewhere (GTX 16xx, Pascal, MPS, CPU).
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F

from backend.audio.mel import linear_spectrogram, log_mel, mel_filterbank
from backend.inference.device import ComputeDevice
from backend.training.dataset import Segment
from backend.training.hparams import TrainingConfig
from backend.training.networks import (
    MultiPeriodDiscriminator,
    TrainableSynthesizer,
    discriminator_loss,
    export_voice,
    feature_loss,
    from_rvc_state,
    generator_loss,
    kl_loss,
    slice_segments,
)

logger = logging.getLogger(__name__)

MAX_BATCH = 16


@dataclass(slots=True)
class Batch:
    phone: torch.Tensor
    phone_lengths: torch.Tensor
    pitch: torch.Tensor
    pitchf: torch.Tensor
    spec: torch.Tensor
    spec_lengths: torch.Tensor
    wave: torch.Tensor
    speaker: torch.Tensor


@dataclass(slots=True)
class Example:
    """A segment aligned to one frame grid (100 frames/s at the model hop)."""

    phone: torch.Tensor   # [frames, 768]
    pitch: torch.Tensor   # [frames]
    pitchf: torch.Tensor  # [frames]
    wave: torch.Tensor    # [frames * hop]


@dataclass(slots=True)
class TrainingProgress:
    epoch: int
    total_epochs: int
    step: int
    steps_per_epoch: int
    losses: dict[str, float] = field(default_factory=dict)
    epoch_seconds: float | None = None
    batch_size: int = 0


def auto_batch_size(compute: ComputeDevice) -> int:
    """Largest batch that comfortably fits: ~0.9 GB per example in fp32, ~0.55 GB with mixed precision."""
    if compute.device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(compute.device)
        per_example = 0.55 if compute.half else 0.9
        return int(np.clip((free / 1024**3 - 1.0) / per_example, 1, MAX_BATCH))
    return 4 if compute.device.type == "mps" else 2


def build_examples(segments: list[Segment], config: TrainingConfig) -> list[Example]:
    examples = []
    for segment in segments:
        assert segment.phone is not None and segment.pitch is not None and segment.pitchf is not None
        phone = np.repeat(segment.phone, 2, axis=0)  # HuBERT 50 frames/s -> 100 frames/s
        frames = min(len(phone), len(segment.pitch), len(segment.audio) // config.hop)
        if frames <= config.segment_frames:
            continue
        examples.append(Example(
            torch.from_numpy(phone[:frames].astype(np.float32)),
            torch.from_numpy(segment.pitch[:frames].astype(np.int64)),
            torch.from_numpy(segment.pitchf[:frames].astype(np.float32)),
            torch.from_numpy(segment.audio[: frames * config.hop].astype(np.float32)),
        ))
    return examples


class Trainer:
    def __init__(self, config: TrainingConfig, compute: ComputeDevice, examples: list[Example],
                 checkpoint_dir: Path, batch_size: int | None = None) -> None:
        if not examples:
            raise ValueError("No usable training audio after preprocessing.")
        self.config = config
        self.compute = compute
        self.device = compute.device
        self.examples = examples
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = min(batch_size or auto_batch_size(compute), len(examples))
        self.generator = TrainableSynthesizer(config).to(self.device)
        self.discriminator = MultiPeriodDiscriminator().to(self.device)
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), config.learning_rate, betas=config.betas,
                                         eps=config.eps)
        self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), config.learning_rate,
                                         betas=config.betas, eps=config.eps)
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, config.lr_decay)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, config.lr_decay)
        self.mel_basis = torch.from_numpy(mel_filterbank(config.sample_rate, config.n_fft, config.n_mels)).to(
            self.device)
        self.autocast_dtype = self._precision()
        self.scaler = GradScaler(self.device.type, enabled=self.autocast_dtype == torch.float16)
        self.epoch = 0
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _precision(self) -> torch.dtype | None:
        if self.device.type != "cuda":
            return None
        if torch.cuda.get_device_capability(self.device)[0] >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if self.compute.half else None

    @property
    def precision_label(self) -> str:
        return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(self.autocast_dtype, "fp32")  # type: ignore[arg-type]

    # --- Checkpoints --------------------------------------------------------

    def load_pretrained(self, generator_path: Path, discriminator_path: Path) -> None:
        generator = torch.load(generator_path, map_location="cpu", weights_only=True)["model"]
        discriminator = torch.load(discriminator_path, map_location="cpu", weights_only=True)["model"]
        self.generator.load_state_dict(from_rvc_state(generator, generator=True))
        self.discriminator.load_state_dict(from_rvc_state(discriminator, generator=False))

    def save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": self.epoch, "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(), "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(), "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(), "scaler": self.scaler.state_dict(),
        }
        partial = self.checkpoint_dir / "latest.pt.part"
        torch.save(state, partial)
        partial.replace(self.checkpoint_dir / "latest.pt")

    def resume(self) -> bool:
        path = self.checkpoint_dir / "latest.pt"
        if not path.is_file():
            return False
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        for name in ("optim_g", "optim_d", "sched_g", "sched_d", "scaler"):
            getattr(self, name).load_state_dict(state[name])
        self.epoch = int(state["epoch"])
        return True

    def export(self, path: Path) -> None:
        partial = path.with_suffix(path.suffix + ".part")
        torch.save(export_voice(self.generator, self.epoch), partial)
        partial.replace(path)

    # --- Training -----------------------------------------------------------

    def _collate(self, items: list[Example]) -> Batch:
        hop = self.config.hop
        frames = max(len(item.pitch) for item in items)
        size = len(items)
        phone = torch.zeros(size, frames, items[0].phone.shape[1])
        pitch = torch.zeros(size, frames, dtype=torch.long)
        pitchf = torch.zeros(size, frames)
        wave = torch.zeros(size, frames * hop)
        lengths = torch.tensor([len(item.pitch) for item in items])
        for row, item in enumerate(items):
            n = len(item.pitch)
            phone[row, :n], pitch[row, :n], pitchf[row, :n] = item.phone, item.pitch, item.pitchf
            wave[row, : n * hop] = item.wave
        device = self.device
        wave = wave.to(device, non_blocking=True)
        spec = torch.zeros(size, self.config.n_fft // 2 + 1, frames, device=device)
        for row, item in enumerate(items):  # per item, so padding never leaks into the edge frames
            n = len(item.pitch)
            spec[row, :, :n] = linear_spectrogram(wave[row:row + 1, : n * hop], self.config.n_fft, hop,
                                                  self.config.win)[0, :, :n]
        lengths = lengths.to(device)
        return Batch(phone.to(device, non_blocking=True), lengths, pitch.to(device, non_blocking=True),
                     pitchf.to(device, non_blocking=True), spec, lengths, wave.unsqueeze(1),
                     torch.zeros(size, dtype=torch.long, device=device))

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def _step(self, batch: Batch) -> dict[str, float]:
        config = self.config
        with self._autocast():
            audio, starts, y_mask, (z_p, m_p, logs_p, logs_q) = self.generator(
                batch.phone, batch.phone_lengths, batch.pitch, batch.pitchf, batch.spec, batch.spec_lengths,
                batch.speaker)
            real = slice_segments(batch.wave.squeeze(1), starts * config.hop, config.segment_samples).unsqueeze(1)
            real_scores, fake_scores, _, _ = self.discriminator(real, audio.detach())
            loss_d = discriminator_loss(real_scores, fake_scores)
        self.optim_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_d).backward()
        self.scaler.step(self.optim_d)

        with self._autocast():
            real_scores, fake_scores, real_features, fake_features = self.discriminator(real, audio)
        real_mel = slice_segments(log_mel(batch.spec, self.mel_basis), starts, config.segment_frames)
        fake_mel = log_mel(linear_spectrogram(audio.float().squeeze(1), config.n_fft, config.hop, config.win),
                           self.mel_basis)
        loss_mel = F.l1_loss(real_mel, fake_mel) * config.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * config.c_kl
        loss_fm = feature_loss(real_features, fake_features)
        loss_gen = generator_loss(fake_scores)
        loss_g = loss_gen + loss_fm + loss_mel + loss_kl
        self.optim_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.optim_g)
        self.scaler.update()
        return {"mel": loss_mel.item(), "kl": loss_kl.item(), "fm": loss_fm.item(), "gen": loss_gen.item(),
                "disc": loss_d.item()}

    def train(self, total_epochs: int, on_progress: Callable[[TrainingProgress], None],
              should_stop: Callable[[], bool], on_epoch_end: Callable[[int], None]) -> str:
        """Train until ``total_epochs``; returns ``"done"`` or ``"stopped"``."""
        self.generator.train()
        self.discriminator.train()
        while self.epoch < total_epochs:
            started = time.perf_counter()
            order = list(self.examples)
            random.shuffle(order)
            steps = math.ceil(len(order) / self.batch_size)
            totals: dict[str, float] = {}
            step = 0
            try:
                for step in range(steps):
                    if should_stop():
                        return "stopped"
                    losses = self._step(self._collate(order[step * self.batch_size:(step + 1) * self.batch_size]))
                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.0) + value
                    if step % 5 == 0:
                        on_progress(TrainingProgress(self.epoch + 1, total_epochs, step + 1, steps,
                                                     {k: v / (step + 1) for k, v in totals.items()},
                                                     batch_size=self.batch_size))
            except torch.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning("Out of GPU memory; retrying the epoch with batch size %d", self.batch_size)
                self.optim_g.zero_grad(set_to_none=True)
                self.optim_d.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            self.epoch += 1
            self.sched_g.step()
            self.sched_d.step()
            on_progress(TrainingProgress(self.epoch, total_epochs, steps, steps,
                                         {k: v / max(step + 1, 1) for k, v in totals.items()},
                                         time.perf_counter() - started, self.batch_size))
            on_epoch_end(self.epoch)
        return "done"
