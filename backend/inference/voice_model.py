"""A loaded RVC voice: synthesizer weights plus checkpoint metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from backend.inference.checkpoints import CheckpointError, fold_weight_norm, load_rvc_checkpoint
from backend.inference.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VoiceModel:
    name: str
    version: str          # "v1" (256-dim features) or "v2" (768-dim)
    use_pitch: bool
    sample_rate: int      # synthesizer output rate
    speaker_count: int
    synthesizer: Synthesizer

    @classmethod
    def load(cls, path: Path, device: torch.device, half: bool, *, allow_unsafe: bool = False) -> VoiceModel:
        checkpoint = load_rvc_checkpoint(path, allow_unsafe=allow_unsafe)
        version = str(checkpoint.get("version", "v1"))
        if version not in {"v1", "v2"}:
            raise CheckpointError(f"{path.name}: unsupported RVC version {version!r}")
        use_pitch = bool(int(checkpoint.get("f0", 1)))
        config = list(checkpoint["config"])
        weights = checkpoint["weight"]
        config[-3] = int(weights["emb_g.weight"].shape[0])  # speaker count from the embedding

        synthesizer = Synthesizer(config, version=version, use_pitch=use_pitch)
        state = {k: v for k, v in fold_weight_norm(weights).items() if not k.startswith("enc_q.")}
        missing, unexpected = synthesizer.load_state_dict(state, strict=False)
        if missing:
            raise CheckpointError(f"{path.name} is missing weights: {', '.join(missing[:5])}")
        if unexpected:
            logger.debug("%s: ignoring %d unused weights", path.name, len(unexpected))
        synthesizer = synthesizer.eval().to(device)
        synthesizer = synthesizer.half() if half else synthesizer.float()
        logger.info("Loaded voice %s (RVC %s, %s, %d Hz, %d speakers)", path.name, version,
                    "pitch" if use_pitch else "no pitch", synthesizer.sample_rate, synthesizer.speaker_count)
        return cls(path.name, version, use_pitch, synthesizer.sample_rate, synthesizer.speaker_count, synthesizer)

    def describe(self) -> dict:
        return {"engine": "rvc", "version": self.version, "f0": self.use_pitch,
                "target_sample_rate": self.sample_rate, "speakers": self.speaker_count}
