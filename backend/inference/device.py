"""Compute device and precision selection, in one place."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# GPUs whose FP16 path is slow or numerically unreliable for RVC: GTX 16xx
# (Turing without tensor cores) and Pascal-era cards, which emulate FP16.
_FP32_ONLY_MARKERS = ("GTX 16", "GTX 10", "1060", "1070", "1080", "P40", "P100", "MX1", "MX2", "MX3")


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    """Where inference runs, plus the upstream RVC segmentation budget.

    ``offline_pad_seconds``/``query``/``center``/``max`` are the RVC
    ``x_pad``/``x_query``/``x_center``/``x_max`` values (seconds) used to split
    and pad whole-file renders; they scale with available memory.
    """

    device: torch.device
    half: bool
    name: str
    memory_gb: float | None
    offline_pad_seconds: int
    query_seconds: int
    center_seconds: int
    max_seconds: int

    @property
    def is_accelerator(self) -> bool:
        return self.device.type != "cpu"

    def describe(self) -> dict:
        return {"device": str(self.device), "name": self.name, "half_precision": self.half,
                "memory_gb": self.memory_gb}


def select_device(preference: str = "auto") -> ComputeDevice:
    """Pick CUDA, MPS or XPU when available (or as requested), else CPU."""
    preference = (preference or "auto").lower()
    device, half, name, memory = torch.device("cpu"), False, "CPU", None

    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        half = not any(marker in name.upper() for marker in _FP32_ONLY_MARKERS)
    elif preference in {"auto", "mps"} and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, half, name = torch.device("mps"), False, "Apple Silicon (MPS)"
    elif preference in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
        device, half, name = torch.device("xpu:0"), True, "Intel XPU"
    elif preference not in {"auto", "cpu"}:
        logger.warning("Requested device %r is not available; using CPU", preference)

    if memory is not None and memory <= 4.5:
        budget = (1, 5, 30, 32)
    elif half:
        budget = (3, 10, 60, 65)
    else:
        budget = (1, 6, 38, 41)
    selected = ComputeDevice(device, half, name, round(memory, 1) if memory else None, *budget)
    logger.info("Inference device: %s (%s, %s)", selected.name, selected.device,
                "fp16" if selected.half else "fp32")
    return selected
