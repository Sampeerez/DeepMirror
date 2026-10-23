"""Hardware detection and realtime-safe setting recommendations.

The result is informational: it is exposed through ``/api/config`` so the UI
can show what the server runs on and which buffer/F0 combination fits the
realtime budget. Detection imports torch (slow on first import), so it runs
once and is cached for the life of the process.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Detected compute capabilities."""

    gpu_type: str = "none"  # cuda | mps | xpu | directml | none
    gpu_name: str = "None"
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    cpu_memory_gb: float = 0.0

    @property
    def has_gpu(self) -> bool:
        return self.gpu_type != "none"


@dataclass(frozen=True, slots=True)
class RecommendedSettings:
    """Streaming settings that keep per-chunk processing under the chunk budget.

    Chunk sizes are in samples at 48 kHz: 2048 is about 43 ms, 4096 about
    85 ms and 8192 about 171 ms. With the fixed playback priming of one chunk,
    end-to-end latency is roughly 2-3 chunks plus network and device buffers.
    """

    chunk_size: int
    f0_method: str


def _detect_gpu() -> tuple[str, str, float]:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return "cuda", torch.cuda.get_device_name(0), props.total_memory / 1024**3
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps", "Apple Silicon", 0.0
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return "xpu", "Intel XPU", 0.0
        except Exception:
            logger.debug("Torch accelerator probe failed", exc_info=True)

    try:
        import onnxruntime as ort

        if "DmlExecutionProvider" in ort.get_available_providers():
            return "directml", "DirectML GPU", 0.0
    except Exception:
        pass
    return "none", "None", 0.0


def detect_hardware() -> HardwareProfile:
    """Probe GPU and CPU without loading any model."""
    gpu_type, gpu_name, gpu_memory_gb = _detect_gpu()
    cpu_memory_gb = 0.0
    try:
        import psutil

        cpu_memory_gb = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    return HardwareProfile(
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        gpu_memory_gb=round(gpu_memory_gb, 1),
        cpu_cores=os.cpu_count() or 1,
        cpu_memory_gb=round(cpu_memory_gb, 1),
    )


def recommend_settings(profile: HardwareProfile) -> RecommendedSettings:
    """Map a hardware profile to a realtime-safe buffer and pitch detector.

    RMVPE is only recommended where torch can run it on an accelerator; on CPU
    its cost per chunk exceeds the budget, so PM (cheap, asset-free) is used.
    DirectML only accelerates ONNX models, so it is treated like CPU for RVC.
    """
    if profile.gpu_type == "cuda":
        # Measured on a 6 GB FP32 card: RMVPE needs ~81 ms of an 85 ms
        # 4096-sample chunk, so smaller cards get a larger buffer.
        chunk = 4096 if profile.gpu_memory_gb >= 8 else 8192
        return RecommendedSettings(chunk_size=chunk, f0_method="rmvpe")
    if profile.gpu_type in ("mps", "xpu"):
        return RecommendedSettings(chunk_size=4096, f0_method="rmvpe")
    chunk = 4096 if profile.cpu_cores >= 8 else 8192
    return RecommendedSettings(chunk_size=chunk, f0_method="pm")


@lru_cache(maxsize=1)
def get_hardware_optimized_config() -> dict[str, Any]:
    """Return the cached hardware profile and its recommended settings."""
    profile = detect_hardware()
    recommended = recommend_settings(profile)
    logger.info(
        "Hardware: gpu=%s (%s), cpu_cores=%d; recommended chunk=%d f0=%s",
        profile.gpu_type, profile.gpu_name, profile.cpu_cores,
        recommended.chunk_size, recommended.f0_method,
    )
    return {
        "hardware_profile": {**asdict(profile), "has_gpu": profile.has_gpu},
        "recommended_settings": asdict(recommended),
    }
