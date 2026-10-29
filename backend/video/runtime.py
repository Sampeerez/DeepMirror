"""ONNX Runtime sessions for the face models, on the best execution provider available.

The CUDA build of ONNX Runtime uses the CUDA and cuDNN libraries that come
with the CUDA torch build, so torch is imported first. Light models (the face
detector, the landmark lifter) run on the CPU even with a GPU: they are fast
there and leave the GPU to the voice and the face swapper.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()


@cache
def gpu_providers() -> tuple[str, ...]:
    """Accelerated execution providers this ONNX Runtime build can use, best first."""
    import onnxruntime as ort
    import torch  # loads the CUDA and cuDNN libraries ONNX Runtime links against

    available = set(ort.get_available_providers())
    preferred = [name for name in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
                 if name in available]
    if "CUDAExecutionProvider" in preferred and not torch.cuda.is_available():
        preferred.remove("CUDAExecutionProvider")
    return tuple(preferred)


def session(path: Path, *, gpu: bool, threads: int | None = None):
    """An inference session for ``path``: on the GPU when ``gpu`` and one is usable, else the CPU."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if threads is not None:
        options.intra_op_num_threads = threads
    providers: list = ["CPUExecutionProvider"]
    accelerated = gpu_providers() if gpu else ()
    if "CUDAExecutionProvider" in accelerated:
        # Exhaustive search once per shape, then the fastest kernels every frame.
        providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC",
                                                       "arena_extend_strategy": "kSameAsRequested"}))
    elif accelerated:
        providers.insert(0, accelerated[0])
    with _lock:   # concurrent session creation on Windows can deadlock the CUDA provider
        created = ort.InferenceSession(str(path), options, providers=providers)
    logger.debug("Loaded %s on %s", path.name, created.get_providers()[0])
    return created


def cpu_threads() -> int:
    """Threads for one CPU face model: a share of the cores, so the voice keeps its own."""
    return max(1, min(4, (os.cpu_count() or 4) // 4))


def infer(model: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run ``model`` (an inference session) and return its outputs as arrays."""
    return cast(list[np.ndarray], model.run(None, feeds))
