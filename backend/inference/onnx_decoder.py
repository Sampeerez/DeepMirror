"""ONNX Runtime execution of the HiFi-GAN decoder on CPU.

On CPU the vocoder is ~75% of each realtime render, and ONNX Runtime's fused
convolution kernels run it about 1.5x faster than PyTorch eager (measured:
304 -> 196 ms for 0.9 s of 40 kHz audio). The decoder is exported once per
voice with the ``torch.export``-based exporter and cached; the PyTorch module
stays as the fallback. GPUs keep the PyTorch path, which is faster there.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from backend.inference.synthesizer import Synthesizer

logger = logging.getLogger(__name__)

Decoder = Callable[..., torch.Tensor]


def _export(synthesizer: Synthesizer, path: Path) -> None:
    decoder = synthesizer.dec.eval()
    frames_dim = torch.export.Dim("frames", min=8, max=60_000)
    z = torch.zeros(1, synthesizer.enc_p.out_channels, 50)
    g = torch.zeros(1, synthesizer.emb_g.embedding_dim, 1)
    if synthesizer.use_pitch:
        inputs: tuple[torch.Tensor, ...] = (z, torch.full((1, 50), 150.0), g)
        shapes: dict = {"x": {2: frames_dim}, "f0": {1: frames_dim}, "g": None}
    else:
        inputs, shapes = (z, g), {"x": {2: frames_dim}, "g": None}
    with torch.inference_mode():
        program = torch.onnx.export(decoder, inputs, dynamo=True, dynamic_shapes=shapes, verbose=False)
    assert program is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.onnx")
    program.save(str(partial))
    partial.replace(path)


def onnx_decoder(synthesizer: Synthesizer, cache_path: Path) -> Decoder | None:
    """A drop-in replacement for ``synthesizer.dec`` on ONNX Runtime, or None if unavailable."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        if not cache_path.is_file():
            logger.info("Exporting the CPU decoder to ONNX (once per voice)...")
            _export(synthesizer, cache_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        # PyTorch's OpenMP threads keep spinning between ops; spinning ORT
        # threads on top of them oversubscribe the CPU (measured: 432 ms with
        # spinning vs 310 ms without, for a full realtime render).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(str(cache_path), options, providers=["CPUExecutionProvider"])
    except Exception:
        logger.exception("ONNX decoder unavailable; using PyTorch")
        cache_path.unlink(missing_ok=True)
        return None
    names = [item.name for item in session.get_inputs()]

    def run(*tensors: torch.Tensor) -> torch.Tensor:
        feed = {name: np.ascontiguousarray(t.float().cpu().numpy()) for name, t in zip(names, tensors, strict=True)}
        return torch.from_numpy(session.run(None, feed)[0])

    logger.info("CPU decoder running on ONNX Runtime")
    return run
