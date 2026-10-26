"""DeepFilterNet3 built from the official checkpoint.

``export_streaming_onnx`` loads ``DeepFilterNet3.zip`` (Rikorose/DeepFilterNet,
MIT or Apache-2.0) into the network in :mod:`.network` and writes a one-hop
streaming ONNX graph, once, on this computer.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_MEMBER = "DeepFilterNet3/checkpoints/model_120.ckpt.best"
EXPORT_VERSION = 2            # bump when the network or the graph interface changes


def load_network(checkpoint_zip: Path):
    """The streaming network with the official weights, in eval mode."""
    import torch

    from backend.audio.deepfilter.network import StreamingDeepFilterNet

    with zipfile.ZipFile(checkpoint_zip) as archive:
        weights = torch.load(io.BytesIO(archive.read(CHECKPOINT_MEMBER)), map_location="cpu", weights_only=True)
    network = StreamingDeepFilterNet()
    missing, unexpected = network.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DeepFilterNet3 checkpoint does not match: missing {missing}, unexpected {unexpected}")
    return network.eval()


def export_streaming_onnx(checkpoint_zip: Path, target: Path) -> Path:
    """Export the one-hop streaming graph to ``target`` (atomically) and check it against PyTorch."""
    import onnxruntime as ort
    import torch

    from backend.audio.deepfilter.network import HOP, initial_state

    network = load_network(checkpoint_zip)
    frame = torch.zeros(HOP)
    state = torch.from_numpy(initial_state())
    limit = torch.tensor(100.0)
    partial = target.with_suffix(".part.onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(network, (frame, state, limit), str(partial), dynamo=False, opset_version=17,
                          input_names=["input_frame", "states", "atten_lim_db"],
                          output_names=["enhanced_audio_frame", "new_states", "lsnr"])
    try:
        _check(network, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    del ort
    logger.info("Exported DeepFilterNet3 to %s", target.name)
    return target


def _check(network, onnx_path: Path) -> None:
    """The exported graph must match PyTorch on noise and a tone (a broken export never ships)."""
    import onnxruntime as ort
    import torch

    from backend.audio.deepfilter.network import HOP, initial_state

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    t = np.arange(HOP * 40) / 48000
    audio = (0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    state_torch = torch.from_numpy(initial_state())
    state_onnx = initial_state()
    worst = 0.0
    with torch.no_grad():
        for k in range(len(audio) // HOP):
            hop = audio[k * HOP:(k + 1) * HOP]
            ref, state_torch, _ = network(torch.from_numpy(hop), state_torch, torch.tensor(100.0))
            out, state_onnx, _ = session.run(None, {"input_frame": hop, "states": state_onnx,
                                                    "atten_lim_db": np.array(100.0, np.float32)})
            worst = max(worst, float(np.abs(np.asarray(out) - ref.numpy()).max()))
    if worst > 1e-4:
        raise RuntimeError(f"DeepFilterNet3 export differs from PyTorch by {worst:.2e}")
