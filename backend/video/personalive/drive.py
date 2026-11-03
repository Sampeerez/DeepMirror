"""PersonaLive's process, in its own environment (models/personalive): head crops in, generated head
crops out, four at a time. The studio's side is client.py. Imports nothing from the studio but the
GPU arbiter, by its file: this is not the studio's Python.

    python drive.py <compute_arbiter.py> <arbiter's name>     (cwd: PersonaLive's code)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from collections import deque
from collections.abc import Callable

import numpy as np

CHUNK = 4         # frames PersonaLive takes and gives at a time
CONFIG = "configs/prompts/personalive_online.yaml"
ARBITER_WAIT = 0.1   # a chunk does not fit in the voice's gaps: this long for a free GPU, then it runs anyway
MEASURE_CHUNKS = 8
CPU_VARIABLE = "DEEPMIRROR_PERSONALIVE_CPU"   # to check it works without a big GPU: about 20 s per frame


def to_tensor(crops: list[np.ndarray], torch, device):
    """BGR uint8 crops to what ``process_input`` takes: RGB, -1..1, (frames, channels, height, width)."""
    stack = torch.from_numpy(np.stack(crops)[..., ::-1].copy()).to(device).float() / 255.0
    return (stack * 2.0 - 1.0).permute(0, 3, 1, 2)


def to_crops(video: np.ndarray) -> list[np.ndarray]:
    """``process_input``'s output (RGB, 0..1, frames x height x width x 3) to BGR uint8 crops."""
    return [np.ascontiguousarray((frame[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)) for frame in video]


def serve(conn, pipeline, tensor: Callable[[list[np.ndarray]], object], arbiter,
          to_pil: Callable[[np.ndarray], object]) -> None:
    """Until "stop". ``pipeline`` is PersonaLive's wrapper: ``reset``, ``fuse_reference``, ``process_input``."""
    pending: list[tuple[int, float, np.ndarray]] = []
    # It denoises several windows at once and gives back the oldest: what comes out is the frames
    # handed in this many chunks ago (and, before that, its fade in from the photo: nothing's frames).
    delay = getattr(pipeline, "temporal_adaptive_step", 1) - 1
    history: deque[list[tuple[int, float]]] = deque()
    expected_ms = 300.0
    portrait: np.ndarray | None = None

    def generate(crops: list[np.ndarray]) -> np.ndarray:
        nonlocal expected_ms
        with arbiter.video(expected_ms, timeout=ARBITER_WAIT):
            started = time.perf_counter()
            video = pipeline.process_input(tensor(crops))
            expected_ms = 0.8 * expected_ms + 0.2 * (time.perf_counter() - started) * 1000.0
        return video

    def fuse() -> None:
        assert portrait is not None
        pipeline.reset()
        with arbiter.video(100.0, timeout=5.0):
            pipeline.fuse_reference(to_pil(portrait))

    while True:
        message = conn.recv()
        kind = message[0]
        if kind == "stop":
            return
        if kind == "portrait":
            portrait = message[1]
            pending.clear()
            history.clear()
            fuse()
            conn.send(("portrait_ready",))
        elif kind == "frame" and portrait is not None:
            pending.append(message[1:])
            if len(pending) == CHUNK:
                chunk, pending = pending, []
                video = generate([crop for _, _, crop in chunk])
                history.append([(seq, t) for seq, t, _ in chunk])
                if len(history) > delay:
                    conn.send(("frames", [(seq, t, image) for (seq, t), image
                                          in zip(history.popleft(), to_crops(video), strict=False)]))
        elif kind == "measure":
            # The photo driving itself: the time does not depend on the motion.
            took = None
            if portrait is not None:
                times, broken = [], False
                for _ in range(MEASURE_CHUNKS):
                    started = time.perf_counter()
                    broken |= bool(np.isnan(generate([portrait] * CHUNK)).any())   # fp16 on a GTX 16
                    times.append((time.perf_counter() - started) * 1000.0 / CHUNK)
                took = None if broken else float(np.median(times[2:]))
                history.clear()
                fuse()
            conn.send(("measured", took))


def main() -> None:
    from multiprocessing.connection import Client
    from types import SimpleNamespace

    on_cpu = bool(os.environ.get(CPU_VARIABLE))
    if on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import torch
    from PIL import Image  # pyright: ignore[reportMissingImports]

    host, port, key = os.environ["DEEPMIRROR_PERSONALIVE"].split(":")
    conn = Client((host, int(port)), authkey=bytes.fromhex(key))
    spec = importlib.util.spec_from_file_location("compute_arbiter", sys.argv[1])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arbiter = module.GpuArbiter(sys.argv[2])
    sys.path.insert(0, os.getcwd())
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    from src.wrapper import PersonaLive  # pyright: ignore[reportMissingImports]

    device = torch.device("cpu" if on_cpu else "cuda")
    config = CONFIG
    if on_cpu:   # half precision is for the GPU
        from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

        settings = OmegaConf.load(CONFIG)
        settings.dtype = "fp32"
        config = os.path.join(os.environ.get("TEMP", "."), "personalive_cpu.yaml")
        OmegaConf.save(settings, config)
    pipeline = PersonaLive(SimpleNamespace(config_path=config), device)
    # It turns xFormers on by itself, but its own note says it crashes on the RTX 50 series.
    if not on_cpu and " RTX 5" in torch.cuda.get_device_name(0):
        pipeline.reference_unet.disable_xformers_memory_efficient_attention()
        pipeline.denoising_unet.disable_xformers_memory_efficient_attention()
    conn.send(("ready",))
    try:
        serve(conn, pipeline, lambda crops: to_tensor(crops, torch, device), arbiter,
              lambda crop: Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    except (EOFError, OSError):
        pass
    finally:
        arbiter.close()


if __name__ == "__main__":
    main()
