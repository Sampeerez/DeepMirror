"""Install the 3D head (LAM, the Large Avatar Model) into models/lam. Windows, NVIDIA GPU.

    uv run python scripts/setup_head3d.py [--status data/head3d-install.json]

The studio runs this by itself on first start when the computer has an NVIDIA GPU with 6 GB or more
(see backend/video/head3d_install.py); it can also be run by hand. Nothing is compiled: everything
is downloaded (about 7 GB) and checked by SHA-256 where the file is ours to pin:

* Python 3.10 (managed by uv) and an environment of its own in models/lam/env, with PyTorch 2.3
  for CUDA 11.8 (the NVIDIA driver is all the GPU needs) and LAM's Python dependencies;
* the CUDA libraries LAM needs, prebuilt for GTX 10 to RTX 40 (and RTX 50 through PTX) by
  scripts/build_head3d_wheels.py, from this project's GitHub release;
* LAM's code at a fixed commit, its weights (LAM-20K) and its assets (FLAME, the face tracker's
  models), from GitHub and Hugging Face;
* MediaPipe's face landmarker.

Each step is skipped when already done, and downloads resume, so it can simply be run again.
Licences: LAM's code is Apache-2.0, its weights CC BY-NC 4.0 (non-commercial); FLAME's model files
are under the MPI FLAME licence (non-commercial); MediaPipe is Apache-2.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.video.avatar import NO_WINDOW, LamInstall  # noqa: E402

LAM_COMMIT = "3166feb82e47b1f4c9ad59d5c4f0f5b58d9b1194"
LAM_ZIP = f"https://github.com/aigc3d/LAM/archive/{LAM_COMMIT}.zip"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
TORCH = ["torch==2.3.0", "torchvision==0.18.0", "xformers==0.0.26.post1"]
# What LAM's model, its face tracker and the live driver import (LAM's requirements.txt also has
# its web demo's, TensorFlow and JAX among them, which neither needs).
REQUIREMENTS = ["numpy==1.23.0", "opencv-python-headless==4.11.0.86", "omegaconf==2.3.0", "safetensors",
                "accelerate", "einops", "transformers==4.41.2", "diffusers", "huggingface_hub==0.23.2",
                "pymcubes==0.1.6", "trimesh==4.4.9", "plyfile", "jaxtyping", "typeguard", "tyro", "loguru",
                "pyyaml", "scikit-image==0.20.0", "scipy", "pandas==2.2.3", "moviepy==1.0.3", "imageio",
                "iopath", "fvcore", "onnxruntime==1.19.2", "absl-py", "flatbuffers", "attrs",
                "protobuf>=4.25.3,<5", "matplotlib", "setuptools", "wheel", "Cython"]
# Built without isolation (its setup imports the environment's own pip and numpy).
SDIST_NO_ISOLATION = ["chumpy==0.70"]
MEDIAPIPE = "mediapipe==0.10.14"   # without its own NumPy 2 / OpenCV requirements

RELEASE = "https://github.com/Sampeerez/DeepMirror/releases/download/head3d-deps-1"
PREBUILT = {   # file: SHA-256 (scripts/build_head3d_wheels.py)
    "pytorch3d-0.7.7-cp310-cp310-win_amd64.whl": "7a1af043d7a0c511048a5399c6ffd1ebfd2889e5a98ba84bf13d3bf0af0a161f",
    "diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl": "e0e419591280fa1f63f89c95072fb45e1a3e1ee19113746dd957ccd88aa644da",
    "nvdiffrast-0.3.3-py3-none-any.whl": "adfe397f15fc3d223238e18734b65b845c3401dfe44bb53f4b09750b91efb237",
    "nvdiffrast_plugin.pyd": "658ba51978ed9798bff51076feb1b18ebc8fba9d972468c5c0d48ecb38ea5e67",
    "cpu_nms.cp310-win_amd64.pyd": "4091b63a2980b312325bbe23f246a6d4d62d25928d44a3d0e57e9c321fe7cb4f",
}
HF = "https://huggingface.co"
WEIGHTS = (f"{HF}/3DAIGC/LAM-20K/resolve/main/model.safetensors",
           "f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925")
WEIGHTS_CONFIG = f"{HF}/3DAIGC/LAM-20K/resolve/main/config.json"
ASSETS = {   # archive: SHA-256 (LAM's demo assets, LAM_assets.tar, are not needed)
    "thirdparty_models.tar": "2ac5eba7c1abca0ab6c682ad900ddb8819b713506043b04874599c82572931c8",
    "LAM_human_model.tar": "f8a1e10301937ac9d6169bf46d1bb9a1e73725a31e380f67d5fca3f12f92ef24",
}
LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
              "face_landmarker.task", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")
# The share of the whole install each step takes (by download size), for the progress shown.
STEPS = {"code": 0.01, "python": 0.02, "torch": 0.40, "packages": 0.10, "prebuilt": 0.02,
         "weights": 0.30, "assets": 0.13, "landmarker": 0.005, "check": 0.015}
SMOKE_TEST = """
import os, sys, importlib
sys.argv = ['x']; sys.path.insert(0, os.getcwd())
import torch
assert torch.cuda.is_available(), 'PyTorch does not see the GPU'
plugins = os.environ['DEEPMIRROR_LAM_PLUGINS']; sys.path.insert(0, plugins)
import nvdiffrast.torch.ops as ops
ops._cached_plugin[False] = importlib.import_module('nvdiffrast_plugin')
ops.RasterizeCudaContext()
from pytorch3d import _C
from diff_gaussian_rasterization import GaussianRasterizer
import mediapipe, onnxruntime
from lam.models import ModelLAM
from lam.runners.infer.head_utils import prepare_motion_seqs
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
print('ok', torch.cuda.get_device_name(0))
"""


class Progress:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.done = 0.0
        self.step = ""
        self._last = 0.0

    def write(self, state: str = "installing", error: str | None = None, force: bool = False,
              within: float = 0.0) -> None:
        now = time.monotonic()
        if self.path is None or (not force and now - self._last < 0.5):
            return
        self._last = now
        share = STEPS.get(self.step, 0.0)
        data = {"state": state, "step": self.step, "progress": round(min(1.0, self.done + share * within), 3),
                "error": error, "updated": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".part")
        partial.write_text(json.dumps(data), encoding="utf-8")
        partial.replace(self.path)

    def start(self, step: str) -> None:
        self.step = step
        print(f"\n== {step}", flush=True)
        self.write(force=True)

    def finish(self) -> None:
        self.done += STEPS.get(self.step, 0.0)
        self.write(force=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str | None, progress: Progress) -> Path:
    """``url`` to ``target``, resumed if a part is already there, checked against ``expected``."""
    if target.is_file() and (not expected or sha256(target) == expected):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        have = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "DeepMirror", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:   # the server ignored the range: start over
                    have = 0
                total = have + int(response.headers.get("Content-Length") or 0)
                with partial.open("ab" if have else "wb") as file:
                    while block := response.read(1 << 20):
                        file.write(block)
                        have += len(block)
                        if total:
                            progress.write(within=have / total)
            break
        except OSError as exc:
            print(f"  download interrupted ({exc}); retrying", flush=True)
            time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError(f"Could not download {url}")
    if expected and sha256(partial) != expected:
        partial.unlink()
        raise RuntimeError(f"{target.name} does not match its checksum")
    partial.replace(target)
    return target


def run(command: list, **kwargs) -> None:
    print("  $", " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, creationflags=NO_WINDOW, **kwargs)


def uv() -> str:
    found = shutil.which("uv")
    if found is None:
        raise RuntimeError("uv is needed (it runs the studio itself)")
    return found


def works(install: LamInstall, code: str) -> bool:
    if not install.python.is_file():
        return False
    return subprocess.run([str(install.python), "-c", code], capture_output=True, cwd=install.source,
                          env=install.run_environment(), creationflags=NO_WINDOW).returncode == 0


def install_all(install: LamInstall, progress: Progress) -> None:
    root = install.root
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    progress.start("code")
    if not (install.source / "app_lam.py").is_file():
        archive = download(LAM_ZIP, downloads / f"LAM-{LAM_COMMIT[:12]}.zip", None, progress)
        staging = root / "src.part"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(staging)
        (next(staging.iterdir())).rename(install.source)
        shutil.rmtree(staging, ignore_errors=True)
    progress.finish()

    progress.start("python")
    if not install.python.is_file():
        run([uv(), "venv", "--python", "3.10", "--seed", root / "env"])
    progress.finish()
    pip = [uv(), "pip", "install", "--python", install.python]

    progress.start("torch")
    if not works(install, "import torch, xformers; assert torch.version.cuda == '11.8'"):
        run([*pip, *TORCH, "--index-url", TORCH_INDEX])
    progress.finish()

    progress.start("packages")
    if not works(install, "import omegaconf, chumpy, mediapipe, onnxruntime, skimage, diffusers"):
        run([*pip, *REQUIREMENTS])
        run([*pip, "--no-build-isolation", *SDIST_NO_ISOLATION])
        run([*pip, "--no-deps", MEDIAPIPE])
    progress.finish()

    progress.start("prebuilt")
    for name, expected in PREBUILT.items():
        file = download(f"{RELEASE}/{name}", downloads / name, expected, progress)
        if name.endswith(".whl"):
            run([*pip, "--no-deps", "--reinstall", file])
        elif name.startswith("nvdiffrast"):
            install.plugins.mkdir(exist_ok=True)
            shutil.copy2(file, install.plugins / name)
        else:
            shutil.copy2(file, install.source / "external/landmark_detection/FaceBoxesV2/utils/nms" / name)
    progress.finish()

    progress.start("weights")
    weights = install.source / "model_zoo/lam_models/releases/lam/lam-20k/step_045500"
    download(WEIGHTS[0], weights / "model.safetensors", WEIGHTS[1], progress)
    download(WEIGHTS_CONFIG, weights / "config.json", None, progress)
    progress.finish()

    progress.start("assets")
    marker = install.source / "model_zoo" / ".assets-ok"
    if not marker.is_file():
        for name, expected in ASSETS.items():
            archive = download(f"{HF}/3DAIGC/LAM-assets/resolve/main/{name}", downloads / name, expected, progress)
            with tarfile.open(archive) as tar:
                tar.extractall(install.source, filter="data")
        marker.write_text("ok", encoding="utf-8")
    progress.finish()

    progress.start("landmarker")
    download(LANDMARKER[0], install.landmarker, LANDMARKER[1], progress)
    progress.finish()

    progress.start("check")
    result = subprocess.run([str(install.python), "-c", SMOKE_TEST], capture_output=True, text=True,
                            cwd=install.source, env=install.run_environment(), creationflags=NO_WINDOW)
    print(result.stdout[-2000:], result.stderr[-4000:], flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"The 3D head does not run here: {(result.stderr.strip().splitlines() or ['?'])[-1]}")
    progress.finish()
    shutil.rmtree(downloads, ignore_errors=True)   # the archives are unpacked: ~3 GB back


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", type=Path, help="write the progress here (JSON)")
    args = parser.parse_args()
    progress = Progress(args.status)
    if sys.platform != "win32":
        progress.write("failed", "The 3D head install is for Windows for now", force=True)
        sys.exit("The 3D head install is for Windows for now; on Linux follow LAM's own README into models/lam.")
    install = LamInstall.default()
    os.environ.pop("VIRTUAL_ENV", None)
    try:
        install_all(install, progress)
    except Exception as exc:
        progress.write("failed", str(exc), force=True)
        raise
    progress.done = 1.0
    progress.write("ready", force=True)
    print("\n3D head installed: Settings > Face > Clone > 3D head.")


if __name__ == "__main__":
    main()
