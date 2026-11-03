"""Install PersonaLive (diffusion: the whole head generated, hair included) into models/personalive.
Windows, NVIDIA GPU with 12 GB or more. Experimental.

    uv run python scripts/setup_personalive.py [--status data/personalive-install.json]

The studio runs this the first time Clone > AI is chosen (backend/video/personalive/client.py);
OVC_PERSONALIVE=off turns that off. About 16 GB of downloads, each weight checked by SHA-256, in an
environment of its own (Python 3.10, PyTorch 2.1): about 20 GB once installed. Licence: code under
Apache-2.0, but its authors release the project for academic research only; so it is for
non-commercial use here too.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup_head3d import Progress, download, run, uv, works  # noqa: E402

from backend.video.avatar import NO_WINDOW  # noqa: E402
from backend.video.personalive.client import PersonaLiveInstall  # noqa: E402

COMMIT = "abdd112e01dcf7d89122c2e5efa29fcff0669740"
CODE_ZIP = f"https://github.com/GVCLab/PersonaLive/archive/{COMMIT}.zip"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
TORCH = ["torch==2.1.0", "torchvision==0.16.0", "xformers==0.0.22.post7"]
# requirements_base.txt, only what its model code imports (not its web demo's).
REQUIREMENTS = ["accelerate==1.12.0", "av", "diffusers==0.27.0", "einops==0.8.1", "huggingface_hub==0.25.1",
                "numpy==1.26.3", "omegaconf==2.3.0", "opencv-python-headless==4.10.0.84", "Pillow==11.3.0",
                "safetensors==0.7.0", "tqdm==4.67.1", "transformers==4.36.2", "peft==0.10.0"]
HF = "https://huggingface.co"
PERSONALIVE = f"{HF}/huaichang/PersonaLive/resolve/main/pretrained_weights/personalive"
BASE = f"{HF}/lambda/sd-image-variations-diffusers/resolve/main"   # PersonaLive's script names its old address
VAE = f"{HF}/stabilityai/sd-vae-ft-mse/resolve/main"
WEIGHTS = {   # file under pretrained_weights: (url, SHA-256)
    "personalive/denoising_unet.pth":
        (f"{PERSONALIVE}/denoising_unet.pth", "d0446c4d2387f259d5f3c1ac54a5aefa93400f4672f942856bff2538df046162"),
    "personalive/motion_encoder.pth":
        (f"{PERSONALIVE}/motion_encoder.pth", "ff7c6b0a84cd750046e7687f7a6f6bbc21317055bfcacef950ed347debae4d2c"),
    "personalive/motion_extractor.pth":
        (f"{PERSONALIVE}/motion_extractor.pth", "251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5"),
    "personalive/pose_guider.pth":
        (f"{PERSONALIVE}/pose_guider.pth", "8b997db63343a6a5d489778172d9544bcccaf27e6756505dc6353d84e877269d"),
    "personalive/reference_unet.pth":
        (f"{PERSONALIVE}/reference_unet.pth", "85eb03e6c34fab69f9246ff14b3016789232e56dc4892d0581fea21a3a8480f6"),
    "personalive/temporal_module.pth":
        (f"{PERSONALIVE}/temporal_module.pth", "295e8942a453adb48756432d99de103ecba9b840b5b8f6635a0687311cdff30e"),
    "sd-vae-ft-mse/diffusion_pytorch_model.bin":
        (f"{VAE}/diffusion_pytorch_model.bin", "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"),
    "sd-vae-ft-mse/config.json": (f"{VAE}/config.json", None),
    "sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin":
        (f"{BASE}/unet/diffusion_pytorch_model.bin", "ee23e3368e4e7c0e4ef636ed61923609c97fcaa583f8bb416e3e0986d4a0cfc6"),
    "sd-image-variations-diffusers/unet/config.json": (f"{BASE}/unet/config.json", None),
    "sd-image-variations-diffusers/image_encoder/pytorch_model.bin":
        (f"{BASE}/image_encoder/pytorch_model.bin", "89d2aa29b5fdf64f3ad4f45fb4227ea98bc45156bbae673b85be1af7783dbabb"),
    "sd-image-variations-diffusers/image_encoder/config.json": (f"{BASE}/image_encoder/config.json", None),
    "sd-image-variations-diffusers/model_index.json": (f"{BASE}/model_index.json", None),
}
STEPS = {"code": 0.01, "python": 0.01, "torch": 0.2, "packages": 0.05, "weights": 0.7, "check": 0.03}
SMOKE_TEST = """
import sys, torch
sys.path.insert(0, ".")
assert torch.cuda.is_available()
# A real operation: this PyTorch may have no kernels for the newest GPUs (it only fails when used).
torch.nn.functional.conv2d(torch.zeros(1, 1, 4, 4).cuda(), torch.zeros(1, 1, 3, 3).cuda())
import src.wrapper
print(torch.cuda.get_device_name(0))
"""


def install_all(install: PersonaLiveInstall, progress: Progress) -> None:
    root = install.root
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    progress.start("code")
    if not (install.source / "inference_online.py").is_file():
        archive = download(CODE_ZIP, downloads / f"PersonaLive-{COMMIT[:12]}.zip", None, progress)
        staging = root / "src.part"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(staging)
        next(staging.iterdir()).rename(install.source)
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
    if not works(install, "import av, cv2, diffusers, einops, omegaconf, transformers"):
        run([*pip, *REQUIREMENTS])
    progress.finish()

    progress.start("weights")
    for name, (url, expected) in WEIGHTS.items():
        download(url, install.source / "pretrained_weights" / name, expected, progress)
    progress.finish()

    progress.start("check")
    install.marker.unlink(missing_ok=True)
    result = subprocess.run([str(install.python), "-c", SMOKE_TEST], capture_output=True, text=True,
                            cwd=install.source, env=install.run_environment(), creationflags=NO_WINDOW)
    print(result.stdout[-2000:], result.stderr[-4000:], flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"PersonaLive does not run here: {(result.stderr.strip().splitlines() or ['?'])[-1]}")
    install.marker.write_text(json.dumps({"personalive": COMMIT, "gpu": result.stdout.strip()}), encoding="utf-8")
    progress.finish()
    shutil.rmtree(downloads, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", type=Path, help="write the progress here (JSON)")
    args = parser.parse_args()
    progress = Progress(args.status, STEPS)
    if sys.platform != "win32":
        progress.write("failed", "The PersonaLive install is for Windows for now", force=True)
        sys.exit("The PersonaLive install is for Windows for now; on Linux follow its own README into models/personalive.")
    os.environ.pop("VIRTUAL_ENV", None)
    try:
        install_all(PersonaLiveInstall.default(), progress)
    except Exception as exc:
        progress.write("failed", str(exc), force=True)
        raise
    progress.done = 1.0
    progress.write("ready", force=True)
    print("\nPersonaLive installed.")


if __name__ == "__main__":
    main()
