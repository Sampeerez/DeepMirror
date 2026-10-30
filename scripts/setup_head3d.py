"""Install the 3D head (LAM, the Large Avatar Model) into models/lam. Windows + NVIDIA GPU.

    uv run python scripts/setup_head3d.py

Needs, installed beforehand (the script checks and says how):
  * Visual Studio 2019 Build Tools with the C++ workload (CUDA 11.8 accepts no newer compiler):
        winget install Microsoft.VisualStudio.2019.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  * CUDA Toolkit 11.8 (the toolkit only; the display driver stays):
        winget install Nvidia.CUDA --version 11.8
  * git and uv (as for the studio itself).

What it does, each step skipped when already done (so it can be run again after a failure):
LAM's code at a fixed commit, a Python 3.10 environment with PyTorch 2.3 (CUDA 11.8), four CUDA
libraries compiled for this GPU (pytorch3d, diff-gaussian-rasterization, nvdiffrast,
simple-knn), LAM's weights and assets (~7 GB, Hugging Face) and MediaPipe's face landmarker.
Licences: LAM's code is Apache-2.0, its weights CC BY-NC 4.0 (non-commercial), FLAME's model
files are under the MPI FLAME licence (non-commercial), MediaPipe Apache-2.0.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.video.avatar import LamInstall, _cuda_home, _vcvars  # noqa: E402

LAM = ("https://github.com/aigc3d/LAM.git", "3166feb82e47b1f4c9ad59d5c4f0f5b58d9b1194")
EXTENSIONS = {   # folder: (repository, commit)
    "pytorch3d": ("https://github.com/facebookresearch/pytorch3d.git", "89653419d0973396f3eff1a381ba09a07fffc2ed"),
    "diffgs": ("https://github.com/ashawkey/diff-gaussian-rasterization.git", "8829d14f814fccdaf840b7b0f3021a616583c0a1"),
    "nvdiffrast": ("https://github.com/ShenhanQian/nvdiffrast.git", "22718580f24a313c429ba2c304794c264351f108"),
    "simpleknn": ("https://github.com/camenduru/simple-knn.git", "60f461f4a56b7967e5d8045bf92f8c33f36976d0"),
}
CHECK = {   # module that proves each one is built
    "pytorch3d": "from pytorch3d import _C", "diffgs": "import diff_gaussian_rasterization",
    "nvdiffrast": "import nvdiffrast.torch", "simpleknn": "import simple_knn._C",
}
TORCH = ["torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0", "xformers==0.0.26.post1"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
# LAM's requirements.txt without the four compiled libraries (built from source below).
REQUIREMENTS = ["pymcubes==0.1.6", "omegaconf==2.3.0", "moviepy==1.0.3", "pandas==2.2.3", "transformers==4.41.2",
                "numpy==1.23.0", "trimesh==4.4.9", "opencv_python_headless==4.11.0.86", "tensorflow==2.12.0",
                "face-detection-tflite==0.6.0", "scikit-image==0.20.0", "jaxlib==0.4.30", "gradio==3.44.3",
                "huggingface_hub==0.23.2", "Cython", "accelerate", "tyro", "einops", "diffusers", "plyfile",
                "jaxtyping", "typeguard", "chumpy", "loguru", "ninja"]
# The studio's side: MediaPipe without its NumPy 2 / OpenCV dependencies, and ONNX Runtime for the parser.
EXTRAS = ["absl-py", "flatbuffers", "attrs", "onnxruntime==1.19.2", "numpy==1.23.0"]
LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
              "face_landmarker.task", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")
WEIGHTS_SHA256 = "f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925"
ASSETS = ("LAM_assets.tar", "thirdparty_models.tar", "LAM_human_model.tar")


def step(text: str) -> None:
    print(f"\n== {text}", flush=True)


def run(command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("  $", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkout(folder: Path, repository: str, commit: str) -> None:
    if not (folder / ".git").is_dir():
        run(["git", "clone", "--recursive", repository, str(folder)])
    run(["git", "-C", str(folder), "checkout", "-q", commit])
    run(["git", "-C", str(folder), "submodule", "update", "--init", "--recursive", "-q"])


def works(python: Path, code: str) -> bool:
    return subprocess.run([str(python), "-c", f"import torch; {code}"], capture_output=True).returncode == 0


def main() -> None:
    install = LamInstall.default()
    root = install.root
    root.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        sys.exit("This installer is for Windows; on Linux follow LAM's own README into models/lam.")
    missing = [name for name, found in (("Visual Studio 2019 Build Tools", _vcvars()), ("CUDA 11.8", _cuda_home()),
                                        ("git", shutil.which("git")), ("uv", shutil.which("uv"))) if not found]
    if missing:
        sys.exit("Install first: " + ", ".join(missing) + " (see the top of this script).")

    step("LAM's code")
    checkout(install.source, *LAM)

    step("Python 3.10 environment")
    python = install.python
    if not python.is_file():
        run(["uv", "venv", "--python", "3.10", "--seed", str(root / "env")])
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if not works(python, "import xformers"):
        run([*pip, *TORCH, "--index-url", TORCH_INDEX])
    if not works(python, "import omegaconf, chumpy, tensorflow"):
        run([*pip, "pip", "setuptools", "wheel", "Cython", "numpy==1.23.0"])
        run([*pip, "--no-build-isolation", *REQUIREMENTS])

    env = install.build_environment()
    env.update(FORCE_CUDA="1", MAX_JOBS="8")
    for folder, (repository, commit) in EXTENSIONS.items():
        step(f"Compiling {folder} (CUDA)")
        if works(python, CHECK[folder]):
            print("  already built")
            continue
        checkout(root / folder, repository, commit)
        if folder == "pytorch3d":   # MSVC: its C++17 flag, and Windows headers without the min/max macros
            setup = root / folder / "setup.py"
            text = setup.read_text(encoding="utf-8")
            text = text.replace('{"cxx": ["-std=c++17"]}',
                                '{"cxx": ["/std:c++17" if sys.platform == "win32" else "-std=c++17"]}')
            if '"-DWIN32_LEAN_AND_MEAN",' not in text:
                text = text.replace('"-DCUDA_HAS_FP16=1",', '"-DCUDA_HAS_FP16=1",\n            "-DWIN32_LEAN_AND_MEAN",')
            setup.write_text(text, encoding="utf-8")
        run([*pip, "--no-build-isolation", "."], cwd=root / folder, env=env)
        if not works(python, CHECK[folder]):
            sys.exit(f"{folder} did not build; see the output above.")

    step("LAM's face box detector")
    boxes = install.source / "external/landmark_detection/FaceBoxesV2/utils"
    if not list((boxes / "nms").glob("cpu_nms*.pyd")):
        run([str(python), "build.py", "build_ext", "--inplace"], cwd=boxes, env=env)

    step("MediaPipe and ONNX Runtime")
    if not works(python, "import mediapipe, onnxruntime"):
        run([*pip, "--no-deps", "mediapipe==0.10.14"])
        run([*pip, *EXTRAS])
    landmarker = install.landmarker
    if not landmarker.is_file() or sha256(landmarker) != LANDMARKER[1]:
        landmarker.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LANDMARKER[0], landmarker)
        if sha256(landmarker) != LANDMARKER[1]:
            sys.exit("The face landmarker download does not match its checksum.")

    step("LAM's weights and assets (~7 GB)")
    weights = install.source / "model_zoo/lam_models/releases/lam/lam-20k/step_045500"
    downloads = root / "dl"
    if not (weights / "model.safetensors").is_file():
        run([str(python), "-c", "from huggingface_hub import snapshot_download as d; "
             f"d('3DAIGC/LAM-20K', local_dir=r'{downloads / 'LAM-20K'}'); "
             f"d('3DAIGC/LAM-assets', local_dir=r'{downloads / 'LAM-assets'}', allow_patterns=['*.tar'])"])
        if sha256(downloads / "LAM-20K" / "model.safetensors") != WEIGHTS_SHA256:
            sys.exit("LAM's weights do not match their checksum.")
        for name in ASSETS:
            with tarfile.open(downloads / "LAM-assets" / name) as archive:
                archive.extractall(install.source, filter="data")
        weights.mkdir(parents=True, exist_ok=True)
        for file in (downloads / "LAM-20K").iterdir():
            if file.is_file():
                shutil.copy2(file, weights / file.name)

    print("\n3D head installed: restart the studio, then Settings > Face > Clone > 3D head.")


if __name__ == "__main__":
    main()
