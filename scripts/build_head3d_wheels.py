"""Maintainers: compile the 3D head's CUDA libraries once, for every recent NVIDIA GPU, so nobody
installing the studio needs a compiler. Windows, with Visual Studio 2019 Build Tools and CUDA 11.8,
and LAM's environment installed (scripts/setup_head3d.py).

    uv run python scripts/build_head3d_wheels.py <output folder>

Writes pytorch3d and diff-gaussian-rasterization wheels, ``nvdiffrast_plugin.pyd`` (nvdiffrast's
CUDA plugin, which nvdiffrast would otherwise compile on each computer) and LAM's face box detector's
``cpu_nms`` module, plus ``SHA256SUMS``. They
go up as a GitHub release; scripts/setup_head3d.py pins their names and checksums.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.video.avatar import LamInstall, _cuda_home, _vcvars  # noqa: E402

# Pascal (GTX 10) to Hopper; newer GPUs (RTX 50) run the PTX, compiled by the driver on first use.
ARCHS = "6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
# nvdiffrast's rasterizer crashes CUDA 11.8's ptxas for sm_90 (stack overflow): Hopper and newer run
# its 8.9 PTX instead.
PLUGIN_ARCHS = "6.1;7.0;7.5;8.0;8.6;8.9+PTX"
SOURCES = {   # folder: (repository, commit)
    "pytorch3d": ("https://github.com/facebookresearch/pytorch3d.git", "89653419d0973396f3eff1a381ba09a07fffc2ed"),
    "diffgs": ("https://github.com/ashawkey/diff-gaussian-rasterization.git", "8829d14f814fccdaf840b7b0f3021a616583c0a1"),
    "nvdiffrast": ("https://github.com/ShenhanQian/nvdiffrast.git", "22718580f24a313c429ba2c304794c264351f108"),
}
NVDIFFRAST_SETUP = textwrap.dedent("""
    # nvdiffrast's CUDA plugin as a normal extension (nvdiffrast/torch/ops.py builds the same at run time)
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    import shutil

    root = "nvdiffrast/"
    # RasterImpl and texture come as both .cu and .cpp: one object file each would clash, so the
    # CUDA halves compile from copies with their own names (same folder, same includes).
    for name in ("common/cudaraster/impl/RasterImpl", "common/texture"):
        shutil.copyfile(root + name + ".cu", root + name + "_cuda.cu")
    sources = [root + "common/cudaraster/impl/Buffer.cpp", root + "common/cudaraster/impl/CudaRaster.cpp",
               root + "common/cudaraster/impl/RasterImpl_cuda.cu", root + "common/cudaraster/impl/RasterImpl.cpp",
               root + "common/common.cpp", root + "common/rasterize.cu", root + "common/interpolate.cu",
               root + "common/texture_cuda.cu", root + "common/texture.cpp", root + "common/antialias.cu",
               root + "torch/torch_bindings.cpp", root + "torch/torch_rasterize.cpp",
               root + "torch/torch_interpolate.cpp", root + "torch/torch_texture.cpp",
               root + "torch/torch_antialias.cpp"]
    setup(name="nvdiffrast_plugin", ext_modules=[CUDAExtension(
        "nvdiffrast_plugin", sources,
        extra_compile_args={"cxx": ["-DNVDR_TORCH", "/wd4067", "/wd4624"], "nvcc": ["-DNVDR_TORCH", "-lineinfo"]})],
        cmdclass={"build_ext": BuildExtension})
""")


def run(command: list[str], cwd: Path, env: dict) -> None:
    print("  $", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    install = LamInstall.default()
    if not install.python.is_file() or _vcvars() is None or _cuda_home() is None:
        sys.exit("Needs LAM's environment, Visual Studio 2019 Build Tools and CUDA 11.8.")
    work = install.root / "release-build"
    work.mkdir(exist_ok=True)
    env = install.compiler_environment()
    env.update(TORCH_CUDA_ARCH_LIST=ARCHS, FORCE_CUDA="1", MAX_JOBS="10")
    python = str(install.python)
    for folder, (repository, commit) in SOURCES.items():
        source = work / folder
        if not (source / ".git").is_dir():
            run(["git", "clone", "--recursive", repository, str(source)], work, env)
        run(["git", "-C", str(source), "checkout", "-q", commit], work, env)
        run(["git", "-C", str(source), "submodule", "update", "--init", "--recursive", "-q"], work, env)
    setup = work / "pytorch3d" / "setup.py"   # MSVC: its C++17 flag; Windows headers without min/max
    text = setup.read_text(encoding="utf-8")
    text = text.replace('{"cxx": ["-std=c++17"]}', '{"cxx": ["/std:c++17" if sys.platform == "win32" else "-std=c++17"]}')
    if '"-DWIN32_LEAN_AND_MEAN",' not in text:
        text = text.replace('"-DCUDA_HAS_FP16=1",', '"-DCUDA_HAS_FP16=1",\n            "-DWIN32_LEAN_AND_MEAN",')
    setup.write_text(text, encoding="utf-8")

    for folder in ("pytorch3d", "diffgs", "nvdiffrast"):   # nvdiffrast's own is Python only
        print(f"\n== {folder} ({ARCHS})", flush=True)
        run([python, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", str(out), "."],
            work / folder, env)
    print("\n== nvdiffrast plugin", flush=True)
    (work / "nvdiffrast" / "setup_plugin.py").write_text(NVDIFFRAST_SETUP, encoding="utf-8")
    run([python, "setup_plugin.py", "build_ext", "--inplace"], work / "nvdiffrast",
        {**env, "TORCH_CUDA_ARCH_LIST": PLUGIN_ARCHS})
    built = next((work / "nvdiffrast").glob("nvdiffrast_plugin*.pyd"))
    shutil.copy2(built, out / "nvdiffrast_plugin.pyd")

    print("\n== LAM's face box detector (plain C, any GPU)", flush=True)
    boxes = install.source / "external/landmark_detection/FaceBoxesV2/utils"
    run([python, "build.py", "build_ext", "--inplace"], boxes, env)
    shutil.copy2(next((boxes / "nms").glob("cpu_nms*.pyd")), out / "cpu_nms.cp310-win_amd64.pyd")

    sums = []
    for file in sorted(out.iterdir()):
        if file.suffix in (".whl", ".pyd"):
            sums.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
