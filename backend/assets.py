"""Shared model files: the voice's (HuBERT, RMVPE, Silero VAD, DeepFilterNet3) and the face clone's.

Downloaded once from their official sources and checked by SHA-256. An asset
can be built after download: the DeepFilterNet3 checkpoint is exported here
to the streaming ONNX graph (see backend.audio.deepfilter), HyperSwap is
stored in fp32 (see backend.video.convert). The voice's files download at the
first start; the face clone's (about 1.8 GB) only when the face clone is used.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.audio.deepfilter import EXPORT_VERSION as DEEPFILTER_EXPORT_VERSION
from backend.inference.checkpoints import file_sha256
from backend.inference.hubert import HUBERT_SHA256

logger = logging.getLogger(__name__)


FACEFUSION = "https://huggingface.co/facefusion/models"

PINNED_REVISION = ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
                   "e6d0c1a17da07c33557852f9dfa2bd44cc75737d")


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int
    required: bool
    product: str | None = None      # file built from the download (what is used), if any
    group: str = "voice"            # voice: downloaded at start; video: when the face clone is first used
    folder: str = ""                # subfolder of the assets directory
    label: str = ""

    @property
    def relative(self) -> str:
        return f"{self.folder}/{self.filename}" if self.folder else self.filename

    @property
    def relative_product(self) -> str | None:
        if self.product is None:
            return None
        return f"{self.folder}/{self.product}" if self.folder else self.product


def download_verified(asset: Asset, target: Path, on_progress: Callable[[int], None] | None = None) -> None:
    """Download ``asset`` to ``target`` atomically, verifying its SHA-256; raises on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    received = 0
    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "VoxClone"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
                received += len(block)
                if on_progress is not None:
                    on_progress(received)
        # An empty checksum means the caller verifies the file another way (a signature).
        if asset.sha256 and file_sha256(partial) != asset.sha256:
            raise RuntimeError(f"{asset.filename}: checksum mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


ASSETS = (
    Asset("hubert", "hubert_base.pt", f"{PINNED_REVISION}/hubert_base.pt", HUBERT_SHA256, 189507909, True),
    Asset("rmvpe", "rmvpe.pt", f"{PINNED_REVISION}/rmvpe.pt",
          "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193", 181184272, False),
    # Silero VAD (MIT): tells speech from noise and breaths in the live stream.
    Asset("vad", "silero_vad.onnx",
          "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx",
          "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3", 2327524, False),
    # DeepFilterNet3 (MIT/Apache-2.0), the official checkpoint from its authors' repository (the
    # latest DeepFilterNet model). It is exported here to a streaming ONNX graph that removes room
    # noise, keyboard clicks and breaths from what the voice model hears.
    Asset("denoiser", "DeepFilterNet3.zip",
          "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip",
          "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2", 7986207, False,
          product=f"deepfilternet3_stream_v{DEEPFILTER_EXPORT_VERSION}.onnx"),
    # The face clone (FaceFusion's model releases on Hugging Face). Non-commercial licences:
    # InsightFace (inswapper, ArcFace, SCRFD) and FaceFusion's HyperSwap are for non-commercial use.
    *(Asset(name, f"{name}.onnx", f"{FACEFUSION}-{tag}/resolve/main/{name}.onnx", sha, size, False, product,
            group="video", folder="face", label=label)
      for name, tag, sha, size, product, label in (
          ("scrfd_2.5g", "3.0.0", "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada", 3295067, None,
           "Face detector (SCRFD)"),
          ("fan_68_5", "3.0.0", "ea05f7d9d014ae1d9cdc6e9c72643d8f00d198b5db3fdd384148ddff499a613e", 944321, None,
           "Face landmarks"),
          ("arcface_w600k_r50", "3.0.0", "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
           174388474, None, "Face identity (ArcFace)"),
          ("hyperswap_1b_256", "3.3.0", "5124031789c42f71b9558fb71954ef7aedb6da7ed9fac79293e23c61a792a73e",
           402742682, "hyperswap_1b_256_fp32.onnx", "Face swapper HyperSwap"),
          ("inswapper_128", "3.0.0", "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7",
           555303150, None, "Face swapper inswapper"),
          ("ghost_1_256", "3.0.0", "4eac28021ae80128e9262080cf0da850e3190e41862650c9e805f9a516b3e924",
           514902950, None, "Face swapper GHOST"),
          ("crossface_ghost", "3.4.0", "9ec5862d9ff1f723a7380ea89baf87cddec9c56670d4db766702657939284957",
           22083800, None, "GHOST identity converter"),
          ("xseg_1", "3.1.0", "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3", 70324286, None,
           "Occlusion mask (XSeg)"),
          ("gpen_bfr_256", "3.0.0", "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25",
           75792988, None, "Face enhancer (GPEN)"),
      )),
)


def build_product(asset: Asset, source: Path, target: Path) -> None:
    """Make ``asset.product`` from the verified download."""
    if asset.name == "denoiser":
        from backend.audio.deepfilter import export_streaming_onnx

        export_streaming_onnx(source, target)
    elif asset.name.startswith("hyperswap"):
        from backend.video.convert import to_float32

        to_float32(source, target)
    else:  # pragma: no cover - every product needs a builder
        raise ValueError(f"No builder for {asset.name}")


@dataclass(slots=True)
class _Progress:
    state: str = "missing"   # missing | downloading | preparing | ready | error
    received: int = 0
    error: str | None = None


class AssetStore:
    """Locates, downloads and builds the shared model assets."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._progress = {asset.name: _Progress() for asset in ASSETS}
        self._threads: dict[str, threading.Thread] = {}

    def _asset(self, name: str) -> Asset:
        return next(a for a in ASSETS if a.name == name)

    def path(self, name: str) -> Path:
        """The file to use: the built product when the asset has one."""
        asset = self._asset(name)
        return self.directory / (asset.relative_product or asset.relative)

    def folder(self, group: str) -> Path:
        """Where a group's files live (the face clone's are in their own folder)."""
        folders = {asset.folder for asset in ASSETS if asset.group == group}
        return self.directory / (folders.pop() if len(folders) == 1 else "")

    def ready(self, name: str) -> bool:
        return self._progress[name].state == "ready"

    def group_ready(self, group: str) -> bool:
        return all(self.ready(asset.name) for asset in ASSETS if asset.group == group)

    def status(self, group: str | None = "voice") -> list[dict]:
        rows = []
        for asset in ASSETS:
            if group is not None and asset.group != group:
                continue
            progress = self._progress[asset.name]
            rows.append({"id": asset.name, "file": asset.filename, "required": asset.required,
                         "state": progress.state, "bytes": asset.size, "received": progress.received,
                         "error": progress.error, "label": asset.label or asset.name})
        return rows

    def prepare(self) -> None:
        """Mark the files already on disk ready (fast, no network)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        products = {asset.product for asset in ASSETS if asset.product}
        for stale in self.directory.glob("deepfilternet3_stream_v*.onnx"):   # older exports
            if stale.name not in products:
                stale.unlink(missing_ok=True)
        for asset in ASSETS:
            target = self.directory / asset.relative
            downloaded = target.is_file() and target.stat().st_size == asset.size
            built = asset.relative_product is None or (self.directory / asset.relative_product).is_file()
            if downloaded and built:
                self._progress[asset.name].state = "ready"

    def download_missing_in_background(self, group: str = "voice") -> None:
        """Fetch a group's missing files (the voice's at start, the face clone's on first use)."""
        thread = self._threads.get(group)
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=self._download_missing, args=(group,),
                                      name=f"asset-download-{group}", daemon=True)
            self._threads[group] = thread
            thread.start()

    def _download_missing(self, group: str = "voice") -> None:
        for asset in ASSETS:
            if asset.group != group or self._progress[asset.name].state == "ready":
                continue
            target = self.directory / asset.relative
            if not (target.is_file() and target.stat().st_size == asset.size):
                self._download(asset)
            elif asset.product is not None:
                self._progress[asset.name].state = "downloaded"
            if asset.product is not None and self._progress[asset.name].state != "error":
                self._build(asset)

    def _build(self, asset: Asset) -> None:
        progress = self._progress[asset.name]
        assert asset.relative_product is not None
        progress.state, progress.error = "preparing", None
        try:
            source = self.directory / asset.relative
            if file_sha256(source) != asset.sha256:
                raise RuntimeError(f"{asset.filename}: checksum mismatch")
            build_product(asset, source, self.directory / asset.relative_product)
            progress.state = "ready"
        except Exception as exc:
            progress.state, progress.error = "error", f"{type(exc).__name__}: {exc}"
            logger.error("Could not prepare %s: %s", asset.product, exc)

    def _download(self, asset: Asset) -> None:
        progress = self._progress[asset.name]
        progress.state, progress.received, progress.error = "downloading", 0, None
        logger.info("Downloading %s (%.0f MB)...", asset.filename, asset.size / 1e6)

        def report(received: int) -> None:
            progress.received = received

        try:
            download_verified(asset, self.directory / asset.relative, report)
            progress.state = "ready" if asset.product is None else "downloaded"
            logger.info("Downloaded %s", asset.filename)
        except Exception as exc:
            progress.state, progress.error = "error", f"{type(exc).__name__}: {exc}"
            logger.error("Could not download %s: %s", asset.filename, exc)
