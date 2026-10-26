"""Shared model files: HuBERT, RMVPE, Silero VAD and DeepFilterNet3.

Downloaded once from their official sources and checked by SHA-256. An asset
can be built after download: the DeepFilterNet3 checkpoint is exported here
to the streaming ONNX graph (see backend.audio.deepfilter).
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
)


def build_product(asset: Asset, source: Path, target: Path) -> None:
    """Make ``asset.product`` from the verified download."""
    if asset.name == "denoiser":
        from backend.audio.deepfilter import export_streaming_onnx

        export_streaming_onnx(source, target)
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
        self._thread: threading.Thread | None = None

    def path(self, name: str) -> Path:
        """The file to use: the built product when the asset has one."""
        asset = next(a for a in ASSETS if a.name == name)
        return self.directory / (asset.product or asset.filename)

    def ready(self, name: str) -> bool:
        return self._progress[name].state == "ready"

    def status(self) -> list[dict]:
        rows = []
        for asset in ASSETS:
            progress = self._progress[asset.name]
            rows.append({"id": asset.name, "file": asset.filename, "required": asset.required,
                         "state": progress.state, "bytes": asset.size, "received": progress.received,
                         "error": progress.error})
        return rows

    def prepare(self) -> None:
        """Mark the files already on disk ready (fast, no network)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        products = {asset.product for asset in ASSETS if asset.product}
        for stale in self.directory.glob("deepfilternet3_stream_v*.onnx"):   # older exports
            if stale.name not in products:
                stale.unlink(missing_ok=True)
        for asset in ASSETS:
            target = self.directory / asset.filename
            downloaded = target.is_file() and target.stat().st_size == asset.size
            built = asset.product is None or (self.directory / asset.product).is_file()
            if downloaded and built:
                self._progress[asset.name].state = "ready"

    def download_missing_in_background(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._download_missing, name="asset-download", daemon=True)
            self._thread.start()

    def _download_missing(self) -> None:
        for asset in ASSETS:
            if self._progress[asset.name].state == "ready":
                continue
            target = self.directory / asset.filename
            if not (target.is_file() and target.stat().st_size == asset.size):
                self._download(asset)
            if asset.product is not None and self._progress[asset.name].state != "error":
                self._build(asset)

    def _build(self, asset: Asset) -> None:
        progress = self._progress[asset.name]
        assert asset.product is not None
        progress.state, progress.error = "preparing", None
        try:
            source = self.directory / asset.filename
            if file_sha256(source) != asset.sha256:
                raise RuntimeError(f"{asset.filename}: checksum mismatch")
            build_product(asset, source, self.directory / asset.product)
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
            download_verified(asset, self.directory / asset.filename, report)
            progress.state = "ready" if asset.product is None else "downloaded"
            logger.info("Downloaded %s", asset.filename)
        except Exception as exc:
            progress.state, progress.error = "error", f"{type(exc).__name__}: {exc}"
            logger.error("Could not download %s: %s", asset.filename, exc)
