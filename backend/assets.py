"""Model assets: the HuBERT content encoder and RMVPE weights every voice needs.

They are fetched once from pinned Hugging Face revisions and verified by
SHA-256 before use. Copies left by earlier OpenVoiceChanger versions are
reused instead of being downloaded again.
"""

from __future__ import annotations

import logging
import shutil
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.inference.checkpoints import file_sha256
from backend.inference.hubert import HUBERT_SHA256

logger = logging.getLogger(__name__)


def legacy_asset_directories(root: Path) -> list[Path]:
    """Where OpenVoiceChanger 2.x left HuBERT/RMVPE (old asset dir, isolated runtime setup)."""
    setup = root / "data" / "rvc-setup"
    return [root / "models" / "assets", setup / "downloads", *sorted(setup.glob("jobs/*/assets"))]


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


def download_verified(asset: Asset, target: Path, on_progress: Callable[[int], None] | None = None) -> None:
    """Download ``asset`` to ``target`` atomically, verifying its SHA-256; raises on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    received = 0
    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "OpenVoiceChanger"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
                received += len(block)
                if on_progress is not None:
                    on_progress(received)
        if file_sha256(partial) != asset.sha256:
            raise RuntimeError(f"{asset.filename}: checksum mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


ASSETS = (
    Asset("hubert", "hubert_base.pt", f"{PINNED_REVISION}/hubert_base.pt", HUBERT_SHA256, 189507909, True),
    Asset("rmvpe", "rmvpe.pt", f"{PINNED_REVISION}/rmvpe.pt",
          "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193", 181184272, False),
)


@dataclass(slots=True)
class _Progress:
    state: str = "missing"   # missing | downloading | ready | error
    received: int = 0
    error: str | None = None


class AssetStore:
    """Locates, migrates and downloads the shared model assets."""

    def __init__(self, directory: Path, legacy_roots: list[Path] | None = None) -> None:
        self.directory = directory
        self.legacy_roots = legacy_roots or []
        self._progress = {asset.name: _Progress() for asset in ASSETS}
        self._thread: threading.Thread | None = None

    def path(self, name: str) -> Path:
        asset = next(a for a in ASSETS if a.name == name)
        return self.directory / asset.filename

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
        """Mark verified files ready and adopt copies from older installs (fast, no network)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for asset in ASSETS:
            target = self.directory / asset.filename
            if not target.is_file():
                self._adopt_legacy_copy(asset, target)
            if target.is_file() and target.stat().st_size == asset.size:
                self._progress[asset.name].state = "ready"

    def _adopt_legacy_copy(self, asset: Asset, target: Path) -> None:
        for root in self.legacy_roots:
            for candidate in (root / asset.filename, root / "rmvpe" / asset.filename):
                if (candidate.is_file() and candidate.stat().st_size == asset.size
                        and file_sha256(candidate) == asset.sha256):
                    shutil.copy2(candidate, target)
                    logger.info("Reused %s from %s", asset.filename, candidate.parent)
                    return

    def download_missing_in_background(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._download_missing, name="asset-download", daemon=True)
            self._thread.start()

    def _download_missing(self) -> None:
        for asset in ASSETS:
            if self._progress[asset.name].state != "ready":
                self._download(asset)

    def _download(self, asset: Asset) -> None:
        progress = self._progress[asset.name]
        progress.state, progress.received, progress.error = "downloading", 0, None
        logger.info("Downloading %s (%.0f MB)...", asset.filename, asset.size / 1e6)

        def report(received: int) -> None:
            progress.received = received

        try:
            download_verified(asset, self.directory / asset.filename, report)
            progress.state = "ready"
            logger.info("Downloaded %s", asset.filename)
        except Exception as exc:
            progress.state, progress.error = "error", f"{type(exc).__name__}: {exc}"
            logger.error("Could not download %s: %s", asset.filename, exc)
