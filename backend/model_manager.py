"""Voice model library: files on disk, upload/delete, and the active voice.

HuBERT and the pitch extractor are shared by every voice and loaded once;
activating a voice only loads its synthesizer and retrieval index. Loading and
warm-up run in a worker thread (see ``backend.api.models``), and the new voice
is published only when it is fully ready, so streams never see a half-loaded
model.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import UploadFile

from backend.assets import AssetStore
from backend.config import Settings
from backend.inference.device import ComputeDevice, select_device
from backend.inference.hubert import HubertModel, load_hubert
from backend.inference.onnx_decoder import onnx_decoder
from backend.inference.pipeline import ConversionSettings, VoiceConverter
from backend.inference.pitch import DEFAULT_F0_METHOD, PitchExtractor, method_capabilities
from backend.inference.retrieval import FeatureIndex, IndexFormatError
from backend.inference.voice_model import VoiceModel

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = {".pth", ".pt"}
INDEX_EXTENSION = ".index"
MAX_MODEL_BYTES = 4 * 1024**3


class ModelNotFoundError(FileNotFoundError):
    pass


def sanitize_filename(name: str) -> str:
    """Keep word characters, dots and hyphens; reject empty results and path parts."""
    name = re.sub(r"_+", "_", re.sub(r"[^\w.\-]", "_", Path(name).name)).strip("_")
    if not name or name.startswith("."):
        raise ValueError("Invalid file name")
    return name


class ModelManager:
    def __init__(self, settings: Settings, assets: AssetStore, compute: ComputeDevice | None = None) -> None:
        self.settings = settings
        self.assets = assets
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._compute = compute
        self._lock = threading.Lock()
        self._hubert: HubertModel | None = None
        self._pitch: PitchExtractor | None = None
        self._converter: VoiceConverter | None = None
        self._details: dict[str, dict] = {}

    # --- Shared engine ------------------------------------------------------

    @property
    def compute(self) -> ComputeDevice:
        if self._compute is None:
            self._compute = select_device(self.settings.DEVICE)
        return self._compute

    def _engine(self) -> tuple[HubertModel, PitchExtractor]:
        if self._hubert is None:
            if not self.assets.ready("hubert"):
                raise RuntimeError("The HuBERT content encoder is still downloading; try again shortly.")
            self._hubert = load_hubert(self.assets.path("hubert"), self.compute.device, self.compute.half)
        if self._pitch is None:
            self._pitch = PitchExtractor(self.compute.device, self.compute.half, self.assets.path("rmvpe"))
        return self._hubert, self._pitch

    def pitch_methods(self) -> list[dict]:
        return method_capabilities(self.assets.path("rmvpe") if self.assets.ready("rmvpe") else Path("missing"))

    def default_f0_method(self) -> str:
        """RMVPE on an accelerator when its weights are ready; PM (cheap, asset-free) otherwise."""
        if self.assets.ready("rmvpe") and self.compute.is_accelerator:
            return "rmvpe"
        return DEFAULT_F0_METHOD

    def stream_defaults(self) -> tuple[int, float, float]:
        """``(chunk_size, lookback_seconds, lookahead_seconds)`` that fit this device in realtime."""
        s = self.settings
        if self.compute.is_accelerator:
            return s.CHUNK_SIZE, s.STREAM_LOOKBACK_SECONDS, s.STREAM_LOOKAHEAD_SECONDS
        return s.CPU_CHUNK_SIZE, s.CPU_STREAM_LOOKBACK_SECONDS, s.CPU_STREAM_LOOKAHEAD_SECONDS

    def default_conversion(self) -> ConversionSettings:
        s = self.settings
        return ConversionSettings(
            f0_method=self.default_f0_method(), index_rate=s.DEFAULT_INDEX_RATE,
            filter_radius=s.DEFAULT_FILTER_RADIUS, rms_mix_rate=s.DEFAULT_RMS_MIX_RATE,
            protect=s.DEFAULT_PROTECT, crepe_hop_length=s.DEFAULT_CREPE_HOP_LENGTH,
            rmvpe_threshold=s.RMVPE_THRESHOLD)

    # --- Library ------------------------------------------------------------

    def _model_files(self) -> list[Path]:
        return sorted(p for p in self.models_dir.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS)

    def find_index(self, model_name: str) -> Path | None:
        """``<stem>.index``, a case-insensitive match, or the only index in the folder."""
        stem = Path(model_name).stem.casefold()
        indexes = sorted(p for p in self.models_dir.iterdir()
                         if p.is_file() and p.suffix.lower() == INDEX_EXTENSION)
        for candidate in indexes:
            if candidate.stem.casefold() == stem:
                return candidate
        return indexes[0] if len(indexes) == 1 else None

    def list_models(self) -> list[dict]:
        active = self.active_model_name
        return [{
            "name": path.name,
            "type": "rvc",
            "size_bytes": path.stat().st_size,
            "active": path.name == active,
            "has_index": self.find_index(path.name) is not None,
            "details": self._details.get(path.name),
        } for path in self._model_files()]

    def _path_of(self, name: str) -> Path:
        path = self.models_dir / Path(name).name
        if path.suffix.lower() not in MODEL_EXTENSIONS or not path.is_file():
            raise ModelNotFoundError(f"Model not found: {name}")
        return path

    async def upload(self, filename: str, file: UploadFile) -> dict:
        safe_name = sanitize_filename(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in MODEL_EXTENSIONS | {INDEX_EXTENSION}:
            raise ValueError(f"Unsupported file type '{extension}'. Upload .pth models and their .index files.")
        destination = self.models_dir / safe_name
        partial = destination.with_suffix(destination.suffix + ".part")
        written = 0
        try:
            with partial.open("wb") as out:
                while block := await file.read(1024 * 1024):
                    written += len(block)
                    if written > MAX_MODEL_BYTES:
                        raise ValueError("File exceeds the 4 GB limit")
                    out.write(block)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        logger.info("Uploaded %s (%d bytes)", safe_name, written)
        kind = "index" if extension == INDEX_EXTENSION else "rvc"
        return {"name": safe_name, "type": kind, "size_bytes": written, "active": False}

    def delete(self, name: str) -> None:
        path = self._path_of(name)
        with self._lock:
            if self._converter is not None and self._converter.voice.name == path.name:
                self._release_locked()
            path.unlink()
            self._details.pop(path.name, None)
        logger.info("Deleted model %s", path.name)

    # --- Active voice -------------------------------------------------------

    def activate(self, name: str) -> dict:
        path = self._path_of(name)
        with self._lock:
            self._release_locked()
            hubert, pitch = self._engine()
            compute = self.compute
            voice = VoiceModel.load(path, compute.device, compute.half,
                                    allow_unsafe=self.settings.RVC_ALLOW_UNSAFE_CHECKPOINTS)
            index = None
            index_path = self.find_index(path.name)
            if index_path is not None:
                try:
                    index = FeatureIndex.load(index_path, compute.device, exact=self.settings.INDEX_EXACT_SEARCH)
                except (IndexFormatError, OSError) as exc:
                    logger.warning("Ignoring index %s: %s", index_path.name, exc)
            if compute.device.type == "cpu" and self.settings.CPU_ONNX_DECODER:
                stat = path.stat()
                cache = self.settings.DATA_DIR / "cache" / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.onnx"
                voice.synthesizer.decoder_override = onnx_decoder(voice.synthesizer, cache)
            converter = VoiceConverter(compute, hubert, pitch, voice, index)
            details = {
                **voice.describe(), **compute.describe(),
                "index": index_path.name if index is not None and index_path else None,
                "index_vectors": index.size if index is not None else 0,
                "warmup_ms": self._warm_up(converter),
            }
            self._converter = converter
            self._details[path.name] = details
        logger.info("Activated %s", path.name)
        return {"name": path.name, "type": "rvc", "status": "active", "details": details}

    def _warm_up(self, converter: VoiceConverter) -> float | None:
        """Render one realtime-sized window so the first live chunk pays no setup cost."""
        s = self.settings
        window = int((s.STREAM_LOOKBACK_SECONDS + s.STREAM_LOOKAHEAD_SECONDS + s.CHUNK_SIZE / s.SAMPLE_RATE)
                     * 16000) // 160 * 160
        silence = np.zeros(window, dtype=np.float32)
        settings = self.default_conversion()
        try:
            converter.convert(silence, settings, pad_seconds=s.STREAM_PAD_SECONDS, split_long_input=False)
            started = time.perf_counter()
            converter.convert(silence, settings, pad_seconds=s.STREAM_PAD_SECONDS, split_long_input=False)
            if converter.compute.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception:
            logger.exception("Warm-up failed")
            return None
        elapsed = (time.perf_counter() - started) * 1000.0
        budget = s.CHUNK_SIZE * 1000.0 / s.SAMPLE_RATE
        level = logging.INFO if elapsed < budget else logging.WARNING
        logger.log(level, "Realtime render: %.0f ms per %.0f ms chunk (f0=%s)", elapsed, budget, settings.f0_method)
        return round(elapsed, 1)

    def deactivate(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if self._converter is None:
            return
        logger.info("Unloading %s", self._converter.voice.name)
        self._converter = None
        if self.compute.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.deactivate()
        if self._pitch is not None:
            self._pitch.close()

    @property
    def converter(self) -> VoiceConverter | None:
        return self._converter

    @property
    def active_model_name(self) -> str | None:
        converter = self._converter
        return converter.voice.name if converter is not None else None

    def active_model(self) -> dict | None:
        converter = self._converter
        if converter is None:
            return None
        name = converter.voice.name
        return {"name": name, "type": "rvc", "active": True, "has_index": converter.index is not None,
                "details": self._details.get(name)}
