"""Voice model library: files on disk, upload/delete, and the active voice.

HuBERT and the pitch extractor are shared by every voice and loaded once;
activating a voice only loads its synthesizer and retrieval index. Loading and
warm-up run in a worker thread (see ``backend.api.models``), and the new voice
is published only when it is fully ready, so streams never see a half-loaded
model.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import replace
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
        # Last automatic pitch shift per voice, so a new stream starts in the right register.
        self._pitch_memory: dict[str, float] = {}
        self._live_f0_methods: set[str] = set()
        self._activation_lock = threading.Lock()
        self._loading: dict | None = None
        self._state_path = Path(settings.DATA_DIR) / "state.json"

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
            rmvpe_threshold=s.RMVPE_THRESHOLD, pitch_expressiveness=s.DEFAULT_EXPRESSIVENESS)

    # --- Last active voice (restored on the next start) ----------------------

    def _remember_active(self, name: str | None) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            partial = self._state_path.with_suffix(".part")
            partial.write_text(json.dumps({"active_voice": name}), encoding="utf-8")
            partial.replace(self._state_path)
        except OSError:
            logger.warning("Could not save the active voice", exc_info=True)

    def last_active_voice(self) -> str | None:
        try:
            name = json.loads(self._state_path.read_text(encoding="utf-8")).get("active_voice")
        except (OSError, ValueError, AttributeError):
            return None
        return name if isinstance(name, str) and (self.models_dir / Path(name).name).is_file() else None

    def restore_last_voice(self, wait_seconds: float = 900.0) -> None:
        """Re-activate the voice used last time once HuBERT is available (runs in a thread)."""
        name = self.last_active_voice()
        if name is None or self._converter is not None:
            return
        deadline = time.monotonic() + wait_seconds
        while not self.assets.ready("hubert"):
            if time.monotonic() > deadline:
                return
            time.sleep(1.0)
        if self._converter is not None:  # the user picked a voice meanwhile
            return
        try:
            self.activate(name)
        except Exception:
            logger.exception("Could not restore the last voice %s", name)

    def vad_path(self) -> Path | None:
        """Silero VAD weights once downloaded (the stream falls back to a level detector before)."""
        return self.assets.path("vad") if self.assets.ready("vad") else None

    def note_f0_method(self, method: str) -> None:
        """A live stream uses ``method``: keep it warm on the next voice activation."""
        self._live_f0_methods.add(method)

    def remembered_pitch_shift(self, voice_name: str) -> float | None:
        return self._pitch_memory.get(voice_name)

    def remember_pitch_shift(self, voice_name: str, semitones: float) -> None:
        self._pitch_memory[voice_name] = semitones

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

    def export_archive(self, name: str) -> tuple[Path, str]:
        """``(temporary zip, download name)`` with the voice's ``.pth`` and its ``.index``.

        Indexes can be hundreds of MB, so the archive is written to a temporary
        file (the caller deletes it) and stored without recompression.
        """
        path = self._path_of(name)
        handle, temporary = tempfile.mkstemp(suffix=".zip", prefix="ovc-export-")
        os.close(handle)
        target = Path(temporary)
        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.write(path, path.name)
                index = self.find_index(path.name)
                if index is not None:
                    archive.write(index, f"{path.stem}.index")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target, f"{path.stem}.zip"

    def delete(self, name: str) -> None:
        path = self._path_of(name)
        with self._lock:
            if self._converter is not None and self._converter.voice.name == path.name:
                self._release_locked()
                self._remember_active(None)
            path.unlink()
            self._details.pop(path.name, None)
        logger.info("Deleted model %s", path.name)

    # --- Active voice -------------------------------------------------------

    @property
    def loading(self) -> dict | None:
        """``{"name", "stage"}`` while a voice is being prepared, else None."""
        return dict(self._loading) if self._loading else None

    def activate(self, name: str) -> dict:
        """Load, index and warm up ``name``, then make it the active voice.

        The previous voice keeps converting until the new one is completely
        ready (weights, index, CPU decoder, every pitch tracker in use, and
        realtime-sized renders), so a stream never falls back to the dry voice
        and the first live chunk after activation is already fast.
        """
        path = self._path_of(name)
        with self._activation_lock:
            self._loading = {"name": path.name, "stage": "loading weights"}
            try:
                converter, details = self._prepare(path)
            finally:
                self._loading = None
            with self._lock:
                self._converter = converter
                self._details[path.name] = details
                self._remember_active(path.name)
        logger.info("Activated %s", path.name)
        return {"name": path.name, "type": "rvc", "status": "active", "details": details}

    def _prepare(self, path: Path) -> tuple[VoiceConverter, dict]:
        with self._lock:
            hubert, pitch = self._engine()
        compute = self.compute
        voice = VoiceModel.load(path, compute.device, compute.half,
                                allow_unsafe=self.settings.RVC_ALLOW_UNSAFE_CHECKPOINTS)
        index = None
        index_path = self.find_index(path.name)
        if index_path is not None:
            self._loading = {"name": path.name, "stage": "loading the index"}
            try:
                index = FeatureIndex.load(index_path, compute.device, exact=self.settings.INDEX_EXACT_SEARCH)
            except (IndexFormatError, OSError) as exc:
                logger.warning("Ignoring index %s: %s", index_path.name, exc)
        if compute.device.type == "cpu" and self.settings.CPU_ONNX_DECODER:
            self._loading = {"name": path.name, "stage": "optimising the CPU decoder"}
            stat = path.stat()
            cache = self.settings.DATA_DIR / "cache" / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.onnx"
            voice.synthesizer.decoder_override = onnx_decoder(voice.synthesizer, cache)
        converter = VoiceConverter(compute, hubert, pitch, voice, index)
        self._loading = {"name": path.name, "stage": "warming up"}
        warm = self._warm_up(converter)
        details = {
            **voice.describe(), **compute.describe(),
            "index": index_path.name if index is not None and index_path else None,
            "index_vectors": index.size if index is not None else 0,
            "warmup_ms": warm.get(self.default_f0_method()),
            "realtime_ms": warm,
        }
        return converter, details

    def _warm_methods(self) -> list[str]:
        methods = [self.default_f0_method(), *sorted(self._live_f0_methods)]
        available = {item["id"] for item in self.pitch_methods() if item.get("available") and item.get("realtime")}
        return [m for m in dict.fromkeys(methods) if m in available or m == self.default_f0_method()]

    def _warm_up(self, converter: VoiceConverter) -> dict[str, float]:
        """Render realtime-sized windows with every pitch tracker in use; returns ms per chunk by method."""
        s = self.settings
        chunk_size, lookback, lookahead = self.stream_defaults()  # this device's live geometry
        window = math.ceil((lookback + lookahead + s.STREAM_CROSSFADE_SECONDS + s.STREAM_SEARCH_SECONDS
                            + chunk_size / s.SAMPLE_RATE) * 16000) // 160 * 160
        # A voiced, speech-like signal so the pitch, protect and index paths all run.
        t = np.arange(window) / 16000
        phase = 2 * np.pi * np.cumsum(140 + 30 * np.sin(2 * np.pi * 1.5 * t)) / 16000
        probe = (0.1 * np.sum([0.6 ** k * np.sin(k * phase) for k in range(1, 12)], axis=0)).astype(np.float32)
        # Like the live stream, vocode only the emitted end of the window.
        tail = (chunk_size / s.SAMPLE_RATE + lookahead + s.STREAM_CROSSFADE_SECONDS + s.STREAM_SEARCH_SECONDS
                + s.STREAM_DECODE_MARGIN_SECONDS)
        budget = chunk_size * 1000.0 / s.SAMPLE_RATE
        timings: dict[str, float] = {}
        live = self._converter
        for method in self._warm_methods():
            settings = replace(self.default_conversion(), f0_method=method)
            try:
                if live is not None:
                    # Another voice is converting, so HuBERT, the pitch trackers and the GPU
                    # kernels are already warm: a new voice's first render is as fast as any
                    # (measured: 80-97 ms, same as steady state). Renders here would only
                    # compete with the live stream for the GPU and cause audible gaps.
                    with live._lock:
                        converter.pitch.prepare(method)
                    continue
                converter.pitch.prepare(method)
                elapsed = []
                for _ in range(3):  # first: lazy kernels and allocations; then steady state
                    started = time.perf_counter()
                    converter.convert(probe, settings, pad_seconds=s.STREAM_PAD_SECONDS, split_long_input=False,
                                      rms_frame_seconds=s.STREAM_RMS_FRAME_SECONDS, tail_seconds=tail)
                    if converter.compute.device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed.append((time.perf_counter() - started) * 1000.0)
            except Exception:
                logger.exception("Warm-up with %s failed", method)
                continue
            timings[method] = round(min(elapsed[1:]), 1)
            level = logging.INFO if timings[method] < budget else logging.WARNING
            logger.log(level, "Realtime render: %.0f ms per %.0f ms chunk (f0=%s)", timings[method], budget, method)
        return timings

    def pitch_ready(self, method: str) -> bool:
        pitch = self._pitch
        return pitch is not None and pitch.ready(method)

    def prepare_pitch(self, method: str) -> None:
        """Load a pitch tracker ahead of use (a live stream switches to it once this returns)."""
        with self._lock:
            _, pitch = self._engine()
        pitch.prepare(method)
        self.note_f0_method(method)

    def deactivate(self) -> None:
        """Unload the voice at the user's request (it is not restored on the next start)."""
        with self._lock:
            self._release_locked()
            self._remember_active(None)

    def _release_locked(self) -> None:
        if self._converter is None:
            return
        logger.info("Unloading %s", self._converter.voice.name)
        self._converter = None
        if self.compute.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:
        with self._lock:
            self._release_locked()
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
                "details": self._details.get(name), "loading": self.loading}
