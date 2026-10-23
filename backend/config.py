"""Application settings. Every value can be overridden with an ``OVC_<NAME>`` environment variable."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# The Vite dev server and the backend itself on loopback. Same-origin requests
# are always allowed (see backend.security), so serving the built frontend
# from this backend works on any host without extra configuration.
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OVC_", env_file=ROOT / ".env", extra="ignore")

    # --- Server -------------------------------------------------------------
    # Loopback by default. OVC_HOST=0.0.0.0 exposes the studio on the LAN.
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    CORS_ORIGINS: list[str] = DEFAULT_ALLOWED_ORIGINS
    # Disables origin checks for HTTP and the audio WebSocket. Trusted networks only.
    ALLOW_ANY_ORIGIN: bool = False

    # --- Storage ------------------------------------------------------------
    MODELS_DIR: Path = ROOT / "models"
    ASSETS_DIR: Path = ROOT / "models" / "assets"
    DATA_DIR: Path = ROOT / "data"          # training jobs and other studio data
    PRESETS_PATH: Path = ROOT / "data" / "presets.json"
    FRONTEND_DIST: Path = ROOT / "frontend" / "dist"

    # --- Inference ----------------------------------------------------------
    # auto | cuda | mps | xpu | cpu
    DEVICE: str = "auto"
    # RVC .pth files are pickles; unpickling one runs code embedded in it.
    # Keep this off unless you trust every checkpoint you load.
    RVC_ALLOW_UNSAFE_CHECKPOINTS: bool = False
    # Exact nearest-neighbour index search (better timbre than faiss IVF with
    # nprobe=1). Set false to reproduce classic RVC retrieval exactly.
    INDEX_EXACT_SEARCH: bool = True
    DEFAULT_INDEX_RATE: float = 0.75
    DEFAULT_FILTER_RADIUS: int = 3
    DEFAULT_RMS_MIX_RATE: float = 0.25
    DEFAULT_PROTECT: float = 0.33
    DEFAULT_CREPE_HOP_LENGTH: int = 160
    RMVPE_THRESHOLD: float = 0.03
    # Run the vocoder with ONNX Runtime when inferring on CPU (~1.5x faster).
    CPU_ONNX_DECODER: bool = True
    # Load the voice that was active when the studio last closed.
    RESTORE_LAST_VOICE: bool = True

    # --- Realtime streaming -------------------------------------------------
    # Rate suggested to the browser; the AudioContext's actual rate is used.
    SAMPLE_RATE: int = 48000
    # Samples per live chunk at SAMPLE_RATE (8192 = 171 ms).
    CHUNK_SIZE: int = 8192
    # Past context re-rendered with every chunk: steadier timbre and pitch.
    STREAM_LOOKBACK_SECONDS: float = 1.0
    # Future context: each emitted sample is rendered with this much real
    # audio after it, so window edges never reach the output. Adds latency.
    STREAM_LOOKAHEAD_SECONDS: float = 0.15
    # SOLA crossfade between consecutive windows and its alignment search.
    STREAM_CROSSFADE_SECONDS: float = 0.05
    STREAM_SEARCH_SECONDS: float = 0.012
    # Reflect padding around each streaming window (file renders use RVC's 1-3 s).
    STREAM_PAD_SECONDS: float = 0.1
    # Loudness envelope window for rms_mix_rate in streaming (shift-invariant).
    STREAM_RMS_FRAME_SECONDS: float = 0.2
    # The vocoder only renders the emitted end of each window plus this margin
    # (its receptive field is a few frames: spectra match within run-to-run
    # noise for any margin); the content encoder still sees the full context.
    STREAM_DECODE_MARGIN_SECONDS: float = 0.1
    # CPU-only machines: the largest window that still renders in realtime
    # (measured on a 12th-gen Core i7 with the ONNX decoder: ~290 ms per 341 ms chunk).
    CPU_CHUNK_SIZE: int = 16384
    CPU_STREAM_LOOKBACK_SECONDS: float = 0.5
    CPU_STREAM_LOOKAHEAD_SECONDS: float = 0.1
    SILENCE_SAVER: bool = True
    # Sleep inference after this much silence. The level is the loudest 10 ms
    # of each chunk, so a single quiet word still wakes it.
    SILENCE_THRESHOLD_DB: float = -60.0
    SILENCE_HOLD_SECONDS: float = 0.3

    # --- File conversion ----------------------------------------------------
    MAX_CONVERT_SECONDS: int = 600
    MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024


settings = Settings()
