"""Shared fixtures: an isolated app instance and a fake voice converter.

The fake converter stands in for a loaded RVC voice so the streaming,
conversion and API code can be exercised without model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.inference.device import ComputeDevice
from backend.model_manager import ModelManager

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class FakeVoice:
    name: str = "voice.pth"
    f0_stats = None  # set to an F0Stats to emulate a voice trained in the studio


class FakeConverter:
    """Renders a 16 kHz window as its negation at 40 kHz (an obviously altered signal)."""

    output_rate = 40000

    def __init__(self) -> None:
        self.voice = FakeVoice()
        self.index = None
        self.calls = 0
        self.last_settings = None
        self.f0_hz = 0.0  # pitch the fake tracker "hears" in every frame
        self.timings: dict[str, float] = {}

    def estimate_f0(self, audio: np.ndarray, settings) -> np.ndarray:
        return np.full(len(audio) // 160, self.f0_hz, dtype=np.float32)

    def convert(self, audio: np.ndarray, settings, *, on_f0=None, **_options) -> np.ndarray:
        from backend.audio.resampling import resample

        self.calls += 1
        self.last_settings = settings
        if on_f0 is not None:
            on_f0(self.estimate_f0(audio, settings))
        return -resample(audio, 16000, self.output_rate)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(MODELS_DIR=tmp_path / "models", ASSETS_DIR=tmp_path / "assets", DATA_DIR=tmp_path / "data",
                    PRESETS_PATH=tmp_path / "presets.json", FRONTEND_DIST=tmp_path / "no-frontend",
                    DEVICE="cpu")


@pytest.fixture
def cpu() -> ComputeDevice:
    import torch

    return ComputeDevice(torch.device("cpu"), False, "CPU", None, 1, 6, 38, 41)


@pytest.fixture
def manager(settings: Settings, cpu: ComputeDevice) -> ModelManager:
    from backend.assets import AssetStore

    return ModelManager(settings, AssetStore(settings.ASSETS_DIR), compute=cpu)


@pytest.fixture
def client(settings: Settings, manager: ModelManager, monkeypatch: pytest.MonkeyPatch):
    # Never download model assets from tests.
    monkeypatch.setattr("backend.assets.AssetStore.download_missing_in_background", lambda self: None)
    app = create_app(settings, manager=manager)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_voice(manager: ModelManager) -> FakeConverter:
    converter = FakeConverter()
    manager._converter = converter  # type: ignore[assignment]
    return converter
