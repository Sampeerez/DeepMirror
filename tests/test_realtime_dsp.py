"""Realtime DSP and default-selection behaviour that keeps the stream audible."""

import numpy as np
import pytest

from backend.services import f0_registry
from backend.services.dsp_effects import EffectsChain, GateEffect, LimiterEffect
from backend.services.f0_adapter import _suppress_octave_spikes
from backend.services.hardware_detector import HardwareProfile, recommend_settings

RATE = 48000


class TestLimiter:
    def test_output_never_exceeds_the_ceiling(self):
        limiter = LimiterEffect(RATE)
        x = (1.8 * np.sin(np.linspace(0, 200 * np.pi, 4096))).astype(np.float32)
        y = limiter.process(x, {"threshold": -1.0})
        assert np.max(np.abs(y)) <= 10 ** (-1.0 / 20.0) + 1e-6

    def test_does_not_mutate_its_input_and_returns_float32(self):
        limiter = LimiterEffect(RATE)
        x = np.full(4096, 1.5, dtype=np.float32)
        before = x.copy()
        y = limiter.process(x, {"threshold": -1.0})
        assert np.array_equal(x, before)
        assert y.dtype == np.float32

    def test_quiet_audio_passes_unchanged(self):
        limiter = LimiterEffect(RATE)
        x = (0.2 * np.sin(np.linspace(0, 50 * np.pi, 4096))).astype(np.float32)
        assert np.allclose(limiter.process(x, {"threshold": -1.0}), x)

    def test_gain_recovers_after_a_peak(self):
        limiter = LimiterEffect(RATE)
        limiter.process(np.full(4096, 2.0, dtype=np.float32), {"threshold": -1.0})
        quiet = np.full(RATE // 2, 0.1, dtype=np.float32)
        y = limiter.process(quiet, {"threshold": -1.0})
        assert y[-1] == pytest.approx(0.1, rel=1e-3)

    def test_chain_runs_limiter_last(self):
        chain = EffectsChain(RATE)
        x = np.full(4096, 0.5, dtype=np.float32)
        effects = {"gain": {"enabled": True, "db": 12}, "limiter": {"enabled": True, "threshold": -3}}
        y = chain.post_process(x, effects)
        assert np.max(np.abs(y)) <= 10 ** (-3.0 / 20.0) + 1e-6


class TestGate:
    def test_speech_level_signal_opens_the_gate(self):
        gate = GateEffect(RATE)
        x = (0.05 * np.sin(np.linspace(0, 400 * np.pi, 8192))).astype(np.float32)  # ~ -29 dBFS
        y = gate.process(x, {"threshold": -45})
        assert np.max(np.abs(y[-1024:])) > 0.04


class TestOctaveSpikes:
    def test_single_frame_octave_jump_is_replaced(self):
        f0 = np.array([200, 200, 400, 200, 200], dtype=np.float32)
        assert np.allclose(_suppress_octave_spikes(f0), 200)

    def test_glides_and_unvoiced_frames_are_kept(self):
        glide = np.array([200, 220, 245, 270, 300], dtype=np.float32)
        assert np.array_equal(_suppress_octave_spikes(glide), glide)
        edges = np.array([0, 0, 200, 0, 400, 400], dtype=np.float32)
        assert np.array_equal(_suppress_octave_spikes(edges), edges)


class TestDefaults:
    def test_rmvpe_needs_an_accelerator_and_weights(self, monkeypatch):
        caps = {"rmvpe": {"available": True}}
        monkeypatch.setattr(f0_registry, "_accelerator_available", lambda: False)
        assert f0_registry.resolve_default_f0_method(caps) == "pm"
        monkeypatch.setattr(f0_registry, "_accelerator_available", lambda: True)
        assert f0_registry.resolve_default_f0_method(caps) == "rmvpe"
        assert f0_registry.resolve_default_f0_method({"rmvpe": {"available": False}}) == "pm"

    @pytest.mark.parametrize("profile", [
        HardwareProfile(gpu_type="cuda", gpu_memory_gb=12.0, cpu_cores=16),
        HardwareProfile(gpu_type="mps", cpu_cores=8),
        HardwareProfile(gpu_type="directml", cpu_cores=4),
        HardwareProfile(cpu_cores=2),
    ])
    def test_recommendations_stay_within_realtime_buffers(self, profile):
        recommended = recommend_settings(profile)
        assert recommended.chunk_size <= 8192
        assert recommended.f0_method in {"pm", "rmvpe"}
        if not profile.has_gpu or profile.gpu_type == "directml":
            assert recommended.f0_method == "pm"
