"""Per-connection streaming behaviour: model/DSP/bypass paths, silence saver, failures, settings."""

from __future__ import annotations

import numpy as np
import pytest

from backend.api.stream import StreamSession

CHUNK = 4096
RATE = 48000


@pytest.fixture
def session(manager, settings) -> StreamSession:
    session = StreamSession(manager, settings)
    session.apply({"sample_rate": RATE, "chunk_size": CHUNK})
    return session


def loud(seed: int = 0) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(CHUNK)).astype(np.float32)


class TestPaths:
    def test_model_path_renders_through_the_realtime_stream(self, session, fake_voice):
        result = session.process(loud())
        assert result.mode == "rvc" and len(result.audio) == CHUNK
        assert fake_voice.calls == 1

    def test_no_model_uses_the_dsp_pitch_shifter(self, session):
        session.apply({"pitch_shift": 12})
        chunk = np.sin(np.arange(CHUNK, dtype=np.float32) * 0.1).astype(np.float32)
        result = session.process(chunk)
        assert result.mode == "dsp" and not np.array_equal(result.audio, chunk)

    def test_bypass_returns_the_input_untouched(self, session, fake_voice):
        session.apply({"bypass": True, "effects": {"gain": {"enabled": True, "db": 12}}})
        chunk = loud()
        result = session.process(chunk)
        assert result.mode == "bypass" and result.audio is chunk
        assert fake_voice.calls == 0
        assert session.status()["effects_active"] == 0

    def test_voice_settings_reach_the_converter(self, session, fake_voice):
        session.apply({"index_rate": 0.4, "f0_method": "pm", "protect": 0.9})
        session.process(loud())
        used = fake_voice.last_settings
        assert (used.index_rate, used.f0_method, used.protect) == (0.4, "pm", 0.5)


class TestSilenceSaver:
    def test_sustained_silence_stops_inference_after_the_hold(self, session, fake_voice):
        quiet = np.zeros(CHUNK, dtype=np.float32)
        hold_chunks = int(np.ceil(session.config.SILENCE_HOLD_SECONDS * RATE / CHUNK))
        for _ in range(hold_chunks - 1):
            session.process(quiet)
        calls = fake_voice.calls
        result = session.process(quiet)
        assert session.sleeping and result.model_ms == 0.0
        assert fake_voice.calls == calls
        np.testing.assert_array_equal(result.audio, quiet)

    def test_voice_wakes_inference_on_the_first_loud_chunk(self, session, fake_voice):
        for _ in range(10):
            session.process(np.zeros(CHUNK, dtype=np.float32))
        assert session.sleeping
        calls = fake_voice.calls
        session.process(loud())
        assert not session.sleeping and fake_voice.calls == calls + 1

    def test_disabled_saver_always_infers(self, session, fake_voice):
        session.apply({"silence_saver": False})
        for _ in range(10):
            session.process(np.zeros(CHUNK, dtype=np.float32))
        assert fake_voice.calls == 10 and not session.sleeping

    def test_duty_cycle_is_reported_and_reset(self, session, fake_voice):
        for _ in range(4):
            session.process(loud())
        status = session.status()
        assert status["inference_duty_percent"] == 100.0 and status["active_model"] == "voice.pth"
        assert session.status()["inference_duty_percent"] == 0.0


class TestFailures:
    def test_inference_errors_mute_the_stream_without_leaking_details(self, session, fake_voice):
        def explode(*_args, **_kwargs):
            raise RuntimeError(r"C:\private\path exploded")

        fake_voice.convert = explode
        result = session.process(loud())
        assert result.mode == "error"
        assert not result.audio.any()
        assert "RuntimeError" in session.processing_error and "private" not in session.processing_error

    def test_recovery_clears_the_error(self, session, fake_voice):
        original = fake_voice.convert
        fake_voice.convert = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
        session.process(loud())
        fake_voice.convert = original
        assert session.process(loud()).mode == "rvc" and session.processing_error is None


class TestSettings:
    @pytest.mark.parametrize(("key", "value"), [("sample_rate", 1000), ("chunk_size", 10), ("chunk_size", "x")])
    def test_invalid_transport_values_are_ignored(self, session, key, value):
        before = getattr(session, key)
        session.apply({key: value})
        assert getattr(session, key) == before

    def test_silence_threshold_is_range_checked(self, session):
        session.apply({"silence_threshold_db": -60})
        session.apply({"silence_threshold_db": -120})
        assert session.silence_threshold_db == -60.0

    def test_sample_rate_change_rebuilds_the_effect_chain(self, session):
        session.apply({"sample_rate": 44100})
        assert session.chain.sr == 44100

    def test_quality_profile_timing_is_applied(self, session):
        session.apply({"lookback_seconds": 0.5, "lookahead_seconds": 0.05})
        assert (session.timing.lookback_seconds, session.timing.lookahead_seconds) == (0.5, 0.05)
        session.apply({"lookahead_seconds": 9})
        assert session.timing.lookahead_seconds == 0.05
