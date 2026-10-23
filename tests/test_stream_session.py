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
        session.process(loud())
        sleep_chunks = int(np.ceil(session._sleep_after_seconds() * RATE / CHUNK))
        for _ in range(sleep_chunks):
            session.process(quiet)
        assert session.sleeping  # the last of those rendered once more, fading out
        calls = fake_voice.calls
        result = session.process(quiet)
        assert session.sleeping and result.model_ms == 0.0
        assert fake_voice.calls == calls
        np.testing.assert_array_equal(result.audio, quiet)

    def test_the_end_of_the_last_word_is_never_cut(self, session, fake_voice):
        # Output lags input by the lookahead and SOLA tail; going to sleep must
        # wait until everything still in flight has been emitted.
        t = np.arange(8 * CHUNK) / RATE  # voice-band tones survive the 16 kHz model rate
        tone = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 1300 * t)).astype(np.float32)
        speech = np.split(tone, 8)
        silence = [np.zeros(CHUNK, dtype=np.float32)] * 30
        output = np.concatenate([session.process(chunk).audio for chunk in speech + silence])
        assert session.sleeping
        heard = float(np.sum(output.astype(np.float64) ** 2))
        spoken = float(sum(np.sum(chunk.astype(np.float64) ** 2) for chunk in speech))
        assert heard > 0.95 * spoken

    def test_a_single_quiet_word_wakes_inference(self, session, fake_voice):
        for _ in range(12):
            session.process(np.zeros(CHUNK, dtype=np.float32))
        assert session.sleeping
        # 120 ms at about -50 dBFS inside an otherwise silent chunk: its whole-chunk
        # RMS is below the threshold, its loudest 10 ms is not.
        chunk = np.zeros(CHUNK, dtype=np.float32)
        t = np.arange(int(0.08 * RATE)) / RATE
        chunk[:len(t)] = 0.0045 * np.sin(2 * np.pi * 180 * t)
        calls = fake_voice.calls
        session.process(chunk)
        assert not session.sleeping and fake_voice.calls == calls + 1

    def test_waking_keeps_the_audio_heard_while_asleep_as_context(self, session, fake_voice):
        for _ in range(12):
            session.process(np.zeros(CHUNK, dtype=np.float32))
        breath = (1e-5 * np.random.default_rng(3).standard_normal(CHUNK)).astype(np.float32)
        session.process(breath)  # below the threshold: listened to, not rendered
        assert session.sleeping
        session.process(loud())
        np.testing.assert_array_equal(session._stream._history[-2 * CHUNK:-CHUNK], breath)

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


class TestAutoPitch:
    @pytest.fixture
    def trained_voice(self, fake_voice):
        from backend.inference.auto_pitch import F0Stats

        fake_voice.voice.f0_stats = F0Stats(220.0, 180.0, 280.0)
        fake_voice.f0_hz = 110.0  # an octave below the voice
        return fake_voice

    def test_moves_the_speaker_onto_the_voice_register(self, session, trained_voice):
        for seed in range(30):
            session.process(loud(seed))
        assert session.auto_pitch_shift == 12.0
        assert trained_voice.last_settings.pitch_shift == 12.0
        assert session.status()["auto_pitch_available"] is True

    def test_manual_mode_uses_the_slider(self, session, trained_voice):
        session.apply({"pitch_mode": "manual", "pitch_shift": 3})
        for seed in range(30):
            session.process(loud(seed))
        assert trained_voice.last_settings.pitch_shift == 3.0 and session.auto_pitch_shift is None

    def test_voices_without_pitch_stats_keep_the_manual_shift(self, session, fake_voice):
        fake_voice.f0_hz = 110.0
        for seed in range(30):
            session.process(loud(seed))
        assert fake_voice.last_settings.pitch_shift == 0.0
        assert session.status()["auto_pitch_available"] is False
