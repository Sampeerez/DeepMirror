"""Wire protocol, resampling, realtime streaming (SOLA) and DSP effects."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from backend.audio.effects import EffectsChain
from backend.audio.effects.dynamics import GateEffect, LimiterEffect
from backend.audio.protocol import HEADER_SIZE, decode_frame, encode_frame
from backend.audio.resampling import resample
from backend.audio.streaming import RealtimeVoiceStream, StreamTiming, _sola_offset

RATE = 48000


def tone(seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class TestProtocol:
    def test_header_is_two_little_endian_uint32(self):
        assert HEADER_SIZE == 8
        frame = encode_frame(np.zeros(2, np.float32), 7, 9)
        assert struct.unpack("<II", frame[:8]) == (7, 9)

    def test_round_trip(self):
        samples = np.linspace(-1, 1, 64, dtype=np.float32)
        decoded, sequence, reserved = decode_frame(encode_frame(samples, 42, 1234))
        np.testing.assert_array_equal(decoded, samples)
        assert (sequence, reserved) == (42, 1234)
        decoded[0] = 5.0  # decoded buffers are writable copies

    def test_reserved_field_wraps(self):
        _, _, reserved = decode_frame(encode_frame(np.zeros(1, np.float32), 1, 2**32 + 5))
        assert reserved == 5

    @pytest.mark.parametrize("data", [b"", b"\x00" * 7])
    def test_short_frames_are_rejected(self, data):
        with pytest.raises(ValueError):
            decode_frame(data)

    def test_misaligned_payload_is_rejected(self):
        with pytest.raises(ValueError):
            decode_frame(b"\x00" * 9)


class TestResample:
    def test_identity_when_rates_match(self):
        x = tone(0.1)
        assert resample(x, RATE, RATE) is x

    def test_length_and_dtype(self):
        y = resample(tone(1.0), RATE, 16000)
        assert len(y) == 16000 and y.dtype == np.float32

    def test_band_limited_tone_survives_a_round_trip(self):
        x = tone(1.0)
        y = resample(resample(x, RATE, 16000), 16000, RATE)
        assert np.max(np.abs(y[2000:-2000] - x[2000:-2000])) < 1e-3


class TestRealtimeStream:
    @pytest.mark.parametrize(("rate", "chunk"), [(48000, 4096), (48000, 8192), (44100, 4096), (48000, 2048)])
    def test_identity_render_reconstructs_the_input_with_a_fixed_delay(self, rate, chunk):
        x = tone(3.0, rate)
        stream = RealtimeVoiceStream(rate, lambda window: (window, 16000))
        out = np.concatenate([stream.process(c) for c in x[: len(x) // chunk * chunk].reshape(-1, chunk)])
        assert len(out) == len(x) // chunk * chunk
        errors = [np.abs(out[rate + d: 2 * rate + d] - x[rate: 2 * rate]).max() for d in range(int(0.3 * rate))]
        delay = int(np.argmin(errors))
        assert errors[delay] < 1e-3, "no samples may be dropped or repeated"
        assert abs(delay / rate - StreamTiming().algorithmic_delay()) < 0.015

    def test_seams_are_as_smooth_as_the_signal(self):
        chunk = 4096
        x = tone(2.0)
        stream = RealtimeVoiceStream(RATE, lambda window: (window, 16000))
        out = np.concatenate([stream.process(c) for c in x[: len(x) // chunk * chunk].reshape(-1, chunk)])
        seams = np.arange(chunk, len(out), chunk) - 1
        assert np.abs(np.diff(out)[seams]).max() <= np.abs(np.diff(x)).max() * 1.05

    def test_output_length_always_matches_the_input(self):
        stream = RealtimeVoiceStream(RATE, lambda window: (window * 0, 16000))
        for size in (128, 1000, 4096, 9000):
            assert len(stream.process(np.ones(size, np.float32))) == size

    def test_silent_tail_keeps_the_centre_alignment(self):
        assert _sola_offset(np.zeros(100), np.random.default_rng(0).standard_normal(400), 200) == 100

    def test_sola_finds_a_shifted_copy(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(1000)
        assert _sola_offset(signal[37:137], signal, 80) == 37


class TestLimiter:
    def test_output_never_exceeds_the_ceiling(self):
        x = (1.8 * np.sin(np.linspace(0, 200 * np.pi, 4096))).astype(np.float32)
        y = LimiterEffect(RATE).process(x, {"threshold": -1.0})
        assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-6

    def test_does_not_mutate_its_input(self):
        x = np.full(4096, 1.5, dtype=np.float32)
        before = x.copy()
        y = LimiterEffect(RATE).process(x, {"threshold": -1.0})
        np.testing.assert_array_equal(x, before)
        assert y.dtype == np.float32

    def test_quiet_audio_passes_unchanged(self):
        x = (0.2 * np.sin(np.linspace(0, 50 * np.pi, 4096))).astype(np.float32)
        np.testing.assert_allclose(LimiterEffect(RATE).process(x, {"threshold": -1.0}), x)

    def test_chain_runs_the_limiter_last(self):
        effects = {"gain": {"enabled": True, "db": 12}, "limiter": {"enabled": True, "threshold": -3}}
        y = EffectsChain(RATE).post_process(np.full(4096, 0.5, np.float32), effects)
        assert np.max(np.abs(y)) <= 10 ** (-3 / 20) + 1e-6


def test_speech_level_signal_opens_the_gate():
    x = (0.05 * np.sin(np.linspace(0, 400 * np.pi, 8192))).astype(np.float32)
    y = GateEffect(RATE).process(x, {"threshold": -45})
    assert np.max(np.abs(y[-1024:])) > 0.04


def test_every_effect_runs_on_a_stream_without_errors():
    chain = EffectsChain(RATE)
    effects = {key: {"enabled": True} for key in ("robot", "whisper", "distortion", "bitcrush", "telephone",
                                                  "chorus", "echo", "reverb", "eq", "compressor", "gain",
                                                  "limiter")}
    for _ in range(4):
        out = chain.post_process(tone(0.085), effects, formant_shift=3.0)
        assert out.dtype == np.float32 and np.isfinite(out).all() and np.abs(out).max() <= 1.0
    assert EffectsChain.count_active(effects, 3.0) == len(effects) + 1
