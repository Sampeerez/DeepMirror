"""Quiet speech, melody and vocal sounds: voicing clean-up, intonation, speaker profile, vocal frames."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from backend.audio.streaming import envelope_ceiling
from backend.audio.voice_activity import FRAME, vocal_frames
from backend.inference.auto_pitch import F0Stats, SpeakerProfile, auto_expressiveness
from backend.inference.pitch import HOP, clean_voicing, expand_intonation

RATE = 16000


def harmonic(seconds: float, f0: float, level: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * sum(0.7 ** k * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 10))).astype(np.float32)


class TestVoicing:
    def test_voice_stays_voiced(self):
        audio = harmonic(1.0, 150)
        f0 = np.full(len(audio) // HOP, 150.0, np.float32)
        assert (clean_voicing(f0, audio) > 0).mean() > 0.95

    def test_a_fricative_called_voiced_is_unvoiced(self):
        rng = np.random.default_rng(0)
        hiss = signal.lfilter(*signal.butter(2, 4000, btype="high", fs=RATE), rng.standard_normal(RATE))
        audio = np.concatenate([harmonic(0.5, 150), (0.05 * hiss).astype(np.float32)])
        f0 = np.full(len(audio) // HOP, 150.0, np.float32)    # the tracker kept "voicing" through the s
        cleaned = clean_voicing(f0, audio)
        assert (cleaned[:45] > 0).mean() > 0.9 and (cleaned[60:] > 0).mean() < 0.1

    def test_blips_and_octave_jumps_are_fixed(self):
        audio = harmonic(1.0, 150)
        f0 = np.full(len(audio) // HOP, 150.0, np.float32)
        f0[40:42] = 0.0          # a 20 ms dropout inside voice
        f0[60:63] = 300.0        # an octave error
        cleaned = clean_voicing(f0, audio)
        assert (cleaned[40:42] > 0).all()
        np.testing.assert_allclose(cleaned[60:63], 150.0, rtol=0.01)


class TestIntonation:
    @staticmethod
    def spread(f0, centre=150.0):
        voiced = f0 > 0
        return np.std(12 * np.log2(f0[voiced] / centre))

    def test_widens_speech_melody_around_the_centre(self):
        frames = np.arange(400)
        f0 = (150 * 2 ** (2.5 * np.sin(frames / 6) / 12)).astype(np.float32)   # a fast, speech-like glide
        assert self.spread(expand_intonation(f0, 1.5, 150.0)) / self.spread(f0) == pytest.approx(1.5, rel=0.08)

    def test_sung_notes_keep_their_pitch(self):
        notes = np.repeat([0, 4, 7, 12, 7, 4, 0], 55)                   # 550 ms per note
        vibrato = 0.35 * np.sin(2 * np.pi * 5.5 * np.arange(len(notes)) / 100)
        f0 = (150 * 2 ** ((notes + vibrato) / 12)).astype(np.float32)
        out = expand_intonation(f0, 1.25, 150.0)
        cents = 1200 * np.log2(out / f0)
        middle = np.concatenate([np.arange(k * 55 + 12, k * 55 + 43) for k in range(7)])
        assert np.median(np.abs(cents[middle])) < 10

    def test_a_laugh_far_above_the_speaking_pitch_is_untouched(self):
        frames = np.arange(300)
        f0 = (150 * 2 ** ((10 + 2 * np.sin(frames / 4)) / 12)).astype(np.float32)   # ~10 semitones up
        np.testing.assert_allclose(expand_intonation(f0, 1.25, 150.0), f0, rtol=2e-3)

    def test_unvoiced_frames_and_factor_one_are_untouched(self):
        f0 = np.array([0, 140, 150, 160, 0, 155, 150], np.float32)
        assert (expand_intonation(f0, 1.4)[[0, 4]] == 0).all()
        np.testing.assert_array_equal(expand_intonation(f0, 1.0), f0)


class TestSpeakerProfile:
    def test_learns_the_speaker_and_calibrates(self):
        profile = SpeakerProfile()
        assert profile.stats is None and profile.calibration == 0
        rng = np.random.default_rng(1)
        profile.observe(110 * 2 ** (rng.normal(0, 2, 1000) / 12))
        assert profile.stats is not None and profile.stats.median_hz == pytest.approx(110, rel=0.05)
        assert profile.calibration == 1.0 and profile.describe()["calibration"] == 1.0

    def test_a_saved_profile_applies_until_new_speech_arrives(self):
        profile = SpeakerProfile(F0Stats(200.0, 170.0, 240.0))
        assert profile.stats.median_hz == 200.0 and profile.calibration == 1.0
        assert profile.observe(np.zeros(10)) is True       # a pause

    def test_expressiveness_matches_the_target_range(self):
        speaker = F0Stats(110.0, 100.0, 121.0)              # a flat speaker
        target = F0Stats(220.0, 180.0, 269.0)               # a lively voice
        assert auto_expressiveness(target, speaker) == pytest.approx(1.8)
        assert auto_expressiveness(speaker, speaker) == pytest.approx(1.0)


class TestVocalSounds:
    def test_pitched_frames_are_periodic_and_noise_is_not(self):
        rng = np.random.default_rng(2)
        laugh = harmonic(FRAME * 4 / RATE, 220).reshape(4, FRAME).astype(np.float64)
        breath = (0.05 * rng.standard_normal((4, FRAME)))
        assert vocal_frames(laugh)[0].min() > 0.8
        assert vocal_frames(breath)[0].max() < 0.4


class TestEnvelopeCeiling:
    def test_invented_sound_over_silence_is_attenuated_and_voice_is_not(self):
        rate = 48000
        t = np.arange(rate) / rate
        voice = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        reference = voice.copy()
        reference[rate // 2:] = 1e-4                      # the speaker stopped
        out = envelope_ceiling(voice, reference, rate)
        assert np.abs(out[: rate // 2 - 2000]).max() == pytest.approx(0.1, rel=0.01)
        assert np.abs(out[rate // 2 + 2000:]).max() < 0.002


class TestEnvelopeFloor:
    def test_a_dip_inside_speech_is_filled_and_pauses_are_not(self):
        from backend.audio.streaming import envelope_floor

        rate = 48000
        t = np.arange(rate) / rate
        voice = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        dipped = voice.copy()
        dipped[24000:25920] *= 0.1                       # a 40 ms, 20 dB dip mid-word
        heard = np.ones(rate, bool)
        filled = envelope_floor(dipped, voice, rate, heard)
        assert np.abs(filled[24480:25440]).max() > 0.04  # back within 6 dB (+-) of the speaker
        quiet = np.zeros(rate, bool)
        np.testing.assert_array_equal(envelope_floor(dipped, voice, rate, quiet), dipped)
        np.testing.assert_array_equal(envelope_floor(dipped, voice, rate, None), dipped)
