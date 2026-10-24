"""Around the model: voice detection, noise reduction, ambience, loudness, and ready-only activation."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from backend.audio.denoise import SpectralDenoiser
from backend.audio.loudness import CEILING, KNEE, LoudnessMatcher, soft_limit
from backend.audio.streaming import RealtimeVoiceStream, StreamCleanup, StreamTiming
from backend.audio.voice_activity import FRAME, RATE, SpeechTimeline

CLIENT = 48000


class ScriptedDetector:
    """Returns one probability per 512 samples from a script (frame index -> probability)."""

    neural = True

    def __init__(self, probability) -> None:
        self.probability = probability
        self.reset()

    def reset(self) -> None:
        self._pending = 0
        self._frame = 0

    def feed(self, audio16: np.ndarray) -> np.ndarray:
        total = self._pending + len(audio16)
        count, self._pending = divmod(total, FRAME)
        values = [self.probability(self._frame + k) for k in range(count)]
        self._frame += count
        return np.array(values, np.float32)


def frames(seconds: float) -> int:
    return round(seconds * RATE / FRAME)


class TestSpeechTimeline:
    def timeline(self, probability) -> SpeechTimeline:
        timeline = SpeechTimeline(ScriptedDetector(probability))  # type: ignore[arg-type]
        timeline.feed(np.zeros(frames(3.0) * FRAME, np.float32))
        return timeline

    def test_speech_is_kept_with_pre_roll_and_hang_over(self):
        start, end = frames(1.0), frames(1.6)
        timeline = self.timeline(lambda k: 0.95 if start <= k < end else 0.01)
        gain = timeline.gain(0, 3 * RATE, RATE)
        assert gain[int(1.0 * RATE):int(1.6 * RATE)].min() == pytest.approx(1.0)
        assert gain[int(0.95 * RATE)] > 0.5            # pre-roll keeps the attack
        assert gain[int(1.85 * RATE)] > 0.5            # hang-over keeps the ending
        assert gain[int(0.5 * RATE)] == 0 and gain[int(2.5 * RATE)] == 0

    def test_a_breath_like_wobble_is_not_speech(self):
        # Two frames just over the threshold, as a breath produces, then nothing.
        timeline = self.timeline(lambda k: 0.62 if k in (frames(1.0), frames(1.0) + 1) else 0.02)
        assert timeline.gain(0, 3 * RATE, RATE).max() == 0

    def test_noise_learning_avoids_anything_speech_like(self):
        timeline = self.timeline(lambda k: 0.3 if k == frames(1.0) else 0.01)
        quiet = timeline.noise_only(0, 3 * RATE, RATE)
        assert not quiet[int(1.0 * RATE)] and quiet[int(0.3 * RATE)]
        assert not quiet[-1]  # the newest frames are not settled yet


class TestNoiseReduction:
    def test_room_noise_drops_and_speech_survives(self):
        rng = np.random.default_rng(0)
        noise = (0.01 * rng.standard_normal(2 * CLIENT)).astype(np.float32)
        denoiser = SpectralDenoiser(CLIENT, 12.0)
        denoiser.learn(noise, np.zeros(len(noise), np.float32))
        cleaned = denoiser.apply(noise)
        assert denoiser.reduction_now_db(noise, cleaned) >= 10.0
        t = np.arange(2 * CLIENT) / CLIENT
        voice = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        kept = denoiser.apply(voice + noise, np.ones(len(voice), np.float32))
        assert abs(10 * np.log10(np.mean(kept ** 2) / np.mean(voice ** 2))) < 1.0


class TestLoudness:
    def test_converted_voice_follows_the_speaker_level(self):
        t = np.arange(CLIENT) / CLIENT
        speaker = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        matcher = LoudnessMatcher(CLIENT)
        gain = matcher.update(speaker, 0.25 * speaker, np.ones(CLIENT, np.float32))
        assert 20 * np.log10(gain) == pytest.approx(12.04, abs=0.1)

    def test_silence_at_the_edges_of_speech_does_not_mislead_it(self):
        t = np.arange(CLIENT) / CLIENT
        speaker = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        speaker[: CLIENT // 2] = 1e-5
        converted = speaker.copy()
        converted[: CLIENT // 2] = 0.01   # model output where the speaker is still silent
        gain = LoudnessMatcher(CLIENT).update(speaker, converted, np.ones(CLIENT, np.float32))
        assert abs(20 * np.log10(gain)) < 0.5

    def test_limiter_is_transparent_below_the_knee_and_never_clips(self):
        quiet = np.linspace(-KNEE, KNEE, 101, dtype=np.float32)
        np.testing.assert_array_equal(soft_limit(quiet), quiet)
        assert np.abs(soft_limit(np.array([-4.0, 2.0, 1.0], np.float32))).max() <= CEILING


class TestAmbience:
    @staticmethod
    def run(ambient_db: float, noise_reduction_db: float, audio: np.ndarray) -> np.ndarray:
        speaking = lambda k: 0.97 if k < frames(1.5) else 0.01  # noqa: E731
        stream = RealtimeVoiceStream(CLIENT, lambda w: (-w, 16000), StreamTiming(),
                                     vad=ScriptedDetector(speaking))  # type: ignore[arg-type]
        stream.set_cleanup(StreamCleanup(isolate_voice=True, noise_reduction_db=noise_reduction_db,
                                         ambient_db=ambient_db, match_loudness=False))
        return np.concatenate([stream.process(audio[i:i + 4096]) for i in range(0, len(audio) - 4096 + 1, 4096)])

    def test_the_raw_input_never_reaches_the_output_outside_speech(self):
        rng = np.random.default_rng(1)
        t = np.arange(4 * CLIENT) / CLIENT
        # A loud breath the speech detector does not call speech, over room noise.
        from scipy import signal

        breath = signal.lfilter(*signal.butter(2, [300, 2500], btype="band", fs=CLIENT), rng.standard_normal(len(t)))
        audio = (0.003 * rng.standard_normal(len(t)) + 0.05 * breath * (t > 2.4)).astype(np.float32)
        for ambient_db, reduction in ((-60.0, 12.0), (0.0, 0.0), (0.0, 12.0)):
            late = self.run(ambient_db, reduction, audio)[int(2.8 * CLIENT):int(3.6 * CLIENT)]
            # The test renderer negates: converted sound correlates negatively, the raw
            # input positively. Only the raw input is forbidden.
            scores = [np.corrcoef(late, audio[int(2.8 * CLIENT) - d:int(3.6 * CLIENT) - d])[0, 1]
                      if late.std() > 0 else 0.0 for d in range(0, int(0.4 * CLIENT), 48)]
            assert max(scores) < 0.1, (ambient_db, reduction)

    def test_the_default_is_silence_between_words(self):
        rng = np.random.default_rng(2)
        room = (0.01 * rng.standard_normal(4 * CLIENT)).astype(np.float32)
        late = self.run(-60.0, 12.0, room)[int(2.8 * CLIENT):int(3.6 * CLIENT)]
        assert np.abs(late).max() < 1e-6


class TestReadyOnlyActivation:
    def test_previous_voice_keeps_converting_until_the_new_one_is_ready(self, manager, fake_voice, monkeypatch,
                                                                     settings):
        (settings.MODELS_DIR / "next.pth").write_bytes(b"x")
        release = threading.Event()
        started = threading.Event()
        new_voice = object()

        def prepare(path):
            started.set()
            release.wait(5)
            return new_voice, {"warmup_ms": 1.0}

        monkeypatch.setattr(manager, "_prepare", prepare)
        worker = threading.Thread(target=manager.activate, args=("next.pth",))
        worker.start()
        assert started.wait(5)
        assert manager.converter is fake_voice and manager.loading == {"name": "next.pth", "stage": "loading weights"}
        release.set()
        worker.join(5)
        assert manager.converter is new_voice and manager.loading is None
