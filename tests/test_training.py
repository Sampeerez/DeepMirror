"""Voice training: data preparation, index, checkpoint layout, one real step, and the jobs API."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import soundfile as sf
import torch

from backend.inference.device import ComputeDevice
from backend.inference.retrieval import FeatureIndex, read_index
from backend.inference.voice_model import VoiceModel
from backend.training.dataset import (
    Segment,
    load_segments,
    normalise,
    save_segments,
    split_on_pauses,
    windows,
)
from backend.training.hparams import CONFIGS
from backend.training.index_builder import build_index, kmeans
from backend.training.networks import TrainableSynthesizer, export_voice, from_rvc_state, to_rvc_state
from backend.training.trainer import Trainer, build_examples

RATE = 40000


def speech_like(seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(140 + 20 * np.sin(2 * np.pi * 0.5 * t)) / rate
    return (0.3 * sum(0.7 ** k * np.sin(k * phase) for k in range(1, 15))).astype(np.float32)


class TestDatasetPreparation:
    def test_pauses_split_the_audio(self):
        speech = speech_like(2.0)
        audio = np.concatenate([speech, np.zeros(RATE), speech, np.zeros(RATE), speech])
        chunks = split_on_pauses(audio, RATE)
        assert len(chunks) == 3
        assert all(1.9 * RATE < len(c) < 2.6 * RATE for c in chunks)

    def test_long_chunks_become_overlapping_windows(self):
        pieces = windows(speech_like(10.0), RATE)
        assert [round(len(p) / RATE, 1) for p in pieces] == [3.7, 3.7, 3.2]  # starts 0, 3.4, 6.8 s

    def test_normalisation_and_clipping_guard(self):
        piece = normalise(speech_like(1.0) * 2)
        assert piece is not None and 0.6 < np.abs(piece).max() <= 0.9 + 1e-6
        assert normalise(np.full(100, 3.0, np.float32)) is None
        assert normalise(np.zeros(100, np.float32)) is None

    def test_segments_round_trip_without_pickles(self, tmp_path):
        segments = [Segment(np.ones(n, np.float32), np.ones((n // 800, 768), np.float16),
                            np.ones(n // 400, np.int64), np.ones(n // 400, np.float32)) for n in (8000, 12000)]
        save_segments(segments, tmp_path / "dataset.npz")
        loaded = load_segments(tmp_path / "dataset.npz")
        assert [len(s.audio) for s in loaded] == [8000, 12000]
        assert loaded[1].phone.shape == (15, 768) and loaded[1].phone.dtype == np.float16


class TestIndex:
    def test_kmeans_finds_separated_clusters(self):
        rng = np.random.default_rng(0)
        points = np.concatenate([rng.normal(c, 0.1, (200, 4)) for c in (0, 5, 10)]).astype(np.float32)
        centroids = kmeans(torch.from_numpy(points), 3).numpy()
        assert sorted(np.round(centroids[:, 0]).astype(int).tolist()) == [0, 5, 10]

    def test_built_index_reads_back_and_searches_exactly(self, tmp_path):
        rng = np.random.default_rng(1)
        features = [rng.standard_normal((300, 16)).astype(np.float16) for _ in range(5)]
        count = build_index(features, tmp_path / "voice.index", torch.device("cpu"))
        data = read_index(tmp_path / "voice.index")
        assert count == 1500 and data.vectors.shape == (1500, 16)
        assert data.centroids is not None and len(data.centroids) == min(int(16 * math.sqrt(1500)), 1500 // 39)
        query = torch.from_numpy(features[2][7:8].astype(np.float32))
        distances, _ = FeatureIndex(data, torch.device("cpu")).search(query)
        assert float(distances[0, 0]) < 1e-3


class TestCheckpointLayout:
    def test_rvc_key_layout_round_trips(self):
        generator = TrainableSynthesizer(CONFIGS[RATE])
        rvc = to_rvc_state(generator.state_dict(), generator=True)
        assert any(k.endswith(".weight_g") for k in rvc) and not any("parametrizations" in k for k in rvc)
        assert generator.load_state_dict(from_rvc_state(rvc, generator=True)).missing_keys == []

    def test_exported_voice_loads_for_inference(self, tmp_path):
        torch.save(export_voice(TrainableSynthesizer(CONFIGS[RATE]), 7), tmp_path / "voice.pth")
        voice = VoiceModel.load(tmp_path / "voice.pth", torch.device("cpu"), False)
        assert (voice.version, voice.sample_rate, voice.use_pitch) == ("v2", RATE, True)


def test_one_training_step_on_cpu_is_finite(tmp_path):
    config = CONFIGS[RATE]
    frames = config.segment_frames + 20
    rng = np.random.default_rng(3)
    segments = [Segment(speech_like(frames * config.hop / RATE), rng.standard_normal((frames // 2, 768)).astype(np.float16),
                        rng.integers(1, 255, frames), rng.uniform(80, 300, frames).astype(np.float32))
                for _ in range(2)]
    cpu = ComputeDevice(torch.device("cpu"), False, "CPU", None, 1, 6, 38, 41)
    trainer = Trainer(config, cpu, build_examples(segments, config), tmp_path / "ckpt", batch_size=2)
    losses = trainer._step(trainer._collate(trainer.examples))
    assert all(math.isfinite(v) for v in losses.values())
    trainer.epoch = 1
    trainer.save_checkpoint()
    resumed = Trainer(config, cpu, trainer.examples, tmp_path / "ckpt", batch_size=2)
    assert resumed.resume() and resumed.epoch == 1


# --------------------------------------------------------------------------
# Jobs API
# --------------------------------------------------------------------------


def wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, speech_like(seconds, 16000), 16000, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def no_worker(monkeypatch):
    started = []
    monkeypatch.setattr("backend.training.jobs.TrainingJobs._start", lambda self, job_id: started.append(job_id))
    return started


class TestJobsApi:
    def test_info_describes_limits(self, client):
        info = client.get("/api/training/info").json()
        assert info["default_sample_rate"] == 40000 and info["min_audio_seconds"] == 60

    def test_too_little_audio_is_rejected(self, client, no_worker):
        response = client.post("/api/training/jobs", data={"name": "voz"},
                               files=[("files", ("a.wav", wav(5), "audio/wav"))])
        assert response.status_code == 400 and "at least" in response.json()["detail"]
        assert no_worker == [] and client.get("/api/training/jobs").json() == []

    def test_unsupported_files_are_rejected(self, client, no_worker):
        response = client.post("/api/training/jobs", data={"name": "voz"},
                               files=[("files", ("notes.txt", b"hello", "text/plain"))])
        assert response.status_code == 400

    def test_create_list_stop_and_delete(self, client, no_worker, settings):
        response = client.post("/api/training/jobs", data={"name": "Mi Voz!", "epochs": "50"},
                               files=[("files", ("a.wav", wav(40), "audio/wav")),
                                      ("files", ("b.wav", wav(40), "audio/wav"))])
        assert response.status_code == 200, response.text
        job = response.json()
        assert job["name"] == "Mi_Voz" and job["epochs"] == 50 and job["state"] == "queued"
        assert job["audio_seconds"] == pytest.approx(80, abs=0.5) and no_worker == [job["id"]]
        assert [j["id"] for j in client.get("/api/training/jobs").json()] == [job["id"]]
        assert client.post(f"/api/training/jobs/{job['id']}/stop").status_code == 200
        assert (settings.DATA_DIR / "training" / job["id"] / "stop").exists()
        assert client.delete(f"/api/training/jobs/{job['id']}").status_code == 200
        assert client.get(f"/api/training/jobs/{job['id']}").status_code == 404

    def test_existing_voice_names_are_protected(self, client, no_worker, settings):
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (settings.MODELS_DIR / "taken.pth").write_bytes(b"x")
        response = client.post("/api/training/jobs", data={"name": "taken"},
                               files=[("files", ("a.wav", wav(70), "audio/wav"))])
        assert response.status_code == 400 and "already exists" in response.json()["detail"]


def test_heartbeat_marks_detached_workers_alive_or_interrupted(settings):
    import json
    import time

    from backend.training.jobs import TrainingJobs

    root = settings.DATA_DIR / "training"
    for job_id, age in (("aaaaaaaaaaaa", 5), ("bbbbbbbbbbbb", 3600)):
        (root / job_id).mkdir(parents=True)
        (root / job_id / "job.json").write_text(json.dumps({"name": job_id}), encoding="utf-8")
        (root / job_id / "status.json").write_text(
            json.dumps({"state": "running", "updated_at": time.time() - age}), encoding="utf-8")
    jobs = TrainingJobs(settings)
    assert jobs.describe("aaaaaaaaaaaa")["state"] == "running"
    assert jobs.describe("bbbbbbbbbbbb")["state"] == "interrupted"
    assert jobs._running_id() == "aaaaaaaaaaaa"


# --------------------------------------------------------------------------
# Automatic plan, validation and best-epoch selection
# --------------------------------------------------------------------------


class TestTrainingPlan:
    def test_five_minutes_gets_a_full_plan_with_validation(self):
        from backend.training.plan import plan_training

        plan = plan_training(windows=85, dataset_seconds=300, vram_batch=5)
        assert plan.validation_windows == 7 and plan.crops_per_example == 2
        assert plan.batch_size == 5
        assert 150 <= plan.epochs <= 1000 and plan.patience_epochs >= 40

    def test_epochs_follow_a_step_budget(self):
        from backend.training.plan import plan_training

        short = plan_training(windows=85, dataset_seconds=300, vram_batch=8)
        long = plan_training(windows=500, dataset_seconds=1800, vram_batch=8)
        assert short.epochs > long.epochs and long.crops_per_example == 1

    def test_explicit_choices_win(self):
        from backend.training.plan import plan_training

        plan = plan_training(windows=85, dataset_seconds=300, vram_batch=8, batch_size=3, epochs=42)
        assert (plan.batch_size, plan.epochs, plan.eval_every) == (3, 42, 2)

    def test_tiny_datasets_skip_validation(self):
        from backend.training.plan import plan_training

        assert plan_training(windows=12, dataset_seconds=60, vram_batch=4).validation_windows == 0


class TestValidationTracker:
    def test_keeps_the_best_epoch_when_the_voice_gets_worse(self):
        from backend.training.plan import ValidationTracker

        tracker = ValidationTracker(patience_epochs=20)
        kept = [tracker.add(epoch, score) for epoch, score in [(5, 3.0), (10, 2.5), (15, 2.7), (20, 2.9)]]
        assert kept == [True, True, False, False] and tracker.best_epoch == 10

    def test_on_a_plateau_the_most_trained_epoch_is_kept(self):
        from backend.training.plan import ValidationTracker

        # The scores of the real 5-minute run from epoch 85 on: noise around a plateau.
        tracker = ValidationTracker(patience_epochs=40)
        for epoch, score in [(80, 0.4098), (85, 0.3999), (90, 0.4086), (95, 0.4109), (100, 0.3957), (105, 0.4067),
                             (110, 0.4046), (115, 0.407), (120, 0.4023), (125, 0.4023), (130, 0.4032)]:
            tracker.add(epoch, score)
        assert tracker.best_epoch == 130
        assert not tracker.add(135, 0.45) and tracker.best_epoch == 130  # a real regression is never kept

    def test_stops_after_patience_without_gain(self):
        from backend.training.plan import ValidationTracker

        tracker = ValidationTracker(patience_epochs=10)
        for epoch, score in [(5, 2.0), (10, 2.1), (15, 2.1)]:
            tracker.add(epoch, score)
        assert not tracker.should_stop(10) and tracker.should_stop(15)
        late = ValidationTracker(patience_epochs=10, min_epochs=30)
        for epoch, score in [(5, 2.0), (10, 2.1), (15, 2.1)]:
            late.add(epoch, score)
        assert not late.should_stop(15)

    def test_restores_from_status(self):
        from backend.training.plan import ValidationTracker

        tracker = ValidationTracker.restore({"validation": [[5, 3.0, 3.0], [10, 2.0, 2.6]]}, 40)
        assert tracker.best_epoch == 10 and not tracker.stopped


def test_validation_scores_the_inference_path(tmp_path):
    config = CONFIGS[RATE]
    frames = config.segment_frames + 20
    rng = np.random.default_rng(4)
    segments = [Segment(speech_like(frames * config.hop / RATE), rng.standard_normal((frames // 2, 768)).astype(np.float16),
                        rng.integers(1, 255, frames), rng.uniform(80, 300, frames).astype(np.float32))
                for _ in range(3)]
    cpu = ComputeDevice(torch.device("cpu"), False, "CPU", None, 1, 6, 38, 41)
    examples = build_examples(segments, config)
    trainer = Trainer(config, cpu, examples[:2], tmp_path / "ckpt", batch_size=2, crops_per_example=2,
                      validation=examples[2:])
    first, second = trainer.evaluate(), trainer.evaluate()
    assert first is not None and math.isfinite(first) and first == pytest.approx(second)  # fixed seed
    assert trainer.generator.training


def test_exported_voice_carries_pitch_stats(tmp_path):
    from backend.inference.auto_pitch import compute_f0_stats

    stats = compute_f0_stats([np.r_[np.zeros(50), np.full(200, 220.0)]])
    assert stats is not None and stats.median_hz == pytest.approx(220.0)
    torch.save(export_voice(TrainableSynthesizer(CONFIGS[RATE]), 7, stats.to_dict()), tmp_path / "voice.pth")
    voice = VoiceModel.load(tmp_path / "voice.pth", torch.device("cpu"), False)
    assert voice.f0_stats is not None and voice.f0_stats.median_hz == pytest.approx(220.0)
    assert voice.describe()["f0_stats"]["median_hz"] == 220.0


class TestRecordingQuality:
    def test_quiet_recordings_are_brought_to_speech_level(self):
        from backend.training.dataset import normalise_loudness

        quiet = 0.01 * speech_like(3.0)
        louder = normalise_loudness(quiet, RATE)
        assert np.abs(louder).max() / np.abs(quiet).max() > 5 and np.abs(louder).max() <= 0.95 + 1e-6

    def test_warnings_explain_weak_material(self):
        from backend.training.dataset import quality_warnings

        warnings = quality_warnings({"speech_seconds": 90, "clipped_percent": 1.0, "snr_db": 12, "voiced_percent": 20})
        assert len(warnings) == 4
        assert quality_warnings({"speech_seconds": 400, "clipped_percent": 0.0, "snr_db": 45,
                                 "voiced_percent": 70}) == []
