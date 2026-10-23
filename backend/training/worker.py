"""Training worker process: ``python -m backend.training.worker <job-dir>``.

Runs one job end to end in its own process (isolated from the studio's audio
loop and freed of all GPU memory when it exits) and reports through
``status.json``. A ``stop`` file in the job directory asks it to finish the
current step, save, export and exit.

Stages: assets -> prepare -> features -> index -> train -> done.
Every stage is resumable: finished artefacts are reused on restart.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from backend.assets import ASSETS, download_verified
from backend.config import settings
from backend.inference.auto_pitch import compute_f0_stats
from backend.inference.device import select_device
from backend.inference.hubert import load_hubert
from backend.inference.rmvpe import Rmvpe
from backend.training.dataset import (
    extract_features,
    load_segments,
    prepare_segments,
    quality_warnings,
    save_segments,
    voiced_percent,
)
from backend.training.hparams import CONFIGS, PRETRAINED
from backend.training.index_builder import build_index
from backend.training.plan import ValidationTracker, plan_training, validation_count
from backend.training.trainer import Trainer, TrainingProgress, auto_batch_size, build_examples

logger = logging.getLogger("backend.training.worker")

# Share of the overall progress bar per stage (training dominates).
_STAGE_SPAN = {"assets": (0.0, 0.03), "prepare": (0.03, 0.06), "features": (0.06, 0.14),
               "index": (0.14, 0.16), "train": (0.16, 1.0)}


class StopRequested(Exception):
    """The user asked to stop before training started."""


class Job:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.spec = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        self.status_path = directory / "status.json"
        self.status: dict = json.loads(self.status_path.read_text(encoding="utf-8")) \
            if self.status_path.is_file() else {}
        self._last_write = 0.0
        self._lock = threading.Lock()

    @property
    def stop_requested(self) -> bool:
        return (self.directory / "stop").exists()

    def check_stop(self) -> None:
        if self.stop_requested:
            raise StopRequested

    def start_heartbeat(self, interval: float = 20.0) -> None:
        """Refresh ``updated_at`` during long silent stages so the server knows we are alive."""
        def beat() -> None:
            while True:
                time.sleep(interval)
                self.report(force=True)

        threading.Thread(target=beat, name="heartbeat", daemon=True).start()

    def report(self, force: bool = False, **fields) -> None:
        with self._lock:
            stage = fields.get("stage", self.status.get("stage"))
            if "stage_progress" in fields and stage in _STAGE_SPAN:
                low, high = _STAGE_SPAN[stage]
                fields["progress"] = round(low + (high - low) * float(fields["stage_progress"]), 4)
            self.status.update(fields, updated_at=time.time())
            now = time.monotonic()
            if force or now - self._last_write > 1.0:
                partial = self.status_path.with_suffix(".part")
                partial.write_text(json.dumps(self.status), encoding="utf-8")
                partial.replace(self.status_path)
                self._last_write = now


def _ensure_assets(job: Job, sample_rate: int, assets_dir: Path) -> dict[str, Path]:
    needed = [*ASSETS, *PRETRAINED[sample_rate]]
    paths = {}
    for number, asset in enumerate(needed):
        directory = assets_dir / ("pretrained_v2" if asset.name.startswith("pretrained") else "")
        target = directory / asset.filename
        if not target.is_file() or target.stat().st_size != asset.size:
            job.report(force=True, stage="assets", message=f"Downloading {asset.filename}",
                       stage_progress=number / len(needed))
            download_verified(asset, target)
        paths[asset.name] = target
    return paths


def run(job: Job) -> None:
    spec = job.spec
    rate = int(spec["sample_rate"])
    config = CONFIGS[rate]
    models_dir = Path(spec.get("models_dir") or settings.MODELS_DIR)
    compute = select_device(spec.get("device") or settings.DEVICE)
    job.report(force=True, state="running", device=compute.describe(), error=None)

    paths = _ensure_assets(job, rate, Path(spec.get("assets_dir") or settings.ASSETS_DIR))
    dataset_path = job.directory / "dataset.npz"
    model_path = models_dir / f"{spec['name']}.pth"
    index_path = models_dir / f"{spec['name']}.index"

    segments = load_segments(dataset_path) if dataset_path.is_file() else []
    if not segments:
        job.report(force=True, stage="prepare", stage_progress=0.0, message="Cleaning and slicing audio")
        sources = sorted((job.directory / "uploads").iterdir())
        report: dict = {}
        segments = prepare_segments(sources, rate, lambda p: job.report(stage="prepare", stage_progress=p), report)
        if not segments:
            raise ValueError("No usable speech found in the uploaded audio.")
        save_segments(segments, dataset_path)
        job.report(force=True, quality=report, warnings=quality_warnings(report))
    job.check_stop()
    job.report(force=True, segments=len(segments),
               dataset_seconds=round(sum(len(s.audio) for s in segments) / rate, 1))

    if segments[0].phone is None:
        job.report(force=True, stage="features", stage_progress=0.0, message="Extracting voice features")
        hubert = load_hubert(paths["hubert"], compute.device, compute.half)
        rmvpe = Rmvpe(paths["rmvpe"], compute.device, compute.half)
        extract_features(segments, rate, hubert, rmvpe, compute.device, compute.half,
                         lambda p: job.report(stage="features", stage_progress=p))
        del hubert, rmvpe
        if compute.device.type == "cuda":
            torch.cuda.empty_cache()
        save_segments(segments, dataset_path)
        quality = {**(job.status.get("quality") or {}), "voiced_percent": voiced_percent(segments)}
        job.report(force=True, quality=quality, warnings=quality_warnings(quality))
    job.check_stop()

    if not index_path.is_file():
        job.report(force=True, stage="index", stage_progress=0.0, message="Building the retrieval index")
        models_dir.mkdir(parents=True, exist_ok=True)
        build_index([s.phone for s in segments if s.phone is not None], index_path, compute.device)

    examples = build_examples(segments, config)
    held_out = validation_count(len(examples))
    # Spread the held-out windows over the whole recording.
    picks = set(np.linspace(0, len(examples) - 1, held_out).round().astype(int).tolist()) if held_out else set()
    validation = [e for i, e in enumerate(examples) if i in picks]
    training = [e for i, e in enumerate(examples) if i not in picks]
    dataset_seconds = float(job.status.get("dataset_seconds") or 0.0)
    plan = plan_training(len(examples), dataset_seconds, auto_batch_size(compute),
                         batch_size=spec.get("batch_size"), epochs=spec.get("epochs"))
    f0_stats = compute_f0_stats([s.pitchf for s in segments if s.pitchf is not None])
    f0_payload = f0_stats.to_dict() if f0_stats else None

    trainer = Trainer(config, compute, training, job.directory / "checkpoints", plan.batch_size,
                      crops_per_example=plan.crops_per_example, validation=validation)
    if not trainer.resume():
        trainer.load_pretrained(paths[PRETRAINED[rate][0].name], paths[PRETRAINED[rate][1].name])
    total = plan.epochs
    save_every = max(1, int(spec.get("save_every", 25)))
    tracker = ValidationTracker.restore(job.status, plan.patience_epochs)
    tracker.last_gain_epoch = max(tracker.last_gain_epoch, trainer.epoch)  # a resume gets a fresh patience
    job.report(force=True, stage="train", stage_progress=trainer.epoch / total, epoch=trainer.epoch,
               total_epochs=total, batch_size=trainer.batch_size, precision=trainer.precision_label,
               plan=plan.describe(), f0_stats=f0_payload, message="Training")
    epoch_times: list[float] = []

    def on_progress(progress: TrainingProgress) -> None:
        if progress.epoch_seconds:
            epoch_times.append(progress.epoch_seconds)
        recent = epoch_times[-5:]
        eta = (sum(recent) / len(recent)) * (progress.total_epochs - progress.epoch) if recent else None
        done = (progress.epoch - 1 + progress.step / progress.steps_per_epoch) / progress.total_epochs
        if progress.epoch_seconds:
            done = progress.epoch / progress.total_epochs
        job.report(stage="train", stage_progress=min(done, 1.0), epoch=progress.epoch, step=progress.step,
                   steps_per_epoch=progress.steps_per_epoch,
                   losses={k: round(v, 4) for k, v in progress.losses.items()},
                   eta_seconds=round(eta) if eta is not None else None, batch_size=progress.batch_size)

    def export_voice_file(epoch: int, score: float | None) -> None:
        trainer.export(model_path, f0_payload)
        job.report(force=True, exported_epoch=epoch, best_epoch=epoch,
                   best_score=None if score is None else round(score, 4),
                   message=f"Best voice so far: epoch {epoch}")

    def on_epoch_end(epoch: int) -> None:
        if epoch % save_every == 0 or epoch == total:
            trainer.save_checkpoint()
            job.report(force=True, checkpoint_epoch=epoch)
        if not trainer.validation:
            if epoch % save_every == 0 or epoch == total:
                export_voice_file(epoch, None)  # nothing to validate against: keep the latest
            return
        if epoch % plan.eval_every and epoch != total:
            return
        score = trainer.evaluate()
        assert score is not None
        if tracker.add(epoch, score):
            export_voice_file(epoch, score)
        job.report(force=True, validation=tracker.history[-400:], validation_score=round(score, 4))
        if tracker.should_stop(epoch):
            logger.info("No validation gain for %d epochs; stopping at epoch %d (best %d)",
                        epoch - tracker.last_gain_epoch, epoch, tracker.best_epoch)

    def should_stop() -> bool:
        return job.stop_requested or tracker.stopped

    outcome = trainer.train(total, on_progress, should_stop, on_epoch_end)
    if trainer.epoch > int(job.status.get("checkpoint_epoch") or 0) and trainer.epoch > 0:
        trainer.save_checkpoint()
        job.report(force=True, checkpoint_epoch=trainer.epoch)
    if not job.status.get("exported_epoch") and trainer.epoch > 0:
        export_voice_file(trainer.epoch, None)  # stopped before the first validation
    finished = outcome == "done" or tracker.stopped
    exported = int(job.status.get("exported_epoch") or 0)
    if tracker.stopped:
        message = (f"Finished at epoch {trainer.epoch}: the voice stopped improving. "
                   f"Saved the best one, epoch {exported}.")
    elif finished:
        message = f"Finished. Saved the best epoch, {exported}."
    else:
        message = f"Stopped at epoch {trainer.epoch}. Saved the best epoch, {exported}."
    job.report(force=True, state="done" if finished else "stopped",
               stage="done" if finished else job.status.get("stage"),
               progress=1.0 if finished else job.status.get("progress"), model=model_path.name,
               index=index_path.name, message=message)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)
    job = Job(Path(sys.argv[1]))
    (job.directory / "stop").unlink(missing_ok=True)
    job.start_heartbeat()
    try:
        run(job)
    except StopRequested:
        job.report(force=True, state="stopped", message="Stopped before training started")
    except Exception as exc:
        logger.error("Training failed:\n%s", traceback.format_exc())
        job.report(force=True, state="error", error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
