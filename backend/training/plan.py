"""Automatic training plan: batch size, epoch budget, validation and early stopping.

The plan depends on how much speech there is and on the GPU:

* **Batch size** fits the free VRAM, but a short dataset also caps it so an
  epoch still has about a dozen optimiser steps (a few large steps per epoch
  learn less than many smaller ones from the same audio).
* **Epochs** come from a budget of optimiser steps, not a fixed count: a short
  dataset gets more epochs, a long one fewer, for a similar amount of learning.
  The budget is an upper bound; early stopping usually ends sooner.
* **Crops per example**: each step trains on a random 0.32 s crop of a 3.7 s
  window. Under 10 minutes of speech, every window gives several crops per
  epoch, so each epoch sees more of the material.
* **Validation**: a few windows spread over the recording are held out and
  scored through the inference path (what realtime conversion plays), so the
  exported voice is the epoch that generalises best, not the last one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

TARGET_STEPS = 7000
MIN_EPOCHS = 150
MAX_AUTO_EPOCHS = 1000
MIN_STEPS_PER_EPOCH = 12
MAX_AUTO_BATCH = 8
SHORT_DATASET_SECONDS = 600
VALIDATION_SHARE = 0.08
VALIDATION_RANGE = (4, 16)
MIN_WINDOWS_FOR_VALIDATION = 20


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    batch_size: int
    epochs: int
    crops_per_example: int
    eval_every: int
    patience_epochs: int
    validation_windows: int

    def describe(self) -> dict:
        return asdict(self)


def validation_count(windows: int) -> int:
    if windows < MIN_WINDOWS_FOR_VALIDATION:
        return 0
    low, high = VALIDATION_RANGE
    return int(min(high, max(low, round(windows * VALIDATION_SHARE))))


def plan_training(windows: int, dataset_seconds: float, vram_batch: int, *,
                  batch_size: int | None = None, epochs: int | None = None) -> TrainingPlan:
    """Plan for ``windows`` training windows; explicit values override the automatic ones."""
    held_out = validation_count(windows)
    train_windows = max(1, windows - held_out)
    crops = 2 if dataset_seconds < SHORT_DATASET_SECONDS else 1
    if batch_size is None:
        batch_size = min(vram_batch, MAX_AUTO_BATCH, max(2, train_windows * crops // MIN_STEPS_PER_EPOCH))
    batch_size = max(1, min(batch_size, train_windows * crops))
    steps_per_epoch = math.ceil(train_windows * crops / batch_size)
    if epochs is None:
        epochs = int(min(MAX_AUTO_EPOCHS, max(MIN_EPOCHS, math.ceil(TARGET_STEPS / steps_per_epoch))))
    eval_every = 5 if epochs >= 100 else 2
    patience = max(40, round(0.15 * epochs))
    return TrainingPlan(batch_size, epochs, crops, eval_every, patience, held_out)



@dataclass(slots=True)
class ValidationTracker:
    """Best epoch and early stopping from validation scores (lower is better).

    The best epoch is picked on the raw score. Early stopping watches a
    smoothed score, so one lucky or unlucky evaluation neither ends training
    nor keeps it going: it stops after ``patience_epochs`` without a real gain.
    """

    patience_epochs: int
    history: list[list[float]] = field(default_factory=list)  # [epoch, score, smoothed]
    best_score: float | None = None
    best_epoch: int = 0
    best_smoothed: float | None = None
    last_gain_epoch: int = 0
    stopped: bool = False

    MIN_RELATIVE_GAIN = 0.002
    SMOOTHING = 0.6

    @classmethod
    def restore(cls, status: dict, patience_epochs: int) -> ValidationTracker:
        tracker = cls(patience_epochs)
        for epoch, score, *_ in status.get("validation") or []:
            tracker.add(int(epoch), float(score))
        tracker.stopped = False
        return tracker

    def add(self, epoch: int, score: float) -> bool:
        """Record a score; True when it is the best so far (export this epoch)."""
        previous = self.history[-1][2] if self.history else None
        smoothed = score if previous is None else self.SMOOTHING * previous + (1 - self.SMOOTHING) * score
        self.history.append([epoch, round(score, 4), round(smoothed, 4)])
        if self.best_smoothed is None or smoothed < self.best_smoothed * (1 - self.MIN_RELATIVE_GAIN):
            self.best_smoothed, self.last_gain_epoch = smoothed, epoch
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        return improved

    def should_stop(self, epoch: int) -> bool:
        self.stopped = self.stopped or epoch - self.last_gain_epoch >= self.patience_epochs
        return self.stopped
