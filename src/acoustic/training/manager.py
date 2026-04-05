"""Training manager: background thread lifecycle, progress state, concurrency guard.

Runs TrainingRunner in a daemon background thread with os.nice(10) priority
reduction. Thread-safe progress state readable from any thread.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum

import torch

from acoustic.classification.config import MelConfig
from acoustic.training.config import TrainingConfig
from acoustic.training.trainer import TrainingRunner

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Status of the training pipeline."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TrainingProgress:
    """Snapshot of training progress, readable from any thread."""

    status: TrainingStatus = TrainingStatus.IDLE
    model_name: str | None = None
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0           # Current batch within epoch
    total_batches: int = 0   # Total batches per epoch
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    best_val_loss: float = float("inf")
    error: str | None = None
    tp: int = 0   # True positives (confusion matrix)
    fp: int = 0   # False positives
    tn: int = 0   # True negatives
    fn: int = 0   # False negatives
    cache_loaded: int = 0   # Audio samples cached in memory
    cache_total: int = 0    # Total audio samples in dataset
    stage: int = 0          # Current training stage (0=N/A, 1-3 for EfficientAT)


class TrainingManager:
    """Manages training lifecycle in a background daemon thread.

    Enforces single concurrent training run. Progress is thread-safe via Lock.
    Training can be cancelled mid-run via stop event.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        mel_config: MelConfig | None = None,
    ) -> None:
        self._config = config or TrainingConfig()
        self._mel_config = mel_config or MelConfig()
        self._lock = threading.Lock()
        self._progress = TrainingProgress()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _compute_total_epochs(cfg: TrainingConfig) -> int:
        """Return total training epochs based on model type."""
        if cfg.model_type == "efficientat_mn10":
            return cfg.stage1_epochs + cfg.stage2_epochs + cfg.stage3_epochs
        return cfg.max_epochs

    def get_progress(self) -> TrainingProgress:
        """Return a copy of the current training progress (thread-safe)."""
        with self._lock:
            return dataclasses.replace(self._progress)

    def is_training(self) -> bool:
        """Return True if the training thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, config: TrainingConfig | None = None, model_name: str | None = None) -> None:
        """Launch training in a background daemon thread.

        Args:
            config: Optional override config. Uses constructor config if None.
            model_name: Name for this training run, persisted in progress.

        Raises:
            RuntimeError: If training is already in progress.
        """
        if self._thread is not None and self._thread.is_alive():
            msg = "Training already in progress"
            raise RuntimeError(msg)

        cfg = config or self._config

        # Reset state
        self._stop_event.clear()
        with self._lock:
            self._progress = TrainingProgress(
                status=TrainingStatus.RUNNING,
                model_name=model_name,
                total_epochs=self._compute_total_epochs(cfg),
            )

        self._thread = threading.Thread(
            target=self._run,
            args=(cfg,),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Signal the training thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30)
        with self._lock:
            if self._progress.status == TrainingStatus.RUNNING:
                self._progress.status = TrainingStatus.CANCELLED

    def _run(self, config: TrainingConfig) -> None:
        """Training thread entry point. Runs with lowered CPU priority."""
        try:
            os.nice(10)
        except OSError:
            logger.warning("Could not set os.nice(10), continuing at normal priority")

        # Cap PyTorch intra-op thread pool to avoid starving live detection (per D-12)
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can only be set once before parallel work starts; ignore if already set
            pass

        try:
            if config.model_type in ("efficientat_mn10", "efficientat_mn05"):
                from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

                runner = EfficientATTrainingRunner(config)
            else:
                runner = TrainingRunner(config, self._mel_config)
            runner.run(self._stop_event, progress_callback=self._on_progress)

            with self._lock:
                if self._stop_event.is_set():
                    self._progress.status = TrainingStatus.CANCELLED
                else:
                    self._progress.status = TrainingStatus.COMPLETED

        except Exception as exc:
            logger.exception("Training failed: %s", exc)
            with self._lock:
                self._progress.status = TrainingStatus.FAILED
                self._progress.error = str(exc)

    def _on_progress(self, update: dict) -> None:
        """Callback invoked by TrainingRunner each epoch with metrics."""
        with self._lock:
            self._progress.epoch = update.get("epoch", self._progress.epoch)
            self._progress.total_epochs = update.get("total_epochs", self._progress.total_epochs)
            self._progress.batch = update.get("batch", self._progress.batch)
            self._progress.total_batches = update.get("total_batches", self._progress.total_batches)
            self._progress.train_loss = update.get("train_loss", self._progress.train_loss)
            self._progress.val_loss = update.get("val_loss", self._progress.val_loss)
            self._progress.val_acc = update.get("val_acc", self._progress.val_acc)
            self._progress.best_val_loss = update.get("best_val_loss", self._progress.best_val_loss)
            self._progress.tp = update.get("tp", self._progress.tp)
            self._progress.fp = update.get("fp", self._progress.fp)
            self._progress.tn = update.get("tn", self._progress.tn)
            self._progress.fn = update.get("fn", self._progress.fn)
            self._progress.cache_loaded = update.get("cache_loaded", self._progress.cache_loaded)
            self._progress.cache_total = update.get("cache_total", self._progress.cache_total)
            self._progress.stage = update.get("stage", self._progress.stage)
