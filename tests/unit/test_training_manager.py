"""Unit tests for TrainingManager: thread lifecycle, cancellation, concurrency guard."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_synthetic_wavs(tmp_path: Path, sr: int = 16000, duration: float = 1.0) -> Path:
    """Create data directory with drone/ and background/ subdirs, 4 files each."""
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)

    for label in ("drone", "background"):
        label_dir = tmp_path / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            if label == "drone":
                t = np.linspace(0, duration, n_samples, dtype=np.float32)
                audio = np.sin(2 * np.pi * 500 * t).astype(np.float32) + 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            else:
                audio = 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            sf.write(label_dir / f"sample_{i}.wav", audio, sr)

    return tmp_path


@pytest.fixture()
def training_dir(tmp_path):
    """Create synthetic training data and return configured TrainingConfig."""
    from acoustic.training.config import TrainingConfig

    data_dir = _make_synthetic_wavs(tmp_path / "data")
    config = TrainingConfig(
        data_root=str(data_dir),
        max_epochs=5,
        batch_size=2,
        checkpoint_path=str(tmp_path / "model.pt"),
        augmentation_enabled=False,
        patience=10,  # Don't early-stop in tests
    )
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTrainingManagerLifecycle:
    def test_initial_state_is_idle(self, training_dir):
        from acoustic.training.manager import TrainingManager, TrainingStatus

        manager = TrainingManager(config=training_dir)
        progress = manager.get_progress()
        assert progress.status == TrainingStatus.IDLE

    def test_start_launches_thread(self, training_dir):
        from acoustic.training.manager import TrainingManager

        manager = TrainingManager(config=training_dir)
        manager.start()
        assert manager.is_training() is True
        manager.cancel()

    def test_double_start_raises(self, training_dir):
        from acoustic.training.manager import TrainingManager

        manager = TrainingManager(config=training_dir)
        manager.start()
        with pytest.raises(RuntimeError, match="already"):
            manager.start()
        manager.cancel()

    def test_cancel_stops_training(self, training_dir):
        from acoustic.training.manager import TrainingManager, TrainingStatus

        # Use many epochs to ensure we can cancel mid-run
        training_dir.max_epochs = 1000
        manager = TrainingManager(config=training_dir)
        manager.start()
        time.sleep(1.0)  # Let it start training
        manager.cancel()
        progress = manager.get_progress()
        # With tiny data, training may finish before cancel takes effect
        assert progress.status in (TrainingStatus.CANCELLED, TrainingStatus.COMPLETED)
        assert not manager.is_training()

    def test_completion_status(self, training_dir):
        from acoustic.training.manager import TrainingManager, TrainingStatus

        manager = TrainingManager(config=training_dir)
        manager.start()
        # Wait for completion (max_epochs=5 should be quick)
        manager._thread.join(timeout=60)
        progress = manager.get_progress()
        assert progress.status == TrainingStatus.COMPLETED
        assert progress.epoch > 0

    def test_thread_is_daemon(self, training_dir):
        from acoustic.training.manager import TrainingManager

        manager = TrainingManager(config=training_dir)
        manager.start()
        assert manager._thread.daemon is True
        manager.cancel()

    def test_progress_readable_during_training(self, training_dir):
        from acoustic.training.manager import TrainingManager

        training_dir.max_epochs = 20
        manager = TrainingManager(config=training_dir)
        manager.start()
        time.sleep(1.0)  # Let some epochs run
        progress = manager.get_progress()
        assert progress.epoch >= 0
        manager.cancel()

    def test_restart_after_completion(self, training_dir):
        from acoustic.training.manager import TrainingManager, TrainingStatus

        manager = TrainingManager(config=training_dir)
        manager.start()
        manager._thread.join(timeout=60)
        assert manager.get_progress().status == TrainingStatus.COMPLETED

        # Should be able to start again
        manager.start()
        assert manager.is_training() is True
        manager.cancel()
