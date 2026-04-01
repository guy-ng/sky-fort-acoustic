"""Unit tests for EarlyStopping and TrainingRunner checkpoint functionality."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Helpers: create synthetic WAV files for training
# ---------------------------------------------------------------------------
def _make_synthetic_wavs(tmp_path: Path, sr: int = 16000, duration: float = 1.0) -> Path:
    """Create a data directory with drone/ and background/ subdirectories, 4 files each."""
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)

    for label in ("drone", "background"):
        label_dir = tmp_path / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            if label == "drone":
                # Sine wave at 500Hz + noise
                t = np.linspace(0, duration, n_samples, dtype=np.float32)
                audio = np.sin(2 * np.pi * 500 * t).astype(np.float32) + 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            else:
                # Just noise
                audio = 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            sf.write(label_dir / f"sample_{i}.wav", audio, sr)

    return tmp_path


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    def test_first_call_returns_improved(self):
        from acoustic.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3)
        assert es.step(1.0) is True

    def test_lower_loss_returns_improved(self):
        from acoustic.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3)
        es.step(1.0)
        assert es.step(0.5) is True

    def test_higher_loss_returns_not_improved(self):
        from acoustic.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3)
        es.step(1.0)
        assert es.step(1.5) is False

    def test_should_stop_after_patience_exhausted(self):
        from acoustic.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3)
        es.step(1.0)  # improved
        es.step(2.0)  # not improved (counter=1)
        es.step(2.0)  # not improved (counter=2)
        es.step(2.0)  # not improved (counter=3 >= patience)
        assert es.should_stop is True

    def test_should_stop_false_within_patience(self):
        from acoustic.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(2.0)  # counter=1
        es.step(2.0)  # counter=2
        assert es.should_stop is False


# ---------------------------------------------------------------------------
# TrainingRunner checkpoint tests
# ---------------------------------------------------------------------------
class TestTrainingRunnerCheckpoint:
    def test_run_saves_checkpoint(self, tmp_path):
        from acoustic.training.config import TrainingConfig
        from acoustic.training.trainer import TrainingRunner

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=3,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=10,
        )
        runner = TrainingRunner(config)
        result = runner.run(stop_event=threading.Event())
        assert result is not None
        assert Path(ckpt_path).exists()

    def test_checkpoint_loadable_by_research_cnn(self, tmp_path):
        from acoustic.classification.research_cnn import ResearchCNN
        from acoustic.training.config import TrainingConfig
        from acoustic.training.trainer import TrainingRunner

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=3,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=10,
        )
        runner = TrainingRunner(config)
        runner.run(stop_event=threading.Event())

        # Load checkpoint into fresh model
        model = ResearchCNN()
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        # Verify forward pass shape
        x = torch.randn(1, 1, 128, 64)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1)

    def test_stop_event_cancels_training(self, tmp_path):
        from acoustic.training.config import TrainingConfig
        from acoustic.training.trainer import TrainingRunner

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=100,  # Many epochs, but stop event should prevent
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=100,
        )
        stop_event = threading.Event()
        stop_event.set()  # Set before run — training should exit within 1 epoch

        runner = TrainingRunner(config)
        start = time.monotonic()
        runner.run(stop_event=stop_event)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Training should have stopped quickly, took {elapsed:.1f}s"

    def test_loaded_model_output_range(self, tmp_path):
        """Verify loaded model outputs drone probability in [0, 1]."""
        from acoustic.classification.research_cnn import ResearchCNN
        from acoustic.training.config import TrainingConfig
        from acoustic.training.trainer import TrainingRunner

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=3,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=10,
        )
        runner = TrainingRunner(config)
        runner.run(stop_event=threading.Event())

        model = ResearchCNN()
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        x = torch.randn(1, 1, 128, 64)
        with torch.no_grad():
            output = model(x)
        prob = output.item()
        assert 0.0 <= prob <= 1.0, f"Output {prob} not in [0, 1]"
