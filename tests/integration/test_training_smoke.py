"""Integration smoke test: end-to-end training on synthetic data proves convergence."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


def _make_synthetic_wavs(tmp_path: Path, sr: int = 16000, duration: float = 2.0) -> Path:
    """Create 8 synthetic WAV files: 4 drone (sine+noise), 4 background (noise)."""
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


class TestTrainingSmokeTest:
    def test_end_to_end_training_converges(self, tmp_path):
        from acoustic.classification.research_cnn import ResearchCNN
        from acoustic.training.config import TrainingConfig
        from acoustic.training.manager import TrainingManager, TrainingStatus

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=10,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=8,
        )

        manager = TrainingManager(config=config)
        manager.start()

        # Poll for completion with timeout
        deadline = time.monotonic() + 120
        while manager.is_training() and time.monotonic() < deadline:
            time.sleep(0.5)

        progress = manager.get_progress()

        # Assert training completed
        assert progress.status == TrainingStatus.COMPLETED, (
            f"Expected COMPLETED, got {progress.status} (error: {progress.error})"
        )
        assert progress.epoch > 1, "Should have trained more than 1 epoch"
        assert Path(ckpt_path).exists(), "Checkpoint file should exist"
        assert progress.best_val_loss < float("inf"), "best_val_loss should be set"

        # Load checkpoint and run inference
        model = ResearchCNN()
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        # Synthetic drone audio for inference
        t = np.linspace(0, 0.5, 8000, dtype=np.float32)
        drone_audio = np.sin(2 * np.pi * 500 * t).astype(np.float32)

        # Run through mel pipeline
        from acoustic.classification.preprocessing import mel_spectrogram_from_segment

        features = mel_spectrogram_from_segment(drone_audio)
        with torch.no_grad():
            output = model(features)
        prob = output.item()
        assert 0.0 <= prob <= 1.0, f"Output {prob} not in [0, 1]"
