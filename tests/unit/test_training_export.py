"""Unit tests for TorchScript export, confusion matrix, and thread limits."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

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
                t = np.linspace(0, duration, n_samples, dtype=np.float32)
                audio = np.sin(2 * np.pi * 500 * t).astype(np.float32) + 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            else:
                audio = 0.1 * rng.standard_normal(n_samples).astype(np.float32)
            sf.write(label_dir / f"sample_{i}.wav", audio, sr)

    return tmp_path


# ---------------------------------------------------------------------------
# TorchScript export tests
# ---------------------------------------------------------------------------
class TestTorchScriptExport:
    def test_torchscript_export_created(self, tmp_path):
        """After TrainingRunner.run(), a .jit file exists alongside the checkpoint."""
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
        jit_path = Path(ckpt_path + ".jit")
        assert jit_path.exists(), f"TorchScript file not found at {jit_path}"

    def test_torchscript_loadable(self, tmp_path):
        """The .jit file loads with torch.jit.load() and produces correct shape and range."""
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

        jit_path = Path(ckpt_path + ".jit")
        scripted_model = torch.jit.load(str(jit_path))
        x = torch.randn(1, 1, 128, 64)
        output = scripted_model(x)
        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
        prob = output.item()
        assert 0.0 <= prob <= 1.0, f"Output {prob} not in [0, 1]"

    def test_no_export_on_cancel(self, tmp_path):
        """If stop_event is set before run(), no .jit file is created."""
        from acoustic.training.config import TrainingConfig
        from acoustic.training.trainer import TrainingRunner

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=100,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=100,
        )
        stop_event = threading.Event()
        stop_event.set()  # Cancel immediately

        runner = TrainingRunner(config)
        runner.run(stop_event=stop_event)

        jit_path = Path(ckpt_path + ".jit")
        assert not jit_path.exists(), "TorchScript file should not exist when training is cancelled"


# ---------------------------------------------------------------------------
# Confusion matrix tests
# ---------------------------------------------------------------------------
class TestConfusionMatrix:
    def test_confusion_matrix_in_progress(self, tmp_path):
        """Progress callback includes tp, fp, tn, fn keys on each epoch."""
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

        updates: list[dict] = []
        runner = TrainingRunner(config)
        runner.run(stop_event=threading.Event(), progress_callback=updates.append)

        assert len(updates) > 0, "No progress updates received"
        last = updates[-1]
        for key in ("tp", "fp", "tn", "fn"):
            assert key in last, f"Missing key '{key}' in progress update"
            assert isinstance(last[key], int), f"'{key}' should be int, got {type(last[key])}"
            assert last[key] >= 0, f"'{key}' should be >= 0"

        # Total confusion matrix should equal total validation samples
        total_cm = last["tp"] + last["fp"] + last["tn"] + last["fn"]
        assert total_cm > 0, "Confusion matrix total should be > 0"


# ---------------------------------------------------------------------------
# Thread limits tests
# ---------------------------------------------------------------------------
class TestThreadLimits:
    def test_thread_limits_applied(self, tmp_path):
        """TrainingManager._run() calls torch.set_num_threads(2) and torch.set_num_interop_threads(1)."""
        from acoustic.training.config import TrainingConfig
        from acoustic.training.manager import TrainingManager

        data_dir = _make_synthetic_wavs(tmp_path / "data")
        ckpt_path = str(tmp_path / "model.pt")

        config = TrainingConfig(
            data_root=str(data_dir),
            max_epochs=2,
            batch_size=2,
            checkpoint_path=ckpt_path,
            augmentation_enabled=False,
            patience=10,
        )

        num_threads_calls: list[int] = []
        interop_threads_calls: list[int] = []

        with patch("acoustic.training.manager.torch") as mock_torch:
            # Let real torch work for training but capture thread calls
            import torch as real_torch
            mock_torch.set_num_threads = lambda n: num_threads_calls.append(n)
            mock_torch.set_num_interop_threads = lambda n: interop_threads_calls.append(n)

            # We need the real TrainingRunner to work, so only mock the thread limit calls
            # Use a direct approach: monkeypatch the _run method indirectly
            pass

        # Better approach: use monkeypatch to track calls
        num_threads_calls.clear()
        interop_threads_calls.clear()

        original_set_num_threads = torch.set_num_threads
        original_set_num_interop_threads = torch.set_num_interop_threads

        def tracking_set_num_threads(n):
            num_threads_calls.append(n)
            original_set_num_threads(n)

        def tracking_set_interop_threads(n):
            interop_threads_calls.append(n)
            original_set_num_interop_threads(n)

        manager = TrainingManager(config=config)

        with patch("acoustic.training.manager.torch.set_num_threads", side_effect=tracking_set_num_threads), \
             patch("acoustic.training.manager.torch.set_num_interop_threads", side_effect=tracking_set_interop_threads):
            manager.start()
            # Wait for training to complete
            thread = manager._thread
            if thread is not None:
                thread.join(timeout=60)

        assert 2 in num_threads_calls, f"torch.set_num_threads(2) not called. Calls: {num_threads_calls}"
        # set_num_interop_threads may fail at runtime (can only be called once),
        # but the code must attempt to call it with value 1
        assert 1 in interop_threads_calls, f"torch.set_num_interop_threads(1) not called. Calls: {interop_threads_calls}"
