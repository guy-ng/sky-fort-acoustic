"""Unit tests for TrainingConfig and augmentation modules."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


class TestTrainingConfig:
    """Tests for TrainingConfig defaults and env-var loading."""

    def test_defaults(self):
        from acoustic.training.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.batch_size == 32
        assert cfg.max_epochs == 50
        assert cfg.patience == 5
        assert cfg.augmentation_enabled is True
        assert cfg.data_root == "audio-data/data/"
        assert cfg.checkpoint_path.endswith(".pt")

    def test_env_var_override(self, monkeypatch):
        from acoustic.training.config import TrainingConfig

        monkeypatch.setenv("ACOUSTIC_TRAINING_LEARNING_RATE", "0.01")
        monkeypatch.setenv("ACOUSTIC_TRAINING_BATCH_SIZE", "64")
        cfg = TrainingConfig()
        assert cfg.learning_rate == pytest.approx(0.01)
        assert cfg.batch_size == 64


class TestWaveformAugmentation:
    """Tests for WaveformAugmentation."""

    def test_output_shape_and_dtype(self):
        from acoustic.training.augmentation import WaveformAugmentation

        aug = WaveformAugmentation()
        audio = np.random.randn(8000).astype(np.float32)
        out = aug(audio)
        assert out.dtype == np.float32
        assert out.shape == audio.shape

    def test_adds_noise(self):
        from acoustic.training.augmentation import WaveformAugmentation

        aug = WaveformAugmentation()
        audio = np.random.randn(8000).astype(np.float32)
        out = aug(audio)
        assert not np.array_equal(out, audio), "Output should differ from input (noise added)"

    def test_gain_variation(self):
        from acoustic.training.augmentation import WaveformAugmentation

        aug = WaveformAugmentation()
        audio = np.random.randn(8000).astype(np.float32)
        magnitudes = set()
        for _ in range(20):
            out = aug(audio)
            magnitudes.add(round(float(np.max(np.abs(out))), 4))
        assert len(magnitudes) > 1, "Multiple calls should produce different magnitudes"


class TestSpecAugment:
    """Tests for SpecAugment."""

    def test_output_shape(self):
        from acoustic.training.augmentation import SpecAugment

        aug = SpecAugment()
        spec = torch.rand(1, 128, 64)
        out = aug(spec)
        assert out.shape == (1, 128, 64)

    def test_produces_zeros(self):
        from acoustic.training.augmentation import SpecAugment

        aug = SpecAugment(time_mask_param=20, freq_mask_param=8)
        spec = torch.ones(1, 128, 64)
        # Run multiple times to account for randomness
        has_zeros = False
        for _ in range(10):
            out = aug(spec)
            if (out == 0).any():
                has_zeros = True
                break
        assert has_zeros, "SpecAugment should produce at least some zero values"

    def test_no_masking_when_params_zero(self):
        from acoustic.training.augmentation import SpecAugment

        aug = SpecAugment(time_mask_param=0, freq_mask_param=0)
        spec = torch.rand(1, 128, 64)
        out = aug(spec)
        assert torch.allclose(out, spec), "Zero params should return input unchanged"
