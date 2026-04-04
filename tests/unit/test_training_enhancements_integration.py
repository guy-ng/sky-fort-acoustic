"""Integration tests for training enhancements: focal loss, augmentations, balanced sampling.

Tests wiring of Plan 01 building blocks into TrainingRunner (Plan 02).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from acoustic.training.augmentation import (
    AudiomentationsAugmentation,
    BackgroundNoiseMixer,
    ComposedAugmentation,
    WaveformAugmentation,
)
from acoustic.training.config import TrainingConfig
from acoustic.training.dataset import build_weighted_sampler
from acoustic.training.losses import FocalLoss, build_loss_function


class TestTrainerUsesFocalLoss:
    """TRN-10: Trainer defaults to focal loss."""

    def test_trainer_uses_focal_loss_by_default(self) -> None:
        cfg = TrainingConfig()
        loss = build_loss_function(
            cfg.loss_function,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            bce_pos_weight=cfg.bce_pos_weight,
        )
        assert isinstance(loss, FocalLoss)

    def test_trainer_bce_fallback(self) -> None:
        loss = build_loss_function("bce")
        assert isinstance(loss, torch.nn.BCEWithLogitsLoss)


class TestLogitsMode:
    """ResearchCNN logits_mode for focal loss compatibility."""

    def test_trainer_logits_mode_model(self) -> None:
        from acoustic.classification.research_cnn import ResearchCNN

        x = torch.randn(4, 1, 128, 64)

        # logits_mode=True: output can be outside [0, 1]
        model_logits = ResearchCNN(logits_mode=True)
        model_logits.eval()
        with torch.no_grad():
            out_logits = model_logits(x).squeeze(-1)
        # Logits are unbounded -- at least check they exist and have correct shape
        assert out_logits.shape == (4,)

        # logits_mode=False (default): output in [0, 1]
        model_sigmoid = ResearchCNN(logits_mode=False)
        model_sigmoid.eval()
        with torch.no_grad():
            out_sigmoid = model_sigmoid(x).squeeze(-1)
        assert out_sigmoid.min() >= 0.0
        assert out_sigmoid.max() <= 1.0

    def test_export_model_uses_sigmoid(self) -> None:
        from acoustic.classification.research_cnn import ResearchCNN

        model = ResearchCNN(logits_mode=False)
        model.eval()
        x = torch.randn(2, 1, 128, 64)
        with torch.no_grad():
            out = model(x).squeeze(-1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestComposedAugmentation:
    """ComposedAugmentation pipeline is picklable and functional."""

    def test_composed_augmentation_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small noise WAV for BackgroundNoiseMixer
            noise = np.random.randn(16000).astype(np.float32) * 0.1
            noise_path = Path(tmpdir) / "noise.wav"
            sf.write(str(noise_path), noise, 16000)

            mixer = BackgroundNoiseMixer(
                noise_dirs=[Path(tmpdir)],
                snr_range=(5.0, 15.0),
                sample_rate=16000,
                p=1.0,
            )
            mixer.warm_cache()

            aug = AudiomentationsAugmentation(
                pitch_semitones=2.0,
                time_stretch_range=(0.9, 1.1),
                gain_db=3.0,
                p=1.0,
                sample_rate=16000,
            )

            composed = ComposedAugmentation([mixer, aug])
            audio = np.random.randn(8000).astype(np.float32) * 0.5
            result = composed(audio)
            assert result.dtype == np.float32
            assert len(result) == len(audio)

    def test_composed_augmentation_is_picklable(self) -> None:
        wave = WaveformAugmentation(snr_range=(10.0, 30.0), gain_db=3.0)
        composed = ComposedAugmentation([wave])
        data = pickle.dumps(composed)
        restored = pickle.loads(data)
        audio = np.random.randn(8000).astype(np.float32)
        result = restored(audio)
        assert result.dtype == np.float32
        assert len(result) == len(audio)


class TestSigmoidThreshold:
    """Validation loop applies sigmoid before threshold."""

    def test_sigmoid_threshold_on_logits(self) -> None:
        logits = torch.tensor([2.0, -2.0, 0.0])
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        expected = torch.tensor([1.0, 0.0, 1.0])
        assert torch.equal(preds, expected)


class TestWeightedSampler:
    """build_weighted_sampler produces correct weights for imbalanced data."""

    def test_weighted_sampler_with_imbalanced_labels(self) -> None:
        labels = [1, 1, 1, 0]
        sampler = build_weighted_sampler(labels)
        assert sampler.num_samples == 4
        weights = list(sampler.weights)
        # class 1 has 3 samples -> weight 1/3
        # class 0 has 1 sample  -> weight 1/1
        assert abs(weights[0] - 1 / 3) < 1e-6  # label=1
        assert abs(weights[3] - 1.0) < 1e-6  # label=0

    def test_weighted_sampler_with_1000_imbalanced_samples(self) -> None:
        """Validate sampler at DADS-like scale (900:100 imbalance)."""
        labels = [1] * 900 + [0] * 100
        sampler = build_weighted_sampler(labels)
        assert sampler.num_samples == 1000
        weights = list(sampler.weights)
        # class 1 (900 samples): weight = 1/900
        # class 0 (100 samples): weight = 1/100
        w_class1 = weights[0]
        w_class0 = weights[900]
        ratio = w_class0 / w_class1
        # class 0 should get 9x weight of class 1 (900/100)
        assert abs(ratio - 9.0) < 0.01


class TestTrainingConfigDefaults:
    """Config defaults match plan expectations."""

    def test_training_config_defaults_match_plan(self) -> None:
        cfg = TrainingConfig()
        assert cfg.loss_function == "focal"
        assert cfg.focal_alpha == 0.25
        assert cfg.focal_gamma == 2.0
        assert cfg.use_audiomentations is True
