"""Unit tests for FocalLoss module and build_loss_function factory."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestFocalLoss:
    """Tests for FocalLoss nn.Module."""

    def test_focal_loss_output_shape(self):
        from acoustic.training.losses import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0, "FocalLoss should return a scalar tensor"

    def test_focal_matches_bce_at_gamma_zero(self):
        from acoustic.training.losses import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()

        focal = FocalLoss(alpha=-1, gamma=0.0)
        bce = nn.BCEWithLogitsLoss()

        focal_val = focal(logits, targets)
        bce_val = bce(logits, targets)
        assert torch.allclose(focal_val, bce_val, atol=1e-5), (
            f"FocalLoss(gamma=0, alpha=-1) should match BCE: focal={focal_val.item():.6f}, bce={bce_val.item():.6f}"
        )

    def test_focal_loss_gradient(self):
        from acoustic.training.losses import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert (logits.grad != 0).any(), "Gradients should be non-zero"

    def test_focal_hard_examples_higher_loss(self):
        from torchvision.ops import sigmoid_focal_loss

        # Hard example: target=1 but logit is very negative (confident wrong prediction)
        hard_logit = torch.tensor([-3.0])
        hard_target = torch.tensor([1.0])

        # Easy example: target=1 and logit is very positive (confident correct prediction)
        easy_logit = torch.tensor([3.0])
        easy_target = torch.tensor([1.0])

        hard_loss = sigmoid_focal_loss(hard_logit, hard_target, alpha=0.25, gamma=2.0, reduction="none")
        easy_loss = sigmoid_focal_loss(easy_logit, easy_target, alpha=0.25, gamma=2.0, reduction="none")

        assert hard_loss.item() > easy_loss.item(), (
            f"Hard examples should have higher loss: hard={hard_loss.item():.6f}, easy={easy_loss.item():.6f}"
        )


class TestBuildLossFunction:
    """Tests for build_loss_function factory."""

    def test_build_loss_function_focal(self):
        from acoustic.training.losses import FocalLoss, build_loss_function

        loss_fn = build_loss_function("focal")
        assert isinstance(loss_fn, FocalLoss)

    def test_build_loss_function_bce(self):
        from acoustic.training.losses import build_loss_function

        loss_fn = build_loss_function("bce")
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_build_loss_function_bce_weighted(self):
        from acoustic.training.losses import build_loss_function

        loss_fn = build_loss_function("bce_weighted", bce_pos_weight=2.0)
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
        assert loss_fn.pos_weight is not None
        assert loss_fn.pos_weight.item() == pytest.approx(2.0)

    def test_build_loss_function_unknown(self):
        from acoustic.training.losses import build_loss_function

        with pytest.raises(ValueError, match="unknown"):
            build_loss_function("unknown")


class TestTrainingConfigLossFields:
    """Tests for new loss-related fields in TrainingConfig."""

    def test_training_config_loss_fields(self):
        from acoustic.training.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.loss_function == "focal"
        assert cfg.focal_alpha == pytest.approx(0.25)
        assert cfg.focal_gamma == pytest.approx(2.0)
        assert cfg.bce_pos_weight == pytest.approx(1.0)

    def test_training_config_noise_fields(self):
        from acoustic.training.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.noise_augmentation_enabled is False
        assert cfg.noise_dirs == []
        assert cfg.noise_snr_range_low == pytest.approx(-10.0)
        assert cfg.noise_snr_range_high == pytest.approx(20.0)
        assert cfg.noise_probability == pytest.approx(0.5)

    def test_training_config_audiomentations_fields(self):
        from acoustic.training.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.use_audiomentations is True
        assert cfg.pitch_shift_semitones == pytest.approx(3.0)
        assert cfg.time_stretch_min == pytest.approx(0.85)
        assert cfg.time_stretch_max == pytest.approx(1.15)
        assert cfg.waveform_gain_db == pytest.approx(6.0)
        assert cfg.augmentation_probability == pytest.approx(0.5)
