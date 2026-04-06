"""Tests for EfficientAT three-stage transfer learning trainer.

Validates freeze/unfreeze logic, cosine schedule, and training loop.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from acoustic.classification.efficientat.model import get_model
from acoustic.training.config import TrainingConfig
from acoustic.training.efficientat_trainer import EfficientATTrainingRunner


@pytest.fixture()
def small_model():
    """Create a small mn10 model for fast tests (input_dim_t=10)."""
    model = get_model(num_classes=527, width_mult=1.0, head_type="mlp",
                      input_dim_f=128, input_dim_t=10)
    return model


@pytest.fixture()
def binary_model(small_model):
    """Small model with binary head (1 output)."""
    in_features = small_model.classifier[-1].in_features
    small_model.classifier[-1] = torch.nn.Linear(in_features, 1)
    return small_model


class TestStage1Freeze:
    """After Stage 1 setup, only classifier params have requires_grad=True."""

    def test_stage1_freeze(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)

        # All features frozen
        for name, p in binary_model.features.named_parameters():
            assert not p.requires_grad, f"features param {name} should be frozen in stage 1"

        # Classifier unfrozen
        for name, p in binary_model.classifier.named_parameters():
            assert p.requires_grad, f"classifier param {name} should be trainable in stage 1"


class TestStage2Unfreeze:
    """After Stage 2, last 3 blocks + classifier are trainable."""

    def test_stage2_unfreeze(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        # First do stage 1 freeze, then stage 2 partial unfreeze
        runner._setup_stage1(binary_model)
        runner._setup_stage2(binary_model)

        # Earlier features still frozen
        for name, p in binary_model.features[:-3].named_parameters():
            assert not p.requires_grad, f"early features param {name} should be frozen in stage 2"

        # Last 3 blocks unfrozen
        for name, p in binary_model.features[-3:].named_parameters():
            assert p.requires_grad, f"late features param {name} should be trainable in stage 2"

        # Classifier still unfrozen
        for name, p in binary_model.classifier.named_parameters():
            assert p.requires_grad, f"classifier param {name} should be trainable in stage 2"


class TestStage3Unfreeze:
    """After Stage 3, ALL model parameters are trainable."""

    def test_stage3_unfreeze(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)
        runner._setup_stage2(binary_model)
        runner._setup_stage3(binary_model)

        for name, p in binary_model.named_parameters():
            assert p.requires_grad, f"param {name} should be trainable in stage 3"


class TestCosineSchedule:
    """CosineAnnealingLR is used and decays LR toward 0."""

    def test_cosine_schedule(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage2(binary_model)

        trainable = [p for p in binary_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == pytest.approx(1e-4)

        # Step through T_max epochs
        for _ in range(10):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        # After T_max steps, LR should be near 0 (eta_min default=0)
        assert final_lr < initial_lr * 0.01


class TestTrainingLoopSmoke:
    """Short training run completes and produces a checkpoint."""

    def test_training_loop_smoke(self, tmp_path):
        config = TrainingConfig(
            model_type="efficientat_mn10",
            pretrained_weights="",  # no pretrained weights for smoke test
            stage1_epochs=2,
            stage2_epochs=2,
            stage3_epochs=2,
            batch_size=4,
            patience=100,  # disable early stopping
            checkpoint_path=str(tmp_path / "test_ckpt.pt"),
            dads_path=str(tmp_path / "nonexistent"),  # force synthetic fallback
            data_root=str(tmp_path / "nonexistent"),
            # D-32: smoke test uses random noise as data, so any "model" is
            # degenerate by construction. Disable the save gate here so the
            # smoke test continues to exercise the full training loop end-
            # to-end. Real runs keep the default save_gate_min_accuracy=0.55.
            save_gate_min_accuracy=0.0,
        )

        runner = EfficientATTrainingRunner(config)
        stop_event = threading.Event()

        progress_updates: list[dict] = []
        def on_progress(info: dict) -> None:
            progress_updates.append(info)

        result = runner.run(stop_event, progress_callback=on_progress, _synthetic=True)

        assert result is not None
        assert Path(result).exists()

        # Check progress updates include stage info
        stages_seen = {u.get("stage") for u in progress_updates if "stage" in u}
        assert stages_seen == {1, 2, 3}

        # Total epochs should be 6 (2+2+2)
        epoch_reports = [u for u in progress_updates if "val_loss" in u]
        assert len(epoch_reports) == 6
