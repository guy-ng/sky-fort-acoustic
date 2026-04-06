"""Tests for trainer loss factory wiring (D-31).

Verifies that efficientat_trainer.py constructs its criterion via
build_loss_function() so that TrainingConfig.loss_function ("focal" by
default) actually takes effect at training time.
See .planning/debug/training-collapse-constant-output.md (PRIMARY-C).
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from acoustic.training.config import TrainingConfig
from acoustic.training.losses import FocalLoss, build_loss_function


def test_focal_selected_when_configured():
    cfg = TrainingConfig(loss_function="focal", focal_alpha=0.25, focal_gamma=2.0)
    loss = build_loss_function(
        loss_type=cfg.loss_function,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        bce_pos_weight=cfg.bce_pos_weight,
    )
    assert isinstance(loss, FocalLoss)
    assert loss.alpha == 0.25
    assert loss.gamma == 2.0


def test_bce_selected_when_configured():
    cfg = TrainingConfig(loss_function="bce", bce_pos_weight=1.0)
    loss = build_loss_function(
        loss_type=cfg.loss_function,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        bce_pos_weight=cfg.bce_pos_weight,
    )
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_bce_weighted_honors_pos_weight():
    cfg = TrainingConfig(loss_function="bce_weighted", bce_pos_weight=2.5)
    loss = build_loss_function(
        loss_type=cfg.loss_function,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        bce_pos_weight=cfg.bce_pos_weight,
    )
    assert isinstance(loss, nn.BCEWithLogitsLoss)
    assert loss.pos_weight is not None
    assert float(loss.pos_weight.item()) == 2.5
    # Callable smoke check
    out = loss(torch.zeros(4, 1), torch.zeros(4, 1))
    assert out is not None


def test_trainer_source_uses_build_loss_function():
    """Static guarantee: trainer source no longer hard-codes BCEWithLogitsLoss
    and instead calls build_loss_function()."""
    from acoustic.training import efficientat_trainer

    src = inspect.getsource(efficientat_trainer)
    assert "build_loss_function(" in src, (
        "efficientat_trainer must call build_loss_function()"
    )
    assert "nn.BCEWithLogitsLoss()" not in src, (
        "efficientat_trainer must not hard-code nn.BCEWithLogitsLoss()"
    )
