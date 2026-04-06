"""Tests for narrowed Stage 1 unfreezing scope (D-33).

_setup_stage1 must unfreeze ONLY the final Linear(1280, 1) binary head,
not the preceding pretrained Linear(1280, 1280) classifier layer. This
prevents Adam@1e-3 from destroying the pretrained head features during
early training on masked inputs.

See .planning/debug/training-collapse-constant-output.md (CONTRIBUTING-F).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from acoustic.classification.efficientat.model import get_model
from acoustic.training.efficientat_trainer import EfficientATTrainingRunner


@pytest.fixture()
def binary_model():
    model = get_model(
        num_classes=527,
        width_mult=1.0,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=10,
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


class TestStage1NarrowUnfreeze:
    """Stage 1 must train ONLY the new final binary head."""

    def test_final_head_trainable(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)

        final_head = binary_model.classifier[-1]
        assert isinstance(final_head, nn.Linear)
        assert final_head.out_features == 1
        for p in final_head.parameters():
            assert p.requires_grad, "final binary head must be trainable in stage 1"

    def test_preceding_classifier_linear_frozen(self, binary_model):
        """The Linear(1280, 1280) layer earlier in the MLP head must stay frozen."""
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)

        prior_linears = [
            m for m in binary_model.classifier if isinstance(m, nn.Linear)
        ][:-1]
        assert len(prior_linears) >= 1, "expected at least one prior Linear in MLP head"
        for layer in prior_linears:
            for p in layer.parameters():
                assert not p.requires_grad, (
                    "Stage 1 must NOT unfreeze the pretrained Linear(1280, 1280) "
                    "(D-33): it causes head collapse under Adam@1e-3 + masked inputs."
                )

    def test_backbone_fully_frozen(self, binary_model):
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)

        for name, p in binary_model.named_parameters():
            if not name.startswith("classifier"):
                assert not p.requires_grad, (
                    f"backbone param {name} must be frozen in stage 1"
                )

    def test_stage2_still_unfreezes_full_classifier(self, binary_model):
        """Stage 2 must continue to unfreeze the whole classifier (+ last 3 blocks)."""
        runner = EfficientATTrainingRunner.__new__(EfficientATTrainingRunner)
        runner._setup_stage1(binary_model)
        runner._setup_stage2(binary_model)

        for name, p in binary_model.classifier.named_parameters():
            assert p.requires_grad, (
                f"stage 2 must leave classifier param {name} trainable"
            )
        for name, p in binary_model.features[-3:].named_parameters():
            assert p.requires_grad, (
                f"stage 2 must unfreeze features[-3:] param {name}"
            )
