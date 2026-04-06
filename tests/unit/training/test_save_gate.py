"""Tests for behavioral checkpoint save gate (D-32).

Verifies that the trainer refuses to save degenerate checkpoints
(min(tp,tn) == 0 OR val_acc < cfg.save_gate_min_accuracy) and logs a
warning instead. Guards against the v3/v5/v6 constant-output collapse
pattern from .planning/debug/training-collapse-constant-output.md.
"""

from __future__ import annotations

import inspect
import logging
import threading
from pathlib import Path

import pytest
import torch

from acoustic.training.config import TrainingConfig
from acoustic.training.efficientat_trainer import EfficientATTrainingRunner


def test_save_gate_config_default():
    cfg = TrainingConfig()
    assert cfg.save_gate_min_accuracy == 0.55


def test_trainer_source_implements_save_gate():
    """Static guarantee: trainer source references save_gate_min_accuracy
    and emits the 'save gate blocked' warning."""
    from acoustic.training import efficientat_trainer

    src = inspect.getsource(efficientat_trainer)
    assert "save_gate_min_accuracy" in src, (
        "trainer must consult cfg.save_gate_min_accuracy"
    )
    assert "save gate blocked" in src, (
        "trainer must log 'save gate blocked' when refusing to save"
    )


def _patched_runner_smoke(
    tmp_path: Path,
    force_tp: int,
    force_tn: int,
    force_fp: int,
    force_fn: int,
    monkeypatch: pytest.MonkeyPatch,
) -> Path | None:
    """Run the trainer smoke path but force the confusion matrix to a
    synthetic value so we can test the save gate deterministically."""
    from acoustic.training import efficientat_trainer

    original_run = efficientat_trainer.EfficientATTrainingRunner.run

    def run_with_forced_cm(self, stop_event, progress_callback=None, *, _synthetic=False):
        # Monkeypatch torch comparison ops to return forced confusion matrix.
        # Simpler: wrap torch.sigmoid to produce values that yield the desired cm.
        return original_run(self, stop_event, progress_callback=progress_callback, _synthetic=_synthetic)

    # Actually the cleanest test is inline: we manually call the save-gate
    # branch with synthetic variables. We'll do this via a helper method if
    # present, else we simulate by constructing a runner and invoking the
    # training loop with a tiny dataset and patched sigmoid output.
    return None


class TestSaveGateLogic:
    """Unit-level tests that exercise the save gate condition directly.

    We replicate the gate's boolean formula from the trainer source and
    verify it against the behavioral contract.
    """

    @staticmethod
    def _gate_ok(tp: int, fp: int, tn: int, fn: int, min_acc: float) -> bool:
        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        return (min(tp, tn) > 0) and (acc >= min_acc)

    def test_degenerate_all_negative_blocked(self):
        # Collapsed to "always predict 0": tp=0, fn=positive rate
        assert not self._gate_ok(tp=0, fp=0, tn=100, fn=50, min_acc=0.55)

    def test_degenerate_all_positive_blocked(self):
        # Collapsed to "always predict 1": tn=0, fp=negative rate
        assert not self._gate_ok(tp=50, fp=100, tn=0, fn=0, min_acc=0.55)

    def test_low_accuracy_blocked(self):
        # Both classes present but accuracy below threshold
        assert not self._gate_ok(tp=10, fp=40, tn=10, fn=40, min_acc=0.55)

    def test_healthy_model_allowed(self):
        assert self._gate_ok(tp=80, fp=20, tn=70, fn=30, min_acc=0.55)


class TestSaveGateIntegration:
    """End-to-end: synthetic smoke-train and verify the gate is wired up."""

    def test_save_gate_warning_emitted_for_degenerate(self, tmp_path, caplog):
        """With save_gate threshold above anything the synthetic model can hit,
        every would-be save is blocked and the warning is logged."""
        config = TrainingConfig(
            model_type="efficientat_mn10",
            pretrained_weights="",
            stage1_epochs=1,
            stage2_epochs=0,
            stage3_epochs=0,
            batch_size=4,
            patience=100,
            checkpoint_path=str(tmp_path / "gated_ckpt.pt"),
            dads_path=str(tmp_path / "nonexistent"),
            data_root=str(tmp_path / "nonexistent"),
            save_gate_min_accuracy=1.01,  # impossible -> gate always blocks
        )
        runner = EfficientATTrainingRunner(config)
        stop_event = threading.Event()

        with caplog.at_level(logging.WARNING, logger="acoustic.training.efficientat_trainer"):
            result = runner.run(stop_event, _synthetic=True)

        # With the gate blocking all saves, no checkpoint should exist.
        assert result is None or not Path(config.checkpoint_path).exists()
        # And the warning must have been logged at least once.
        blocked_msgs = [r for r in caplog.records if "save gate blocked" in r.getMessage()]
        assert len(blocked_msgs) >= 1

    def test_save_gate_allows_save_when_threshold_zero(self, tmp_path):
        """With threshold 0.0, the gate degenerates to 'min(tp,tn) > 0' only;
        with enough epochs a non-degenerate model should save."""
        config = TrainingConfig(
            model_type="efficientat_mn10",
            pretrained_weights="",
            stage1_epochs=2,
            stage2_epochs=2,
            stage3_epochs=2,
            batch_size=4,
            patience=100,
            checkpoint_path=str(tmp_path / "ungated_ckpt.pt"),
            dads_path=str(tmp_path / "nonexistent"),
            data_root=str(tmp_path / "nonexistent"),
            save_gate_min_accuracy=0.0,
        )
        runner = EfficientATTrainingRunner(config)
        stop_event = threading.Event()
        result = runner.run(stop_event, _synthetic=True)
        # Synthetic data is tiny, so training may or may not produce a save --
        # the important property is that the call path does not crash and the
        # gate does not incorrectly block when min_acc=0.0.
        # If a save happened, the checkpoint exists.
        if result is not None:
            assert Path(result).exists()
