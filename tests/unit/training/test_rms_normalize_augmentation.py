"""Tests for the RmsNormalize augmentation and its wiring into the trainer chain.

Phase 20 D-34: RMS normalization must run as the LAST step of BOTH the train
and eval augmentation chains so the model sees identical amplitude
distributions at train and inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acoustic.training.augmentation import (
    ComposedAugmentation,
    RmsNormalize,
)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Direct class-level behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [0.003, 0.05, 0.5, 3.0])
def test_rms_normalize_snaps_to_target(amplitude: float) -> None:
    rng = np.random.default_rng(int(amplitude * 1000))
    n = 16000
    audio = rng.standard_normal(n).astype(np.float32)
    audio *= amplitude / _rms(audio)
    aug = RmsNormalize(target=0.1)
    out = aug(audio, sample_rate=16000)
    assert abs(_rms(out) - 0.1) < 1e-5


def test_rms_normalize_target_override() -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(8000).astype(np.float32) * 0.02
    aug = RmsNormalize(target=0.25)
    out = aug(audio, sample_rate=16000)
    assert abs(_rms(out) - 0.25) < 1e-5


def test_rms_normalize_silence_unchanged() -> None:
    audio = np.zeros(4000, dtype=np.float32)
    aug = RmsNormalize(target=0.1)
    out = aug(audio, sample_rate=16000)
    assert np.array_equal(out, audio)


def test_rms_normalize_inside_composed_augmentation() -> None:
    """RmsNormalize works when driven through ComposedAugmentation.__call__."""
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(8000).astype(np.float32) * 5.0
    composed = ComposedAugmentation([RmsNormalize(target=0.1)])
    out = composed(audio)
    assert abs(_rms(out) - 0.1) < 1e-5


def test_rms_normalize_is_picklable() -> None:
    """Must be pickle-safe for DataLoader num_workers>0 (Phase 20 D-34)."""
    import pickle

    aug = RmsNormalize(target=0.1)
    restored = pickle.loads(pickle.dumps(aug))
    assert isinstance(restored, RmsNormalize)
    assert restored.target == 0.1


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


def _build_runner(noise_dir: Path):
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig(
        wide_gain_db=40.0,
        rir_enabled=True,
        rir_probability=0.7,
        rir_pool_size=4,
        noise_augmentation_enabled=True,
        noise_dirs=[str(noise_dir)],
        use_audiomentations=True,
    )
    return EfficientATTrainingRunner(config=cfg)


def _build_runner_no_noise():
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig(
        wide_gain_db=0.0,
        rir_enabled=False,
        noise_augmentation_enabled=False,
        noise_dirs=[],
        use_audiomentations=False,
    )
    return EfficientATTrainingRunner(config=cfg)


def test_train_chain_ends_with_rms_normalize(
    temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer, "warm_cache", lambda self: None, raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    train_aug = runner._build_train_augmentation()
    last = train_aug._augmentations[-1]
    assert isinstance(last, RmsNormalize)
    assert last.target == 0.1


def test_eval_chain_ends_with_rms_normalize(
    temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer, "warm_cache", lambda self: None, raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    eval_aug = runner._build_eval_augmentation()
    last = eval_aug._augmentations[-1]
    assert isinstance(last, RmsNormalize)
    assert last.target == 0.1


def test_eval_chain_has_rms_normalize_even_without_noise() -> None:
    """When all augmentations are disabled, eval still runs RmsNormalize so
    val/test metrics match live inference amplitude."""
    runner = _build_runner_no_noise()
    eval_aug = runner._build_eval_augmentation()
    names = [type(a).__name__ for a in eval_aug._augmentations]
    assert names == ["RmsNormalize"]
