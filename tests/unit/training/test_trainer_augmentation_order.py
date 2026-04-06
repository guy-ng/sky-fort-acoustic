"""RED stub locking the train/eval augmentation chain order (D-02, D-07, D-08).

Plan 20-04 wires WideGain → RoomIR → Audiomentations → BackgroundNoiseMixer
into ``EfficientATTrainingRunner._build_train_augmentation``. The eval chain
must explicitly EXCLUDE ``RoomIRAugmentation`` (eval data is already real or
synthetically clean — adding RIR at eval time would corrupt the metric).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _build_phase20_config(noise_dir: Path):
    from acoustic.training.config import TrainingConfig

    return TrainingConfig(
        wide_gain_db=40.0,
        rir_enabled=True,
        rir_probability=0.7,
        rir_pool_size=8,
        noise_augmentation_enabled=True,
        noise_dirs=[str(noise_dir)],
        use_audiomentations=True,
    )


def _build_runner(noise_dir: Path):
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = _build_phase20_config(noise_dir)
    return EfficientATTrainingRunner(config=cfg)


def test_train_chain_order(temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train chain must be exactly:
    [WideGainAugmentation, RoomIRAugmentation, AudiomentationsAugmentation, BackgroundNoiseMixer]
    """
    # Patch warm_cache so we don't need real noise loading.
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer,
        "warm_cache",
        lambda self: None,
        raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    train_aug = runner._build_train_augmentation()
    names = [type(a).__name__ for a in train_aug._augmentations]
    assert names == [
        "WideGainAugmentation",
        "RoomIRAugmentation",
        "AudiomentationsAugmentation",
        "BackgroundNoiseMixer",
    ]


def test_eval_chain_excludes_rir(temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval chain must NOT contain RoomIRAugmentation (D-08)."""
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer,
        "warm_cache",
        lambda self: None,
        raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    eval_aug = runner._build_eval_augmentation()
    names = [type(a).__name__ for a in eval_aug._augmentations]
    assert "RoomIRAugmentation" not in names
