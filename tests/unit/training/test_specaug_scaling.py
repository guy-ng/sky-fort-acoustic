"""Tests for SpecAugment mask scaling (D-30).

Verifies that SpecAugment time/freq mask params are driven by TrainingConfig
fields `specaug_freq_mask` / `specaug_time_mask` and that defaults are
proportional to the actual ~100-frame EfficientAT input, not the legacy
AudioSet values (48/192) that caused training collapse.

See .planning/debug/training-collapse-constant-output.md (PRIMARY-A).
"""

from __future__ import annotations

from acoustic.classification.efficientat.config import EfficientATMelConfig
from acoustic.classification.efficientat.preprocess import AugmentMelSTFT
from acoustic.training.config import TrainingConfig


def test_training_config_has_specaug_fields_with_safe_defaults():
    cfg = TrainingConfig()
    assert cfg.specaug_freq_mask == 8
    assert cfg.specaug_time_mask == 10


def test_specaug_time_mask_less_than_input_dim():
    """timem must be strictly less than input_dim_t // 5 (~20 for 100 frames)."""
    cfg = TrainingConfig()
    input_dim_t = EfficientATMelConfig().input_dim_t
    assert cfg.specaug_time_mask < input_dim_t // 5, (
        f"specaug_time_mask={cfg.specaug_time_mask} must be << input_dim_t={input_dim_t}"
    )


def test_augment_melstft_constructed_with_config_values():
    """AugmentMelSTFT built from config exposes the configured masks."""
    cfg = TrainingConfig()
    mel_cfg = EfficientATMelConfig()
    mel = AugmentMelSTFT(
        n_mels=mel_cfg.n_mels,
        sr=mel_cfg.sample_rate,
        win_length=mel_cfg.win_length,
        hopsize=mel_cfg.hop_size,
        n_fft=mel_cfg.n_fft,
        freqm=cfg.specaug_freq_mask,
        timem=cfg.specaug_time_mask,
    )
    # AugmentMelSTFT stores freqm/timem as attributes (per preprocess.py)
    assert mel.freqm == 8
    assert mel.timem == 10
