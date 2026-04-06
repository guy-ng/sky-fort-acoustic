"""RED stubs for Phase 20 TrainingConfig fields (D-01, D-05..D-12).

These tests MUST currently fail with AttributeError because the new fields
do not yet exist on ``TrainingConfig``. Plan 20-02 adds them.
"""

from __future__ import annotations

import pytest

from acoustic.training.config import TrainingConfig


def test_wide_gain_db_default() -> None:
    """Default wide-gain range is ±40 dB (D-01)."""
    cfg = TrainingConfig()
    assert cfg.wide_gain_db == 40.0


def test_rir_enabled_default_false() -> None:
    """Room-IR augmentation must be opt-in (D-05)."""
    cfg = TrainingConfig()
    assert cfg.rir_enabled is False


def test_rir_probability_default() -> None:
    """RIR probability default is 0.7 (D-07)."""
    cfg = TrainingConfig()
    assert cfg.rir_probability == 0.7


def test_rir_pool_size_default() -> None:
    """Pre-built RIR pool size default is 500 (Research recommendation)."""
    cfg = TrainingConfig()
    assert cfg.rir_pool_size == 500


def test_window_overlap_ratio_default() -> None:
    """Sliding-window overlap is opt-in: default 0.0 (D-13/D-14)."""
    cfg = TrainingConfig()
    assert cfg.window_overlap_ratio == 0.0


def test_uma16_ambient_snr_range() -> None:
    """UMA-16 ambient noise SNR range default is (-5.0, 15.0) dB (D-11)."""
    cfg = TrainingConfig()
    assert cfg.uma16_ambient_snr_low == -5.0
    assert cfg.uma16_ambient_snr_high == 15.0


def test_uma16_pure_negative_ratio() -> None:
    """10% of negatives sourced as raw UMA-16 ambient with no drone (D-12)."""
    cfg = TrainingConfig()
    assert cfg.uma16_pure_negative_ratio == 0.10


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACOUSTIC_TRAINING_* env vars override field defaults."""
    monkeypatch.setenv("ACOUSTIC_TRAINING_RIR_ENABLED", "true")
    monkeypatch.setenv("ACOUSTIC_TRAINING_WIDE_GAIN_DB", "20.0")
    cfg = TrainingConfig()
    assert cfg.rir_enabled is True
    assert cfg.wide_gain_db == 20.0
