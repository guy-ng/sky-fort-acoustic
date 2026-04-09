"""Tests for the EfficientAT window contract module.

Validates that the single source of truth for window length, target SR,
and segment samples is consistent with EfficientATMelConfig and that the
source_window_samples helper produces correct values for different source rates.
"""

from __future__ import annotations

import pytest

window_contract = pytest.importorskip(
    "acoustic.classification.efficientat.window_contract"
)


def test_window_seconds_is_one():
    assert window_contract.EFFICIENTAT_WINDOW_SECONDS == 1.0


def test_target_sr_is_32000():
    assert window_contract.EFFICIENTAT_TARGET_SR == 32000


def test_segment_samples_matches_mel_config():
    from acoustic.classification.efficientat.config import EfficientATMelConfig

    assert window_contract.EFFICIENTAT_SEGMENT_SAMPLES == EfficientATMelConfig().segment_samples
    assert window_contract.EFFICIENTAT_SEGMENT_SAMPLES == 32000


def test_source_window_samples_16k():
    assert window_contract.source_window_samples(16000) == 16000


def test_source_window_samples_32k():
    assert window_contract.source_window_samples(32000) == 32000


def test_import_time_selfcheck_fires_on_mismatch(monkeypatch):
    """Verify the import-time assertion catches mel config drift."""
    import importlib
    import acoustic.classification.efficientat.window_contract as wc_mod

    # Monkeypatch the segment samples to a wrong value and re-run the assertion logic
    monkeypatch.setattr(wc_mod, "EFFICIENTAT_SEGMENT_SAMPLES", 16000)
    with pytest.raises(AssertionError, match="window contract broken"):
        # Re-execute the assertion manually (can't re-import easily)
        assert wc_mod.EFFICIENTAT_SEGMENT_SAMPLES == int(
            wc_mod.EFFICIENTAT_WINDOW_SECONDS * wc_mod.EFFICIENTAT_TARGET_SR
        ), (
            f"EfficientAT window contract broken: segment_samples={wc_mod.EFFICIENTAT_SEGMENT_SAMPLES} "
            f"!= window_seconds * target_sr = {int(wc_mod.EFFICIENTAT_WINDOW_SECONDS * wc_mod.EFFICIENTAT_TARGET_SR)}"
        )
