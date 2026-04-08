"""Phase 22 Wave 0: pins v7 regression vector. Green after Plan 02."""
import pytest

window_contract = pytest.importorskip(
    "acoustic.classification.efficientat.window_contract",
    reason="Phase 22 Plan 02 creates this module",
)


def test_window_seconds_is_one():
    assert window_contract.EFFICIENTAT_WINDOW_SECONDS == 1.0


def test_segment_samples_is_32000():
    assert window_contract.EFFICIENTAT_SEGMENT_SAMPLES == 32000


def test_target_sr_is_32000():
    assert window_contract.EFFICIENTAT_TARGET_SR == 32000


def test_source_window_samples_at_16k():
    assert window_contract.source_window_samples(16000) == 16000


def test_source_window_samples_at_32k():
    assert window_contract.source_window_samples(32000) == 32000


def test_contract_matches_mel_config():
    from acoustic.classification.efficientat.config import EfficientATMelConfig
    assert window_contract.EFFICIENTAT_SEGMENT_SAMPLES == EfficientATMelConfig().segment_samples
