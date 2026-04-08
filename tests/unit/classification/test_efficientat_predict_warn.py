"""Phase 22 Wave 0: EfficientATClassifier WARN on length mismatch. Green after Plan 03."""
import logging
import pytest
import torch

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="Phase 22 Plan 03 adds WARN in EfficientATClassifier.predict",
)


def test_predict_warns_on_length_mismatch(caplog):
    from acoustic.classification.efficientat.classifier import EfficientATClassifier
    from acoustic.classification.efficientat.config import EfficientATMelConfig
    # Build a minimal classifier with a mock model + real mel config
    clf = EfficientATClassifier.__new__(EfficientATClassifier)
    # Use real constructor when Plan 03 lands -- stub out model for Wave 0
    pytest.skip("classifier construction API depends on Plan 03 changes")


def test_predict_does_not_raise_on_length_mismatch():
    pytest.skip("same as above -- Plan 03 will flesh this out")
