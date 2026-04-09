"""Phase 22 Wave 2 (Plan 03): EfficientATClassifier WARN on length mismatch.

Guardrail tests for REQ-22-W3. The v7 regression shipped 0.5 s windows through
to production undetected because the classifier silently ran a shape-agnostic
model on out-of-domain inputs. Plan 03 adds a one-shot WARN that makes drift
operator-visible without killing detection.

Contract:
    - Matching length (32000 samples @ 32 kHz) → silent, returns a float.
    - Mismatched length → ONE WARN log at WARNING level, still returns a float,
      NEVER raises.
    - Second mismatched call does NOT log a second WARN (one-shot per instance).
"""
from __future__ import annotations

import logging

import torch

from acoustic.classification.efficientat.classifier import EfficientATClassifier
from acoustic.classification.efficientat.model import get_model


def _make_clf() -> EfficientATClassifier:
    """Build a minimal EfficientATClassifier with a fresh binary head.

    Uses the same ``get_model`` path as ``tests/unit/test_efficientat.py``.
    This does NOT load a trained checkpoint — we only care about the
    predict() code path (length check + logging), not the model output.
    """
    model = get_model(
        num_classes=1,
        width_mult=1.0,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=100,
    )
    return EfficientATClassifier(model)


def test_predict_silent_on_matching_length(caplog) -> None:
    """32000-sample input → no WARN, returns float."""
    clf = _make_clf()
    with caplog.at_level(
        logging.WARNING,
        logger="acoustic.classification.efficientat.classifier",
    ):
        out = clf.predict(torch.zeros(32000))
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0
    mismatch_records = [
        r for r in caplog.records if "v7 regression signature" in r.message
    ]
    assert mismatch_records == [], (
        "matching-length input must NOT trigger the mismatch WARN"
    )


def test_predict_warns_on_length_mismatch(caplog) -> None:
    """Half-length input (16000 samples) → WARN logged, returns float."""
    clf = _make_clf()
    with caplog.at_level(
        logging.WARNING,
        logger="acoustic.classification.efficientat.classifier",
    ):
        out = clf.predict(torch.zeros(16000))
    assert isinstance(out, float)
    records = [
        r for r in caplog.records if "v7 regression signature" in r.message
    ]
    assert len(records) == 1, (
        f"expected exactly 1 mismatch WARN, got {len(records)}"
    )
    msg = records[0].message
    # Assert the WARN carries enough context for an operator to diagnose
    assert "16000" in msg
    assert "32000" in msg
    assert records[0].levelno == logging.WARNING


def test_predict_warns_only_once_per_instance(caplog) -> None:
    """Two mismatched calls → one WARN (one-shot flag)."""
    clf = _make_clf()
    with caplog.at_level(
        logging.WARNING,
        logger="acoustic.classification.efficientat.classifier",
    ):
        clf.predict(torch.zeros(16000))
        clf.predict(torch.zeros(16000))
        clf.predict(torch.zeros(8000))  # different bad length — still no repeat
    records = [
        r for r in caplog.records if "v7 regression signature" in r.message
    ]
    assert len(records) == 1, (
        f"expected 1 WARN across 3 mismatched calls (one-shot per instance), "
        f"got {len(records)}"
    )


def test_predict_does_not_raise_on_mismatch() -> None:
    """Mismatch is a WARN, never a raise — operational pipeline must keep running."""
    clf = _make_clf()
    # Must not raise
    out1 = clf.predict(torch.zeros(16000))
    out2 = clf.predict(torch.zeros(8000))
    out3 = clf.predict(torch.zeros(24000))
    assert all(isinstance(o, float) for o in (out1, out2, out3))


def test_predict_separate_instances_each_get_one_warn(caplog) -> None:
    """One-shot flag is PER INSTANCE — two classifiers can each log once."""
    clf_a = _make_clf()
    clf_b = _make_clf()
    with caplog.at_level(
        logging.WARNING,
        logger="acoustic.classification.efficientat.classifier",
    ):
        clf_a.predict(torch.zeros(16000))
        clf_b.predict(torch.zeros(16000))
        clf_a.predict(torch.zeros(16000))  # suppressed
        clf_b.predict(torch.zeros(16000))  # suppressed
    records = [
        r for r in caplog.records if "v7 regression signature" in r.message
    ]
    assert len(records) == 2, (
        f"each classifier instance should log once; got {len(records)}"
    )
