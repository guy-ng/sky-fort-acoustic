"""EfficientAT train/serve window contract -- single source of truth.

Phase 22 remediation for the v7 regression (`.planning/debug/efficientat-v7-regression-vs-v6.md`).
Every consumer -- trainer, inference pipeline, dataset, classifier runtime check --
imports from here. If this file is wrong, EVERYTHING is wrong in the same way,
which is detectable by the integration tests. If call sites hard-code their own
value, the failure mode is silent (see v7 post-mortem).
"""

from __future__ import annotations

from .config import EfficientATMelConfig

EFFICIENTAT_WINDOW_SECONDS: float = 1.0
EFFICIENTAT_TARGET_SR: int = 32000
EFFICIENTAT_SEGMENT_SAMPLES: int = EfficientATMelConfig().segment_samples  # 32000


def source_window_samples(source_sr: int) -> int:
    """Window length in SOURCE-rate samples (pre-resample).

    Used by the training loop which operates on 16 kHz source audio;
    returns 16000 for source_sr=16000 so the post-resample tensor lands at
    the 32000-sample inference contract.
    """
    return int(EFFICIENTAT_WINDOW_SECONDS * source_sr)


# Contract self-check -- runs at import time, crashes on mismatch.
# If this ever fires, someone broke EfficientATMelConfig and this module is
# the last line of defense before a silent train/serve split.
assert EFFICIENTAT_SEGMENT_SAMPLES == int(
    EFFICIENTAT_WINDOW_SECONDS * EFFICIENTAT_TARGET_SR
), (
    f"EfficientAT window contract broken: segment_samples={EFFICIENTAT_SEGMENT_SAMPLES} "
    f"!= window_seconds * target_sr = {int(EFFICIENTAT_WINDOW_SECONDS * EFFICIENTAT_TARGET_SR)}"
)
