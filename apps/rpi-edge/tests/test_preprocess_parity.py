"""Wave 0 RED stub — parity test between vendored preprocess and training source.

Covers: D-04 (preprocess vendoring + drift guard).
Owner: Plan 21-02 (vendor preprocess.py + mel_banks_128_1024_32k.pt into apps/rpi-edge/).
"""
from __future__ import annotations

import pytest


def test_vendored_preprocess_matches_training_reference_within_atol_1e_5(golden_drone_wav):
    pytest.fail(
        "not implemented — Plan 21-02 must vendor preprocess.py and assert "
        "np.allclose(vendored_mel, training_mel, atol=1e-5)"
    )
