"""Wave 0 RED stub — 48 kHz -> 32 kHz polyphase resampling.

Covers: D-02 (USB mic 48 kHz capture, model trained on 32 kHz mel input).
Owner: Plan 21-02 (skyfort_edge/audio.py — resample_poly(2, 3)).
"""
from __future__ import annotations

import pytest


def test_resample_poly_2_3_correctness():
    pytest.fail(
        "not implemented — Plan 21-02: scipy.signal.resample_poly(x, up=2, down=3) on "
        "1s of 48k must yield exactly 32000 samples and match expected sinusoid frequency"
    )


def test_resample_latency_under_50ms_for_1s_window():
    pytest.fail(
        "not implemented — Plan 21-02: resampling 1 s of 48 kHz audio must complete in "
        "< 50 ms on host (latency budget proxy)"
    )
