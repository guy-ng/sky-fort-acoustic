"""Wave 0 RED stub — hysteresis state-machine tests for detection latching.

Covers: D-12 (enter/exit thresholds + confirm/release hits), D-14 (min_on_seconds).
Owner: Plan 21-04 (skyfort_edge/hysteresis.py).
"""
from __future__ import annotations

import pytest


def test_rising_edge_latches_after_confirm_hits():
    pytest.fail(
        "not implemented — Plan 21-04 must implement Hysteresis state machine; "
        "feed N consecutive p>=enter_threshold, assert state transitions to LATCHED"
    )


def test_min_on_seconds_held_after_last_positive():
    pytest.fail(
        "not implemented — Plan 21-04: after latch, drop to p<exit_threshold and assert "
        "state stays LATCHED until min_on_seconds elapsed"
    )


def test_release_after_release_hits_below_exit_threshold():
    pytest.fail(
        "not implemented — Plan 21-04: after min_on_seconds, M consecutive p<exit_threshold "
        "must transition state to RELEASED"
    )
