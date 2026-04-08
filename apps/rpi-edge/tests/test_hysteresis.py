"""D-12/D-14: Hysteresis state machine tests.

Owner: Plan 21-05 (skyfort_edge/hysteresis.py).
"""
from __future__ import annotations

import pytest

from skyfort_edge.hysteresis import EventType, HysteresisStateMachine, State


def _sm() -> HysteresisStateMachine:
    return HysteresisStateMachine(
        enter_threshold=0.6,
        exit_threshold=0.4,
        confirm_hits=3,
        release_hits=5,
        min_on_seconds=2.0,
    )


def test_rising_edge_latches_after_confirm_hits():
    sm = _sm()
    assert sm.update(0.7, 0.0) is None   # hit 1
    assert sm.update(0.7, 0.5) is None   # hit 2
    ev = sm.update(0.7, 1.0)              # hit 3 -> latch
    assert ev is not None
    assert ev.type == EventType.RISING_EDGE
    assert sm.is_latched
    assert sm.state == State.LATCHED
    assert sm.latch_start_time == 1.0


def test_below_enter_resets_confirm_counter():
    sm = _sm()
    sm.update(0.7, 0.0)  # hit 1
    sm.update(0.7, 0.1)  # hit 2
    sm.update(0.5, 0.2)  # not above enter -> reset
    assert not sm.is_latched
    # Need three fresh consecutive above-hits to latch now.
    assert sm.update(0.7, 0.3) is None
    assert sm.update(0.7, 0.4) is None
    ev = sm.update(0.7, 0.5)
    assert ev is not None and ev.type == EventType.RISING_EDGE


def test_min_on_seconds_held_after_last_positive():
    sm = _sm()
    sm.update(0.7, 0.0)
    sm.update(0.7, 0.5)
    sm.update(0.7, 1.0)  # latched at t=1.0
    # Scores drop immediately. release_hits=5 would trip by t=1.5, but
    # min_on_seconds=2.0 means we must hold LATCHED until at least t=3.0.
    for t in [1.1, 1.2, 1.3, 1.4, 1.5]:
        sm.update(0.1, t)
    assert sm.is_latched, "released before min_on_seconds"


def test_release_after_release_hits_below_exit_threshold():
    sm = _sm()
    sm.update(0.7, 0.0)
    sm.update(0.7, 0.5)
    sm.update(0.7, 1.0)  # latched at t=1.0
    # Wait past min_on_seconds, then feed release_hits low scores.
    ev = None
    for t in [3.1, 3.2, 3.3, 3.4, 3.5]:
        ev = sm.update(0.1, t)
    assert ev is not None and ev.type == EventType.FALLING_EDGE
    assert not sm.is_latched
    assert ev.latch_duration_seconds > 2.0


def test_exit_threshold_must_be_below_enter_threshold():
    with pytest.raises(ValueError):
        HysteresisStateMachine(
            enter_threshold=0.4,
            exit_threshold=0.6,
            confirm_hits=3,
            release_hits=5,
            min_on_seconds=2.0,
        )
