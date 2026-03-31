"""Tests for hysteresis detection state machine."""

from __future__ import annotations

import pytest


class TestDetectionStateMachine:
    def test_initial_state_no_drone(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine()
        assert sm.state == DetectionState.NO_DRONE

    def test_high_prob_transitions_to_candidate(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, confirm_hits=2)
        sm.update(0.85)
        assert sm.state == DetectionState.CANDIDATE

    def test_confirm_hits_transitions_to_confirmed(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, confirm_hits=2)
        sm.update(0.85)  # -> CANDIDATE (hit_count=1)
        sm.update(0.90)  # -> CONFIRMED (hit_count=2)
        assert sm.state == DetectionState.CONFIRMED

    def test_low_prob_candidate_resets_to_no_drone(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, exit_threshold=0.40, confirm_hits=2)
        sm.update(0.85)  # -> CANDIDATE
        sm.update(0.30)  # -> NO_DRONE (below exit)
        assert sm.state == DetectionState.NO_DRONE

    def test_confirmed_stays_while_above_exit(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, exit_threshold=0.40, confirm_hits=2)
        sm.update(0.85)
        sm.update(0.90)  # CONFIRMED
        sm.update(0.50)  # still above exit
        assert sm.state == DetectionState.CONFIRMED

    def test_confirmed_drops_to_candidate_on_low(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, exit_threshold=0.40, confirm_hits=2)
        sm.update(0.85)
        sm.update(0.90)  # CONFIRMED
        sm.update(0.30)  # below exit -> CANDIDATE
        assert sm.state == DetectionState.CANDIDATE

    def test_confirmed_to_no_drone_requires_two_lows(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, exit_threshold=0.40, confirm_hits=2)
        sm.update(0.85)
        sm.update(0.90)  # CONFIRMED
        sm.update(0.30)  # -> CANDIDATE
        sm.update(0.30)  # -> NO_DRONE
        assert sm.state == DetectionState.NO_DRONE

    def test_reset(self):
        from acoustic.classification.state_machine import DetectionState, DetectionStateMachine

        sm = DetectionStateMachine(enter_threshold=0.80, confirm_hits=2)
        sm.update(0.85)
        sm.update(0.90)  # CONFIRMED
        sm.reset()
        assert sm.state == DetectionState.NO_DRONE
