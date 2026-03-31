"""Hysteresis state machine for drone detection — prevents flickering."""

from __future__ import annotations

from enum import Enum


class DetectionState(str, Enum):
    """Possible states for drone detection."""

    NO_DRONE = "NO_DRONE"
    CANDIDATE = "DRONE_CANDIDATE"
    CONFIRMED = "DRONE_CONFIRMED"


class DetectionStateMachine:
    """Three-state hysteresis detector.

    Transitions:
        NO_DRONE   --[prob >= enter]--> CANDIDATE (hit_count=1)
        CANDIDATE  --[prob >= enter, hits >= confirm]--> CONFIRMED
        CANDIDATE  --[prob <= exit]--> NO_DRONE
        CONFIRMED  --[prob <= exit]--> CANDIDATE (hit_count reset)
        CANDIDATE  --[prob <= exit]--> NO_DRONE
    """

    def __init__(
        self,
        enter_threshold: float = 0.80,
        exit_threshold: float = 0.40,
        confirm_hits: int = 2,
    ) -> None:
        self._enter = enter_threshold
        self._exit = exit_threshold
        self._confirm = confirm_hits
        self._state = DetectionState.NO_DRONE
        self._hit_count = 0

    @property
    def state(self) -> DetectionState:
        """Current detection state."""
        return self._state

    def update(self, drone_probability: float) -> DetectionState:
        """Feed a new probability and advance the state machine.

        Args:
            drone_probability: CNN output in [0.0, 1.0].

        Returns:
            The new DetectionState after this update.
        """
        if self._state == DetectionState.NO_DRONE:
            if drone_probability >= self._enter:
                self._state = DetectionState.CANDIDATE
                self._hit_count = 1

        elif self._state == DetectionState.CANDIDATE:
            if drone_probability >= self._enter:
                self._hit_count += 1
                if self._hit_count >= self._confirm:
                    self._state = DetectionState.CONFIRMED
            elif drone_probability <= self._exit:
                self._state = DetectionState.NO_DRONE
                self._hit_count = 0

        elif self._state == DetectionState.CONFIRMED:
            if drone_probability <= self._exit:
                self._state = DetectionState.CANDIDATE
                self._hit_count = 0

        return self._state

    def reset(self) -> None:
        """Reset to initial NO_DRONE state."""
        self._state = DetectionState.NO_DRONE
        self._hit_count = 0
