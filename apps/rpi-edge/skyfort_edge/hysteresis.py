"""Hysteresis state machine for drone detection latching (D-12/D-14).

Semantics ported (not imported — D-25) from
src/acoustic/classification/state_machine.py. The edge app must not reach into
the main service package on the Pi.

Binary-head note (21-01-SUMMARY): efficientat_mn10_v6 is a single-logit sigmoid
head, so 'score' here is the scalar drone probability in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(Enum):
    IDLE = "idle"
    LATCHED = "latched"


class EventType(Enum):
    RISING_EDGE = "rising_edge"
    FALLING_EDGE = "falling_edge"


@dataclass
class StateEvent:
    type: EventType
    timestamp: float
    score: float
    latch_duration_seconds: float = 0.0


class HysteresisStateMachine:
    """K-of-N latch state machine with min-on hold.

    Args:
        enter_threshold: score >= this counts as an "above" hit.
        exit_threshold: score <= this counts as a "below" hit. MUST be <= enter.
        confirm_hits: consecutive above-hits required to latch (rising edge).
        release_hits: consecutive below-hits required to release (falling edge).
        min_on_seconds: minimum time in LATCHED state before a release is allowed.
    """

    def __init__(
        self,
        enter_threshold: float,
        exit_threshold: float,
        confirm_hits: int,
        release_hits: int,
        min_on_seconds: float,
    ) -> None:
        if exit_threshold > enter_threshold:
            raise ValueError(
                f"exit_threshold ({exit_threshold}) must be <= "
                f"enter_threshold ({enter_threshold})"
            )
        if confirm_hits < 1 or release_hits < 1:
            raise ValueError("confirm_hits and release_hits must be >= 1")
        if min_on_seconds < 0:
            raise ValueError("min_on_seconds must be >= 0")

        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.confirm_hits = int(confirm_hits)
        self.release_hits = int(release_hits)
        self.min_on_seconds = float(min_on_seconds)

        self._state = State.IDLE
        self._above_count = 0
        self._below_count = 0
        self._latch_start_time: Optional[float] = None
        self._last_positive_time: Optional[float] = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_latched(self) -> bool:
        return self._state == State.LATCHED

    @property
    def latch_start_time(self) -> Optional[float]:
        return self._latch_start_time

    def update(self, score: float, timestamp: float) -> Optional[StateEvent]:
        """Feed one score sample. Returns a StateEvent on edge transitions."""
        score = float(score)
        if self._state == State.IDLE:
            if score >= self.enter_threshold:
                self._above_count += 1
                if self._above_count >= self.confirm_hits:
                    self._state = State.LATCHED
                    self._latch_start_time = timestamp
                    self._last_positive_time = timestamp
                    self._above_count = 0
                    self._below_count = 0
                    return StateEvent(EventType.RISING_EDGE, timestamp, score)
            else:
                self._above_count = 0
            return None

        # LATCHED
        if score >= self.enter_threshold:
            self._last_positive_time = timestamp
        if score <= self.exit_threshold:
            self._below_count += 1
        else:
            self._below_count = 0

        if self._below_count >= self.release_hits:
            latch_start = self._latch_start_time if self._latch_start_time is not None else timestamp
            last_pos = self._last_positive_time if self._last_positive_time is not None else latch_start
            elapsed = timestamp - latch_start
            since_positive = timestamp - last_pos
            if elapsed >= self.min_on_seconds and since_positive >= self.min_on_seconds:
                latch_duration = elapsed
                self._state = State.IDLE
                self._latch_start_time = None
                self._last_positive_time = None
                self._below_count = 0
                self._above_count = 0
                return StateEvent(
                    EventType.FALLING_EDGE, timestamp, score, latch_duration
                )
        return None
