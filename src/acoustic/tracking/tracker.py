"""Target tracker with UUID lifecycle management.

Single-target tracking for Phase 3. Assigns a UUID on first confirmed detection,
updates bearing on subsequent detections, and marks lost after TTL expiry.
Doppler speed (TRK-02) deferred to milestone 2 per D-07 -- speed_mps is always None.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType, TargetEvent

logger = logging.getLogger(__name__)


@dataclass
class TrackedTarget:
    """A tracked target with UUID and spatial state."""

    id: str
    class_label: str
    az_deg: float
    el_deg: float
    confidence: float
    speed_mps: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    lost: bool = False


class TargetTracker:
    """Manages target lifecycle: create, update, lose.

    Single active target at a time (Phase 3 scope). Multi-target is a future enhancement.

    Args:
        ttl: Seconds of no detection before target is marked lost.
        broadcaster: Optional EventBroadcaster for emitting tracking events.
    """

    def __init__(self, ttl: float = 5.0, broadcaster: EventBroadcaster | None = None) -> None:
        self._ttl = ttl
        self._broadcaster = broadcaster
        self._targets: dict[str, TrackedTarget] = {}

    def update(
        self,
        az_deg: float,
        el_deg: float,
        confidence: float,
        class_label: str = "drone",
    ) -> TrackedTarget:
        """Create or update the active target.

        If no active (non-lost) target exists, creates a new one with a UUID4.
        If an active target exists, updates its bearing, confidence, and last_seen.

        Returns the TrackedTarget.
        """
        active = self._get_active_target()

        if active is None:
            # Create new target
            target = TrackedTarget(
                id=str(uuid.uuid4()),
                class_label=class_label,
                az_deg=az_deg,
                el_deg=el_deg,
                confidence=confidence,
            )
            self._targets[target.id] = target
            self._emit(EventType.NEW, target)
            logger.info("New target %s (%.1f°, %.1f°) conf=%.2f", target.id[:8], az_deg, el_deg, confidence)
            return target

        # Update existing target
        active.az_deg = az_deg
        active.el_deg = el_deg
        active.confidence = confidence
        active.class_label = class_label
        active.last_seen = time.monotonic()
        self._emit(EventType.UPDATE, active)
        return active

    def tick(self) -> list[str]:
        """Check for expired targets and mark them lost.

        Returns list of target IDs that were just marked lost.
        """
        now = time.monotonic()
        lost_ids: list[str] = []

        for target in list(self._targets.values()):
            if not target.lost and (now - target.last_seen) > self._ttl:
                target.lost = True
                lost_ids.append(target.id)
                self._emit(EventType.LOST, target)
                logger.info("Target lost: %s (TTL expired)", target.id[:8])

        return lost_ids

    def get_active_targets(self) -> list[TrackedTarget]:
        """Return list of non-lost targets."""
        return [t for t in self._targets.values() if not t.lost]

    def get_target_states(self) -> list[dict]:
        """Return list of dicts compatible with TargetState model.

        Used by /ws/targets and /api/targets endpoints.
        """
        return [
            {
                "id": t.id,
                "class_label": t.class_label,
                "speed_mps": t.speed_mps,
                "az_deg": t.az_deg,
                "el_deg": t.el_deg,
                "confidence": t.confidence,
            }
            for t in self._targets.values()
            if not t.lost
        ]

    def _get_active_target(self) -> TrackedTarget | None:
        """Return the single active (non-lost) target, or None."""
        for t in self._targets.values():
            if not t.lost:
                return t
        return None

    def _emit(self, event_type: EventType, target: TrackedTarget) -> None:
        """Emit a tracking event via the broadcaster if available."""
        if self._broadcaster is None:
            return
        event = TargetEvent(
            event=event_type,
            target_id=target.id,
            class_label=target.class_label,
            confidence=target.confidence,
            az_deg=target.az_deg,
            el_deg=target.el_deg,
            speed_mps=target.speed_mps,
            timestamp=time.monotonic(),
        )
        self._broadcaster.broadcast(event)
