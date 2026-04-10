"""Target tracker with UUID lifecycle management.

Multi-target tracking with nearest-neighbor peak-to-target association (DOA-03).
Assigns UUIDs on first detection, updates bearing as sources move, and marks
targets lost after TTL expiry. Supports configurable EMA direction smoothing (D-05)
and pan/tilt output via DOA coordinate transform.

Doppler speed (TRK-02) deferred to milestone 2 per D-07 -- speed_mps is always None.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field

from acoustic.tracking.doa import MountingOrientation, array_to_world
from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.schema import EventType, TargetEvent
from acoustic.types import PeakDetection

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
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    lost: bool = False


class TargetTracker:
    """Manages target lifecycle: create, update, lose.

    Supports multiple simultaneous targets with nearest-neighbor association (D-03/D-04).
    Pan/tilt populated via array_to_world coordinate transform with configurable
    EMA smoothing (alpha=1.0 = no smoothing).

    Args:
        ttl: Seconds of no detection before target is marked lost.
        broadcaster: Optional EventBroadcaster for emitting tracking events.
        mounting: Physical mounting orientation of the array.
        association_threshold_deg: Max angular distance for peak-to-target match.
        smoothing_alpha: EMA alpha for direction smoothing (1.0 = raw pass-through).
    """

    def __init__(
        self,
        ttl: float = 5.0,
        broadcaster: EventBroadcaster | None = None,
        mounting: MountingOrientation = MountingOrientation.VERTICAL_Y_UP,
        association_threshold_deg: float = 7.5,
        smoothing_alpha: float = 1.0,
    ) -> None:
        self._ttl = ttl
        self._broadcaster = broadcaster
        self._mounting = mounting
        self._association_threshold = association_threshold_deg
        self._smoothing_alpha = smoothing_alpha
        self._targets: dict[str, TrackedTarget] = {}

    @staticmethod
    def _angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
        """Euclidean angular distance in degrees (sufficient for small angles)."""
        return math.sqrt((az1 - az2) ** 2 + (el1 - el2) ** 2)

    def update_multi(
        self,
        peaks: list[PeakDetection],
        confidence: float,
        class_label: str = "drone",
    ) -> list[TrackedTarget]:
        """Associate peaks to existing targets, create new targets for unmatched peaks.

        Uses greedy nearest-neighbor: for each active target, find the closest
        unmatched peak within the association threshold. Unmatched peaks become
        new targets. Unmatched targets are NOT lost (TTL handles expiry via tick()).

        Returns list of targets that were created or updated this call.
        """
        active = self.get_active_targets()
        matched_peak_indices: set[int] = set()
        updated: list[TrackedTarget] = []

        # Greedy nearest-neighbor: for each target, find closest unmatched peak
        for target in active:
            best_idx = -1
            best_dist = float("inf")
            for i, peak in enumerate(peaks):
                if i in matched_peak_indices:
                    continue
                dist = self._angular_distance(
                    target.az_deg, target.el_deg, peak.az_deg, peak.el_deg
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0 and best_dist < self._association_threshold:
                matched_peak_indices.add(best_idx)
                pk = peaks[best_idx]
                raw_pan, raw_tilt = array_to_world(pk.az_deg, pk.el_deg, self._mounting)
                # EMA smoothing (alpha=1.0 means no smoothing)
                alpha = self._smoothing_alpha
                target.pan_deg = alpha * raw_pan + (1 - alpha) * target.pan_deg
                target.tilt_deg = alpha * raw_tilt + (1 - alpha) * target.tilt_deg
                target.az_deg = pk.az_deg
                target.el_deg = pk.el_deg
                target.confidence = confidence
                target.class_label = class_label
                target.last_seen = time.monotonic()
                self._emit(EventType.UPDATE, target)
                updated.append(target)

        # Unmatched peaks become new targets
        for i, peak in enumerate(peaks):
            if i not in matched_peak_indices:
                pan, tilt = array_to_world(peak.az_deg, peak.el_deg, self._mounting)
                target = TrackedTarget(
                    id=str(uuid.uuid4()),
                    class_label=class_label,
                    az_deg=peak.az_deg,
                    el_deg=peak.el_deg,
                    pan_deg=pan,
                    tilt_deg=tilt,
                    confidence=confidence,
                )
                self._targets[target.id] = target
                self._emit(EventType.NEW, target)
                logger.info(
                    "New target %s (%.1f deg, %.1f deg) pan=%.1f tilt=%.1f",
                    target.id[:8],
                    peak.az_deg,
                    peak.el_deg,
                    pan,
                    tilt,
                )
                updated.append(target)

        return updated

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

    def clear(self) -> None:
        """Remove all targets (used when stopping a detection session)."""
        self._targets.clear()

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
                "pan_deg": t.pan_deg,
                "tilt_deg": t.tilt_deg,
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
            pan_deg=target.pan_deg,
            tilt_deg=target.tilt_deg,
            speed_mps=target.speed_mps,
            timestamp=time.monotonic(),
        )
        self._broadcaster.broadcast(event)
