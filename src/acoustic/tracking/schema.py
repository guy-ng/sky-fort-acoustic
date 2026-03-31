"""Target event schema for tracking events."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class EventType(str, Enum):
    """Type of tracking event."""

    NEW = "new"
    UPDATE = "update"
    LOST = "lost"


class TargetEvent(BaseModel):
    """A tracking event emitted when target state changes.

    Fields:
        event: Type of event (new, update, lost).
        target_id: UUID string identifying the target.
        class_label: Classification label (e.g. "drone", "background").
        confidence: Detection confidence 0.0-1.0.
        az_deg: Azimuth bearing in degrees.
        el_deg: Elevation bearing in degrees.
        speed_mps: Speed in m/s. Always None until Doppler is implemented (D-07).
        timestamp: Monotonic timestamp of the event.
    """

    event: EventType
    target_id: str
    class_label: str
    confidence: float
    az_deg: float
    el_deg: float
    speed_mps: float | None = None
    timestamp: float
