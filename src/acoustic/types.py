"""Shared types for the acoustic service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

# Type alias for documentation — a chunk of audio data
AudioChunk = np.ndarray


@dataclass
class DeviceInfo:
    """Information about a detected audio device."""

    index: int
    name: str
    channels: int
    default_samplerate: float
    # True when this device was selected because the UMA-16v2 was not present.
    # Callers may use this to switch to detection-only (mono) capture.
    is_fallback: bool = False


@dataclass
class PeakDetection:
    """A detected peak in the beamforming spatial map."""

    az_deg: float
    el_deg: float
    power: float
    threshold: float


@dataclass
class HealthStatus:
    """Service health status."""

    status: str
    device_detected: bool
    pipeline_running: bool
    overflow_count: int
    last_frame_time: float | None


# Placeholder target ID used until real CNN classification in Phase 3
PLACEHOLDER_TARGET_ID = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))


def placeholder_target_from_peak(peak: PeakDetection) -> dict:
    """Generate a placeholder target from peak detection data.

    Fallback when CNN is not available. Real targets come from TargetTracker.
    """
    return {
        "id": PLACEHOLDER_TARGET_ID,
        "class_label": "unknown",
        "speed_mps": None,
        "az_deg": peak.az_deg,
        "el_deg": peak.el_deg,
        "confidence": min(peak.power / (peak.threshold * 2), 1.0),
    }
