"""Shared types for the acoustic service."""

from __future__ import annotations

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
