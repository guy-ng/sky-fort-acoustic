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
    # 0-indexed channels to extract from the device stream. None means "open
    # `channels` raw and pass straight through". Used to skip on-board DSP
    # channels (e.g. ReSpeaker XMOS firmware exposes channel 0 as the
    # AEC/AGC/beamformer output and channels 1-4 as raw mic capsules — we want
    # the raw mic, not the processed channel which is out-of-distribution for
    # a CNN trained on raw UMA-16 audio).
    mic_channels: tuple[int, ...] | None = None
    # Device-specific cap on the CNN preprocessor input gain. None means
    # "use whatever the user/UI sends". The UMA-16v2 was characterized at
    # input_gain=500; smaller-capsule fallback mics (e.g. ReSpeaker raw mic 1)
    # have completely different sensitivity and clip badly at that level, so
    # we cap to a safe value when known. Only consulted by the /pipeline/start
    # route — never applied automatically at capture time.
    recommended_gain: float | None = None
    # Streaming Butterworth high-pass cutoff applied at capture time, in Hz.
    # None means "no high-pass". Used to scrub mains hum (50/60 Hz fundamental
    # plus harmonics) on small USB mics whose power isolation is worse than
    # the UMA-16's. When `lowpass_hz` is also set, the capture path builds
    # one combined bandpass instead of two cascaded filters.
    highpass_hz: float | None = None
    # Streaming Butterworth low-pass cutoff applied at capture time, in Hz.
    # None means "no low-pass". Used to drop HF ambient/HVAC junk above the
    # drone harmonic band (~4 kHz). When set together with `highpass_hz`,
    # AudioCapture builds a single bandpass SOS rather than cascading filters.
    lowpass_hz: float | None = None


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
