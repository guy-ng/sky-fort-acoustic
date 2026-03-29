"""Shared types for the acoustic service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PeakDetection:
    """Result of peak detection on a beamforming spatial map.

    Attributes:
        az_deg: Azimuth of the peak in degrees
        el_deg: Elevation of the peak in degrees
        power: Power value at the peak
        threshold: Noise threshold that was exceeded
    """

    az_deg: float
    el_deg: float
    power: float
    threshold: float
