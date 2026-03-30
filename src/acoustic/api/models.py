"""Pydantic response models for the acoustic service API."""

from __future__ import annotations

from pydantic import BaseModel


class TargetState(BaseModel):
    """A detected target with current state."""

    id: str  # UUID string
    class_label: str  # "unknown" for Phase 2 placeholder
    speed_mps: float | None  # None until Phase 3 Doppler
    az_deg: float  # Azimuth in degrees
    el_deg: float  # Elevation in degrees
    confidence: float  # 0.0-1.0


class BeamformingMapResponse(BaseModel):
    """Beamforming map as JSON grid with metadata."""

    az_min: float  # -90.0
    az_max: float  # 90.0
    el_min: float  # -45.0
    el_max: float  # 45.0
    az_resolution: float  # 1.0
    el_resolution: float  # 1.0
    width: int  # 181 (azimuth grid points)
    height: int  # 91 (elevation grid points)
    data: list[list[float]]  # 2D grid [elevation][azimuth] -- row-major for canvas
    peak: dict | None  # {"az_deg": float, "el_deg": float, "power": float} if detected


class HeatmapHandshake(BaseModel):
    """Initial WebSocket handshake message with grid dimensions."""

    type: str = "handshake"
    width: int  # 181
    height: int  # 91
    az_min: float
    az_max: float
    el_min: float
    el_max: float
