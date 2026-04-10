"""REST API endpoints for beamforming map and target data."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from acoustic.api.models import BeamformingMapResponse, TargetState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class SettingsUpdate(BaseModel):
    """Partial settings update for runtime-configurable parameters."""

    bf_nu: float | None = Field(None, ge=1.0, le=1000.0)


@router.get("/map", response_model=BeamformingMapResponse)
async def get_map(request: Request) -> BeamformingMapResponse | JSONResponse:
    """Return the current beamforming map as JSON with grid metadata.

    Returns 503 if the pipeline has not yet produced a map.
    """
    pipeline = request.app.state.pipeline
    settings = request.app.state.settings

    latest_map = pipeline.latest_map
    if latest_map is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Pipeline not ready"},
        )

    # Transpose so data is [elevation][azimuth] (row-major for canvas rendering)
    data_2d = latest_map.T.tolist()

    # Build peak dict if detected
    peak_dict = None
    if pipeline.latest_peak is not None:
        p = pipeline.latest_peak
        peak_dict = {
            "az_deg": p.az_deg,
            "el_deg": p.el_deg,
            "power": p.power,
        }

    return BeamformingMapResponse(
        az_min=-settings.az_range,
        az_max=settings.az_range,
        el_min=-settings.el_range,
        el_max=settings.el_range,
        az_resolution=settings.az_resolution,
        el_resolution=settings.el_resolution,
        width=latest_map.shape[0],
        height=latest_map.shape[1],
        data=data_2d,
        peak=peak_dict,
    )


@router.get("/targets", response_model=list[TargetState])
async def get_targets(request: Request) -> list[dict]:
    """Return current detected targets as a JSON array.

    Returns real targets from TargetTracker when CNN is enabled,
    or placeholder targets from peak detection as fallback.
    """
    pipeline = request.app.state.pipeline
    return pipeline.latest_targets


@router.patch("/settings")
async def update_settings(request: Request, body: SettingsUpdate) -> dict:
    """Update runtime-configurable acoustic settings.

    Currently supports: bf_nu (functional beamforming exponent).
    """
    settings = request.app.state.settings
    updated = {}
    for field_name, value in body.model_dump(exclude_none=True).items():
        setattr(settings, field_name, value)
        updated[field_name] = value
    return {"updated": updated}
