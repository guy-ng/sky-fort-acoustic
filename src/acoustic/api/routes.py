"""REST API endpoints for beamforming map and target data."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from acoustic.api.models import BeamformingMapResponse, TargetState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class SettingsUpdate(BaseModel):
    """Partial settings update for runtime-configurable parameters."""

    bf_nu: float | None = Field(None, ge=1.0, le=1000.0)
    bf_peak_threshold: float | None = Field(None, ge=0.1, le=50.0)
    bf_mass_threshold: float | None = Field(None, ge=0.01, le=0.99)
    bf_always_on: bool | None = None


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


@router.post("/raw-recording/start")
async def start_raw_recording(request: Request) -> dict:
    """Start a raw 16-channel 48kHz recording (60s max, auto-stops)."""
    pipeline = request.app.state.pipeline
    state = pipeline.raw_recording_state
    if state["status"] == "recording":
        return JSONResponse(status_code=409, content={"detail": "Already recording", **state})
    rec_id = pipeline.start_raw_recording()
    return {"id": rec_id, "status": "recording", "max_seconds": 60}


@router.post("/raw-recording/stop")
async def stop_raw_recording(request: Request) -> dict:
    """Stop the current raw recording."""
    pipeline = request.app.state.pipeline
    info = pipeline.stop_raw_recording()
    if info is None:
        return JSONResponse(status_code=404, content={"detail": "No active recording"})
    return info


@router.get("/raw-recording/status")
async def raw_recording_status(request: Request) -> dict:
    """Return current raw recording state."""
    pipeline = request.app.state.pipeline
    return pipeline.raw_recording_state


RAW_REC_DIR = Path("data/raw_recordings")


@router.get("/raw-recordings")
async def list_raw_recordings() -> list[dict]:
    """List all raw 16-channel recordings."""
    if not RAW_REC_DIR.exists():
        return []
    import soundfile as sf
    recs = []
    for wav in sorted(RAW_REC_DIR.glob("*.wav"), reverse=True):
        try:
            info = sf.info(str(wav))
            recs.append({
                "id": wav.stem,
                "filename": wav.name,
                "channels": info.channels,
                "sample_rate": info.samplerate,
                "duration_s": round(info.duration, 1),
                "size_bytes": wav.stat().st_size,
            })
        except Exception:
            continue
    return recs


@router.get("/raw-recordings/{rec_id}/audio")
async def download_raw_recording(rec_id: str) -> FileResponse:
    """Download a raw recording WAV file."""
    path = RAW_REC_DIR / f"{rec_id}.wav"
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": "Recording not found"})  # type: ignore[return-value]
    return FileResponse(str(path), media_type="audio/wav", filename=path.name)


@router.post("/raw-recordings/{rec_id}/playback")
async def start_playback(rec_id: str, request: Request) -> dict:
    """Play a raw 16ch recording through the beamforming pipeline.

    Replaces live audio capture with file playback. The pipeline processes
    the recording in real time so the heatmap, spectrum, and peaks update.
    """
    path = RAW_REC_DIR / f"{rec_id}.wav"
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": "Recording not found"})  # type: ignore[return-value]
    pipeline = request.app.state.pipeline
    pipeline.start_file_playback(str(path))
    return {"status": "playing", "id": rec_id, "path": str(path)}


@router.post("/raw-recordings/stop-playback")
async def stop_playback(request: Request) -> dict:
    """Stop file playback and return to live audio."""
    pipeline = request.app.state.pipeline
    pipeline.stop_file_playback()
    return {"status": "stopped"}


# --- Target location recording -----------------------------------------------

TARGET_REC_DIR = Path("data/target_recordings")


@router.post("/target-recording/start")
async def start_target_recording(request: Request) -> dict:
    """Start recording target locations to a JSON file."""
    pipeline = request.app.state.pipeline
    state = pipeline.target_recording_state
    if state["status"] == "recording":
        return JSONResponse(status_code=409, content={"detail": "Already recording", **state})
    rec_id = pipeline.start_target_recording()
    return {"id": rec_id, "status": "recording"}


@router.post("/target-recording/stop")
async def stop_target_recording(request: Request) -> dict:
    """Stop the current target recording and save JSON file."""
    pipeline = request.app.state.pipeline
    info = pipeline.stop_target_recording()
    if info is None:
        return JSONResponse(status_code=404, content={"detail": "No active target recording"})  # type: ignore[return-value]
    return info


@router.get("/target-recording/status")
async def target_recording_status(request: Request) -> dict:
    """Return current target recording state."""
    pipeline = request.app.state.pipeline
    return pipeline.target_recording_state


@router.get("/target-recordings")
async def list_target_recordings() -> list[dict]:
    """List saved target recording JSON files."""
    import json

    if not TARGET_REC_DIR.exists():
        return []
    recs = []
    for jf in sorted(TARGET_REC_DIR.glob("*.json"), reverse=True):
        try:
            with open(jf) as f:
                data = json.load(f)
            recs.append({
                "id": data.get("id", jf.stem),
                "duration_s": data.get("duration_s", 0),
                "total_samples": data.get("total_samples", 0),
                "size_bytes": jf.stat().st_size,
            })
        except Exception:
            continue
    return recs
