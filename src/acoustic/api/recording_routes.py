"""REST API endpoints for field recording lifecycle and metadata CRUD."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class RecordingStartResponse(BaseModel):
    id: str
    status: str  # "recording"


class RecordingStopResponse(BaseModel):
    id: str
    duration_s: float
    status: str  # "stopped"


class LabelRequest(BaseModel):
    label: str  # Must be "drone", "background", or "other"
    sub_label: str | None = None
    distance_m: float | None = None
    altitude_m: float | None = None
    conditions: str | None = None
    notes: str | None = None


class MetadataUpdateRequest(BaseModel):
    sub_label: str | None = None
    distance_m: float | None = None
    altitude_m: float | None = None
    conditions: str | None = None
    notes: str | None = None


class RecordingDeleteResponse(BaseModel):
    id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=None)
async def start_recording(request: Request) -> RecordingStartResponse | JSONResponse:
    """Start a new recording session.

    Returns 409 if already recording. Returns 503 if no audio device.
    """
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    try:
        rec_id = manager.start_recording()
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"message": str(exc)})
    return RecordingStartResponse(id=rec_id, status="recording")


@router.post("/stop", response_model=None)
async def stop_recording(request: Request) -> RecordingStopResponse | JSONResponse:
    """Stop the current recording session.

    Returns 409 if not currently recording.
    """
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    try:
        rec_id, duration = manager.stop_recording()
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"message": str(exc)})
    return RecordingStopResponse(id=rec_id, duration_s=round(duration, 2), status="stopped")


@router.get("", response_model=None)
async def list_recordings(request: Request) -> list[dict]:
    """List all recordings with metadata."""
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    return manager.list_recordings()


@router.get("/{rec_id}", response_model=None)
async def get_recording(rec_id: str, request: Request) -> dict | JSONResponse:
    """Get a single recording's metadata by ID."""
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    result = manager.get_recording(rec_id)
    if result is None:
        return JSONResponse(status_code=404, content={"message": f"Recording {rec_id} not found"})
    return result


@router.post("/{rec_id}/label", response_model=None)
async def label_recording(rec_id: str, body: LabelRequest, request: Request) -> dict | JSONResponse:
    """Assign a label to a recording and move it to the label directory.

    Returns 400 if the label is invalid, 404 if the recording is not found.
    """
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    extra = body.model_dump(exclude={"label"}, exclude_none=True)
    try:
        manager.label_recording(rec_id, body.label, extra=extra if extra else None)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"message": str(exc)})
    # Return updated metadata
    result = manager.get_recording(rec_id)
    return result if result is not None else JSONResponse(
        status_code=404, content={"message": f"Recording {rec_id} not found after labeling"}
    )


@router.patch("/{rec_id}", response_model=None)
async def update_recording(rec_id: str, body: MetadataUpdateRequest, request: Request) -> dict | JSONResponse:
    """Update metadata fields for a recording."""
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    updates = body.model_dump(exclude_none=True)
    result = manager.update_recording(rec_id, updates)
    if result is None:
        return JSONResponse(status_code=404, content={"message": f"Recording {rec_id} not found"})
    return result


@router.delete("/{rec_id}", response_model=None)
async def delete_recording(rec_id: str, request: Request) -> RecordingDeleteResponse | JSONResponse:
    """Delete a recording and its metadata."""
    from acoustic.recording.manager import RecordingManager

    manager: RecordingManager = request.app.state.recording_manager
    deleted = manager.delete_recording(rec_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"message": f"Recording {rec_id} not found"})
    return RecordingDeleteResponse(id=rec_id, deleted=True)
