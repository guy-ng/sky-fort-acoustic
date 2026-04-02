"""Recording metadata: sidecar JSON schema and CRUD operations."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class RecordingMetadata(BaseModel):
    """Sidecar JSON metadata for a field recording.

    Required: label (str). All other fields are optional.
    Matches the D-10 field set from the context decisions.
    """

    label: str
    sub_label: str | None = None
    distance_m: float | None = None
    altitude_m: float | None = None
    conditions: str | None = None
    notes: str | None = None
    recorded_at: str | None = None
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    original_sr: int | None = None
    filename: str | None = None


def write_metadata(json_path: Path, meta: RecordingMetadata) -> None:
    """Write metadata to a sidecar JSON file (exclude None fields, indent=2)."""
    data = meta.model_dump(exclude_none=True)
    json_path.write_text(json.dumps(data, indent=2) + "\n")


def read_metadata(json_path: Path) -> RecordingMetadata:
    """Read metadata from a sidecar JSON file."""
    data = json.loads(json_path.read_text())
    return RecordingMetadata(**data)


def update_metadata(json_path: Path, updates: dict) -> RecordingMetadata:
    """Merge updates into an existing sidecar JSON and return the updated metadata."""
    data = json.loads(json_path.read_text())
    data.update(updates)
    meta = RecordingMetadata(**data)
    write_metadata(json_path, meta)
    return meta
