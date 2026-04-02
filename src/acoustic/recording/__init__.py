"""Recording module: audio capture sessions, metadata, and recording management."""

from acoustic.recording.config import RecordingConfig
from acoustic.recording.metadata import (
    RecordingMetadata,
    read_metadata,
    update_metadata,
    write_metadata,
)
from acoustic.recording.recorder import RecordingSession

__all__ = [
    "RecordingConfig",
    "RecordingManager",
    "RecordingMetadata",
    "RecordingSession",
    "read_metadata",
    "update_metadata",
    "write_metadata",
]


def __getattr__(name: str):
    """Lazy import RecordingManager to avoid circular imports."""
    if name == "RecordingManager":
        from acoustic.recording.manager import RecordingManager

        return RecordingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
