"""RecordingManager: orchestrates recording sessions, file lifecycle, and metadata CRUD."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from acoustic.recording.config import RecordingConfig
from acoustic.recording.metadata import (
    RecordingMetadata,
    read_metadata,
    update_metadata,
    write_metadata,
)
from acoustic.recording.recorder import RecordingSession


class RecordingManager:
    """Manages the record-first/label-later workflow for field audio collection.

    Thread-safe: feed_chunk() is called from the audio pipeline thread,
    while start/stop/label operations are called from the async API context.
    """

    def __init__(self, config: RecordingConfig) -> None:
        self._config = config
        self._root = Path(config.data_root)
        self._session: RecordingSession | None = None
        self._session_id: str | None = None
        self._session_path: Path | None = None
        self._start_time: float | None = None
        self._lock = threading.Lock()

    def start_recording(self) -> str:
        """Start a new recording session. Returns the recording ID.

        Raises RuntimeError if already recording.
        """
        with self._lock:
            if self._session is not None:
                raise RuntimeError("Already recording")

            now = datetime.now(timezone.utc)
            hex_suffix = uuid.uuid4().hex[:6]
            stem = f"{now.strftime('%Y%m%d_%H%M%S')}_{hex_suffix}"
            rec_id = stem

            unlabeled_dir = self._root / "_unlabeled"
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            wav_path = unlabeled_dir / f"{stem}.wav"

            session = RecordingSession(
                output_path=wav_path,
                source_sr=self._config.source_sample_rate,
                target_sr=self._config.target_sample_rate,
                gain_db=self._config.gain_db,
            )
            session.start()

            self._session = session
            self._session_id = rec_id
            self._session_path = wav_path
            self._start_time = time.monotonic()
            return rec_id

    def stop_recording(self) -> tuple[str, float]:
        """Stop the current recording session. Returns (recording_id, duration_s).

        Raises RuntimeError if not recording.
        """
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> tuple[str, float]:
        """Internal stop implementation. Must be called with self._lock held."""
        if self._session is None:
            raise RuntimeError("Not recording")

        duration = self._session.stop()
        rec_id = self._session_id
        wav_path = self._session_path

        # Write partial metadata (no label yet -- record-first workflow D-01)
        meta = RecordingMetadata(
            label="",
            recorded_at=datetime.now(timezone.utc).isoformat(),
            duration_s=round(duration, 2),
            sample_rate=self._config.target_sample_rate,
            channels=1,
            original_sr=self._config.source_sample_rate,
            filename=wav_path.name,
        )
        write_metadata(wav_path.with_suffix(".json"), meta)

        self._session = None
        self._session_id = None
        self._session_path = None
        self._start_time = None
        return rec_id, duration

    def feed_chunk(self, chunk) -> None:
        """Forward an audio chunk to the active recording session.

        Called by the pipeline thread. Thread-safe via atomic reference read
        and lock escalation only for auto-stop.
        """
        session = self._session  # Atomic reference read
        if session is not None and session.running:
            session.write_chunk(chunk)
            # Auto-stop check
            if session.duration_s >= self._config.max_duration_s:
                with self._lock:
                    if (
                        self._session is not None
                        and self._session.duration_s >= self._config.max_duration_s
                    ):
                        self._stop_locked()

    def label_recording(
        self, rec_id: str, label: str, extra: dict | None = None
    ) -> Path:
        """Assign a label to a recording, moving it from _unlabeled/ to {label}/.

        Args:
            rec_id: Recording identifier (filename stem).
            label: Top-level label (must be in top_labels).
            extra: Additional metadata fields to merge.

        Returns:
            Path to the moved WAV file.

        Raises:
            ValueError: If label is not in top_labels.
            FileNotFoundError: If recording not found in _unlabeled/.
        """
        if label not in self._config.top_labels:
            raise ValueError(
                f"Invalid label '{label}'. Must be one of: {self._config.top_labels}"
            )

        unlabeled_wav = self._root / "_unlabeled" / f"{rec_id}.wav"
        unlabeled_json = self._root / "_unlabeled" / f"{rec_id}.json"
        if not unlabeled_wav.exists():
            raise FileNotFoundError(f"Recording {rec_id} not found in _unlabeled/")

        target_dir = self._root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_wav = target_dir / f"{rec_id}.wav"
        target_json = target_dir / f"{rec_id}.json"

        # Move WAV first
        shutil.move(str(unlabeled_wav), str(target_wav))

        # Update metadata with label + extras, then move JSON
        updates: dict = {"label": label}
        if extra:
            updates.update(extra)
        update_metadata(unlabeled_json, updates)
        shutil.move(str(unlabeled_json), str(target_json))

        # Create Parquet version for training pipeline (D-08)
        label_int = 1 if label == "drone" else 0
        wav_bytes = target_wav.read_bytes()
        parquet_table = pa.table({
            "audio": [{"bytes": wav_bytes, "path": target_wav.name}],
            "label": [label_int],
        })
        parquet_path = target_wav.with_suffix(".parquet")
        pq.write_table(parquet_table, str(parquet_path))

        return target_wav

    def list_recordings(self) -> list[dict]:
        """List all recordings (labeled + unlabeled) with metadata."""
        results: list[dict] = []
        if not self._root.exists():
            return results

        for subdir in sorted(self._root.iterdir()):
            if not subdir.is_dir():
                continue
            for json_file in sorted(subdir.glob("*.json")):
                meta = read_metadata(json_file)
                results.append(
                    {
                        "id": json_file.stem,
                        "label": meta.label if meta.label else "unlabeled",
                        "labeled": bool(meta.label),
                        "directory": subdir.name,
                        **meta.model_dump(exclude_none=True),
                    }
                )
        return results

    def get_recording(self, rec_id: str) -> dict | None:
        """Get a single recording's metadata by ID."""
        if not self._root.exists():
            return None
        for subdir in self._root.iterdir():
            if not subdir.is_dir():
                continue
            json_path = subdir / f"{rec_id}.json"
            if json_path.exists():
                meta = read_metadata(json_path)
                return {
                    "id": rec_id,
                    "directory": subdir.name,
                    **meta.model_dump(exclude_none=True),
                }
        return None

    def update_recording(self, rec_id: str, updates: dict) -> dict | None:
        """Update metadata fields for a recording."""
        if not self._root.exists():
            return None
        for subdir in self._root.iterdir():
            if not subdir.is_dir():
                continue
            json_path = subdir / f"{rec_id}.json"
            if json_path.exists():
                meta = update_metadata(json_path, updates)
                return {
                    "id": rec_id,
                    "directory": subdir.name,
                    **meta.model_dump(exclude_none=True),
                }
        return None

    def delete_recording(self, rec_id: str) -> bool:
        """Delete a recording's WAV and JSON files."""
        if not self._root.exists():
            return False
        for subdir in self._root.iterdir():
            if not subdir.is_dir():
                continue
            wav_path = subdir / f"{rec_id}.wav"
            json_path = subdir / f"{rec_id}.json"
            if wav_path.exists() or json_path.exists():
                wav_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
                return True
        return False

    def get_state(self) -> dict:
        """Get current recording state for WebSocket broadcast."""
        session = self._session
        if session is None or not session.running:
            return {
                "status": "idle",
                "elapsed_s": 0,
                "remaining_s": 0,
                "level_db": -100.0,
            }
        elapsed = session.duration_s
        remaining = max(0, self._config.max_duration_s - elapsed)
        return {
            "status": "recording",
            "elapsed_s": round(float(elapsed), 1),
            "remaining_s": round(float(remaining), 1),
            "level_db": round(float(session.rms_db), 1),
        }
