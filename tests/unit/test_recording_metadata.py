"""Unit tests for RecordingMetadata sidecar JSON CRUD."""

from __future__ import annotations

import json

import pytest

from acoustic.recording.metadata import (
    RecordingMetadata,
    read_metadata,
    update_metadata,
    write_metadata,
)


class TestRecordingMetadata:
    """RecordingMetadata dataclass validation."""

    def test_required_label_field(self):
        meta = RecordingMetadata(label="drone")
        assert meta.label == "drone"

    def test_optional_fields_default_none(self):
        meta = RecordingMetadata(label="background")
        assert meta.sub_label is None
        assert meta.distance_m is None
        assert meta.altitude_m is None
        assert meta.conditions is None
        assert meta.notes is None

    def test_all_fields_populated(self):
        meta = RecordingMetadata(
            label="drone",
            sub_label="Mavic",
            distance_m=50.0,
            altitude_m=30.0,
            conditions="windy",
            notes="test flight",
            recorded_at="2026-04-02T12:00:00Z",
            duration_s=10.5,
            sample_rate=16000,
            channels=1,
            original_sr=48000,
            filename="rec_001.wav",
        )
        assert meta.sub_label == "Mavic"
        assert meta.distance_m == 50.0
        assert meta.duration_s == 10.5


class TestMetadataIO:
    """write_metadata / read_metadata / update_metadata cycle."""

    def test_write_and_read_roundtrip(self, tmp_path):
        json_path = tmp_path / "rec.json"
        meta = RecordingMetadata(
            label="drone",
            duration_s=5.0,
            sample_rate=16000,
            channels=1,
            filename="rec.wav",
        )
        write_metadata(json_path, meta)
        assert json_path.exists()

        loaded = read_metadata(json_path)
        assert loaded.label == "drone"
        assert loaded.duration_s == 5.0
        assert loaded.filename == "rec.wav"

    def test_write_excludes_none_fields(self, tmp_path):
        json_path = tmp_path / "rec.json"
        meta = RecordingMetadata(label="background")
        write_metadata(json_path, meta)

        raw = json.loads(json_path.read_text())
        assert "label" in raw
        assert "sub_label" not in raw
        assert "distance_m" not in raw

    def test_update_merges_fields(self, tmp_path):
        json_path = tmp_path / "rec.json"
        meta = RecordingMetadata(label="", duration_s=3.0, filename="rec.wav")
        write_metadata(json_path, meta)

        updated = update_metadata(json_path, {"label": "drone", "notes": "good capture"})
        assert updated.label == "drone"
        assert updated.notes == "good capture"
        assert updated.duration_s == 3.0  # preserved

        # Verify persisted
        reloaded = read_metadata(json_path)
        assert reloaded.label == "drone"
        assert reloaded.notes == "good capture"

    def test_sidecar_json_is_indented(self, tmp_path):
        json_path = tmp_path / "rec.json"
        meta = RecordingMetadata(label="other")
        write_metadata(json_path, meta)

        text = json_path.read_text()
        # Indented JSON has newlines
        assert "\n" in text
