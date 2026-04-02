"""Unit tests for RecordingManager: lifecycle, auto-stop, label workflow, compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acoustic.recording.config import RecordingConfig
from acoustic.recording.manager import RecordingManager
from acoustic.recording.metadata import read_metadata


def _make_chunk(duration_s: float = 0.15, sr: int = 48000, channels: int = 16) -> np.ndarray:
    """Synthetic 16-channel audio chunk."""
    samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, samples, dtype=np.float32)
    signal = np.sin(2 * np.pi * 440 * t)
    return np.tile(signal[:, np.newaxis], (1, channels))


@pytest.fixture()
def config(tmp_path) -> RecordingConfig:
    return RecordingConfig(data_root=str(tmp_path / "field"))


@pytest.fixture()
def manager(config) -> RecordingManager:
    return RecordingManager(config)


class TestStartStop:
    def test_start_stop_creates_wav_and_json(self, manager, config):
        rec_id = manager.start_recording()
        chunk = _make_chunk()
        manager.feed_chunk(chunk)
        returned_id, duration = manager.stop_recording()

        assert returned_id == rec_id
        assert duration > 0

        unlabeled = Path(config.data_root) / "_unlabeled"
        wav = unlabeled / f"{rec_id}.wav"
        json_f = unlabeled / f"{rec_id}.json"
        assert wav.exists()
        assert json_f.exists()

        # WAV is valid
        data, sr = sf.read(str(wav))
        assert sr == 16000
        assert data.ndim == 1

    def test_double_start_raises(self, manager):
        manager.start_recording()
        with pytest.raises(RuntimeError, match="Already recording"):
            manager.start_recording()
        manager.stop_recording()

    def test_stop_without_start_raises(self, manager):
        with pytest.raises(RuntimeError, match="Not recording"):
            manager.stop_recording()


class TestAutoStop:
    def test_auto_stop_on_max_duration(self, tmp_path):
        config = RecordingConfig(data_root=str(tmp_path / "field"), max_duration_s=0.1)
        mgr = RecordingManager(config)
        mgr.start_recording()

        # Feed enough chunks to exceed 0.1s
        for _ in range(10):
            mgr.feed_chunk(_make_chunk(duration_s=0.15))

        # After auto-stop, session should be None
        state = mgr.get_state()
        assert state["status"] == "idle"


class TestLabelWorkflow:
    def test_label_moves_files(self, manager, config):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        result_path = manager.label_recording(rec_id, "drone")

        drone_dir = Path(config.data_root) / "drone"
        assert (drone_dir / f"{rec_id}.wav").exists()
        assert (drone_dir / f"{rec_id}.json").exists()

        # Original unlabeled files should be gone
        unlabeled = Path(config.data_root) / "_unlabeled"
        assert not (unlabeled / f"{rec_id}.wav").exists()
        assert not (unlabeled / f"{rec_id}.json").exists()

        # Metadata has label
        meta = read_metadata(drone_dir / f"{rec_id}.json")
        assert meta.label == "drone"

    def test_label_invalid_raises(self, manager):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        with pytest.raises(ValueError, match="Invalid label"):
            manager.label_recording(rec_id, "invalid_label")

    def test_label_with_extra_metadata(self, manager, config):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        manager.label_recording(rec_id, "drone", extra={"sub_label": "Mavic", "distance_m": 50.0})

        meta = read_metadata(Path(config.data_root) / "drone" / f"{rec_id}.json")
        assert meta.label == "drone"
        assert meta.sub_label == "Mavic"
        assert meta.distance_m == 50.0


class TestCRUD:
    def test_list_recordings(self, manager):
        rec1 = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        rec2 = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        recordings = manager.list_recordings()
        assert len(recordings) == 2
        ids = {r["id"] for r in recordings}
        assert rec1 in ids
        assert rec2 in ids

    def test_get_recording(self, manager):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        rec = manager.get_recording(rec_id)
        assert rec is not None
        assert rec["id"] == rec_id

    def test_get_recording_not_found(self, manager):
        assert manager.get_recording("nonexistent") is None

    def test_update_recording(self, manager):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        updated = manager.update_recording(rec_id, {"notes": "updated note"})
        assert updated is not None
        assert updated["notes"] == "updated note"

    def test_delete_recording(self, manager, config):
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        assert manager.delete_recording(rec_id)
        assert not (Path(config.data_root) / "_unlabeled" / f"{rec_id}.wav").exists()
        assert not (Path(config.data_root) / "_unlabeled" / f"{rec_id}.json").exists()

    def test_delete_nonexistent_returns_false(self, manager):
        assert not manager.delete_recording("nonexistent")


class TestState:
    def test_get_state_idle(self, manager):
        state = manager.get_state()
        assert state["status"] == "idle"
        assert state["elapsed_s"] == 0
        assert state["level_db"] == -100.0

    def test_get_state_while_recording(self, manager):
        manager.start_recording()
        manager.feed_chunk(_make_chunk())

        state = manager.get_state()
        assert state["status"] == "recording"
        assert state["elapsed_s"] > 0
        assert state["remaining_s"] > 0
        assert state["level_db"] > -100.0

        manager.stop_recording()


class TestTrainingPipelineCompatibility:
    def test_collect_wav_files_compatibility(self, manager, config):
        """Labeled field recordings are discoverable by collect_wav_files()."""
        # Import the training pipeline scanner
        # Note: We import from the main repo's training module
        # For this test, we replicate the core scan logic to avoid torch dependency
        rec_id = manager.start_recording()
        manager.feed_chunk(_make_chunk())
        manager.stop_recording()

        manager.label_recording(rec_id, "drone")

        # Replicate collect_wav_files logic
        root = Path(config.data_root)
        label_map = {"drone": 1, "background": 0, "other": 0}
        found_files = []
        found_labels = []
        for label_name, label_int in sorted(label_map.items()):
            label_dir = root / label_name
            if not label_dir.is_dir():
                continue
            wavs = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
            for wav_path in wavs:
                found_files.append(wav_path)
                found_labels.append(label_int)

        assert len(found_files) == 1
        assert found_labels == [1]
        assert found_files[0].name == f"{rec_id}.wav"

        # Verify the WAV is valid training input
        data, sr = sf.read(str(found_files[0]))
        assert sr == 16000
        assert data.ndim == 1  # mono
