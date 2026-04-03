"""Integration tests: ParquetDataset -> TrainingRunner wiring, recording Parquet output."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from acoustic.training.config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(num_samples: int = 16000, sample_rate: int = 16000) -> bytes:
    """Create valid WAV bytes: 44-byte header + int16 PCM mono data."""
    rng = np.random.default_rng(42)
    samples = (rng.uniform(-0.5, 0.5, num_samples) * 32767).astype(np.int16)
    pcm_bytes = samples.tobytes()
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _make_parquet_shard(path: Path, num_rows: int, label_value: int, sr: int = 16000) -> None:
    """Create a single DADS-format Parquet shard."""
    audio_structs = []
    for i in range(num_rows):
        wav = _make_wav_bytes(num_samples=sr, sample_rate=sr)
        audio_structs.append({"bytes": wav, "path": f"sample_{i}.wav"})

    table = pa.table({
        "audio": audio_structs,
        "label": [label_value] * num_rows,
    })
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestTrainingConfigDadsPath:
    def test_training_config_has_dads_path(self) -> None:
        cfg = TrainingConfig()
        assert cfg.dads_path == "data/"

    def test_training_config_dads_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACOUSTIC_TRAINING_DADS_PATH", "/tmp/test")
        cfg = TrainingConfig()
        assert cfg.dads_path == "/tmp/test"


# ---------------------------------------------------------------------------
# Trainer wiring tests
# ---------------------------------------------------------------------------


class TestTrainerParquetBranch:
    def test_trainer_uses_parquet_when_shards_exist(self, tmp_path: Path) -> None:
        """When dads_path contains Parquet shards, TrainingRunner loads ParquetDataset."""
        # Create 2 shards with 10 rows each (mixed labels)
        _make_parquet_shard(tmp_path / "train-00000.parquet", 10, label_value=1)
        _make_parquet_shard(tmp_path / "train-00001.parquet", 10, label_value=0)

        cfg = TrainingConfig(dads_path=str(tmp_path), max_epochs=1, batch_size=4)

        from acoustic.training.trainer import TrainingRunner
        import threading

        runner = TrainingRunner(cfg)
        stop = threading.Event()

        # Run training -- should use Parquet path and complete 1 epoch
        result = runner.run(stop)
        # Result is the checkpoint path (should exist since training ran)
        assert result is not None
        assert result.exists()

    def test_trainer_falls_back_to_wav(self, tmp_path: Path) -> None:
        """When dads_path has no shards, TrainingRunner falls back to WAV path."""
        cfg = TrainingConfig(
            dads_path=str(tmp_path / "nonexistent"),
            data_root=str(tmp_path / "also_nonexistent"),
        )

        from acoustic.training.trainer import TrainingRunner
        import threading

        runner = TrainingRunner(cfg)
        stop = threading.Event()

        # Should fall back to WAV path and raise ValueError from collect_wav_files
        with pytest.raises(ValueError, match="Data root does not exist"):
            runner.run(stop)


# ---------------------------------------------------------------------------
# Recording Parquet output tests
# ---------------------------------------------------------------------------


class TestRecordingSessionToParquet:
    def test_recording_session_to_parquet(self, tmp_path: Path) -> None:
        """RecordingSession.to_parquet() creates a DADS-compatible Parquet file."""
        from acoustic.recording.recorder import RecordingSession

        wav_path = tmp_path / "test_rec.wav"
        session = RecordingSession(output_path=wav_path, source_sr=48000, target_sr=16000)
        session.start()

        # Write a chunk of 48kHz 16-channel audio (0.1s)
        chunk = np.random.randn(4800, 16).astype(np.float32) * 0.1
        session.write_chunk(chunk)
        session.stop()

        assert wav_path.exists()

        # Convert to Parquet
        parquet_path = session.to_parquet(label=1)
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

        # Verify schema
        table = pq.read_table(str(parquet_path))
        assert "audio" in table.column_names
        assert "label" in table.column_names
        assert table.num_rows == 1
        assert table.column("label")[0].as_py() == 1

        audio_struct = table.column("audio")[0].as_py()
        assert "bytes" in audio_struct
        assert "path" in audio_struct
        assert audio_struct["path"] == "test_rec.wav"

    def test_recording_session_path_property(self, tmp_path: Path) -> None:
        """RecordingSession.path returns the output WAV path."""
        from acoustic.recording.recorder import RecordingSession

        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        assert session.path == wav_path


class TestLabelRecordingCreatesParquet:
    def test_label_recording_creates_parquet(self, tmp_path: Path) -> None:
        """RecordingManager.label_recording() creates .parquet alongside .wav."""
        from acoustic.recording.config import RecordingConfig
        from acoustic.recording.manager import RecordingManager
        from acoustic.recording.recorder import RecordingSession

        config = RecordingConfig(data_root=str(tmp_path))
        manager = RecordingManager(config)

        # Start and stop a recording
        rec_id = manager.start_recording()

        # Feed a small chunk
        chunk = np.random.randn(4800, 16).astype(np.float32) * 0.1
        manager.feed_chunk(chunk)

        rec_id, duration = manager.stop_recording()

        # Label it
        target_wav = manager.label_recording(rec_id, "drone")

        # Verify Parquet file created alongside WAV
        parquet_path = target_wav.with_suffix(".parquet")
        assert parquet_path.exists(), f"Expected Parquet at {parquet_path}"

        # Verify schema
        table = pq.read_table(str(parquet_path))
        assert table.num_rows == 1
        assert table.column("label")[0].as_py() == 1  # drone = 1
        audio_struct = table.column("audio")[0].as_py()
        assert "bytes" in audio_struct

    def test_label_recording_background_label_zero(self, tmp_path: Path) -> None:
        """Background label produces label_int=0 in Parquet."""
        from acoustic.recording.config import RecordingConfig
        from acoustic.recording.manager import RecordingManager

        config = RecordingConfig(data_root=str(tmp_path))
        manager = RecordingManager(config)

        rec_id = manager.start_recording()
        chunk = np.random.randn(4800, 16).astype(np.float32) * 0.1
        manager.feed_chunk(chunk)
        rec_id, _ = manager.stop_recording()

        target_wav = manager.label_recording(rec_id, "background")
        parquet_path = target_wav.with_suffix(".parquet")
        assert parquet_path.exists()

        table = pq.read_table(str(parquet_path))
        assert table.column("label")[0].as_py() == 0  # background = 0
