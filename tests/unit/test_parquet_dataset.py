"""Unit tests for ParquetDataset: DADS Parquet shard loading, WAV decoding, splitting."""

from __future__ import annotations

import struct
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from acoustic.training.parquet_dataset import (
    ParquetDataset,
    ParquetDatasetBuilder,
    decode_wav_bytes,
    split_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(num_samples: int = 16000, sample_rate: int = 16000) -> bytes:
    """Create valid WAV bytes: 44-byte header + int16 PCM mono data."""
    rng = np.random.default_rng(42)
    samples = (rng.uniform(-0.5, 0.5, num_samples) * 32767).astype(np.int16)
    pcm_bytes = samples.tobytes()
    data_size = len(pcm_bytes)
    # Standard 44-byte WAV header (PCM, mono, 16-bit)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,      # ChunkSize
        b"WAVE",
        b"fmt ",
        16,                  # Subchunk1Size (PCM)
        1,                   # AudioFormat (PCM)
        1,                   # NumChannels
        sample_rate,         # SampleRate
        sample_rate * 2,     # ByteRate (sr * channels * bits/8)
        2,                   # BlockAlign (channels * bits/8)
        16,                  # BitsPerSample
        b"data",
        data_size,           # Subchunk2Size
    )
    return header + pcm_bytes


def _make_parquet_shard(path: Path, num_rows: int, label_value: int, sr: int = 16000) -> None:
    """Create a single DADS-format Parquet shard with audio struct and label columns."""
    audio_structs = []
    for i in range(num_rows):
        wav = _make_wav_bytes(num_samples=sr, sample_rate=sr)  # 1 second of audio
        audio_structs.append({"bytes": wav, "path": f"sample_{i}.wav"})

    table = pa.table({
        "audio": audio_structs,
        "label": [label_value] * num_rows,
    })
    pq.write_table(table, path)


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    """Create synthetic DADS Parquet shards: 3 shards with 10, 8, 12 rows = 30 total."""
    _make_parquet_shard(tmp_path / "train-00000-of-00003.parquet", num_rows=10, label_value=1)
    _make_parquet_shard(tmp_path / "train-00001-of-00003.parquet", num_rows=8, label_value=0)
    _make_parquet_shard(tmp_path / "train-00002-of-00003.parquet", num_rows=12, label_value=1)
    return tmp_path


# ---------------------------------------------------------------------------
# split_indices tests
# ---------------------------------------------------------------------------

class TestSplitIndices:
    """Tests for the split_indices function."""

    def test_sizes_sum_to_total(self):
        train, val, test = split_indices(100, seed=42)
        assert len(train) + len(val) + len(test) == 100

    def test_approximate_proportions(self):
        train, val, test = split_indices(100, seed=42)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self):
        train, val, test = split_indices(100, seed=42)
        all_indices = set(train) | set(val) | set(test)
        assert len(all_indices) == 100

    def test_deterministic_same_seed(self):
        a = split_indices(100, seed=42)
        b = split_indices(100, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = split_indices(100, seed=42)
        b = split_indices(100, seed=99)
        assert a != b

    def test_small_total(self):
        """Edge case: small dataset still produces valid splits."""
        train, val, test = split_indices(5, seed=42)
        assert len(train) + len(val) + len(test) == 5


# ---------------------------------------------------------------------------
# decode_wav_bytes tests
# ---------------------------------------------------------------------------

class TestDecodeWavBytes:
    """Tests for WAV byte decoding."""

    def test_produces_float32(self):
        wav = _make_wav_bytes(num_samples=100)
        audio = decode_wav_bytes(wav)
        assert audio.dtype == np.float32

    def test_values_in_range(self):
        wav = _make_wav_bytes(num_samples=1000)
        audio = decode_wav_bytes(wav)
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_correct_length(self):
        wav = _make_wav_bytes(num_samples=500)
        audio = decode_wav_bytes(wav)
        assert len(audio) == 500


# ---------------------------------------------------------------------------
# ParquetDataset tests
# ---------------------------------------------------------------------------

class TestParquetDataset:
    """Tests for ParquetDataset loading and __getitem__."""

    def test_total_rows(self, parquet_dir: Path):
        builder = ParquetDatasetBuilder(parquet_dir)
        assert builder.total_rows == 30  # 10 + 8 + 12

    def test_len_matches_split(self, parquet_dir: Path):
        builder = ParquetDatasetBuilder(parquet_dir)
        indices = list(range(10))
        ds = builder.build(split_indices=indices)
        assert len(ds) == 10

    def test_getitem_shape_and_dtype(self, parquet_dir: Path):
        builder = ParquetDatasetBuilder(parquet_dir)
        ds = builder.build(split_indices=[0, 1, 2])
        features, label = ds[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1, 128, 64)
        assert label.dtype == torch.float32
        assert label.item() in (0.0, 1.0)

    def test_labels_property(self, parquet_dir: Path):
        builder = ParquetDatasetBuilder(parquet_dir)
        # Shard 0 has 10 rows with label=1, shard 1 has 8 rows with label=0
        # Index 0 -> label=1, index 10 -> label=0
        ds = builder.build(split_indices=[0, 10])
        assert ds.labels == [1, 0]

    def test_locate_first_row(self, parquet_dir: Path):
        builder = ParquetDatasetBuilder(parquet_dir)
        ds = builder.build(split_indices=list(range(30)))
        shard_idx, local_idx = ds._locate(0)
        assert shard_idx == 0
        assert local_idx == 0

    def test_locate_shard_boundary(self, parquet_dir: Path):
        """Last row of shard 0 (idx=9) and first row of shard 1 (idx=10)."""
        builder = ParquetDatasetBuilder(parquet_dir)
        ds = builder.build(split_indices=list(range(30)))
        # Last row of shard 0
        shard_idx, local_idx = ds._locate(9)
        assert shard_idx == 0
        assert local_idx == 9
        # First row of shard 1
        shard_idx, local_idx = ds._locate(10)
        assert shard_idx == 1
        assert local_idx == 0

    def test_locate_last_shard_boundary(self, parquet_dir: Path):
        """Last row of shard 1 (idx=17) and first row of shard 2 (idx=18)."""
        builder = ParquetDatasetBuilder(parquet_dir)
        ds = builder.build(split_indices=list(range(30)))
        shard_idx, local_idx = ds._locate(17)
        assert shard_idx == 1
        assert local_idx == 7
        shard_idx, local_idx = ds._locate(18)
        assert shard_idx == 2
        assert local_idx == 0

    def test_short_audio_zero_padded(self, tmp_path: Path):
        """Audio shorter than segment_samples (8000) is zero-padded without error."""
        # Create shard with very short audio (500 samples < 8000)
        wav = _make_wav_bytes(num_samples=500, sample_rate=16000)
        audio_structs = [{"bytes": wav, "path": "short.wav"}]
        table = pa.table({"audio": audio_structs, "label": [1]})
        pq.write_table(table, tmp_path / "train-00000-of-00001.parquet")

        builder = ParquetDatasetBuilder(tmp_path)
        ds = builder.build(split_indices=[0])
        features, label = ds[0]
        # Should not raise, shape should be correct
        assert features.shape == (1, 128, 64)
        assert label.item() == 1.0
