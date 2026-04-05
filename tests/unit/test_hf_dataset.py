"""Unit tests for HFDroneDataset: HuggingFace Datasets wrapper for DADS training."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from acoustic.training.hf_dataset import HFDatasetBuilder, HFDroneDataset
from acoustic.training.parquet_dataset import split_indices


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
        16, 1, 1,
        sample_rate,
        sample_rate * 2,
        2, 16,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _make_fake_hf_dataset(num_rows: int = 20) -> list[dict]:
    """Create a list of dicts mimicking HF dataset rows."""
    rows = []
    for i in range(num_rows):
        wav_bytes = _make_wav_bytes(num_samples=16000, sample_rate=16000)
        rows.append({
            "audio": {"bytes": wav_bytes, "path": f"sample_{i}.wav"},
            "label": 1 if i % 2 == 0 else 0,
        })
    return rows


class FakeHFDataset:
    """Minimal mock of a HuggingFace Dataset object."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]

    def __iter__(self):
        return iter(self._rows)


@pytest.fixture
def fake_hf_ds() -> FakeHFDataset:
    """20-row fake HF dataset with alternating labels."""
    return FakeHFDataset(_make_fake_hf_dataset(20))


# ---------------------------------------------------------------------------
# HFDroneDataset tests
# ---------------------------------------------------------------------------

class TestHFDroneDataset:
    """Tests for HFDroneDataset loading and __getitem__."""

    def test_len_matches_split(self, fake_hf_ds):
        indices = list(range(10))
        ds = HFDroneDataset(fake_hf_ds, split_indices=indices)
        assert len(ds) == 10

    def test_getitem_shape_and_dtype(self, fake_hf_ds):
        ds = HFDroneDataset(fake_hf_ds, split_indices=[0, 1, 2])
        features, label = ds[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1, 128, 64)
        assert label.dtype == torch.float32
        assert label.item() in (0.0, 1.0)

    def test_labels_property(self, fake_hf_ds):
        # Even indices have label=1, odd have label=0
        ds = HFDroneDataset(fake_hf_ds, split_indices=[0, 1, 2, 3])
        assert ds.labels == [1, 0, 1, 0]

    def test_total_rows(self, fake_hf_ds):
        ds = HFDroneDataset(fake_hf_ds, split_indices=[0, 1])
        assert ds.total_rows == 20

    def test_waveform_augmentation_called(self, fake_hf_ds):
        aug_called = []

        def mock_aug(segment: np.ndarray) -> np.ndarray:
            aug_called.append(True)
            return segment

        ds = HFDroneDataset(fake_hf_ds, split_indices=[0], waveform_aug=mock_aug)
        ds[0]
        assert len(aug_called) == 1

    def test_short_audio_zero_padded(self):
        """Audio shorter than segment_samples is zero-padded."""
        short_wav = _make_wav_bytes(num_samples=500, sample_rate=16000)
        rows = [{"audio": {"bytes": short_wav, "path": "short.wav"}, "label": 1}]
        fake_ds = FakeHFDataset(rows)
        ds = HFDroneDataset(fake_ds, split_indices=[0])
        features, label = ds[0]
        assert features.shape == (1, 128, 64)
        assert label.item() == 1.0


# ---------------------------------------------------------------------------
# HFDatasetBuilder tests
# ---------------------------------------------------------------------------

class TestHFDatasetBuilder:
    """Tests for HFDatasetBuilder factory."""

    def test_builder_loads_and_builds(self, fake_hf_ds):
        builder = HFDatasetBuilder.__new__(HFDatasetBuilder)
        builder._hf_ds = fake_hf_ds
        builder._all_labels = [int(row["label"]) for row in fake_hf_ds]

        assert builder.total_rows == 20
        assert len(builder.all_labels) == 20

        # Build a split
        ds = builder.build(split_indices=[0, 2, 4, 6])
        assert len(ds) == 4
        features, label = ds[0]
        assert features.shape == (1, 128, 64)

    def test_load_raw_waveforms(self, fake_hf_ds):
        builder = HFDatasetBuilder.__new__(HFDatasetBuilder)
        builder._hf_ds = fake_hf_ds
        builder._all_labels = [int(row["label"]) for row in fake_hf_ds]

        waveforms, labels = builder.load_raw_waveforms([0, 1, 2])
        assert len(waveforms) == 3
        assert len(labels) == 3
        assert all(isinstance(w, np.ndarray) for w in waveforms)
        assert all(w.dtype == np.float32 for w in waveforms)
        assert labels == [1, 0, 1]

    def test_split_indices_integration(self, fake_hf_ds):
        """Verify split_indices works with builder total_rows."""
        builder = HFDatasetBuilder.__new__(HFDatasetBuilder)
        builder._hf_ds = fake_hf_ds
        builder._all_labels = [int(row["label"]) for row in fake_hf_ds]

        train_idx, val_idx, test_idx = split_indices(builder.total_rows, seed=42)
        assert len(train_idx) + len(val_idx) + len(test_idx) == 20

        train_ds = builder.build(train_idx)
        val_ds = builder.build(val_idx)
        assert len(train_ds) == len(train_idx)
        assert len(val_ds) == len(val_idx)
