"""DADS Parquet dataset: load audio from Parquet shards without disk extraction.

Implements a PyTorch Dataset that reads WAV bytes directly from Parquet files,
decodes them in-memory, extracts random 0.5s segments, and produces mel-spectrogram
tensors compatible with the existing training pipeline.

Includes an in-memory audio cache to eliminate repeated Parquet I/O across epochs.
"""

from __future__ import annotations

import bisect
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import mel_spectrogram_from_segment
from acoustic.training.augmentation import SpecAugment, WaveformAugmentation

logger = logging.getLogger(__name__)


def split_indices(
    total: int,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic train/val/test index splits.

    Shuffles range(total) with a fixed seed, then slices into three non-overlapping
    partitions at approximately 70/15/15 ratio.

    Args:
        total: Total number of samples.
        seed: Random seed for reproducibility.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.

    Returns:
        (train_indices, val_indices, test_indices) as lists of ints.
    """
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    n_train = int(total * train_frac)
    n_val = int(total * val_frac)

    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]

    return train, val, test


def decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 audio array without writing to disk.

    Skips the standard 44-byte WAV header and interprets the remaining bytes
    as int16 PCM, normalizing to [-1.0, 1.0].

    Args:
        wav_bytes: Raw WAV file bytes (44-byte header + PCM data).

    Returns:
        1-D float32 numpy array with values in [-1.0, 1.0].
    """
    return np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0


class ParquetDataset(Dataset):
    """PyTorch Dataset that loads DADS audio from Parquet shards.

    Each __getitem__ call reads a single row from the appropriate shard,
    decodes WAV bytes, extracts a random 0.5s segment, and returns a
    mel-spectrogram tensor with its label.

    Designed to be constructed via ParquetDatasetBuilder to avoid repeated
    shard scanning when creating train/val/test splits.
    """

    def __init__(
        self,
        shards: list[Path],
        shard_offsets: list[int],
        all_labels: list[int],
        split_indices: list[int],
        mel_config: MelConfig | None = None,
        waveform_aug: WaveformAugmentation | None = None,
        spec_aug: SpecAugment | None = None,
    ) -> None:
        self._shards = shards
        self._shard_offsets = shard_offsets
        self._all_labels = all_labels
        self._indices = split_indices
        self._mel_config = mel_config or MelConfig()
        self._waveform_aug = waveform_aug
        self._spec_aug = spec_aug
        self._labels_cache = [all_labels[i] for i in split_indices]
        # In-memory cache: global_idx -> decoded float32 numpy array
        self._audio_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._indices)

    def _locate(self, global_idx: int) -> tuple[int, int]:
        """Map a global row index to (shard_index, local_row_index).

        Uses bisect on cumulative shard offsets for O(log n) lookup.
        """
        shard_idx = bisect.bisect_right(self._shard_offsets, global_idx) - 1
        local_idx = global_idx - self._shard_offsets[shard_idx]
        return shard_idx, local_idx

    def _load_audio(self, global_idx: int) -> np.ndarray:
        """Load audio for *global_idx*, returning cached copy when available."""
        if global_idx in self._audio_cache:
            return self._audio_cache[global_idx]

        shard_idx, local_idx = self._locate(global_idx)
        table = pq.read_table(self._shards[shard_idx], columns=["audio"])
        audio_struct = table.column("audio")[local_idx].as_py()
        wav_bytes = audio_struct["bytes"]
        audio = decode_wav_bytes(wav_bytes)

        self._audio_cache[global_idx] = audio
        return audio

    def warm_cache(self, limit: int = 0) -> None:
        """Pre-load audio samples into memory, reading each shard once.

        Args:
            limit: Maximum number of samples to load.  0 means load all.
                   Use a small value (e.g. 1000) to quickly prime the cache
                   before training starts, then call again with 0 from a
                   background thread to finish the rest.
        """
        needed = [i for i in self._indices if i not in self._audio_cache]
        if not needed:
            return
        if limit > 0:
            needed = needed[:limit]

        # Group needed indices by shard for sequential bulk reads
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for global_idx in needed:
            shard_idx, local_idx = self._locate(global_idx)
            by_shard.setdefault(shard_idx, []).append((global_idx, local_idx))

        loaded = 0
        for shard_idx, pairs in by_shard.items():
            table = pq.read_table(self._shards[shard_idx], columns=["audio"])
            audio_col = table.column("audio")
            for global_idx, local_idx in pairs:
                wav_bytes = audio_col[local_idx].as_py()["bytes"]
                self._audio_cache[global_idx] = decode_wav_bytes(wav_bytes)
                loaded += 1

        logger.info(
            "ParquetDataset cache: %d/%d samples loaded (~%.1f MB)",
            len(self._audio_cache), len(self._indices), self.cache_size_mb,
        )

    @property
    def cache_size_mb(self) -> float:
        """Approximate memory used by the audio cache in megabytes."""
        return sum(a.nbytes for a in self._audio_cache.values()) / (1024 * 1024)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample: decode WAV from Parquet, extract segment, return mel-spec.

        Returns:
            (features, label) where features is (1, 128, 64) and label is float32 scalar.
        """
        global_idx = self._indices[idx]
        audio = self._load_audio(global_idx)

        # Random segment extraction
        n = self._mel_config.segment_samples
        if len(audio) >= n:
            start = random.randint(0, len(audio) - n)
            segment = audio[start : start + n]
        else:
            # Zero-pad short audio
            segment = np.zeros(n, dtype=np.float32)
            segment[: len(audio)] = audio

        # Waveform augmentation
        if self._waveform_aug is not None:
            segment = self._waveform_aug(segment)

        # Mel spectrogram: returns (1, 1, 128, 64)
        features = mel_spectrogram_from_segment(segment, self._mel_config)

        # Remove batch dim -> (1, 128, 64)
        features = features.squeeze(0)

        # Spectrogram augmentation
        if self._spec_aug is not None:
            features = self._spec_aug(features)

        label_tensor = torch.tensor(self._labels_cache[idx], dtype=torch.float32)
        return features, label_tensor

    @property
    def labels(self) -> list[int]:
        """Integer labels for the split indices (needed by build_weighted_sampler)."""
        return self._labels_cache

    @property
    def total_rows(self) -> int:
        """Total rows across all shards (before splitting)."""
        return len(self._all_labels)


class ParquetDatasetBuilder:
    """Lightweight factory that scans Parquet shards once and builds split datasets.

    Avoids scanning shards 3 times (once per train/val/test split) by
    pre-computing shard metadata on construction.
    """

    def __init__(self, data_dir: str | Path) -> None:
        data_path = Path(data_dir)
        self._shards = sorted(data_path.glob("train-*.parquet"))

        if not self._shards:
            msg = f"No Parquet shards found in {data_path}"
            raise ValueError(msg)

        # Read only label column from each shard to build global index
        self._shard_offsets: list[int] = []
        self._all_labels: list[int] = []
        offset = 0

        for shard_path in self._shards:
            self._shard_offsets.append(offset)
            table = pq.read_table(shard_path, columns=["label"])
            labels = table.column("label").to_pylist()
            self._all_labels.extend(labels)
            offset += len(labels)

    @property
    def total_rows(self) -> int:
        """Total rows across all shards."""
        return len(self._all_labels)

    @property
    def all_labels(self) -> list[int]:
        """All labels in shard order."""
        return self._all_labels

    def build(
        self,
        split_indices: list[int],
        mel_config: MelConfig | None = None,
        waveform_aug: WaveformAugmentation | None = None,
        spec_aug: SpecAugment | None = None,
    ) -> ParquetDataset:
        """Create a ParquetDataset for the given split indices without re-scanning shards."""
        return ParquetDataset(
            shards=self._shards,
            shard_offsets=self._shard_offsets,
            all_labels=self._all_labels,
            split_indices=split_indices,
            mel_config=mel_config,
            waveform_aug=waveform_aug,
            spec_aug=spec_aug,
        )
