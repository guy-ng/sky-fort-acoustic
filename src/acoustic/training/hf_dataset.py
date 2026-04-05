"""HuggingFace Datasets wrapper for DADS drone audio dataset.

Replaces per-row Parquet I/O with memory-mapped Arrow format for near-instant
random access. The HF dataset is loaded once (downloaded or from cache) and
converted to Arrow on first use.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import mel_spectrogram_from_segment
from acoustic.training.augmentation import SpecAugment
from acoustic.training.parquet_dataset import decode_wav_bytes, split_indices

logger = logging.getLogger(__name__)


class HFDroneDataset(Dataset):
    """PyTorch Dataset backed by a HuggingFace Dataset split.

    Each __getitem__ reads from memory-mapped Arrow (near zero-copy),
    decodes WAV bytes, extracts a random 0.5s segment, and returns
    a mel-spectrogram tensor with its label.

    Interface matches ParquetDataset for drop-in replacement.
    """

    def __init__(
        self,
        hf_dataset,
        split_indices: list[int],
        mel_config: MelConfig | None = None,
        waveform_aug: Callable[[np.ndarray], np.ndarray] | None = None,
        spec_aug: SpecAugment | None = None,
    ) -> None:
        self._hf_ds = hf_dataset
        self._indices = split_indices
        self._mel_config = mel_config or MelConfig()
        self._waveform_aug = waveform_aug
        self._spec_aug = spec_aug
        # Pre-fetch labels via Arrow column (zero-copy) instead of per-row access
        all_labels = list(hf_dataset["label"])
        self._labels_cache = [all_labels[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample from memory-mapped Arrow data.

        Returns:
            (features, label) where features is (1, 128, 64) and label is float32 scalar.
        """
        global_idx = self._indices[idx]
        row = self._hf_ds[global_idx]

        # Decode audio bytes
        audio_struct = row["audio"]
        wav_bytes = audio_struct["bytes"]
        audio = decode_wav_bytes(wav_bytes)

        # Random 0.5s segment extraction
        n = self._mel_config.segment_samples
        if len(audio) >= n:
            start = random.randint(0, len(audio) - n)
            segment = audio[start : start + n]
        else:
            from acoustic.classification.preprocessing import pad_or_loop
            segment = pad_or_loop(audio, n)

        # Waveform augmentation
        if self._waveform_aug is not None:
            segment = self._waveform_aug(segment)

        # Mel spectrogram: returns (1, 1, 128, 64)
        features = mel_spectrogram_from_segment(segment, self._mel_config)
        features = features.squeeze(0)  # (1, 128, 64)

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
        """Total rows in the underlying HF dataset (before splitting)."""
        return len(self._hf_ds)


class HFDatasetBuilder:
    """Loads an HF dataset once and builds split datasets without re-downloading.

    Usage:
        builder = HFDatasetBuilder("geronimobasso/drone-audio-detection-samples")
        train_idx, val_idx, test_idx = split_indices(builder.total_rows)
        train_ds = builder.build(train_idx, mel_config=mel_cfg, waveform_aug=aug)
        val_ds = builder.build(val_idx, mel_config=mel_cfg)
    """

    def __init__(self, repo_id: str) -> None:
        from datasets import Audio, load_dataset

        logger.info("Loading HF dataset: %s", repo_id)
        ds = load_dataset(repo_id, split="train")
        # Disable automatic audio decoding — our code decodes WAV bytes directly
        # via decode_wav_bytes(). This prevents torchcodec/soundfile conflicts
        # and keeps memory usage low (raw bytes stay in Arrow, not decoded).
        ds = ds.cast_column("audio", Audio(decode=False))
        self._hf_ds = ds
        # Use Arrow column access (zero-copy) instead of iterating rows
        self._all_labels = list(ds["label"])
        logger.info("HF dataset loaded: %d rows", len(ds))

    @property
    def total_rows(self) -> int:
        return len(self._hf_ds)

    @property
    def all_labels(self) -> list[int]:
        return self._all_labels

    def build(
        self,
        split_indices: list[int],
        mel_config: MelConfig | None = None,
        waveform_aug: Callable[[np.ndarray], np.ndarray] | None = None,
        spec_aug: SpecAugment | None = None,
    ) -> HFDroneDataset:
        """Create an HFDroneDataset for the given split indices."""
        return HFDroneDataset(
            hf_dataset=self._hf_ds,
            split_indices=split_indices,
            mel_config=mel_config,
            waveform_aug=waveform_aug,
            spec_aug=spec_aug,
        )

    def load_raw_waveforms(
        self, indices: list[int],
    ) -> tuple[list[np.ndarray], list[int]]:
        """Load raw decoded waveforms for given indices (for EfficientAT).

        Returns:
            (waveforms, labels) — list of float32 arrays and int labels.
        """
        waveforms = []
        labels = []
        for i in indices:
            row = self._hf_ds[i]
            wav_bytes = row["audio"]["bytes"]
            audio = decode_wav_bytes(wav_bytes)
            waveforms.append(audio)
            labels.append(int(row["label"]))
        return waveforms, labels
