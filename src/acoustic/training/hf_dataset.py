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
import torchaudio.functional as F_audio
from torch.utils.data import Dataset

from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import (
    mel_spectrogram_from_segment,
    pad_or_loop,
)
from acoustic.training.augmentation import SpecAugment
from acoustic.training.parquet_dataset import decode_wav_bytes, split_indices

logger = logging.getLogger(__name__)

# Sample rate constants for the EfficientAT path. DADS source is 16 kHz; the
# EfficientAT mn10 model expects 32 kHz waveforms (matches AudioSet pretrain).
_SOURCE_SR = 16000
_TARGET_SR = 32000


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


# ---------------------------------------------------------------------------
# Phase 20 D-13/D-14/D-15/D-16: Sliding-window dataset
# ---------------------------------------------------------------------------


class WindowedHFDroneDataset(Dataset):
    """Sliding-window dataset for Phase 20 v7 EfficientAT training (D-13..D-16).

    Returns ``(raw_waveform_32k, label)`` pairs — NOT mel spectrograms. The
    EfficientAT trainer applies its own ``AugmentMelSTFT`` (mel + spec-aug)
    per-batch on device, so this dataset hands it raw audio at the EfficientAT
    sample rate (32 kHz). Quick task 260407-ls3 fixed an earlier version that
    returned mel features, which crashed ``mel_train(batch_wav)`` with
    ``RuntimeError: Expected 2D/3D input to conv1d, got [B,1,1,128,64]``.

    CRITICAL: must be constructed with FILE INDICES (from
    ``parquet_dataset.split_file_indices``), not window indices, to preserve
    session-level split isolation (D-15). Adjacent overlapping windows from the
    same source file would otherwise leak across splits and inflate val/test
    metrics by 10-20% (compass doc §4 "Data splitting: session-level grouping
    is non-negotiable"; Plotz 2021).

    Window math (in 16 kHz source-domain samples):
        num_windows_per_file = max(1, 1 + (n_samples - window_samples) // hop_samples)

    For DADS uniform 1s @ 16k clips, window=8000, hop=3200:
        num_windows = 1 + (16000-8000)//3200 = 3 windows per file (60% overlap, D-13/D-14)

    For test split with hop=8000 (no overlap, D-16):
        num_windows = 1 + (16000-8000)//8000 = 2 windows per file

    Each window is sliced at 16 kHz then resampled to 32 kHz before return,
    yielding length ``window_samples * 2`` (e.g. 16000 samples = 0.5 s @ 32 kHz).
    """

    def __init__(
        self,
        hf_dataset,
        file_indices: list[int],
        window_samples: int = 8000,
        hop_samples: int = 3200,
        waveform_aug: Callable[[np.ndarray], np.ndarray] | None = None,
        assumed_clip_samples: int = 16000,
        sample_rate: int = 16000,
    ) -> None:
        # Duplicate file indices would silently break session-level isolation
        assert len(set(file_indices)) == len(file_indices), (
            "duplicate file indices not allowed — would break session-level split"
        )

        self._hf_ds = hf_dataset
        self._window_samples = int(window_samples)
        self._hop_samples = int(hop_samples)
        self._waveform_aug = waveform_aug
        self._sr = int(sample_rate)
        self._assumed_clip_samples = int(assumed_clip_samples)
        # One-shot warning flag — DADS contains rare outlier clips that violate
        # the uniform-length assumption (e.g. file_idx=174151 in the v7 corpus
        # decodes to 8000 samples instead of 16000). We tolerate them in
        # __getitem__ via pad_or_loop / truncation rather than crashing
        # training, but warn once so the divergence is visible in logs.
        self._warned_non_uniform = False

        # Build flat index list of (file_idx, window_offset)
        # _items[k] = (file_idx, byte-offset-in-samples) for the k-th window
        self._items: list[tuple[int, int]] = []
        self._labels_cache: list[int] = []

        # Pre-fetch labels via Arrow column (zero-copy) instead of per-row access
        all_labels = list(hf_dataset["label"])

        n = self._assumed_clip_samples
        # Number of sliding windows per (uniform-length) clip
        num_w = max(1, 1 + max(0, (n - self._window_samples)) // self._hop_samples)

        for file_idx in file_indices:
            label_int = int(all_labels[file_idx])
            for w in range(num_w):
                self._items.append((int(file_idx), w * self._hop_samples))
                self._labels_cache.append(label_int)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (raw_waveform_32k, label) tuple for the idx-th window."""
        file_idx, offset = self._items[idx]
        row = self._hf_ds[file_idx]

        # Decode WAV bytes (same path as HFDroneDataset)
        audio_struct = row["audio"]
        wav_bytes = audio_struct["bytes"]
        audio = decode_wav_bytes(wav_bytes)

        # Tolerate non-uniform clips (Research A1 fallback). The DADS corpus
        # contains rare outlier files whose decoded length deviates from the
        # uniform 1 s assumption (e.g. file_idx=174151 in v7 decodes to 8000
        # samples). Pre-computed window offsets in self._items assume
        # _assumed_clip_samples, so we reconcile the actual audio against that
        # assumption rather than crash mid-training.
        if len(audio) != self._assumed_clip_samples:
            if not self._warned_non_uniform:
                logger.warning(
                    "WindowedHFDroneDataset: non-uniform clip detected "
                    "(file_idx=%d, %d samples vs assumed %d). "
                    "Falling back to pad_or_loop/truncate; suppressing further "
                    "warnings for this dataset instance.",
                    file_idx,
                    len(audio),
                    self._assumed_clip_samples,
                )
                self._warned_non_uniform = True
            if len(audio) < self._assumed_clip_samples:
                # Tile the short clip up to the assumed length so every
                # pre-allocated (file_idx, offset) window contains real audio
                # rather than zero padding.
                audio = pad_or_loop(audio, self._assumed_clip_samples)
            else:
                # Longer-than-expected: truncate so window offsets stay valid.
                audio = audio[: self._assumed_clip_samples]

        # Slice the deterministic window (in 16 kHz source-domain samples)
        segment = audio[offset : offset + self._window_samples]
        if len(segment) < self._window_samples:
            pad = np.zeros(self._window_samples - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, pad])

        # Waveform augmentation at 16 kHz BEFORE resampling. Augmentations like
        # WideGain / RoomIR / BackgroundNoiseMixer expect source-rate audio.
        if self._waveform_aug is not None:
            segment = self._waveform_aug(segment)

        # Resample 16 kHz → 32 kHz for EfficientAT. The trainer's AugmentMelSTFT
        # will compute mel + spec-aug per-batch on device; this dataset hands
        # back raw waveform only. Quick task 260407-ls3 — see class docstring.
        segment_t = torch.from_numpy(np.ascontiguousarray(segment, dtype=np.float32))
        segment_t = F_audio.resample(segment_t, _SOURCE_SR, _TARGET_SR)

        label_tensor = torch.tensor(self._labels_cache[idx], dtype=torch.float32)
        return segment_t, label_tensor

    @property
    def labels(self) -> list[int]:
        """Integer labels for every flat (file, window) item — needed for samplers."""
        return self._labels_cache

    @property
    def total_rows(self) -> int:
        """Total rows in the underlying HF dataset (before windowing/splitting)."""
        return len(self._hf_ds)
