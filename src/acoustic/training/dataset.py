"""Training dataset: lazy WAV loading, random segment extraction, augmentation pipeline.

Includes an in-memory audio cache to eliminate repeated disk I/O across epochs.
After the first read (or an explicit warm_cache() call), audio data is served
from RAM, reducing per-sample latency from ~5ms (disk) to ~0.01ms (memory).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import mel_spectrogram_from_segment
from acoustic.training.augmentation import SpecAugment, WaveformAugmentation

logger = logging.getLogger(__name__)


def collect_wav_files(
    data_root: str | Path,
    label_map: dict[str, int],
) -> tuple[list[Path], list[int]]:
    """Scan data_root for labeled WAV files organized in subdirectories.

    Each subdirectory whose name is a key in label_map is scanned for .wav files
    (case-insensitive). Subdirectories not in label_map are skipped.

    Args:
        data_root: Root directory containing label subdirectories.
        label_map: Mapping from subdirectory name to integer label.

    Returns:
        Parallel lists of (file_paths, labels).

    Raises:
        ValueError: If data_root does not exist or no WAV files are found.
    """
    root = Path(data_root)
    if not root.is_dir():
        msg = f"Data root does not exist: {root}"
        raise ValueError(msg)

    file_paths: list[Path] = []
    labels: list[int] = []

    for label_name, label_int in sorted(label_map.items()):
        label_dir = root / label_name
        if not label_dir.is_dir():
            continue
        wavs = sorted(
            p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
        )
        for wav_path in wavs:
            file_paths.append(wav_path)
            labels.append(label_int)

    if not file_paths:
        msg = f"No WAV files found in {root} for label_map keys: {list(label_map.keys())}"
        raise ValueError(msg)

    return file_paths, labels


class DroneAudioDataset(Dataset):
    """PyTorch Dataset for drone audio classification training.

    Loads WAV files, extracts a random 0.5s segment per __getitem__ call,
    applies waveform and spectrogram augmentation, and returns model-ready tensors.

    Audio data is cached in memory after first read to eliminate disk I/O on
    subsequent epochs.  Call warm_cache() to pre-load all files before training.
    """

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        mel_config: MelConfig,
        waveform_aug: WaveformAugmentation | None = None,
        spec_aug: SpecAugment | None = None,
    ) -> None:
        self._paths = file_paths
        self._labels = labels
        self._mel_config = mel_config
        self._waveform_aug = waveform_aug
        self._spec_aug = spec_aug
        # In-memory cache: index -> mono float32 numpy array
        self._audio_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._paths)

    def _load_audio(self, idx: int) -> np.ndarray:
        """Load audio for *idx*, returning cached copy when available."""
        if idx in self._audio_cache:
            return self._audio_cache[idx]

        audio, _sr = sf.read(self._paths[idx], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self._audio_cache[idx] = audio
        return audio

    def warm_cache(self, limit: int = 0) -> None:
        """Pre-load audio files into memory.

        Args:
            limit: Maximum number of files to load.  0 means load all.
        """
        remaining = [i for i in range(len(self._paths)) if i not in self._audio_cache]
        if limit > 0:
            remaining = remaining[:limit]
        for idx in remaining:
            self._load_audio(idx)
        logger.info(
            "DroneAudioDataset cache: %d/%d files loaded (~%.1f MB)",
            len(self._audio_cache), len(self._paths), self.cache_size_mb,
        )

    @property
    def cache_size_mb(self) -> float:
        """Approximate memory used by the audio cache in megabytes."""
        return sum(a.nbytes for a in self._audio_cache.values()) / (1024 * 1024)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load WAV, extract random segment, apply augmentation, return mel-spec + label.

        Returns:
            (features, label) where features is (1, 128, 64) and label is float32 scalar.
        """
        audio = self._load_audio(idx)

        n = self._mel_config.segment_samples

        # Random 0.5s segment extraction
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

        # Remove batch dim -> (1, 128, 64) for spec augmentation
        features = features.squeeze(0)

        # Spectrogram augmentation
        if self._spec_aug is not None:
            features = self._spec_aug(features)

        label_tensor = torch.tensor(self._labels[idx], dtype=torch.float32)
        return features, label_tensor


def build_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances classes by inverse frequency.

    Args:
        labels: List of integer labels.

    Returns:
        WeightedRandomSampler with replacement, num_samples=len(labels).
    """
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
