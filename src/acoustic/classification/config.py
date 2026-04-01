"""Research-validated mel-spectrogram preprocessing parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MelConfig:
    """Research-validated mel-spectrogram preprocessing parameters.

    All values match Acoustic-UAV-Identification train_strong_cnn.py.
    """

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    max_frames: int = 128
    segment_seconds: float = 0.5
    db_range: float = 80.0

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)
