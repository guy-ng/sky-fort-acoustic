"""EfficientAT mel-spectrogram preprocessing configuration.

Parameters match AugmentMelSTFT defaults for pretrained weight compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientATMelConfig:
    """Preprocessing parameters for EfficientAT models.

    Defaults match the AudioSet-pretrained mn10 expectations:
    32kHz sample rate, 128 mel bands, hop=320, win=800, n_fft=1024.
    """

    sample_rate: int = 32000
    n_mels: int = 128
    win_length: int = 800
    hop_size: int = 320
    n_fft: int = 1024
    input_dim_t: int = 100  # ~1s at 32kHz/hop=320

    @property
    def segment_samples(self) -> int:
        """Number of raw audio samples per segment."""
        return self.input_dim_t * self.hop_size  # 32000 = 1s at 32kHz
