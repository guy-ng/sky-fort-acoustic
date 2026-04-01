"""Data augmentation for drone audio training: waveform-level and spectrogram-level."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T


class WaveformAugmentation:
    """Waveform-level augmentation: Gaussian noise injection + random gain.

    Applied to raw audio segments before mel-spectrogram conversion (D-10).
    """

    def __init__(
        self,
        snr_range: tuple[float, float] = (10.0, 40.0),
        gain_db: float = 6.0,
    ) -> None:
        self._snr_range = snr_range
        self._gain_db = gain_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise injection and gain scaling to a 1-D float32 audio segment.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Augmented audio as float32 array of same length.
        """
        rng = np.random.default_rng()
        out = audio.copy()

        # Gaussian noise at random SNR
        signal_power = float(np.mean(out**2))
        if signal_power > 1e-10:
            snr_db = rng.uniform(self._snr_range[0], self._snr_range[1])
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0, np.sqrt(noise_power), size=out.shape).astype(
                np.float32
            )
            out = out + noise

        # Random gain in +/- gain_db
        gain_db = rng.uniform(-self._gain_db, self._gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        out = out * gain_linear

        return out.astype(np.float32)


class SpecAugment:
    """Spectrogram-level augmentation: time and frequency masking (D-09).

    Uses torchaudio TimeMasking and FrequencyMasking transforms.
    """

    def __init__(
        self,
        time_mask_param: int = 20,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        self._time_mask_param = time_mask_param
        self._freq_mask_param = freq_mask_param
        self._num_time_masks = num_time_masks
        self._num_freq_masks = num_freq_masks

        # Build mask transforms (only if params > 0)
        self._time_masks: list[T.TimeMasking] = []
        self._freq_masks: list[T.FrequencyMasking] = []
        if time_mask_param > 0:
            self._time_masks = [
                T.TimeMasking(time_mask_param=time_mask_param)
                for _ in range(num_time_masks)
            ]
        if freq_mask_param > 0:
            self._freq_masks = [
                T.FrequencyMasking(freq_mask_param=freq_mask_param)
                for _ in range(num_freq_masks)
            ]

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masking to a spectrogram.

        Args:
            spectrogram: Tensor of shape (1, time, freq) = (1, 128, 64).

        Returns:
            Masked spectrogram of same shape (1, 128, 64).
        """
        if not self._time_masks and not self._freq_masks:
            return spectrogram

        # torchaudio masks expect (..., freq, time) layout
        # Input is (1, time=128, freq=64) -> transpose to (1, freq=64, time=128)
        spec = spectrogram.transpose(-2, -1)

        for mask in self._freq_masks:
            spec = mask(spec)
        for mask in self._time_masks:
            spec = mask(spec)

        # Transpose back to (1, time=128, freq=64)
        return spec.transpose(-2, -1)
