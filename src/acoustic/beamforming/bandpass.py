"""Bandpass pre-filter for beamforming frequency band enforcement.

Applies a Butterworth bandpass filter (default 500-4000 Hz) to enforce the
spatial aliasing limit of the UMA-16v2 array (BF-10, BF-11). Uses SOS format
for numerical stability and maintains per-channel filter state for streaming.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class BandpassFilter:
    """Streaming bandpass filter with per-channel state tracking.

    Designed for real-time chunk-by-chunk processing of multi-channel audio.
    Filter state is maintained between apply() calls so that consecutive
    chunks produce a continuous filtered signal without transient artifacts.

    Args:
        fs: Sample rate in Hz.
        fmin: Lower cutoff frequency in Hz.
        fmax: Upper cutoff frequency in Hz.
        order: Butterworth filter order (total order; SOS sections = order).
    """

    def __init__(
        self,
        fs: int,
        fmin: float = 500.0,
        fmax: float = 4000.0,
        order: int = 4,
    ) -> None:
        nyq = fs / 2.0
        low = fmin / nyq
        high = fmax / nyq
        self._sos = butter(order, [low, high], btype="band", output="sos")
        self._zi: np.ndarray | None = None
        self._n_channels: int = 0

    def reset(self, n_channels: int) -> None:
        """Reset filter state for the given number of channels.

        Args:
            n_channels: Number of audio channels (e.g. 16 for UMA-16v2).
        """
        zi = sosfilt_zi(self._sos)  # shape (n_sections, 2)
        self._zi = np.repeat(zi[np.newaxis, :, :], n_channels, axis=0)
        self._n_channels = n_channels

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to multi-channel audio.

        Args:
            signals: Audio data, shape (n_mics, n_samples).

        Returns:
            Filtered audio, same shape as input.
        """
        if self._zi is None or signals.shape[0] != self._n_channels:
            self.reset(signals.shape[0])

        filtered = np.empty_like(signals)
        for ch in range(signals.shape[0]):
            filtered[ch], self._zi[ch] = sosfilt(
                self._sos, signals[ch], zi=self._zi[ch]
            )
        return filtered
