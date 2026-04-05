"""MCRA (Minima Controlled Recursive Averaging) noise estimator for SRP maps.

Provides an adaptive noise floor that replaces the static percentile-based
threshold (BF-14). The algorithm tracks the minimum of a smoothed power
spectrum to distinguish between signal-present and signal-absent frames,
updating the noise estimate conservatively when signal is detected.

Reference: Israel Cohen, "Noise Spectrum Estimation in Adverse Environments:
Improved Minima Controlled Recursive Averaging", IEEE Trans. Speech and Audio
Processing, 2003.
"""

from __future__ import annotations

import numpy as np


class MCRANoiseEstimator:
    """Adaptive noise floor estimator using Minima Controlled Recursive Averaging.

    Tracks the noise floor of an SRP-PHAT spatial power map over time.
    Signal-present cells are updated slowly (alpha_d) while noise-only cells
    adapt quickly (0.5), allowing the estimator to track non-stationary noise
    without being corrupted by persistent signal peaks.

    Args:
        alpha_s: Smoothing factor for power spectrum S (0-1). Higher = slower.
        alpha_d: Noise update rate when signal is present (0-1). Higher = slower.
        delta: Signal presence threshold — ratio of S/S_min above which
               a cell is considered signal-present.
        min_window: Number of frames between S_min resets. Prevents S_min
                    from tracking a rising noise floor too slowly.
    """

    def __init__(
        self,
        alpha_s: float = 0.8,
        alpha_d: float = 0.95,
        delta: float = 5.0,
        min_window: int = 50,
    ) -> None:
        self._alpha_s = alpha_s
        self._alpha_d = alpha_d
        self._delta = delta
        self._min_window = min_window

        self._frame_count: int = 0
        self._S: np.ndarray | None = None
        self._S_min: np.ndarray | None = None
        self._noise: np.ndarray | None = None

    def update(self, srp_map: np.ndarray) -> np.ndarray:
        """Update the noise estimate with a new SRP map frame.

        Args:
            srp_map: 2D spatial power map, shape (n_az, n_el).

        Returns:
            Noise floor estimate with the same shape as srp_map.
        """
        power = srp_map.ravel()

        # First frame: initialize all state from input
        if self._S is None:
            self._S = power.copy()
            self._S_min = power.copy()
            self._noise = power.copy()
            self._frame_count = 1
            return self._noise.reshape(srp_map.shape)

        # Step 1: Smooth power spectrum
        self._S = self._alpha_s * self._S + (1 - self._alpha_s) * power

        # Step 2: Track minimum and periodically reset
        self._S_min = np.minimum(self._S_min, self._S)
        self._frame_count += 1
        if self._frame_count % self._min_window == 0:
            self._S_min = self._S.copy()

        # Step 3: Signal presence detection
        # Primary: S/S_min ratio (classic MCRA — detects transient signals)
        ratio = self._S / (self._S_min + 1e-10)
        # Secondary: compare each cell against the global median of the
        # smoothed map. This catches persistent signals that S_min cannot
        # distinguish because they were present from the start.
        global_median = np.median(self._S)
        global_ratio = self._S / (global_median + 1e-10)
        signal_present = (ratio > self._delta) | (global_ratio > self._delta)

        # Step 4: Conditional noise update
        # Signal absent  -> adapt noise toward current power (fast)
        # Signal present -> adapt noise toward global median (slow),
        #                   preventing noise from rising to the signal level
        target = np.where(signal_present, global_median, power)
        alpha = np.where(signal_present, self._alpha_d, 0.5)
        self._noise = alpha * self._noise + (1 - alpha) * target

        return self._noise.reshape(srp_map.shape)

    def reset(self) -> None:
        """Clear all internal state. Next update() will re-initialize."""
        self._S = None
        self._S_min = None
        self._noise = None
        self._frame_count = 0
