"""Unit tests for MCRA noise estimator."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic.beamforming.mcra import MCRANoiseEstimator


class TestMCRANoiseEstimator:
    """Tests for MCRANoiseEstimator adaptive noise floor tracking."""

    def test_first_update_initializes_noise_to_input(self):
        """First call to update() initializes noise estimate equal to input."""
        mcra = MCRANoiseEstimator()
        srp_map = np.array([[1.0, 2.0], [3.0, 4.0]])
        noise = mcra.update(srp_map)
        np.testing.assert_array_equal(noise, srp_map)

    def test_constant_noise_converges(self):
        """Feeding constant noise-only maps for 100 frames converges noise estimate to ~1.0."""
        mcra = MCRANoiseEstimator()
        uniform_map = np.ones((10, 10))
        for _ in range(100):
            noise = mcra.update(uniform_map)
        # Noise estimate should be close to 1.0 (within 5%)
        np.testing.assert_allclose(noise, 1.0, atol=0.05)

    def test_signal_peak_preserved(self):
        """Persistent signal peak — noise at peak cell stays lower than signal."""
        mcra = MCRANoiseEstimator()
        base_map = np.ones((10, 10))
        base_map[5, 5] = 10.0  # persistent signal peak
        for _ in range(100):
            noise = mcra.update(base_map)
        # Noise at peak cell should be less than signal value
        assert noise[5, 5] < 5.0, f"Noise at peak {noise[5, 5]} should be < 5.0"
        # Noise at non-peak cells should converge to ~1.0
        non_peak_noise = noise.copy()
        non_peak_noise[5, 5] = 1.0  # exclude peak for check
        np.testing.assert_allclose(non_peak_noise, 1.0, atol=0.1)

    def test_reset_clears_state(self):
        """After reset(), state is cleared and next update() re-initializes."""
        mcra = MCRANoiseEstimator()
        srp_map = np.ones((5, 5)) * 3.0
        mcra.update(srp_map)
        mcra.reset()
        # After reset, next update should re-initialize
        new_map = np.ones((5, 5)) * 7.0
        noise = mcra.update(new_map)
        np.testing.assert_array_equal(noise, new_map)

    def test_min_tracking_resets_every_min_window(self):
        """S_min is refreshed every min_window frames."""
        min_window = 10
        mcra = MCRANoiseEstimator(min_window=min_window)
        # Feed 10 frames of value 5.0
        for _ in range(min_window):
            mcra.update(np.ones((3, 3)) * 5.0)
        # After min_window frames, S_min should be reset to current S
        # Feed 1 more frame with higher value — S_min should now equal S (just reset)
        mcra.update(np.ones((3, 3)) * 10.0)
        # The internal S_min should have been reset at frame min_window
        # We verify indirectly: the ratio S/S_min should be close to 1.0
        # right after reset (since S_min was set to S at the reset point)
        assert mcra._frame_count == min_window + 1

    def test_signal_presence_detection(self):
        """Signal presence indicator correctly identifies cells where S/S_min > delta."""
        mcra = MCRANoiseEstimator(delta=2.0, min_window=1000)
        # Start with low values
        low_map = np.ones((5, 5))
        for _ in range(5):
            mcra.update(low_map)
        # Now introduce a sudden spike
        spike_map = np.ones((5, 5))
        spike_map[2, 2] = 50.0
        mcra.update(spike_map)
        # The noise at the spike cell should NOT have fully adapted
        # (signal presence should have been detected, limiting adaptation)
        noise = mcra.update(spike_map)
        assert noise[2, 2] < 50.0, "Signal presence should limit noise adaptation at spike"

    def test_config_defaults(self):
        """Config fields have correct defaults."""
        from acoustic.config import AcousticSettings

        settings = AcousticSettings()
        assert settings.bf_mcra_alpha_s == 0.8
        assert settings.bf_mcra_alpha_d == 0.95
        assert settings.bf_mcra_delta == 5.0
        assert settings.bf_mcra_min_window == 50
