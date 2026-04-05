"""Unit tests for BandpassFilter and Phase 17 config extensions."""

import numpy as np
import pytest

from acoustic.beamforming.bandpass import BandpassFilter
from acoustic.config import AcousticSettings


class TestBandpassFilter:
    """Tests for the BandpassFilter class."""

    def test_sos_coefficients_shape(self):
        """Test 1: SOS coefficients have correct shape (n_sections, 6)."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        sos = bp._sos
        assert sos.ndim == 2
        assert sos.shape[1] == 6
        # order=4 bandpass -> 4 second-order sections
        assert sos.shape[0] == 4

    def test_below_passband_attenuated(self):
        """Test 2: 250 Hz tone attenuated by >20 dB (below passband)."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        fs = 48000
        n_samples = 48000  # 1 second
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 250 * t)
        signals = signal.reshape(1, -1)
        filtered = bp.apply(signals)
        rms_orig = np.sqrt(np.mean(signal**2))
        rms_filt = np.sqrt(np.mean(filtered[0] ** 2))
        attenuation_db = 20 * np.log10(rms_filt / rms_orig)
        assert attenuation_db < -20, f"Expected >20 dB attenuation, got {attenuation_db:.1f} dB"

    def test_within_passband_passes(self):
        """Test 3: 2000 Hz tone passes with <3 dB attenuation."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        fs = 48000
        n_samples = 48000
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 2000 * t)
        signals = signal.reshape(1, -1)
        filtered = bp.apply(signals)
        rms_orig = np.sqrt(np.mean(signal**2))
        rms_filt = np.sqrt(np.mean(filtered[0] ** 2))
        attenuation_db = 20 * np.log10(rms_filt / rms_orig)
        assert attenuation_db > -3, f"Expected <3 dB attenuation, got {attenuation_db:.1f} dB"

    def test_above_passband_attenuated(self):
        """Test 4: 10000 Hz tone attenuated by >20 dB (well above passband)."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        fs = 48000
        n_samples = 48000
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10000 * t)
        signals = signal.reshape(1, -1)
        filtered = bp.apply(signals)
        rms_orig = np.sqrt(np.mean(signal**2))
        rms_filt = np.sqrt(np.mean(filtered[0] ** 2))
        attenuation_db = 20 * np.log10(rms_filt / rms_orig)
        assert attenuation_db < -20, f"Expected >20 dB attenuation, got {attenuation_db:.1f} dB"

    def test_streaming_state_preserved(self):
        """Test 5: Filter preserves state across chunks -- second call differs from first."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        fs = 48000
        n_samples = 7200  # 150ms chunk
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 1000 * t).reshape(1, -1)
        out1 = bp.apply(signal.copy())
        out2 = bp.apply(signal.copy())
        # With maintained state, second call output should differ from first
        assert not np.allclose(out1, out2), "Second chunk should differ due to filter state"

    def test_reset_clears_state(self):
        """Test 6: reset() clears filter state -- after reset output matches first-call output."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        fs = 48000
        n_samples = 7200
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 1000 * t).reshape(1, -1)
        out_first = bp.apply(signal.copy())
        # Apply again (state changed)
        bp.apply(signal.copy())
        # Reset and apply again
        bp.reset(1)
        out_after_reset = bp.apply(signal.copy())
        np.testing.assert_allclose(out_first, out_after_reset, rtol=1e-10)

    def test_multichannel_input(self):
        """Test 7: apply() works on multi-channel input (16, 7200) shape."""
        bp = BandpassFilter(fs=48000, fmin=500, fmax=4000, order=4)
        signals = np.random.randn(16, 7200)
        filtered = bp.apply(signals)
        assert filtered.shape == (16, 7200)

    def test_config_freq_band_defaults(self):
        """Test 8: Config fields bf_freq_min=500.0, bf_freq_max=4000.0, bf_filter_order=4."""
        settings = AcousticSettings()
        assert settings.bf_freq_min == 500.0
        assert settings.bf_freq_max == 4000.0
        assert settings.bf_filter_order == 4

    def test_config_all_bf_fields(self):
        """Test 9: All bf_* config fields have correct defaults."""
        settings = AcousticSettings()
        assert settings.bf_min_separation_deg == 15.0
        assert settings.bf_max_peaks == 5
        assert settings.bf_peak_threshold == 3.0
        assert settings.bf_mcra_alpha_s == 0.8
        assert settings.bf_mcra_alpha_d == 0.95
        assert settings.bf_mcra_delta == 5.0
        assert settings.bf_mcra_min_window == 50
        assert settings.bf_holdoff_seconds == 5.0
