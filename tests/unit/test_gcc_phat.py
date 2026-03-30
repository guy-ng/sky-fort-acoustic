"""Tests for GCC-PHAT cross-correlation and FFT preparation."""

import numpy as np
import numpy.testing as npt

from acoustic.beamforming.gcc_phat import gcc_phat_from_fft, prepare_fft


class TestPrepareFFT:
    def test_prepare_fft_shape(self):
        """With (16, 7200) input at fs=48000, returns X shape (16, nfft//2+1)
        where nfft is next power of 2 >= 2*7200 = 16384."""
        signals = np.random.randn(16, 7200).astype(np.float64)
        X, nfft, max_shift, band_mask = prepare_fft(signals, fs=48000)
        assert nfft == 16384  # next power of 2 >= 14400
        assert X.shape == (16, nfft // 2 + 1)
        assert max_shift == nfft // 2

    def test_prepare_fft_band_mask(self):
        """band_mask selects only frequencies between 100-2000 Hz."""
        signals = np.random.randn(16, 7200).astype(np.float64)
        X, nfft, max_shift, band_mask = prepare_fft(signals, fs=48000, fmin=100.0, fmax=2000.0)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / 48000)
        # All selected freqs should be in [100, 2000]
        selected_freqs = freqs[band_mask]
        assert len(selected_freqs) > 0
        assert np.all(selected_freqs >= 100.0)
        assert np.all(selected_freqs <= 2000.0)
        # Frequencies outside band should not be selected
        assert not np.any(band_mask[freqs < 100.0])
        assert not np.any(band_mask[freqs > 2000.0])


class TestGccPhat:
    def test_gcc_phat_zero_delay(self):
        """Two identical signals produce GCC-PHAT peak at shift=0."""
        n_samples = 7200
        fs = 48000
        signal = np.random.randn(n_samples)
        signals = np.vstack([signal, signal])
        X, nfft, max_shift, band_mask = prepare_fft(signals, fs)
        cc = gcc_phat_from_fft(X[0], X[1], nfft, max_shift, band_mask)
        # Peak should be at center (shift=0 -> index=max_shift)
        peak_idx = np.argmax(cc)
        assert peak_idx == max_shift, f"Expected peak at {max_shift}, got {peak_idx}"

    def test_gcc_phat_known_delay(self):
        """Signal with known delay produces GCC-PHAT peak at correct shift.

        When Xn is delayed by 1 sample relative to Xm, GCC-PHAT(Xm, Xn)
        peaks at shift=-1 (Xn arrives later). We verify the magnitude.
        """
        n_samples = 7200
        fs = 48000
        signal = np.random.randn(n_samples)
        delayed = np.zeros(n_samples)
        delayed[1:] = signal[:-1]  # Xn delayed by 1 sample
        signals = np.vstack([signal, delayed])
        X, nfft, max_shift, band_mask = prepare_fft(signals, fs)
        cc = gcc_phat_from_fft(X[0], X[1], nfft, max_shift, band_mask)
        peak_idx = np.argmax(cc)
        peak_shift = peak_idx - max_shift
        assert abs(peak_shift) == 1, f"Expected peak at |shift|=1, got {peak_shift}"
