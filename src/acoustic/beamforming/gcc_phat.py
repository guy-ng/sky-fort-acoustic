"""GCC-PHAT cross-correlation and FFT preparation.

Ported from POC: radar_gui_all_mics_fast_drone.py
  - prepare_fft (lines 92-118)
  - gcc_phat_from_fft (lines 121-134)
"""

import numpy as np


def prepare_fft(
    signals: np.ndarray,
    fs: int,
    fmin: float = 100.0,
    fmax: float = 2000.0,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Compute FFT once per microphone for all signals.

    Args:
        signals: shape (n_mics, n_samples)
        fs: sample rate in Hz
        fmin: lower frequency bound for band mask
        fmax: upper frequency bound for band mask

    Returns:
        X: shape (n_mics, nfft//2+1) -- rfft of each channel
        nfft: FFT size (next power of 2 >= 2*n_samples)
        max_shift: half FFT size
        band_mask: boolean mask selecting frequencies in [fmin, fmax]
    """
    n_mics, n_samples = signals.shape

    # Next power of 2 >= 2 * n_samples
    nfft = 1
    while nfft < 2 * n_samples:
        nfft *= 2

    # Remove DC per channel
    signals = signals - np.mean(signals, axis=1, keepdims=True)

    # FFT along time axis
    X = np.fft.rfft(signals, n=nfft, axis=1)

    # Frequency vector and band mask
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band_mask = (freqs >= fmin) & (freqs <= fmax)

    max_shift = nfft // 2

    return X, nfft, max_shift, band_mask


def gcc_phat_from_fft(
    Xm: np.ndarray,
    Xn: np.ndarray,
    nfft: int,
    max_shift: int,
    band_mask: np.ndarray,
) -> np.ndarray:
    """Compute GCC-PHAT cross-correlation from precomputed FFTs.

    Uses only frequency bins inside band_mask (drone-focused band).

    Args:
        Xm: FFT of microphone m, shape (nfft//2+1,)
        Xn: FFT of microphone n, shape (nfft//2+1,)
        nfft: FFT size
        max_shift: half FFT size
        band_mask: boolean mask for frequency band of interest

    Returns:
        cc: cross-correlation array, shape (2*max_shift,), centered at max_shift
    """
    # Apply band mask
    Xm_f = Xm * band_mask
    Xn_f = Xn * band_mask

    # Cross-spectrum with PHAT weighting
    R = Xm_f * np.conj(Xn_f)
    R /= np.abs(R) + 1e-12

    # Inverse FFT and rearrange to centered correlation
    cc = np.fft.irfft(R, n=nfft)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))

    return cc
