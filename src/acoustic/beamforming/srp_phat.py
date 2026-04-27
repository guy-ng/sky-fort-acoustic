"""2D SRP-PHAT beamforming engine.

Extended from POC's srp_phat_1d_fast (lines 137-181) to scan over
both azimuth and elevation angles using 2D steering vectors.
"""

import itertools

import numpy as np

from acoustic.beamforming.gcc_phat import gcc_phat_from_fft, prepare_fft
from acoustic.beamforming.geometry import build_steering_vectors_2d


def srp_phat_2d(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    c: float,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    fmin: float = 500.0,
    fmax: float = 4000.0,
) -> np.ndarray:
    """2D SRP-PHAT beamforming.

    Scans over all (azimuth, elevation) directions using GCC-PHAT
    cross-correlations for all microphone pairs.

    Args:
        signals: shape (n_mics, n_samples) -- transposed from sounddevice format
        mic_positions: shape (3, n_mics) -- [x, y, z] per mic
        fs: sample rate in Hz
        c: speed of sound in m/s
        az_grid_deg: 1D array of azimuth angles in degrees
        el_grid_deg: 1D array of elevation angles in degrees
        fmin: lower frequency bound for band filtering
        fmax: upper frequency bound for band filtering

    Returns:
        np.ndarray: shape (n_az, n_el) spatial power map
    """
    n_mics, n_samples = signals.shape

    # Build 2D direction vectors
    dirs = build_steering_vectors_2d(az_grid_deg, el_grid_deg)
    n_dirs = dirs.shape[0]

    # Compute max physical TDOA from array geometry
    # Find the largest baseline (distance between any two mics)
    max_baseline = 0.0
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            d = np.linalg.norm(mic_positions[:, i] - mic_positions[:, j])
            if d > max_baseline:
                max_baseline = d
    max_tdoa_samples = int(np.ceil(max_baseline / c * fs)) + 1

    # FFT once per mic, with physically bounded correlation window
    X, nfft, max_shift, band_mask = prepare_fft(
        signals, fs, fmin, fmax, max_tdoa_samples=max_tdoa_samples,
    )

    # Accumulate SRP power for each direction
    srp = np.zeros(n_dirs, dtype=np.float64)

    # Iterate over all mic pairs
    pairs = list(itertools.combinations(range(n_mics), 2))
    for m, n in pairs:
        cc = gcc_phat_from_fft(X[m], X[n], nfft, max_shift, band_mask)

        # Mic pair displacement
        delta_p = mic_positions[:, m] - mic_positions[:, n]  # (3,)

        # Predicted TDOA for each direction
        tdoa_pred = dirs @ delta_p / c
        shift_pred = np.round(tdoa_pred * fs).astype(int)
        shift_pred = np.clip(shift_pred, -max_shift + 1, max_shift - 1)

        # Accumulate correlation at predicted shift
        srp += cc[shift_pred + max_shift]

    return srp.reshape(len(az_grid_deg), len(el_grid_deg))
