"""Simulated 16-channel audio source for development without hardware."""

from __future__ import annotations

import numpy as np

from acoustic.config import AcousticSettings

# UMA-16v2 geometry constants
SPACING = 0.042  # meters between adjacent microphones
NUM_CHANNELS = 16


def build_mic_positions() -> np.ndarray:
    """Build UMA-16v2 microphone positions as a (3, 16) array.

    Ported from POC: radar_gui_all_mics_fast_drone.py lines 26-72.

    Coordinate system:
        x: left (-) to right (+)
        y: back (-) to front (+), top row is +y
        z: all zeros (planar array)

    Layout (top view):
        Row 0 (top):    MIC8   MIC7   MIC10  MIC9
        Row 1:          MIC6   MIC5   MIC12  MIC11
        Row 2:          MIC4   MIC3   MIC14  MIC13
        Row 3 (bottom): MIC2   MIC1   MIC16  MIC15
    """
    mic_rc = {
        8: (0, 0), 7: (0, 1), 10: (0, 2), 9: (0, 3),
        6: (1, 0), 5: (1, 1), 12: (1, 2), 11: (1, 3),
        4: (2, 0), 3: (2, 1), 14: (2, 2), 13: (2, 3),
        2: (3, 0), 1: (3, 1), 16: (3, 2), 15: (3, 3),
    }

    xs = np.array([-1.5, -0.5, 0.5, 1.5]) * SPACING
    ys = np.array([+1.5, +0.5, -0.5, -1.5]) * SPACING

    xs_all = []
    ys_all = []
    zs_all = []

    for ch in range(NUM_CHANNELS):
        mic_num = ch + 1
        row, col = mic_rc[mic_num]
        xs_all.append(xs[col])
        ys_all.append(ys[row])
        zs_all.append(0.0)

    return np.vstack([np.array(xs_all), np.array(ys_all), np.array(zs_all)])


def generate_simulated_chunk(
    mic_positions: np.ndarray,
    fs: int,
    chunk_samples: int,
    source_az_deg: float,
    source_el_deg: float,
    freq: float,
    c: float,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Generate a synthetic 16-channel audio chunk with a plane wave from a given direction.

    Args:
        mic_positions: (3, N) array of microphone positions in meters.
        fs: Sample rate in Hz.
        chunk_samples: Number of samples per chunk.
        source_az_deg: Source azimuth in degrees.
        source_el_deg: Source elevation in degrees.
        freq: Frequency of the sine wave in Hz.
        c: Speed of sound in m/s.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        (chunk_samples, N) float32 array.
    """
    n_mics = mic_positions.shape[1]

    # Convert angles to radians
    az_rad = np.radians(source_az_deg)
    el_rad = np.radians(source_el_deg)

    # Direction unit vector (pointing FROM source TO array)
    direction = np.array([
        np.cos(el_rad) * np.sin(az_rad),
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad),
    ])

    # Time delays per microphone: tau_i = (mic_pos . direction) / c
    delays = mic_positions.T @ direction / c  # (N,)

    # Time vector
    t = np.arange(chunk_samples, dtype=np.float64) / fs

    # Generate delayed sine waves: signal[t, mic] = sin(2*pi*freq*(t - delay))
    signal = np.sin(2 * np.pi * freq * (t[:, None] - delays[None, :]))

    # Add Gaussian noise at specified SNR
    signal_power = 0.5  # RMS of sine wave = 1/sqrt(2), power = 0.5
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng()
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape)

    return (signal + noise).astype(np.float32)


class SimulatedAudioSource:
    """Generates synthetic 16-channel audio with configurable direction-of-arrival.

    Used for development and testing when no UMA-16v2 hardware is available.
    """

    def __init__(self, settings: AcousticSettings) -> None:
        self._settings = settings
        self._mic_positions = build_mic_positions()

    def get_chunk(
        self,
        source_az_deg: float = 0.0,
        source_el_deg: float = 0.0,
        freq: float = 500.0,
        snr_db: float = 20.0,
    ) -> np.ndarray:
        """Generate a synthetic audio chunk from the given direction.

        Returns:
            (chunk_samples, num_channels) float32 array.
        """
        return generate_simulated_chunk(
            mic_positions=self._mic_positions,
            fs=self._settings.sample_rate,
            chunk_samples=self._settings.chunk_samples,
            source_az_deg=source_az_deg,
            source_el_deg=source_el_deg,
            freq=freq,
            c=self._settings.speed_of_sound,
            snr_db=snr_db,
        )
