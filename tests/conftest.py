"""Shared test fixtures for the acoustic service."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pytest

from acoustic.config import AcousticSettings


@pytest.fixture
def settings() -> AcousticSettings:
    """Return AcousticSettings with default values."""
    return AcousticSettings()


@pytest.fixture
def mic_positions() -> np.ndarray:
    """Return UMA-16v2 mic positions as (3, 16) array.

    Lazy import to avoid circular dependency — build_mic_positions is defined
    in src/acoustic/audio/simulator.py which is created in Task 2.
    """
    from acoustic.audio.simulator import build_mic_positions

    return build_mic_positions()


@pytest.fixture
def synthetic_audio(settings: AcousticSettings) -> Callable[..., np.ndarray]:
    """Factory fixture that generates synthetic 16-channel audio with a plane wave.

    Returns a callable: (az_deg, el_deg, freq) -> (chunk_samples, 16) float32 array.
    """

    def _generate(
        az_deg: float = 0.0,
        el_deg: float = 0.0,
        freq: float = 500.0,
        snr_db: float = 20.0,
    ) -> np.ndarray:
        from acoustic.audio.simulator import generate_simulated_chunk, build_mic_positions

        positions = build_mic_positions()
        return generate_simulated_chunk(
            mic_positions=positions,
            fs=settings.sample_rate,
            chunk_samples=settings.chunk_samples,
            source_az_deg=az_deg,
            source_el_deg=el_deg,
            freq=freq,
            c=settings.speed_of_sound,
            snr_db=snr_db,
        )

    return _generate


@pytest.fixture
def wav_audio_fixture():
    """Load a short WAV snippet from audio-data/ for deterministic tests.

    Skipped if audio-data/ is not present (CI environments).
    """
    audio_data_dir = os.path.join(os.path.dirname(__file__), "..", "audio-data", "data", "background")
    if not os.path.isdir(audio_data_dir):
        pytest.skip("audio-data/ directory not available (CI)")

    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    # Find first available WAV file
    wav_files = sorted(f for f in os.listdir(audio_data_dir) if f.endswith(".wav"))
    if not wav_files:
        pytest.skip("No WAV files found in audio-data/data/background/")

    # Read up to 4 WAV files and stack them
    channels = []
    for wav_file in wav_files[:4]:
        data, sr = sf.read(os.path.join(audio_data_dir, wav_file), dtype="float32")
        if data.ndim == 1:
            channels.append(data)
        else:
            channels.append(data[:, 0])

    # Stack into (samples, n_channels) — trim to shortest
    min_len = min(len(c) for c in channels)
    stacked = np.column_stack([c[:min_len] for c in channels])
    return stacked
