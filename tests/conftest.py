"""Shared test fixtures for the acoustic service."""

from __future__ import annotations

import os
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Phase 20 Wave 0 fixtures (D-01..D-29)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_waveform() -> np.ndarray:
    """1.0 s mono float32 sine wave at 1 kHz, sampled at 16 kHz, amplitude 0.01.

    Used by Phase 20 augmentation tests as a deterministic input that has
    enough energy to compute SNR ratios but is far below the clipping bound.

    Amplitude 0.01 (RMS ~0.00707) leaves ~43 dB of headroom before the
    [-1, 1] clip in WideGainAugmentation saturates, so a +30 dB gain is
    measurable post-clip (0.01 * 31.6 ~= 0.316, well under 1.0). The earlier
    0.1 amplitude saturated at ~+23 dB and broke
    test_gain_range_uniform's max(observed_db) >= 30 assertion.
    """
    sample_rate = 16000
    duration_s = 1.0
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    return (0.01 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_rir() -> np.ndarray:
    """Tiny synthetic room impulse response: float32 length 800 (50 ms @ 16 kHz).

    Single Dirac at sample 0 followed by exponential decay. Long enough to
    convolve meaningfully but short enough to keep tests fast.
    """
    n = 800
    decay = np.exp(-np.arange(n, dtype=np.float32) / 80.0).astype(np.float32)
    decay[0] = 1.0
    return decay


@pytest.fixture
def temp_noise_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 5 short WAV files for noise mixing tests.

    Each WAV is 0.5s of float32 white noise at 16 kHz mono. Returns the parent
    directory containing the noise files (suitable for BackgroundNoiseMixer
    noise_dirs argument).
    """
    import soundfile as sf

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    sample_rate = 16000
    n_samples = sample_rate // 2  # 0.5 s
    for i in range(5):
        noise = rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
        sf.write(str(noise_dir / f"noise_{i:02d}.wav"), noise, sample_rate)
    return noise_dir
