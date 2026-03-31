"""Application configuration loaded from environment variables with ACOUSTIC_ prefix."""

from __future__ import annotations

import math

from pydantic_settings import BaseSettings, SettingsConfigDict


class AcousticSettings(BaseSettings):
    """Settings for the acoustic service, loaded from ACOUSTIC_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="ACOUSTIC_")

    # Audio device
    audio_device: str | None = None
    audio_source: str = "hardware"

    # Audio parameters
    sample_rate: int = 48000
    num_channels: int = 16
    chunk_seconds: float = 0.15

    # Frequency band for drone detection
    freq_min: float = 100.0
    freq_max: float = 2000.0

    # Beamforming grid
    az_range: float = 90.0
    el_range: float = 45.0
    az_resolution: float = 1.0
    el_resolution: float = 1.0

    # Noise threshold
    noise_percentile: float = 95.0
    noise_margin: float = 1.5

    # CNN classification
    cnn_model_path: str = "models/uav_melspec_cnn.onnx"
    cnn_enter_threshold: float = 0.80
    cnn_exit_threshold: float = 0.40
    cnn_confirm_hits: int = 2
    cnn_target_ttl: float = 5.0  # seconds before target marked lost

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Physics
    speed_of_sound: float = 343.0

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk: sample_rate * chunk_seconds."""
        return int(self.sample_rate * self.chunk_seconds)

    @property
    def ring_chunks(self) -> int:
        """Number of chunks in the ring buffer (~2 seconds of audio per D-03)."""
        return int(math.ceil(2.0 / self.chunk_seconds))
