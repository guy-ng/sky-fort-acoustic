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

    # Beamforming map normalization (POC logic)
    ignore_origin_deg: float = 3.5  # Suppress broadside artifact within this radius of (0,0)

    # Multi-peak detection (BF-13)
    bf_min_separation_deg: float = 15.0
    bf_max_peaks: int = 5
    bf_peak_threshold: float = 3.0

    # MCRA noise estimation (BF-14)
    bf_mcra_alpha_s: float = 0.8
    bf_mcra_alpha_d: float = 0.95
    bf_mcra_delta: float = 5.0
    bf_mcra_min_window: int = 50

    # Demand-driven activation (BF-16)
    bf_holdoff_seconds: float = 5.0

    # CNN classification
    cnn_model_type: str = "research_cnn"
    cnn_model_path: str = "models/uav_melspec_cnn.pt"
    cnn_enter_threshold: float = 0.80
    cnn_exit_threshold: float = 0.40
    cnn_confirm_hits: int = 2
    cnn_target_ttl: float = 5.0  # seconds before target marked lost

    # Aggregation weights for multi-segment classification
    cnn_agg_w_max: float = 0.5
    cnn_agg_w_mean: float = 0.5

    # Ensemble configuration
    ensemble_config_path: str | None = None

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
