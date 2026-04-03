"""Recording configuration loaded from ACOUSTIC_RECORDING_* environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RecordingConfig(BaseSettings):
    """Configuration for field audio recording sessions.

    All fields can be overridden via ACOUSTIC_RECORDING_* environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="ACOUSTIC_RECORDING_")

    data_root: str = "data/field"
    max_duration_s: float = 300.0
    target_sample_rate: int = 16000
    source_sample_rate: int = 48000
    gain_db: float = 20.0
    top_labels: list[str] = ["drone", "background", "other"]
