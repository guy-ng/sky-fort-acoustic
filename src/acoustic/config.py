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

    # Frequency band for drone detection (legacy, kept for backward compat)
    freq_min: float = 100.0
    freq_max: float = 2000.0

    # Beamforming frequency band (BF-10, BF-11) - upgraded from 100-2000 Hz
    bf_freq_min: float = 500.0
    bf_freq_max: float = 4000.0
    bf_filter_order: int = 4

    # Multi-peak detection (BF-13)
    bf_min_separation_deg: float = 15.0
    bf_max_peaks: int = 5
    bf_peak_threshold: float = 3.0

    # VIZ-02: Functional beamforming exponent for sidelobe suppression
    bf_nu: float = 100.0

    # MCRA noise estimation (BF-14)
    bf_mcra_alpha_s: float = 0.8
    bf_mcra_alpha_d: float = 0.95
    bf_mcra_delta: float = 5.0
    bf_mcra_min_window: int = 50

    # Demand-driven activation (BF-16)
    bf_holdoff_seconds: float = 5.0

    # Direction of Arrival (Phase 18)
    mounting_orientation: str = "vertical_y_up"
    doa_smoothing_alpha: float = 1.0  # EMA alpha: 1.0 = no smoothing, lower = smoother (D-05)
    doa_association_threshold_deg: float = 7.5  # Max angular distance for peak-to-target match (D-03)

    # WebSocket target broadcast rate (DIR-02, D-07)
    ws_targets_hz: float = 2.0  # /ws/targets update rate in Hz. Range: 0.5-10 Hz.

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

    # CNN classification
    cnn_model_path: str = "models/uav_melspec_cnn.onnx"
    cnn_model_type: str = "research_cnn"
    cnn_enter_threshold: float = 0.80
    cnn_exit_threshold: float = 0.40
    cnn_confirm_hits: int = 2
    cnn_target_ttl: float = 5.0  # seconds before target marked lost

    # Weighted aggregator for segment probabilities
    cnn_agg_w_max: float = 0.5
    cnn_agg_w_mean: float = 0.5

    # CNN inference cadence.
    # `cnn_interval_seconds` — how often the pipeline pushes a new audio window
    # to the classifier. The window LENGTH is NOT configurable — it must always
    # match what the model was trained on, so it is derived from the model type
    # at session start (research_cnn → 0.5 s, efficientat/mn10/mn05 → 1.0 s).
    # Default 0.2 s gives a 5 Hz inference rate.
    cnn_interval_seconds: float = 0.2

    # CNN input gain — DEPRECATED legacy mic calibration knob. Phase 20 D-34
    # replaces per-chunk gain scaling with per-chunk RMS normalization
    # (``cnn_rms_normalize_target``), which makes the gain knob a no-op for
    # correctness. The field is preserved for backwards compatibility and
    # still applied before normalization, so setting it affects the debug dump
    # only. Default is now 1.0 (was 500.0 pre-D-34). See D-34 +
    # .planning/debug/training-collapse-constant-output.md.
    cnn_input_gain: float = 1.0

    # D-34: per-sample RMS normalization target. Applied as the LAST step of
    # ``RawAudioPreprocessor.process()`` AND as the LAST step of the trainer's
    # augmentation chain (both train and eval splits), so the model sees
    # identical amplitude distributions at train and inference time. Closes
    # the ~50x domain shift documented in
    # ``scripts/verify_rms_domain_mismatch.py``.
    cnn_rms_normalize_target: float = 0.1

    # CNN silence gate — reject chunks whose mono RMS is below this.
    # Tuned for UMA-16v2 MEMS mic ambient (~8e-5 RMS / −82 dBFS). The old
    # 1e-3 default (−60 dBFS) was 20 dB too aggressive and rejected every
    # chunk from the live array, preventing any CNN inference. 1e-5 catches
    # only truly dead signal (all-zero / disconnected stream).
    cnn_silence_threshold: float = 1.0e-5

    # Ensemble classifier config file (optional; if set, overrides single model)
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
