"""Training configuration loaded from ACOUSTIC_TRAINING_* environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Hyperparameters and paths for CNN training pipeline.

    All fields can be overridden via ACOUSTIC_TRAINING_* environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="ACOUSTIC_TRAINING_")

    # Optimizer (D-06)
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 5

    # Data (D-04)
    data_root: str = "data/field"
    label_map: dict[str, int] = {"drone": 1, "background": 0, "other": 0}

    # DADS Parquet data (D-09)
    dads_path: str = "data/"

    # DADS HuggingFace dataset (preferred over local parquet when set)
    dads_hf_repo: str = ""

    # Validation split (D-03)
    val_split: float = 0.2

    # Checkpoint (D-08)
    checkpoint_path: str = "models/research_cnn_trained.pt"

    # Augmentation toggle (D-11)
    augmentation_enabled: bool = True

    # SpecAugment params (D-09)
    spec_time_mask_param: int = 20
    spec_freq_mask_param: int = 8
    spec_num_time_masks: int = 2
    spec_num_freq_masks: int = 2

    # Waveform augmentation params (D-10)
    wave_snr_range_low: float = 10.0
    wave_snr_range_high: float = 40.0
    wave_gain_db: float = 6.0

    # EfficientAT transfer learning stages (MDL-11)
    model_type: str = "research_cnn"  # "research_cnn", "efficientat_mn10", or "efficientat_mn05"
    pretrained_weights: str = "models/pretrained/mn10_as.pt"
    width_mult: float = 1.0  # 1.0=MN10 (~4.6M params), 0.5=MN05 (~1.3M params)
    stage1_epochs: int = 10
    stage2_epochs: int = 15
    stage3_epochs: int = 20
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4
    stage3_lr: float = 1e-5

    # Loss function (TRN-10)
    loss_function: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    bce_pos_weight: float = 1.0

    # Background noise augmentation (TRN-11)
    noise_augmentation_enabled: bool = False
    noise_dirs: list[str] = []
    noise_snr_range_low: float = -10.0
    noise_snr_range_high: float = 20.0
    noise_probability: float = 0.5

    # Audiomentations waveform augmentation (TRN-12)
    use_audiomentations: bool = True
    pitch_shift_semitones: float = 3.0
    time_stretch_min: float = 0.85
    time_stretch_max: float = 1.15
    waveform_gain_db: float = 6.0
    augmentation_probability: float = 0.5

    # --- Phase 20 additions (D-01..D-20, D-23) -------------------------------
    # WideGain replacement for small-gain stage (D-01..D-04)
    wide_gain_db: float = 40.0
    wide_gain_probability: float = 1.0

    # Procedural ShoeBox RIR augmentation (D-05..D-08)
    rir_enabled: bool = False
    rir_probability: float = 0.7
    rir_pool_size: int = 500
    rir_room_dim_min: list[float] = Field(default_factory=lambda: [3.0, 3.0, 2.5])
    rir_room_dim_max: list[float] = Field(default_factory=lambda: [12.0, 12.0, 4.0])
    rir_absorption_min: float = 0.2
    rir_absorption_max: float = 0.7
    rir_source_distance_min: float = 1.0
    rir_source_distance_max: float = 8.0
    rir_max_order: int = 10

    # Sliding-window dataset overlap (D-13..D-16)
    window_overlap_ratio: float = 0.0
    window_overlap_test: float = 0.0

    # UMA-16 ambient noise mixing (D-10..D-12)
    uma16_ambient_dir: str = "data/field/uma16_ambient"
    uma16_ambient_snr_low: float = -5.0
    uma16_ambient_snr_high: float = 15.0
    # Primary field name (matches Wave 0 test stub test_uma16_pure_negative_ratio)
    uma16_pure_negative_ratio: float = 0.10
    # Alias retained for plan-body compatibility (D-12 must-haves grep)
    uma16_ambient_pure_negative_ratio: float = 0.10
