"""Training configuration loaded from ACOUSTIC_TRAINING_* environment variables."""

from __future__ import annotations

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
    model_type: str = "research_cnn"  # "research_cnn" or "efficientat_mn10"
    pretrained_weights: str = "models/pretrained/mn10_as.pt"
    stage1_epochs: int = 10
    stage2_epochs: int = 15
    stage3_epochs: int = 20
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4
    stage3_lr: float = 1e-5
