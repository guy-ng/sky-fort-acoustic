"""Three-stage transfer learning trainer for EfficientAT mn10.

Stage 1: Head-only (lr=1e-3) — learns task-specific features.
Stage 2: Head + last 3 blocks (lr=1e-4, CosineAnnealingLR) — fine-tunes high-level features.
Stage 3: All layers (lr=1e-5, CosineAnnealingLR) — full model fine-tuning.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from acoustic.classification.efficientat.config import EfficientATMelConfig
from acoustic.classification.efficientat.model import get_model
from acoustic.classification.efficientat.preprocess import AugmentMelSTFT
from acoustic.training.augmentation import (
    AudiomentationsAugmentation,
    BackgroundNoiseMixer,
    ComposedAugmentation,
    RoomIRAugmentation,
    WideGainAugmentation,
)
from acoustic.training.config import TrainingConfig
from acoustic.training.hf_dataset import WindowedHFDroneDataset
from acoustic.training.losses import build_loss_function
from acoustic.training.parquet_dataset import split_file_indices
from acoustic.training.trainer import EarlyStopping

logger = logging.getLogger(__name__)

_SOURCE_SR = 16000  # DADS/field data sample rate
_TARGET_SR = 32000  # EfficientAT expected sample rate


class _EfficientATDataset(Dataset):
    """Dataset wrapper that serves raw waveforms at 32kHz for EfficientAT.

    Handles resampling from 16kHz source audio and random segment extraction.
    Mel spectrogram computation is deferred to AugmentMelSTFT in the training loop
    so that SpecAugment (freqm/timem) is applied per-batch on device.
    """

    def __init__(
        self,
        waveforms: list[np.ndarray],
        labels: list[int],
        segment_samples: int,
    ) -> None:
        self._waveforms = waveforms
        self._labels = labels
        self._segment_samples = segment_samples

    def __len__(self) -> int:
        return len(self._waveforms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self._waveforms[idx]
        n = self._segment_samples

        if len(audio) >= n:
            start = random.randint(0, len(audio) - n)
            segment = audio[start : start + n]
        else:
            from acoustic.classification.preprocessing import pad_or_loop
            segment = pad_or_loop(audio, n)

        waveform = torch.from_numpy(segment)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return waveform, label

    @property
    def labels(self) -> list[int]:
        return self._labels


class _LazyEfficientATDataset(Dataset):
    """Lazy dataset that loads audio from HF on-the-fly for EfficientAT.

    Instead of pre-loading all waveforms into RAM, reads and resamples
    one sample at a time from the memory-mapped HF Arrow dataset.
    Memory usage stays constant regardless of dataset size.
    """

    def __init__(
        self,
        hf_dataset,
        split_indices: list[int],
        labels: list[int],
        segment_samples: int,
    ) -> None:
        self._hf_ds = hf_dataset
        self._indices = split_indices
        self._labels = labels
        self._segment_samples = segment_samples

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from acoustic.training.parquet_dataset import decode_wav_bytes

        global_idx = self._indices[idx]
        row = self._hf_ds[global_idx]

        # Decode WAV bytes and resample 16kHz -> 32kHz
        wav_bytes = row["audio"]["bytes"]
        audio = decode_wav_bytes(wav_bytes)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampled = F_audio.resample(waveform, _SOURCE_SR, _TARGET_SR).squeeze(0).numpy()

        # Random segment extraction
        n = self._segment_samples
        if len(resampled) >= n:
            start = random.randint(0, len(resampled) - n)
            segment = resampled[start : start + n]
        else:
            from acoustic.classification.preprocessing import pad_or_loop
            segment = pad_or_loop(resampled, n)

        waveform_out = torch.from_numpy(segment)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return waveform_out, label

    @property
    def labels(self) -> list[int]:
        return self._labels


class EfficientATTrainingRunner:
    """Executes three-stage unfreezing transfer learning for EfficientAT mn10."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._mel_config = EfficientATMelConfig()

    def _build_train_augmentation(self) -> ComposedAugmentation:
        """Build Phase 20 train augmentation chain in LOCKED order (D-02, D-07).

        Order: WideGain -> RoomIR -> Audiomentations -> BackgroundNoiseMixer.
        Each stage is gated on its enable flag so legacy/v6 configs that disable
        wide_gain (db<=0), RIR, audiomentations or noise still produce a valid
        ComposedAugmentation (possibly empty).
        """
        cfg = self._config
        augs: list = []

        # Stage 1: wide gain (D-01..D-04)
        if cfg.wide_gain_db > 0:
            augs.append(
                WideGainAugmentation(
                    wide_gain_db=cfg.wide_gain_db,
                    p=cfg.wide_gain_probability,
                )
            )

        # Stage 2: room impulse response (D-05..D-08)
        if cfg.rir_enabled:
            augs.append(
                RoomIRAugmentation(
                    sample_rate=_SOURCE_SR,
                    pool_size=cfg.rir_pool_size,
                    room_dim_min=tuple(cfg.rir_room_dim_min),
                    room_dim_max=tuple(cfg.rir_room_dim_max),
                    absorption_range=(
                        cfg.rir_absorption_min,
                        cfg.rir_absorption_max,
                    ),
                    source_distance_range=(
                        cfg.rir_source_distance_min,
                        cfg.rir_source_distance_max,
                    ),
                    max_order=cfg.rir_max_order,
                    p=cfg.rir_probability,
                )
            )

        # Stage 3: audiomentations (pitch / stretch / small-gain per D-04)
        if cfg.use_audiomentations:
            augs.append(
                AudiomentationsAugmentation(
                    pitch_semitones=cfg.pitch_shift_semitones,
                    time_stretch_range=(cfg.time_stretch_min, cfg.time_stretch_max),
                    gain_db=cfg.waveform_gain_db,
                    p=cfg.augmentation_probability,
                    sample_rate=_SOURCE_SR,
                )
            )

        # Stage 4: background noise mixer (ESC-50 + UrbanSound8K + UMA-16 ambient
        # per D-10/D-12). UMA-16 ambient is handled via dir_snr_overrides + the
        # uma16_ambient_dir kwarg so the tighter (-5, +15) dB SNR is applied.
        if cfg.noise_augmentation_enabled and cfg.noise_dirs:
            dir_snr_overrides = {
                "uma16_ambient": (
                    cfg.uma16_ambient_snr_low,
                    cfg.uma16_ambient_snr_high,
                ),
            }
            mixer = BackgroundNoiseMixer(
                noise_dirs=[Path(d) for d in cfg.noise_dirs],
                snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
                sample_rate=_SOURCE_SR,
                p=cfg.noise_probability,
                dir_snr_overrides=dir_snr_overrides,
                uma16_ambient_dir=cfg.uma16_ambient_dir or None,
                uma16_ambient_pure_negative_ratio=cfg.uma16_pure_negative_ratio,
            )
            mixer.warm_cache()
            augs.append(mixer)

        return ComposedAugmentation(augs)

    def _build_eval_augmentation(self) -> ComposedAugmentation | None:
        """Build eval augmentation chain.

        Per D-08 the eval pipeline EXCLUDES RoomIRAugmentation -- adding RIR at
        eval time would corrupt the metric since the eval data already reflects
        the deployment distribution (or, for the synthetic eval set, is meant
        to represent clean inputs). Wide gain and audiomentations are also
        omitted; only background noise is mixed in so val/test SNR matches
        the train chain's noise floor.
        """
        cfg = self._config
        if not (cfg.noise_augmentation_enabled and cfg.noise_dirs):
            return ComposedAugmentation([])
        mixer = BackgroundNoiseMixer(
            noise_dirs=[Path(d) for d in cfg.noise_dirs],
            snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
            sample_rate=_SOURCE_SR,
            p=cfg.noise_probability,
        )
        mixer.warm_cache()
        return ComposedAugmentation([mixer])

    @staticmethod
    def _setup_stage1(model: nn.Module) -> None:
        """Freeze all parameters, then unfreeze classifier only."""
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    @staticmethod
    def _setup_stage2(model: nn.Module) -> None:
        """Unfreeze last 3 feature blocks (classifier stays unfrozen)."""
        for block in model.features[-3:]:
            for p in block.parameters():
                p.requires_grad = True

    @staticmethod
    def _setup_stage3(model: nn.Module) -> None:
        """Unfreeze all layers."""
        for p in model.parameters():
            p.requires_grad = True

    def _load_data(
        self, *, _synthetic: bool = False,
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]] | None:
        """Load and resample audio data, returning train/val waveforms+labels."""
        cfg = self._config

        if _synthetic:
            return self._synthetic_data()

        # Priority: 1) HF Dataset  2) Local Parquet  3) WAV files
        use_hf = bool(cfg.dads_hf_repo)
        dads_dir = Path(cfg.dads_path) if cfg.dads_path else None
        use_parquet = (
            not use_hf
            and dads_dir is not None
            and dads_dir.is_dir()
            and list(dads_dir.glob("train-*.parquet"))
        )

        if use_hf:
            # Return None to signal lazy loading — handled in run()
            return None

        if use_parquet:
            from acoustic.training.parquet_dataset import ParquetDatasetBuilder, split_indices as split_idx_fn

            logger.info("Loading DADS Parquet data from %s", dads_dir)
            builder = ParquetDatasetBuilder(dads_dir)
            train_indices, val_indices, _ = split_idx_fn(builder.total_rows, seed=42)
            all_labels = builder.all_labels

            # Load audio from parquet
            from acoustic.training.parquet_dataset import decode_wav_bytes
            import pyarrow.parquet as pq

            all_audio: list[np.ndarray | None] = [None] * builder.total_rows
            for shard_path in sorted(Path(cfg.dads_path).glob("train-*.parquet")):
                table = pq.read_table(shard_path, columns=["audio"])
                for local_idx in range(len(table)):
                    audio_bytes = table.column("audio")[local_idx].as_py()["bytes"]
                    audio = decode_wav_bytes(audio_bytes)
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                    resampled = F_audio.resample(waveform, _SOURCE_SR, _TARGET_SR).squeeze(0).numpy()
                    global_idx = local_idx
                    all_audio[global_idx] = resampled

            train_wav = [all_audio[i] for i in train_indices if all_audio[i] is not None]
            train_lbl = [all_labels[i] for i in train_indices if all_audio[i] is not None]
            val_wav = [all_audio[i] for i in val_indices if all_audio[i] is not None]
            val_lbl = [all_labels[i] for i in val_indices if all_audio[i] is not None]
            return train_wav, train_lbl, val_wav, val_lbl

        # WAV fallback
        from acoustic.training.dataset import collect_wav_files

        logger.info("Loading WAV data from %s", cfg.data_root)
        all_paths, all_labels = collect_wav_files(cfg.data_root, cfg.label_map)
        indices = list(range(len(all_paths)))
        rng = random.Random(42)
        rng.shuffle(indices)
        split_pt = max(1, int(len(indices) * (1.0 - cfg.val_split)))

        import soundfile as sf

        def _load_and_resample(path: Path) -> np.ndarray:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            return F_audio.resample(waveform, _SOURCE_SR, _TARGET_SR).squeeze(0).numpy()

        train_wav = [_load_and_resample(all_paths[i]) for i in indices[:split_pt]]
        train_lbl = [all_labels[i] for i in indices[:split_pt]]
        val_wav = [_load_and_resample(all_paths[i]) for i in indices[split_pt:]]
        val_lbl = [all_labels[i] for i in indices[split_pt:]]
        return train_wav, train_lbl, val_wav, val_lbl

    @staticmethod
    def _synthetic_data() -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
        """Generate tiny synthetic dataset for smoke tests."""
        seg_len = 32000  # 1s at 32kHz
        train_wav = [np.random.randn(seg_len).astype(np.float32) for _ in range(10)]
        train_lbl = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        val_wav = [np.random.randn(seg_len).astype(np.float32) for _ in range(4)]
        val_lbl = [1, 0, 1, 0]
        return train_wav, train_lbl, val_wav, val_lbl

    def run(
        self,
        stop_event: threading.Event,
        progress_callback: Callable[[dict], None] | None = None,
        *,
        _synthetic: bool = False,
    ) -> Path | None:
        """Run three-stage transfer learning.

        Args:
            stop_event: Set to gracefully stop training.
            progress_callback: Called per-epoch with metrics dict.
            _synthetic: Use synthetic data (for testing only).

        Returns:
            Path to best checkpoint, or None if no checkpoint saved.
        """
        cfg = self._config
        mel_cfg = self._mel_config
        total_epochs = cfg.stage1_epochs + cfg.stage2_epochs + cfg.stage3_epochs

        # --- Load data ---
        result = self._load_data(_synthetic=_synthetic)

        if result is None:
            # HF lazy loading path — constant memory usage
            from acoustic.training.hf_dataset import HFDatasetBuilder
            from acoustic.training.parquet_dataset import split_indices as split_idx_fn

            logger.info("Using lazy HF dataset for EfficientAT")
            hf_builder = HFDatasetBuilder(cfg.dads_hf_repo)

            # Phase 20 path: sliding-window dataset + new augmentation chain.
            # Activated when overlap is configured OR RIR is enabled (D-13/D-16).
            use_phase20_path = (
                cfg.window_overlap_ratio > 0 or cfg.rir_enabled
            )

            if use_phase20_path:
                logger.info(
                    "Phase 20 path active: window_overlap_ratio=%.2f rir_enabled=%s",
                    cfg.window_overlap_ratio,
                    cfg.rir_enabled,
                )
                num_files = hf_builder.total_rows
                train_files, val_files, _test_files = split_file_indices(
                    num_files=num_files,
                    seed=42,
                    train=1.0 - cfg.val_split,
                    val=cfg.val_split / 2.0,
                )
                window_samples = int(0.5 * _SOURCE_SR)  # 8000 samples = 0.5 s @ 16 kHz
                train_hop = max(
                    1,
                    int(window_samples * (1.0 - cfg.window_overlap_ratio)),
                )
                test_hop = window_samples  # D-16: non-overlapping test split

                train_ds = WindowedHFDroneDataset(
                    hf_builder._hf_ds,
                    file_indices=train_files,
                    window_samples=window_samples,
                    hop_samples=train_hop,
                    waveform_aug=self._build_train_augmentation(),
                )
                val_ds = WindowedHFDroneDataset(
                    hf_builder._hf_ds,
                    file_indices=val_files,
                    window_samples=window_samples,
                    hop_samples=test_hop,
                    waveform_aug=self._build_eval_augmentation(),
                )
                train_lbl = [hf_builder.all_labels[i] for i in train_files]
                val_lbl = [hf_builder.all_labels[i] for i in val_files]
            else:
                train_indices, val_indices, _ = split_idx_fn(
                    hf_builder.total_rows, seed=42,
                )

                train_lbl = [hf_builder.all_labels[i] for i in train_indices]
                val_lbl = [hf_builder.all_labels[i] for i in val_indices]

                train_ds = _LazyEfficientATDataset(
                    hf_builder._hf_ds, train_indices, train_lbl, mel_cfg.segment_samples,
                )
                val_ds = _LazyEfficientATDataset(
                    hf_builder._hf_ds, val_indices, val_lbl, mel_cfg.segment_samples,
                )
        else:
            train_wav, train_lbl, val_wav, val_lbl = result
            train_ds = _EfficientATDataset(train_wav, train_lbl, mel_cfg.segment_samples)
            val_ds = _EfficientATDataset(val_wav, val_lbl, mel_cfg.segment_samples)

        dataset_len = len(train_ds)
        num_workers = 0 if dataset_len < 32 else min(4, os.cpu_count() or 1)
        loader_kwargs: dict = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        sampler = WeightedRandomSampler(
            [1.0 / max(1, train_lbl.count(l)) for l in train_lbl],
            num_samples=len(train_lbl), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

        # --- Device ---
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("EfficientAT training on device: %s", device)

        # --- Model setup ---
        width_mult = cfg.width_mult
        if cfg.model_type == "efficientat_mn05":
            width_mult = 0.5
        logger.info("EfficientAT width_mult=%.1f (%s)", width_mult,
                     "MN10 ~4.6M params" if width_mult == 1.0 else f"MN{int(width_mult*10):02d} ~{width_mult**2 * 4.6:.1f}M params")

        model = get_model(
            num_classes=527, width_mult=width_mult, head_type="mlp",
            input_dim_f=mel_cfg.n_mels, input_dim_t=mel_cfg.input_dim_t,
        )

        # Load pretrained weights if available (must match width_mult)
        pretrained_path = Path(cfg.pretrained_weights) if cfg.pretrained_weights else None
        if pretrained_path and pretrained_path.exists():
            state_dict = torch.load(str(pretrained_path), map_location="cpu", weights_only=True)
            try:
                model.load_state_dict(state_dict)
                logger.info("Loaded pretrained weights from %s", pretrained_path)
            except RuntimeError:
                logger.warning("Pretrained weights don't match model (width_mult=%.1f), training from scratch", width_mult)
        elif width_mult != 1.0:
            logger.info("No pretrained weights for width_mult=%.1f, training from scratch", width_mult)

        model.float()

        # Replace head for binary classification
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        model = model.to(device)

        # Mel preprocessors (training with SpecAugment, eval without)
        mel_train = AugmentMelSTFT(
            n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
            win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
            n_fft=mel_cfg.n_fft,
            freqm=cfg.specaug_freq_mask,
            timem=cfg.specaug_time_mask,
        ).to(device)
        mel_eval = AugmentMelSTFT(
            n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
            win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
            n_fft=mel_cfg.n_fft, freqm=0, timem=0,
        ).to(device)

        # D-31: honor cfg.loss_function ("focal" by default) instead of
        # hard-coding BCE. See .planning/debug/training-collapse-constant-output.md
        criterion = build_loss_function(
            loss_type=cfg.loss_function,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            bce_pos_weight=cfg.bce_pos_weight,
        ).to(device)

        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Stage definitions ---
        stages = [
            (1, cfg.stage1_epochs, cfg.stage1_lr, self._setup_stage1, False),
            (2, cfg.stage2_epochs, cfg.stage2_lr, self._setup_stage2, True),
            (3, cfg.stage3_epochs, cfg.stage3_lr, self._setup_stage3, True),
        ]

        global_epoch = 0

        for stage_num, stage_epochs, lr, setup_fn, use_cosine in stages:
            setup_fn(model)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=lr)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
                if use_cosine else None
            )

            # Reset early stopping per stage so stages 2/3 always run
            early_stopping = EarlyStopping(patience=cfg.patience)

            logger.info("Stage %d: %d epochs, lr=%.1e, %d trainable params",
                        stage_num, stage_epochs, lr, sum(p.numel() for p in trainable))

            for stage_epoch in range(stage_epochs):
                if stop_event.is_set():
                    logger.info("Stop event set at stage %d epoch %d", stage_num, stage_epoch)
                    return ckpt_path if ckpt_path.exists() else None

                global_epoch += 1

                # --- Train ---
                model.train()
                mel_train.train()
                train_loss_sum = 0.0
                train_batches = 0
                total_batches = len(train_loader)

                for batch_wav, batch_y in train_loader:
                    batch_wav = batch_wav.to(device)
                    batch_y = batch_y.to(device)

                    # Mel spectrogram on device
                    mel = mel_train(batch_wav)  # (B, n_mels, T)
                    mel = mel.unsqueeze(1)       # (B, 1, n_mels, T)

                    optimizer.zero_grad()
                    logits, _ = model(mel)
                    logits = logits.squeeze(-1)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss_sum += loss.item()
                    train_batches += 1

                    # Batch-level progress (every 10 batches)
                    if progress_callback is not None and train_batches % 10 == 0:
                        progress_callback({
                            "epoch": global_epoch,
                            "total_epochs": total_epochs,
                            "stage": stage_num,
                            "batch": train_batches,
                            "total_batches": total_batches,
                            "train_loss": train_loss_sum / train_batches,
                        })

                avg_train_loss = train_loss_sum / max(train_batches, 1)

                # --- Validate ---
                model.eval()
                mel_eval.eval()
                val_loss_sum = 0.0
                val_batches = 0
                val_correct = 0
                val_total = 0
                tp = fp = tn = fn = 0

                with torch.no_grad():
                    for batch_wav, batch_y in val_loader:
                        batch_wav = batch_wav.to(device)
                        batch_y = batch_y.to(device)

                        mel = mel_eval(batch_wav)
                        mel = mel.unsqueeze(1)

                        logits, _ = model(mel)
                        logits = logits.squeeze(-1)
                        loss = criterion(logits, batch_y)

                        val_loss_sum += loss.item()
                        val_batches += 1

                        preds = (torch.sigmoid(logits) >= 0.5).float()
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_y.numel()

                        # Confusion matrix
                        tp += int(((preds == 1) & (batch_y == 1)).sum().item())
                        fp += int(((preds == 1) & (batch_y == 0)).sum().item())
                        tn += int(((preds == 0) & (batch_y == 0)).sum().item())
                        fn += int(((preds == 0) & (batch_y == 1)).sum().item())

                avg_val_loss = val_loss_sum / max(val_batches, 1)
                val_accuracy = val_correct / max(val_total, 1)

                # --- Checkpoint ---
                improved = early_stopping.step(avg_val_loss)
                if improved:
                    torch.save(model.state_dict(), str(ckpt_path))
                    logger.info("Stage %d epoch %d: val_loss=%.4f (improved), saved",
                                stage_num, stage_epoch + 1, avg_val_loss)

                if scheduler is not None:
                    scheduler.step()

                # --- Progress ---
                if progress_callback is not None:
                    progress_callback({
                        "epoch": global_epoch,
                        "total_epochs": total_epochs,
                        "stage": stage_num,
                        "stage_epoch": stage_epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_acc": val_accuracy,
                        "best_val_loss": early_stopping.best_loss,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    })

                if early_stopping.should_stop:
                    logger.info("Early stopping at stage %d epoch %d", stage_num, stage_epoch + 1)
                    break

        if ckpt_path.exists():
            return ckpt_path
        return None
