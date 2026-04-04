"""Training loop: EarlyStopping and TrainingRunner for ResearchCNN.

Executes Adam/BCE training with early stopping on validation loss.
Saves best checkpoint as .pt state_dict for ResearchClassifier.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from acoustic.classification.config import MelConfig
from acoustic.classification.research_cnn import ResearchCNN
from acoustic.training.augmentation import (
    AudiomentationsAugmentation,
    BackgroundNoiseMixer,
    ComposedAugmentation,
    SpecAugment,
    WaveformAugmentation,
)
from acoustic.training.config import TrainingConfig
from acoustic.training.dataset import DroneAudioDataset, build_weighted_sampler, collect_wav_files
from acoustic.training.losses import build_loss_function

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitor validation loss and signal when to stop training.

    Tracks the best validation loss seen. If val_loss does not improve by
    at least min_delta for `patience` consecutive calls, should_stop becomes True.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float | None = None
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Check if validation loss improved.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if val_loss improved (new best), False otherwise.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class TrainingRunner:
    """Executes the full training loop for ResearchCNN.

    Uses Adam optimizer, BCELoss, early stopping, and optional LR scheduling.
    Produces a .pt state_dict checkpoint at the configured path.
    """

    def __init__(
        self,
        config: TrainingConfig,
        mel_config: MelConfig | None = None,
    ) -> None:
        self._config = config
        self._mel_config = mel_config or MelConfig()

    @staticmethod
    def _warm_caches(*datasets: object, limit: int = 0) -> None:
        """Call warm_cache() on each dataset that supports it.

        Args:
            limit: Max samples per dataset.  0 = load everything.
        """
        for ds in datasets:
            if hasattr(ds, "warm_cache"):
                ds.warm_cache(limit=limit)  # type: ignore[union-attr]

    def run(
        self,
        stop_event: threading.Event,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> Path | None:
        """Run the training loop.

        Args:
            stop_event: Threading event checked at each epoch. If set, training
                exits gracefully, preserving the best checkpoint saved so far.
            progress_callback: Optional callback invoked per epoch with a dict
                of metrics (epoch, train_loss, val_loss, val_acc, best_val_loss).

        Returns:
            Path to saved checkpoint, or None if no checkpoint was saved.
        """
        cfg = self._config

        # Build augmentation objects
        wave_aug = None
        noise_mixer: BackgroundNoiseMixer | None = None
        spec_aug: SpecAugment | None = None

        if cfg.augmentation_enabled:
            # Waveform augmentation: audiomentations (TRN-12) or legacy
            if cfg.use_audiomentations:
                wave_aug = AudiomentationsAugmentation(
                    pitch_semitones=cfg.pitch_shift_semitones,
                    time_stretch_range=(cfg.time_stretch_min, cfg.time_stretch_max),
                    gain_db=cfg.waveform_gain_db,
                    p=cfg.augmentation_probability,
                    sample_rate=self._mel_config.sample_rate,
                )
            else:
                wave_aug = WaveformAugmentation(
                    snr_range=(cfg.wave_snr_range_low, cfg.wave_snr_range_high),
                    gain_db=cfg.wave_gain_db,
                )

            # Background noise augmentation (TRN-11)
            if cfg.noise_augmentation_enabled and cfg.noise_dirs:
                noise_mixer = BackgroundNoiseMixer(
                    noise_dirs=[Path(d) for d in cfg.noise_dirs],
                    snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
                    sample_rate=self._mel_config.sample_rate,
                    p=cfg.noise_probability,
                )
                noise_mixer.warm_cache()
                logger.info("Noise mixer loaded %d files", len(noise_mixer._noise_cache))

            spec_aug = SpecAugment(
                time_mask_param=cfg.spec_time_mask_param,
                freq_mask_param=cfg.spec_freq_mask_param,
                num_time_masks=cfg.spec_num_time_masks,
                num_freq_masks=cfg.spec_num_freq_masks,
            )

        # Compose waveform augmentation pipeline using picklable ComposedAugmentation
        # (closures are NOT picklable and break DataLoader num_workers > 0)
        composed_wave_aug = None
        if noise_mixer is not None and wave_aug is not None:
            composed_wave_aug = ComposedAugmentation([noise_mixer, wave_aug])
        elif noise_mixer is not None:
            composed_wave_aug = noise_mixer
        elif wave_aug is not None:
            composed_wave_aug = wave_aug

        # --- Data source selection ---
        dads_dir = Path(cfg.dads_path) if cfg.dads_path else None
        use_parquet = dads_dir is not None and dads_dir.is_dir() and list(dads_dir.glob("train-*.parquet"))

        if use_parquet:
            from acoustic.training.parquet_dataset import ParquetDatasetBuilder, split_indices

            logger.info("Using DADS Parquet data from %s", dads_dir)
            builder = ParquetDatasetBuilder(dads_dir)
            train_idx, val_idx, _test_idx = split_indices(builder.total_rows, seed=42)

            train_ds = builder.build(
                train_idx, mel_config=self._mel_config,
                waveform_aug=composed_wave_aug, spec_aug=spec_aug,
            )
            val_ds = builder.build(val_idx, mel_config=self._mel_config)
            train_labels = train_ds.labels
            val_labels = val_ds.labels
        else:
            # Legacy WAV path (existing code, unchanged)
            logger.info("Using WAV data from %s", cfg.data_root)
            all_paths, all_labels = collect_wav_files(cfg.data_root, cfg.label_map)
            indices = list(range(len(all_paths)))
            rng = random.Random(42)
            rng.shuffle(indices)

            split_idx = max(1, int(len(indices) * (1.0 - cfg.val_split)))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            train_paths = [all_paths[i] for i in train_indices]
            train_labels = [all_labels[i] for i in train_indices]
            val_paths = [all_paths[i] for i in val_indices]
            val_labels = [all_labels[i] for i in val_indices]

            # Create datasets (val set never augmented)
            train_ds = DroneAudioDataset(
                train_paths, train_labels, self._mel_config,
                waveform_aug=composed_wave_aug, spec_aug=spec_aug,
            )
            val_ds = DroneAudioDataset(
                val_paths, val_labels, self._mel_config,
            )

        # Two-phase cache warm-up:
        # Phase 1 (blocking): quick-load first 1000 samples so epoch 1 starts fast.
        # Phase 2 (background): load the rest while training runs.
        _QUICK_PRIME = 1000
        self._warm_caches(train_ds, val_ds, limit=_QUICK_PRIME)
        logger.info("Quick-primed %d samples, starting background cache load", _QUICK_PRIME)

        cache_thread = threading.Thread(
            target=self._warm_caches, args=(train_ds, val_ds), daemon=True,
        )
        cache_thread.start()

        # Create data loaders (workers prefetch data while GPU trains)
        # Scale workers with dataset size — spawning 8 workers for 8 files is wasteful.
        # For tiny datasets (<32 samples) skip multiprocessing entirely to avoid
        # process-spawn overhead that dominates on small workloads.
        dataset_len = len(train_ds)
        if dataset_len < 32:
            num_workers = 0
        else:
            num_workers = min(8, os.cpu_count() or 1)
        from collections import Counter
        label_counts = Counter(train_labels)
        logger.info("Class distribution: %s (total=%d)", dict(label_counts), len(train_labels))
        train_sampler = build_weighted_sampler(train_labels)
        loader_kwargs: dict = dict(
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            **loader_kwargs,
        )

        # Device selection: MPS (Apple Silicon GPU) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Training on device: %s", device)

        # Model, optimizer, loss, scheduler
        model = ResearchCNN(logits_mode=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = build_loss_function(
            cfg.loss_function,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            bce_pos_weight=cfg.bce_pos_weight,
        )
        logger.info("Loss function: %s", cfg.loss_function)
        early_stopping = EarlyStopping(patience=cfg.patience)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5,
        )

        # Ensure checkpoint directory exists
        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Don't block — background thread continues loading while we train.
        # Cache misses are handled gracefully via lazy per-sample I/O.

        # Helper to gather cache stats for progress callbacks
        def _cache_info() -> dict:
            cached = 0
            total = 0
            for ds in (train_ds, val_ds):
                if hasattr(ds, "_audio_cache"):
                    cached += len(ds._audio_cache)
                    total += len(ds)
            return {"cache_loaded": cached, "cache_total": total}

        # Training loop
        for epoch in range(cfg.max_epochs):
            if stop_event.is_set():
                logger.info("Stop event set, exiting training at epoch %d", epoch)
                break

            # --- Train epoch ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            total_batches = len(train_loader)
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x).squeeze(-1)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_batches += 1

                # Batch-level progress (every 10 batches to avoid callback overhead)
                if progress_callback is not None and train_batches % 10 == 0:
                    progress_callback({
                        "epoch": epoch + 1,
                        "total_epochs": cfg.max_epochs,
                        "batch": train_batches,
                        "total_batches": total_batches,
                        "train_loss": train_loss_sum / train_batches,
                        **_cache_info(),
                    })

            avg_train_loss = train_loss_sum / max(train_batches, 1)

            # --- Validation epoch ---
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_correct = 0
            val_total = 0
            tp = fp = tn = fn = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x).squeeze(-1)
                    loss = criterion(output, batch_y)
                    val_loss_sum += loss.item()
                    val_batches += 1

                    # Accuracy: threshold at 0.5
                    preds = (torch.sigmoid(output) >= 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.numel()

                    # Confusion matrix
                    tp += int(((preds == 1) & (batch_y == 1)).sum().item())
                    fp += int(((preds == 1) & (batch_y == 0)).sum().item())
                    tn += int(((preds == 0) & (batch_y == 0)).sum().item())
                    fn += int(((preds == 0) & (batch_y == 1)).sum().item())

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            val_accuracy = val_correct / max(val_total, 1)

            # --- Early stopping ---
            improved = early_stopping.step(avg_val_loss)
            if improved:
                torch.save(model.state_dict(), str(ckpt_path))
                logger.info(
                    "Epoch %d: val_loss=%.4f (improved), saved checkpoint",
                    epoch + 1, avg_val_loss,
                )

            # --- LR scheduler ---
            scheduler.step(avg_val_loss)

            # --- Progress callback ---
            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch + 1,
                    "total_epochs": cfg.max_epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": val_accuracy,
                    "best_val_loss": early_stopping.best_loss,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    **_cache_info(),
                })

            if early_stopping.should_stop:
                logger.info(
                    "Early stopping triggered after %d epochs", epoch + 1,
                )
                break

        # Export to TorchScript deployable format (ROADMAP SC3)
        if early_stopping.best_loss is not None and ckpt_path.exists():
            try:
                export_model = ResearchCNN(logits_mode=False)
                export_model.load_state_dict(torch.load(str(ckpt_path), weights_only=True, map_location="cpu"))
                export_model.eval()
                jit_path = Path(str(ckpt_path) + ".jit")
                scripted = torch.jit.script(export_model)
                scripted.save(str(jit_path))
                logger.info("Exported TorchScript model to %s", jit_path)
            except Exception:
                logger.exception("TorchScript export failed, state_dict checkpoint still available")

        # Return checkpoint path if one was saved
        if early_stopping.best_loss is not None and ckpt_path.exists():
            return ckpt_path
        return None
