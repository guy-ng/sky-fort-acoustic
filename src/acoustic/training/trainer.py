"""Training loop: EarlyStopping and TrainingRunner for ResearchCNN.

Executes Adam/BCE training with early stopping on validation loss.
Saves best checkpoint as .pt state_dict for ResearchClassifier.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from acoustic.classification.config import MelConfig
from acoustic.classification.research_cnn import ResearchCNN
from acoustic.training.augmentation import SpecAugment, WaveformAugmentation
from acoustic.training.config import TrainingConfig
from acoustic.training.dataset import DroneAudioDataset, build_weighted_sampler, collect_wav_files

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
        wave_aug: WaveformAugmentation | None = None
        spec_aug: SpecAugment | None = None
        if cfg.augmentation_enabled:
            wave_aug = WaveformAugmentation(
                snr_range=(cfg.wave_snr_range_low, cfg.wave_snr_range_high),
                gain_db=cfg.wave_gain_db,
            )
            spec_aug = SpecAugment(
                time_mask_param=cfg.spec_time_mask_param,
                freq_mask_param=cfg.spec_freq_mask_param,
                num_time_masks=cfg.spec_num_time_masks,
                num_freq_masks=cfg.spec_num_freq_masks,
            )

        # Collect files and split at file level (sklearn-free)
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
            waveform_aug=wave_aug, spec_aug=spec_aug,
        )
        val_ds = DroneAudioDataset(
            val_paths, val_labels, self._mel_config,
        )

        # Create data loaders
        train_sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

        # Model, optimizer, loss, scheduler
        model = ResearchCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.BCELoss()
        early_stopping = EarlyStopping(patience=cfg.patience)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5,
        )

        # Ensure checkpoint directory exists
        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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
                    output = model(batch_x).squeeze(-1)
                    loss = criterion(output, batch_y)
                    val_loss_sum += loss.item()
                    val_batches += 1

                    # Accuracy: threshold at 0.5
                    preds = (output >= 0.5).float()
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
                })

            if early_stopping.should_stop:
                logger.info(
                    "Early stopping triggered after %d epochs", epoch + 1,
                )
                break

        # Export to TorchScript deployable format (ROADMAP SC3)
        if early_stopping.best_loss is not None and ckpt_path.exists():
            try:
                export_model = ResearchCNN()
                export_model.load_state_dict(torch.load(str(ckpt_path), weights_only=True))
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
