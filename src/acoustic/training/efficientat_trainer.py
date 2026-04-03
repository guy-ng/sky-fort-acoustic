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
from acoustic.training.config import TrainingConfig
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
            segment = np.zeros(n, dtype=np.float32)
            segment[: len(audio)] = audio

        waveform = torch.from_numpy(segment)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return waveform, label

    @property
    def labels(self) -> list[int]:
        return self._labels


class EfficientATTrainingRunner:
    """Executes three-stage unfreezing transfer learning for EfficientAT mn10."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._mel_config = EfficientATMelConfig()

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
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
        """Load and resample audio data, returning train/val waveforms+labels."""
        cfg = self._config

        if _synthetic:
            return self._synthetic_data()

        # Try Parquet first, then WAV
        from acoustic.training.parquet_dataset import ParquetDatasetBuilder, split_indices as split_idx_fn

        dads_dir = Path(cfg.dads_path) if cfg.dads_path else None
        use_parquet = dads_dir is not None and dads_dir.is_dir() and list(dads_dir.glob("train-*.parquet"))

        if use_parquet:
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
                # Reconstruct global offset
                for local_idx in range(len(table)):
                    audio_bytes = table.column("audio")[local_idx].as_py()["bytes"]
                    audio = decode_wav_bytes(audio_bytes)
                    # Resample 16kHz -> 32kHz
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                    resampled = F_audio.resample(waveform, _SOURCE_SR, _TARGET_SR).squeeze(0).numpy()
                    # Find global index
                    global_idx = local_idx  # simplified; builder handles offset
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
        train_wav, train_lbl, val_wav, val_lbl = self._load_data(_synthetic=_synthetic)

        train_ds = _EfficientATDataset(train_wav, train_lbl, mel_cfg.segment_samples)
        val_ds = _EfficientATDataset(val_wav, val_lbl, mel_cfg.segment_samples)

        dataset_len = len(train_ds)
        num_workers = 0 if dataset_len < 32 else min(8, os.cpu_count() or 1)
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
        model = get_model(
            num_classes=527, width_mult=1.0, head_type="mlp",
            input_dim_f=mel_cfg.n_mels, input_dim_t=mel_cfg.input_dim_t,
        )

        # Load pretrained weights if available
        pretrained_path = Path(cfg.pretrained_weights) if cfg.pretrained_weights else None
        if pretrained_path and pretrained_path.exists():
            state_dict = torch.load(str(pretrained_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Loaded pretrained weights from %s", pretrained_path)

        model.float()

        # Replace head for binary classification
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        model = model.to(device)

        # Mel preprocessors (training with SpecAugment, eval without)
        mel_train = AugmentMelSTFT(
            n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
            win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
            n_fft=mel_cfg.n_fft, freqm=48, timem=192,
        ).to(device)
        mel_eval = AugmentMelSTFT(
            n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
            win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
            n_fft=mel_cfg.n_fft, freqm=0, timem=0,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=cfg.patience)

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

                avg_train_loss = train_loss_sum / max(train_batches, 1)

                # --- Validate ---
                model.eval()
                mel_eval.eval()
                val_loss_sum = 0.0
                val_batches = 0
                val_correct = 0
                val_total = 0

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
                    })

                if early_stopping.should_stop:
                    logger.info("Early stopping at stage %d epoch %d", stage_num, stage_epoch + 1)
                    break

            if early_stopping.should_stop:
                break

        if ckpt_path.exists():
            return ckpt_path
        return None
