#!/usr/bin/env python3
"""Train EfficientNet-B0 drone classifier on local audio data.

Reads WAV files from audio-data/data/{drone,background,other}/, creates
mel-spectrograms matching the POC parameters, trains a binary classifier
(drone vs not-drone), and exports to ONNX.

Usage:
    .venv/bin/python scripts/train_cnn.py [--epochs 20] [--batch-size 32] [--lr 1e-3]

Output:
    models/uav_melspec_cnn.onnx
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path for preprocessing imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic.classification.preprocessing import (
    MAX_FRAMES,
    N_MELS,
    SR_CNN,
    fast_resample,
    make_melspec,
    norm_spec,
    pad_or_trim,
)

DATA_DIR = ROOT / "audio-data" / "data"
MODEL_OUT = ROOT / "models" / "uav_melspec_cnn.onnx"

# EfficientNet expects 224x224 RGB — we'll resize the spectrogram and repeat channels
EFFICIENTNET_SIZE = 224


def collect_wav_files() -> list[tuple[Path, int]]:
    """Collect WAV files with labels. drone=1, background/other=0."""
    samples: list[tuple[Path, int]] = []

    # Drone files (label=1)
    drone_dir = DATA_DIR / "drone"
    if drone_dir.exists():
        for wav in drone_dir.rglob("*.wav"):
            samples.append((wav, 1))

    # Background files (label=0)
    bg_dir = DATA_DIR / "background"
    if bg_dir.exists():
        for wav in bg_dir.rglob("*.wav"):
            samples.append((wav, 0))

    # Other files (label=0 — not drone)
    other_dir = DATA_DIR / "other"
    if other_dir.exists():
        for wav in other_dir.rglob("*.wav"):
            samples.append((wav, 0))

    return samples


def wav_to_spectrogram(wav_path: Path) -> np.ndarray | None:
    """Convert a WAV file to a normalized mel-spectrogram.

    Returns shape (1, MAX_FRAMES, N_MELS) float32, or None on failure.
    """
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        # If multi-channel, take first channel
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Resample to CNN rate
        audio = fast_resample(audio, sr, SR_CNN)
        # Take last 2 seconds
        n_samples = int(SR_CNN * 2.0)
        if len(audio) >= n_samples:
            audio = audio[-n_samples:]
        else:
            audio = np.pad(audio, (n_samples - len(audio), 0), mode="constant").astype(
                np.float32
            )
        spec = make_melspec(audio, SR_CNN)
        spec = pad_or_trim(spec, MAX_FRAMES)
        spec = norm_spec(spec)
        return spec.astype(np.float32)  # (128, 64)
    except Exception as e:
        print(f"  Skipping {wav_path.name}: {e}")
        return None


class DroneAudioDataset(Dataset):
    """Precomputed spectrogram dataset."""

    def __init__(self, specs: list[np.ndarray], labels: list[int]):
        self.specs = specs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.specs[idx]  # (128, 64)
        # Resize to EfficientNet input: (224, 224) via torch interpolation
        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 128, 64)
        tensor = nn.functional.interpolate(
            tensor.unsqueeze(0), size=(EFFICIENTNET_SIZE, EFFICIENTNET_SIZE), mode="bilinear", align_corners=False
        ).squeeze(0)  # (1, 224, 224)
        # Repeat grayscale to 3 channels
        tensor = tensor.repeat(3, 1, 1)  # (3, 224, 224)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def build_model() -> nn.Module:
    """Build EfficientNet-B0 with binary classification head."""
    import timm

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    return model


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect and preprocess
    print("Collecting WAV files...")
    samples = collect_wav_files()
    random.shuffle(samples)

    drone_count = sum(1 for _, l in samples if l == 1)
    bg_count = len(samples) - drone_count
    print(f"Found {len(samples)} files: {drone_count} drone, {bg_count} background/other")

    if len(samples) < 10:
        print("Not enough data to train. Need at least 10 samples.")
        sys.exit(1)

    # Balance classes — undersample the majority class
    drone_samples = [(p, l) for p, l in samples if l == 1]
    neg_samples = [(p, l) for p, l in samples if l == 0]
    min_count = min(len(drone_samples), len(neg_samples))
    if len(drone_samples) > min_count:
        drone_samples = random.sample(drone_samples, min_count)
    if len(neg_samples) > min_count:
        neg_samples = random.sample(neg_samples, min_count)
    balanced = drone_samples + neg_samples
    random.shuffle(balanced)
    print(f"Balanced to {len(balanced)} samples ({min_count} per class)")

    # Apply a sample limit for faster iteration if dataset is huge
    max_samples = args.max_samples
    if max_samples and len(balanced) > max_samples:
        balanced = balanced[:max_samples]
        print(f"Limited to {max_samples} samples for training speed")

    print("Computing spectrograms...")
    specs: list[np.ndarray] = []
    labels: list[int] = []
    for i, (wav_path, label) in enumerate(balanced):
        if i % 200 == 0:
            print(f"  {i}/{len(balanced)}...")
        spec = wav_to_spectrogram(wav_path)
        if spec is not None:
            specs.append(spec)
            labels.append(label)

    print(f"Processed {len(specs)} spectrograms")

    # Split train/val
    dataset = DroneAudioDataset(specs, labels)
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}")

    # Build model
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss / max(train_total, 1):.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss / max(val_total, 1):.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Load best model and export to ONNX
    model.load_state_dict(best_state)
    model.eval()
    model = model.to("cpu")

    # Export with sigmoid applied — output is drone probability [0, 1]
    class ModelWithSigmoid(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.base(x))

    export_model = ModelWithSigmoid(model)
    export_model.eval()

    # ONNX export — input matches preprocessing: we handle resize in inference
    # But ONNX model takes the EfficientNet-ready (1, 3, 224, 224) input
    dummy_input = torch.randn(1, 3, EFFICIENTNET_SIZE, EFFICIENTNET_SIZE)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(MODEL_OUT),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nModel exported to {MODEL_OUT}")
    print(f"File size: {MODEL_OUT.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 drone classifier")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class (for quick testing)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
