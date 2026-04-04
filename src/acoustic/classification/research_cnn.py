"""ResearchCNN model ported from Acoustic-UAV-Identification train_strong_cnn.py.

Architecture: 3-layer Conv2D (32/64/128) with BatchNorm + MaxPool,
GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid output.

Input shape:  (N, 1, 128, 64) -- (batch, channel, time_frames, n_mels)
Output shape: (N, 1)           -- drone probability per sample
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResearchCNN(nn.Module):
    """Binary drone classifier CNN matching the research build_model() architecture."""

    def __init__(self, logits_mode: bool = False) -> None:
        super().__init__()
        self._logits_mode = logits_mode
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features extraction then classification."""
        x = self.features(x)
        x = self.classifier(x)
        if not self._logits_mode:
            x = self._sigmoid(x)
        return x


class ResearchClassifier:
    """Wrapper that satisfies the Classifier protocol for ResearchCNN.

    Sets model to eval mode and runs inference under torch.no_grad().
    """

    def __init__(self, model: ResearchCNN) -> None:
        self._model = model
        self._model.eval()

    def predict(self, features: torch.Tensor) -> float:
        """Run inference and return drone probability as a Python float."""
        with torch.no_grad():
            output = self._model(features)
        return output.item()
