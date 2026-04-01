"""Classifier and Preprocessor protocols for clean model swaps."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Preprocessor(Protocol):
    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Process raw audio into model-ready features.

        Returns: Tensor of shape (1, 1, max_frames, n_mels).
        """
        ...


@runtime_checkable
class Classifier(Protocol):
    def predict(self, features: torch.Tensor) -> float:
        """Run inference on preprocessed features.

        Returns: Drone probability in [0.0, 1.0].
        """
        ...
