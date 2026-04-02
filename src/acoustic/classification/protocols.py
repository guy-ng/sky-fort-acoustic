"""Runtime-checkable protocols for classification pipeline components."""

from __future__ import annotations

import numpy as np
import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class Preprocessor(Protocol):
    """Converts raw audio into a feature tensor for classification."""

    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor: ...


@runtime_checkable
class Classifier(Protocol):
    """Produces a drone probability from a feature tensor."""

    def predict(self, features: torch.Tensor) -> float: ...


@runtime_checkable
class Aggregator(Protocol):
    """Aggregates multiple segment probabilities into a single score."""

    def aggregate(self, probabilities: list[float]) -> float: ...
