"""Ensemble classification: weighted soft voting over multiple model architectures.

Provides:
  - EnsembleClassifier: combines N Classifier predictions via weighted soft voting
  - ModelRegistry: maps model type strings to loader functions
  - EnsembleConfig / ModelEntry: JSON config file parsing via Pydantic
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Union

import torch
from pydantic import BaseModel

from acoustic.classification.protocols import Classifier

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """Single model entry in an ensemble configuration file."""

    type: str
    path: str
    weight: float


class EnsembleConfig(BaseModel):
    """Parsed ensemble configuration with a list of model entries."""

    models: list[ModelEntry]

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> EnsembleConfig:
        """Read and validate an ensemble config JSON file."""
        raw = Path(path).read_text()
        data = json.loads(raw)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ModelLoader = Callable[[str], Classifier]

_REGISTRY: dict[str, ModelLoader] = {}


def register_model(type_name: str, loader: ModelLoader) -> None:
    """Register a model loader function under the given type name."""
    _REGISTRY[type_name] = loader


def load_model(type_name: str, path: str) -> Classifier:
    """Load a classifier by type name and checkpoint path."""
    if type_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model type: {type_name}. Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[type_name](path)


def get_registered_types() -> list[str]:
    """Return all registered model type names."""
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# EnsembleClassifier
# ---------------------------------------------------------------------------


class EnsembleClassifier:
    """Combines multiple Classifier predictions via weighted soft voting.

    Weights are F1-derived and normalized to sum to 1.0 at construction.
    Live mode enforces a hard cap on model count for latency control.
    """

    def __init__(
        self,
        classifiers: list[Classifier],
        weights: list[float],
        *,
        max_live_models: int = 3,
        live_mode: bool = True,
    ) -> None:
        if len(classifiers) != len(weights):
            raise ValueError(
                f"classifiers and weights must have same length, "
                f"got {len(classifiers)} and {len(weights)}"
            )
        if live_mode and len(classifiers) > max_live_models:
            raise ValueError(
                f"Live mode allows max {max_live_models} models, "
                f"got {len(classifiers)}"
            )
        self._classifiers = classifiers
        total = sum(weights)
        self._weights = [w / total for w in weights]

    def predict(self, features: torch.Tensor) -> float:
        """Weighted soft voting: sum(w_i * clf_i.predict(features))."""
        return sum(
            w * clf.predict(features)
            for clf, w in zip(self._classifiers, self._weights)
        )

    @property
    def model_count(self) -> int:
        """Number of models in the ensemble."""
        return len(self._classifiers)

    @property
    def weights(self) -> list[float]:
        """Copy of normalized weights."""
        return list(self._weights)


# ---------------------------------------------------------------------------
# Built-in model registrations
# ---------------------------------------------------------------------------


def _load_research_cnn(path: str) -> Classifier:
    """Load a ResearchCNN checkpoint and wrap in ResearchClassifier."""
    from acoustic.classification.research_cnn import ResearchClassifier, ResearchCNN

    model = ResearchCNN()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return ResearchClassifier(model)


register_model("research_cnn", _load_research_cnn)
