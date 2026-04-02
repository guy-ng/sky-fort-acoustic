"""Unit tests for the ensemble classification module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from acoustic.classification.ensemble import (
    EnsembleClassifier,
    EnsembleConfig,
    ModelEntry,
    load_model,
    register_model,
)
from acoustic.classification.protocols import Classifier


class MockClassifier:
    """A simple mock that returns a fixed probability."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, features: torch.Tensor) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Weighted soft voting
# ---------------------------------------------------------------------------

def test_weighted_soft_voting() -> None:
    """EnsembleClassifier with 2 mock classifiers and weights [0.9, 0.7] produces correct weighted average."""
    c1 = MockClassifier(0.8)
    c2 = MockClassifier(0.6)
    ensemble = EnsembleClassifier(classifiers=[c1, c2], weights=[0.9, 0.7], live_mode=False)
    result = ensemble.predict(torch.zeros(1))
    # (0.9/1.6)*0.8 + (0.7/1.6)*0.6 = 0.45 + 0.2625 = 0.7125
    assert abs(result - 0.7125) < 1e-6


def test_weight_normalization() -> None:
    """Weights [0.85, 0.92] are normalized to sum to 1.0."""
    c1 = MockClassifier(1.0)
    c2 = MockClassifier(1.0)
    ensemble = EnsembleClassifier(classifiers=[c1, c2], weights=[0.85, 0.92], live_mode=False)
    assert abs(sum(ensemble.weights) - 1.0) < 1e-9


def test_single_model_ensemble() -> None:
    """EnsembleClassifier with 1 model returns that model's prediction exactly."""
    c = MockClassifier(0.42)
    ensemble = EnsembleClassifier(classifiers=[c], weights=[1.0], live_mode=False)
    assert ensemble.predict(torch.zeros(1)) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def test_model_registry() -> None:
    """register_model then load_model calls the loader function."""
    called_with: list[str] = []

    def loader(path: str) -> MockClassifier:
        called_with.append(path)
        return MockClassifier(0.5)

    register_model("test_type", loader)
    clf = load_model("test_type", "/fake/path")
    assert called_with == ["/fake/path"]
    assert clf.predict(torch.zeros(1)) == 0.5


def test_registry_unknown_type() -> None:
    """load_model with unknown type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model type: unknown"):
        load_model("unknown", "/fake")


def test_research_cnn_registered() -> None:
    """After importing ensemble module, 'research_cnn' is in the registry."""
    from acoustic.classification.ensemble import get_registered_types

    assert "research_cnn" in get_registered_types()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_config_parsing(tmp_path: Path) -> None:
    """EnsembleConfig.from_file loads JSON correctly."""
    config_data = {
        "models": [
            {"type": "research_cnn", "path": "m.pt", "weight": 0.85},
        ]
    }
    config_file = tmp_path / "ensemble.json"
    config_file.write_text(json.dumps(config_data))

    config = EnsembleConfig.from_file(config_file)
    assert len(config.models) == 1
    assert config.models[0].type == "research_cnn"
    assert config.models[0].path == "m.pt"
    assert config.models[0].weight == pytest.approx(0.85)


def test_config_invalid_json(tmp_path: Path) -> None:
    """EnsembleConfig.from_file with invalid JSON raises appropriate error."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all")
    with pytest.raises(Exception):
        EnsembleConfig.from_file(bad_file)


# ---------------------------------------------------------------------------
# Live mode cap
# ---------------------------------------------------------------------------

def test_live_mode_cap() -> None:
    """EnsembleClassifier with 4 models in live mode raises ValueError."""
    classifiers = [MockClassifier(0.5) for _ in range(4)]
    weights = [0.8, 0.7, 0.6, 0.5]
    with pytest.raises(ValueError, match="max 3 models"):
        EnsembleClassifier(classifiers=classifiers, weights=weights, live_mode=True)


def test_offline_no_cap() -> None:
    """EnsembleClassifier with 5 models in offline mode succeeds."""
    classifiers = [MockClassifier(0.5) for _ in range(5)]
    weights = [0.8, 0.7, 0.6, 0.5, 0.4]
    ensemble = EnsembleClassifier(classifiers=classifiers, weights=weights, live_mode=False)
    assert ensemble.model_count == 5


# ---------------------------------------------------------------------------
# Protocol compliance and validation
# ---------------------------------------------------------------------------

def test_protocol_compliance() -> None:
    """EnsembleClassifier satisfies the Classifier protocol."""
    c = MockClassifier(0.5)
    ensemble = EnsembleClassifier(classifiers=[c], weights=[1.0], live_mode=False)
    assert isinstance(ensemble, Classifier)


def test_mismatched_lengths() -> None:
    """EnsembleClassifier raises ValueError when classifiers and weights have different lengths."""
    c1 = MockClassifier(0.5)
    c2 = MockClassifier(0.6)
    with pytest.raises(ValueError, match="same length"):
        EnsembleClassifier(classifiers=[c1, c2], weights=[0.5])
