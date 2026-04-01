"""Tests for classification protocol definitions."""

from __future__ import annotations

import numpy as np
import torch

from acoustic.classification.protocols import Aggregator, Classifier, Preprocessor


class TestPreprocessorProtocol:
    """Verify Preprocessor protocol runtime checking."""

    def test_conforming_class_satisfies(self):
        class GoodPreprocessor:
            def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
                return torch.zeros(1)

        assert isinstance(GoodPreprocessor(), Preprocessor)

    def test_missing_method_fails(self):
        class BadPreprocessor:
            pass

        assert not isinstance(BadPreprocessor(), Preprocessor)


class TestClassifierProtocol:
    """Verify Classifier protocol runtime checking."""

    def test_conforming_class_satisfies(self):
        class GoodClassifier:
            def predict(self, features: torch.Tensor) -> float:
                return 0.5

        assert isinstance(GoodClassifier(), Classifier)

    def test_missing_method_fails(self):
        class BadClassifier:
            pass

        assert not isinstance(BadClassifier(), Classifier)


class TestAggregatorProtocol:
    """Verify Aggregator protocol runtime checking."""

    def test_conforming_class_satisfies(self):
        class GoodAggregator:
            def aggregate(self, probabilities: list[float]) -> float:
                return 0.5

        assert isinstance(GoodAggregator(), Aggregator)

    def test_missing_method_fails(self):
        class BadAggregator:
            pass

        assert not isinstance(BadAggregator(), Aggregator)
