"""Tests for Classifier and Preprocessor protocols."""

import numpy as np
import torch

from acoustic.classification.protocols import Classifier, Preprocessor


class TestPreprocessorProtocol:
    def test_conforming_class_satisfies(self):
        class Good:
            def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
                return torch.zeros(1, 1, 128, 64)

        assert isinstance(Good(), Preprocessor)

    def test_missing_method_fails(self):
        class Bad:
            pass

        assert not isinstance(Bad(), Preprocessor)


class TestClassifierProtocol:
    def test_conforming_class_satisfies(self):
        class Good:
            def predict(self, features: torch.Tensor) -> float:
                return 0.5

        assert isinstance(Good(), Classifier)

    def test_missing_method_fails(self):
        class Bad:
            pass

        assert not isinstance(Bad(), Classifier)
