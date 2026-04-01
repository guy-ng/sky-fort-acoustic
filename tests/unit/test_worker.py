"""Tests for CNNWorker protocol injection and constructor."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic.classification.worker import CNNWorker


class TestCNNWorkerConstructor:
    def test_instantiate_with_none_deps(self):
        """CNNWorker can be created with preprocessor=None, classifier=None."""
        worker = CNNWorker(preprocessor=None, classifier=None)
        assert worker._preprocessor is None
        assert worker._classifier is None

    def test_no_onnx_import(self):
        """worker.py must not import OnnxDroneClassifier."""
        import acoustic.classification.worker as wmod

        source = open(wmod.__file__).read()
        assert "OnnxDroneClassifier" not in source
        assert "from acoustic.classification.inference" not in source

    def test_no_preprocess_for_cnn_import(self):
        """worker.py must not import old preprocess_for_cnn."""
        import acoustic.classification.worker as wmod

        source = open(wmod.__file__).read()
        assert "preprocess_for_cnn" not in source

    def test_accepts_protocol_typed_preprocessor(self):
        """CNNWorker accepts an object satisfying Preprocessor protocol."""
        import torch

        from acoustic.classification.protocols import Preprocessor

        class FakePreprocessor:
            def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
                return torch.zeros(1, 1, 128, 64)

        assert isinstance(FakePreprocessor(), Preprocessor)
        worker = CNNWorker(preprocessor=FakePreprocessor(), classifier=None)
        assert worker._preprocessor is not None

    def test_accepts_protocol_typed_classifier(self):
        """CNNWorker accepts an object satisfying Classifier protocol."""
        import torch

        from acoustic.classification.protocols import Classifier

        class FakeClassifier:
            def predict(self, features: torch.Tensor) -> float:
                return 0.5

        assert isinstance(FakeClassifier(), Classifier)
        worker = CNNWorker(preprocessor=None, classifier=FakeClassifier())
        assert worker._classifier is not None


class TestPipelineSegmentDuration:
    def test_segment_uses_half_second(self):
        """Pipeline must use 0.5s segments, not 2.0s."""
        import acoustic.pipeline as pmod

        source = open(pmod.__file__).read()
        assert "* 2.0" not in source, "Pipeline still uses 2.0s segments"
        assert "* 0.5" in source or "segment_seconds" in source, (
            "Pipeline must reference 0.5s segment duration"
        )
