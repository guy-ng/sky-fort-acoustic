"""Tests for CNNWorker protocol injection, constructor, and segment buffer."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from acoustic.classification.protocols import Aggregator, Classifier, Preprocessor
from acoustic.classification.worker import CNNWorker


class FakePreprocessor:
    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        return torch.zeros(1, 1, 128, 64)


class FakeClassifier:
    def predict(self, features: torch.Tensor) -> float:
        return 0.9


class FakeAggregator:
    def aggregate(self, probabilities: list[float]) -> float:
        return sum(probabilities) / len(probabilities) if probabilities else 0.0


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
        assert isinstance(FakePreprocessor(), Preprocessor)
        worker = CNNWorker(preprocessor=FakePreprocessor(), classifier=None)
        assert worker._preprocessor is not None

    def test_accepts_protocol_typed_classifier(self):
        """CNNWorker accepts an object satisfying Classifier protocol."""
        assert isinstance(FakeClassifier(), Classifier)
        worker = CNNWorker(preprocessor=None, classifier=FakeClassifier())
        assert worker._classifier is not None

    def test_accepts_aggregator(self):
        """CNNWorker accepts an object satisfying Aggregator protocol."""
        assert isinstance(FakeAggregator(), Aggregator)
        worker = CNNWorker(aggregator=FakeAggregator())
        assert worker._aggregator is not None

    def test_default_aggregator_none(self):
        """CNNWorker() has _aggregator == None by default."""
        worker = CNNWorker()
        assert worker._aggregator is None

    def test_default_segment_buffer_size(self):
        """CNNWorker() has _segment_probs.maxlen == 4 by default."""
        worker = CNNWorker()
        assert worker._segment_probs.maxlen == 4


class TestSegmentBuffer:
    def test_segment_probs_deque_maxlen(self):
        """CNNWorker with segment_buffer_size=4 has _segment_probs deque with maxlen=4."""
        worker = CNNWorker(segment_buffer_size=4)
        assert worker._segment_probs.maxlen == 4

    def test_aggregation_applied(self):
        """After 4 pushes with classifier returning 0.9, result uses aggregated probability."""
        worker = CNNWorker(
            preprocessor=FakePreprocessor(),
            classifier=FakeClassifier(),
            aggregator=FakeAggregator(),
            fs_in=48000,
            segment_buffer_size=4,
        )
        worker.start()
        try:
            loud_audio = np.ones(24000, dtype=np.float32) * 0.1
            for _ in range(4):
                worker.push(loud_audio, az_deg=0.0, el_deg=0.0)
                time.sleep(0.5)
            result = worker.get_latest()
            assert result is not None
            # FakeAggregator returns mean of [0.9, 0.9, ...] = 0.9
            assert abs(result.drone_probability - 0.9) < 0.01
        finally:
            worker.stop()

    def test_deque_rolls_over(self):
        """After 5 pushes to a maxlen=4 deque, len(_segment_probs) == 4."""
        worker = CNNWorker(
            preprocessor=FakePreprocessor(),
            classifier=FakeClassifier(),
            aggregator=FakeAggregator(),
            fs_in=48000,
            segment_buffer_size=4,
        )
        worker.start()
        try:
            loud_audio = np.ones(24000, dtype=np.float32) * 0.1
            for _ in range(5):
                worker.push(loud_audio, az_deg=0.0, el_deg=0.0)
                time.sleep(0.5)
            assert len(worker._segment_probs) == 4
        finally:
            worker.stop()

    def test_aggregator_none_uses_raw_prob(self):
        """With aggregator=None, result uses raw classifier probability."""
        worker = CNNWorker(
            preprocessor=FakePreprocessor(),
            classifier=FakeClassifier(),
            aggregator=None,
            fs_in=48000,
        )
        worker.start()
        try:
            loud_audio = np.ones(24000, dtype=np.float32) * 0.1
            worker.push(loud_audio, az_deg=0.0, el_deg=0.0)
            time.sleep(1.0)
            result = worker.get_latest()
            assert result is not None
            assert result.drone_probability == 0.9
        finally:
            worker.stop()

    def test_silence_clears_nothing(self):
        """Silence-gated segment does not append to deque."""
        worker = CNNWorker(
            preprocessor=FakePreprocessor(),
            classifier=FakeClassifier(),
            aggregator=FakeAggregator(),
            fs_in=48000,
            silence_threshold=0.001,
        )
        worker.start()
        try:
            # Push silence (all zeros)
            silence = np.zeros(24000, dtype=np.float32)
            worker.push(silence, az_deg=0.0, el_deg=0.0)
            time.sleep(0.5)
            # Deque should remain empty
            assert len(worker._segment_probs) == 0
            # But result should have prob=0.0
            result = worker.get_latest()
            assert result is not None
            assert result.drone_probability == 0.0
        finally:
            worker.stop()


class TestPipelineSegmentDuration:
    def test_segment_uses_training_window(self):
        """Pipeline CNN window matches the model's training window; interval comes from settings."""
        from acoustic.config import AcousticSettings
        from acoustic.pipeline import BeamformingPipeline

        settings = AcousticSettings()
        pipe = BeamformingPipeline(settings)
        # Pre-session placeholder is research_cnn training window (0.5s)
        assert pipe._cnn_segment_samples == int(settings.sample_rate * 0.5)
        assert pipe._cnn_interval == settings.cnn_interval_seconds
