"""Integration tests for CNN classification pipeline integration."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from acoustic.classification.state_machine import DetectionState, DetectionStateMachine
from acoustic.classification.worker import ClassificationResult, CNNWorker
from acoustic.config import AcousticSettings
from acoustic.pipeline import BeamformingPipeline
from acoustic.tracking.events import EventBroadcaster
from acoustic.tracking.tracker import TargetTracker


@pytest.fixture
def settings():
    return AcousticSettings(audio_source="simulated")


@pytest.fixture
def mock_classifier():
    """Mock ONNX classifier that always returns 0.9 probability."""
    clf = MagicMock()
    clf.predict.return_value = 0.9
    return clf


class TestCNNWorker:
    """Tests for the CNNWorker background thread."""

    def test_worker_start_stop(self, mock_classifier):
        worker = CNNWorker(mock_classifier, fs_in=48000)
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        worker.stop()
        assert worker._thread is None

    def test_push_and_get_latest(self, mock_classifier):
        worker = CNNWorker(mock_classifier, fs_in=16000)
        worker.start()
        try:
            # Push a 2-second mono segment
            mono = np.random.randn(32000).astype(np.float32)
            worker.push(mono, az_deg=10.0, el_deg=5.0)
            # Wait for inference (librosa mel-spectrogram is slow on first call)
            time.sleep(3.0)
            result = worker.get_latest()
            assert result is not None
            assert isinstance(result, ClassificationResult)
            assert result.drone_probability == 0.9
            assert result.az_deg == 10.0
            assert result.el_deg == 5.0
        finally:
            worker.stop()

    def test_get_latest_returns_none_initially(self, mock_classifier):
        worker = CNNWorker(mock_classifier, fs_in=48000)
        assert worker.get_latest() is None


class TestPipelineWithCNN:
    """Tests for BeamformingPipeline with CNN integration."""

    def test_pipeline_initializes_with_cnn(self, settings, mock_classifier):
        worker = CNNWorker(mock_classifier, fs_in=settings.sample_rate)
        sm = DetectionStateMachine()
        broadcaster = EventBroadcaster()
        tracker = TargetTracker(ttl=5.0, broadcaster=broadcaster)
        pipeline = BeamformingPipeline(
            settings, cnn_worker=worker, state_machine=sm, tracker=tracker
        )
        assert pipeline._cnn_worker is worker
        assert pipeline._state_machine is sm
        assert pipeline._tracker is tracker

    def test_pipeline_initializes_without_cnn(self, settings):
        pipeline = BeamformingPipeline(settings)
        assert pipeline._cnn_worker is None
        assert pipeline._state_machine is None
        assert pipeline._tracker is None

    def test_latest_targets_returns_list(self, settings):
        pipeline = BeamformingPipeline(settings)
        targets = pipeline.latest_targets
        assert isinstance(targets, list)

    def test_latest_targets_with_tracker(self, settings, mock_classifier):
        broadcaster = EventBroadcaster()
        tracker = TargetTracker(ttl=5.0, broadcaster=broadcaster)
        pipeline = BeamformingPipeline(
            settings,
            cnn_worker=CNNWorker(mock_classifier, fs_in=settings.sample_rate),
            state_machine=DetectionStateMachine(),
            tracker=tracker,
        )
        # No targets initially
        assert pipeline.latest_targets == []

        # Simulate adding a target directly
        tracker.update(az_deg=10.0, el_deg=5.0, confidence=0.95)
        targets = pipeline.latest_targets
        assert len(targets) == 1
        assert targets[0]["class_label"] == "drone"
        assert "id" in targets[0]
        assert "az_deg" in targets[0]
        assert "el_deg" in targets[0]
        assert "confidence" in targets[0]
        assert "speed_mps" in targets[0]

    def test_latest_targets_fallback_placeholder(self, settings):
        """Without tracker, falls back to placeholder_target_from_peak."""
        from acoustic.types import PeakDetection

        pipeline = BeamformingPipeline(settings)
        # Set a peak manually
        pipeline.latest_peak = PeakDetection(az_deg=15.0, el_deg=3.0, power=5.0, threshold=2.0)
        targets = pipeline.latest_targets
        assert len(targets) == 1
        assert targets[0]["class_label"] == "unknown"
        assert targets[0]["az_deg"] == 15.0

    def test_process_cnn_accumulates_mono_buffer(self, settings, mock_classifier):
        """Verify that _process_cnn accumulates mono audio on peak detection."""
        from acoustic.types import PeakDetection

        worker = CNNWorker(mock_classifier, fs_in=settings.sample_rate)
        sm = DetectionStateMachine()
        pipeline = BeamformingPipeline(
            settings, cnn_worker=worker, state_machine=sm, tracker=None
        )
        # Create a fake chunk (chunk_samples x 16 channels)
        chunk = np.random.randn(settings.chunk_samples, settings.num_channels).astype(np.float32)
        peak = PeakDetection(az_deg=10.0, el_deg=5.0, power=3.0, threshold=1.0)

        pipeline._process_cnn(chunk, peak)
        assert pipeline._mono_buffer_samples == settings.chunk_samples
        assert len(pipeline._mono_buffer) == 1

    def test_process_cnn_clears_buffer_on_no_peak(self, settings, mock_classifier):
        """Verify that mono buffer is cleared when no peak is detected."""
        worker = CNNWorker(mock_classifier, fs_in=settings.sample_rate)
        pipeline = BeamformingPipeline(settings, cnn_worker=worker)

        # Pre-fill some buffer
        pipeline._mono_buffer = [np.zeros(100, dtype=np.float32)]
        pipeline._mono_buffer_samples = 100

        pipeline._process_cnn(np.zeros((100, 16), dtype=np.float32), None)
        assert pipeline._mono_buffer_samples == 0
        assert len(pipeline._mono_buffer) == 0


class TestGracefulDegradation:
    """Test that the pipeline works gracefully without CNN model."""

    @pytest.mark.asyncio
    async def test_app_starts_without_cnn_model(self, running_app):
        """Service starts even when CNN model file is missing."""
        app = running_app
        assert app.state.pipeline is not None
        assert app.state.pipeline.running
        # event_broadcaster may or may not be set depending on model availability
        # but the app should not crash
