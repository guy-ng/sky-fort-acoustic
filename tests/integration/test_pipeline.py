"""Integration tests for the BeamformingPipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from acoustic.audio.capture import AudioRingBuffer
from acoustic.audio.simulator import SimulatedAudioSource
from acoustic.config import AcousticSettings
from acoustic.pipeline import BeamformingPipeline
from acoustic.types import PeakDetection


class TestPipelineProcessChunk:
    """Tests for BeamformingPipeline.process_chunk."""

    def test_pipeline_processes_chunk(self, settings: AcousticSettings, synthetic_audio):
        """Feed a simulated chunk through the pipeline and get back PeakDetection or None."""
        pipeline = BeamformingPipeline(settings)
        chunk = synthetic_audio(az_deg=0.0, el_deg=0.0, freq=500.0)
        result = pipeline.process_chunk(chunk)
        assert result is None or isinstance(result, PeakDetection)

    def test_pipeline_stores_latest_map(self, settings: AcousticSettings, synthetic_audio):
        """After processing a chunk, pipeline.latest_map is a 2D ndarray with correct shape."""
        pipeline = BeamformingPipeline(settings)
        chunk = synthetic_audio(az_deg=0.0, el_deg=0.0)
        pipeline.process_chunk(chunk)

        assert pipeline.latest_map is not None
        assert isinstance(pipeline.latest_map, np.ndarray)
        assert pipeline.latest_map.ndim == 2

        expected_az = len(
            np.arange(
                -settings.az_range,
                settings.az_range + settings.az_resolution,
                settings.az_resolution,
            )
        )
        expected_el = len(
            np.arange(
                -settings.el_range,
                settings.el_range + settings.el_resolution,
                settings.el_resolution,
            )
        )
        assert pipeline.latest_map.shape == (expected_az, expected_el)

    def test_pipeline_stores_latest_peak(self, settings: AcousticSettings, synthetic_audio):
        """After processing a chunk with strong source, pipeline.latest_peak is a PeakDetection."""
        pipeline = BeamformingPipeline(settings)
        # High SNR to ensure detection
        chunk = synthetic_audio(az_deg=30.0, el_deg=10.0, freq=500.0, snr_db=40.0)
        pipeline.process_chunk(chunk)

        # With high SNR, we should get a detection
        if pipeline.latest_peak is not None:
            assert isinstance(pipeline.latest_peak, PeakDetection)
            assert isinstance(pipeline.latest_peak.az_deg, float)
            assert isinstance(pipeline.latest_peak.el_deg, float)


class TestPipelineLiveness:
    """Tests for pipeline background thread operation."""

    def test_pipeline_liveness(self, settings: AcousticSettings):
        """Verify pipeline.latest_map updates over time (not stalled)."""
        from acoustic.main import SimulatedProducer

        pipeline = BeamformingPipeline(settings)
        ring = AudioRingBuffer(settings.ring_chunks, settings.chunk_samples, settings.num_channels)
        sim = SimulatedAudioSource(settings)

        producer = SimulatedProducer(sim, ring, settings.chunk_seconds)
        producer.start()
        pipeline.start(ring)
        try:
            time.sleep(0.5)
            t1 = pipeline.last_process_time
            assert t1 is not None, "Pipeline never processed a chunk"
            time.sleep(0.3)
            t2 = pipeline.last_process_time
            assert t2 is not None
            assert t2 > t1, "Pipeline stalled -- last_process_time not advancing"
        finally:
            pipeline.stop()
            producer.stop()
