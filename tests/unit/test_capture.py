"""Tests for AudioCapture and SimulatedAudioSource."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from acoustic.audio.capture import AudioCapture
from acoustic.audio.simulator import SimulatedAudioSource
from acoustic.config import AcousticSettings


class TestAudioCapture:
    """Tests for callback-based audio capture with mocked sounddevice."""

    @patch("acoustic.audio.capture.sd")
    def test_audio_capture_creates_stream(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture = AudioCapture(
            device="hw:2,0",
            fs=48000,
            channels=16,
            chunk_samples=7200,
            ring_chunks=14,
        )

        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args
        assert call_kwargs.kwargs["device"] == "hw:2,0"
        assert call_kwargs.kwargs["samplerate"] == 48000
        assert call_kwargs.kwargs["channels"] == 16
        assert call_kwargs.kwargs["blocksize"] == 7200
        assert call_kwargs.kwargs["dtype"] == "float32"
        assert call_kwargs.kwargs["callback"] == capture._callback

    @patch("acoustic.audio.capture.sd")
    def test_audio_capture_callback_writes_to_ring(self, mock_sd):
        mock_sd.InputStream.return_value = MagicMock()

        capture = AudioCapture(
            device=None,
            fs=48000,
            channels=16,
            chunk_samples=7200,
            ring_chunks=14,
        )

        # Simulate callback invocation
        test_data = np.random.randn(7200, 16).astype(np.float32)
        capture._callback(test_data, 7200, None, None)

        # Verify data is in ring buffer
        result = capture.ring.read()
        assert result is not None
        np.testing.assert_array_equal(result, test_data)


class TestSimulatedAudioSource:
    """Tests for simulated 16-channel audio generation."""

    def test_simulated_source_generates_correct_shape(self):
        settings = AcousticSettings()
        source = SimulatedAudioSource(settings)
        chunk = source.get_chunk()
        assert chunk.shape == (7200, 16)
        assert chunk.dtype == np.float32

    def test_simulated_source_direction(self):
        """Generate chunk at az=30 deg; verify the signal has directional content.

        We check that the signal is not all zeros and has reasonable amplitude.
        Full SRP-PHAT validation is deferred to beamforming tests in plan 01-02.
        """
        settings = AcousticSettings()
        source = SimulatedAudioSource(settings)
        chunk = source.get_chunk(source_az_deg=30.0, source_el_deg=0.0, freq=500.0)
        assert chunk.shape == (7200, 16)
        # Signal should have non-trivial amplitude
        assert np.max(np.abs(chunk)) > 0.1
        # Different channels should have slightly different phases (time delays)
        # Check that not all channels are identical
        assert not np.allclose(chunk[:, 0], chunk[:, 8], atol=1e-6)
