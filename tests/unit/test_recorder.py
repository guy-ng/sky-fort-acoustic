"""Unit tests for RecordingSession: mono downmix, resample, WAV write."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acoustic.recording.recorder import RecordingSession


class TestRecordingSession:
    """RecordingSession captures 16-channel 48kHz audio as mono 16kHz WAV."""

    def _make_chunk(self, duration_s: float = 0.15, sr: int = 48000, channels: int = 16) -> np.ndarray:
        """Create a synthetic 16-channel audio chunk."""
        samples = int(sr * duration_s)
        t = np.linspace(0, duration_s, samples, dtype=np.float32)
        # 440 Hz sine wave on all channels
        signal = np.sin(2 * np.pi * 440 * t)
        return np.tile(signal[:, np.newaxis], (1, channels))

    def test_start_creates_wav_file(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()
        assert wav_path.exists()
        session.stop()

    def test_write_chunk_and_stop_produces_valid_wav(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        chunk = self._make_chunk()
        session.write_chunk(chunk)
        duration = session.stop()

        # WAV is readable
        data, sr = sf.read(str(wav_path))
        assert sr == 16000
        assert data.ndim == 1  # mono
        assert len(data) > 0
        assert duration > 0

    def test_mono_downmix(self, tmp_path):
        """16 channels averaged to mono."""
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path, gain_db=0.0)
        session.start()

        # Create chunk with known values: channel 0=1.0, channel 1=3.0, rest=0
        samples = 7200  # 150ms at 48kHz
        chunk = np.zeros((samples, 16), dtype=np.float32)
        chunk[:, 0] = 1.0
        chunk[:, 1] = 3.0
        # Mean should be (1+3+0*14)/16 = 0.25
        session.write_chunk(chunk)
        session.stop()

        data, sr = sf.read(str(wav_path))
        # After resample, signal should be around 0.25 (constant DC)
        assert np.abs(data.mean() - 0.25) < 0.05

    def test_resample_48k_to_16k(self, tmp_path):
        """Output sample count is ~1/3 of input (48kHz -> 16kHz)."""
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        # 48000 samples of 16-channel audio = 1 second
        chunk = self._make_chunk(duration_s=1.0)
        session.write_chunk(chunk)
        session.stop()

        data, sr = sf.read(str(wav_path))
        # Should be ~16000 samples for 1 second at 16kHz
        assert sr == 16000
        assert abs(len(data) - 16000) < 10  # allow small rounding

    def test_duration_property(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        chunk = self._make_chunk(duration_s=0.15)
        session.write_chunk(chunk)
        # 150ms at 48kHz -> 50ms at 16kHz = 0.05s worth of samples
        assert session.duration_s > 0
        assert session.duration_s == pytest.approx(0.15, abs=0.02)

        session.stop()

    def test_stop_returns_duration(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        chunk = self._make_chunk(duration_s=0.3)
        session.write_chunk(chunk)

        duration = session.stop()
        assert duration == pytest.approx(0.3, abs=0.02)

    def test_running_property(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        assert not session.running
        session.start()
        assert session.running
        session.stop()
        assert not session.running

    def test_rms_db_updates(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        chunk = self._make_chunk()
        session.write_chunk(chunk)
        # Should be a reasonable dB value (not -100 default)
        assert session.rms_db > -100.0
        session.stop()

    def test_write_chunk_after_stop_is_noop(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()
        session.stop()

        # Writing after stop should not raise
        chunk = self._make_chunk()
        session.write_chunk(chunk)

    def test_multiple_chunks(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        session = RecordingSession(output_path=wav_path)
        session.start()

        for _ in range(5):
            chunk = self._make_chunk(duration_s=0.15)
            session.write_chunk(chunk)

        duration = session.stop()
        assert duration == pytest.approx(0.75, abs=0.05)

        data, sr = sf.read(str(wav_path))
        assert sr == 16000
        assert len(data) > 0
