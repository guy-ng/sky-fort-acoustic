"""Unit tests for BackgroundNoiseMixer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


class TestBackgroundNoiseMixer:
    """Tests for BackgroundNoiseMixer class."""

    def _make_wav(self, path: Path, sr: int = 16000, duration: float = 1.0) -> None:
        """Create a synthetic WAV file with a sine wave."""
        t = np.arange(int(sr * duration)) / sr
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(path), audio, sr)

    def test_empty_noise_dir(self):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        mixer = BackgroundNoiseMixer(noise_dirs=[], snr_range=(-10.0, 20.0))
        audio = np.random.randn(8000).astype(np.float32)
        out = mixer(audio)
        np.testing.assert_array_equal(out, audio)

    def test_warm_cache_loads_files(self, tmp_path: Path):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        self._make_wav(noise_dir / "test.wav", sr=16000, duration=1.0)

        mixer = BackgroundNoiseMixer(noise_dirs=[noise_dir], snr_range=(-10.0, 20.0))
        mixer.warm_cache()
        assert len(mixer._noise_cache) == 1

    def test_snr_range(self, tmp_path: Path):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        self._make_wav(noise_dir / "test.wav", sr=16000, duration=2.0)

        mixer = BackgroundNoiseMixer(noise_dirs=[noise_dir], snr_range=(0.0, 10.0), p=1.0)
        mixer.warm_cache()

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        out = mixer(audio)
        assert out.shape == audio.shape
        assert out.dtype == np.float32

    def test_resample(self, tmp_path: Path):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        # Create WAV at 44100 Hz, 1 second
        self._make_wav(noise_dir / "high_sr.wav", sr=44100, duration=1.0)

        mixer = BackgroundNoiseMixer(
            noise_dirs=[noise_dir], snr_range=(0.0, 10.0), sample_rate=16000
        )
        mixer.warm_cache()
        # 44100 samples at 44100 Hz = 1s -> resampled to 16000 samples at 16000 Hz
        cached = mixer._noise_cache[0]
        assert cached.shape[0] == 16000, f"Expected 16000 samples, got {cached.shape[0]}"

    def test_no_mix_when_probability_zero(self, tmp_path: Path):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        self._make_wav(noise_dir / "test.wav", sr=16000)

        mixer = BackgroundNoiseMixer(noise_dirs=[noise_dir], snr_range=(0.0, 10.0), p=0.0)
        mixer.warm_cache()

        audio = np.random.randn(8000).astype(np.float32)
        out = mixer(audio)
        np.testing.assert_array_equal(out, audio)

    def test_output_clipped(self, tmp_path: Path):
        from acoustic.training.augmentation import BackgroundNoiseMixer

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        # Create loud noise file
        loud_noise = np.ones(16000, dtype=np.float32)
        sf.write(str(noise_dir / "loud.wav"), loud_noise, 16000)

        mixer = BackgroundNoiseMixer(
            noise_dirs=[noise_dir], snr_range=(-10.0, -10.0), p=1.0
        )
        mixer.warm_cache()

        audio = np.ones(16000, dtype=np.float32) * 0.9
        out = mixer(audio)
        assert np.all(out >= -1.0) and np.all(out <= 1.0), "Output should be clipped to [-1, 1]"
