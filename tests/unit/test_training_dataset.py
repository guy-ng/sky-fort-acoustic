"""Unit tests for mel_spectrogram_from_segment, DroneAudioDataset, and utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def mel_config():
    from acoustic.classification.config import MelConfig

    return MelConfig()


@pytest.fixture
def tmp_dataset(tmp_path: Path):
    """Create a temporary dataset with drone and background WAV files."""
    rng = np.random.default_rng(42)

    drone_dir = tmp_path / "drone"
    drone_dir.mkdir()
    bg_dir = tmp_path / "background"
    bg_dir.mkdir()

    # 2 drone files, 2 background files (1s at 16kHz)
    for i in range(2):
        sf.write(str(drone_dir / f"d{i}.wav"), rng.standard_normal(16000).astype(np.float32), 16000)
        sf.write(str(bg_dir / f"b{i}.wav"), rng.standard_normal(16000).astype(np.float32), 16000)

    return tmp_path


class TestMelSpectrogramFromSegment:
    """Tests for the standalone mel_spectrogram_from_segment utility."""

    def test_output_shape(self, mel_config):
        from acoustic.classification.preprocessing import mel_spectrogram_from_segment

        segment = np.random.randn(8000).astype(np.float32)
        out = mel_spectrogram_from_segment(segment, mel_config)
        assert out.shape == (1, 1, 128, 64)

    def test_output_range(self, mel_config):
        from acoustic.classification.preprocessing import mel_spectrogram_from_segment

        segment = np.random.randn(8000).astype(np.float32)
        out = mel_spectrogram_from_segment(segment, mel_config)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_parity_with_preprocessor(self, mel_config):
        """Same segment produces same output as ResearchPreprocessor.process()."""
        from acoustic.classification.preprocessing import (
            ResearchPreprocessor,
            mel_spectrogram_from_segment,
        )

        # Create segment of exact segment_samples length
        n = mel_config.segment_samples
        segment = np.random.randn(n).astype(np.float32)

        # mel_spectrogram_from_segment
        out_util = mel_spectrogram_from_segment(segment, mel_config)

        # ResearchPreprocessor.process — takes the last n samples
        proc = ResearchPreprocessor(mel_config)
        out_proc = proc.process(segment, mel_config.sample_rate)

        assert torch.allclose(out_util, out_proc, atol=1e-5), (
            f"Utility and preprocessor outputs differ: max diff = {(out_util - out_proc).abs().max()}"
        )


class TestCollectWavFiles:
    """Tests for collect_wav_files utility."""

    def test_scans_and_labels(self, tmp_dataset):
        from acoustic.training.dataset import collect_wav_files

        label_map = {"drone": 1, "background": 0}
        paths, labels = collect_wav_files(str(tmp_dataset), label_map)
        assert len(paths) == 4
        assert len(labels) == 4
        assert sum(1 for l in labels if l == 1) == 2  # 2 drone files
        assert sum(1 for l in labels if l == 0) == 2  # 2 background files

    def test_label_mapping(self, tmp_dataset):
        from acoustic.training.dataset import collect_wav_files

        label_map = {"drone": 1, "background": 0, "other": 0}
        paths, labels = collect_wav_files(str(tmp_dataset), label_map)
        # Only drone and background dirs exist; "other" is just skipped
        assert len(paths) == 4

    def test_skips_unknown_dirs(self, tmp_dataset):
        from acoustic.training.dataset import collect_wav_files

        # Create an "unknown" dir that's not in label_map
        (tmp_dataset / "unknown").mkdir()
        sf.write(
            str(tmp_dataset / "unknown" / "u.wav"),
            np.random.randn(16000).astype(np.float32),
            16000,
        )
        label_map = {"drone": 1, "background": 0}
        paths, labels = collect_wav_files(str(tmp_dataset), label_map)
        assert len(paths) == 4  # unknown dir skipped


class TestDroneAudioDataset:
    """Tests for DroneAudioDataset."""

    def test_len(self, tmp_dataset, mel_config):
        from acoustic.training.dataset import DroneAudioDataset, collect_wav_files

        paths, labels = collect_wav_files(str(tmp_dataset), {"drone": 1, "background": 0})
        ds = DroneAudioDataset(paths, labels, mel_config)
        assert len(ds) == 4

    def test_getitem_shape(self, tmp_dataset, mel_config):
        from acoustic.training.dataset import DroneAudioDataset, collect_wav_files

        paths, labels = collect_wav_files(str(tmp_dataset), {"drone": 1, "background": 0})
        ds = DroneAudioDataset(paths, labels, mel_config)
        features, label = ds[0]
        assert features.shape == (1, 128, 64)
        assert label.dtype == torch.float32

    def test_random_segment_extraction(self, tmp_dataset, mel_config):
        from acoustic.training.dataset import DroneAudioDataset, collect_wav_files

        paths, labels = collect_wav_files(str(tmp_dataset), {"drone": 1, "background": 0})
        ds = DroneAudioDataset(paths, labels, mel_config)
        t1, _ = ds[0]
        t2, _ = ds[0]
        # With 1s audio and 0.5s segments, random starts should yield different results
        assert not torch.allclose(t1, t2), "Repeated calls should yield different segments"

    def test_no_augmentation(self, tmp_dataset, mel_config):
        from acoustic.training.dataset import DroneAudioDataset, collect_wav_files

        paths, labels = collect_wav_files(str(tmp_dataset), {"drone": 1, "background": 0})
        ds = DroneAudioDataset(paths, labels, mel_config, waveform_aug=None, spec_aug=None)
        features, label = ds[0]
        assert features.shape == (1, 128, 64)

    def test_short_audio_pads(self, tmp_path, mel_config):
        """Audio shorter than 0.5s should be zero-padded, not crash."""
        from acoustic.training.dataset import DroneAudioDataset

        # 0.1s of audio at 16kHz = 1600 samples (less than 8000)
        short_dir = tmp_path / "short"
        short_dir.mkdir()
        sf.write(str(short_dir / "s.wav"), np.random.randn(1600).astype(np.float32), 16000)

        ds = DroneAudioDataset([short_dir / "s.wav"], [0], mel_config)
        features, label = ds[0]
        assert features.shape == (1, 128, 64)


class TestBuildWeightedSampler:
    """Tests for build_weighted_sampler."""

    def test_returns_sampler(self):
        from torch.utils.data import WeightedRandomSampler

        from acoustic.training.dataset import build_weighted_sampler

        labels = [0, 0, 0, 1, 1]
        sampler = build_weighted_sampler(labels)
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(labels)
