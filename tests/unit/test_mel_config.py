"""Tests for MelConfig frozen dataclass with research constants."""

import pytest

from acoustic.classification.config import MelConfig


class TestMelConfigDefaults:
    def test_sample_rate(self):
        assert MelConfig().sample_rate == 16000

    def test_n_fft(self):
        assert MelConfig().n_fft == 1024

    def test_hop_length(self):
        assert MelConfig().hop_length == 256

    def test_n_mels(self):
        assert MelConfig().n_mels == 64

    def test_max_frames(self):
        assert MelConfig().max_frames == 128

    def test_segment_seconds(self):
        assert MelConfig().segment_seconds == 0.5

    def test_db_range(self):
        assert MelConfig().db_range == 80.0

    def test_segment_samples(self):
        assert MelConfig().segment_samples == 8000

    def test_frozen(self):
        c = MelConfig()
        with pytest.raises(AttributeError):
            c.sample_rate = 44100
