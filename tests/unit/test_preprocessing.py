"""Tests for CNN preprocessing pipeline — mel-spectrogram feature extraction."""

from __future__ import annotations

import numpy as np
import pytest


def _synth_mono(duration_s: float, sr: int) -> np.ndarray:
    """Generate a synthetic mono signal (440 Hz sine)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


class TestFastResample:
    def test_downsample_48k_to_16k(self):
        from acoustic.classification.preprocessing import fast_resample

        y = _synth_mono(1.0, 48000)
        out = fast_resample(y, 48000, 16000)
        expected_len = len(y) // 3
        assert abs(len(out) - expected_len) <= 1
        assert out.dtype == np.float32

    def test_same_rate_returns_unchanged(self):
        from acoustic.classification.preprocessing import fast_resample

        y = _synth_mono(0.5, 16000)
        out = fast_resample(y, 16000, 16000)
        np.testing.assert_array_equal(out, y)


class TestMakeMelspec:
    def test_output_shape(self):
        from acoustic.classification.preprocessing import make_melspec

        y = np.zeros(16000 * 2, dtype=np.float32)  # 2s at 16kHz
        spec = make_melspec(y, 16000)
        assert spec.ndim == 2
        assert spec.shape[1] == 64  # n_mels


class TestPadOrTrim:
    def test_pad_short(self):
        from acoustic.classification.preprocessing import pad_or_trim

        spec = np.ones((50, 64), dtype=np.float32)
        out = pad_or_trim(spec, 128)
        assert out.shape == (128, 64)
        # Padded region should be zeros
        np.testing.assert_array_equal(out[50:, :], 0.0)

    def test_trim_long(self):
        from acoustic.classification.preprocessing import pad_or_trim

        spec = np.ones((200, 64), dtype=np.float32)
        out = pad_or_trim(spec, 128)
        assert out.shape == (128, 64)


class TestNormSpec:
    def test_normalized_stats(self):
        from acoustic.classification.preprocessing import norm_spec

        rng = np.random.default_rng(42)
        spec = rng.standard_normal((128, 64)).astype(np.float32) * 10 + 5
        out = norm_spec(spec)
        assert abs(float(out.mean())) < 0.1
        assert abs(float(out.std()) - 1.0) < 0.1


class TestPreprocessForCnn:
    def test_full_pipeline_shape(self):
        from acoustic.classification.preprocessing import preprocess_for_cnn

        y = _synth_mono(2.0, 48000)
        out = preprocess_for_cnn(y, 48000)
        assert out.shape == (1, 3, 224, 224)
        assert out.dtype == np.float32

    def test_short_audio_pads(self):
        from acoustic.classification.preprocessing import preprocess_for_cnn

        y = _synth_mono(0.5, 48000)
        out = preprocess_for_cnn(y, 48000)
        assert out.shape == (1, 3, 224, 224)
