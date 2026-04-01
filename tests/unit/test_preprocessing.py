"""Tests for ResearchPreprocessor -- torchaudio mel-spectrogram pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import ResearchPreprocessor
from acoustic.classification.protocols import Preprocessor


def _sine_440(sr: int = 16000, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


class TestResearchPreprocessor:
    def test_output_shape(self):
        pp = ResearchPreprocessor()
        out = pp.process(_sine_440(), 16000)
        assert out.shape == (1, 1, 128, 64)

    def test_output_dtype(self):
        pp = ResearchPreprocessor()
        out = pp.process(_sine_440(), 16000)
        assert out.dtype == torch.float32

    def test_values_in_unit_range(self):
        pp = ResearchPreprocessor()
        out = pp.process(_sine_440(), 16000)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_satisfies_preprocessor_protocol(self):
        pp = ResearchPreprocessor()
        assert isinstance(pp, Preprocessor)

    def test_short_audio_padded(self):
        pp = ResearchPreprocessor()
        short = _sine_440(duration=0.1)
        out = pp.process(short, 16000)
        assert out.shape == (1, 1, 128, 64)

    def test_48khz_resampled(self):
        pp = ResearchPreprocessor()
        audio_48k = _sine_440(sr=48000, duration=0.5)
        out = pp.process(audio_48k, 48000)
        assert out.shape == (1, 1, 128, 64)

    def test_silence_valid_output(self):
        pp = ResearchPreprocessor()
        silence = np.zeros(8000, dtype=np.float32)
        out = pp.process(silence, 16000)
        assert out.shape == (1, 1, 128, 64)
        # Output must be in valid [0, 1] range even for silence
        assert out.min() >= 0.0
        assert out.max() <= 1.0
