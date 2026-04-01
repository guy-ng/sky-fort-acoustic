"""Numerical parity tests: torchaudio preprocessor vs librosa reference fixtures.

Per D-07: Compare against .npy fixtures generated from research code.
No TensorFlow or librosa dependency needed at test time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acoustic.classification.preprocessing import ResearchPreprocessor

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestParityWithLibrosaReference:
    @pytest.fixture
    def preprocessor(self):
        return ResearchPreprocessor()

    @pytest.fixture
    def sine_440(self):
        t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    @pytest.fixture
    def reference_440(self):
        path = FIXTURES / "reference_melspec_440hz.npy"
        if not path.exists():
            pytest.skip("Reference fixture not found. Run scripts/generate_reference_fixtures.py")
        return np.load(path)

    def test_440hz_parity(self, preprocessor, sine_440, reference_440):
        """Core parity test: torchaudio output matches librosa reference within atol=1e-4."""
        out = preprocessor.process(sine_440, 16000)
        # out shape: (1, 1, 128, 64), reference shape: (128, 64)
        actual = out.squeeze().numpy()
        np.testing.assert_allclose(
            actual,
            reference_440,
            atol=1e-4,
            err_msg="Torchaudio preprocessor does not match librosa reference within atol=1e-4",
        )

    def test_output_shape_matches_reference(self, preprocessor, sine_440, reference_440):
        out = preprocessor.process(sine_440, 16000)
        actual = out.squeeze().numpy()
        assert actual.shape == reference_440.shape, (
            f"Shape mismatch: got {actual.shape}, expected {reference_440.shape}"
        )

    def test_value_range_matches_reference(self, preprocessor, sine_440, reference_440):
        out = preprocessor.process(sine_440, 16000)
        actual = out.squeeze().numpy()
        assert actual.min() >= 0.0
        assert actual.max() <= 1.0
        # Reference should also be in [0, 1]
        assert reference_440.min() >= 0.0
        assert reference_440.max() <= 1.0
