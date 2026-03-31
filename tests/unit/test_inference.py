"""Tests for ONNX-based drone classifier."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DUMMY_MODEL = FIXTURES / "dummy_model.onnx"


class TestOnnxDroneClassifier:
    def test_load_success(self):
        from acoustic.classification.inference import OnnxDroneClassifier

        clf = OnnxDroneClassifier(str(DUMMY_MODEL))
        assert clf is not None

    def test_load_bad_path(self):
        from acoustic.classification.inference import OnnxDroneClassifier

        with pytest.raises(FileNotFoundError):
            OnnxDroneClassifier("/nonexistent/model.onnx")

    def test_predict_returns_float_in_range(self):
        from acoustic.classification.inference import OnnxDroneClassifier

        clf = OnnxDroneClassifier(str(DUMMY_MODEL))
        inp = np.random.randn(1, 128, 64, 1).astype(np.float32)
        result = clf.predict(inp)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_accepts_correct_shape(self):
        from acoustic.classification.inference import OnnxDroneClassifier

        clf = OnnxDroneClassifier(str(DUMMY_MODEL))
        inp = np.zeros((1, 128, 64, 1), dtype=np.float32)
        result = clf.predict(inp)
        assert isinstance(result, float)
