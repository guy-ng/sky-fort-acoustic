"""ONNX Runtime-based drone classifier."""

from __future__ import annotations

import logging
import os

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class OnnxDroneClassifier:
    """Binary drone/not-drone classifier using an ONNX model.

    Accepts preprocessed mel-spectrogram input of shape (1, 128, 64, 1)
    and returns a drone probability in [0.0, 1.0].
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("Loaded ONNX drone classifier from %s", model_path)

    def predict(self, preprocessed: np.ndarray) -> float:
        """Run inference on preprocessed input.

        Args:
            preprocessed: Array of shape (1, 128, 64, 1) float32.

        Returns:
            Drone probability clamped to [0.0, 1.0].
        """
        outputs = self._session.run(None, {self._input_name: preprocessed})
        raw = float(outputs[0].flat[0])
        return max(0.0, min(1.0, raw))
