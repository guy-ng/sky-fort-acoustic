"""D-05/D-07: ONNX inference latency + fallback on host (proxy for Pi 4 budget).

Owner: Plan 21-05 (skyfort_edge/inference.py).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from skyfort_edge.config import ModelConfig
from skyfort_edge.inference import OnnxClassifier
from skyfort_edge.preprocess import NumpyMelSTFT

REPO_ROOT = Path(__file__).resolve().parents[3]
FP32 = REPO_ROOT / "models" / "efficientat_mn10_v6_fp32.onnx"
INT8 = REPO_ROOT / "models" / "efficientat_mn10_v6_int8.onnx"


@pytest.fixture
def int8_classifier():
    if not INT8.exists() or not FP32.exists():
        pytest.skip("ONNX artifacts not built yet -- run Plan 21-03 first")
    pytest.importorskip("onnxruntime")
    cfg = ModelConfig(
        onnx_path=str(INT8),
        fallback_onnx_path=str(FP32),
        prefer_int8=True,
        num_threads=2,
    )
    return OnnxClassifier(cfg)


def test_int8_onnx_inference_under_150ms(int8_classifier):
    pp = NumpyMelSTFT()
    silence = np.zeros(32000, dtype=np.float32)
    mel = pp.forward(silence)
    assert mel.shape[0] == 128
    # Warmup
    for _ in range(3):
        int8_classifier.classify(mel)
    times: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        int8_classifier.classify(mel)
        times.append(time.perf_counter() - start)
    p50 = sorted(times)[len(times) // 2]
    assert p50 < 0.150, f"int8 p50 latency {p50 * 1000:.1f} ms >= 150 ms on host"


def test_fallback_to_fp32_on_missing_int8(tmp_path):
    if not FP32.exists():
        pytest.skip("FP32 artifact missing -- run Plan 21-03 first")
    pytest.importorskip("onnxruntime")
    cfg = ModelConfig(
        onnx_path=str(tmp_path / "nonexistent.onnx"),
        fallback_onnx_path=str(FP32),
        prefer_int8=True,
        num_threads=2,
    )
    classifier = OnnxClassifier(cfg)
    assert classifier.active_model_path == FP32


def test_raises_runtime_error_when_no_model_loads(tmp_path):
    pytest.importorskip("onnxruntime")
    cfg = ModelConfig(
        onnx_path=str(tmp_path / "missing_int8.onnx"),
        fallback_onnx_path=str(tmp_path / "missing_fp32.onnx"),
        prefer_int8=True,
        num_threads=2,
    )
    with pytest.raises(RuntimeError, match="Could not load any ONNX model"):
        OnnxClassifier(cfg)


def test_classify_output_is_num_classes_vector(int8_classifier):
    pp = NumpyMelSTFT()
    mel = pp.forward(np.zeros(32000, dtype=np.float32))
    logits = int8_classifier.classify(mel)
    assert logits.ndim == 1
    # efficientat_mn10_v6 is a single-logit binary sigmoid head (num_classes=1).
    assert logits.shape == (1,)
    assert logits.dtype == np.float32
