"""Wave 0 RED stub — int8 ONNX inference latency budget on host.

Covers: D-05 (onnxruntime CPU EP), D-07 (int8 latency budget — host proxy for Pi 4).
Owner: Plan 21-03 (ONNX export) + Plan 21-05 (inference wiring).
"""
from __future__ import annotations

import pytest


def test_int8_onnx_inference_under_150ms():
    pytest.fail(
        "not implemented — Plan 21-03/05: load int8 ONNX session, run inference on "
        "(1, 1, 128, T_FRAMES) mel input, assert single-shot latency < 150 ms on host "
        "(proxy for Pi 4 ~500 ms budget)"
    )
