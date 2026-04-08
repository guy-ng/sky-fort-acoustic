"""Wave 0 RED stub — ONNX conversion sanity check vs PyTorch reference.

Covers: D-06 (ONNX export), D-07 (int8 dynamic quant), D-08 (top-1 agreement budget).
Owner: Plan 21-03 (scripts/convert_efficientat_to_onnx.py).

Note: 21-VALIDATION.md suggested ≥97% int8 top-1 agreement, but Plan 03 pins ≥95%
per 21-RESEARCH.md finding 3 (Conv-layer dynamic-quant accuracy caveat). Stub name
reflects the executable threshold.
"""
from __future__ import annotations

import pytest


def test_fp32_onnx_top1_agreement_ge_99pct():
    pytest.fail(
        "not implemented — Plan 21-03 must export FP32 ONNX and assert top-1 agreement "
        ">= 99% vs PyTorch reference on a held-out audio batch"
    )


def test_int8_onnx_top1_agreement_ge_95pct():
    pytest.fail(
        "not implemented — Plan 21-03 must produce int8 dynamic-quant ONNX and assert "
        "top-1 agreement >= 95% vs FP32 (per research finding 3 caveat)"
    )


def test_mean_logit_delta_below_threshold():
    pytest.fail(
        "not implemented — Plan 21-03 must assert mean(|onnx_logit - torch_logit|) < 0.05"
    )
