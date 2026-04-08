"""D-06, D-07, D-08: ONNX conversion sanity gate (re-verification from committed artifacts).

This test re-runs the same sanity check that scripts/convert_efficientat_to_onnx.py
runs at conversion time. It exists for two reasons:

1. T-21-05 tamper detection: catches any post-commit corruption or hand-editing
   of the .onnx artifacts vs the committed sha256 manifest.
2. Drift guard: if a future change to the conversion script (or src.acoustic.
   classification.efficientat construction) silently regresses agreement, this
   test fires loudly in CI before the broken artifact reaches the Pi.

The torch construction here MUST mirror scripts/convert_efficientat_to_onnx.py
exactly (same num_classes, width_mult, head_type, input dims). If the script
changes, update both sides.

Plan: 21-03. Owner construction reference: src/acoustic/classification/efficientat/__init__.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

# apps/rpi-edge/tests/test_*.py -> parents[0]=tests, [1]=rpi-edge, [2]=apps, [3]=repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
FP32_ONNX = REPO_ROOT / "models" / "efficientat_mn10_v6_fp32.onnx"
INT8_ONNX = REPO_ROOT / "models" / "efficientat_mn10_v6_int8.onnx"
CHECKSUMS = REPO_ROOT / "models" / "efficientat_mn10_v6_onnx.sha256"
CHECKPOINT = REPO_ROOT / "models" / "efficientat_mn10_v6.pt"

N_SAMPLES = 50
RNG_SEED = 1337

# Construction constants — keep in sync with scripts/convert_efficientat_to_onnx.py
NUM_CLASSES = 1
WIDTH_MULT = 1.0
HEAD_TYPE = "mlp"
INPUT_DIM_F = 128
INPUT_DIM_T = 100


def _make_ref_batch() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return rng.standard_normal((N_SAMPLES, 1, INPUT_DIM_F, INPUT_DIM_T)).astype(np.float32) * 0.5


def _torch_logits(batch_np: np.ndarray) -> np.ndarray:
    """Build the torch reference model and run forward pass on batch_np."""
    torch = pytest.importorskip("torch")
    if not CHECKPOINT.exists():
        pytest.skip(
            f"checkpoint missing: {CHECKPOINT} — required for re-validation; "
            f"obtain from training pipeline (Phase 20) before running this test"
        )
    # Make src/ importable so we can construct the same MN module
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from acoustic.classification.efficientat.model import get_model

    model = get_model(
        num_classes=NUM_CLASSES,
        width_mult=WIDTH_MULT,
        head_type=HEAD_TYPE,
        input_dim_f=INPUT_DIM_F,
        input_dim_t=INPUT_DIM_T,
    )
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(batch_np))
        if isinstance(out, tuple):
            out = out[0]
    arr = out.cpu().numpy()
    # Normalize to (N, num_classes) — MN may squeeze singleton dims
    return arr.reshape(batch_np.shape[0], NUM_CLASSES)


def _onnx_logits(onnx_path: Path, batch_np: np.ndarray) -> np.ndarray:
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: batch_np})[0]
    return np.asarray(out).reshape(batch_np.shape[0], NUM_CLASSES)


def _top1(logits: np.ndarray) -> np.ndarray:
    """For binary sigmoid head, top-1 == sign of logit. For multi-class, argmax."""
    if logits.shape[-1] == 1:
        return (logits[:, 0] > 0).astype(np.int64)
    return np.argmax(logits, axis=-1)


@pytest.fixture(scope="module")
def ref_batch() -> np.ndarray:
    return _make_ref_batch()


@pytest.fixture(scope="module")
def torch_reference(ref_batch: np.ndarray) -> np.ndarray:
    return _torch_logits(ref_batch)


def test_fp32_onnx_top1_agreement_ge_99pct(
    ref_batch: np.ndarray, torch_reference: np.ndarray
) -> None:
    assert FP32_ONNX.exists(), (
        f"missing {FP32_ONNX} — run scripts/convert_efficientat_to_onnx.py first"
    )
    onnx_out = _onnx_logits(FP32_ONNX, ref_batch)
    torch_top1 = _top1(torch_reference)
    onnx_top1 = _top1(onnx_out)
    agreement = float(np.mean(torch_top1 == onnx_top1))
    assert agreement >= 0.99, f"FP32 top-1 agreement {agreement:.4f} < 0.99"


def test_int8_onnx_top1_agreement_ge_95pct(
    ref_batch: np.ndarray, torch_reference: np.ndarray
) -> None:
    assert INT8_ONNX.exists(), (
        f"missing {INT8_ONNX} — run scripts/convert_efficientat_to_onnx.py first"
    )
    onnx_out = _onnx_logits(INT8_ONNX, ref_batch)
    torch_top1 = _top1(torch_reference)
    onnx_top1 = _top1(onnx_out)
    agreement = float(np.mean(torch_top1 == onnx_top1))
    assert agreement >= 0.95, (
        f"int8 top-1 agreement {agreement:.4f} < 0.95 "
        f"(research note: Conv layers are NOT quantized by quantize_dynamic)"
    )


def test_mean_logit_delta_below_threshold(
    ref_batch: np.ndarray, torch_reference: np.ndarray
) -> None:
    assert FP32_ONNX.exists(), f"missing {FP32_ONNX}"
    onnx_out = _onnx_logits(FP32_ONNX, ref_batch)
    delta = float(np.mean(np.abs(onnx_out - torch_reference)))
    assert delta < 0.05, f"mean logit delta {delta:.6f} >= 0.05"


def test_checksum_file_valid() -> None:
    assert CHECKSUMS.exists(), f"missing checksum manifest: {CHECKSUMS}"
    expected: dict[str, str] = {}
    for line in CHECKSUMS.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            expected[parts[1].strip()] = parts[0].strip()
    for onnx_file in (FP32_ONNX, INT8_ONNX):
        assert onnx_file.exists(), f"missing {onnx_file}"
        actual = hashlib.sha256(onnx_file.read_bytes()).hexdigest()
        name = onnx_file.name
        assert name in expected, f"no checksum recorded for {name}"
        assert actual == expected[name], (
            f"sha256 mismatch for {name}: expected {expected[name]}, got {actual}"
        )
