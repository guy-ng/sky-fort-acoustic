"""Host-side PyTorch -> ONNX conversion for EfficientAT mn10 v6 (Plan 21-03).

Decisions addressed: D-05, D-06, D-07, D-08.

This script is the single source of truth for how `models/efficientat_mn10_v6.pt`
becomes the two ONNX artifacts the Pi ships:

  - models/efficientat_mn10_v6_fp32.onnx  (opset 17, dynamic batch axis)
  - models/efficientat_mn10_v6_int8.onnx  (dynamic-quant from FP32)

Both artifacts pass through a sanity gate (D-08): if top-1 agreement against the
torch reference falls below the configured tolerance, the offending .onnx file is
deleted and the script exits non-zero. A broken export must NEVER silently land
in models/.

Pi-side preprocessing: the ONNX input is the *mel tensor*, NOT raw audio. The
vendored numpy mel preprocess (apps/rpi-edge/skyfort_edge/preprocess.py, Plan 21-02)
runs on the Pi side and feeds (1, 1, 128, 100) into the ONNX session. Time axis
is FIXED at 100 frames (~1s at 32 kHz / hop=320). Only the batch axis is dynamic.

IMPORTANT: onnxruntime quantize_dynamic only quantizes MatMul and Gemm ops.
EfficientAT MN10 is Conv-heavy (MobileNetV3), so int8 speedup on the Pi will
be MARGINAL (5-15%), not the 3-4x you might see for a transformer. The int8
artifact exists primarily for size reduction and as a hedge for future static
calibration (deferred per D-28 discretion). FP32 remains the default fallback
(D-07).

Usage:
    python scripts/convert_efficientat_to_onnx.py \
        --checkpoint models/efficientat_mn10_v6.pt \
        --output-dir models/

    # FP32 only (skip int8 quantization)
    python scripts/convert_efficientat_to_onnx.py --skip-int8

    # Re-validate already-written artifacts (no re-export)
    python scripts/convert_efficientat_to_onnx.py --validate
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Construction constants must mirror src/acoustic/classification/efficientat/__init__.py
NUM_CLASSES = 1
WIDTH_MULT = 1.0
HEAD_TYPE = "mlp"
INPUT_DIM_F = 128
INPUT_DIM_T = 100  # ~1s at 32 kHz / hop=320

# Sanity-gate calibration batch
SANITY_BATCH_SIZE = 20
SANITY_SEED = 42


class LogitsOnly(nn.Module):
    """Adapter that strips the (logits, features) tuple returned by MN.forward.

    The MN model returns (logits, pooled_features). For ONNX export we want a
    single output tensor — hence this thin wrapper.

    NOTE: MN._forward_impl calls .squeeze() on the classifier output which
    collapses *all* singleton dims. For a num_classes=1 binary head this
    yields shape () for batch=1 or (batch,) for batch>1 — inconsistent and
    breaks the declared dynamic_axes={0: "batch"}, {0: "batch"} contract for
    ONNX export. We forcibly reshape the logits back to (batch, num_classes)
    so the exported graph has a consistent rank-2 output across all batch
    sizes — which is what the Pi inference code (Plan 21-05) and the sanity
    test expect.
    """

    def __init__(self, mn: nn.Module, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.mn = mn
        self.num_classes = num_classes

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.mn(mel)
        if isinstance(out, tuple):
            out = out[0]
        # Restore (batch, num_classes) shape regardless of upstream squeezes.
        return out.reshape(mel.shape[0], self.num_classes)


def build_torch_model(checkpoint_path: Path) -> LogitsOnly:
    """Load v6 checkpoint and wrap in LogitsOnly adapter."""
    from acoustic.classification.efficientat.model import get_model

    model = get_model(
        num_classes=NUM_CLASSES,
        width_mult=WIDTH_MULT,
        head_type=HEAD_TYPE,
        input_dim_f=INPUT_DIM_F,
        input_dim_t=INPUT_DIM_T,
    )
    state_dict = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True
    )
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    wrapped = LogitsOnly(model)
    wrapped.eval()
    return wrapped


def make_reference_batch(batch_size: int = SANITY_BATCH_SIZE) -> torch.Tensor:
    """Deterministic mel-shaped batch for the sanity gate.

    Shape: (batch_size, 1, 128, 100). Scaled by 0.5 to keep values in a
    reasonable mel-spectrogram dynamic range.
    """
    g = torch.Generator()
    g.manual_seed(SANITY_SEED)
    return torch.randn(
        batch_size, 1, INPUT_DIM_F, INPUT_DIM_T, generator=g, dtype=torch.float32
    ) * 0.5


def export_fp32(model: LogitsOnly, output_path: Path) -> None:
    """Export the LogitsOnly-wrapped model to FP32 ONNX with dynamic batch axis."""
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, INPUT_DIM_F, INPUT_DIM_T)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(str(output_path))


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Run onnxruntime dynamic quantization (MatMul + Gemm only)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def onnx_logits(onnx_path: Path, batch_np: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: batch_np})[0]


def torch_logits(model: LogitsOnly, batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(batch).cpu().numpy()


def _top1(logits: np.ndarray) -> np.ndarray:
    """Compute top-1 index. For binary sigmoid head (num_classes=1), the top-1
    index is determined by sign of the logit (>0 -> drone, <=0 -> not-drone).
    For multi-class heads we fall back to argmax.
    """
    if logits.ndim == 1 or logits.shape[-1] == 1:
        flat = logits.reshape(logits.shape[0], -1)[:, 0]
        return (flat > 0).astype(np.int64)
    return np.argmax(logits, axis=-1)


def sanity_gate(
    onnx_path: Path,
    torch_model: LogitsOnly,
    ref_batch: torch.Tensor,
    *,
    label: str,
    tolerance: float,
    logit_delta_max: float,
) -> tuple[float, float]:
    """Run torch vs onnx comparison. Delete onnx_path and exit on failure.

    Returns:
        (top1_agreement, mean_logit_delta) on success.
    """
    ref_np = ref_batch.numpy().astype(np.float32)
    torch_out = torch_logits(torch_model, ref_batch)
    onnx_out = onnx_logits(onnx_path, ref_np)

    torch_top1 = _top1(torch_out)
    onnx_top1 = _top1(onnx_out)
    agreement = float(np.mean(torch_top1 == onnx_top1))
    delta = float(np.mean(np.abs(onnx_out - torch_out)))

    failures: list[str] = []
    if agreement < tolerance:
        failures.append(
            f"top-1 agreement {agreement:.4f} < tolerance {tolerance:.4f}"
        )
    if delta >= logit_delta_max:
        failures.append(
            f"mean |logit delta| {delta:.6f} >= max {logit_delta_max:.6f}"
        )

    if failures:
        if onnx_path.exists():
            onnx_path.unlink()
        msg = (
            f"[FAIL] {label} sanity gate failed for {onnx_path.name}: "
            + "; ".join(failures)
            + ". Artifact deleted."
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    print(
        f"[OK]   {label} sanity gate passed for {onnx_path.name}: "
        f"top1={agreement:.4f}, mean_delta={delta:.6f}"
    )
    return agreement, delta


def write_sha256_manifest(paths: list[Path], manifest_path: Path) -> dict[str, str]:
    """Write `<hex>  <filename>` lines, sha256sum-compatible."""
    digests: dict[str, str] = {}
    lines: list[str] = []
    for p in paths:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        digests[p.name] = h
        lines.append(f"{h}  {p.name}\n")
    manifest_path.write_text("".join(lines))
    return digests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EfficientAT mn10 v6 PyTorch checkpoint to ONNX (FP32 + int8)."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/efficientat_mn10_v6.pt"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("models"))
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument(
        "--validate",
        action="store_true",
        help="Re-run sanity gate against existing on-disk ONNX artifacts (no re-export).",
    )
    p.add_argument("--tolerance-fp32", type=float, default=0.99)
    p.add_argument("--tolerance-int8", type=float, default=0.95)
    p.add_argument("--logit-delta-max", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint: Path = args.checkpoint
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / "efficientat_mn10_v6_fp32.onnx"
    int8_path = output_dir / "efficientat_mn10_v6_int8.onnx"
    manifest_path = output_dir / "efficientat_mn10_v6_onnx.sha256"

    if not checkpoint.exists():
        print(f"[FAIL] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    print(f"[INFO] loading checkpoint: {checkpoint}")
    torch_model = build_torch_model(checkpoint)

    ref_batch = make_reference_batch()
    print(f"[INFO] sanity batch shape: {tuple(ref_batch.shape)}, seed={SANITY_SEED}")

    if not args.validate:
        print(f"[INFO] exporting FP32 ONNX -> {fp32_path}")
        export_fp32(torch_model, fp32_path)

    fp32_top1, fp32_delta = sanity_gate(
        fp32_path,
        torch_model,
        ref_batch,
        label="FP32",
        tolerance=args.tolerance_fp32,
        logit_delta_max=args.logit_delta_max,
    )

    int8_top1: float | None = None
    int8_delta: float | None = None
    artifacts = [fp32_path]

    if not args.skip_int8:
        if not args.validate:
            print(f"[INFO] dynamic-quantizing -> {int8_path}")
            quantize_int8(fp32_path, int8_path)
        # int8 keeps FP32 logit numerics close enough that the same logit-delta
        # budget applies to a binary head; relax it slightly via the int8 path
        # if your run shows headroom-bound failures.
        int8_top1, int8_delta = sanity_gate(
            int8_path,
            torch_model,
            ref_batch,
            label="INT8",
            tolerance=args.tolerance_int8,
            logit_delta_max=args.logit_delta_max,
        )
        artifacts.append(int8_path)

    digests = write_sha256_manifest(artifacts, manifest_path)
    print(f"[INFO] sha256 manifest: {manifest_path}")

    # Summary table
    print("\n=== Conversion Summary ===")
    header = f"{'artifact':<40} {'bytes':>12} {'top1':>8} {'mean_delta':>12}  sha256"
    print(header)
    print("-" * len(header))
    print(
        f"{fp32_path.name:<40} {fp32_path.stat().st_size:>12} "
        f"{fp32_top1:>8.4f} {fp32_delta:>12.6f}  {digests[fp32_path.name][:16]}..."
    )
    if not args.skip_int8:
        assert int8_top1 is not None and int8_delta is not None
        print(
            f"{int8_path.name:<40} {int8_path.stat().st_size:>12} "
            f"{int8_top1:>8.4f} {int8_delta:>12.6f}  {digests[int8_path.name][:16]}..."
        )

    print("\n[DONE] All artifacts written and sanity-validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
