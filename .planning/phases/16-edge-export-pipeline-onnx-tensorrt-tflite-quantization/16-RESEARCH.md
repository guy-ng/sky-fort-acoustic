# Phase 16: Edge Export Pipeline - ONNX TensorRT TFLite Quantization - Research

**Researched:** 2026-04-04
**Domain:** Model export, quantization, edge deployment
**Confidence:** HIGH

## Summary

This phase adds a model export pipeline that converts trained PyTorch models (both ResearchCNN and EfficientAT MN10) to ONNX format, with optional downstream conversion to TensorRT FP16/INT8 engines and TFLite INT8 quantized models. The critical architectural decision is that mel-spectrogram preprocessing (AugmentMelSTFT) must be **excluded** from the ONNX graph because `torch.stft` is not exportable to ONNX. This matches EfficientAT's own deployment strategy: export the CNN backbone only, with mel preprocessing handled separately on the host.

The project already has ONNX (1.21.0) and ONNX Runtime (1.24.4) installed. TensorRT and TFLite converters are NOT available on the dev machine (macOS) -- this is expected since TensorRT requires NVIDIA GPUs (Jetson target) and TFLite conversion requires `onnx2tf`. The export pipeline should generate ONNX on any platform, and TensorRT/TFLite conversion should be available as optional steps that gracefully skip when dependencies are missing.

**Primary recommendation:** Build an `ExportPipeline` class in `src/acoustic/export/` that takes a PyTorch model checkpoint, loads it, creates an export-ready wrapper (removing dual-output for EfficientAT), runs `torch.onnx.export` with opset 17, validates numerical parity, and optionally converts to TensorRT/TFLite. Expose via a REST API endpoint `POST /api/export`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEP-01 | Trained PyTorch model exports to ONNX (opset 13+) with verified numerical parity (atol=1e-4) | torch.onnx.export with opset 17 for both ResearchCNN and EfficientAT MN10; parity check via onnxruntime inference comparison |
| DEP-02 | ONNX model converts to TensorRT FP16 engine with <30ms inference on Jetson; converts to TFLite INT8 with PTQ calibration | TensorRT via trtexec/tensorrt Python API (Jetson only); TFLite via onnx2tf with -oiqt flag; both optional with graceful skip |
| DEP-03 | REST API endpoint allows model export with format selection (onnx, tensorrt, tflite) | POST /api/export endpoint with format parameter, background task for conversion, status polling |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnx | 1.21.0 | ONNX model format | Already installed; standard for model interchange |
| onnxruntime | 1.24.4 | ONNX inference + validation | Already installed; numerical parity verification |
| torch.onnx | (PyTorch 2.11) | Export PyTorch to ONNX | Built into PyTorch; supports opset 17 |

### Supporting (Optional Edge Dependencies)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnx2tf | 2.4.0 | ONNX to TFLite conversion | TFLite INT8 export (Raspberry Pi target) |
| tensorrt | 10.x | TensorRT engine building | Jetson deployment only (NVIDIA GPU required) |
| onnxsim | >=0.4 | ONNX graph simplification | Pre-processing before TensorRT/TFLite conversion |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| onnx2tf | ai-edge-litert (Google) | Newer but less mature for ONNX input; onnx2tf has proven MobileNetV3 support |
| torch.onnx.export | torch.onnx.dynamo_export | Dynamo exporter is newer but less proven for audio models with custom ops |
| tensorrt Python API | trtexec CLI | CLI is simpler but Python API gives better control over calibration |

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/export/
    __init__.py
    exporter.py          # ExportPipeline class (core ONNX export logic)
    onnx_utils.py         # Validation, simplification helpers
    tensorrt_convert.py   # TensorRT conversion (optional, graceful skip)
    tflite_convert.py     # TFLite conversion via onnx2tf (optional, graceful skip)
    wrappers.py           # Export-ready model wrappers (single output, no mel)
src/acoustic/api/
    export_routes.py      # POST /api/export endpoint
```

### Pattern 1: Separate Mel Preprocessing from CNN Backbone

**What:** The ONNX model accepts mel spectrogram input, NOT raw audio. Mel preprocessing stays in Python/NumPy at inference time on the edge device.

**When to use:** Always -- `torch.stft` cannot be exported to ONNX.

**Why:** PyTorch's `torch.stft` operator is not supported for ONNX export (open issue since 2021, still unresolved as of PyTorch 2.11). The ONNX opset 17 STFT operator exists but PyTorch's exporter does not map `aten::stft` to it. EfficientAT's own deployment approach splits the inference into two stages: mel preprocessing on host, CNN inference on accelerator.

**Example:**
```python
# Export wrapper for EfficientAT -- single output, accepts mel input
class EfficientATExportWrapper(nn.Module):
    """Wraps MN model for ONNX export: single logit output, mel input."""

    def __init__(self, model: MN) -> None:
        super().__init__()
        self._model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Input: (batch, 1, n_mels, time). Output: (batch, 1) logits."""
        logits, _ = self._model(mel)  # Discard features output
        return logits
```

### Pattern 2: ResearchCNN Export Wrapper

**What:** ResearchCNN already has a simple single-output forward(), but needs `logits_mode=False` for export (include sigmoid) to match the inference contract.

**Example:**
```python
# ResearchCNN exports directly -- simpler case
model = ResearchCNN(logits_mode=False)
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()
dummy = torch.randn(1, 1, 128, 64)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17,
                   input_names=["mel_spectrogram"],
                   output_names=["probability"],
                   dynamic_axes={"mel_spectrogram": {0: "batch"}, "probability": {0: "batch"}})
```

### Pattern 3: Numerical Parity Verification

**What:** After ONNX export, run the same input through both PyTorch and ONNX Runtime, assert allclose.

**Example:**
```python
import onnxruntime as ort
import numpy as np

# PyTorch reference
with torch.no_grad():
    pt_output = model(dummy_input).numpy()

# ONNX Runtime
session = ort.InferenceSession("model.onnx")
ort_output = session.run(None, {"mel_spectrogram": dummy_input.numpy()})[0]

np.testing.assert_allclose(pt_output, ort_output, atol=1e-4)
```

### Pattern 4: Optional Converter with Graceful Skip

**What:** TensorRT and TFLite converters import-guard their dependencies and return a status indicating whether conversion was possible.

**Example:**
```python
def convert_to_tflite(onnx_path: str, output_path: str, calibration_data: np.ndarray | None = None) -> dict:
    try:
        import onnx2tf
    except ImportError:
        return {"status": "skipped", "reason": "onnx2tf not installed"}

    # ... conversion logic ...
    return {"status": "success", "path": output_path}
```

### Anti-Patterns to Avoid
- **Including AugmentMelSTFT in ONNX graph:** torch.stft will fail export. Always export backbone only.
- **Hardcoding batch size:** Use dynamic_axes for batch dimension so edge devices can use batch=1.
- **Exporting in training mode:** Always call model.eval() and set model to logits_mode=False before export.
- **Blocking API on export:** TensorRT/TFLite conversion can take minutes. Use background task.
- **Assuming GPU on dev machine:** macOS has no NVIDIA GPU. TensorRT tests must be skippable.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX export | Custom graph builder | torch.onnx.export | PyTorch's exporter handles op mapping, shape inference |
| ONNX validation | Manual tensor comparison | onnxruntime + np.testing.assert_allclose | ORT is the reference runtime |
| ONNX to TFLite | Custom TF graph construction | onnx2tf | Handles NCHW->NHWC transpose, op mapping, quantization |
| ONNX simplification | Manual graph passes | onnxsim | Folds constants, removes redundant nodes |
| Calibration dataset | Random noise | Real audio clips from training data | Quantization accuracy depends on representative data |

## Common Pitfalls

### Pitfall 1: EfficientAT Dual Output
**What goes wrong:** MN.forward() returns `(logits, features)` tuple. ONNX export fails or produces unexpected graph with two outputs.
**Why it happens:** The model was designed for embedding extraction + classification simultaneously.
**How to avoid:** Wrap in EfficientATExportWrapper that discards the features output.
**Warning signs:** ONNX model has 2 outputs instead of 1; downstream converters confused by extra output.

### Pitfall 2: torch.stft Not Exportable
**What goes wrong:** Including AugmentMelSTFT in the export graph causes `RuntimeError: Exporting the operator 'aten::stft' to ONNX opset version X is not supported`.
**Why it happens:** PyTorch's ONNX exporter has never implemented the stft->STFT opset 17 mapping.
**How to avoid:** Export backbone only. Mel preprocessing runs in Python/NumPy on the edge device.
**Warning signs:** Export error mentioning aten::stft or prims.fft_r2c.

### Pitfall 3: SiLU/HardSwish Quantization Issues
**What goes wrong:** MobileNetV3 uses HardSwish and Squeeze-Excitation which can produce catastrophic INT8 quantization errors.
**Why it happens:** Non-linear activation functions with narrow value ranges lose precision at INT8.
**How to avoid:** Use calibration data from real audio (not random); verify quantized accuracy; FP16 is safer than INT8 for TensorRT.
**Warning signs:** Quantized model accuracy drops >5% vs full precision.

### Pitfall 4: Missing torchvision Dependency in Export
**What goes wrong:** EfficientAT model.py imports `from torchvision.ops.misc import ConvNormActivation` at init time.
**Why it happens:** ConvNormActivation is used in the first conv layer construction.
**How to avoid:** Ensure torchvision is available in the export environment.
**Warning signs:** ImportError when loading model for export.

### Pitfall 5: Batch Dimension Squeeze in Single-Sample Inference
**What goes wrong:** MN._forward_impl squeezes batch dim when batch=1, causing shape mismatch.
**Why it happens:** Lines 164-170 of model.py: `features.squeeze()` removes batch dim for single samples, then re-adds it.
**How to avoid:** The export wrapper handles this, but test with batch_size=1 specifically.
**Warning signs:** ONNX output shape is (1,) instead of (1, 1).

### Pitfall 6: TFLite NCHW to NHWC Transposition
**What goes wrong:** PyTorch models use NCHW layout; TFLite uses NHWC. Naive conversion inserts many Transpose ops.
**Why it happens:** Framework layout mismatch.
**How to avoid:** onnx2tf handles this automatically (its primary purpose). Do not use onnx-tf.
**Warning signs:** TFLite model is much slower than expected due to excessive Transpose ops.

## Code Examples

### Complete ONNX Export Flow
```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
) -> dict:
    """Export PyTorch model to ONNX with validation."""
    model.eval()

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        do_constant_folding=True,
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Numerical parity check
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    sess = ort.InferenceSession(output_path)
    ort_out = sess.run(None, {input_names[0]: dummy_input.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    parity_ok = max_diff < 1e-4

    return {
        "status": "success" if parity_ok else "parity_failed",
        "path": output_path,
        "max_diff": max_diff,
        "opset": 17,
    }
```

### EfficientAT-Specific Export (with mel separation)
```python
from acoustic.classification.efficientat.model import get_model, MN
from acoustic.classification.efficientat.config import EfficientATMelConfig

def export_efficientat(ckpt_path: str, output_path: str) -> dict:
    mel_cfg = EfficientATMelConfig()

    # Load model with binary classification head
    model = get_model(
        num_classes=527, width_mult=1.0, head_type="mlp",
        input_dim_f=mel_cfg.n_mels, input_dim_t=mel_cfg.input_dim_t,
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 1)

    state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)

    # Wrap for single-output export
    wrapper = EfficientATExportWrapper(model)
    wrapper.eval()

    # Dummy: (batch=1, channels=1, n_mels=128, time=100)
    dummy = torch.randn(1, 1, mel_cfg.n_mels, mel_cfg.input_dim_t)

    return export_to_onnx(
        wrapper, dummy, output_path,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}},
    )
```

### TFLite Conversion with INT8 Quantization
```python
def convert_onnx_to_tflite(
    onnx_path: str,
    output_dir: str,
    calibration_data: np.ndarray | None = None,
) -> dict:
    try:
        import onnx2tf
    except ImportError:
        return {"status": "skipped", "reason": "onnx2tf not installed"}

    args = ["-i", onnx_path, "-o", output_dir]
    if calibration_data is not None:
        # Save calibration data for onnx2tf
        np.save(os.path.join(output_dir, "calibration.npy"), calibration_data)
        args.extend(["-oiqt"])  # Output INT8 quantized TFLite

    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=output_dir,
                    copy_onnx_input_output_names_to_tflite=True,
                    non_verbose=True)

    return {"status": "success", "path": output_dir}
```

### REST API Endpoint
```python
from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/api", tags=["export"])

@router.post("/export")
async def export_model(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
) -> ExportResponse:
    """Start model export to specified format (onnx, tensorrt, tflite)."""
    # Validate model exists
    # Queue background export task
    # Return job ID for status polling
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.onnx.export (TorchScript) | torch.onnx.export with dynamo=True | PyTorch 2.1+ | New exporter available but legacy still works; use legacy for stability |
| tensorflow-lite converter | onnx2tf with flatbuffer_direct backend | onnx2tf 2.4.0 | Faster, higher success rate; no TF dependency needed |
| Manual NCHW->NHWC | onnx2tf auto-handles | Always | onnx2tf primary value proposition |
| TorchScript for deployment | ONNX for cross-platform | Industry trend | ONNX is the standard interchange format |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| onnx | ONNX export | Yes | 1.21.0 | -- |
| onnxruntime | Parity validation | Yes | 1.24.4 | -- |
| torch | Model loading | Yes | 2.11.0 | -- |
| torchvision | EfficientAT ConvNormActivation | Needs check | -- | Must install |
| onnx2tf | TFLite conversion | No | -- | Skip TFLite on dev; install for Raspberry Pi target |
| tensorrt | TensorRT engine | No | -- | Skip on macOS; available on Jetson |
| onnxsim | Graph simplification | No | -- | Optional; `pip install onnxsim` |

**Missing dependencies with no fallback:**
- None (ONNX export works with current stack)

**Missing dependencies with fallback:**
- `onnx2tf`: TFLite conversion skipped on dev machine; install on target or CI
- `tensorrt`: TensorRT conversion skipped on macOS; runs on Jetson with NVIDIA GPU
- `onnxsim`: Nice-to-have graph optimization; export works without it

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/test_export.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEP-01a | ResearchCNN exports to ONNX opset 17 | unit | `pytest tests/unit/test_export.py::test_research_cnn_onnx_export -x` | Wave 0 |
| DEP-01b | EfficientAT exports to ONNX opset 17 | unit | `pytest tests/unit/test_export.py::test_efficientat_onnx_export -x` | Wave 0 |
| DEP-01c | ONNX numerical parity atol=1e-4 | unit | `pytest tests/unit/test_export.py::test_onnx_parity -x` | Wave 0 |
| DEP-02a | TFLite conversion (skip if no onnx2tf) | unit | `pytest tests/unit/test_export.py::test_tflite_conversion -x` | Wave 0 |
| DEP-02b | TensorRT conversion (skip if no tensorrt) | unit | `pytest tests/unit/test_export.py::test_tensorrt_conversion -x` | Wave 0 |
| DEP-03 | REST API export endpoint | integration | `pytest tests/integration/test_export_api.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_export.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_export.py` -- covers DEP-01a, DEP-01b, DEP-01c, DEP-02a, DEP-02b
- [ ] `tests/integration/test_export_api.py` -- covers DEP-03
- [ ] `pip install onnxsim` -- optional but recommended for graph optimization

## Open Questions

1. **torchvision availability in venv**
   - What we know: EfficientAT model.py imports `torchvision.ops.misc.ConvNormActivation` at model construction time
   - What's unclear: Whether torchvision is installed in the current venv (not in requirements.txt)
   - Recommendation: Verify and add to requirements if missing; it is required for EfficientAT export

2. **EfficientAT input dimensions for export**
   - What we know: Default config uses input_dim_t=100 (100 time frames = ~1s at 32kHz/hop=320)
   - What's unclear: Whether edge devices will use different segment lengths
   - Recommendation: Use dynamic_axes for time dimension to support variable-length input

3. **Calibration dataset size for INT8 quantization**
   - What we know: PTQ needs representative data (typically 100-1000 samples)
   - What's unclear: How many calibration samples are available from training data
   - Recommendation: Use 200 random mel spectrograms from training/validation set

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/acoustic/classification/efficientat/model.py` -- MN forward returns (logits, features) tuple
- Codebase analysis: `src/acoustic/classification/efficientat/preprocess.py` -- AugmentMelSTFT uses torch.stft
- Installed packages: onnx 1.21.0, onnxruntime 1.24.4, torch 2.11.0 verified via pip
- [ONNX STFT operator spec](https://onnx.ai/onnx/operators/onnx__STFT.html) -- exists in opset 17

### Secondary (MEDIUM confidence)
- [PyTorch ONNX export docs](https://docs.pytorch.org/docs/stable/onnx.html) -- opset 17 supported
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf) -- v2.4.0 with flatbuffer_direct backend
- [PyTorch issue #65666](https://github.com/pytorch/pytorch/issues/65666) -- torch.stft ONNX export still open
- [Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices](https://arxiv.org/html/2509.14049) -- EfficientAT edge deployment approach

### Tertiary (LOW confidence)
- [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) -- alternative to standalone TensorRT
- onnx2tf SiLU quantization warning -- from GitHub issue, needs validation with HardSwish specifically

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - onnx/onnxruntime already installed and verified; torch.onnx.export is well-established
- Architecture: HIGH - mel/backbone separation is the proven EfficientAT deployment pattern; torch.stft limitation is well-documented
- Pitfalls: HIGH - dual-output wrapper need confirmed by code inspection; stft limitation confirmed by multiple sources
- TFLite/TensorRT specifics: MEDIUM - onnx2tf INT8 flow documented but not tested on this model; TensorRT untestable on macOS

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (30 days -- stable domain, libraries are mature)
