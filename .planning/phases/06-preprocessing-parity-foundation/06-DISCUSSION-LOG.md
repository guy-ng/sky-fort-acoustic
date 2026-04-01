# Phase 6: Preprocessing Parity Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 06-preprocessing-parity-foundation
**Areas discussed:** Coexistence strategy, Protocol design scope, Parity test approach, librosa vs torchaudio

---

## Coexistence Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Config-switched | Single config flag selects which preprocessor+classifier pair loads at startup | |
| Side-by-side shadow mode | Both preprocessors run simultaneously, old drives detection, new logs for comparison | |
| Remove old entirely | Old pipeline is non-functional, replace entirely | ✓ |

**User's choice:** Free text — "we can remove old one - it is not working"
**Notes:** The existing EfficientNet-B0 preprocessing pipeline is confirmed non-functional. No coexistence needed.

### Follow-up: ONNX Model Status

| Option | Description | Selected |
|--------|-------------|----------|
| Remove ONNX entirely | Delete OnnxDroneClassifier and .onnx model. No classifier until Phase 7. | ✓ |
| Keep ONNX wrapped | Wrap behind Classifier protocol so pipeline has a working classifier during Phase 6 | |
| You decide | Let Claude choose | |

**User's choice:** Remove ONNX entirely
**Notes:** ONNX model is also dead. Pipeline will have no classifier until Phase 7.

---

## Protocol Design Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal | Classifier: predict(tensor) -> float. Preprocessor: process(audio, sr) -> tensor. | ✓ |
| With metadata | Adds name/version properties and config() method for logging and evaluation harness | |
| You decide | Let Claude determine | |

**User's choice:** Minimal (Recommended)
**Notes:** None

---

## Parity Test Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Saved reference tensors | Run TF code once, save .npy fixtures, PyTorch tests compare against them. No TF in CI. | ✓ |
| TF as test dependency | Install TensorFlow in test env, run both pipelines live and compare | |
| You decide | Let Claude choose | |

**User's choice:** Saved reference tensors (Recommended)
**Notes:** None

---

## librosa vs torchaudio

| Option | Description | Selected |
|--------|-------------|----------|
| torchaudio | Aligns with PyTorch stack, GPU-acceleratable, native tensors | ✓ |
| Keep librosa | Exact match with research code, zero parity risk, but NumPy-only | |
| You decide | Let Claude choose | |

**User's choice:** torchaudio (Recommended)
**Notes:** None

---

## Claude's Discretion

- MelConfig placement (which module)
- CNNWorker restructuring for protocol injection
- Parity test fixture WAV file selection
- Whether to keep fast_resample() or use torchaudio resampling
- SILENCE_RMS_THRESHOLD handling in new preprocessor

## Deferred Ideas

None — discussion stayed within phase scope
