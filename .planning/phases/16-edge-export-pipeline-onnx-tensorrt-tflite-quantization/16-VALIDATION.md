---
phase: 16
slug: edge-export-pipeline-onnx-tensorrt-tflite-quantization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_export.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_export.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | DEP-01 | unit | `pytest tests/test_export.py::test_onnx_export` | ❌ W0 | ⬜ pending |
| 16-01-02 | 01 | 1 | DEP-01 | unit | `pytest tests/test_export.py::test_numerical_parity` | ❌ W0 | ⬜ pending |
| 16-02-01 | 02 | 2 | DEP-02 | unit | `pytest tests/test_export.py::test_tensorrt_conversion` | ❌ W0 | ⬜ pending |
| 16-02-02 | 02 | 2 | DEP-03 | unit | `pytest tests/test_export.py::test_tflite_conversion` | ❌ W0 | ⬜ pending |
| 16-03-01 | 03 | 2 | DEP-01 | integration | `pytest tests/test_export.py::test_export_api_endpoint` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_export.py` — stubs for DEP-01, DEP-02, DEP-03
- [ ] `tests/conftest.py` — shared fixtures (mock model, calibration data)

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorRT <30ms latency on Jetson | DEP-02 | Requires Jetson hardware | Export model, deploy to Jetson, run benchmark script |
| TFLite INT8 on Raspberry Pi | DEP-03 | Requires RPi hardware | Export model, deploy to RPi, verify inference runs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
