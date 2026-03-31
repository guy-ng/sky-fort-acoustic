---
phase: 03
slug: cnn-classification-and-target-tracking
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-31
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-asyncio |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/unit/ -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/ -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | CLS-01 | unit | `pytest tests/unit/test_preprocessing.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | CLS-01, CLS-04 | unit | `pytest tests/unit/test_inference.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | CLS-03 | unit | `pytest tests/unit/test_state_machine.py -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | TRK-01 | unit | `pytest tests/unit/test_tracker.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | TRK-03, TRK-04 | integration | `pytest tests/integration/test_zmq_publisher.py -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 2 | TRK-05 | unit | `pytest tests/unit/test_target_schema.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_preprocessing.py` — mel-spectrogram pipeline tests (CLS-01)
- [ ] `tests/unit/test_inference.py` — ONNX model loading and inference (CLS-01, CLS-04)
- [ ] `tests/unit/test_state_machine.py` — hysteresis state machine transitions (CLS-03)
- [ ] `tests/unit/test_tracker.py` — UUID assignment, TTL, lost detection (TRK-01)
- [ ] `tests/unit/test_target_schema.py` — ZMQ message schema validation (TRK-05)
- [ ] `tests/integration/test_zmq_publisher.py` — ZMQ publish/subscribe (TRK-03, TRK-04)
- [ ] `tests/fixtures/dummy_model.onnx` — small ONNX model for testing without real model file
- [ ] Framework install: `pip install librosa>=0.11.0` — not currently installed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Target appears in web UI within seconds | CLS-01 | Requires browser + live audio | Open UI, play drone audio, verify target marker appears |
| Detection does not flicker | CLS-03 | Requires sustained real-time observation | Watch UI for 30s with drone audio, verify no rapid on/off |
| ZMQ subscribers receive events | TRK-03, TRK-04 | End-to-end with external subscriber | Run `zmq_sub.py` listener, verify new/update/lost events |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
