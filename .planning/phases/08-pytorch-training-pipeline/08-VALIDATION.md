---
phase: 8
slug: pytorch-training-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-01
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 + pytest-asyncio 1.3.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `.venv/bin/pytest tests/unit/ -x -q` |
| **Full suite command** | `.venv/bin/pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/pytest tests/unit/ -x -q`
- **After every plan wave:** Run `.venv/bin/pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | TRN-01 | unit | `.venv/bin/pytest tests/unit/test_training_dataset.py -x` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | TRN-04 | unit | `.venv/bin/pytest tests/unit/test_augmentation.py -x` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | TRN-02 | unit | `.venv/bin/pytest tests/unit/test_training_manager.py -x` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 1 | TRN-03 | unit | `.venv/bin/pytest tests/unit/test_training_checkpoint.py -x` | ❌ W0 | ⬜ pending |
| 08-03-01 | 03 | 2 | TRN-01 | integration | `.venv/bin/pytest tests/integration/test_training_smoke.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_training_dataset.py` — stubs for TRN-01 (dataset, segment extraction, class balancing)
- [ ] `tests/unit/test_augmentation.py` — stubs for TRN-04 (SpecAugment, waveform augmentation)
- [ ] `tests/unit/test_training_manager.py` — stubs for TRN-02 (thread lifecycle, cancellation, progress)
- [ ] `tests/unit/test_training_checkpoint.py` — stubs for TRN-03 (save/load .pt checkpoint)
- [ ] `tests/integration/test_training_smoke.py` — stubs for TRN-01 (end-to-end training on synthetic data)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training thread does not degrade live detection latency below 150ms | TRN-02 | Requires real-time audio pipeline running concurrently | Start live detection, trigger training, observe beamforming latency |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
