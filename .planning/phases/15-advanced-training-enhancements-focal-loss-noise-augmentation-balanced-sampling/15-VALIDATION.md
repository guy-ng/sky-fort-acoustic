---
phase: 15
slug: advanced-training-enhancements-focal-loss-noise-augmentation-balanced-sampling
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 15 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/ -x -q --timeout=30` |
| **Full suite command** | `python -m pytest tests/ -v --timeout=60` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --timeout=30`
- **After every plan wave:** Run `python -m pytest tests/ -v --timeout=60`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | TRN-10 | unit | `python -m pytest tests/unit/test_focal_loss.py -v` | W0 | pending |
| 15-01-02 | 01 | 1 | TRN-11 | unit | `python -m pytest tests/unit/test_noise_augmentation.py -v` | W0 | pending |
| 15-01-03 | 01 | 1 | TRN-12 | unit | `python -m pytest tests/unit/test_audiomentations_aug.py -v` | W0 | pending |
| 15-02-01 | 02 | 2 | TRN-10,TRN-11,TRN-12 | integration | `python -m pytest tests/unit/test_training_enhancements_integration.py -v` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_focal_loss.py` -- stubs for TRN-10 (focal loss)
- [ ] `tests/unit/test_noise_augmentation.py` -- stubs for TRN-11 (background noise mixing)
- [ ] `tests/unit/test_audiomentations_aug.py` -- stubs for TRN-12 (audiomentations pipeline + ComposedAugmentation picklability)
- [ ] `tests/unit/test_training_enhancements_integration.py` -- stubs for TRN-10/11/12 integration + weighted sampler at DADS scale

*Existing pytest infrastructure covers framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| <5% FPR with >95% recall on DADS test | TRN-10,TRN-11,TRN-12 | Requires full DADS dataset and trained model | Run evaluation harness on DADS test split after training |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
