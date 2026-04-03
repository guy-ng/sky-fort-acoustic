---
phase: 14
slug: efficientat-model-architecture-with-audioset-transfer-learning
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 with pytest-asyncio |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/unit/test_efficientat.py -x` |
| **Full suite command** | `python -m pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/test_efficientat.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | MDL-10 | unit | `python -m pytest tests/unit/test_efficientat.py::test_model_loads_pretrained -x` | ❌ W0 | ⬜ pending |
| 14-01-02 | 01 | 1 | MDL-10 | unit | `python -m pytest tests/unit/test_efficientat.py::test_classifier_protocol -x` | ❌ W0 | ⬜ pending |
| 14-01-03 | 01 | 1 | MDL-10 | unit | `python -m pytest tests/unit/test_efficientat.py::test_param_count -x` | ❌ W0 | ⬜ pending |
| 14-01-04 | 01 | 1 | MDL-11 | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage1_freeze -x` | ❌ W0 | ⬜ pending |
| 14-01-05 | 01 | 1 | MDL-11 | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage2_unfreeze -x` | ❌ W0 | ⬜ pending |
| 14-01-06 | 01 | 1 | MDL-11 | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage3_unfreeze -x` | ❌ W0 | ⬜ pending |
| 14-01-07 | 01 | 1 | MDL-11 | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_cosine_schedule -x` | ❌ W0 | ⬜ pending |
| 14-01-08 | 01 | 1 | MDL-12 | unit | `python -m pytest tests/unit/test_efficientat.py::test_registry_load -x` | ❌ W0 | ⬜ pending |
| 14-01-09 | 01 | 1 | MDL-12 | unit | `python -m pytest tests/unit/test_config.py::test_model_type_config -x` | ✅ Extend | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_efficientat.py` — model loading, protocol, param count, registry tests
- [ ] `tests/unit/test_efficientat_training.py` — stage freeze/unfreeze, cosine schedule tests

*Existing infrastructure covers pytest framework and conftest fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| >95% accuracy on DADS test set | MDL-11 | Requires full training run with DADS dataset | Train model with DADS data, run evaluation harness, verify accuracy >95% |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending