---
phase: 11
slug: late-fusion-ensemble-conditional
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x with pytest-asyncio |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/unit/test_ensemble.py -x -q` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/test_ensemble.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | ENS-01 | unit | `pytest tests/unit/test_ensemble.py::test_weighted_soft_voting -x` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | ENS-01 | unit | `pytest tests/unit/test_ensemble.py::test_weight_normalization -x` | ❌ W0 | ⬜ pending |
| 11-01-03 | 01 | 1 | ENS-01 | unit | `pytest tests/unit/test_ensemble.py::test_model_registry -x` | ❌ W0 | ⬜ pending |
| 11-01-04 | 01 | 1 | ENS-01 | unit | `pytest tests/unit/test_ensemble.py::test_config_parsing -x` | ❌ W0 | ⬜ pending |
| 11-01-05 | 01 | 1 | ENS-02 | unit | `pytest tests/unit/test_ensemble.py::test_live_mode_cap -x` | ❌ W0 | ⬜ pending |
| 11-01-06 | 01 | 1 | ENS-02 | unit | `pytest tests/unit/test_ensemble.py::test_offline_no_cap -x` | ❌ W0 | ⬜ pending |
| 11-01-07 | 01 | 1 | ENS-02 | unit | `pytest tests/unit/test_ensemble.py::test_protocol_compliance -x` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 2 | ENS-02 | unit | `pytest tests/unit/test_evaluator.py -x` | ✅ (extend) | ⬜ pending |
| 11-02-02 | 02 | 2 | ENS-02 | integration | `pytest tests/integration/test_eval_api.py -x` | ✅ (extend) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_ensemble.py` — stubs for ENS-01 (weighted voting, normalization, registry, config) and ENS-02 (live cap, offline no cap, protocol compliance)
- [ ] Extend `tests/unit/test_evaluator.py` — cover ensemble classifier evaluation
- [ ] Extend `tests/integration/test_eval_api.py` — cover ensemble eval endpoint

*Existing test infrastructure (pytest, conftest.py, fixtures) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Ensemble improves over single-model | ENS-02 (SC-3) | Requires trained models with real data | Run evaluation harness with ensemble config, compare F1 to single-model baseline |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
