---
phase: 9
slug: evaluation-harness-and-api
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 with pytest-asyncio |
| **Config file** | pyproject.toml (`asyncio_mode = "auto"`) |
| **Quick run command** | `python -m pytest tests/unit/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | EVL-01, EVL-02 | unit | `python -m pytest tests/unit/test_evaluator.py -x` | ❌ W0 | ⬜ pending |
| 09-02-01 | 02 | 2 | API-01 | integration | `python -m pytest tests/integration/test_training_api.py tests/integration/test_eval_api.py -x` | ❌ W0 | ⬜ pending |
| 09-02-02 | 02 | 2 | API-02 | integration | `python -m pytest tests/integration/test_training_ws.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_evaluator.py` — stubs for EVL-01, EVL-02
- [ ] `tests/integration/test_training_api.py` — stubs for API-01 (training endpoints)
- [ ] `tests/integration/test_eval_api.py` — stubs for API-01 (eval endpoints)
- [ ] `tests/integration/test_training_ws.py` — stubs for API-02

*Existing test infrastructure (conftest, fixtures) covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| WebSocket real-time epoch streaming | API-02 | Requires active training run with real model | Start training via POST /api/training/start, connect to /ws/training, verify epoch messages arrive |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
