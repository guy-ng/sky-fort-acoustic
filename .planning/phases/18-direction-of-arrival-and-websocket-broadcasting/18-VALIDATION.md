---
phase: 18
slug: direction-of-arrival-and-websocket-broadcasting
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `python -m pytest tests/ -x -q --tb=short` |
| **Full suite command** | `python -m pytest tests/ -v --tb=long` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `python -m pytest tests/ -v --tb=long`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | DOA-01 | — | N/A | unit | `python -m pytest tests/unit/test_doa.py -k "test_coordinate_transform"` | ❌ W0 | ⬜ pending |
| 18-01-02 | 01 | 1 | DOA-02 | — | N/A | unit | `python -m pytest tests/unit/test_doa.py -k "test_broadside_zero"` | ❌ W0 | ⬜ pending |
| 18-02-01 | 02 | 2 | DOA-03 | — | N/A | unit | `python -m pytest tests/unit/test_tracker.py -k "test_multi_target"` | ❌ W0 | ⬜ pending |
| 18-02-02 | 02 | 2 | DOA-03 | — | N/A | unit | `python -m pytest tests/unit/test_tracker.py -k "test_association"` | ❌ W0 | ⬜ pending |
| 18-03-01 | 03 | 3 | DIR-01 | — | N/A | unit | `python -m pytest tests/unit/test_target_schema.py -k "test_target_event_direction"` | ❌ W0 | ⬜ pending |
| 18-03-02 | 03 | 3 | DIR-02 | — | N/A | integration | `python -m pytest tests/unit/test_events_ws.py -k "test_ws_direction"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_doa.py` — stubs for DOA-01, DOA-02 (coordinate transform)
- [ ] `tests/unit/test_tracker.py` — extend with multi-target tests for DOA-03
- [ ] `tests/unit/test_target_schema.py` — extend with pan_deg/tilt_deg tests for DIR-01
- [ ] `tests/unit/test_events_ws.py` — extend with direction broadcast tests for DIR-02

*Existing infrastructure covers framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Physical broadside produces pan=0, tilt=0 | DOA-02 | Requires physical UMA-16v2 hardware | Place sound source directly in front of array, verify pan=0/tilt=0 in /ws/targets output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
