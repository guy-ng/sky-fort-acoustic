---
phase: 10
slug: field-data-collection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-asyncio |
| **Config file** | `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/unit/test_recording*.py tests/unit/test_recorder.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/test_recording*.py tests/unit/test_recorder.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | COL-01 | unit | `pytest tests/unit/test_recorder.py -x -q` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | COL-02 | unit | `pytest tests/unit/test_recording_metadata.py -x -q` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 1 | COL-03 | unit | `pytest tests/unit/test_recorder.py::test_label_moves_file -x -q` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 1 | COL-01 | integration | `pytest tests/integration/test_recording_api.py -x -q` | ❌ W0 | ⬜ pending |
| 10-02-02 | 02 | 1 | COL-02 | integration | `pytest tests/integration/test_recording_api.py::test_update_metadata -x -q` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | COL-01 | manual | N/A (frontend) | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_recorder.py` — RecordingSession unit tests (mono downmix, resampling, WAV write, label file move)
- [ ] `tests/unit/test_recording_metadata.py` — Sidecar JSON CRUD operations
- [ ] `tests/integration/test_recording_api.py` — REST endpoint tests (start, stop, list, metadata update)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Recording panel UI renders in dashboard | COL-01 | Frontend rendering requires browser | Open web UI, verify recording panel visible in dashboard |
| Inline label form appears after stop | COL-01 | UI interaction flow | Start recording, stop, verify inline form appears (not modal) |
| Audio level meter updates during recording | COL-01 | Real-time visual feedback | Start recording, verify level meter animates |
| Metadata edit from recordings list | COL-02 | UI interaction flow | Open recordings list, click edit, verify fields editable |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
