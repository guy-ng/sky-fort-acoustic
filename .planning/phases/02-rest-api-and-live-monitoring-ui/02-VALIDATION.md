---
phase: 2
slug: rest-api-and-live-monitoring-ui
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-31
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-asyncio 0.24+ |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-xx | 01 | 1 | API-01 | integration | `pytest tests/integration/test_api.py::test_map_endpoint -x` | No -- Wave 0 | pending |
| 02-01-xx | 01 | 1 | API-02 | integration | `pytest tests/integration/test_api.py::test_targets_endpoint -x` | No -- Wave 0 | pending |
| 02-01-xx | 01 | 1 | API-03 | integration | `pytest tests/integration/test_websocket.py::test_heatmap_ws -x` | No -- Wave 0 | pending |
| 02-02-xx | 02 | 1 | UI-01 | manual | Browser check -- Canvas rendering | N/A | pending |
| 02-02-xx | 02 | 1 | UI-02 | manual | Browser check -- target overlay | N/A | pending |
| 02-02-xx | 02 | 1 | UI-03 | manual | Browser check -- target strip | N/A | pending |
| 02-02-xx | 02 | 1 | UI-08 | manual | Visual comparison with sky-fort-dashboard | N/A | pending |
| 02-01-xx | 01 | 1 | INF-02 | integration | `docker build -t sky-fort-acoustic .` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/integration/test_api.py` -- stubs for API-01, API-02
- [ ] `tests/integration/test_websocket.py` -- stubs for API-03
- [ ] Integration test fixtures for FastAPI TestClient with mock pipeline

*Frontend tests (UI-01, UI-02, UI-03, UI-08) are manual browser verification -- no automated test infrastructure needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live heatmap renders on canvas | UI-01 | Canvas 2D rendering not testable via pytest | Open browser, verify colored heatmap updates in real time |
| Target overlay markers appear | UI-02 | Visual overlay on canvas | Open browser, verify target markers at correct positions |
| Target strip shows cards | UI-03 | React component rendering | Open browser, verify cards with ID, class, speed, bearing |
| Styling matches sky-fort-dashboard | UI-08 | Visual comparison | Compare side-by-side: dark theme, Panel components, fonts |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
