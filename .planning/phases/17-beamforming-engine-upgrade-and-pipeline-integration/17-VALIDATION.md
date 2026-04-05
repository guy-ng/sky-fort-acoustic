---
phase: 17
slug: beamforming-engine-upgrade-and-pipeline-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/unit/test_bandpass.py tests/unit/test_interpolation.py tests/unit/test_mcra.py tests/unit/test_multi_peak.py tests/unit/test_bf_gate.py tests/integration/test_pipeline_beamforming.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run the relevant task test file (see Per-Task map below)
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | BF-10, BF-11 | — | N/A | unit | `python -m pytest tests/unit/test_bandpass.py -x` | :x: W0 | :white_large_square: pending |
| 17-01-02 | 01 | 1 | BF-12 | — | N/A | unit | `python -m pytest tests/unit/test_interpolation.py -x` | :x: W0 | :white_large_square: pending |
| 17-02-01 | 02 | 1 | BF-14 | — | N/A | unit | `python -m pytest tests/unit/test_mcra.py -x` | :x: W0 | :white_large_square: pending |
| 17-02-02 | 02 | 1 | BF-13 | — | N/A | unit | `python -m pytest tests/unit/test_multi_peak.py -x` | :x: W0 | :white_large_square: pending |
| 17-03-01 | 03 | 2 | BF-16 | — | N/A | unit | `python -m pytest tests/unit/test_bf_gate.py -x` | :x: W0 | :white_large_square: pending |
| 17-03-02 | 03 | 2 | BF-15 | — | N/A | integration | `python -m pytest tests/integration/test_pipeline_beamforming.py -x` | :x: W0 | :white_large_square: pending |

*Status: :white_large_square: pending · :white_check_mark: green · :x: red · :warning: flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_bandpass.py` — stubs for BF-10, BF-11
- [ ] `tests/unit/test_interpolation.py` — stubs for BF-12
- [ ] `tests/unit/test_mcra.py` — stubs for BF-14
- [ ] `tests/unit/test_multi_peak.py` — stubs for BF-13
- [ ] `tests/unit/test_bf_gate.py` — stubs for BF-16
- [ ] `tests/integration/test_pipeline_beamforming.py` — stubs for BF-15
- [ ] `tests/conftest.py` — shared fixtures (mock audio chunks, array geometry)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-time spatial map updates at 150ms rate | BF-14 | Requires live audio hardware | Start pipeline with mic array, verify WebSocket spatial map updates in browser |
| MCRA adapts to changing outdoor noise | BF-13 | Requires environmental changes | Run system outdoors, introduce varying background noise, verify noise floor tracks |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
