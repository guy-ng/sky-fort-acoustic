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
| **Quick run command** | `python -m pytest tests/test_beamforming.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_beamforming.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | BF-10 | — | N/A | unit | `python -m pytest tests/test_beamforming.py::test_bandpass_filter -x` | ❌ W0 | ⬜ pending |
| 17-01-02 | 01 | 1 | BF-11 | — | N/A | unit | `python -m pytest tests/test_beamforming.py::test_parabolic_interpolation -x` | ❌ W0 | ⬜ pending |
| 17-01-03 | 01 | 1 | BF-12 | — | N/A | unit | `python -m pytest tests/test_beamforming.py::test_multi_peak_detection -x` | ❌ W0 | ⬜ pending |
| 17-02-01 | 02 | 1 | BF-13 | — | N/A | unit | `python -m pytest tests/test_beamforming.py::test_mcra_noise_estimation -x` | ❌ W0 | ⬜ pending |
| 17-03-01 | 03 | 2 | BF-14, BF-15, BF-16 | — | N/A | integration | `python -m pytest tests/test_pipeline_beamforming.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_beamforming.py` — stubs for BF-10, BF-11, BF-12, BF-13
- [ ] `tests/test_pipeline_beamforming.py` — stubs for BF-14, BF-15, BF-16
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
