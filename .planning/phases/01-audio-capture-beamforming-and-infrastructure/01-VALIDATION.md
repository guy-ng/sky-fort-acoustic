---
phase: 1
slug: audio-capture-beamforming-and-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x + pytest-asyncio 1.3.x |
| **Config file** | pyproject.toml [tool.pytest.ini_options] — Wave 0 installs |
| **Quick run command** | `pytest tests/unit/ -x -q` |
| **Full suite command** | `pytest tests/ -x -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/ -x -q`
- **After every plan wave:** Run `pytest tests/ -x -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-xx | 01 | 1 | AUD-01 | unit (simulated) | `pytest tests/unit/test_capture.py -x` | ❌ W0 | ⬜ pending |
| 01-01-xx | 01 | 1 | AUD-02 | unit | `pytest tests/unit/test_ring_buffer.py -x` | ❌ W0 | ⬜ pending |
| 01-01-xx | 01 | 1 | AUD-03 | unit (mocked) | `pytest tests/unit/test_device.py -x` | ❌ W0 | ⬜ pending |
| 01-02-xx | 02 | 1 | BF-01 | unit | `pytest tests/unit/test_srp_phat.py -x` | ❌ W0 | ⬜ pending |
| 01-02-xx | 02 | 1 | BF-02 | unit | `pytest tests/unit/test_srp_phat.py::test_freq_band -x` | ❌ W0 | ⬜ pending |
| 01-02-xx | 02 | 1 | BF-03 | unit | `pytest tests/unit/test_peak.py -x` | ❌ W0 | ⬜ pending |
| 01-02-xx | 02 | 1 | BF-04 | unit | `pytest tests/unit/test_peak.py::test_noise_threshold -x` | ❌ W0 | ⬜ pending |
| 01-01-xx | 01 | 1 | INF-01 | integration | `docker build -t acoustic . && docker run --rm acoustic python -c "import acoustic"` | ❌ W0 | ⬜ pending |
| 01-01-xx | 01 | 1 | INF-02 | manual | Visual inspection | ❌ W0 | ⬜ pending |
| 01-01-xx | 01 | 1 | INF-03 | unit | `pytest tests/unit/test_config.py -x` | ❌ W0 | ⬜ pending |
| 01-03-xx | 03 | 2 | INF-04 | integration | `pytest tests/integration/test_health.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — project config with [tool.pytest.ini_options] (asyncio_mode = "auto")
- [ ] `tests/conftest.py` — shared fixtures (mic positions, synthetic audio generator, mock device list)
- [ ] `tests/unit/test_ring_buffer.py` — AUD-02
- [ ] `tests/unit/test_srp_phat.py` — BF-01, BF-02
- [ ] `tests/unit/test_peak.py` — BF-03, BF-04
- [ ] `tests/unit/test_config.py` — INF-03
- [ ] `tests/unit/test_device.py` — AUD-03
- [ ] `tests/unit/test_capture.py` — AUD-01
- [ ] `tests/integration/test_health.py` — INF-04
- [ ] Framework install: `pip install pytest pytest-asyncio httpx`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dockerfile exists (single-stage for Phase 1) | INF-02 | Build artifact — visual check sufficient | Verify `Dockerfile` exists with `FROM python:3.11-slim` and ALSA deps |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
