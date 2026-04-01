---
phase: 7
slug: research-cnn-and-inference-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-01
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x with pytest-asyncio |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `python -m pytest tests/unit/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -x` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | MDL-01 | unit | `python -m pytest tests/unit/test_research_cnn.py -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | MDL-01 | unit | `python -m pytest tests/unit/test_research_cnn.py::test_architecture_matches_spec -x` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 1 | MDL-02 | unit | `python -m pytest tests/unit/test_aggregation.py -x` | ❌ W0 | ⬜ pending |
| 07-01-04 | 01 | 1 | MDL-03 | unit | `python -m pytest tests/unit/test_protocols.py::TestAggregatorProtocol -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | MDL-02 | unit | `python -m pytest tests/unit/test_worker.py::TestSegmentBuffer -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | MDL-03 | unit | `python -m pytest tests/unit/test_worker.py::TestCNNWorkerConstructor -x` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | MDL-03 | integration | `python -m pytest tests/integration/test_cnn_pipeline.py -x` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 2 | MDL-04 | unit | `python -m pytest tests/unit/test_config.py -x` | existing (extend) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_research_cnn.py` — stubs for MDL-01 (model shape, architecture)
- [ ] `tests/unit/test_aggregation.py` — stubs for MDL-02 (WeightedAggregator, edge cases)
- [ ] Extend `tests/unit/test_protocols.py` with `TestAggregatorProtocol` — MDL-03
- [ ] Extend `tests/unit/test_worker.py` with `TestSegmentBuffer`, `TestCNNWorkerConstructor` — MDL-02/MDL-03
- [ ] Extend `tests/unit/test_config.py` with aggregation weight fields — MDL-04

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end beamforming + CNN no regression | MDL-04 (SC5) | Requires live audio or full simulation | Start service, verify beamforming heatmap still updates, check no errors in logs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
