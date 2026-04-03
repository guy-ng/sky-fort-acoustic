---
phase: 13
slug: dads-dataset-integration-and-training-data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-asyncio |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/unit/test_parquet_dataset.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/test_parquet_dataset.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | DAT-01 | unit | `pytest tests/unit/test_parquet_dataset.py::TestDADSValidation -x` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | DAT-02 | unit | `pytest tests/unit/test_parquet_dataset.py::TestParquetDataset -x` | ❌ W0 | ⬜ pending |
| 13-01-03 | 01 | 1 | DAT-03 | unit | `pytest tests/unit/test_parquet_dataset.py::TestSplitIndices -x` | ❌ W0 | ⬜ pending |
| 13-02-01 | 02 | 2 | DAT-02 | integration | `pytest tests/integration/test_parquet_training.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_parquet_dataset.py` — stubs for DAT-01, DAT-02, DAT-03
- [ ] `tests/integration/test_parquet_training.py` — end-to-end training with Parquet data
- [ ] Framework install: None needed — pytest already configured

*Existing infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end training produces >90% accuracy | DAT-02 | Requires full dataset and GPU time | Run training with DADS data, check final accuracy metric |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
