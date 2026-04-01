---
phase: 6
slug: preprocessing-parity-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-01
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ with pytest-asyncio |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `python -m pytest tests/unit/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

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
| 06-01-01 | 01 | 1 | PRE-01 | unit | `pytest tests/unit/test_mel_config.py -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | PRE-02 | unit | `pytest tests/unit/test_protocols.py -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | PRE-03 | unit | `pytest tests/unit/test_preprocessing.py -x` | ✅ needs rewrite | ⬜ pending |
| 06-01-04 | 01 | 1 | PRE-04 | unit | `pytest tests/unit/test_parity.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_mel_config.py` — stubs for PRE-01 (MelConfig constants, no duplicate magic numbers)
- [ ] `tests/unit/test_protocols.py` — stubs for PRE-02 (protocol structural typing, isinstance checks)
- [ ] `tests/unit/test_parity.py` — stubs for PRE-04 (numerical parity with .npy fixtures)
- [ ] `tests/fixtures/reference_melspec_440hz.npy` — reference tensor from librosa for parity test
- [ ] `tests/unit/test_preprocessing.py` — EXISTS but needs complete rewrite for new preprocessor
- [ ] `tests/unit/test_inference.py` — EXISTS, must be DELETED (OnnxDroneClassifier removed)
- [ ] `tests/fixtures/dummy_model.onnx` — EXISTS, must be DELETED (ONNX removed)
- [ ] torchaudio installation: `pip install torchaudio==2.11.0`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No magic numbers in codebase | PRE-01 | Requires grep sweep | `grep -rn "16000\|1024\|256\|N_MELS\|64\|128" src/ --include="*.py"` — only MelConfig should define these |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
