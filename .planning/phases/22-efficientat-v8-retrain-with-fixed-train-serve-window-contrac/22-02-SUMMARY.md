---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 02
subsystem: efficientat-window-contract

# Dependency graph
requires:
  - phase: 22
    plan: 01
    provides: wave-0 test scaffolds (test_window_contract.py exists as importorskip scaffold)
provides:
  - src/acoustic/classification/efficientat/window_contract.py — single source of truth for EfficientAT window length, target SR, and segment samples
  - Eliminates v7 regression literal int(0.5 * _SOURCE_SR) from trainer
  - Pipeline + hf_dataset wired to window_contract imports
affects: [22-03, 22-04, 22-06, 22-07, 22-08]

# Tech tracking
tech-stack:
  added: []
  patterns: [import-time contract self-check, source-SR-aware helper, single-source-of-truth constants]

key-files:
  created:
    - src/acoustic/classification/efficientat/window_contract.py
  modified:
    - src/acoustic/training/efficientat_trainer.py
    - src/acoustic/pipeline.py
    - src/acoustic/training/hf_dataset.py
    - tests/unit/classification/efficientat/test_window_contract.py
---

# Plan 22-02 — Window Contract Module (The Linchpin)

## What was built

Phase 22 linchpin: `src/acoustic/classification/efficientat/window_contract.py` is the single authoritative declaration of the EfficientAT train/serve window contract:

- `EFFICIENTAT_WINDOW_SECONDS = 1.0`
- `EFFICIENTAT_TARGET_SR = 32000`
- `EFFICIENTAT_SEGMENT_SAMPLES = EfficientATMelConfig().segment_samples` (32000)
- `source_window_samples(source_sr)` helper for the training path (returns 16000 at 16kHz source)
- Import-time self-check: asserts `EFFICIENTAT_SEGMENT_SAMPLES == int(EFFICIENTAT_WINDOW_SECONDS * EFFICIENTAT_TARGET_SR)`. Crashes at import if the mel config ever drifts.

## v7 bug elimination

The regression literal at `src/acoustic/training/efficientat_trainer.py:456` —

```python
window_samples = int(0.5 * _SOURCE_SR)  # 8000 samples = 0.5 s @ 16 kHz
```

— was replaced with:

```python
window_samples = source_window_samples(_SOURCE_SR)  # 1.0 s @ source SR (= 16000 @ 16kHz)
```

The trainer now imports `EFFICIENTAT_SEGMENT_SAMPLES` and `source_window_samples` at the top of the file. `grep "int(0.5 \* _SOURCE_SR)" src/acoustic/` returns zero matches.

## Consumer wiring

- **pipeline.py**: `_training_window_seconds` returns `EFFICIENTAT_WINDOW_SECONDS` for the efficientat branch (research_cnn still returns 0.5 intentionally).
- **hf_dataset.py**: `WindowedHFDroneDataset` defaults updated to `window_samples=16000`, `hop_samples=8000` matching the 1.0s / 50% overlap contract at 16kHz source SR. Imports `EFFICIENTAT_SEGMENT_SAMPLES` and `source_window_samples` (consumed by Plan 22-03 for the length assertion).
- **test_window_contract.py** (Wave 0 scaffold): updated from importorskip-guarded scaffold into a normal test module that exercises the now-existing constants and helper. 6 tests pass.

## Verification

```
$ python -m pytest tests/unit/classification/efficientat/test_window_contract.py -x -q
......                                                                   [100%]
6 passed in 0.84s
```

## Notes

- SUMMARY.md was written post-hoc by the orchestrator because the spawned executor agent hit the usage limit on its final wrap-up message. All code commits landed correctly in the worktree and have been merged to main.
- Commits on main: `3489745`, `d156891`, `fecacec` (plus merge commit `9af37ba`).
