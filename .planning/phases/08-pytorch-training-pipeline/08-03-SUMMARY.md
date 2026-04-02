---
phase: 08-pytorch-training-pipeline
plan: 03
subsystem: training
tags: [torchscript, confusion-matrix, thread-limits, gap-closure]
dependency_graph:
  requires: [08-02]
  provides: [torchscript-export, confusion-matrix-metrics, thread-isolation]
  affects: [training-pipeline, training-progress-api]
tech_stack:
  added: []
  patterns: [torchscript-export, confusion-matrix-tracking, thread-pool-isolation]
key_files:
  created:
    - tests/unit/test_training_export.py
  modified:
    - src/acoustic/training/trainer.py
    - src/acoustic/training/manager.py
decisions:
  - TorchScript export alongside state_dict (not replacing it) for deployable format
  - Graceful handling of set_num_interop_threads RuntimeError (only callable once per process)
metrics:
  duration: 6min
  completed: 2026-04-02
---

# Phase 08 Plan 03: Gap Closure (TorchScript, Thread Limits, Confusion Matrix) Summary

TorchScript .jit export after training, torch.set_num_threads(2) for CPU isolation, and per-epoch confusion matrix (TP/FP/TN/FN) in progress state.

## What Was Done

### Task 1: TorchScript export, confusion matrix, and thread limits (TDD)

**TorchScript Export (ROADMAP SC3):**
- After training saves best checkpoint, loads it into a fresh ResearchCNN, runs `torch.jit.script()`, saves as `{checkpoint_path}.jit`
- Export failure is non-fatal -- logged but state_dict checkpoint remains available
- .jit file is an additional deployable format alongside the primary .pt state_dict

**Confusion Matrix (TRN-03):**
- TP/FP/TN/FN computed per validation epoch using threshold 0.5
- Included in every progress callback dict alongside existing metrics
- Four new fields added to TrainingProgress dataclass (tp, fp, tn, fn)
- _on_progress() updated to read confusion matrix keys from callback

**Thread Limits (ROADMAP SC2):**
- `torch.set_num_threads(2)` called in TrainingManager._run() before creating TrainingRunner
- `torch.set_num_interop_threads(1)` called with try/except for RuntimeError (can only be set once per process)
- Combined, training uses at most 2-3 CPU threads for PyTorch ops, leaving rest for beamforming

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 8df6b24 | test | Failing tests for TorchScript export, confusion matrix, thread limits (TDD RED) |
| 9670558 | feat | Implementation: TorchScript export, confusion matrix, thread limits (TDD GREEN) |

## Test Results

- `tests/unit/test_training_export.py`: 5 passed (new)
- `tests/unit/test_training_checkpoint.py`: 5 passed (no regression)
- `tests/unit/test_training_manager.py`: 8 passed (no regression)
- `tests/integration/test_training_smoke.py`: 5 passed (no regression)
- Full unit suite: 187 passed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Graceful handling of torch.set_num_interop_threads**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** `torch.set_num_interop_threads()` raises RuntimeError if called after parallel work has started or if called more than once per process. Existing test_training_manager tests triggered this.
- **Fix:** Wrapped in try/except RuntimeError with pass -- the limit is best-effort since it can only be set once.
- **Files modified:** src/acoustic/training/manager.py
- **Commit:** 9670558

## Verification Results

All acceptance criteria verified:
- `grep "torch.jit.script" trainer.py` -- matches line 242
- `grep "torch.set_num_threads" manager.py` -- matches line 128
- `grep "torch.set_num_interop_threads" manager.py` -- matches line 130
- `grep "import torch" manager.py` -- matches line 16
- Confusion matrix in callback -- lines 183, 226
- TrainingProgress fields tp/fp/tn/fn -- lines 47-50
- All tests pass with zero regressions

## Known Stubs

None -- all implementations are fully wired.

## Self-Check: PASSED
