---
phase: 22
plan: 07
subsystem: evaluation
tags: [eval-harness, promotion-gate, d27, uma16-holdout]
dependency_graph:
  requires: [22-04]
  provides: [evaluate_on_uma16, promote_if_gates_pass, promote_efficientat_cli]
  affects: [22-08, 22-09]
tech_stack:
  added: []
  patterns: [inference-path-eval, sha256-verification, threshold-gating]
key_files:
  created:
    - src/acoustic/evaluation/uma16_eval.py
    - src/acoustic/evaluation/promotion.py
    - scripts/promote_efficientat.py
  modified:
    - src/acoustic/evaluation/__init__.py
    - tests/e2e/test_eval_harness.py
decisions:
  - "Reused _sha256 from uma16_eval in promotion.py via import to avoid duplication"
  - "Preserved existing __init__.py exports (Evaluator, EvaluationResult, etc.) and added new modules"
metrics:
  duration: 5m
  completed: 2026-04-12
---

# Phase 22 Plan 07: Eval Harness and Promotion Gate Summary

UMA-16 real-device evaluation harness and D-27 promotion gate CLI enforcing REAL_TPR >= 0.80 and REAL_FPR <= 0.05 on frozen holdout, running through EfficientATClassifier.predict inference code path.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create evaluation/uma16_eval.py + promotion.py with D-27 constants | d77ae1d (RED), 54a4bd6 (GREEN) | uma16_eval.py, promotion.py, __init__.py |
| 2 | Create scripts/promote_efficientat.py CLI | f4945da | scripts/promote_efficientat.py |

## Implementation Details

### uma16_eval.py
- `evaluate_on_uma16(classifier, manifest_path)` loads each holdout file, verifies sha256, resamples to 32 kHz, slices into 1s non-overlapping segments, runs `classifier.predict` on each segment
- Returns metrics dict with real_TPR, real_FPR, num_drone_segments, num_bg_segments, threshold, per_file breakdown
- Uses EFFICIENTAT_SEGMENT_SAMPLES and EFFICIENTAT_TARGET_SR from window_contract for segment sizing

### promotion.py
- `promote_if_gates_pass(source, target, metrics)` copies checkpoint only if all gates pass
- D-27 constants: REAL_TPR_MIN=0.80, REAL_FPR_MAX=0.05, DADS_ACC_MIN=0.95
- Optional sha256 verification of source checkpoint before promotion
- Optional metrics_out JSON dump

### scripts/promote_efficientat.py
- End-to-end CLI: load checkpoint -> eval on holdout -> gate check -> promote (or abort)
- Exit codes: 0=promoted, 1=gate-failed, 2=precondition-error
- Supports --dry-run, --expected-sha256, --tpr-min/--fpr-max overrides
- Loads classifier via EfficientATClassifier.load() (inference code path)

## Verification Results

- `from acoustic.evaluation.promotion import promote_if_gates_pass, REAL_TPR_MIN, REAL_FPR_MAX` -- OK
- `assert REAL_TPR_MIN == 0.80 and REAL_FPR_MAX == 0.05` -- OK
- `from acoustic.evaluation.uma16_eval import evaluate_on_uma16` -- OK
- `scripts/promote_efficientat.py --help` exits 0 -- OK
- `pytest tests/e2e/test_eval_harness.py`: 4 passed, 1 skipped

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] Preserved existing __init__.py exports**
- **Found during:** Task 1
- **Issue:** Plan said to make __init__.py empty, but it already exports Evaluator, EvaluationResult, FileResult, DistributionStats used by other code
- **Fix:** Kept existing exports and added new module exports alongside them
- **Files modified:** src/acoustic/evaluation/__init__.py

**2. [Rule 2 - DRY] Reused _sha256 helper**
- **Found during:** Task 1
- **Issue:** Plan had _sha256 duplicated in both uma16_eval.py and promotion.py
- **Fix:** Defined _sha256 once in uma16_eval.py, imported it in promotion.py
- **Files modified:** src/acoustic/evaluation/promotion.py

## Known Stubs

None -- all code is fully wired. Full end-to-end execution requires a trained v8 checkpoint (Plan 08/09's job).
