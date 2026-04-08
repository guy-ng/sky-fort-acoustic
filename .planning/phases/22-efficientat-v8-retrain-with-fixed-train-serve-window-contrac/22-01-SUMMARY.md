---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 01
subsystem: testing
tags: [pytest, efficientat, window-contract, model-provenance, sha256]

# Dependency graph
requires:
  - phase: 20
    provides: efficientat_mn10_v6.pt checkpoint and training pipeline
provides:
  - 8 pytest test scaffold files for Phase 22 validation (window contract, dataset length, predict WARN, RMS parity, preflight, cardinality, holdout split, eval harness)
  - MODEL_PROVENANCE.md with sha256 hashes for v6, v7, and default checkpoint
  - Identity mismatch flag (efficientat_mn10.pt is NOT v6)
affects: [22-02, 22-03, 22-04, 22-06, 22-07]

# Tech tracking
tech-stack:
  added: []
  patterns: [importorskip for future-module tests, xfail scaffolds for wave-0-first validation]

key-files:
  created:
    - tests/unit/classification/efficientat/test_window_contract.py
    - tests/unit/training/test_windowed_dataset_length.py
    - tests/unit/classification/test_efficientat_predict_warn.py
    - tests/unit/training/test_rmsnormalize_parity.py
    - tests/integration/test_data_integrity_preflight.py
    - tests/integration/test_dataset_cardinality.py
    - tests/integration/test_holdout_split.py
    - tests/e2e/test_eval_harness.py
    - tests/fixtures/efficientat_v8/README.md
    - models/MODEL_PROVENANCE.md
  modified: []

key-decisions:
  - "efficientat_mn10.pt is NOT v6 -- different sha256 and size. Flagged as pre-v6 checkpoint requiring resolution before v8 promotion."
  - "Live service default loads research_cnn ONNX model, not EfficientAT. EfficientAT requires ACOUSTIC_CNN_MODEL_PATH + ACOUSTIC_CNN_MODEL_TYPE env vars."

patterns-established:
  - "importorskip pattern: test files for modules that don't exist yet use pytest.importorskip at module level"
  - "xfail scaffold pattern: tests for future-wave features use pytest.mark.xfail(strict=False) so they turn green automatically"

requirements-completed: [REQ-22-W1, REQ-22-W2, REQ-22-W3, REQ-22-W4, REQ-22-D3, REQ-22-G1]

# Metrics
duration: 6min
completed: 2026-04-08
---

# Phase 22 Plan 01: Wave 0 Test Scaffolds + Model Provenance Summary

**8 pytest test scaffolds for Phase 22 validation targets plus sha256-locked model provenance identifying efficientat_mn10.pt as a pre-v6 checkpoint**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-08T14:27:59Z
- **Completed:** 2026-04-08T14:34:11Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments
- Created 8 pytest-collectible test files covering all Phase 22 validation requirements (unit, integration, e2e)
- Locked model provenance with sha256 hashes for efficientat_mn10_v6.pt, efficientat_mn10.pt, and efficientat_mn10_v7.pt
- Discovered and documented that efficientat_mn10.pt is NOT v6 (different hash: 1b9a5162 vs c8828b5d) -- critical finding for v8 promotion safety

## Task Commits

Each task was committed atomically:

1. **Task 1: Create unit test scaffolds** - `a1927a9` (test)
2. **Task 2: Create integration + e2e test scaffolds** - `ea788d4` (test)
3. **Task 3: Lock model provenance** - `f5a84c7` (docs)

## Files Created/Modified
- `tests/unit/classification/efficientat/__init__.py` - New test package
- `tests/unit/classification/__init__.py` - New test package
- `tests/unit/classification/efficientat/test_window_contract.py` - Window contract constants (importorskip until Plan 02)
- `tests/unit/training/test_windowed_dataset_length.py` - Dataset length assertion (xfail until Plan 03)
- `tests/unit/classification/test_efficientat_predict_warn.py` - Classifier WARN test (xfail until Plan 03)
- `tests/unit/training/test_rmsnormalize_parity.py` - Train/serve RMS parity (xfail until Plan 03)
- `tests/integration/test_data_integrity_preflight.py` - Field data preflight (xfail until Plan 04)
- `tests/integration/test_dataset_cardinality.py` - Dataset cardinality (xfail until Plan 04/06)
- `tests/integration/test_holdout_split.py` - Holdout split determinism (xfail until Plan 04)
- `tests/e2e/__init__.py` - New e2e test package
- `tests/e2e/test_eval_harness.py` - Eval harness D-27 gates (xfail until Plan 07)
- `tests/fixtures/efficientat_v8/README.md` - Fixture directory documentation
- `models/MODEL_PROVENANCE.md` - sha256 provenance for all EfficientAT checkpoints

## Decisions Made
- **efficientat_mn10.pt identity mismatch:** sha256 1b9a5162... does NOT match v6 (c8828b5d...). Size difference confirms it (17020041 vs 17019638 bytes). This is a pre-v6 checkpoint. Flagged in MODEL_PROVENANCE.md for resolution before v8 promotion.
- **Live service model path:** Default is research_cnn ONNX, not EfficientAT. Operators must set ACOUSTIC_CNN_MODEL_PATH and ACOUSTIC_CNN_MODEL_TYPE env vars for EfficientAT inference.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Model .pt files are gitignored and not present in worktrees. Used main repo path for sha256 computation. MODEL_PROVENANCE.md is committed as documentation regardless.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 8 test files exist and are pytest-collectible (0 collection errors)
- MODEL_PROVENANCE.md committed with sha256 values and code path citations
- Wave 1+ (Plans 02-07) can now land code that turns these xfail/skip tests green
- **Action needed before v8 promotion:** Resolve efficientat_mn10.pt identity -- confirm no deployment depends on current (pre-v6) contents

## Self-Check: PASSED

All 10 created files verified on disk. All 3 task commits (a1927a9, ea788d4, f5a84c7) verified in git log.

---
*Phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac*
*Completed: 2026-04-08*
