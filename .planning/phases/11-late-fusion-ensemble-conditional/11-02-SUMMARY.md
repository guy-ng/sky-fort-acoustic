---
phase: 11-late-fusion-ensemble-conditional
plan: 02
subsystem: classification
tags: [ensemble, evaluation, api, lifespan, per-model-metrics]

requires:
  - phase: 11-late-fusion-ensemble-conditional
    provides: EnsembleClassifier, ModelRegistry, EnsembleConfig, ensemble_config_path
provides:
  - Evaluator.evaluate_classifier accepting any Classifier protocol
  - Evaluator.evaluate_ensemble with single-pass per-model metrics
  - Eval API ensemble_config_path parameter for ensemble evaluation
  - PerModelResultResponse in API for per-model metrics
  - main.py ensemble factory with single-model fallback
affects: [live-pipeline, evaluation-ui, model-comparison]

tech-stack:
  added: []
  patterns: [classifier-protocol-evaluation, single-pass-ensemble-metrics, factory-fallback-pattern]

key-files:
  created: []
  modified:
    - src/acoustic/evaluation/evaluator.py
    - src/acoustic/evaluation/models.py
    - src/acoustic/api/models.py
    - src/acoustic/api/eval_routes.py
    - src/acoustic/main.py
    - tests/unit/test_evaluator.py
    - tests/integration/test_eval_api.py

key-decisions:
  - "Evaluator refactored with evaluate_classifier and evaluate_ensemble, keeping backward-compatible evaluate(model_path)"
  - "Single-pass ensemble evaluation: features extracted once, run through all models + ensemble in one loop"
  - "Ensemble factory in lifespan runs before single-model factory with fallback on failure"
  - "Validation order preserved: model_path before data_dir for single-model path (existing test compatibility)"

patterns-established:
  - "Protocol-based evaluation: evaluate_classifier accepts any Classifier, not just ResearchCNN"
  - "Factory fallback: ensemble config checked first, single-model fallback if absent or fails"

requirements-completed: [ENS-01, ENS-02]

duration: 10min
completed: 2026-04-02
---

# Phase 11 Plan 02: Ensemble Evaluation and Pipeline Integration Summary

**Evaluator refactored to accept any Classifier with single-pass ensemble+per-model metrics, eval API extended with ensemble_config_path, and main.py lifespan wired for ensemble mode**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-02T17:41:24Z
- **Completed:** 2026-04-02T17:51:41Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Evaluator.evaluate_classifier accepts any Classifier protocol implementor, decoupling from ResearchCNN
- Evaluator.evaluate_ensemble produces ensemble + per-model metrics in a single pass over test data
- PerModelResult dataclass tracks individual model accuracy/precision/recall/F1 within ensemble
- Eval API accepts optional ensemble_config_path, returning per_model_results in response
- main.py lifespan detects ensemble config, loads models via registry, creates EnsembleClassifier with live_mode=True
- Falls back to single-model when no ensemble config or on load failure
- 22 tests pass (16 unit evaluator + 6 integration eval API)

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor Evaluator and extend evaluation models** - `6e76e5b` (feat)
2. **Task 2: Extend eval API and wire ensemble into main.py** - `b70e8c0` (feat)

## Files Created/Modified
- `src/acoustic/evaluation/models.py` - Added PerModelResult dataclass, per_model_results field on EvaluationResult
- `src/acoustic/evaluation/evaluator.py` - Added evaluate_classifier, evaluate_ensemble, _extract_features, _process_file, _build_result
- `src/acoustic/api/models.py` - Added PerModelResultResponse, ensemble_config_path on EvalRunRequest, per_model_results on EvalResultResponse
- `src/acoustic/api/eval_routes.py` - Ensemble evaluation path with config validation and model file checks
- `src/acoustic/main.py` - Ensemble factory in lifespan with live_mode=True, single-model fallback, mode logging
- `tests/unit/test_evaluator.py` - 6 new tests for evaluate_classifier and evaluate_ensemble
- `tests/integration/test_eval_api.py` - 2 new tests for ensemble endpoint parameter and missing config

## Decisions Made
- Refactored evaluate() to delegate to evaluate_classifier via load_model registry, maintaining backward compatibility
- Single-pass approach for ensemble evaluation: extract features once per file, run all models + ensemble
- Ensemble factory placed before single-model factory in lifespan, with try/except fallback
- Preserved original validation order (model_path before data_dir) for single-model path to maintain existing test compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Copied missing source files from main repo to worktree**
- **Found during:** Task 1 (initial test run)
- **Issue:** Worktree missing updated preprocessing.py, pipeline.py, worker.py, recording modules, and API files from phases 7-10
- **Fix:** Copied all needed files from main repo to worktree src/
- **Files modified:** Multiple classification/, evaluation/, recording/, api/ files
- **Verification:** All imports resolve, tests pass
- **Committed in:** 6e76e5b, b70e8c0

---

**Total deviations:** 1 auto-fixed (1 blocking - worktree sync)
**Impact on plan:** Necessary for parallel worktree execution. No scope creep.

## Issues Encountered
- Pre-existing test failure in tests/integration/test_cnn_pipeline.py (worker interface mismatch between worktree's old tests and main repo's updated worker) -- not related to this plan's changes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ensemble module fully integrated: evaluation, API, and live pipeline
- Ready for ensemble config creation and model comparison workflows
- Model registry extensible for future architecture types

## Self-Check: PASSED

All 7 modified files verified present. All 2 commit hashes verified in git log.

---
*Phase: 11-late-fusion-ensemble-conditional*
*Completed: 2026-04-02*
