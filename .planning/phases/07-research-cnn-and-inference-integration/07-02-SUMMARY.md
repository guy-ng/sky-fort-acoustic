---
phase: 07-research-cnn-and-inference-integration
plan: 02
subsystem: classification
tags: [pytorch, cnn, aggregation, pipeline, inference, factory-pattern]

requires:
  - phase: 07-research-cnn-and-inference-integration
    plan: 01
    provides: "ResearchCNN, ResearchClassifier, WeightedAggregator, Aggregator protocol"
provides:
  - "CNNWorker with segment buffer (deque maxlen=4) and aggregator injection"
  - "Pipeline 50% overlap (0.25s push interval for 0.5s segments)"
  - "Classifier factory loading ResearchCNN from cnn_model_path with validation"
  - "Aggregator wiring with config-driven weights"
affects: [training-pipeline, live-inference, model-deployment]

tech-stack:
  added: []
  patterns: [segment-aggregation-buffer, classifier-factory-with-validation, overlap-push-pattern]

key-files:
  created:
    - tests/integration/test_cnn_pipeline.py (TestFactoryWiring, TestPipelineOverlap classes)
  modified:
    - src/acoustic/classification/worker.py
    - src/acoustic/pipeline.py
    - src/acoustic/main.py
    - tests/unit/test_worker.py
    - tests/integration/test_cnn_pipeline.py

key-decisions:
  - "Silence-gated segments do not append to aggregation deque (avoids contaminating rolling window)"
  - "Classifier factory validates model with dummy forward pass before accepting"
  - "Integration tests fixed to use keyword args matching new constructor signature"

patterns-established:
  - "Segment aggregation: deque(maxlen=N) accumulates per-segment probs, aggregator produces final p_agg"
  - "Factory pattern: load model from config path, validate, or fall back to dormant mode"

requirements-completed: [MDL-02, MDL-03, MDL-04]

duration: 7min
completed: 2026-04-01
---

# Phase 7 Plan 2: Pipeline Integration and Classifier Factory Summary

**CNNWorker segment buffer with deque(maxlen=4) aggregation, 50% overlap push (0.25s interval), and classifier factory loading ResearchCNN from config path**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-01T12:09:06Z
- **Completed:** 2026-04-01T12:16:00Z
- **Tasks:** 2 (Task 1 TDD, Task 2 standard)
- **Files modified:** 5

## Accomplishments
- CNNWorker now accepts Aggregator as third protocol dependency with deque-based segment buffer
- Pipeline pushes 0.5s segments every 0.25s for 50% overlap, improving classification stability
- Classifier factory in main.py loads ResearchCNN from cnn_model_path with dummy forward pass validation
- WeightedAggregator wired with config-driven weights (cnn_agg_w_max, cnn_agg_w_mean)
- State machine receives aggregated p_agg without any state machine code changes
- All 145 unit tests + 15 integration tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for segment buffer** - `8041a62` (test)
2. **Task 1 GREEN: CNNWorker segment buffer implementation** - `6a9b009` (feat)
3. **Task 2: Pipeline overlap, factory, integration tests** - `8e28568` (feat)

## Files Created/Modified
- `src/acoustic/classification/worker.py` - Added aggregator param, deque segment buffer, aggregation in _loop
- `src/acoustic/pipeline.py` - Changed _cnn_interval from 0.5 to 0.25 for 50% overlap
- `src/acoustic/main.py` - Classifier factory with ResearchCNN loading, WeightedAggregator wiring
- `tests/unit/test_worker.py` - TestSegmentBuffer class, extended TestCNNWorkerConstructor
- `tests/integration/test_cnn_pipeline.py` - TestFactoryWiring, TestPipelineOverlap, fixed positional args

## Decisions Made
- Silence-gated segments do not append to deque to avoid contaminating the rolling aggregation window
- Classifier factory validates model output shape with dummy forward pass before accepting
- Fixed pre-existing integration test bug where mock_classifier was passed as preprocessor positionally

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed integration test positional arg misuse**
- **Found during:** Task 2 (integration tests)
- **Issue:** `test_push_and_get_latest` passed `mock_classifier` as first positional arg (preprocessor), which worked by accident with old signature. With new keyword-arg constructor, test needed preprocessor to produce results.
- **Fix:** Added `mock_preprocessor` fixture and updated test to use keyword args
- **Files modified:** tests/integration/test_cnn_pipeline.py
- **Verification:** All integration tests pass
- **Committed in:** 8e28568

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Bug fix necessary for test correctness. No scope creep.

## Issues Encountered
- Python sys.path in worktree resolved to main repo source; required PYTHONPATH=src override for test execution

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Live inference path complete: ResearchCNN model loads from config path, preprocesses, classifies, aggregates
- Model deployment: place trained .pt file at cnn_model_path to activate inference
- Training pipeline can now be built to produce models consumable by the factory

## Self-Check: PASSED

All 5 modified files verified present. All 3 commits (8041a62, 6a9b009, 8e28568) verified in git log.

---
*Phase: 07-research-cnn-and-inference-integration*
*Completed: 2026-04-01*
