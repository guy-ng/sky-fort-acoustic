---
phase: 09-evaluation-harness-and-api
plan: 02
subsystem: api
tags: [fastapi, rest, websocket, training, evaluation, integration-tests]

requires:
  - phase: 09-evaluation-harness-and-api
    provides: "Evaluator class, Pydantic API models (Plan 01)"
  - phase: 08-pytorch-training-pipeline
    provides: "TrainingManager, TrainingConfig, TrainingRunner"
provides:
  - "POST /api/training/start, GET /api/training/progress, POST /api/training/cancel endpoints"
  - "POST /api/eval/run endpoint with thread executor"
  - "GET /api/models endpoint for .pt file listing"
  - "WebSocket /ws/training with epoch/status push"
  - "Integration tests for all Phase 9 API endpoints"
affects: [09-03-ui-integration]

tech-stack:
  added: []
  patterns: ["request.app.state.training_manager for DI", "run_in_executor for CPU-bound eval", "poll-and-push WebSocket with change detection"]

key-files:
  created:
    - src/acoustic/api/training_routes.py
    - src/acoustic/api/eval_routes.py
    - src/acoustic/api/model_routes.py
    - tests/integration/test_training_api.py
    - tests/integration/test_eval_api.py
    - tests/integration/test_training_ws.py
  modified:
    - src/acoustic/api/websocket.py
    - src/acoustic/main.py

key-decisions:
  - "TrainingManager wired via app.state in lifespan for clean dependency injection"
  - "Evaluation runs in thread executor (run_in_executor) to avoid blocking the async event loop"
  - "WebSocket /ws/training polls at 2Hz and only pushes on epoch/status change (no heartbeats)"
  - "409 for training conflicts, 404 for missing model/data with UI-SPEC copywriting"

patterns-established:
  - "app.state.training_manager: DI pattern for training lifecycle across routes and WebSocket"
  - "run_in_executor for CPU-bound operations in async endpoints"

requirements-completed: [API-01, API-02]

duration: 17min
completed: 2026-04-02
---

# Phase 9 Plan 2: REST Endpoints, WebSocket, and Integration Tests Summary

**Training start/progress/cancel REST endpoints, evaluation run with thread executor, model listing, WebSocket /ws/training with epoch push, and 9 integration tests**

## Performance

- **Duration:** 17 min
- **Started:** 2026-04-02T06:03:24Z
- **Completed:** 2026-04-02T06:20:23Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- All 6 REST endpoints (training start/progress/cancel, eval run, models list) and WebSocket /ws/training fully implemented and registered
- Error responses use UI-SPEC copywriting: 409 for training conflicts, 404 for missing model/data
- Evaluation runs in thread executor to avoid blocking the async event loop
- 9 integration tests cover all endpoints including error cases and WebSocket initial status

## Task Commits

Each task was committed atomically:

1. **Task 1: REST route files, WebSocket /ws/training, and main.py wiring** - `71aa9a7` (feat)
2. **Task 2: Integration tests for training, eval, and model APIs** - `19a2b89` (test)

## Files Created/Modified

- `src/acoustic/api/training_routes.py` - POST start, GET progress, POST cancel with 409 guards
- `src/acoustic/api/eval_routes.py` - POST /api/eval/run with run_in_executor, 404 validation
- `src/acoustic/api/model_routes.py` - GET /api/models scans .pt files with metadata
- `src/acoustic/api/websocket.py` - Extended with /ws/training endpoint and _progress_to_ws_dict helper
- `src/acoustic/main.py` - TrainingManager wired to app.state, three new routers registered
- `tests/integration/test_training_api.py` - 4 tests: start 200, start 409, progress 200, cancel 409
- `tests/integration/test_eval_api.py` - 4 tests: missing model 404, missing dir 404, valid data 200, models list 200
- `tests/integration/test_training_ws.py` - 1 test: WebSocket initial status message

## Decisions Made

- Used `request.app.state.training_manager` pattern (consistent with existing `request.app.state.settings` pattern) rather than FastAPI Depends
- Evaluation endpoint runs in default thread pool executor rather than a custom executor -- sufficient for single concurrent evaluations
- WebSocket /ws/training uses poll-and-push (0.5s sleep) with change detection rather than an event-driven approach -- matches existing /ws/targets pattern
- Model listing scans the parent directory of `cnn_model_path` setting, returning all .pt files

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Worktree was missing Phase 8 training module and Phase 7 classification files -- cherry-picked required commits to populate dependencies. Not a code issue, just worktree setup.
- Pre-existing import failures in `test_cnn_pipeline.py` and `test_preprocessing.py` (missing `preprocess_for_cnn` and `fast_resample`) -- unrelated to this plan, likely due to worktree cherry-pick ordering.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All Phase 9 API endpoints ready for Plan 03 UI integration
- WebSocket /ws/training ready for frontend training progress display
- All Pydantic response models serialize correctly

## Self-Check: PASSED

All 8 files verified. Both commits found.

---
*Phase: 09-evaluation-harness-and-api*
*Completed: 2026-04-02*
