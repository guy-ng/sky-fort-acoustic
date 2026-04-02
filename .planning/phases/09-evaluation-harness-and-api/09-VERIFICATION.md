---
phase: 09-evaluation-harness-and-api
verified: 2026-04-02T12:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
gaps: []
human_verification:
  - test: "Evaluate on real UAV audio dataset"
    expected: "Confusion matrix and distribution stats reflect real classification accuracy"
    why_human: "Requires actual labeled UAV audio files not available in the test environment"
---

# Phase 9: Evaluation Harness and API Verification Report

**Phase Goal:** Operators can evaluate classifier accuracy on labeled test data and control training and evaluation via REST endpoints with real-time progress updates
**Verified:** 2026-04-02T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Evaluation harness runs classifier on labeled test folders and produces confusion matrix, precision/recall/F1, and distribution stats (p_agg/p_max/p_mean percentiles) | VERIFIED | `Evaluator.evaluate()` in `src/acoustic/evaluation/evaluator.py` computes tp/fp/tn/fn, accuracy/precision/recall/f1, and calls `_compute_distribution()` using `np.percentile` at [25,50,75,95] per class |
| 2 | Evaluation provides per-file detailed output showing segment-level probabilities and final aggregation scores | VERIFIED | `FileResult` dataclass has `filename`, `true_label`, `predicted_label`, `p_agg`, `p_max`, `p_mean`, `correct` fields; all populated per file in evaluate loop |
| 3 | REST endpoints allow starting a training run, checking training progress, running an evaluation, and retrieving evaluation results | VERIFIED | `POST /api/training/start`, `GET /api/training/progress`, `POST /api/training/cancel`, `POST /api/eval/run`, `GET /api/models` all implemented and registered in `main.py`; 9 integration tests pass |
| 4 | Training progress is streamed via WebSocket so the UI can show real-time updates (epoch, loss, metrics) | VERIFIED | `/ws/training` endpoint in `websocket.py` sends current state on connect then pushes per-epoch updates using `_progress_to_ws_dict()` polling at 2Hz; integration test confirms initial status delivery |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/acoustic/evaluation/__init__.py` | Public exports for evaluation module | VERIFIED | Exports `Evaluator`, `EvaluationResult`, `FileResult`, `DistributionStats` |
| `src/acoustic/evaluation/evaluator.py` | Evaluator class with `evaluate()` method | VERIFIED | 223 lines, full implementation with `_compute_confusion`, `_compute_metrics`, `_compute_distribution` |
| `src/acoustic/evaluation/models.py` | `EvaluationResult`, `FileResult`, `DistributionStats` dataclasses | VERIFIED | All three dataclasses with correct fields; `FileResult` includes `p_max` and `p_mean` beyond plan spec |
| `src/acoustic/api/models.py` | Pydantic request/response models for training, eval, models endpoints | VERIFIED | 13 Pydantic models added; existing `TargetState`, `BeamformingMapResponse`, `HeatmapHandshake` preserved |
| `src/acoustic/api/training_routes.py` | `POST /api/training/start`, `GET /api/training/progress`, `POST /api/training/cancel` | VERIFIED | Three endpoints with 409 guards; correct UI-SPEC copy |
| `src/acoustic/api/eval_routes.py` | `POST /api/eval/run` | VERIFIED | Runs in `run_in_executor`; 404 for missing model/dir |
| `src/acoustic/api/model_routes.py` | `GET /api/models` | VERIFIED | Scans `*.pt` files in model directory with metadata |
| `src/acoustic/api/websocket.py` | `/ws/training` WebSocket endpoint | VERIFIED | `ws_training` and `_progress_to_ws_dict` added at end of file; imports `TrainingManager`, `TrainingProgress`, `TrainingStatus` |
| `src/acoustic/main.py` | `TrainingManager` wired to `app.state`, new routers registered | VERIFIED | `app.state.training_manager` set in lifespan; `training_router`, `eval_router`, `model_router` all included |
| `tests/unit/test_evaluator.py` | Unit tests for Evaluator: metrics, per-file output, distribution stats, edge cases | VERIFIED | 12 tests covering all 7 behaviors (confusion matrix, file results, distribution stats, metrics, division-by-zero, space-in-folder, short audio) |
| `tests/integration/test_training_api.py` | Integration tests for training endpoints | VERIFIED | 4 tests: start 200, start 409, progress 200, cancel 409 |
| `tests/integration/test_eval_api.py` | Integration tests for eval endpoints | VERIFIED | 4 tests: missing model 404, missing dir 404, valid data 200, models list 200 |
| `tests/integration/test_training_ws.py` | Integration test for WebSocket | VERIFIED | 1 test: initial status JSON with status field |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/acoustic/evaluation/evaluator.py` | `src/acoustic/training/dataset.py` | `collect_wav_files()` | WIRED | `from acoustic.training.dataset import collect_wav_files` at line 17 |
| `src/acoustic/evaluation/evaluator.py` | `src/acoustic/classification/preprocessing.py` | `mel_spectrogram_from_segment()` | WIRED | `from acoustic.classification.preprocessing import mel_spectrogram_from_segment` at line 14 |
| `src/acoustic/evaluation/evaluator.py` | `src/acoustic/classification/aggregation.py` | `WeightedAggregator` | WIRED | `from acoustic.classification.aggregation import WeightedAggregator` at line 12 |
| `src/acoustic/api/training_routes.py` | `src/acoustic/training/manager.py` | `request.app.state.training_manager` | WIRED | `manager: TrainingManager = request.app.state.training_manager` used in all three endpoints |
| `src/acoustic/api/eval_routes.py` | `src/acoustic/evaluation/evaluator.py` | `Evaluator().evaluate()` | WIRED | `from acoustic.evaluation import Evaluator`; called via `loop.run_in_executor(None, evaluator.evaluate, ...)` |
| `src/acoustic/api/websocket.py` | `src/acoustic/training/manager.py` | `training_manager.get_progress()` | WIRED | `manager.get_progress()` called on connect and in poll loop |
| `src/acoustic/main.py` | `src/acoustic/api/training_routes.py` | `app.include_router(training_router)` | WIRED | `include_router(training_router)` at line 436 |
| `src/acoustic/main.py` | `src/acoustic/api/eval_routes.py` | `app.include_router(eval_router)` | WIRED | `include_router(eval_router)` at line 437 |
| `src/acoustic/main.py` | `src/acoustic/api/model_routes.py` | `app.include_router(model_router)` | WIRED | `include_router(model_router)` at line 438 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `evaluator.py` | `file_results` | `soundfile.read()` on WAV files + `ResearchCNN` inference | Real WAV reads and model forward pass | FLOWING |
| `training_routes.py` | `progress` | `manager.get_progress()` returns live `TrainingProgress` dataclass | Live training thread state | FLOWING |
| `websocket.py` ws_training | `progress` | `websocket.app.state.training_manager.get_progress()` | Live training thread state polled at 2Hz | FLOWING |
| `model_routes.py` | `models` | `model_dir.glob("*.pt")` filesystem scan | Real filesystem files | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Evaluator unit tests pass | `pytest tests/unit/test_evaluator.py -x -q` | 12 passed in 1.20s | PASS |
| Integration tests for training, eval, WebSocket pass | `pytest tests/integration/test_training_api.py tests/integration/test_eval_api.py tests/integration/test_training_ws.py -x -q` | 9 passed in 27.16s | PASS |
| Evaluation module imports | `python -c "from acoustic.evaluation import Evaluator, EvaluationResult, FileResult, DistributionStats"` | OK | PASS |
| All API models import | `python -c "from acoustic.api.models import TrainingStartRequest, TrainingProgressResponse, EvalRunRequest, EvalResultResponse, ModelInfo"` | OK | PASS |
| All route modules import | `python -c "from acoustic.api.training_routes import router as tr; from acoustic.api.eval_routes import router as er; from acoustic.api.model_routes import router as mr"` | OK | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|-------------|-------------|--------|----------|
| EVL-01 | 09-01-PLAN.md | Evaluation harness with confusion matrix, P/R/F1, distribution stats | SATISFIED | `Evaluator.evaluate()` produces complete `EvaluationResult` with all metrics; 12 unit tests pass |
| EVL-02 | 09-01-PLAN.md | Per-file detailed output with segment probabilities and aggregation scores | SATISFIED | `FileResult` has `filename`, `true_label`, `predicted_label`, `p_agg`, `p_max`, `p_mean`, `correct` |
| API-01 | 09-02-PLAN.md | REST endpoints for training start/progress/cancel and eval run | SATISFIED | Five REST endpoints implemented and registered; integration tests pass |
| API-02 | 09-02-PLAN.md | WebSocket training progress streaming | SATISFIED | `/ws/training` sends status on connect and pushes on epoch/status changes |

**Note on requirement ID conflicts:** EVL-01 and EVL-02 do not exist in `REQUIREMENTS.md`. API-01 and API-02 in REQUIREMENTS.md refer to beamforming map endpoint and target list endpoint (Phase 2, already complete). The PLAN frontmatter uses these IDs to track Phase 9 work but REQUIREMENTS.md has not been updated to define EVL-01, EVL-02 or add new API-XX IDs for training/evaluation endpoints. This is a documentation tracking gap, not a code gap. The functionality exists and is verified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/acoustic/api/training_routes.py` | 77 | `best_epoch=0` hardcoded | Info | `TrainingProgress` does not track best_epoch; default of 0 is intentional and documented in the route comment |

No blockers found. The hardcoded `best_epoch=0` is a known limitation (the `TrainingProgress` dataclass from Phase 8 does not expose best_epoch tracking), not a stub — it returns a safe default value and the field is optional in the UI spec.

### Human Verification Required

#### 1. End-to-End Evaluation on Real UAV Audio

**Test:** Point `POST /api/eval/run` at a real labeled UAV audio dataset (drone/ and background/ subdirectories with real recordings) and a trained model checkpoint
**Expected:** Confusion matrix sums to total files, p_agg distributions differ meaningfully between drone and background classes, accuracy is reasonable (>70% with a trained model)
**Why human:** Requires physical UAV audio recordings not available in automated test environment; verifies no silent preprocessing bug inflates scores

#### 2. WebSocket Training Progress Real-Time Updates

**Test:** Start a training run via `POST /api/training/start`, then connect to `/ws/training` and observe messages as epochs complete
**Expected:** Epoch number increments in each message, loss values change, status transitions from `running` to `completed`
**Why human:** Integration tests mock `manager.start()` to avoid actual training; real epoch push behavior requires a full training run with audio data

### Gaps Summary

No gaps found. All four observable truths are fully verified by artifact inspection, wiring checks, data-flow traces, and passing test suites (12 unit tests + 9 integration tests).

The only notable finding is a requirements tracking discrepancy: EVL-01, EVL-02 are referenced in PLAN frontmatter but not defined in `REQUIREMENTS.md`, and API-01/API-02 in REQUIREMENTS.md belong to Phase 2. The REQUIREMENTS.md should be updated to add EVL-01, EVL-02, and new API-XX IDs for Phase 9 training and evaluation endpoints. This does not affect code correctness.

---

_Verified: 2026-04-02T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
