# Phase 9: Evaluation Harness and API - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build an evaluation harness that runs the classifier on labeled test folders and produces metrics (confusion matrix, precision/recall/F1, distribution stats), plus REST endpoints for training control (start/cancel/progress) and evaluation (trigger/results), a model listing endpoint, and a WebSocket for real-time training progress streaming. This phase does NOT include recording UI (Phase 10), ensemble support (Phase 11), or frontend UI for these endpoints (future phase).

</domain>

<decisions>
## Implementation Decisions

### Evaluation Output
- **D-01:** Binary classification evaluation only (drone vs not-drone). Matches current ResearchCNN sigmoid output and BCE loss. Multi-class is a future concern.
- **D-02:** Per-file summary output: filename, true label, predicted label, aggregated p_agg score, correct/incorrect flag. No segment-level detail.
- **D-03:** Distribution stats: percentiles (25th, 50th, 75th, 95th) of p_agg, p_max, p_mean per class (drone/background). Helps tune detection thresholds.
- **D-04:** Evaluation results stored in-memory only (ephemeral, like TrainingProgress from Phase 8 D-13). Not persisted to disk. Run again to re-evaluate.

### API Endpoint Design
- **D-05:** Separate route prefixes: `/api/training/...` for training control, `/api/eval/...` for evaluation. Alongside existing `/api/map` and `/api/targets`.
- **D-06:** Training start endpoint (POST `/api/training/start`) accepts optional hyperparameter overrides in request body (lr, batch_size, epochs, patience, augmentation toggle). Defaults from TrainingConfig when omitted.
- **D-07:** Evaluation endpoint (POST `/api/eval/run`) accepts optional `data_dir` path parameter. Defaults to research test data directory.
- **D-08:** Evaluation endpoint accepts optional `model_path` parameter. Defaults to `cnn_model_path` from AcousticSettings. Allows evaluating any .pt checkpoint without swapping the live model.
- **D-09:** Add GET `/api/models` endpoint listing available .pt model files with metadata (file size, modification date). Useful for future UI model selection.
- **D-10:** Other training endpoints: GET `/api/training/progress`, POST `/api/training/cancel`.

### Training Progress Streaming
- **D-11:** Dedicated WebSocket at `/ws/training` for training progress updates. Consistent with existing `/ws/heatmap`, `/ws/targets`, `/ws/events`, `/ws/status` pattern.
- **D-12:** Push frequency: one JSON message per epoch containing epoch number, train_loss, val_loss, val_acc, confusion matrix (tp/fp/tn/fn), status.
- **D-13:** On WebSocket connect, send one JSON message with current status (idle/running/completed/failed). If completed/failed, include last results. Then go quiet until training starts or next epoch completes. No periodic heartbeats.

### Evaluation Execution Model
- **D-14:** Evaluation runs synchronously — POST `/api/eval/run` blocks and returns results directly in the response. Evaluation is inference-only (no gradients), expected to complete in seconds to low minutes.
- **D-15:** Evaluation is on-demand only. No auto-run after training completes. User explicitly triggers evaluation, choosing model and test set.

### Test Data
- **D-16:** Default evaluation test data: `Acoustic-UAV-Identification-main-main/Recorded Audios/Real World Testing/` with `drone/` (272 files) and `no drone/` (284 files) subdirectories. Label derived from folder name.
- **D-17:** Research recorded audio data (`Acoustic-UAV-Identification-main-main/Recorded Audios/`) should also be usable as training data, following the same `{root}/{label}/` directory convention established in Phase 8 D-04.

### Claude's Discretion
- How to structure the evaluation module (e.g., `src/acoustic/evaluation/` or `src/acoustic/training/evaluator.py`)
- Pydantic request/response models for training and evaluation endpoints
- How `/api/models` scans for .pt files (directory walk from configured model path, or configurable search path)
- Whether training and evaluation routes live in new files or extend existing `routes.py`
- How the evaluation harness handles the `no drone` folder name (space in directory name) for label mapping

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Research Test Data
- `Acoustic-UAV-Identification-main-main/Recorded Audios/Real World Testing/` -- Default evaluation test data. `drone/` subfolder (272 WAV files: drone+helicopter, drone+traffic scenarios), `no drone/` subfolder (284 WAV files: traffic backgrounds). Binary labels from folder names.
- `Acoustic-UAV-Identification-main-main/Recorded Audios/Unseen Drone Audio/` -- Additional drone recordings (DJI Mini 2, RED5 Eagle). Can be included in evaluation for unseen-drone testing.
- `Acoustic-UAV-Identification-main-main/Recorded Audios/README.txt` -- Attribution notes for recorded audio.

### Existing Service Code (Phase 9 touches/extends these)
- `src/acoustic/training/manager.py` -- `TrainingManager` and `TrainingProgress` dataclass. Phase 9 REST endpoints expose this. Manager has start/cancel/progress methods and single-run concurrency guard.
- `src/acoustic/training/trainer.py` -- `TrainingRunner` and `EarlyStopping`. Training loop that Phase 9 triggers via API.
- `src/acoustic/training/config.py` -- `TrainingConfig` with hyperparameters. API overrides merge with these defaults.
- `src/acoustic/training/dataset.py` -- `DroneAudioDataset`, `collect_wav_files()`, `build_weighted_sampler()`. Evaluation harness reuses `collect_wav_files()` for test data loading.
- `src/acoustic/api/routes.py` -- Existing REST routes (`/api/map`, `/api/targets`). Add training and eval routes alongside.
- `src/acoustic/api/websocket.py` -- Existing WebSocket endpoints. Add `/ws/training` here.
- `src/acoustic/api/models.py` -- Existing Pydantic response models. Add training/eval models here.
- `src/acoustic/classification/research_cnn.py` -- `ResearchCNN` model class and `ResearchClassifier`. Evaluation loads model from path using same pattern.
- `src/acoustic/classification/preprocessing.py` -- `ResearchPreprocessor`. Evaluation harness reuses for feature extraction.
- `src/acoustic/classification/protocols.py` -- Classifier, Preprocessor, Aggregator protocols. Evaluation uses these.
- `src/acoustic/classification/config.py` -- `MelConfig` dataclass. Evaluation preprocessing uses shared config.
- `src/acoustic/config.py` -- `AcousticSettings` with `cnn_model_path`. Default model path for evaluation.

### Prior Phase Context
- `.planning/phases/08-pytorch-training-pipeline/08-CONTEXT.md` -- Training pipeline decisions (D-13 progress state, D-14 cancellation, D-15 single concurrent run).
- `.planning/phases/07-research-cnn-and-inference-integration/07-CONTEXT.md` -- Aggregation strategy (D-03 weighted p_agg), model loading pattern (D-05 .pt only, D-06 factory).

### Requirements
- `.planning/REQUIREMENTS.md` -- EVL-01, EVL-02 (evaluation), API-01, API-02 (REST endpoints), TRN-04 (training progress in UI).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrainingManager` in `manager.py` -- Already has start/cancel/get_progress methods and `TrainingProgress` dataclass with confusion matrix fields. REST endpoints directly wrap these.
- `collect_wav_files()` in `dataset.py` -- Scans directory for WAV files by label folder. Reusable for evaluation test data loading.
- `ResearchPreprocessor` in `preprocessing.py` -- torchaudio mel-spectrogram pipeline. Evaluation uses same preprocessing.
- `ResearchClassifier` in `research_cnn.py` -- Loads .pt state_dict, implements Classifier protocol. Evaluation instantiates one per model path.
- `WeightedAggregator` in protocols/aggregation -- Computes p_agg from segment probabilities. Evaluation uses same aggregation.

### Established Patterns
- Config via Pydantic `BaseSettings` with `ACOUSTIC_` env prefix
- FastAPI `APIRouter` with prefix for route grouping
- WebSocket pattern: accept, send initial state, then poll/push loop (see `/ws/heatmap`, `/ws/status`)
- Daemon threads for background processing with thread-safe state objects
- `request.app.state` for accessing shared service state from route handlers

### Integration Points
- Training endpoints wrap `TrainingManager` methods already attached to `app.state`
- `/ws/training` polls `TrainingManager.get_progress()` like `/ws/status` polls device monitor
- Evaluation harness creates a temporary `ResearchClassifier` instance (does not affect live classifier)
- `/api/models` scans directory around `cnn_model_path` for .pt files

</code_context>

<specifics>
## Specific Ideas

- Research test data at `Acoustic-UAV-Identification-main-main/Recorded Audios/Real World Testing/` provides a ready-made labeled test set (272 drone + 284 no-drone files) for immediate evaluation after training
- The same research recordings should be usable as training data too, following Phase 8's `{root}/{label}/` directory convention
- Folder `no drone` has a space in the name -- evaluation harness must handle this gracefully for label mapping

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 09-evaluation-harness-and-api*
*Context gathered: 2026-04-02*
