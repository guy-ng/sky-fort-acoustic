# Phase 9: Evaluation Harness and API - Research

**Researched:** 2026-04-02
**Domain:** FastAPI REST/WebSocket endpoints, binary classification evaluation metrics, PyTorch inference
**Confidence:** HIGH

## Summary

Phase 9 adds three capabilities to the existing acoustic service: (1) an evaluation harness that runs ResearchCNN inference on labeled test folders and computes binary classification metrics, (2) REST endpoints for training control and evaluation triggering, and (3) a WebSocket endpoint for real-time training progress streaming. All building blocks exist in the codebase already -- this phase wires them together through the API layer.

The evaluation harness reuses `collect_wav_files()`, `mel_spectrogram_from_segment()`, `ResearchCNN`, and `WeightedAggregator` from existing modules. No new ML libraries are needed. The REST endpoints wrap the existing `TrainingManager` and a new evaluator module. The WebSocket follows the established pattern from `/ws/status` (send initial state, then push updates).

**Primary recommendation:** Structure evaluation as a standalone module (`src/acoustic/evaluation/`) with an `Evaluator` class that takes a model path and data directory, runs inference, and returns a results dataclass. Keep it decoupled from the training pipeline -- it only shares preprocessing and model loading code.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Binary classification evaluation only (drone vs not-drone). Matches current ResearchCNN sigmoid output and BCE loss.
- **D-02:** Per-file summary output: filename, true label, predicted label, aggregated p_agg score, correct/incorrect flag. No segment-level detail.
- **D-03:** Distribution stats: percentiles (25th, 50th, 75th, 95th) of p_agg, p_max, p_mean per class (drone/background).
- **D-04:** Evaluation results stored in-memory only (ephemeral). Not persisted to disk.
- **D-05:** Separate route prefixes: `/api/training/...` for training control, `/api/eval/...` for evaluation.
- **D-06:** Training start endpoint (POST `/api/training/start`) accepts optional hyperparameter overrides.
- **D-07:** Evaluation endpoint (POST `/api/eval/run`) accepts optional `data_dir` path parameter.
- **D-08:** Evaluation endpoint accepts optional `model_path` parameter.
- **D-09:** Add GET `/api/models` endpoint listing available .pt model files with metadata.
- **D-10:** Other training endpoints: GET `/api/training/progress`, POST `/api/training/cancel`.
- **D-11:** Dedicated WebSocket at `/ws/training` for training progress updates.
- **D-12:** Push frequency: one JSON message per epoch containing epoch number, train_loss, val_loss, val_acc, confusion matrix (tp/fp/tn/fn), status.
- **D-13:** On WebSocket connect, send current status. If completed/failed, include last results. No periodic heartbeats.
- **D-14:** Evaluation runs synchronously -- POST blocks and returns results directly.
- **D-15:** Evaluation is on-demand only. No auto-run after training.
- **D-16:** Default evaluation test data: `Acoustic-UAV-Identification-main-main/Recorded Audios/Real World Testing/` with `drone/` and `no drone/` subdirectories.
- **D-17:** Research recorded audio data should also be usable as training data.

### Claude's Discretion
- How to structure the evaluation module (e.g., `src/acoustic/evaluation/` or `src/acoustic/training/evaluator.py`)
- Pydantic request/response models for training and evaluation endpoints
- How `/api/models` scans for .pt files
- Whether training and evaluation routes live in new files or extend existing `routes.py`
- How the evaluation harness handles the `no drone` folder name (space in directory name) for label mapping

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVL-01 | Evaluation harness runs classifier on labeled test folders and produces confusion matrix, precision/recall/F1, and distribution stats | Evaluator class reuses collect_wav_files(), mel_spectrogram_from_segment(), ResearchCNN model loading, WeightedAggregator for p_agg. Metrics computed from numpy arrays. |
| EVL-02 | Evaluation provides per-file detailed output showing segment-level probabilities and final aggregation scores | Per D-02, simplified to per-file summary (filename, true_label, predicted_label, p_agg, correct flag). Segments aggregated internally. |
| API-01 | REST endpoints for starting training, checking progress, running evaluation, and retrieving results | TrainingManager already has start/cancel/get_progress methods. New routes wrap these + new Evaluator. |
| API-02 | Training progress streamed via WebSocket for real-time UI updates | New `/ws/training` WebSocket follows established `/ws/status` pattern: send initial state on connect, push per-epoch updates. |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.135 | REST + WebSocket endpoints | Already used, async-native, auto-generated OpenAPI docs |
| Pydantic | v2 (via FastAPI) | Request/response validation | Already used in `api/models.py` |
| PyTorch | >=2.11 | Model loading and inference | Already used for training and inference |
| NumPy | >=1.26 | Metric computation (percentiles, confusion matrix) | Already used throughout |

### No New Dependencies Required
This phase requires zero new pip packages. All evaluation metrics (precision, recall, F1, confusion matrix, percentiles) are trivially computed from Python stdlib + numpy. Using sklearn for this would be overkill for binary classification with known formulas.

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py       # Evaluator class: run inference on test data, compute metrics
│   └── models.py           # EvaluationResult, FileResult, DistributionStats dataclasses
├── api/
│   ├── routes.py           # Existing routes (keep as-is)
│   ├── training_routes.py  # NEW: /api/training/* endpoints
│   ├── eval_routes.py      # NEW: /api/eval/* endpoints  
│   ├── model_routes.py     # NEW: /api/models endpoint
│   ├── websocket.py        # Existing WebSocket endpoints (extend with /ws/training)
│   └── models.py           # Extend with training/eval Pydantic models
└── training/
    └── (unchanged)
```

### Pattern 1: Evaluation Module
**What:** Standalone `Evaluator` class that loads a model, runs inference on a directory of labeled WAV files, and returns a structured result.
**When to use:** Whenever evaluation is triggered (API or future CLI).
**Example:**
```python
@dataclass
class FileResult:
    filename: str
    true_label: str          # "drone" or "no_drone"
    predicted_label: str     # "drone" or "no_drone"
    p_agg: float             # aggregated probability
    correct: bool

@dataclass
class DistributionStats:
    p25: float
    p50: float
    p75: float
    p95: float

@dataclass
class EvaluationResult:
    # Confusion matrix
    tp: int
    fp: int
    tn: int
    fn: int
    # Derived metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    # Per-class distribution stats
    drone_p_agg: DistributionStats
    drone_p_max: DistributionStats
    drone_p_mean: DistributionStats
    background_p_agg: DistributionStats
    background_p_max: DistributionStats
    background_p_mean: DistributionStats
    # Per-file detail
    files: list[FileResult]
    total_files: int
    total_correct: int
```

### Pattern 2: Route Separation
**What:** Separate `APIRouter` files per domain (training, eval, models) rather than stuffing everything into routes.py.
**When to use:** When adding multiple related endpoints that don't share state with existing routes.
**Example:**
```python
# training_routes.py
router = APIRouter(prefix="/api/training")

@router.post("/start")
async def start_training(request: Request, body: TrainingStartRequest) -> TrainingStartResponse:
    manager: TrainingManager = request.app.state.training_manager
    ...

# main.py -- register routers
app.include_router(training_router)
app.include_router(eval_router)
app.include_router(model_router)
```

### Pattern 3: WebSocket Training Progress (follows /ws/status pattern)
**What:** On connect, send current status JSON. Then poll `TrainingManager.get_progress()` and push when epoch changes.
**When to use:** `/ws/training` endpoint.
**Example:**
```python
@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    await websocket.accept()
    manager: TrainingManager = websocket.app.state.training_manager
    
    # Send current state immediately (D-13)
    progress = manager.get_progress()
    await websocket.send_json(_progress_to_dict(progress))
    
    last_epoch = progress.epoch
    last_status = progress.status
    try:
        while True:
            progress = manager.get_progress()
            if progress.epoch != last_epoch or progress.status != last_status:
                await websocket.send_json(_progress_to_dict(progress))
                last_epoch = progress.epoch
                last_status = progress.status
            await asyncio.sleep(0.5)  # Poll at 2 Hz -- epochs take seconds
    except (WebSocketDisconnect, RuntimeError):
        pass
```

### Pattern 4: Synchronous Evaluation via run_in_executor
**What:** Evaluation is CPU-bound (model inference on hundreds of files). Run it in a thread pool executor so the FastAPI event loop is not blocked.
**When to use:** POST `/api/eval/run` endpoint.
**Example:**
```python
@router.post("/run")
async def run_evaluation(request: Request, body: EvalRunRequest) -> EvalResultResponse:
    evaluator = Evaluator(mel_config=MelConfig())
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # default thread pool
        evaluator.evaluate,
        body.model_path or settings.cnn_model_path,
        body.data_dir or DEFAULT_TEST_DIR,
    )
    return EvalResultResponse.from_result(result)
```

### Pattern 5: Label Mapping for Directory Names with Spaces
**What:** The test data has a `no drone` folder (with space). Map directory names to normalized labels.
**When to use:** Evaluation label map configuration.
**Example:**
```python
# Evaluation-specific label map (handles spaces in folder names)
EVAL_LABEL_MAP: dict[str, int] = {
    "drone": 1,
    "no drone": 0,       # Real World Testing directory name
    "background": 0,     # Training data directory name
    "other": 0,
}
```
This works because `collect_wav_files()` iterates `label_map.items()` and joins with `root / label_name` -- Path handles spaces natively.

### Anti-Patterns to Avoid
- **Importing sklearn for binary metrics:** Precision/recall/F1 from a 2x2 confusion matrix is 4 lines of arithmetic. Don't add a 50MB dependency.
- **Persisting evaluation results to disk:** D-04 says in-memory only. Don't write JSON files.
- **Running evaluation in the main thread:** Even though D-14 says synchronous response, use `run_in_executor` to avoid blocking the event loop during inference.
- **Sharing a single model instance between live detection and evaluation:** D-08 says evaluation can use any model path. Always create a fresh `ResearchCNN` instance for evaluation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Percentile computation | Manual sorting + index math | `numpy.percentile()` | Handles edge cases (interpolation, empty arrays) |
| WAV file scanning | Custom glob/walk | `collect_wav_files()` from `dataset.py` | Already handles label mapping, skips non-WAV, tested |
| Mel spectrogram | New preprocessing | `mel_spectrogram_from_segment()` from `preprocessing.py` | Must match training preprocessing exactly |
| Probability aggregation | New aggregation logic | `WeightedAggregator` from `aggregation.py` | Must match live inference aggregation |

**Key insight:** Evaluation MUST use the exact same preprocessing and aggregation as live inference. Any divergence means evaluation metrics don't predict real-world performance. Reuse, don't rebuild.

## Common Pitfalls

### Pitfall 1: Preprocessing Mismatch Between Training and Evaluation
**What goes wrong:** Evaluation uses different mel-spectrogram parameters or segment extraction logic than training, producing misleading metrics.
**Why it happens:** Copy-pasting preprocessing code instead of importing the shared function.
**How to avoid:** Import `mel_spectrogram_from_segment()` and `MelConfig` from the same modules used by training. Never duplicate preprocessing logic.
**Warning signs:** Evaluation accuracy is dramatically different from training validation accuracy.

### Pitfall 2: Blocking the Event Loop with Synchronous Inference
**What goes wrong:** POST `/api/eval/run` blocks all WebSocket streams and other endpoints during evaluation (could be minutes for 500+ files).
**Why it happens:** Running CPU-bound PyTorch inference in an async handler without `run_in_executor`.
**How to avoid:** Use `asyncio.get_event_loop().run_in_executor(None, evaluator.evaluate, ...)`.
**Warning signs:** WebSocket disconnections during evaluation, health check timeouts.

### Pitfall 3: Not Handling Short Audio Files
**What goes wrong:** Some test WAV files may be shorter than the 0.5s segment window (8000 samples at 16kHz). Evaluation crashes or produces garbage features.
**Why it happens:** Assuming all test files meet minimum length.
**How to avoid:** Use the same padding logic from `mel_spectrogram_from_segment()` which already zero-pads short segments. For multi-segment evaluation: if file is shorter than one segment, produce one zero-padded segment.
**Warning signs:** IndexError or empty probability lists.

### Pitfall 4: Division by Zero in Metrics
**What goes wrong:** Precision is TP/(TP+FP) -- if no positive predictions, division by zero. Same for recall.
**Why it happens:** Edge case where model predicts all negative or all positive.
**How to avoid:** Guard with `max(denominator, 1)` or return 0.0 when denominator is 0.
**Warning signs:** NaN or Infinity in API responses.

### Pitfall 5: TrainingManager Not Wired to app.state
**What goes wrong:** `request.app.state.training_manager` raises AttributeError.
**Why it happens:** Phase 8 built TrainingManager but the lifespan function in `main.py` does not yet attach it to `app.state`.
**How to avoid:** Add `app.state.training_manager = TrainingManager(config, mel_config)` in the lifespan function.
**Warning signs:** 500 errors on all training endpoints.

### Pitfall 6: WebSocket Poll Rate vs Epoch Duration
**What goes wrong:** Polling too slowly misses epoch updates; polling too fast wastes CPU.
**Why it happens:** Training epochs can range from sub-second (tiny data) to minutes (large dataset).
**How to avoid:** Poll at 0.5s (2 Hz). Compare epoch number and status -- only send when changed.
**Warning signs:** Missed epochs in UI, or excessive CPU from WebSocket handler.

## Code Examples

### Evaluation Core Logic
```python
# Source: Pattern derived from existing codebase components
def evaluate(self, model_path: str, data_dir: str) -> EvaluationResult:
    # Load model
    model = ResearchCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Collect test files
    paths, labels = collect_wav_files(data_dir, self._label_map)
    
    file_results = []
    aggregator = WeightedAggregator()
    
    with torch.no_grad():
        for path, true_label_int in zip(paths, labels):
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Extract segments (non-overlapping 0.5s windows)
            segment_probs = []
            n = self._mel_config.segment_samples
            num_segments = max(1, len(audio) // n)
            for i in range(num_segments):
                start = i * n
                segment = audio[start:start + n]
                if len(segment) < n:
                    segment = np.pad(segment, (0, n - len(segment)))
                features = mel_spectrogram_from_segment(segment, self._mel_config)
                prob = model(features).item()
                segment_probs.append(prob)
            
            p_agg = aggregator.aggregate(segment_probs)
            predicted_int = 1 if p_agg >= 0.5 else 0
            
            true_label_str = "drone" if true_label_int == 1 else "no_drone"
            pred_label_str = "drone" if predicted_int == 1 else "no_drone"
            
            file_results.append(FileResult(
                filename=path.name,
                true_label=true_label_str,
                predicted_label=pred_label_str,
                p_agg=p_agg,
                correct=(predicted_int == true_label_int),
            ))
    
    # Compute metrics from file_results
    return self._compute_metrics(file_results)
```

### Binary Metrics Computation
```python
def _compute_metrics(self, files: list[FileResult]) -> EvaluationResult:
    tp = sum(1 for f in files if f.true_label == "drone" and f.predicted_label == "drone")
    fp = sum(1 for f in files if f.true_label == "no_drone" and f.predicted_label == "drone")
    tn = sum(1 for f in files if f.true_label == "no_drone" and f.predicted_label == "no_drone")
    fn = sum(1 for f in files if f.true_label == "drone" and f.predicted_label == "no_drone")
    
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Distribution stats per class
    drone_files = [f for f in files if f.true_label == "drone"]
    bg_files = [f for f in files if f.true_label == "no_drone"]
    # ... compute percentiles with np.percentile()
```

### Training Start Request/Response Models
```python
class TrainingStartRequest(BaseModel):
    learning_rate: float | None = None
    batch_size: int | None = None
    max_epochs: int | None = None
    patience: int | None = None
    augmentation_enabled: bool | None = None
    data_root: str | None = None

class TrainingProgressResponse(BaseModel):
    status: str  # idle/running/completed/cancelled/failed
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_acc: float
    best_val_loss: float
    error: str | None
    tp: int
    fp: int
    tn: int
    fn: int
```

### Model Listing Endpoint
```python
@router.get("/models")
async def list_models(request: Request) -> list[ModelInfo]:
    settings: AcousticSettings = request.app.state.settings
    model_dir = Path(settings.cnn_model_path).parent
    models = []
    if model_dir.is_dir():
        for pt_file in sorted(model_dir.glob("*.pt")):
            stat = pt_file.stat()
            models.append(ModelInfo(
                filename=pt_file.name,
                path=str(pt_file),
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    return models
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 with pytest-asyncio |
| Config file | pyproject.toml (`asyncio_mode = "auto"`) |
| Quick run command | `python -m pytest tests/unit/ -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVL-01 | Evaluator produces confusion matrix, precision/recall/F1, distribution stats from labeled test folders | unit | `python -m pytest tests/unit/test_evaluator.py -x` | Wave 0 |
| EVL-02 | Per-file output includes filename, true label, predicted label, p_agg, correct flag | unit | `python -m pytest tests/unit/test_evaluator.py::test_per_file_output -x` | Wave 0 |
| API-01 | Training start/progress/cancel and eval run/results endpoints return correct status codes and schemas | integration | `python -m pytest tests/integration/test_training_api.py tests/integration/test_eval_api.py -x` | Wave 0 |
| API-02 | WebSocket /ws/training sends initial status on connect and epoch updates during training | integration | `python -m pytest tests/integration/test_training_ws.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_evaluator.py` -- covers EVL-01, EVL-02
- [ ] `tests/integration/test_training_api.py` -- covers API-01 (training endpoints)
- [ ] `tests/integration/test_eval_api.py` -- covers API-01 (eval endpoints)
- [ ] `tests/integration/test_training_ws.py` -- covers API-02

## Project Constraints (from CLAUDE.md)

- **Runtime:** Python >=3.11, FastAPI >=0.135
- **Testing:** pytest >=8.0 with pytest-asyncio, httpx for HTTP client
- **Linting:** Ruff >=0.9 (replaces flake8+black+isort)
- **Type checking:** mypy >=1.14
- **Config pattern:** Pydantic `BaseSettings` with `ACOUSTIC_` env prefix
- **Frontend:** Not in scope for Phase 9 (backend API only)
- **GSD Workflow:** All edits through GSD commands

## Sources

### Primary (HIGH confidence)
- `src/acoustic/training/manager.py` -- TrainingManager API (start/cancel/get_progress)
- `src/acoustic/training/trainer.py` -- TrainingRunner progress callback format
- `src/acoustic/training/dataset.py` -- collect_wav_files() signature and behavior
- `src/acoustic/training/config.py` -- TrainingConfig fields (hyperparameter override targets)
- `src/acoustic/classification/preprocessing.py` -- mel_spectrogram_from_segment() and ResearchPreprocessor
- `src/acoustic/classification/research_cnn.py` -- ResearchCNN architecture and ResearchClassifier wrapper
- `src/acoustic/classification/aggregation.py` -- WeightedAggregator.aggregate()
- `src/acoustic/api/routes.py` -- Existing route pattern (APIRouter prefix, request.app.state)
- `src/acoustic/api/websocket.py` -- Existing WebSocket pattern (accept, send initial, poll loop)
- `src/acoustic/api/models.py` -- Existing Pydantic response models
- `src/acoustic/config.py` -- AcousticSettings.cnn_model_path
- `src/acoustic/main.py` -- Lifespan function, app.state wiring
- `tests/integration/conftest.py` -- running_app fixture pattern
- `tests/unit/test_training_manager.py` -- Training test patterns (synthetic WAV generation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project, no new deps needed
- Architecture: HIGH - patterns established by existing codebase, extending not inventing
- Pitfalls: HIGH - based on direct code analysis of existing components

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable -- all patterns established, no external API changes expected)
