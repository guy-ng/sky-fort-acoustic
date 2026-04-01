# Architecture Research

**Domain:** Acoustic UAV classification pipeline migration
**Researched:** 2026-04-01
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
                              EXISTING (keep)                          NEW (add)
                    +----------------------------+        +----------------------------+
                    |  AudioCapture / Simulator   |        |    DatasetCollector (API)   |
                    |  -> AudioRingBuffer          |        |  record + label + metadata  |
                    +-------------+--------------+        +-------------+--------------+
                                  |                                      |
                                  v                                      v
                    +----------------------------+        +----------------------------+
                    |   BeamformingPipeline       |        |     data/recordings/        |
                    |   process_chunk()           |        |     {label}/{bin}/*.wav      |
                    |   _process_cnn() <-- MODIFY |        |     + .json metadata         |
                    +-------------+--------------+        +----------------------------+
                                  |                                      |
                        +---------+---------+                            v
                        |                   |              +----------------------------+
                        v                   v              |   TrainingPipeline          |
              +----------------+  +------------------+    |   PyTorch CNN (3-layer)     |
              | CNNWorker      |  | StateMachine     |    |   -> .pt checkpoints        |
              | (background    |  | (3-state          |    |   -> ONNX export            |
              |  thread)       |  |  hysteresis)      |    +-------------+--------------+
              | MODIFY: swap   |  |                   |                  |
              | classifier     |  +------------------+                  v
              +-------+--------+                          +----------------------------+
                      |                                   |   EvaluationHarness         |
                      v                                   |   confusion matrix, dist    |
              +----------------+                          |   stats, per-file analysis  |
              | Classifier     |                          +----------------------------+
              | (Protocol)     |
              | REPLACE:       |
              | OnnxDrone.. -> |                          +----------------------------+
              | ResearchCNN +  |                          |   EnsembleClassifier        |
              | Ensemble       |                          |   soft/hard voting          |
              +-------+--------+                          |   weighted combination      |
                      |                                   +----------------------------+
                      v
              +----------------+
              | SegmentAggr.   |
              | NEW: p_max,    |
              | p_mean, p_agg  |
              +-------+--------+
                      |
                      v
              +----------------+
              | TargetTracker  |
              | (keep as-is)   |
              +----------------+
```

### Component Responsibilities

| Component | Status | Responsibility | Communicates With |
|-----------|--------|---------------|-------------------|
| `AudioRingBuffer` | KEEP | Ring buffer for 16-ch audio chunks | Pipeline reads, Capture writes |
| `BeamformingPipeline` | MODIFY | Orchestrates chunk processing + CNN dispatch | CNNWorker, StateMachine, Tracker |
| `CNNWorker` | MODIFY | Background inference thread, single-slot queue | Classifier (via protocol), Pipeline |
| `OnnxDroneClassifier` | REPLACE | ONNX EfficientNet-B0 inference | CNNWorker calls `.predict()` |
| `preprocessing.py` | REPLACE | Mel-spec for EfficientNet (224x224, 3-ch) | CNNWorker calls `preprocess_for_cnn()` |
| `DetectionStateMachine` | KEEP | 3-state hysteresis (NO_DRONE/CANDIDATE/CONFIRMED) | Pipeline feeds probabilities |
| `TargetTracker` | KEEP | Track targets by bearing + confidence | Pipeline feeds on CONFIRMED |
| `AcousticSettings` | EXTEND | Pydantic settings from env vars | All components read config |
| `routes.py` | EXTEND | REST API for map, targets | Pipeline, new training/eval endpoints |
| **ResearchClassifier** | NEW | PyTorch CNN inference (research arch) | CNNWorker via Classifier protocol |
| **ResearchPreprocessor** | NEW | Mel-spec matching research params (128x64, 1-ch) | CNNWorker, TrainingPipeline |
| **SegmentAggregator** | NEW | p_max, p_mean, p_agg from segment probs | CNNWorker (post-inference) |
| **EnsembleClassifier** | NEW | Multi-model soft/hard voting | CNNWorker (wraps multiple classifiers) |
| **TrainingPipeline** | NEW | PyTorch training loop, dataset loading | REST API triggers, filesystem |
| **EvaluationHarness** | NEW | Confusion matrix, distribution stats | REST API triggers, filesystem |
| **DatasetCollector** | NEW | Record + label audio via web UI | REST API, AudioCapture |

## Recommended Project Structure

```
src/acoustic/
  classification/
    __init__.py                    # KEEP
    inference.py                   # MODIFY: add Classifier protocol, keep OnnxDroneClassifier
    preprocessing.py               # MODIFY: rename current to efficientnet_preprocessing.py
    state_machine.py               # KEEP (no changes)
    worker.py                      # MODIFY: support new classifier + aggregation
    # --- NEW FILES ---
    protocol.py                    # Classifier protocol (interface)
    research_model.py              # PyTorch 3-layer CNN architecture definition
    research_classifier.py         # PyTorch/ONNX inference implementing Classifier protocol
    research_preprocessing.py      # Research mel-spec params (64 mels, 128 frames, 16kHz)
    aggregation.py                 # SegmentAggregator: p_max, p_mean, p_agg, weighted
    ensemble.py                    # EnsembleClassifier: soft/hard voting, weighted combo
  training/                        # NEW PACKAGE
    __init__.py
    dataset.py                     # PyTorch Dataset for WAV files (replaces tf.data)
    trainer.py                     # Training loop (replaces train_strong_cnn.py)
    export.py                      # PyTorch -> ONNX export
    config.py                      # Training hyperparameters (Pydantic model)
  evaluation/                      # NEW PACKAGE
    __init__.py
    harness.py                     # Evaluation runner (replaces eval_folder_with_strong_cnn.py)
    metrics.py                     # Confusion matrix, precision, recall, F1, dist stats
  collection/                      # NEW PACKAGE
    __init__.py
    recorder.py                    # Record labeled clips (replaces uma16_dataset_collector_gui.py)
    metadata.py                    # JSON metadata schema for recordings
  pipeline.py                      # MODIFY: wire new aggregation into _process_cnn
  config.py                        # EXTEND: add training/eval/collection settings
  api/
    routes.py                      # EXTEND: add /api/training/*, /api/eval/*, /api/collection/*
    models.py                      # EXTEND: add request/response models for new endpoints

models/                            # Model artifacts directory
  uav_melspec_cnn.onnx             # KEEP (legacy EfficientNet, for fallback)
  research_cnn_v1.pt               # NEW: PyTorch checkpoint
  research_cnn_v1.onnx             # NEW: Exported ONNX for production inference
  ensemble/                        # NEW: Multiple model checkpoints for voting
    model_1.onnx
    model_2.onnx
    ...

data/                              # Training data (gitignored)
  train/
    uav/*.wav
    background/*.wav
  test/
    uav/*.wav
    background/*.wav
  recordings/                      # NEW: UMA-16 field recordings
    drone/{distance_bin}/*.wav
    background/{distance_bin}/*.wav
```

## Architectural Patterns

### Pattern 1: Classifier Protocol (Strategy Pattern)

The current code hardcodes `OnnxDroneClassifier`. Replace with a protocol so `CNNWorker` is classifier-agnostic.

**What:** Define a `Classifier` protocol, make all classifiers implement it. `CNNWorker` depends on the protocol, not the concrete class.

**Why:** Enables swapping EfficientNet ONNX for research CNN, ONNX-exported research CNN, or ensemble -- without touching worker logic.

```python
# classification/protocol.py
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Classifier(Protocol):
    """Interface for any drone classifier."""
    def predict(self, preprocessed: np.ndarray) -> float:
        """Return drone probability in [0.0, 1.0]."""
        ...

@runtime_checkable
class Preprocessor(Protocol):
    """Interface for audio-to-model-input preprocessing."""
    def __call__(self, mono_audio: np.ndarray, fs_in: int) -> np.ndarray | None:
        """Return preprocessed tensor or None if silence."""
        ...
```

### Pattern 2: Segment Aggregation (Post-Inference)

The research pipeline splits audio into overlapping 0.5s segments and aggregates predictions. This happens AFTER per-segment inference, BEFORE the state machine.

**What:** `SegmentAggregator` accumulates per-segment probabilities and produces file-level scores.

**When:** Real-time inference. The CNNWorker currently processes one 2-second chunk. With the research approach, it processes multiple 0.5s overlapping segments from that chunk and aggregates.

```python
# classification/aggregation.py
import numpy as np

class SegmentAggregator:
    """Aggregate segment-level drone probabilities into a single score."""

    def __init__(self, w_max: float = 0.7, w_mean: float = 0.3):
        self._w_max = w_max
        self._w_mean = w_mean

    def aggregate(self, segment_probs: np.ndarray) -> dict:
        """Return p_max, p_mean, p_agg, and weighted combination."""
        p_max = float(np.max(segment_probs))
        p_mean = float(np.mean(segment_probs))
        # Probability at least one segment is drone
        p_agg = 1.0 - float(np.prod(1.0 - segment_probs))
        p_weighted = self._w_max * p_max + self._w_mean * p_mean
        return {
            "p_max": p_max,
            "p_mean": p_mean,
            "p_agg": p_agg,
            "p_weighted": p_weighted,
        }
```

### Pattern 3: Ensemble as Composite Classifier

The ensemble wraps multiple classifiers and implements the same `Classifier` protocol. Transparent to the worker.

**What:** `EnsembleClassifier` holds N classifiers, runs all, combines via soft/hard voting.

**When:** Production inference when multiple models are available.

```python
# classification/ensemble.py
class EnsembleClassifier:
    """Late fusion ensemble implementing Classifier protocol."""

    def __init__(
        self,
        classifiers: list[Classifier],
        weights: list[float] | None = None,
        mode: str = "soft",  # "soft" or "hard"
    ):
        self._classifiers = classifiers
        self._weights = weights or [1.0 / len(classifiers)] * len(classifiers)
        self._mode = mode

    def predict(self, preprocessed: np.ndarray) -> float:
        probs = [c.predict(preprocessed) for c in self._classifiers]
        if self._mode == "hard":
            votes = [1 if p >= 0.5 else 0 for p in probs]
            return float(np.average(votes, weights=self._weights))
        else:
            return float(np.average(probs, weights=self._weights))
```

### Pattern 4: Training as Background Task (not blocking the pipeline)

**What:** Training runs in a separate thread, never on the inference path. FastAPI endpoint triggers it, WebSocket streams progress.

**Why:** Training is CPU/GPU-bound and long-running. Must not interfere with real-time audio processing.

```python
# training/trainer.py -- simplified structure
class TrainingJob:
    """Manages a single training run in a background thread."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.status = "pending"
        self.progress = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._train, daemon=True)
        self._thread.start()

    def _train(self) -> None:
        self.status = "running"
        # ... PyTorch training loop ...
        # On completion: self.status = "complete"
        # Export to ONNX for production use
```

### Pattern 5: Preprocessing Normalization Mismatch Guard

**Critical:** The research pipeline uses `(S_db + 80) / 80` normalization (0-1 range), while the current service uses `(S_db - mean) / std` (zero-mean, unit-variance). These produce different distributions and are NOT interchangeable. The new preprocessor MUST match the training normalization exactly.

**What:** The `ResearchPreprocessor` must use the research normalization: `(S_db + 80.0) / 80.0` clipped to [0, 1]. The current `norm_spec()` with zero-mean/unit-variance is ONLY for the legacy EfficientNet path.

## Data Flow

### Training Flow

```
[WAV files on disk]
    |
    v
TrainingDataset (PyTorch Dataset)
    | - list WAV files from data/train/{uav,background}/
    | - random 0.5s segment per file (data augmentation)
    | - segment_to_melspec() with research params
    | - (S_db + 80) / 80 normalization
    | - pad_or_trim to (128, 64)
    | - expand dims -> (128, 64, 1)
    |
    v
DataLoader (batch=32, shuffle=True)
    |
    v
ResearchCNN (PyTorch nn.Module)
    | - Conv2d(1, 32) -> BN -> MaxPool
    | - Conv2d(32, 64) -> BN -> MaxPool
    | - Conv2d(64, 128) -> BN -> MaxPool
    | - GlobalAvgPool2d
    | - Linear(128, 128) -> ReLU -> Dropout(0.3)
    | - Linear(128, 1) -> Sigmoid
    |
    v
BCELoss + Adam(lr=1e-3)
    | - EarlyStopping(patience=8)
    | - ReduceLR(patience=3)
    |
    v
[Save .pt checkpoint] -> [Export to ONNX]
```

### Inference Flow (updated)

```
[Pipeline._process_cnn(chunk, peak)]
    |
    v
Accumulate mono audio in rolling buffer (KEEP existing logic)
    |
    v
When buffer >= segment_length:
    |
    +-- CHANGE: Split into overlapping 0.5s segments (was: single 2s chunk)
    |   hop = 0.25s (50% overlap), yielding ~7 segments from 2s buffer
    |
    v
For each segment:
    ResearchPreprocessor
    | - resample to 16kHz (KEEP, same target SR)
    | - silence gate (KEEP, same RMS check)
    | - melspectrogram (n_fft=1024, hop=256, 64 mels, power=2.0)
    | - (S_db + 80) / 80, clip to [0,1]   <-- CHANGE from zero-mean/std
    | - pad_or_trim to (128, 64)
    | - reshape to (1, 1, 128, 64)          <-- CHANGE from (1, 3, 224, 224)
    |
    v
Classifier.predict(preprocessed) -> float per segment
    | (ResearchClassifier or EnsembleClassifier via protocol)
    |
    v
SegmentAggregator.aggregate(segment_probs)
    | -> p_max, p_mean, p_agg, p_weighted
    |
    v
Feed p_weighted (or p_agg) into DetectionStateMachine.update()
    | (KEEP existing 3-state hysteresis, just different input value)
    |
    v
On CONFIRMED -> TargetTracker.update() (KEEP)
```

### Evaluation Flow

```
[REST: POST /api/eval/run]
    |
    v
EvaluationHarness
    | - Load model from models/ directory
    | - Iterate WAV files in test/{uav,background}/
    | - Per file: split into segments, predict, aggregate
    | - Compute confusion matrix, precision, recall, F1
    | - Compute distribution stats (p1, p5, median, p95, p99)
    | - Per-file detail for debugging
    |
    v
[REST: GET /api/eval/results/{run_id}]
    | -> JSON with metrics, confusion matrix, per-file predictions
    |
    v
[WebSocket: /ws/eval/progress]
    | -> Real-time progress updates during evaluation
```

## Integration Points

### Modified Components

#### 1. `classification/worker.py` (CNNWorker)

**Current:** Hardcoded `OnnxDroneClassifier` and `preprocess_for_cnn`.

**Change:** Accept `Classifier` and `Preprocessor` protocols. Add segment splitting and aggregation.

```python
class CNNWorker:
    def __init__(
        self,
        classifier: Classifier,        # Was: OnnxDroneClassifier
        preprocessor: Preprocessor,     # NEW: injectable preprocessing
        aggregator: SegmentAggregator,  # NEW: segment aggregation
        fs_in: int = 48000,
        segment_seconds: float = 0.5,   # NEW: research uses 0.5s segments
        segment_hop: float = 0.25,      # NEW: 50% overlap
    ) -> None:
```

The `_loop` method changes from:
- preprocess single chunk -> predict -> store result

To:
- split chunk into overlapping segments -> preprocess each -> predict each -> aggregate -> store result

**Risk:** LOW. The worker's external interface (`push()`, `get_latest()`) does not change. Only internal processing logic changes.

#### 2. `classification/preprocessing.py`

**Current:** `preprocess_for_cnn()` produces `(1, 3, 224, 224)` for EfficientNet.

**Change:** Rename file to `efficientnet_preprocessing.py` (keep for legacy/fallback). Create `research_preprocessing.py` with research params.

**Key differences:**

| Parameter | Current (EfficientNet) | Research CNN |
|-----------|----------------------|--------------|
| Output shape | (1, 3, 224, 224) | (1, 1, 128, 64) |
| Segment length | 2.0s | 0.5s |
| Normalization | (x - mean) / std | (x + 80) / 80, clip [0,1] |
| Resize | scipy.ndimage.zoom to 224x224 | None (native 128x64) |
| Channels | 3 (grayscale repeated) | 1 |

**Risk:** LOW. New file, old file preserved.

#### 3. `classification/inference.py`

**Current:** Only `OnnxDroneClassifier`.

**Change:** Add `Classifier` protocol import. Keep `OnnxDroneClassifier` as-is (it already satisfies the protocol). Add `ResearchClassifier` in separate file.

**Risk:** LOW. Additive change only.

#### 4. `pipeline.py` (BeamformingPipeline)

**Current:** Constructs `CNNWorker` dependencies are injected via `__init__`.

**Change:** No pipeline change needed if aggregation happens inside `CNNWorker`. The `ClassificationResult.drone_probability` field already carries the final score -- the pipeline just reads it. The aggregation is transparent.

**Risk:** NONE. The pipeline reads `result.drone_probability` which will now be the aggregated value.

#### 5. `config.py` (AcousticSettings)

**Current:** CNN settings for EfficientNet ONNX model.

**Change:** Add settings for research model, training, evaluation, collection.

```python
class AcousticSettings(BaseSettings):
    # ... existing ...

    # Research CNN (replaces EfficientNet)
    cnn_classifier_type: str = "research"  # "onnx_efficientnet" | "research" | "ensemble"
    research_model_path: str = "models/research_cnn_v1.onnx"
    ensemble_model_dir: str = "models/ensemble/"
    ensemble_mode: str = "soft"  # "soft" | "hard"

    # Segment aggregation
    segment_seconds: float = 0.5
    segment_hop_seconds: float = 0.25
    agg_w_max: float = 0.7
    agg_w_mean: float = 0.3

    # Training
    training_data_dir: str = "data/train"
    training_batch_size: int = 32
    training_epochs: int = 60
    training_patience: int = 8
    training_lr: float = 1e-3

    # Collection
    collection_output_dir: str = "data/recordings"
```

#### 6. `main.py` (FastAPI lifespan)

**Current:** Hardcoded `OnnxDroneClassifier` initialization.

**Change:** Factory logic based on `cnn_classifier_type` setting.

```python
# In lifespan():
if settings.cnn_classifier_type == "research":
    classifier = ResearchClassifier(settings.research_model_path)
    preprocessor = research_preprocess_for_cnn
elif settings.cnn_classifier_type == "ensemble":
    classifiers = load_ensemble(settings.ensemble_model_dir)
    classifier = EnsembleClassifier(classifiers, mode=settings.ensemble_mode)
    preprocessor = research_preprocess_for_cnn
else:
    classifier = OnnxDroneClassifier(settings.cnn_model_path)
    preprocessor = preprocess_for_cnn  # legacy

aggregator = SegmentAggregator(w_max=settings.agg_w_max, w_mean=settings.agg_w_mean)
cnn_worker = CNNWorker(classifier, preprocessor, aggregator, fs_in=settings.sample_rate)
```

#### 7. `api/routes.py`

**Current:** `/api/map`, `/api/targets`.

**Change:** Add training, evaluation, and collection route groups. Use separate APIRouters for modularity.

New endpoints:
- `POST /api/training/start` -- trigger training job
- `GET /api/training/status` -- poll training progress
- `POST /api/eval/run` -- trigger evaluation
- `GET /api/eval/results/{run_id}` -- get evaluation results
- `POST /api/collection/record` -- record labeled clip
- `GET /api/collection/stats` -- dataset statistics

### New Components

#### 1. `classification/protocol.py`
Classifier and Preprocessor protocols. Foundation for all other changes. **Build first.**

#### 2. `classification/research_model.py`
PyTorch `nn.Module` defining the 3-layer CNN. Used by both training and inference.

```python
class ResearchCNN(nn.Module):
    """3-layer CNN matching Acoustic-UAV-Identification architecture."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.classifier(self.features(x))
```

#### 3. `classification/research_classifier.py`
Implements `Classifier` protocol. Loads either `.pt` (PyTorch) or `.onnx` (ONNX Runtime) model file. Prefer ONNX for production (faster, no PyTorch dependency at runtime).

#### 4. `classification/research_preprocessing.py`
Research mel-spec pipeline: 0.5s segments, 16kHz, 64 mels, `(S_db+80)/80` normalization, output `(1, 1, 128, 64)`.

#### 5. `classification/aggregation.py`
Segment aggregation logic. Pure NumPy, no external dependencies.

#### 6. `classification/ensemble.py`
Multi-model voting. Loads N models from a directory, implements `Classifier` protocol.

#### 7. `training/dataset.py`
PyTorch `Dataset` replacing `tf.data` pipeline. Loads WAV files lazily, extracts random segments, applies research preprocessing.

#### 8. `training/trainer.py`
Training loop replacing `train_strong_cnn.py`. Uses PyTorch DataLoader, Adam optimizer, early stopping, LR scheduling. Saves `.pt` checkpoints.

#### 9. `training/export.py`
`torch.onnx.export()` wrapper to convert `.pt` to `.onnx` for production inference.

#### 10. `evaluation/harness.py`
Port of `eval_folder_with_strong_cnn.py`. Runs model on test folders, computes metrics.

#### 11. `evaluation/metrics.py`
Confusion matrix, accuracy, precision, recall, F1, distribution stats. Pure Python/NumPy.

#### 12. `collection/recorder.py`
Port of `uma16_dataset_collector_gui.py` for web-based recording. Records single-channel mono from UMA-16, saves WAV + JSON metadata.

#### 13. `collection/metadata.py`
Pydantic models for recording metadata (label, distance, altitude, session, etc.).

## Build Order (Suggested Phases)

Dependencies flow downward -- each phase builds on the previous.

### Phase 1: Foundation (Protocols + Research Preprocessing)

**Build:** `protocol.py`, `research_preprocessing.py`, `research_model.py`

**Rationale:** These have zero dependencies on existing code changes. They are pure additions. Tests can verify preprocessing matches research output exactly. The model architecture can be unit-tested with random tensors.

**Validates:** Preprocessing produces correct shapes and normalization. Model forward pass works.

### Phase 2: Research Classifier + Aggregation

**Build:** `research_classifier.py`, `aggregation.py`

**Rationale:** Depends on Phase 1 (protocol, preprocessing, model). Still pure additions -- no existing code modified yet. Can test end-to-end: audio -> preprocess -> predict -> aggregate.

**Validates:** Inference produces correct probabilities. Aggregation matches research behavior.

### Phase 3: Worker + Pipeline Integration

**Build:** Modify `worker.py` to use protocols and aggregation. Modify `config.py` for new settings. Modify `main.py` lifespan for classifier factory.

**Rationale:** This is where new code meets existing code. The worker's external interface (`push()`, `get_latest()`) stays the same, so the pipeline and state machine are unaffected. The key change is internal: segment splitting + aggregation inside the worker loop.

**Validates:** Real-time inference works end-to-end through the existing pipeline.

### Phase 4: Training Pipeline

**Build:** `training/dataset.py`, `training/trainer.py`, `training/export.py`, `training/config.py`

**Rationale:** Independent of the inference path. Can be built and tested in isolation. Depends on Phase 1 for the model architecture and preprocessing.

**Validates:** Training produces a model that passes Phase 2 inference tests.

### Phase 5: Evaluation Harness

**Build:** `evaluation/harness.py`, `evaluation/metrics.py`

**Rationale:** Depends on Phase 2 (classifier + aggregation) for running inference. Pure analysis code, no impact on production path.

**Validates:** Evaluation metrics match research baseline numbers.

### Phase 6: Ensemble Support

**Build:** `classification/ensemble.py`, extend classifier factory in `main.py`

**Rationale:** Depends on Phase 2 (multiple classifiers). Requires multiple trained models from Phase 4. Most complex ML feature, build last.

**Validates:** Ensemble outperforms single model on evaluation harness.

### Phase 7: Data Collection + API

**Build:** `collection/recorder.py`, `collection/metadata.py`, API routes for training/eval/collection

**Rationale:** Can be built in parallel with Phases 4-6 since it is mostly REST + recording logic. Placed last because the core inference path (Phases 1-3) is the critical integration. API routes for training/eval depend on those packages existing.

**Validates:** End-to-end flow: collect -> train -> evaluate -> deploy.

## Anti-Patterns

### Anti-Pattern 1: Replacing preprocessing without exact parameter matching
**What:** Using different mel-spec parameters (n_fft, hop, n_mels, normalization) between training and inference.
**Why bad:** The model sees a completely different feature distribution at inference time. Accuracy drops to random chance.
**Instead:** Extract preprocessing constants into a shared config. Both training and inference import from the same source. Add an integration test that verifies training preprocessing matches inference preprocessing bit-for-bit on the same audio input.

### Anti-Pattern 2: Running training on the inference thread
**What:** Blocking the audio processing pipeline with model training.
**Why bad:** Training takes minutes to hours. The real-time pipeline must process audio within 150ms chunks. Any blocking kills detection latency.
**Instead:** Training runs in a separate thread/process. The pipeline only loads the finished model. Hot-swap via the classifier factory after training completes.

### Anti-Pattern 3: Modifying the state machine for aggregation
**What:** Changing `DetectionStateMachine` to understand segment-level probabilities.
**Why bad:** The state machine's job is simple hysteresis on a single probability value. Making it segment-aware adds complexity and couples it to the research pipeline.
**Instead:** Aggregation produces a single probability. The state machine consumes it unchanged. Separation of concerns.

### Anti-Pattern 4: Storing models in the Docker image
**What:** Baking trained models into the Docker container.
**Why bad:** Models change frequently during development. Rebuilding the container for each model update is slow. Models can be 10-100MB.
**Instead:** Mount a `models/` volume. The service loads models from the filesystem at startup. Model hot-reload endpoint allows swapping without restart.

### Anti-Pattern 5: TensorFlow + PyTorch coexistence
**What:** Keeping TensorFlow as a dependency alongside PyTorch.
**Why bad:** Both frameworks are 500MB+ each. Docker image size doubles. Dependency conflicts (NumPy version wars). Confusing for developers.
**Instead:** Port everything to PyTorch. Export to ONNX for production inference (ONNX Runtime is 50MB). Remove TensorFlow entirely.

## Scalability Considerations

| Concern | Current (1 model) | With Ensemble (3-5 models) | Mitigation |
|---------|-------------------|---------------------------|------------|
| Inference latency | ~15ms (ONNX) | ~75ms (5x ONNX) | Parallelize with ThreadPool. ONNX sessions are thread-safe. |
| Memory usage | ~100MB (1 ONNX session) | ~500MB (5 sessions) | Acceptable for edge device. Could share backbone if architectures match. |
| Segment processing | 1 inference/push | ~7 inferences/push (0.5s segments from 2s buffer) | Batch the segments: single ONNX call with batch dim. (1, 1, 128, 64) -> (7, 1, 128, 64). |
| Training data growth | N/A | 10K+ WAV files | PyTorch DataLoader with num_workers for async loading. Index file (JSONL) for fast metadata access. |

## Sources

- Existing codebase: `src/acoustic/classification/` (inference.py, preprocessing.py, worker.py, state_machine.py)
- Existing codebase: `src/acoustic/pipeline.py`, `src/acoustic/config.py`, `src/acoustic/main.py`
- Research pipeline: `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` (TF training, research CNN arch)
- Research pipeline: `Acoustic-UAV-Identification-main-main/mic_realtime_inference.py` (real-time inference pattern)
- Research pipeline: `Acoustic-UAV-Identification-main-main/run_strong_inference.py` (segment aggregation: p_max, p_mean, weighted)
- Research pipeline: `Acoustic-UAV-Identification-main-main/eval_folder_with_strong_cnn.py` (evaluation: p_agg, confusion matrix, dist stats)
- Research pipeline: `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/Performance_Soft_Voting_Calcs.py` (soft voting, weighted ensemble)
- Research pipeline: `Acoustic-UAV-Identification-main-main/uma16_dataset_collector_gui.py` (UMA-16 recording + metadata)
- Project config: `.planning/PROJECT.md` (milestone v2.0 requirements)
- CLAUDE.md stack decisions (PyTorch over TensorFlow, ONNX export, FastAPI)
