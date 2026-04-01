# Roadmap: Sky Fort Acoustic Service

## Milestones

- 🚧 **v1.0 MVP** - Phases 1-5 (in progress)
- 📋 **v2.0 Research Classification Migration** - Phases 6-11 (planned)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>v1.0 MVP (Phases 1-5)</summary>

- [x] **Phase 1: Audio Capture, Beamforming, and Infrastructure** - Reliable 16-channel audio capture with real-time beamforming in a Docker container
- [ ] **Phase 2: REST API and Live Monitoring UI** - Visual feedback on beamforming output via WebSocket-driven React app
- [ ] **Phase 3: CNN Classification and Target Tracking** - Drone detection intelligence with WebSocket event publishing
- [ ] **Phase 4: Recording and Playback** - Capture field audio with metadata and replay through the full pipeline
- [ ] **Phase 5: CNN Training Pipeline** - In-service model training from labeled recordings

</details>

### v2.0 Research Classification Migration (Phases 6-11)

- [ ] **Phase 6: Preprocessing Parity Foundation** - Shared MelConfig, research preprocessing, Classifier/Preprocessor protocols, numerical parity tests
- [ ] **Phase 7: Research CNN and Inference Integration** - Research CNN architecture, segment aggregation, protocol-based worker injection, state machine recalibration
- [ ] **Phase 8: PyTorch Training Pipeline** - Background training with resource isolation, data augmentation, model checkpoint and export
- [ ] **Phase 9: Evaluation Harness and API** - Model evaluation with metrics, REST endpoints for training and evaluation, WebSocket progress streaming
- [ ] **Phase 10: Field Data Collection** - Record labeled audio from live UMA-16 via web UI with metadata and auto-organized directory structure
- [ ] **Phase 11: Late Fusion Ensemble (Conditional)** - Multi-model ensemble with accuracy-weighted soft voting, conditional on single-model accuracy results

## Phase Details

<details>
<summary>v1.0 MVP Phase Details (Phases 1-5)</summary>

### Phase 1: Audio Capture, Beamforming, and Infrastructure
**Goal**: The service captures 16-channel audio from the UMA-16v2 and produces a real-time beamforming spatial map inside a Docker container
**Depends on**: Nothing (first phase)
**Requirements**: AUD-01, AUD-02, AUD-03, BF-01, BF-02, BF-03, BF-04, INF-01, INF-03, INF-04
**Success Criteria** (what must be TRUE):
  1. Service starts in Docker, detects the UMA-16v2 device, and logs its presence (or logs a clear error if absent)
  2. 16-channel audio streams continuously at 48kHz without buffer overflows or dropped frames
  3. Beamforming produces an updating spatial map with a visible peak when a sound source is present
  4. Peak azimuth and elevation (pan/tilt degrees) are calculated and logged for the strongest source
  5. Configuration via environment variables controls device selection, frequency band, and service ports
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffolding, Docker setup, config, device detection, audio capture pipeline with ring buffer and simulator
- [x] 01-02-PLAN.md — SRP-PHAT beamforming engine (geometry, GCC-PHAT, 2D spatial map) with peak detection and noise gate
- [x] 01-03-PLAN.md — Integration: FastAPI app with health endpoint, beamforming pipeline wiring, end-to-end validation

### Phase 2: REST API and Live Monitoring UI
**Goal**: Users can see a live beamforming heatmap and target state through a web browser
**Depends on**: Phase 1
**Requirements**: API-01, API-02, API-03, UI-01, UI-02, UI-03, UI-08, INF-02
**Success Criteria** (what must be TRUE):
  1. Opening the web UI in a browser shows a live-updating beamforming heatmap (WebSocket, not polling)
  2. REST endpoint returns the current beamforming map on demand (image or JSON)
  3. REST endpoint returns a list of active targets with their current state
  4. Web UI displays target markers overlaid on the heatmap with class, speed, bearing, and ID (placeholder data until Phase 3)
  5. Web UI styling matches sky-fort-dashboard (React 19, Tailwind 4, same component patterns)
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Backend REST endpoints, WebSocket streaming, Pydantic models, and integration tests
- [ ] 02-02-PLAN.md — React web UI scaffold with live beamforming heatmap, target overlay, and dashboard layout
- [ ] 02-03-PLAN.md — Multi-stage Dockerfile and end-to-end human verification

### Phase 3: CNN Classification and Target Tracking
**Goal**: The service detects drones from audio using a binary CNN classifier, assigns persistent target IDs, and publishes tracking events over a dedicated `/ws/events` WebSocket
**Depends on**: Phase 2 (visual validation needed for tuning)
**Requirements**: CLS-01, CLS-02, CLS-03, CLS-04, TRK-01, TRK-02, TRK-03, TRK-04, TRK-05
**Success Criteria** (what must be TRUE):
  1. When a drone is present, the CNN classifies it as drone/not-drone and a target ID appears in the web UI within seconds
  2. Target ID persists across consecutive detections and disappears only after the source is lost (5s timeout)
  3. Detection does not flicker -- the hysteresis state machine prevents rapid on/off transitions
  4. `/ws/events` WebSocket subscribers receive detection events (new target), periodic updates (bearing), and lost events
  5. CNN model loads from a configurable file path at startup
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — ONNX CNN inference with mel-spectrogram preprocessing, hysteresis state machine, config extensions, and unit tests
- [x] 03-02-PLAN.md — Target tracker with UUID lifecycle, WebSocket event broadcaster, event schema, and unit tests
- [x] 03-03-PLAN.md — Pipeline integration: CNN worker thread, tracker wiring, /ws/events endpoint, placeholder swap to real data, end-to-end verification

### Phase 4: Recording and Playback
**Goal**: Users can record raw 16-channel audio from the web UI, attach metadata, and replay recordings through the full detection pipeline
**Depends on**: Phase 3 (replay validates the full pipeline including CNN + tracking)
**Requirements**: UI-04, UI-05, UI-06, UI-07
**Success Criteria** (what must be TRUE):
  1. User can start and stop a recording from the web UI, producing a raw 16-channel WAV file
  2. User can browse a list of recordings showing label, date, duration, and notes
  3. User can play back a recording and see it processed through beamforming, CNN, and tracking as if it were live
  4. User can attach and edit metadata on recordings (drone type, distance, conditions, notes)
**Plans**: TBD

Plans:
- [ ] 04-01: Recording manager (capture, storage, metadata CRUD)
- [ ] 04-02: Pipeline replay and recording UI

### Phase 5: CNN Training Pipeline
**Goal**: Users can train a new CNN model from labeled recordings directly through the web UI
**Depends on**: Phase 4 (training consumes labeled recordings)
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04
**Success Criteria** (what must be TRUE):
  1. User can select labeled recordings as a training dataset from the web UI
  2. Training runs as a background process without interrupting live detection
  3. Training produces a new model file and reports validation metrics (accuracy, confusion matrix)
  4. User can see training progress and results in the web UI
**Plans**: TBD

Plans:
- [ ] 05-01: Training pipeline (dataset preparation, PyTorch training loop, model output)
- [ ] 05-02: Training UI and model management

</details>

### v2.0 Research Classification Migration

**Milestone Goal:** Replace the EfficientNet-B0 ONNX classifier with the research-proven 3-layer CNN architecture, including PyTorch training, segment aggregation, model evaluation, field data collection, and optional late fusion ensemble.

### Phase 6: Preprocessing Parity Foundation
**Goal**: All audio preprocessing uses a single shared configuration with research-validated parameters, and protocols decouple classifiers from the pipeline
**Depends on**: Phase 3 (existing classification infrastructure)
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04
**Success Criteria** (what must be TRUE):
  1. A single MelConfig dataclass defines all preprocessing constants (SR=16000, N_FFT=1024, HOP=256, N_MELS=64, MAX_FRAMES=128, normalization) with no duplicate magic numbers anywhere in the codebase
  2. Classifier and Preprocessor protocols exist and the existing OnnxDroneClassifier can be wrapped to implement the Classifier protocol without changing its internals
  3. Feeding a 0.5s audio segment through the research preprocessor produces a (1, 1, 128, 64) tensor with values in [0, 1] using (S_db+80)/80 normalization
  4. Numerical parity tests pass: the same WAV file processed through both TF research code and the new PyTorch preprocessor produces tensors within atol=1e-4
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md — MelConfig dataclass, Classifier/Preprocessor protocols, ONNX removal, reference fixtures
- [x] 06-02-PLAN.md — ResearchPreprocessor (torchaudio), CNNWorker protocol refactor, pipeline 0.5s segments, parity tests

### Phase 7: Research CNN and Inference Integration
**Goal**: The live detection pipeline uses the research CNN architecture with segment aggregation, swappable via protocol injection at startup
**Depends on**: Phase 6
**Requirements**: MDL-01, MDL-02, MDL-03, MDL-04
**Success Criteria** (what must be TRUE):
  1. ResearchCNN model (3-layer Conv2D 32/64/128, BN, MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid) accepts (N, 1, 128, 64) input and produces a single probability per sample
  2. Audio chunks are split into overlapping 0.5s segments, each classified independently, and aggregated via configurable p_max/p_mean/p_agg weights before feeding the state machine
  3. CNNWorker accepts injected Classifier, Preprocessor, and Aggregator via protocols; a factory in main.py selects the implementation at startup based on config
  4. State machine thresholds (enter/exit) are configurable via environment variables to accommodate the new CNN's different confidence distribution
  5. The pipeline processes audio end-to-end with the new classifier without crashing or regressing beamforming performance
**Plans**: 2 plans

Plans:
- [ ] 07-01-PLAN.md — ResearchCNN model, Aggregator protocol, WeightedAggregator, config extensions, unit tests
- [ ] 07-02-PLAN.md — CNNWorker segment buffer, pipeline overlap push, classifier factory wiring, integration tests

### Phase 8: PyTorch Training Pipeline
**Goal**: Users can train a research CNN model from labeled WAV files with the training process isolated from live detection
**Depends on**: Phase 6 (shared preprocessing), Phase 7 (model architecture)
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04
**Success Criteria** (what must be TRUE):
  1. Training pipeline loads WAV files lazily, extracts random 0.5s segments, and trains with Adam optimizer, BCE loss, and early stopping
  2. Training runs as a background thread with resource isolation (os.nice, thread limits) and does not degrade live detection latency below the 150ms beamforming deadline
  3. Training produces a model checkpoint (.pt) and exports to a deployable format on completion
  4. Training data augmentation (SpecAugment time/frequency masking and waveform augmentation) is applied during training and can be toggled via config
**Plans**: TBD

### Phase 9: Evaluation Harness and API
**Goal**: Operators can evaluate classifier accuracy on labeled test data and control training and evaluation via REST endpoints with real-time progress updates
**Depends on**: Phase 7 (inference path), Phase 8 (training pipeline)
**Requirements**: EVL-01, EVL-02, API-01, API-02
**Success Criteria** (what must be TRUE):
  1. Evaluation harness runs the classifier on labeled test folders and produces confusion matrix, precision/recall/F1, and distribution stats (p_agg/p_max/p_mean percentiles)
  2. Evaluation provides per-file detailed output showing segment-level probabilities and final aggregation scores
  3. REST endpoints allow starting a training run, checking training progress, running an evaluation, and retrieving evaluation results
  4. Training progress is streamed via WebSocket so the UI can show real-time updates (epoch, loss, metrics)
**Plans**: TBD

### Phase 10: Field Data Collection
**Goal**: Users can record labeled audio clips from the live UMA-16 microphone array through the web UI, building a training dataset
**Depends on**: Phase 8 (recordings feed training pipeline)
**Requirements**: COL-01, COL-02, COL-03
**Success Criteria** (what must be TRUE):
  1. User can start a labeled recording session from the web UI, specifying drone type and recording conditions
  2. User can attach and edit metadata on recordings (drone type, distance, altitude, conditions, notes)
  3. Recordings are automatically saved into a directory structure (data/recordings/{label}/{bin}/) that the training pipeline can directly consume without manual reorganization
**Plans**: TBD
**UI hint**: yes

### Phase 11: Late Fusion Ensemble (Conditional)
**Goal**: Multiple classifiers combine via accuracy-weighted soft voting to improve detection accuracy beyond what a single model achieves
**Depends on**: Phase 9 (evaluation harness measures single-model accuracy; build only if accuracy is insufficient)
**Requirements**: ENS-01, ENS-02
**Success Criteria** (what must be TRUE):
  1. EnsembleClassifier wraps N models via the Classifier protocol with accuracy-weighted soft voting and normalized weights
  2. Ensemble inference runs within real-time latency budget (max 3 models for live detection, N models for offline evaluation)
  3. Ensemble evaluation on the harness shows measurable improvement over the best single-model baseline
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 6 -> 7 -> 8 -> 9 -> 10 -> 11

Note: Phase 11 is conditional -- build only if Phase 9 evaluation shows single-model accuracy is insufficient.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Audio Capture, Beamforming, and Infrastructure | v1.0 | 3/3 | Complete | 2026-03-30 |
| 2. REST API and Live Monitoring UI | v1.0 | 1/3 | In Progress | |
| 3. CNN Classification and Target Tracking | v1.0 | 3/3 | Complete | |
| 4. Recording and Playback | v1.0 | 0/2 | Not started | - |
| 5. CNN Training Pipeline | v1.0 | 0/2 | Not started | - |
| 6. Preprocessing Parity Foundation | v2.0 | 2/2 | Complete   | 2026-04-01 |
| 7. Research CNN and Inference Integration | v2.0 | 0/2 | Not started | - |
| 8. PyTorch Training Pipeline | v2.0 | 0/? | Not started | - |
| 9. Evaluation Harness and API | v2.0 | 0/? | Not started | - |
| 10. Field Data Collection | v2.0 | 0/? | Not started | - |
| 11. Late Fusion Ensemble (Conditional) | v2.0 | 0/? | Not started | - |
