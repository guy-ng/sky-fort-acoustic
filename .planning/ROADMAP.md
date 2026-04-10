# Roadmap: Sky Fort Acoustic Service

## Milestones

- 🚧 **v1.0 MVP** - Phases 1-5 (in progress)
- 📋 **v2.0 Research Classification Migration** - Phases 6-12 (planned)
- 📋 **v3.0 DADS-Powered Detection Upgrade** - Phases 13-16 (planned)
- 📋 **v4.0 Research-Based Beamforming & Direction Calculation** - Phases 17-19 (planned)

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

### v3.0 DADS-Powered Detection Upgrade (Phases 13-16)

- [ ] **Phase 13: DADS Dataset Integration and Training Data Pipeline** - Download, validate, integrate DADS dataset with session-level splitting
- [ ] **Phase 14: EfficientAT Model Architecture with AudioSet Transfer Learning** - MobileNetV3 mn10 with three-stage unfreezing transfer learning
- [ ] **Phase 15: Advanced Training Enhancements** - Focal loss, noise augmentation, balanced sampling, waveform augmentations
- [ ] **Phase 16: Edge Export Pipeline** - ONNX, TensorRT, TFLite quantization for edge deployment

### v4.0 Research-Based Beamforming & Direction Calculation (Phases 17-19)

- [ ] **Phase 17: Beamforming Engine Upgrade and Pipeline Integration** - Research-validated SRP-PHAT with bandpass filtering, MCRA noise estimation, sub-grid interpolation, multi-peak detection, wired into live pipeline
- [ ] **Phase 18: Direction of Arrival and WebSocket Broadcasting** - Pan/tilt calculation with coordinate transforms, per-target direction tracking, WebSocket direction events
- [ ] **Phase 19: Functional Beamforming Visualization** - Sidelobe-suppressed heatmap display with corrected frequency band and configurable nu parameter

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
**Plans**: 3 plans

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
**Plans**: 3 plans

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
- [x] 07-01-PLAN.md — ResearchCNN model, Aggregator protocol, WeightedAggregator, config extensions, unit tests
- [x] 07-02-PLAN.md — CNNWorker segment buffer, pipeline overlap push, classifier factory wiring, integration tests

### Phase 8: PyTorch Training Pipeline
**Goal**: Users can train a research CNN model from labeled WAV files with the training process isolated from live detection
**Depends on**: Phase 6 (shared preprocessing), Phase 7 (model architecture)
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04
**Success Criteria** (what must be TRUE):
  1. Training pipeline loads WAV files lazily, extracts random 0.5s segments, and trains with Adam optimizer, BCE loss, and early stopping
  2. Training runs as a background thread with resource isolation (os.nice, thread limits) and does not degrade live detection latency below the 150ms beamforming deadline
  3. Training produces a model checkpoint (.pt) and exports to a deployable format on completion
  4. Training data augmentation (SpecAugment time/frequency masking and waveform augmentation) is applied during training and can be toggled via config
**Plans**: 3 plans

Plans:
- [x] 08-01-PLAN.md — TrainingConfig, data augmentation (SpecAugment + waveform), DroneAudioDataset with lazy loading and random segment extraction
- [x] 08-02-PLAN.md — TrainingRunner (training loop + early stopping + checkpoint), TrainingManager (background thread + progress + cancellation)
- [x] 08-03-PLAN.md — Gap closure: TorchScript export, torch.set_num_threads isolation, confusion matrix tracking

### Phase 9: Evaluation Harness and API
**Goal**: Operators can evaluate classifier accuracy on labeled test data and control training and evaluation via REST endpoints with real-time progress updates
**Depends on**: Phase 7 (inference path), Phase 8 (training pipeline)
**Requirements**: EVL-01, EVL-02, API-01, API-02
**Success Criteria** (what must be TRUE):
  1. Evaluation harness runs the classifier on labeled test folders and produces confusion matrix, precision/recall/F1, and distribution stats (p_agg/p_max/p_mean percentiles)
  2. Evaluation provides per-file detailed output showing segment-level probabilities and final aggregation scores
  3. REST endpoints allow starting a training run, checking training progress, running an evaluation, and retrieving evaluation results
  4. Training progress is streamed via WebSocket so the UI can show real-time updates (epoch, loss, metrics)
**Plans**: 2 plans

Plans:
- [x] 09-01-PLAN.md — Evaluation module (Evaluator class, metrics, distribution stats, per-file output) and Pydantic API models
- [x] 09-02-PLAN.md — REST endpoints (training, eval, models), WebSocket /ws/training, main.py wiring, integration tests

### Phase 10: Field Data Collection
**Goal**: Users can record labeled audio clips from the live UMA-16 microphone array through the web UI, building a training dataset
**Depends on**: Phase 8 (recordings feed training pipeline)
**Requirements**: COL-01, COL-02, COL-03
**Success Criteria** (what must be TRUE):
  1. User can start a labeled recording session from the web UI, specifying drone type and recording conditions
  2. User can attach and edit metadata on recordings (drone type, distance, altitude, conditions, notes)
  3. Recordings are automatically saved into a directory structure (data/field/{label}/) that the training pipeline can directly consume without manual reorganization
**Plans**: 3 plans
**UI hint**: yes

Plans:
- [x] 10-01-PLAN.md — Backend recording module: config, metadata, recorder session, manager with auto-stop and label workflow
- [x] 10-02-PLAN.md — REST API endpoints, WebSocket /ws/recording, pipeline chunk forwarding, main.py wiring, integration tests
- [x] 10-03-PLAN.md — Frontend recording UI: controls panel, recordings list, metadata editor, sidebar tab integration

### Phase 11: Late Fusion Ensemble (Conditional)
**Goal**: Multiple classifiers combine via accuracy-weighted soft voting to improve detection accuracy beyond what a single model achieves
**Depends on**: Phase 9 (evaluation harness measures single-model accuracy; build only if accuracy is insufficient)
**Requirements**: ENS-01, ENS-02
**Success Criteria** (what must be TRUE):
  1. EnsembleClassifier wraps N models via the Classifier protocol with accuracy-weighted soft voting and normalized weights
  2. Ensemble inference runs within real-time latency budget (max 3 models for live detection, N models for offline evaluation)
  3. Ensemble evaluation on the harness shows measurable improvement over the best single-model baseline
**Plans**: 2 plans

Plans:
- [ ] 11-01-PLAN.md — Core ensemble module: EnsembleClassifier, model registry, config parsing, AcousticSettings extension, unit tests
- [ ] 11-02-PLAN.md — Integration: Evaluator refactor for ensemble, eval API per-model metrics, main.py ensemble factory wiring

### Phase 12: Add ML Training & Testing UI Tab
**Goal**: Operators can control CNN training, view evaluation results, and manage models from a new Training tab in the web UI sidebar
**Depends on**: Phase 9 (backend training/eval APIs), Phase 10 (recording UI patterns)
**Requirements**: TRN-04
**Success Criteria** (what must be TRUE):
  1. Training tab accessible from sidebar with collapsible Train/Evaluate/Models accordion sections
  2. Operator can start training with configurable hyperparameters and see live loss chart via WebSocket
  3. Evaluation results display accuracy, precision, recall, F1, confusion matrix, and per-file details
  4. Model list shows available .pt files with active model highlighted
**Plans**: 2 plans

Plans:
- [x] 12-01-PLAN.md — Install Recharts, TypeScript interfaces for training/eval/model APIs, data hooks (useTraining, useTrainingSocket, useEvaluation, useModels)
- [ ] 12-02-PLAN.md — UI components (TrainingPanel accordion, TrainSection, TrainingProgress chart, EvalSection, EvaluationResults, ModelsSection) + Sidebar integration + visual verification

### v3.0 DADS-Powered Detection Upgrade

**Milestone Goal:** Upgrade the detection model using the DADS public dataset (180K files, 60.9 hours), adopt EfficientAT MobileNetV3 architecture with AudioSet pretraining, enhance training with focal loss and noise augmentation, and add edge export pipeline for TensorRT/TFLite deployment.

### Phase 13: DADS Dataset Integration and Training Data Pipeline
**Goal**: Download, validate, and integrate the DADS dataset (geronimobasso/drone-audio-detection-samples from HuggingFace) into the training pipeline with proper session-level data splitting
**Depends on**: Phase 8 (training pipeline), Phase 9 (evaluation harness)
**Requirements**: DAT-01, DAT-02, DAT-03
**Success Criteria** (what must be TRUE):
  1. DADS dataset (180,320 WAV files, ~60.9 hours) is downloaded and validated (PCM 16-bit, mono, 16 kHz)
  2. Dataset loader handles the DADS directory structure (drone/no-drone folders) and integrates with the existing DroneAudioDataset
  3. Session-level (file-level) data splitting prevents data leakage — all segments from the same recording go into the same split (70/15/15 train/val/test)
  4. Training pipeline can train on DADS data end-to-end and produce a model with >90% baseline accuracy
**Plans**: 2 plans

Plans:
- [x] 13-01-PLAN.md — ParquetDataset class with shard scanning, WAV byte decoding, deterministic 70/15/15 split, unit tests
- [x] 13-02-PLAN.md — TrainingRunner Parquet integration, config extension (dads_path), field recording Parquet output, integration tests

### Phase 14: EfficientAT Model Architecture with AudioSet Transfer Learning
**Goal**: Replace the custom 3-layer CNN with EfficientAT MobileNetV3 (mn10, ~4.5M params) pretrained on AudioSet, using the three-stage unfreezing transfer learning recipe
**Depends on**: Phase 13 (DADS dataset for training), Phase 6 (Classifier protocol)
**Requirements**: MDL-10, MDL-11, MDL-12
**Success Criteria** (what must be TRUE):
  1. EfficientAT mn10 model (~4.5M params, ~18MB) loads with AudioSet-pretrained weights and implements the Classifier protocol
  2. Three-stage transfer learning works: Stage 1 (head only, lr=1e-3), Stage 2 (last 2-3 blocks, lr=1e-4), Stage 3 (all layers, lr=1e-5) with cosine annealing
  3. Fine-tuned model achieves >95% binary detection accuracy on DADS test set
  4. Model can be swapped in at startup via config without code changes (classifier factory)
**Plans**: 2 plans

Plans:
- [x] 14-01-PLAN.md — Vendor EfficientAT mn10 model, classifier wrapper, registry, config, download script
- [x] 14-02-PLAN.md — Three-stage unfreezing transfer learning runner, training manager integration

### Phase 15: Advanced Training Enhancements - Focal Loss, Noise Augmentation, Balanced Sampling
**Goal**: Enhance training with focal loss, background noise augmentation (ESC-50/UrbanSound8K mixing), class-balanced sampling, and waveform augmentations for robust real-world performance
**Depends on**: Phase 14 (EfficientAT model), Phase 13 (DADS dataset)
**Requirements**: TRN-10, TRN-11, TRN-12
**Success Criteria** (what must be TRUE):
  1. Focal Loss (gamma=2.0, alpha=0.25) replaces BCE as the default training loss, with fallback to weighted BCE
  2. Background noise augmentation mixes drone audio with ESC-50/UrbanSound8K at SNR -10 to +20 dB during training (most impactful augmentation per research)
  3. Class-balanced sampling targets ~50/50 drone/no-drone ratio per batch regardless of dataset imbalance
  4. Waveform augmentations (pitch shift ±3 semitones, time stretch 0.85-1.15x, gain -6 to +6 dB) are applied via audiomentations with configurable probabilities
  5. Model achieves <5% false positive rate with >95% recall on DADS test set
**Plans**: 2 plans

Plans:
- [x] 15-01-PLAN.md — FocalLoss module, BackgroundNoiseMixer, AudiomentationsAugmentation, config extensions, unit tests
- [x] 15-02-PLAN.md — Trainer integration: wire focal loss, new augmentations, ResearchCNN logits mode, dataset type widening, integration tests

### Phase 16: Edge Export Pipeline - ONNX TensorRT TFLite Quantization
**Goal**: Export trained models to ONNX format with optional TensorRT FP16/INT8 and TFLite INT8 quantization for edge deployment on Jetson and Raspberry Pi
**Depends on**: Phase 14 (EfficientAT model to export)
**Requirements**: DEP-01, DEP-02, DEP-03
**Success Criteria** (what must be TRUE):
  1. Trained PyTorch model exports to ONNX (opset 13+) with verified numerical parity (atol=1e-4)
  2. ONNX model converts to TensorRT FP16 engine with <30ms inference latency on Jetson
  3. ONNX model converts to TFLite INT8 with post-training quantization using calibration dataset
  4. Quantized models maintain >94% accuracy (within 1% of full-precision baseline)
  5. REST API endpoint allows model export with format selection (onnx, tensorrt, tflite)
**Plans**: 3 plans

Plans:
- [ ] 16-01-PLAN.md — Core ONNX export pipeline: model wrappers, ExportPipeline class, parity validation, unit tests
- [ ] 16-02-PLAN.md — Optional TensorRT FP16 and TFLite INT8 converters with graceful skip
- [ ] 16-03-PLAN.md — REST API export endpoint, background task execution, main.py wiring, integration tests

### v4.0 Research-Based Beamforming & Direction Calculation

**Milestone Goal:** Upgrade the beamforming engine based on research to produce accurate azimuth/elevation DOA estimates, wire beamforming back into the live pipeline (replacing the current stub), add functional beamforming for clean visualization, and publish direction data over WebSocket.

### Phase 17: Beamforming Engine Upgrade and Pipeline Integration
**Goal**: The beamforming engine operates with research-validated parameters and is wired into the live pipeline, producing real-time spatial maps with accurate peak detection
**Depends on**: Phase 1 (existing beamforming modules in src/acoustic/beamforming/)
**Requirements**: BF-10, BF-11, BF-12, BF-13, BF-14, BF-15, BF-16
**Success Criteria** (what must be TRUE):
  1. Beamforming processes audio in the 500-4000 Hz band with a 4th-order Butterworth bandpass pre-filter applied per-channel, and the spatial aliasing limit is not exceeded
  2. Peak DOA is refined to sub-degree accuracy via parabolic interpolation on the SRP-PHAT grid, producing smoother bearing estimates than grid-only resolution
  3. Multiple simultaneous sources are detected as separate peaks with configurable minimum angular separation and threshold
  4. MCRA noise estimator provides an adaptive noise floor that adjusts to changing outdoor conditions without manual recalibration
  5. The live pipeline's process_chunk calls the real beamforming engine (not the zero-map stub) and produces updating spatial maps at the 150ms chunk rate
  6. Beamforming activates only after CNN drone detection and deactivates after 5 seconds of no detection — idle state returns zero maps to save compute
**Plans**: 3 plans

Plans:
- [x] 17-01-PLAN.md — Bandpass pre-filter (500-4000 Hz Butterworth), parabolic sub-grid interpolation, config extensions
- [x] 17-02-PLAN.md — MCRA adaptive noise estimator, multi-peak detection with angular separation
- [x] 17-03-PLAN.md — Pipeline integration: wire real beamforming into process_chunk, demand-driven CNN-gated activation

### Phase 18: Direction of Arrival and WebSocket Broadcasting
**Goal**: Each detected source has accurate pan/tilt degrees that update as the source moves, and direction data is broadcast to WebSocket subscribers in real time
**Depends on**: Phase 17 (beamforming peaks feed DOA calculation)
**Requirements**: DOA-01, DOA-02, DOA-03, DIR-01, DIR-02
**Success Criteria** (what must be TRUE):
  1. Pan (azimuth) and tilt (elevation) degrees are calculated from each beamforming peak and correctly account for the UMA-16v2 vertical mounting orientation
  2. Coordinate transform correctly maps the array's x-y plane to world azimuth/elevation so that a source at physical 0/0 produces 0/0 in the output
  3. Per-target direction tracking persists bearing across updates and smoothly tracks a moving source without jumps or resets
  4. WebSocket /ws/events broadcasts detection events containing target ID, azimuth, elevation, pan, and tilt degrees for each active target
  5. Periodic direction updates are published per active target at a configurable rate (default matching the beamforming chunk rate)
**Plans**: 3 plans

Plans:
- [ ] 18-01-PLAN.md — DOA coordinate transform module (array-to-world pan/tilt with mounting orientation)
- [ ] 18-02-PLAN.md — Multi-target tracker upgrade (nearest-neighbor association, EMA smoothing, pan/tilt fields)
- [ ] 18-03-PLAN.md — Pipeline wiring, schema enrichment, configurable /ws/targets rate

### Phase 19: Functional Beamforming Visualization
**Goal**: The heatmap displays a clean, sidelobe-suppressed beamforming map using functional beamforming with the corrected frequency band
**Depends on**: Phase 17 (upgraded beamforming engine output)
**Requirements**: VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. The heatmap reflects beamforming output computed in the 500-4000 Hz band, showing sharper source localization than the previous 100-2000 Hz band
  2. Functional beamforming with a configurable nu parameter (default nu=100) suppresses sidelobes so that the heatmap shows distinct source peaks instead of smeared energy
  3. The nu parameter is adjustable at runtime via config or API so operators can tune visualization sharpness for different environments
**Plans**: 3 plans
**UI hint**: yes

Plans:
- [ ] TBD (run /gsd:plan-phase 19 to break down)

## Progress

**Execution Order:**
Phases execute in numeric order. Phase 11 is conditional. v3.0 phases: 13 -> 14 -> 15 -> 16. v4.0 phases: 17 -> 18 -> 19.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Audio Capture, Beamforming, and Infrastructure | v1.0 | 3/3 | Complete | 2026-03-30 |
| 2. REST API and Live Monitoring UI | v1.0 | 1/3 | In Progress | |
| 3. CNN Classification and Target Tracking | v1.0 | 3/3 | Complete | |
| 4. Recording and Playback | v1.0 | 0/2 | Not started | - |
| 5. CNN Training Pipeline | v1.0 | 0/2 | Not started | - |
| 6. Preprocessing Parity Foundation | v2.0 | 2/2 | Complete   | 2026-04-01 |
| 7. Research CNN and Inference Integration | v2.0 | 0/2 | Not started | - |
| 8. PyTorch Training Pipeline | v2.0 | 2/3 | In Progress|  |
| 9. Evaluation Harness and API | v2.0 | 2/2 | Complete | 2026-04-02 |
| 10. Field Data Collection | v2.0 | 3/3 | Complete   | 2026-04-02 |
| 11. Late Fusion Ensemble (Conditional) | v2.0 | 0/2 | Complete    | 2026-04-02 |
| 12. Add ML Training & Testing UI Tab | v2.0 | 1/2 | In Progress|  |
| 13. DADS Dataset Integration and Training Data Pipeline | v3.0 | 2/2 | Complete   | 2026-04-03 |
| 14. EfficientAT Model Architecture with AudioSet Transfer Learning | v3.0 | 2/2 | Complete    | 2026-04-04 |
| 15. Advanced Training Enhancements | v3.0 | 2/2 | Complete   | 2026-04-04 |
| 16. Edge Export Pipeline - ONNX TensorRT TFLite Quantization | v3.0 | 0/3 | Not started | - |
| 17. Beamforming Engine Upgrade and Pipeline Integration | v4.0 | 3/3 | Complete    | 2026-04-06 |
| 18. Direction of Arrival and WebSocket Broadcasting | v4.0 | 0/0 | Not started | - |
| 19. Functional Beamforming Visualization | v4.0 | 0/0 | Not started | - |
| 20. Retrain v7 with wide gain + RIR + Vertex remote | v1.0 | 7/9 + 1 partial | ⚠ Blocked (data acq.) | - |

### Phase 20: Retrain v7 with wide gain + room-IR augmentation, Vertex remote training, 60% overlap windows, expanded BG noise negatives

**Goal:** Retrain v7 with deployment-matched augmentation so the model generalizes to real UMA-16 captures. Target augmentations: ±40 dB random gain, room impulse-response (RIR) convolution, additive UMA-16 ambient noise. Data pipeline: 60% window overlap for more samples; expand negative set with background noise sources from `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md`. Training executes remotely on Vertex AI (remote Docker), not locally. Deliverable: retrained v7 checkpoint with measurable improvement on real captures.

**Requirements**: TBD
**Depends on:** Phase 19
**Plans:** 7/9 complete + 1 code-complete-but-blocked (20-05 Task 1)
**Status:** ⚠ BLOCKED on noise-corpora data acquisition (2026-04-07)

**Blocker:** `Dockerfile.vertex-base` and `build_env_vars_v7()` reference `data/noise/{esc50,urbansound8k,fsd50k_subset}/` — only `.gitkeep` placeholders exist on disk. Plan 20-00 created the placeholder dirs but the actual ESC50 / UrbanSound8K / FSD50K downloads were always meant to be a manual capture step that has not happened. Submitting v7 as-is would silently train against ONE noise source (UMA-16 outdoor_quiet ambient pool, 31 min) instead of FOUR — almost certainly failing the locked promotion gate (DADS≥0.95 / real_TPR≥0.80 / real_FPR≤0.05). See `.planning/phases/20-.../20-05-SUMMARY.md` "BLOCKED ON DATA ACQUISITION" section.

**Resolution path:** Create a new phase (e.g. Phase 20.1 via `/gsd-insert-phase` or Phase 21 via `/gsd-add-phase`) that:
1. Acquires ESC50 (~600 MB, CC BY-NC 4.0), UrbanSound8K (~6 GB, registration required), and an FSD50K subset (~2–4 GB curated)
2. Places them under `data/noise/{esc50,urbansound8k,fsd50k_subset}/`
3. Adds a host-side preflight test (e.g., `tests/integration/test_noise_corpora_present.py`) that fails if any noise dir contains <N audio files — so this can never silently happen again
4. Verifies the parquet ambient/eval shards are still in sync with the WAV trees

After that phase completes, run `/gsd-execute-phase 20 --wave 4` to trigger the human checkpoint (build/push/submit) on the ALREADY-COMMITTED Plan 20-05 code, then `/gsd-execute-phase 20 --wave 5` for the eval harness + promotion gate.

Plans:
- [x] 20-00 Wave 0 test stubs and data acquisition placeholders (commits f6af9ed..03e7d39)
- [x] 20-01 New augmentations: WideGainAugmentation + RoomIRAugmentation (commit 3319920)
- [x] 20-02 BackgroundNoiseMixer UMA-16 + TrainingConfig fields (commits 660da2c..5c3c993)
- [x] 20-03 Sliding-window dataset + session-level split (commits 6a3e434..ea2d6fe)
- [x] 20-04 Trainer wiring (commits 1948e14..59391ae)
- [x] 20-07 Trainer correctness fixes D-30..D-33 (commits 567afe7..b63404b)
- [x] 20-08 RMS normalization D-34 (commits d48dbb9..b5244db)
- [~] 20-05 Vertex Docker + submit — Task 1 code-complete (commits 428176c..70b02bb); **Task 2 BLOCKED on noise-corpora data acquisition**
- [ ] 20-06 Eval harness + promotion gate — not yet started; depends on v7 checkpoint from 20-05 Task 2

### Phase 20.1: Acquire noise corpora ESC50 UrbanSound8K FSD50K subset and add host preflight test (INSERTED)

**Goal:** Acquire ESC-50, UrbanSound8K, and an FSD50K 6-class subset onto disk under `data/noise/{esc50,urbansound8k,fsd50k_subset}/` via a single idempotent script (`scripts/acquire_noise_corpora.py`), and add hard preflight tests that fail loudly if any noise corpus is missing or if the parquet ambient/eval shards have drifted from their WAV source trees — so Phase 20-05 Task 2 (Vertex submit) can be unblocked and the silent failure can never recur.
**Requirements**: 19 D-XX decisions in 20.1-CONTEXT.md (no formal REQ-IDs — phase is an unblocker for Phase 20)
**Depends on:** Phase 20
**Plans:** 3 plans

Plans:
- [ ] 20.1-01-PLAN.md — soundata dep + scripts/acquire_noise_corpora.py (CLI, disk guard, idempotency marker, FSD50K class filter) + unit tests (D-01..D-10) + operator-gated download checkpoint
- [ ] 20.1-02-PLAN.md — tests/integration/test_noise_corpora_present.py preflight gate with module-level ACOUSTIC_SKIP_NOISE_PREFLIGHT opt-out + D-15 actionable failure messages + meta-tests (D-11..D-15)
- [ ] 20.1-03-PLAN.md — tests/integration/test_parquet_shards_in_sync.py (ambient rglob + eval labels.json row formulas + mtime gate) + drift meta-tests (D-16..D-19)

### Phase 21: Build Raspberry Pi 4 edge drone-detection app using efficientat_mn10_v6.pt with configurable detection params, GPIO LED alarm, and model conversion

**Goal:** Ship a standalone Raspberry Pi 4 edge application that loads a host-converted ONNX build of `efficientat_mn10_v6.pt`, runs continuous audio classification from a single USB microphone, drives a GPIO LED alarm through a hysteresis state machine, persists every detection to an always-on rotating JSONL log, and installs cleanly via a systemd unit + bare-venv script. The edge app lives in a new `apps/rpi-edge/` tree, vendors a pure-numpy port of the training-side mel preprocessing (CI drift-guarded), and exposes a localhost-only `/health` + `/status` HTTP endpoint.
**Requirements**: 28 D-XX decisions in 21-CONTEXT.md (no formal REQ-IDs — this phase replicates CLS-01..CLS-04 / AUD-01..AUD-03 behavior on the Pi edge)
**Depends on:** Phase 20
**Plans:** 8 plans

Plans:
- [ ] 21-01-PLAN.md — Wave 0: apps/rpi-edge/ skeleton, pinned deps, 13 RED pytest stubs, conftest fixtures, model head-shape inspection
- [ ] 21-02-PLAN.md — Pure-numpy mel preprocessing (no torch) + golden-parity test + main-repo CI drift guard (D-02, D-04)
- [ ] 21-03-PLAN.md — Host-side scripts/convert_efficientat_to_onnx.py with FP32 + int8 exports, sanity gate, sha256 checksums (D-05, D-06, D-07, D-08)
- [ ] 21-04-PLAN.md — EdgeConfig dataclass tree + YAML loader (yaml.safe_load) + CLI overrides, all four D-11 param groups, loopback-only validation (D-09, D-10, D-11)
- [ ] 21-05-PLAN.md — AudioCapture (48→32 kHz resample_poly), OnnxClassifier (int8-preferred + FP32 fallback + checksum verify), HysteresisStateMachine (D-01, D-03, D-12)
- [ ] 21-06-PLAN.md — GPIO LED (SIGTERM-safe), AudioAlarm (silent-degrade), DetectionLogger (always-on rotating JSONL) (D-13..D-23)
- [ ] 21-07-PLAN.md — Stdlib HTTP /health + /status (127.0.0.1 only), RuntimeState, __main__ composition root, e2e golden WAV test (D-24)
- [ ] 21-08-PLAN.md — systemd unit (hardened), scripts/install_edge_rpi.sh (idempotent), README, on-device smoke test checkpoint (D-23, D-25, D-26, D-27, D-28)

### Phase 22: EfficientAT v8 retrain with fixed train/serve window contract and 2026-04-08 field recordings

**Goal:** Train `efficientat_mn10_v8.pt` that beats v6 on a real-device hold-out and replaces v6 in operational use. Fix the train/serve window-length contract bug (root cause of v7 regression — see `.planning/debug/efficientat-v7-regression-vs-v6.md`), include 2026-04-08 field recordings as new training/eval data, train on Vertex AI L4 in `us-east1`, and gate promotion on the D-27 real-device TPR/FPR metrics.
**Requirements**: REQ-22-W1, REQ-22-W2, REQ-22-W3, REQ-22-W4, REQ-22-D1, REQ-22-D2, REQ-22-D3, REQ-22-G1, REQ-22-G2 (phase-local — see 22-RESEARCH.md § Phase Requirements)
**Depends on:** Phase 20.1 (noise corpora), Phase 21 (edge consumer)
**Plans:** 3/9 plans executed

**User constraints (locked before planning):**
- Window: 1.0 second @ 32 kHz (32000 samples) — must equal `EfficientATMelConfig().segment_samples`
- Sliding-window overlap: 50% (`window_overlap_ratio=0.5`)
- New training data: all `data/field/drone/20260408_*.wav` and `data/field/background/20260408_*.wav`. The `20260408_091054_136dc5.wav` "10inch payload 4kg" recording was trimmed to 61.4s on 2026-04-08; use as-is.
- Cloud: Vertex AI region `us-east1`, NVIDIA L4 GPU
- Data integrity preflight required: assert every recording is correctly transferred, decoded, and reaches the training loop (no silent drops, no SR mismatches, no label flips)
- Carry over from Phase 20: wide-gain aug, room-IR aug, BG noise negatives (ESC50/UrbanSound8K/FSD50K subset), focal loss, save gate D-32, narrow Stage 1
- Must fix from v7 post-mortem: derive `window_samples` from `EfficientATMelConfig` (not the 0.5 literal at `efficientat_trainer.py:456`); single source of truth for window length shared with `pipeline.py`; `WindowedHFDroneDataset` length assertion; runtime length-mismatch WARN in `EfficientATClassifier.predict`; move `RmsNormalize` post-resample for train/serve domain parity
- Promotion gate: execute Plan 20-06 eval harness; require real_TPR ≥ 0.80 / real_FPR ≤ 0.05 on UMA-16 hold-out before v8 replaces v6
- Hold-out: split 2026-04-08 recordings into train vs eval — no double-dipping

Plans:
- [x] 22-01-PLAN.md — Wave 0 test scaffolds + model provenance lock
- [x] 22-02-PLAN.md — window_contract.py + literal swaps (BLOCKING)
- [x] 22-03-PLAN.md — length assertion + runtime WARN + RmsNormalize parity + dataset generalization
- [ ] 22-04-PLAN.md — data integrity preflight + frozen holdout manifest
- [ ] 22-05-PLAN.md — Kaggle DroneAudioDataset investigation + ingest/reject decision
- [ ] 22-06-PLAN.md — ConcatDataset(DADS+field) + fine-tune from v6 + Vertex v8 submit path + Dockerfile:v2
- [ ] 22-07-PLAN.md — eval harness (uma16_eval + promotion) + promote_efficientat.py CLI
- [ ] 22-08-PLAN.md — Vertex L4 us-east1 training run + v8 checkpoint + sha256 sidecar
- [ ] 22-09-PLAN.md — D-27 promotion gate execution + v8 operational swap

### Phase 23: Evaluate AUDRON multi-branch hybrid architecture (MFCC+STFT-CNN+BiLSTM+Autoencoder fusion) as alternative or complement to EfficientAT classifier line

**Goal:** Research-first evaluation — deliver adopt/reject/hybrid decision on AUDRON architecture with evidence, comparison table, and revisit conditions. NOT a trained model.
**Requirements**: RES-01 (research recommendation), RES-02 (comparison table), RES-03 (follow-on phases), RES-04 (reject documentation), RES-05 (source verification)
**Depends on:** Phase 22
**Plans:** 1/1 plans complete

- [x] 23-01-PLAN.md — Verify sources + produce DECISION.md (adopt/reject/hybrid record)

