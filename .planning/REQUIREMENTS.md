# Requirements

**Project:** Sky Fort Acoustic Service
**Version:** v2.0 — Research Classification Migration
**Defined:** 2026-04-01

## v2.0 Requirements

### Preprocessing

- [ ] **PRE-01**: Service uses a shared MelConfig with research parameters (SR=16000, N_FFT=1024, HOP=256, N_MELS=64, MAX_FRAMES=128, (S_db+80)/80 normalization)
- [ ] **PRE-02**: Classifier and Preprocessor protocols enable clean model swaps without modifying pipeline or state machine code
- [ ] **PRE-03**: Preprocessing outputs (1, 1, 128, 64) tensors from 0.5s audio segments with research normalization
- [ ] **PRE-04**: Numerical parity tests verify PyTorch preprocessing matches research TF output within atol=1e-4

### Model & Inference

- [ ] **MDL-01**: Research CNN architecture (3-layer Conv2D 32/64/128 + BN + MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid) implemented in PyTorch
- [ ] **MDL-02**: Segment aggregation splits audio into overlapping 0.5s segments and computes p_max, p_mean, p_agg with configurable weights
- [ ] **MDL-03**: CNNWorker accepts injected Classifier/Preprocessor/Aggregator via protocols; classifier factory selects implementation at startup
- [ ] **MDL-04**: State machine thresholds are re-calibratable via config for the new CNN's confidence distribution

### Training

- [ ] **TRN-01**: PyTorch training pipeline loads WAV files lazily, extracts random 0.5s segments, trains with Adam/BCE/early stopping
- [ ] **TRN-02**: Training runs as a background thread with resource isolation (nice, thread limits) without degrading live detection
- [ ] **TRN-03**: Training produces a model checkpoint and exports to deployable format on completion
- [ ] **TRN-04**: Training data augmentation applies SpecAugment and waveform augmentation during training

### Evaluation

- [ ] **EVL-01**: Evaluation harness runs classifier on labeled test folders and produces confusion matrix, precision/recall/F1, distribution stats
- [ ] **EVL-02**: Evaluation provides per-file detailed output with segment-level probabilities and aggregation scores

### API

- [ ] **API-01**: REST endpoints enable starting training, checking progress, running evaluation, and retrieving results
- [ ] **API-02**: Training progress is streamed via WebSocket for real-time UI updates

### Data Collection

- [ ] **COL-01**: User can record labeled audio clips from live UMA-16 via the web UI
- [ ] **COL-02**: User can attach metadata to recordings (drone type, distance, conditions, notes)
- [ ] **COL-03**: Recordings are auto-organized into directory structure compatible with the training pipeline

### Ensemble

- [ ] **ENS-01**: Late fusion ensemble wraps N classifiers via the Classifier protocol with accuracy-weighted soft voting
- [ ] **ENS-02**: Ensemble inference respects real-time latency budget (max 3 models for live, N models for offline evaluation)

## v1.0 Requirements (Previous Milestone)

### Audio Capture (Complete)

- [x] **AUD-01**: Service captures real-time 16-channel audio from UMA-16v2 at 48kHz using callback-based streaming
- [x] **AUD-02**: Audio capture runs continuously in a dedicated thread with a ring buffer for downstream consumers
- [x] **AUD-03**: Service detects and reports UMA-16v2 device presence/absence at startup and during operation

### Beamforming (Complete)

- [x] **BF-01**: Service produces a beamforming spatial map (SRP-PHAT) from 16-channel audio in real time
- [x] **BF-02**: Beamforming frequency band is configurable at runtime (default 100-2000 Hz for drone detection)
- [x] **BF-03**: Service calculates peak azimuth and elevation (pan/tilt degrees) from beamforming map
- [x] **BF-04**: Service applies adaptive noise threshold (percentile-based calibration with configurable margin)

### Classification (Complete — Being Replaced by v2.0)

- [x] **CLS-01**: Service runs CNN inference on audio segments to classify detected sources as drone/not-drone
- [x] **CLS-02**: CNN classifier identifies drone type (multi-class)
- [x] **CLS-03**: Detection uses hysteresis state machine to prevent flickering
- [x] **CLS-04**: Service loads CNN model from configurable file path at startup

### Tracking (Partially Complete)

- [x] **TRK-01**: Service assigns a unique target ID (UUID) on first detection and maintains it until target is lost
- [ ] **TRK-02**: Service estimates target speed via Doppler frequency shift analysis (deferred)
- [x] **TRK-03**: Service publishes initial detection event with target ID and drone class via WebSocket
- [x] **TRK-04**: Service publishes periodic update events with speed, pan degree, and tilt degree per target
- [x] **TRK-05**: Event WebSocket uses JSON message schema with defined event types

### API (Complete)

- [x] **API-01** (v1): REST endpoint serves current beamforming map
- [x] **API-02** (v1): REST endpoint serves list of active targets with current state
- [x] **API-03** (v1): WebSocket endpoint streams beamforming map updates in real time

### Web UI (Partially Complete)

- [x] **UI-01**: React app displays live beamforming heatmap updated via WebSocket
- [x] **UI-02**: Web UI shows active target overlay on heatmap
- [x] **UI-03**: Web UI displays target details panel
- [x] **UI-08**: Web UI consistent with sky-fort-dashboard styling

### Infrastructure (Partially Complete)

- [x] **INF-01**: Service runs in a single Docker container with USB passthrough for UMA-16v2
- [ ] **INF-02**: Dockerfile uses multi-stage build (Python backend + React frontend) (deferred)
- [x] **INF-03**: Service configurable via environment variables
- [x] **INF-04**: Service includes health check endpoint

## Out of Scope

- PTZ camera control — separate service, publish bearing over ZMQ instead
- Visual/YOLO drone detection — acoustic-only service
- TensorFlow/Keras preservation — port to PyTorch, do not maintain dual frameworks
- MFCC feature extraction — mel-spectrograms only (research strong CNN uses mel-specs)
- 10-model ensemble as initial architecture — start with single model, add 2-3 model ensemble later
- Hard voting — soft voting strictly dominates for binary classification
- librosa as runtime dependency — use torchaudio for mel-spectrograms
- JSON-materialized dataset — use lazy loading with PyTorch DataLoader
- 22kHz sample rate — standardized at 16kHz throughout
- Doppler speed estimation — deferred from v1.0, not in v2.0 scope

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| PRE-01 | Phase 6 | Pending |
| PRE-02 | Phase 6 | Pending |
| PRE-03 | Phase 6 | Pending |
| PRE-04 | Phase 6 | Pending |
| MDL-01 | Phase 7 | Pending |
| MDL-02 | Phase 7 | Pending |
| MDL-03 | Phase 7 | Pending |
| MDL-04 | Phase 7 | Pending |
| TRN-01 | Phase 8 | Pending |
| TRN-02 | Phase 8 | Pending |
| TRN-03 | Phase 8 | Pending |
| TRN-04 | Phase 8 | Pending |
| EVL-01 | Phase 9 | Pending |
| EVL-02 | Phase 9 | Pending |
| API-01 | Phase 9 | Pending |
| API-02 | Phase 9 | Pending |
| COL-01 | Phase 10 | Pending |
| COL-02 | Phase 10 | Pending |
| COL-03 | Phase 10 | Pending |
| ENS-01 | Phase 11 | Pending |
| ENS-02 | Phase 11 | Pending |
