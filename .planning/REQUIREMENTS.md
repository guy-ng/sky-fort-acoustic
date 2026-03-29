# Requirements

**Project:** Sky Fort Acoustic Service
**Version:** v1
**Defined:** 2026-03-29

## v1 Requirements

### Audio Capture

- [ ] **AUD-01**: Service captures real-time 16-channel audio from UMA-16v2 at 48kHz using callback-based streaming (not blocking)
- [ ] **AUD-02**: Audio capture runs continuously in a dedicated thread with a ring buffer for downstream consumers
- [ ] **AUD-03**: Service detects and reports UMA-16v2 device presence/absence at startup and during operation

### Beamforming

- [x] **BF-01**: Service produces a beamforming spatial map (SRP-PHAT) from 16-channel audio in real time
- [x] **BF-02**: Beamforming frequency band is configurable at runtime (default 100-2000 Hz for drone detection)
- [x] **BF-03**: Service calculates peak azimuth and elevation (pan/tilt degrees) from beamforming map
- [x] **BF-04**: Service applies adaptive noise threshold (percentile-based calibration with configurable margin) to filter false detections

### Classification

- [ ] **CLS-01**: Service runs CNN inference on audio segments to classify detected sources as drone/not-drone
- [ ] **CLS-02**: CNN classifier identifies drone type (multi-class: 5-inch, Mavic, Matrice, EvoMax, FlyCart, etc.)
- [ ] **CLS-03**: Detection uses hysteresis state machine (enter/exit thresholds with confirmation hits) to prevent flickering
- [ ] **CLS-04**: Service loads CNN model from configurable file path at startup

### Tracking

- [ ] **TRK-01**: Service assigns a unique target ID (UUID) on first detection and maintains it until target is lost (timeout)
- [ ] **TRK-02**: Service estimates target speed via Doppler frequency shift analysis
- [ ] **TRK-03**: Service publishes initial ZeroMQ detection event with target ID and drone class
- [ ] **TRK-04**: Service publishes periodic ZeroMQ update events with speed, pan degree, and tilt degree per target
- [ ] **TRK-05**: ZeroMQ uses PUB/SUB pattern with a defined message schema (JSON)

### API

- [ ] **API-01**: REST endpoint serves current beamforming map (image or JSON grid)
- [ ] **API-02**: REST endpoint serves list of active targets with current state (ID, class, speed, pan, tilt)
- [ ] **API-03**: WebSocket endpoint streams beamforming map updates to connected clients in real time

### Web UI

- [ ] **UI-01**: React app (Vite + TypeScript + Tailwind) displays live beamforming heatmap updated via WebSocket
- [ ] **UI-02**: Web UI shows active target overlay on heatmap (markers with class, speed, bearing, target ID)
- [ ] **UI-03**: Web UI displays target details panel (speed, pan/tilt degrees, class, target ID)
- [ ] **UI-04**: Web UI provides controls to start/stop recording of raw 16-channel audio
- [ ] **UI-05**: Web UI lists available recordings with metadata (label, date, duration, notes)
- [ ] **UI-06**: Web UI plays back recordings through full processing pipeline (simulates live detection)
- [ ] **UI-07**: Web UI allows attaching/editing metadata on recordings (drone type, distance, conditions, notes)
- [ ] **UI-08**: Web UI consistent with sky-fort-dashboard styling (React 19, Tailwind 4, same component patterns)

### Training

- [ ] **TRN-01**: Web UI provides interface to select labeled recordings as training dataset
- [ ] **TRN-02**: Service runs CNN training as a background subprocess (does not block live detection)
- [ ] **TRN-03**: Training produces a new model file with validation metrics (accuracy, confusion matrix)
- [ ] **TRN-04**: Web UI displays training progress and results

### Infrastructure

- [ ] **INF-01**: Service runs in a single Docker container with USB passthrough for UMA-16v2
- [ ] **INF-02**: Dockerfile uses multi-stage build (Python backend + React frontend)
- [ ] **INF-03**: Service configurable via environment variables (device, ports, model path, ZMQ endpoint, frequency band)
- [ ] **INF-04**: Service includes health check endpoint reporting device status and pipeline state

## v2 Requirements (Deferred)

- Multi-array support -- multiple UMA-16v2 arrays for triangulation
- Range estimation from multiple arrays
- Model hot-swap without service restart
- Automatic model retraining on new data
- RF detection fusion

## Out of Scope

- PTZ camera control -- separate service, publish bearing over ZMQ instead
- Visual/YOLO drone detection -- acoustic-only service
- Stereo camera triangulation -- separate POC concern
- BLE scanning -- unrelated POC tool
- Video tagging -- separate POC component
- Audio playback through speakers -- creates feedback loops with mic array
- Range estimation from single array -- single planar array cannot reliably estimate distance

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| AUD-01 | Phase 1 | Pending |
| AUD-02 | Phase 1 | Pending |
| AUD-03 | Phase 1 | Pending |
| BF-01 | Phase 1 | Complete |
| BF-02 | Phase 1 | Complete |
| BF-03 | Phase 1 | Complete |
| BF-04 | Phase 1 | Complete |
| INF-01 | Phase 1 | Pending |
| INF-02 | Phase 2 | Pending |
| INF-03 | Phase 1 | Pending |
| INF-04 | Phase 1 | Pending |
| API-01 | Phase 2 | Pending |
| API-02 | Phase 2 | Pending |
| API-03 | Phase 2 | Pending |
| UI-01 | Phase 2 | Pending |
| UI-02 | Phase 2 | Pending |
| UI-03 | Phase 2 | Pending |
| UI-08 | Phase 2 | Pending |
| CLS-01 | Phase 3 | Pending |
| CLS-02 | Phase 3 | Pending |
| CLS-03 | Phase 3 | Pending |
| CLS-04 | Phase 3 | Pending |
| TRK-01 | Phase 3 | Pending |
| TRK-02 | Phase 3 | Pending |
| TRK-03 | Phase 3 | Pending |
| TRK-04 | Phase 3 | Pending |
| TRK-05 | Phase 3 | Pending |
| UI-04 | Phase 4 | Pending |
| UI-05 | Phase 4 | Pending |
| UI-06 | Phase 4 | Pending |
| UI-07 | Phase 4 | Pending |
| TRN-01 | Phase 5 | Pending |
| TRN-02 | Phase 5 | Pending |
| TRN-03 | Phase 5 | Pending |
| TRN-04 | Phase 5 | Pending |
