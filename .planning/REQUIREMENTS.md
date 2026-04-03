# Requirements

**Project:** Sky Fort Acoustic Service
**Version:** v1
**Defined:** 2026-03-29

## v1 Requirements

### Audio Capture

- [x] **AUD-01**: Service captures real-time 16-channel audio from UMA-16v2 at 48kHz using callback-based streaming (not blocking)
- [x] **AUD-02**: Audio capture runs continuously in a dedicated thread with a ring buffer for downstream consumers
- [x] **AUD-03**: Service detects and reports UMA-16v2 device presence/absence at startup and during operation

### Beamforming

- [x] **BF-01**: Service produces a beamforming spatial map (SRP-PHAT) from 16-channel audio in real time
- [x] **BF-02**: Beamforming frequency band is configurable at runtime (default 100-2000 Hz for drone detection)
- [x] **BF-03**: Service calculates peak azimuth and elevation (pan/tilt degrees) from beamforming map
- [x] **BF-04**: Service applies adaptive noise threshold (percentile-based calibration with configurable margin) to filter false detections

### Classification

- [x] **CLS-01**: Service runs CNN inference on audio segments to classify detected sources as drone/not-drone
- [x] **CLS-02**: CNN classifier identifies drone type (multi-class: 5-inch, Mavic, Matrice, EvoMax, FlyCart, etc.)
- [x] **CLS-03**: Detection uses hysteresis state machine (enter/exit thresholds with confirmation hits) to prevent flickering
- [x] **CLS-04**: Service loads CNN model from configurable file path at startup

### Tracking

- [x] **TRK-01**: Service assigns a unique target ID (UUID) on first detection and maintains it until target is lost (timeout)
- [ ] **TRK-02**: Service estimates target speed via Doppler frequency shift analysis
- [x] **TRK-03**: Service publishes initial detection event with target ID and drone class via dedicated `/ws/events` WebSocket
- [x] **TRK-04**: Service publishes periodic update events with speed, pan degree, and tilt degree per target via `/ws/events` WebSocket
- [x] **TRK-05**: Event WebSocket uses JSON message schema with defined event types (new, update, lost)

### API

- [x] **API-01**: REST endpoint serves current beamforming map (image or JSON grid)
- [x] **API-02**: REST endpoint serves list of active targets with current state (ID, class, speed, pan, tilt)
- [x] **API-03**: WebSocket endpoint streams beamforming map updates to connected clients in real time

### Web UI

- [x] **UI-01**: React app (Vite + TypeScript + Tailwind) displays live beamforming heatmap updated via WebSocket
- [x] **UI-02**: Web UI shows active target overlay on heatmap (markers with class, speed, bearing, target ID)
- [x] **UI-03**: Web UI displays target details panel (speed, pan/tilt degrees, class, target ID)
- [ ] **UI-04**: Web UI provides controls to start/stop recording of raw 16-channel audio
- [ ] **UI-05**: Web UI lists available recordings with metadata (label, date, duration, notes)
- [ ] **UI-06**: Web UI plays back recordings through full processing pipeline (simulates live detection)
- [ ] **UI-07**: Web UI allows attaching/editing metadata on recordings (drone type, distance, conditions, notes)
- [x] **UI-08**: Web UI consistent with sky-fort-dashboard styling (React 19, Tailwind 4, same component patterns)

### Training

- [x] **TRN-01**: Web UI provides interface to select labeled recordings as training dataset
- [x] **TRN-02**: Service runs CNN training as a background subprocess (does not block live detection)
- [x] **TRN-03**: Training produces a new model file with validation metrics (accuracy, confusion matrix)
- [x] **TRN-04**: Web UI displays training progress and results

### Infrastructure

- [x] **INF-01**: Service runs in a single Docker container with USB passthrough for UMA-16v2
- [ ] **INF-02**: Dockerfile uses multi-stage build (Python backend + React frontend)
- [x] **INF-03**: Service configurable via environment variables (device, ports, model path, ZMQ endpoint, frequency band)
- [x] **INF-04**: Service includes health check endpoint reporting device status and pipeline state

## v4.0 Requirements — Research-Based Beamforming & Direction Calculation

### Beamforming Engine

- [ ] **BF-10**: Beamforming operates in 500–4000 Hz frequency band respecting UMA-16v2 spatial aliasing limit at ~4083 Hz
- [ ] **BF-11**: Bandpass filter (50–4000 Hz, 4th-order Butterworth) applied per-channel before beamforming
- [ ] **BF-12**: Sub-grid parabolic interpolation refines peak DOA to sub-degree accuracy
- [ ] **BF-13**: Multi-peak detection identifies multiple simultaneous sources with configurable threshold and minimum separation
- [ ] **BF-14**: MCRA noise estimator tracks adaptive noise floor for outdoor robustness
- [ ] **BF-15**: Beamforming is wired into the live pipeline's process_chunk (replacing current stub)

### Direction of Arrival

- [ ] **DOA-01**: Pan (azimuth) and tilt (elevation) degrees calculated from beamforming peak for each detected source
- [ ] **DOA-02**: Vertical mounting coordinate transform maps array x-y plane to world azimuth/elevation correctly
- [ ] **DOA-03**: Per-target persistent direction tracking updates bearing as the source moves

### Visualization

- [ ] **VIZ-01**: Heatmap displays beamforming output using corrected 500–4000 Hz frequency band
- [ ] **VIZ-02**: Functional Beamforming with configurable ν parameter produces sidelobe-suppressed clean maps for display

### Event Publishing

- [ ] **PUB-01**: ZeroMQ PUB/SUB publishes detection events with target ID, bearing (az/el), pan, and tilt degrees
- [ ] **PUB-02**: ZeroMQ publishes periodic updates (bearing, pan, tilt) per active target at configurable rate

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
| AUD-01 | Phase 1 | Complete |
| AUD-02 | Phase 1 | Complete |
| AUD-03 | Phase 1 | Complete |
| BF-01 | Phase 1 | Complete |
| BF-02 | Phase 1 | Complete |
| BF-03 | Phase 1 | Complete |
| BF-04 | Phase 1 | Complete |
| INF-01 | Phase 1 | Complete |
| INF-02 | Phase 2 | Pending |
| INF-03 | Phase 1 | Complete |
| INF-04 | Phase 1 | Complete |
| API-01 | Phase 2 | Complete |
| API-02 | Phase 2 | Complete |
| API-03 | Phase 2 | Complete |
| UI-01 | Phase 2 | Complete |
| UI-02 | Phase 2 | Complete |
| UI-03 | Phase 2 | Complete |
| UI-08 | Phase 2 | Complete |
| CLS-01 | Phase 3 | Complete |
| CLS-02 | Phase 3 | Complete |
| CLS-03 | Phase 3 | Complete |
| CLS-04 | Phase 3 | Complete |
| TRK-01 | Phase 3 | Complete |
| TRK-02 | Phase 3 | Pending |
| TRK-03 | Phase 3 | Complete |
| TRK-04 | Phase 3 | Complete |
| TRK-05 | Phase 3 | Complete |
| UI-04 | Phase 4 | Pending |
| UI-05 | Phase 4 | Pending |
| UI-06 | Phase 4 | Pending |
| UI-07 | Phase 4 | Pending |
| TRN-01 | Phase 5 | Complete |
| TRN-02 | Phase 5 | Complete |
| TRN-03 | Phase 5 | Complete |
| TRN-04 | Phase 5 | Complete |
