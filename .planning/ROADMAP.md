# Roadmap: Sky Fort Acoustic Service

## Overview

This roadmap delivers a real-time acoustic drone detection microservice from the ground up, following the dependency chain dictated by the processing pipeline: audio capture feeds beamforming, beamforming feeds visual validation, visual validation enables CNN tuning, CNN feeds tracking and events, recordings feed training. Five phases take the project from raw 16-channel audio capture to a self-improving CNN training loop, each delivering a verifiable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Audio Capture, Beamforming, and Infrastructure** - Reliable 16-channel audio capture with real-time beamforming in a Docker container
- [ ] **Phase 2: REST API and Live Monitoring UI** - Visual feedback on beamforming output via WebSocket-driven React app
- [ ] **Phase 3: CNN Classification and Target Tracking** - Drone detection intelligence with ZeroMQ event publishing
- [ ] **Phase 4: Recording and Playback** - Capture field audio with metadata and replay through the full pipeline
- [ ] **Phase 5: CNN Training Pipeline** - In-service model training from labeled recordings

## Phase Details

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
- [ ] 01-01-PLAN.md — Project scaffolding, Docker setup, config, device detection, audio capture pipeline with ring buffer and simulator
- [ ] 01-02-PLAN.md — SRP-PHAT beamforming engine (geometry, GCC-PHAT, 2D spatial map) with peak detection and noise gate
- [ ] 01-03-PLAN.md — Integration: FastAPI app with health endpoint, beamforming pipeline wiring, end-to-end validation

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
**Plans**: TBD

Plans:
- [ ] 02-01: FastAPI REST endpoints and WebSocket streaming
- [ ] 02-02: React web UI with live beamforming heatmap and target overlay

### Phase 3: CNN Classification and Target Tracking
**Goal**: The service detects drones from audio, assigns target IDs, and publishes tracking events over ZeroMQ
**Depends on**: Phase 2 (visual validation needed for tuning)
**Requirements**: CLS-01, CLS-02, CLS-03, CLS-04, TRK-01, TRK-02, TRK-03, TRK-04, TRK-05
**Success Criteria** (what must be TRUE):
  1. When a drone is present, the CNN classifies it as drone (with type) and a target ID appears in the web UI within seconds
  2. Target ID persists across consecutive detections and disappears only after the source is lost (timeout)
  3. Detection does not flicker -- the hysteresis state machine prevents rapid on/off transitions
  4. ZeroMQ subscribers receive detection events (new target), periodic updates (speed, bearing), and lost events
  5. CNN model loads from a configurable file path at startup
**Plans**: TBD

Plans:
- [ ] 03-01: PyTorch CNN model architecture and mel-spectrogram inference pipeline
- [ ] 03-02: Target tracker state machine with ID assignment and Doppler speed estimation
- [ ] 03-03: ZeroMQ PUB/SUB event publishing and web UI integration

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

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Audio Capture, Beamforming, and Infrastructure | 0/3 | Planned | - |
| 2. REST API and Live Monitoring UI | 0/2 | Not started | - |
| 3. CNN Classification and Target Tracking | 0/3 | Not started | - |
| 4. Recording and Playback | 0/2 | Not started | - |
| 5. CNN Training Pipeline | 0/2 | Not started | - |
