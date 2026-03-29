# Sky Fort Acoustic Service

## What This Is

A standalone, Dockerized Python microservice that performs real-time acoustic drone detection and tracking using a UMA-16v2 (16-channel) microphone array. It replaces the scattered POC code with a clean, single-responsibility service that does beamforming, CNN-based drone classification, Doppler speed estimation, and publishes tracking events over ZeroMQ — plus a web interface for live monitoring, recording, and model training.

## Core Value

Reliably detect and classify drones acoustically in real time, publishing target events (ID, class, speed, bearing) over ZeroMQ so downstream systems can act on them.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Real-time 16-channel audio capture from UMA-16v2 mic array
- [ ] Beamforming processing to produce spatial sound map (beamforming map)
- [ ] CNN-based drone classification (drone type identification from audio)
- [ ] Doppler-based speed estimation for detected targets
- [ ] Pan and tilt degree calculation for detected targets
- [ ] ZeroMQ event publishing: initial detection event (target ID + drone class)
- [ ] ZeroMQ periodic updates: speed, pan degree, tilt degree per target
- [ ] REST API endpoint serving the current beamforming map
- [ ] Standalone web UI: live beamforming map visualization
- [ ] Web UI: display speed, pan/tilt, class, and target ID for active targets
- [ ] Web UI: record raw 16-channel audio for testing
- [ ] Web UI: play back recordings through full processing pipeline (simulate live)
- [ ] Web UI: attach metadata to recordings (labels, notes, conditions)
- [ ] CNN training pipeline using labeled recordings
- [ ] Dockerized deployment (single container, independent service)

### Out of Scope

- PTZ camera control — separate service responsibility (exists in POC, not this service)
- Visual/YOLO drone detection — this is acoustic-only
- Stereo camera triangulation — separate concern from POC
- BLE scanning — unrelated POC tool
- Video tagging — separate POC component
- Multi-array support — UMA-16v2 only for v1

## Context

- **Existing POC**: `POC-code/` contains a working but tangled prototype mixing acoustic processing, PTZ control, camera feeds, YOLO detection, stereo triangulation, and web GUIs into monolithic scripts
- **Key POC files to extract from**: `radar_gui_all_mics_fast_drone.py` (beamforming core), `POC_Recorder.py` (recording logic), `unified_drone_collection_web_gui.py` (web orchestration patterns)
- **Audio libraries in POC**: `sounddevice` for capture, `acoular` for beamforming, `numpy` for DSP
- **Mic array**: UMA-16v2 — 4x4 uniform rectangular array, 42mm spacing, 16 USB channels at 48kHz
- **Training data**: Existing labeled recordings in `audio-data/data/` (background, drone, other categories with JSON metadata). New recordings from the web UI will add to this
- **audio-data/ and POC-code/ are excluded from git** — large/legacy assets kept local only
- **Downstream consumers**: Other services subscribe to ZeroMQ for target tracking data

## Constraints

- **Hardware**: UMA-16v2 mic array must be accessible from Docker (USB passthrough)
- **Runtime**: Python backend, single Docker container
- **Web UI**: React app (Vite + TypeScript + Tailwind CSS) — consistent with sky-fort-dashboard
- **Messaging**: ZeroMQ for event publishing (PUB/SUB pattern)
- **Real-time**: Audio processing must keep up with 48kHz 16-channel stream
- **Deployment**: Must run independently — no dependency on other POC components

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ZeroMQ PUB/SUB for events | Fresh protocol design, no legacy format to match | — Pending |
| UMA-16v2 only (no multi-array) | Focused v1, known hardware | — Pending |
| Full pipeline replay for recordings | Recordings serve as training data and testing validation | — Pending |
| CNN training included in service | Service owns its own model lifecycle | — Pending |
| Standalone web UI (not API-only) | Self-contained monitoring without external frontend dependency | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-29 after initialization*
