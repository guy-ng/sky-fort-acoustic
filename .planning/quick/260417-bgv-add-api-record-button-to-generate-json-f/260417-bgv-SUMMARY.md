---
phase: quick
plan: 260417-bgv
subsystem: pipeline, api, web-ui
tags: [target-recording, beamforming, websocket]
dependency_graph:
  requires: [pipeline.py, routes.py, websocket.py, BeamformingControls.tsx, MonitorPage.tsx, useBfPeaksSocket.ts]
  provides: [target-recording-api, target-recording-ui, target-recording-json-output]
  affects: [ws/bf-peaks, monitor-page]
tech_stack:
  patterns: [raw-recording-mirror, websocket-state-broadcast]
key_files:
  modified:
    - src/acoustic/pipeline.py
    - src/acoustic/api/routes.py
    - src/acoustic/api/websocket.py
    - web/src/hooks/useBfPeaksSocket.ts
    - web/src/pages/MonitorPage.tsx
    - web/src/components/heatmap/BeamformingControls.tsx
decisions:
  - Amber color for target recording button to visually distinguish from red raw audio recording
  - Target samples appended at CNN cadence (every _process_cnn call with targets present)
  - JSON output mirrors raw recording pattern with timestamped frames
metrics:
  duration: ~5min
  completed: 2026-04-17
  tasks: 2
  files: 6
---

# Quick Task 260417-bgv: Add Target Recording Feature Summary

Target location recording to JSON files via UI button, following existing raw recording pattern.

## One-liner

Record Targets button in Monitor controls captures timestamped target locations (az, el, pan, tilt, confidence, class) to JSON files on disk.

## Completed Tasks

| # | Task | Commit | Key Files |
|---|------|--------|-----------|
| 1 | Add target recording logic to pipeline and API endpoints | ef19933 | pipeline.py, routes.py |
| 2 | Add target recording button to UI and wire WebSocket state | a7f0767 | websocket.py, useBfPeaksSocket.ts, MonitorPage.tsx, BeamformingControls.tsx |

## What Was Built

### Backend (pipeline.py)
- `start_target_recording()` / `stop_target_recording()` methods mirroring raw recording pattern
- `_sample_targets()` called after every CNN/tracker cycle to accumulate frames
- `target_recording_state` property for WebSocket broadcasting
- JSON output to `data/target_recordings/{YYYYMMDD_HHMMSS}.json` with full frame history

### API (routes.py)
- `POST /api/target-recording/start` -- starts recording (409 if already active)
- `POST /api/target-recording/stop` -- stops and saves JSON (404 if not active)
- `GET /api/target-recording/status` -- returns current state
- `GET /api/target-recordings` -- lists saved JSON files with metadata

### WebSocket (websocket.py)
- `target_recording` field added to `/ws/bf-peaks` payload alongside `raw_recording` and `playback`

### Frontend
- `TargetRecordingState` interface added to `useBfPeaksSocket.ts`
- `BeamformingControls.tsx` -- amber "Record Targets" button with pulsing indicator, elapsed time, and sample count
- `MonitorPage.tsx` -- wires start/stop handlers and passes state to controls

## JSON Output Format

```json
{
  "id": "20260417_143022",
  "started_at": 1745123422.123,
  "stopped_at": 1745123482.456,
  "duration_s": 60.333,
  "total_samples": 1200,
  "frames": [
    {"t": 1745123422.200, "targets": [{"id": "...", "az_deg": 12.3, "el_deg": 5.1, ...}]}
  ]
}
```

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- Backend imports: PASSED (pipeline and routes import cleanly)
- Frontend TypeScript: PASSED (tsc --noEmit with zero errors)

## Self-Check: PASSED

All 6 files found. Both commits (ef19933, a7f0767) verified in git log.
