---
phase: 18-direction-of-arrival-and-websocket-broadcasting
plan: 03
subsystem: pipeline
tags: [pipeline-wiring, websocket, multi-target, pan-tilt, configurable-rate]

requires:
  - phase: 18-direction-of-arrival-and-websocket-broadcasting
    plan: 02
    provides: Multi-target TargetTracker with update_multi() and pan_deg/tilt_deg fields
provides:
  - Pipeline wired to tracker.update_multi() with multi-peak data
  - TargetState REST model enriched with pan_deg/tilt_deg
  - Configurable /ws/targets rate via ws_targets_hz setting (clamped 0.5-10 Hz)
  - TargetTracker constructed with mounting, smoothing, and association params from settings
affects: [websocket, api, pipeline]

tech-stack:
  added: []
  patterns: [configurable WebSocket poll rate read inside loop for hot-reload, multi-peak fallback to single-peak]

key-files:
  created: []
  modified: [src/acoustic/pipeline.py, src/acoustic/config.py, src/acoustic/main.py, src/acoustic/api/models.py, src/acoustic/api/websocket.py, tests/unit/test_target_schema.py, tests/unit/test_events_ws.py]

key-decisions:
  - "Read ws_targets_hz inside the WebSocket loop (not at connection time) so runtime config changes take effect immediately"
  - "Clamp ws_targets_hz to 0.5-10 Hz to prevent DoS from extreme values (T-18-06 mitigation)"
  - "Fallback to single-peak tracker.update() when no beamforming peaks available (edge case resilience)"

patterns-established:
  - "Configurable WebSocket poll rate pattern: read setting inside loop with range clamping"
  - "Multi-peak/single-peak fallback: prefer update_multi but degrade gracefully"

requirements-completed: [DIR-01, DIR-02]

duration: 3min
completed: 2026-04-10
---

# Phase 18 Plan 03: Pipeline Wiring and WebSocket Broadcasting Summary

**Pipeline wired to multi-target tracker via update_multi(), TargetState enriched with pan/tilt, /ws/targets rate configurable via ws_targets_hz (0.5-10 Hz clamped)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-10T20:11:45Z
- **Completed:** 2026-04-10T20:14:48Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Pipeline _process_cnn now passes latest_peaks to tracker.update_multi() on CONFIRMED detection with fallback to single-peak
- TargetTracker constructed in main.py with mounting orientation, association threshold, and smoothing alpha from settings
- TargetState REST model gains pan_deg and tilt_deg fields (backward-compatible defaults to 0.0)
- /ws/targets poll rate configurable via ws_targets_hz setting, read inside loop for hot-reload, clamped 0.5-10 Hz
- ws_targets_hz config setting added with ACOUSTIC_ env var prefix
- All 47 phase unit tests pass (test_doa, test_tracker, test_target_schema, test_events_ws)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire pipeline to multi-target tracker and add ws_targets_hz config** - `e18476e` (feat)
2. **Task 2: Enrich TargetState model and make /ws/targets rate configurable** - `ab8e344` (feat)

## Files Created/Modified
- `src/acoustic/pipeline.py` - _process_cnn uses update_multi with latest_peaks, fallback to single-peak
- `src/acoustic/config.py` - Added ws_targets_hz setting (default 2.0 Hz)
- `src/acoustic/main.py` - TargetTracker constructed with mounting, association_threshold_deg, smoothing_alpha
- `src/acoustic/api/models.py` - TargetState gains pan_deg and tilt_deg fields
- `src/acoustic/api/websocket.py` - /ws/targets rate reads ws_targets_hz setting with 0.5-10 Hz clamping
- `tests/unit/test_target_schema.py` - 3 new tests for pan_deg/tilt_deg in TargetEvent and TargetState
- `tests/unit/test_events_ws.py` - 1 new test for pan_deg/tilt_deg in broadcast events

## Decisions Made
- Read ws_targets_hz inside the WebSocket loop (not at connection time) so runtime config changes take effect immediately (Pitfall 5 avoided)
- Clamp ws_targets_hz to 0.5-10 Hz range to prevent DoS from extreme values (T-18-06 threat mitigation)
- Fallback to single-peak tracker.update() when no beamforming peaks are available, ensuring graceful degradation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 18 complete: DOA transform, multi-target tracker, and pipeline wiring all integrated
- All 47 phase tests green across test_doa, test_tracker, test_target_schema, test_events_ws
- Ready for Phase 19 (Functional Beamforming Visualization) or downstream consumers

## Self-Check: PASSED

---
*Phase: 18-direction-of-arrival-and-websocket-broadcasting*
*Completed: 2026-04-10*
