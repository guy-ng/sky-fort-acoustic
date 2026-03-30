---
phase: 02-rest-api-and-live-monitoring-ui
plan: 01
subsystem: api
tags: [fastapi, websocket, pydantic, rest, beamforming, heatmap]

requires:
  - phase: 01-audio-capture-beamforming-docker
    provides: BeamformingPipeline with latest_map/latest_peak, FastAPI app with lifespan, AcousticSettings

provides:
  - REST endpoint GET /api/map returning beamforming map JSON with grid metadata
  - REST endpoint GET /api/targets returning placeholder target list
  - WebSocket /ws/heatmap streaming binary float32 heatmap frames with handshake
  - WebSocket /ws/targets streaming JSON target state at 2 Hz
  - Pydantic response models (BeamformingMapResponse, TargetState, HeatmapHandshake)
  - SPA static file serving with catch-all fallback
  - placeholder_target_from_peak() for Phase 2 stub targets

affects: [02-02, 02-03, 03-cnn-classification]

tech-stack:
  added: [starlette.staticfiles]
  patterns: [APIRouter prefix pattern, WebSocket binary streaming with JSON handshake, row-major transpose for canvas rendering]

key-files:
  created:
    - src/acoustic/api/__init__.py
    - src/acoustic/api/models.py
    - src/acoustic/api/routes.py
    - src/acoustic/api/websocket.py
    - src/acoustic/api/static.py
    - tests/integration/conftest.py
    - tests/integration/test_api.py
    - tests/integration/test_websocket.py
  modified:
    - src/acoustic/types.py
    - src/acoustic/main.py
    - tests/integration/test_health.py

key-decisions:
  - "Map data transposed to [elevation][azimuth] row-major for canvas rendering compatibility"
  - "WebSocket heatmap uses JSON handshake then binary float32 frames for efficiency"
  - "Heatmap polls at 20 Hz, targets at 2 Hz matching update frequency needs"
  - "SPA catch-all registered last to avoid shadowing API routes"

patterns-established:
  - "APIRouter with prefix for route grouping (api_router prefix=/api)"
  - "WebSocket binary protocol: JSON handshake then raw bytes for high-throughput data"
  - "Shared running_app fixture in integration conftest for lifespan-dependent tests"
  - "placeholder_target_from_peak pattern for Phase 2 stub data (replaced by CNN in Phase 3)"

requirements-completed: [API-01, API-02, API-03]

duration: 7min
completed: 2026-03-31
---

# Phase 2 Plan 1: REST and WebSocket API Endpoints Summary

**FastAPI REST endpoints for beamforming map and targets, plus WebSocket binary heatmap streaming with JSON handshake protocol**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-30T21:55:04Z
- **Completed:** 2026-03-30T22:02:57Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments
- REST endpoints /api/map and /api/targets serving live beamforming data and placeholder target state
- WebSocket /ws/heatmap streaming binary float32 frames at 20 Hz with JSON handshake for grid metadata
- WebSocket /ws/targets streaming JSON target arrays at 2 Hz
- Full integration test coverage: 9 new tests, all 71 tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Pydantic response models and TargetState type** - `1b597e7` (feat)
2. **Task 2: Create REST and WebSocket endpoints with SPA static mount** - `33a1a8c` (feat)
3. **Task 3: Create integration tests for REST and WebSocket endpoints** - `f9e411b` (test)

## Files Created/Modified
- `src/acoustic/api/__init__.py` - Package init for API module
- `src/acoustic/api/models.py` - Pydantic models: BeamformingMapResponse, TargetState, HeatmapHandshake
- `src/acoustic/api/routes.py` - REST endpoints: GET /api/map, GET /api/targets
- `src/acoustic/api/websocket.py` - WebSocket endpoints: /ws/heatmap (binary), /ws/targets (JSON)
- `src/acoustic/api/static.py` - SPA static file serving with catch-all fallback
- `src/acoustic/types.py` - Added placeholder_target_from_peak() and PLACEHOLDER_TARGET_ID
- `src/acoustic/main.py` - Wired API and WebSocket routers, added mount_static call
- `tests/integration/conftest.py` - Shared running_app fixture for integration tests
- `tests/integration/test_api.py` - 6 tests for map and targets REST endpoints
- `tests/integration/test_websocket.py` - 3 tests for heatmap and targets WebSocket endpoints
- `tests/integration/test_health.py` - Refactored to use shared conftest fixture

## Decisions Made
- Map data transposed to [elevation][azimuth] row-major order for direct canvas rendering
- WebSocket heatmap sends JSON handshake first (grid dimensions), then binary float32 frames for efficiency
- Heatmap polling at 20 Hz (50ms sleep), targets at 2 Hz (500ms) matching data change rates
- SPA catch-all route registered after all API/WS routes to prevent shadowing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- API layer complete, ready for React frontend (Plan 02) to consume these endpoints
- WebSocket binary protocol documented via HeatmapHandshake model for frontend integration
- Placeholder targets ready to be replaced by CNN classification in Phase 3

---
*Phase: 02-rest-api-and-live-monitoring-ui*
*Completed: 2026-03-31*
