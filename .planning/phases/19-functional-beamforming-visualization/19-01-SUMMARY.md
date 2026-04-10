---
phase: 19-functional-beamforming-visualization
plan: 01
subsystem: beamforming
tags: [srp-phat, functional-beamforming, sidelobe-suppression, fastapi, pydantic]

# Dependency graph
requires:
  - phase: 17-beamforming-engine-upgrade
    provides: SRP-PHAT engine, bandpass filter, MCRA noise estimation
  - phase: 18-direction-of-arrival-and-websocket-broadcasting
    provides: Pipeline with multi-peak detection and WebSocket broadcasting
provides:
  - bf_nu config field for functional beamforming exponent
  - Power-map exponent post-processing in pipeline (sidelobe suppression)
  - PATCH /api/settings endpoint for runtime parameter adjustment
  - Updated srp_phat_2d default frequency band (500-4000 Hz)
affects: [19-02, frontend-heatmap, visualization]

# Tech tracking
tech-stack:
  added: []
  patterns: [functional-beamforming-exponent, runtime-settings-patch]

key-files:
  created:
    - tests/unit/test_functional_beamforming.py
    - tests/integration/test_settings_api.py
  modified:
    - src/acoustic/config.py
    - src/acoustic/pipeline.py
    - src/acoustic/beamforming/srp_phat.py
    - src/acoustic/api/routes.py

key-decisions:
  - "Functional beamforming applied only to latest_map (visualization) — srp_map left untouched for MCRA and peak detection"
  - "SettingsUpdate model uses generic setattr loop for future extensibility"

patterns-established:
  - "Runtime settings mutation via PATCH /api/settings with Pydantic validation"
  - "Visualization-only transforms applied after beamforming, before map storage"

requirements-completed: [VIZ-01, VIZ-02]

# Metrics
duration: 4m39s
completed: 2026-04-10
---

# Phase 19 Plan 01: Functional Beamforming Backend Summary

**Functional beamforming with configurable nu exponent (default 100) for sidelobe suppression, runtime PATCH /api/settings endpoint, and 500-4000 Hz frequency band defaults**

## Performance

- **Duration:** 4m39s
- **Started:** 2026-04-10T21:16:11Z
- **Completed:** 2026-04-10T21:20:50Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added bf_nu config field (default 100.0) with ACOUSTIC_BF_NU env var support
- Pipeline applies element-wise nu-th power to normalized SRP-PHAT map, clamping values below 1e-6 to zero, producing [0,1] float32 visualization maps with clean sidelobe suppression
- Updated srp_phat_2d default frequency args from 100-2000 Hz to 500-4000 Hz per research analysis
- Added PATCH /api/settings endpoint with Pydantic validation (bf_nu: ge=1.0, le=1000.0) for runtime adjustment without service restart
- 16 tests total: 11 unit tests + 5 integration tests, all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: bf_nu config, functional beamforming pipeline transform, srp_phat_2d defaults** - `783384a` (feat)
2. **Task 2: PATCH /api/settings endpoint** - `989a77d` (feat)

## Files Created/Modified
- `src/acoustic/config.py` - Added bf_nu field with default 100.0
- `src/acoustic/pipeline.py` - Functional beamforming post-processing (normalize, exponentiate, clamp)
- `src/acoustic/beamforming/srp_phat.py` - Updated default fmin/fmax to 500/4000 Hz
- `src/acoustic/api/routes.py` - Added SettingsUpdate model and PATCH /settings endpoint
- `tests/unit/test_functional_beamforming.py` - 11 tests: config, math, defaults, edge cases
- `tests/integration/test_settings_api.py` - 5 tests: valid/invalid PATCH, persistence

## Decisions Made
- Functional beamforming transform applied only to `self.latest_map` (visualization path); `srp_map` variable left untouched for MCRA noise estimation and multi-peak detection downstream
- SettingsUpdate uses generic field iteration via `model_dump(exclude_none=True)` for easy future extension with additional runtime-configurable parameters

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing test_health.py failure (MacBook mic detected as fallback, causing device_detected=True in simulated mode) -- unrelated to this plan, not addressed

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Backend functional beamforming complete; pipeline produces [0,1] normalized maps with sidelobe suppression
- PATCH /api/settings endpoint ready for frontend integration
- Plan 19-02 (frontend visualization) can proceed: heatmap already consumes latest_map via WebSocket

---
*Phase: 19-functional-beamforming-visualization*
*Completed: 2026-04-10*
