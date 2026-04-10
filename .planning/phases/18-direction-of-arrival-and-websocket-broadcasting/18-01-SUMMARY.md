---
phase: 18-direction-of-arrival-and-websocket-broadcasting
plan: 01
subsystem: tracking
tags: [doa, coordinate-transform, beamforming, pydantic-settings]

requires:
  - phase: 17-beamforming-engine-upgrade-and-pipeline-integration
    provides: multi-peak detection producing az_deg/el_deg in array frame
provides:
  - MountingOrientation enum for configurable array orientation
  - array_to_world() coordinate transform from array-frame to world pan/tilt
  - mounting_orientation config setting with ACOUSTIC_ env var prefix
affects: [18-02, 18-03, tracking, pipeline]

tech-stack:
  added: []
  patterns: [identity coordinate transform for vertical_y_up mounting, str enum for pydantic-settings compatibility]

key-files:
  created: [src/acoustic/tracking/doa.py, tests/unit/test_doa.py]
  modified: [src/acoustic/config.py]

key-decisions:
  - "Identity transform for vertical_y_up: array az/el convention already matches world pan/tilt convention per geometry.py analysis"
  - "Config uses str field (not enum) for mounting_orientation because pydantic-settings loads env vars as strings"

patterns-established:
  - "DOA transform pattern: thin pure-function module converting array coordinates to world coordinates"
  - "Mounting orientation as str config with enum conversion at usage site"

requirements-completed: [DOA-01, DOA-02]

duration: 2min
completed: 2026-04-10
---

# Phase 18 Plan 01: DOA Coordinate Transform Summary

**MountingOrientation enum and array_to_world() transform mapping array-frame az/el to world pan/tilt with vertical_y_up identity, plus mounting_orientation config setting**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-10T20:02:22Z
- **Completed:** 2026-04-10T20:03:52Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- Created DOA coordinate transform module with MountingOrientation enum and array_to_world() function
- Broadside identity validated: (0,0) maps to pan=0, tilt=0 (DOA-02 / D-10)
- Sign conventions match D-09: pan positive = right, tilt positive = up
- mounting_orientation config setting added with ACOUSTIC_ env var support
- All 6 unit tests pass covering identity, sign preservation, error handling

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for DOA transform** - `ea86d35` (test)
2. **Task 1 GREEN: Implement DOA transform + config** - `ec1e0e4` (feat)

## Files Created/Modified
- `src/acoustic/tracking/doa.py` - MountingOrientation enum and array_to_world() coordinate transform
- `tests/unit/test_doa.py` - 6 unit tests covering DOA-01, DOA-02 transform correctness
- `src/acoustic/config.py` - Added mounting_orientation setting (default "vertical_y_up")

## Decisions Made
- Identity transform for vertical_y_up mounting: geometry.py's az/el convention (az from y-axis broadside positive right, el from xy-plane positive up) already matches the D-08/D-09 world convention, so no rotation needed
- Config field is str (not enum) for pydantic-settings env var compatibility; tracker will convert to enum at init

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DOA transform ready for Plan 02 (multi-target tracker upgrade) to call array_to_world() when updating target bearings
- MountingOrientation enum ready for tracker constructor parameter
- Config mounting_orientation ready to be read by tracker at init

---
*Phase: 18-direction-of-arrival-and-websocket-broadcasting*
*Completed: 2026-04-10*
