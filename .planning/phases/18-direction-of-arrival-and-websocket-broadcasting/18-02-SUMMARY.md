---
phase: 18-direction-of-arrival-and-websocket-broadcasting
plan: 02
subsystem: tracking
tags: [multi-target, tracker, nearest-neighbor, association, ema-smoothing, pan-tilt]

requires:
  - phase: 18-direction-of-arrival-and-websocket-broadcasting
    plan: 01
    provides: MountingOrientation enum and array_to_world() coordinate transform
provides:
  - Multi-target TargetTracker with update_multi() and nearest-neighbor association
  - TrackedTarget with pan_deg/tilt_deg fields populated via array_to_world
  - EMA direction smoothing with configurable alpha
  - TargetEvent schema enriched with pan_deg/tilt_deg
affects: [18-03, pipeline, websocket]

tech-stack:
  added: []
  patterns: [greedy nearest-neighbor association, EMA direction smoothing, angular distance metric]

key-files:
  created: []
  modified: [src/acoustic/tracking/tracker.py, src/acoustic/tracking/schema.py, src/acoustic/config.py, tests/unit/test_tracker.py]

key-decisions:
  - "EMA smoothing with alpha=1.0 default (no smoothing) per D-05 — downstream PTZ has its own PID, over-smoothing adds latency"
  - "Greedy nearest-neighbor association (not Hungarian) — with max 5 peaks and 15-deg min separation, greedy is sufficient"
  - "PeakDetection imported from acoustic.types (not acoustic.beamforming.multi_peak) — corrected from plan to match actual codebase"
  - "Association threshold 7.5 degrees (half of bf_min_separation_deg) ensures unambiguous matching"

patterns-established:
  - "update_multi() as the multi-peak entry point, keeping old update() for backward compatibility"
  - "Pan/tilt fields flow through TrackedTarget -> TargetEvent -> EventBroadcaster -> WebSocket"

requirements-completed: [DOA-03]

duration: 3min
completed: 2026-04-10
---

# Phase 18 Plan 02: Multi-Target Tracker with Nearest-Neighbor Association Summary

**Upgraded TargetTracker to multi-target with greedy nearest-neighbor peak association, pan/tilt via DOA transform, configurable EMA smoothing (alpha=1.0 default), and enriched TargetEvent schema**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-10T20:06:00Z
- **Completed:** 2026-04-10T20:08:50Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Upgraded TargetTracker with update_multi() supporting multiple simultaneous targets
- Greedy nearest-neighbor association matches existing targets to closest peaks within configurable threshold (default 7.5 degrees)
- TrackedTarget gains pan_deg/tilt_deg fields populated via array_to_world coordinate transform from Plan 01
- EMA direction smoothing with configurable alpha (1.0 = no smoothing, lower = smoother)
- TargetEvent schema enriched with pan_deg/tilt_deg (backward compatible, defaults to 0.0)
- get_target_states() output includes pan_deg and tilt_deg for WebSocket consumers
- Config settings added: doa_smoothing_alpha, doa_association_threshold_deg
- Old update() method preserved for backward compatibility
- 11 new tests covering all multi-target behaviors, all pass alongside 12 existing tests

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for multi-target tracker** - `593a889` (test)
2. **Task 1 GREEN: Implement multi-target tracker + schema + config** - `fc683c8` (feat)

## Files Created/Modified
- `src/acoustic/tracking/tracker.py` - Multi-target update_multi(), TrackedTarget pan_deg/tilt_deg, EMA smoothing, angular distance
- `src/acoustic/tracking/schema.py` - TargetEvent gains pan_deg/tilt_deg fields (backward compatible)
- `src/acoustic/config.py` - Added doa_smoothing_alpha and doa_association_threshold_deg settings
- `tests/unit/test_tracker.py` - 11 new tests in TestMultiTargetTracker class

## Decisions Made
- EMA smoothing alpha=1.0 default (raw pass-through): at 6-7Hz update rate with downstream PTZ PID control, software smoothing adds unnecessary latency. Operators can lower alpha for smoother tracking.
- Greedy nearest-neighbor over Hungarian algorithm: with max 5 peaks and 15-degree minimum separation enforced by multi_peak.py, greedy O(n*m) is sufficient and simpler.
- PeakDetection import corrected from plan's `acoustic.beamforming.multi_peak` to actual location `acoustic.types` (Rule 3 - blocking fix).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] PeakDetection import path corrected**
- **Found during:** Task 1 (RED phase)
- **Issue:** Plan specified `from acoustic.beamforming.multi_peak import PeakDetection` but the dataclass actually lives in `acoustic.types`
- **Fix:** Used correct import `from acoustic.types import PeakDetection` in both tracker.py and test file
- **Files modified:** src/acoustic/tracking/tracker.py, tests/unit/test_tracker.py

## Issues Encountered
None.

## User Setup Required
None.

## Next Phase Readiness
- Multi-target tracker ready for Plan 03 to wire into pipeline.py (_process_cnn calling update_multi)
- TargetEvent schema already has pan_deg/tilt_deg for WebSocket broadcast enrichment
- Config settings ready for pipeline to pass to TargetTracker constructor

---
*Phase: 18-direction-of-arrival-and-websocket-broadcasting*
*Completed: 2026-04-10*
