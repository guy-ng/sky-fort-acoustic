---
phase: 03-cnn-classification-and-target-tracking
plan: 03
subsystem: integration
tags: [cnn, onnx, websocket, event-broadcasting, pipeline-integration, asyncio-queue]

# Dependency graph
requires:
  - phase: 03-cnn-classification-and-target-tracking
    plan: 01
    provides: "OnnxDroneClassifier, DetectionStateMachine, preprocess_for_cnn, CNN config fields"
  - phase: 03-cnn-classification-and-target-tracking
    plan: 02
    provides: "TargetTracker, EventBroadcaster, TargetEvent schema"
provides:
  - "CNNWorker background inference thread with drop-semantics queue"
  - "Pipeline CNN integration with mono buffer accumulation and state machine gating"
  - "/ws/events WebSocket endpoint for detection event broadcasting"
  - "Real target data in /ws/targets and /api/targets (replaces placeholders)"
  - "Graceful degradation when CNN model is missing"
affects: [04-web-ui-recording-playback]

# Tech tracking
tech-stack:
  added: []
  patterns: [background-cnn-worker-thread, mono-buffer-accumulation, call_soon_threadsafe-broadcast]

key-files:
  created:
    - src/acoustic/classification/worker.py
    - tests/integration/test_cnn_pipeline.py
    - tests/integration/test_events_endpoint.py
  modified:
    - src/acoustic/pipeline.py
    - src/acoustic/main.py
    - src/acoustic/types.py
    - src/acoustic/api/websocket.py
    - src/acoustic/api/routes.py
    - src/acoustic/tracking/events.py

key-decisions:
  - "CNNWorker uses single-slot queue with drop semantics (maxsize=1) to never block beamforming"
  - "Mono buffer accumulates 2s of audio before pushing to CNN worker"
  - "Fixed EventBroadcaster to use call_soon_threadsafe for cross-thread delivery"
  - "REST /api/targets also updated to use pipeline.latest_targets for consistency"

patterns-established:
  - "Pipeline optional components: cnn_worker, state_machine, tracker all default to None"
  - "latest_targets property with placeholder fallback when CNN is not available"
  - "call_soon_threadsafe pattern for thread-to-asyncio event delivery"

requirements-completed: [CLS-01, CLS-03, CLS-04, TRK-01, TRK-03, TRK-04, TRK-05]

# Metrics
duration: 12min
completed: 2026-03-31
---

# Phase 3 Plan 03: CNN Pipeline Integration and Event Broadcasting Summary

**Background CNN inference thread wired into beamforming pipeline with state machine gating, /ws/events broadcast endpoint, and real target data replacing placeholders**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-31T16:40:19Z
- **Completed:** 2026-03-31T16:52:00Z
- **Tasks:** 2 completed (Task 3 is checkpoint:human-verify)
- **Files modified:** 9

## Accomplishments
- CNNWorker background thread with non-blocking single-slot queue for inference
- Full pipeline integration: peak detection -> mono accumulation -> CNN inference -> state machine -> tracker
- /ws/events endpoint broadcasting new/update/lost detection events via asyncio.Queue fanout
- /ws/targets and /api/targets now serve real TargetTracker data instead of placeholder
- Graceful degradation: service starts and runs without CNN model file
- Fixed EventBroadcaster thread safety for cross-thread event delivery

## Task Commits

Each task was committed atomically:

1. **Task 1: CNN worker thread and pipeline integration** - `59aea36` (feat)
2. **Task 2: /ws/events endpoint and swap /ws/targets to real data** - `51e3caf` (feat)

## Files Created/Modified
- `src/acoustic/classification/worker.py` - CNNWorker background inference thread with ClassificationResult
- `src/acoustic/pipeline.py` - CNN integration (cnn_worker, state_machine, tracker, mono_buffer, latest_targets)
- `src/acoustic/main.py` - Lifespan wiring: EventBroadcaster, OnnxDroneClassifier, CNNWorker, TargetTracker
- `src/acoustic/types.py` - Updated placeholder_target_from_peak docstring (fallback only)
- `src/acoustic/api/websocket.py` - /ws/events endpoint, /ws/targets using pipeline.latest_targets
- `src/acoustic/api/routes.py` - /api/targets using pipeline.latest_targets
- `src/acoustic/tracking/events.py` - Fixed thread safety with call_soon_threadsafe
- `tests/integration/test_cnn_pipeline.py` - 11 tests for CNN worker and pipeline integration
- `tests/integration/test_events_endpoint.py` - 5 tests for /ws/events and /ws/targets endpoints

## Decisions Made
- CNNWorker uses maxsize=1 queue with drop semantics -- latest audio only, never blocks beamforming
- Mono buffer accumulates 2 seconds of audio at capture rate before pushing to CNN
- EventBroadcaster fixed to use loop.call_soon_threadsafe() for cross-thread delivery
- /api/targets REST endpoint also updated for consistency (Rule 2 deviation)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed EventBroadcaster thread safety**
- **Found during:** Task 2 (testing /ws/events endpoint)
- **Issue:** asyncio.Queue.put_nowait() from non-event-loop thread does not wake up await queue.get() on the event loop
- **Fix:** Added loop capture in subscribe() and call_soon_threadsafe() dispatch in broadcast()
- **Files modified:** src/acoustic/tracking/events.py
- **Verification:** Integration tests pass with cross-thread event delivery
- **Committed in:** 51e3caf (Task 2 commit)

**2. [Rule 2 - Missing Critical] Updated /api/targets REST endpoint**
- **Found during:** Task 2 (removing placeholder from API layer)
- **Issue:** /api/targets still used placeholder_target_from_peak, inconsistent with /ws/targets
- **Fix:** Replaced with pipeline.latest_targets, removed import
- **Files modified:** src/acoustic/api/routes.py
- **Verification:** Integration tests pass
- **Committed in:** 51e3caf (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes essential for correct event delivery and API consistency. No scope creep.

## Issues Encountered

- Pre-existing test failure in test_health.py::test_health_simulated_mode (device_detected assertion) -- not caused by this plan's changes, confirmed by testing on prior commit

## Known Stubs

None - all functions are fully implemented with real logic.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 3 complete: CNN classification, target tracking, and event broadcasting all wired into live pipeline
- Web UI (Phase 4) can consume /ws/events for detection event notifications
- /ws/targets serves real UUID-tracked targets with class_label and confidence
- Service works with or without CNN model file (graceful degradation)

## Self-Check: PASSED

- All 3 created files exist on disk
- Both commits found (59aea36, 51e3caf)
- 129/129 tests passing (excluding 1 pre-existing failure in test_health.py)
- Zero zmq/pyzmq references in production code
- Zero placeholder_target_from_peak references in API layer
- Imports verified: CNNWorker and EventBroadcaster importable

---
*Phase: 03-cnn-classification-and-target-tracking*
*Completed: 2026-03-31*
