---
phase: 18-direction-of-arrival-and-websocket-broadcasting
reviewed: 2026-04-10T12:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - src/acoustic/api/models.py
  - src/acoustic/api/websocket.py
  - src/acoustic/config.py
  - src/acoustic/main.py
  - src/acoustic/pipeline.py
  - src/acoustic/tracking/doa.py
  - src/acoustic/tracking/schema.py
  - src/acoustic/tracking/tracker.py
  - tests/unit/test_doa.py
  - tests/unit/test_events_ws.py
  - tests/unit/test_target_schema.py
  - tests/unit/test_tracker.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-04-10T12:00:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Phase 18 introduces Direction of Arrival (DOA) coordinate transforms, multi-target tracking with nearest-neighbor association, EMA smoothing, and WebSocket event broadcasting. The code is well-structured with clean separation of concerns (doa.py, schema.py, tracker.py, events.py). The tracker design correctly defers TTL expiry to `tick()` rather than immediately losing unmatched targets, which is the right approach for a real-time tracking system.

Four warnings were identified: a thread-safety gap in the tracker accessed from both the pipeline thread and async WebSocket readers, the `update()` single-target method not computing pan/tilt (inconsistent with `update_multi()`), stale lost targets accumulating without cleanup, and direct access to private monitor attributes in the lifecycle code. Three informational items noted.

## Warnings

### WR-01: Single-target `update()` does not compute pan/tilt

**File:** `src/acoustic/tracking/tracker.py:153-190`
**Issue:** The `update()` method (single-target fallback used when no beamforming peaks are available, called from `pipeline.py:412`) creates and updates targets without computing `pan_deg`/`tilt_deg` via `array_to_world()`. New targets created by `update()` get default `pan_deg=0.0, tilt_deg=0.0` regardless of actual az/el. Updated targets never have their pan/tilt refreshed. In contrast, `update_multi()` correctly computes and smooths pan/tilt for both new and existing targets.
**Fix:**
```python
def update(
    self,
    az_deg: float,
    el_deg: float,
    confidence: float,
    class_label: str = "drone",
) -> TrackedTarget:
    active = self._get_active_target()

    if active is None:
        pan, tilt = array_to_world(az_deg, el_deg, self._mounting)
        target = TrackedTarget(
            id=str(uuid.uuid4()),
            class_label=class_label,
            az_deg=az_deg,
            el_deg=el_deg,
            pan_deg=pan,
            tilt_deg=tilt,
            confidence=confidence,
        )
        self._targets[target.id] = target
        self._emit(EventType.NEW, target)
        return target

    # Update existing target
    raw_pan, raw_tilt = array_to_world(az_deg, el_deg, self._mounting)
    alpha = self._smoothing_alpha
    active.pan_deg = alpha * raw_pan + (1 - alpha) * active.pan_deg
    active.tilt_deg = alpha * raw_tilt + (1 - alpha) * active.tilt_deg
    active.az_deg = az_deg
    active.el_deg = el_deg
    active.confidence = confidence
    active.class_label = class_label
    active.last_seen = time.monotonic()
    self._emit(EventType.UPDATE, active)
    return active
```

### WR-02: Thread-safety gap -- tracker accessed from pipeline thread and async readers

**File:** `src/acoustic/tracking/tracker.py:72` and `src/acoustic/pipeline.py:404-413`
**Issue:** `TargetTracker._targets` is a plain dict mutated by `update_multi()`, `update()`, and `tick()` from the pipeline's background thread (via `_process_cnn`), while `get_target_states()` and `get_active_targets()` are read from the async event loop thread (via `/ws/targets` endpoint). There is no lock protecting `_targets`. Although CPython's GIL makes dict operations atomic at the bytecode level, the iteration in `get_target_states()` and mutation in `update_multi()` can interleave between bytecodes, potentially causing a `RuntimeError: dictionary changed size during iteration`.
**Fix:** Add a `threading.Lock` around reads and writes to `_targets`, or use a snapshot pattern where `get_target_states()` copies `list(self._targets.values())` under a lock before iterating.

### WR-03: Lost targets accumulate indefinitely in tracker

**File:** `src/acoustic/tracking/tracker.py:200-207`
**Issue:** `tick()` marks targets as `lost=True` but never removes them from `_targets`. Over a long-running session with many targets appearing and disappearing, the dict grows without bound. `clear()` is only called when a detection session is stopped. For a service expected to run continuously, this is a slow memory leak.
**Fix:** After marking a target as lost and emitting the LOST event, remove it from `_targets`:
```python
def tick(self) -> list[str]:
    now = time.monotonic()
    lost_ids: list[str] = []
    for target in list(self._targets.values()):
        if not target.lost and (now - target.last_seen) > self._ttl:
            target.lost = True
            lost_ids.append(target.id)
            self._emit(EventType.LOST, target)
            del self._targets[target.id]
    return lost_ids
```

### WR-04: Direct access to private DeviceMonitor attributes in lifecycle code

**File:** `src/acoustic/main.py:224-228` and `src/acoustic/main.py:296-299`
**Issue:** Both `_initial_scan_task` and `_reconnect_loop` directly set `monitor._detected`, `monitor._device_info`, `monitor._stall_disconnected`, and call `monitor._stream_aborted.clear()` and `monitor._broadcast()`. This couples the lifecycle code tightly to DeviceMonitor internals and bypasses any invariants the monitor might enforce. If DeviceMonitor's internal representation changes, these two callsites will silently break.
**Fix:** Add a public method to DeviceMonitor like `monitor.mark_connected(device_info)` that encapsulates updating internal state and broadcasting the status change.

## Info

### IN-01: `_CaptureShim` accesses private `_last_write_time` attribute

**File:** `src/acoustic/main.py:91`
**Issue:** `_CaptureShim.last_frame_time` accesses `self._producer._last_write_time`, breaking encapsulation of `SimulatedProducer`.
**Fix:** Add a `last_write_time` property to `SimulatedProducer`.

### IN-02: Unused import in websocket.py

**File:** `src/acoustic/api/websocket.py:14`
**Issue:** `RecordingManager` is imported but only used via `websocket.app.state.recording_manager`, not directly referenced elsewhere in the file. Similarly, `TrainingManager`, `TrainingProgress`, and `TrainingStatus` are used but could be imported more narrowly.
**Fix:** The import is used indirectly for type annotation at line 220. Consider using `TYPE_CHECKING` guard for cleaner separation of runtime vs type-check imports, or leave as-is since it serves as documentation.

### IN-03: `array_to_world` horizontal mount is identical to vertical

**File:** `src/acoustic/tracking/doa.py:55-58`
**Issue:** The `HORIZONTAL` mounting case returns the same `(az_deg, el_deg)` as `VERTICAL_Y_UP` with only a comment noting poor elevation accuracy. This is technically correct for the current phase (identity transform for both) but could mislead future developers into thinking horizontal mounting is fully supported when it actually needs a different coordinate swap for a physically horizontal array.
**Fix:** Add a TODO or docstring note clarifying that horizontal mounting will need a proper coordinate transform when actual horizontal deployment is planned.

---

_Reviewed: 2026-04-10T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
