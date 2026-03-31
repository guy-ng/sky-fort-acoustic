---
phase: 03-cnn-classification-and-target-tracking
plan: 02
subsystem: tracking
tags: [tracking, events, websocket, uuid, state-machine]
dependency_graph:
  requires: [src/acoustic/config.py, src/acoustic/types.py, src/acoustic/api/models.py]
  provides: [src/acoustic/tracking/tracker.py, src/acoustic/tracking/schema.py, src/acoustic/tracking/events.py]
  affects: [src/acoustic/api/websocket.py]
tech_stack:
  added: []
  patterns: [asyncio.Queue fanout, thread-safe broadcaster, Pydantic event schema]
key_files:
  created:
    - src/acoustic/tracking/__init__.py
    - src/acoustic/tracking/tracker.py
    - src/acoustic/tracking/schema.py
    - src/acoustic/tracking/events.py
    - tests/unit/test_tracker.py
    - tests/unit/test_target_schema.py
    - tests/unit/test_events_ws.py
  modified: []
decisions:
  - "Single-target tracking for Phase 3 (multi-target is future enhancement)"
  - "WebSocket broadcast via asyncio.Queue, not ZeroMQ PUB/SUB (per D-10)"
  - "speed_mps always None -- Doppler deferred to milestone 2 (per D-07)"
metrics:
  duration: "5m 19s"
  completed: "2026-03-31"
  tasks_completed: 1
  tasks_total: 1
  test_count: 26
  test_pass: 26
---

# Phase 3 Plan 02: Target Tracker State Machine Summary

UUID-based target tracker with lifecycle management, JSON event schema, and thread-safe WebSocket event broadcaster using asyncio.Queue fanout

## What Was Built

### Target Tracker (`tracker.py`)
- `TrackedTarget` dataclass: UUID4 id, class_label, bearing (az/el), confidence, speed_mps (always None), created_at, last_seen, lost flag
- `TargetTracker` class: single active target management
  - `update()`: creates new target on first detection (UUID4), updates existing on subsequent
  - `tick()`: marks targets lost after configurable TTL (default 5s), returns lost IDs
  - `get_active_targets()`: returns non-lost targets
  - `get_target_states()`: returns TargetState-compatible dicts for API endpoints
  - Emits TargetEvent via optional EventBroadcaster on create/update/lost

### Event Schema (`schema.py`)
- `EventType` enum: NEW, UPDATE, LOST (string values "new", "update", "lost")
- `TargetEvent` Pydantic model: event, target_id, class_label, confidence, az_deg, el_deg, speed_mps (nullable), timestamp

### Event Broadcaster (`events.py`)
- `EventBroadcaster`: thread-safe fanout to asyncio.Queue subscribers
- `subscribe()` / `unsubscribe()` for WebSocket handler lifecycle
- `broadcast()` serializes TargetEvent to dict and distributes to all queues (non-blocking, drops on full)
- No ZeroMQ dependency -- pure asyncio.Queue pattern per D-10

## Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| test_tracker.py | 11 | All pass |
| test_target_schema.py | 6 | All pass |
| test_events_ws.py | 5 | All pass |
| **Total** | **26** | **All pass** |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| e4bab15 | test | Add failing tests for tracker, schema, and broadcaster (TDD RED) |
| 3a3c286 | feat | Implement tracker, schema, and broadcaster (TDD GREEN) |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

- `speed_mps` is always `None` in TrackedTarget -- intentional per D-07, Doppler deferred to milestone 2
- Tracker is single-target only -- multi-target tracking is a documented future enhancement

## Requirements Addressed

| REQ-ID | Status | Notes |
|--------|--------|-------|
| TRK-01 | Complete | UUID assigned on first detection, maintained until lost |
| TRK-02 | Deferred | speed_mps field present but null (Doppler deferred per D-07) |
| TRK-03 | Partial | Event broadcaster ready, WebSocket endpoint wiring in Plan 03 |
| TRK-04 | Partial | Update events emitted, WebSocket endpoint wiring in Plan 03 |
| TRK-05 | Complete | JSON schema with new/update/lost event types defined |

## Self-Check: PASSED

All 7 created files verified on disk. Both commits (e4bab15, 3a3c286) verified in git log.
