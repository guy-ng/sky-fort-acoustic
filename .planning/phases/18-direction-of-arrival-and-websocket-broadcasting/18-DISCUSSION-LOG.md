# Phase 18: Direction of Arrival and WebSocket Broadcasting - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 18-direction-of-arrival-and-websocket-broadcasting
**Areas discussed:** Vertical mounting transform, Multi-target tracking, Direction broadcast strategy, Pan/tilt output convention

---

## Vertical Mounting Transform

| Option | Description | Selected |
|--------|-------------|----------|
| Vertical — y-axis points up | Array face points forward. y-axis maps to world elevation, x-axis maps to world azimuth. | ✓ |
| Vertical — x-axis points up | Array rotated 90°. x-axis maps to elevation, y-axis maps to azimuth. | |
| Horizontal — face up | Array lies flat. Current geometry works as-is. | |
| You decide | Claude picks a sensible default. | |

**User's choice:** Vertical — y-axis points up
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded vertical y-up | Simplest. One transform baked in. | |
| Configurable via settings | Add a mounting_orientation setting that selects the transform. | ✓ |
| You decide | Claude picks based on project constraints. | |

**User's choice:** Configurable via settings
**Notes:** None

---

## Multi-target Tracking

| Option | Description | Selected |
|--------|-------------|----------|
| Nearest-neighbor (Recommended) | For each existing target, find closest peak by angular distance. Simple, fast, works with 15° separation. | ✓ |
| Greedy power-first | Assign strongest peak to nearest target first. Biases toward high-power sources. | |
| You decide | Claude picks based on existing constraints. | |

**User's choice:** Nearest-neighbor
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Exponential moving average | Smooth bearing with EMA. Reduces jitter but adds lag. | |
| Raw peak positions | Pass interpolated peak directly. Most responsive. | |
| You decide | Claude picks based on update rate and PTZ needs. | ✓ |

**User's choice:** You decide (Claude's discretion)
**Notes:** None

---

## Direction Broadcast Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Enrich existing endpoints (Recommended) | Add pan_deg/tilt_deg to TargetEvent and /ws/targets. No new endpoints. | ✓ |
| New /ws/direction endpoint | Dedicated WebSocket for direction-only data at higher rate. | |
| You decide | Claude picks simplest approach. | |

**User's choice:** Enrich existing endpoints
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 2Hz | Current rate. Targets don't move fast enough for more. | |
| Match beamforming rate | Push at every beamforming cycle (~6-7Hz). | |
| Configurable rate | Add setting for direction update rate. Default 2Hz, adjustable. | ✓ |
| You decide | Claude picks based on consumer needs. | |

**User's choice:** Configurable rate
**Notes:** None

---

## Pan/Tilt Output Convention

| Option | Description | Selected |
|--------|-------------|----------|
| Broadside center | pan=0 = broadside, tilt=0 = horizontal. Pan is azimuth, tilt is elevation. | ✓ |
| North-referenced | pan=0 = magnetic/true north. Requires heading offset config. | |
| You decide | Claude picks sensible convention. | |

**User's choice:** Broadside center
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Pan+ right, Tilt+ up | Standard PTZ convention. | ✓ |
| Pan+ left, Tilt+ up | Less common, matches certain servo setups. | |
| You decide | Claude picks standard convention. | |

**User's choice:** Pan+ right, Tilt+ up
**Notes:** None

---

## Claude's Discretion

- Direction smoothing approach (EMA vs raw peak positions) — Claude will decide based on update rate and downstream PTZ needs.

## Deferred Ideas

None — discussion stayed within phase scope.
