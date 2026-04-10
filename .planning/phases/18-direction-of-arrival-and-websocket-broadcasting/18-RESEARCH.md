# Phase 18: Direction of Arrival and WebSocket Broadcasting - Research

**Researched:** 2026-04-10
**Domain:** DOA coordinate transforms, multi-target tracking, WebSocket broadcasting
**Confidence:** HIGH

## Summary

Phase 18 bridges beamforming peaks (from Phase 17) to usable pan/tilt degrees for PTZ control and WebSocket consumers. The core technical challenge is the **coordinate transform** from the array's internal coordinate system (azimuth from y-axis broadside, elevation from xy-plane) to world pan/tilt degrees that account for the UMA-16v2's vertical mounting orientation. The second challenge is upgrading the single-target `TargetTracker` to handle multiple simultaneous targets with nearest-neighbor peak-to-target association.

The existing codebase is well-structured for this work. `geometry.py` already defines the steering vector convention (az from y-axis, el from xy-plane), `multi_peak.py` returns `list[PeakDetection]` with `az_deg`/`el_deg`, and `EventBroadcaster` already fans out `TargetEvent` to `/ws/events` subscribers. The work is primarily: (1) add a coordinate transform module, (2) upgrade `TargetTracker` to multi-target, (3) add `pan_deg`/`tilt_deg` fields to `TrackedTarget`, `TargetEvent`, `TargetState`, and (4) make `/ws/targets` rate configurable.

**Primary recommendation:** Build a thin `doa.py` module for the coordinate transform (vertical mounting maps array az to world pan, array el to world tilt with sign adjustment), upgrade `TargetTracker` with nearest-neighbor association, and enrich existing schemas -- no new endpoints needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** UMA-16v2 is mounted vertically with y-axis pointing up. The coordinate transform must map: array y-axis -> world elevation, array x-axis -> world azimuth. Broadside (array face forward) = horizontal look direction.
- **D-02:** Mounting orientation is configurable via settings (e.g., `mounting_orientation` setting with values like `vertical_y_up`, `horizontal`). Default is `vertical_y_up`. This allows future installations with different orientations without code changes.
- **D-03:** Multi-peak to multi-target association uses nearest-neighbor by angular distance. For each existing target, find the closest peak. Unmatched peaks become new targets. This works well given the 15-degree minimum separation already enforced by `multi_peak.py`.
- **D-04:** The current single-target `TargetTracker` must be upgraded to support multiple simultaneous targets with independent lifecycles (create, update, lose per target).
- **D-05:** Claude's Discretion -- choose the best smoothing approach (EMA vs raw) based on update rate and downstream PTZ needs. Document the choice and make it tunable if appropriate.
- **D-06:** Enrich existing endpoints -- add `pan_deg` and `tilt_deg` fields to the `TargetEvent` schema and `/ws/targets` payload. No new WebSocket endpoints. Direction data flows through `/ws/events` (lifecycle events) and `/ws/targets` (periodic state).
- **D-07:** The `/ws/targets` update rate becomes configurable via settings. Default stays at 2Hz, adjustable up to the beamforming chunk rate (~6-7Hz). This lets operators tune the tradeoff between responsiveness and traffic.
- **D-08:** pan=0, tilt=0 = directly in front of the array (broadside), at the horizontal plane. Pan is azimuth from broadside, tilt is elevation from horizontal.
- **D-09:** Sign convention: pan positive = target to the right (looking at array from behind), tilt positive = target above horizontal. Standard PTZ convention.
- **D-10:** A source at physical broadside center must produce pan=0, tilt=0 in the output -- this is the validation criterion for DOA-02.

### Claude's Discretion
- **D-05:** Direction smoothing approach (EMA vs raw). See recommendation in Architecture Patterns below.

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOA-01 | Pan (azimuth) and tilt (elevation) degrees calculated from beamforming peak for each detected source | Coordinate transform module (`doa.py`) applies mounting-aware conversion from array coordinates to world pan/tilt |
| DOA-02 | Vertical mounting coordinate transform maps array x-y plane to world azimuth/elevation correctly | Transform function with configurable `mounting_orientation` setting; broadside (0,0) -> pan=0, tilt=0 validated by unit test |
| DOA-03 | Per-target persistent direction tracking updates bearing as the source moves | Multi-target `TargetTracker` with nearest-neighbor peak-to-target association and independent target lifecycles |
| DIR-01 | WebSocket /ws/events broadcasts detection events with target ID, bearing (az/el), pan, and tilt degrees | `pan_deg` and `tilt_deg` fields added to `TargetEvent` schema; flows through existing `EventBroadcaster` |
| DIR-02 | WebSocket publishes periodic direction updates (bearing, pan, tilt) per active target at configurable rate | `/ws/targets` poll rate made configurable via `ws_targets_hz` setting (default 2.0); `pan_deg`/`tilt_deg` added to target state dicts |
</phase_requirements>

## Standard Stack

No new external dependencies required. This phase uses only existing project libraries (NumPy, Pydantic, FastAPI).

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.26 | Angular distance math, coordinate transforms | Already in project; pure math operations |
| pydantic | >=2.0 | Schema extension for TargetEvent, TargetState | Already used by all API models |
| fastapi | >=0.135 | WebSocket endpoint configuration | Already the web framework |
| pydantic-settings | >=2.0 | New settings fields (mounting_orientation, ws_targets_hz) | Already used by AcousticSettings |

[VERIFIED: codebase inspection -- all libraries already present in project]

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/
  tracking/
    doa.py          # NEW: coordinate transform functions
    tracker.py      # MODIFY: multi-target upgrade
    schema.py       # MODIFY: add pan_deg, tilt_deg fields
    events.py       # UNCHANGED: EventBroadcaster already works
  api/
    models.py       # MODIFY: add pan_deg, tilt_deg to TargetState
    websocket.py    # MODIFY: configurable /ws/targets rate
  config.py         # MODIFY: add mounting_orientation, ws_targets_hz
  pipeline.py       # MODIFY: pass multi-peak list to tracker.update()
```

### Pattern 1: Coordinate Transform (DOA-01, DOA-02)

**What:** A pure-function module that converts array-frame (az, el) to world-frame (pan, tilt) given a mounting orientation.

**Why this approach:** The array's internal coordinate system (defined in `geometry.py`) uses azimuth from y-axis broadside and elevation from the xy-plane. When the array is mounted vertically with y-axis up (D-01), what the beamforming engine calls "azimuth" actually corresponds to left/right in the world frame (pan), and "elevation" corresponds to up/down (tilt). However, because the array's coordinate system already has az=0 at broadside and el=0 at horizontal, the transform for `vertical_y_up` is essentially **identity with sign conventions enforced**. [VERIFIED: geometry.py lines 69-71 -- "Az from y-axis broadside, el from xy-plane, at az=0 el=0: direction = (0,1,0)"]

**Critical insight from geometry.py:**
- `az=0, el=0` points along +y (broadside) -- this IS the "front of array" direction [VERIFIED: geometry.py line 71]
- Azimuth is measured from y-axis in the xy-plane -- positive az = positive x = rightward
- Elevation is measured from xy-plane upward -- positive el = upward

**For `vertical_y_up` mounting (D-01):**
- Array face points forward (broadside = horizontal look direction) -- CONFIRMED by D-08
- Array az (from y-axis broadside) maps directly to world pan (from forward) -- same reference
- Array el (from xy-plane) maps directly to world tilt (from horizontal) -- same reference
- Sign: D-09 says pan positive = right (looking from behind array). Array az positive = positive x = right when looking from behind. MATCH.
- Sign: D-09 says tilt positive = above horizontal. Array el positive = above xy-plane. MATCH.

**Transform for `vertical_y_up`:**
```python
pan_deg = az_deg   # array az already measures from broadside, positive right
tilt_deg = el_deg  # array el already measures from horizontal, positive up
```

**For `horizontal` mounting (future, configurable per D-02):**
The axes rotate -- array y-axis would point forward horizontally, array x-axis right, but z-axis (currently all zeros for planar array) would need to become the elevation axis. Since the planar array has zero z-baseline, elevation discrimination is poor in horizontal mount anyway (confirmed by Phase 1 decision). The horizontal transform would swap conventions:
```python
pan_deg = az_deg   # still from broadside
tilt_deg = el_deg  # limited accuracy with planar array in horizontal mount
```

[VERIFIED: codebase -- geometry.py coordinate convention, config.py settings pattern]

**Example:**
```python
# src/acoustic/tracking/doa.py

from __future__ import annotations
from enum import Enum

class MountingOrientation(str, Enum):
    VERTICAL_Y_UP = "vertical_y_up"
    HORIZONTAL = "horizontal"

def array_to_world(
    az_deg: float,
    el_deg: float,
    mounting: MountingOrientation = MountingOrientation.VERTICAL_Y_UP,
) -> tuple[float, float]:
    """Convert array-frame (az, el) to world-frame (pan, tilt).
    
    Returns (pan_deg, tilt_deg) per D-08/D-09 convention:
    - pan=0, tilt=0 = broadside center (front of array)
    - pan positive = right (looking from behind array)
    - tilt positive = above horizontal
    """
    if mounting == MountingOrientation.VERTICAL_Y_UP:
        return az_deg, el_deg
    elif mounting == MountingOrientation.HORIZONTAL:
        return az_deg, el_deg  # Same mapping; poor el accuracy noted
    else:
        raise ValueError(f"Unknown mounting orientation: {mounting}")
```
[ASSUMED: The identity transform is correct for vertical_y_up based on matching the geometry.py convention with D-08/D-09 sign conventions. Needs physical validation.]

### Pattern 2: Multi-Target Tracker Upgrade (DOA-03, D-03, D-04)

**What:** Upgrade `TargetTracker.update()` from single-peak input to multi-peak input with nearest-neighbor association.

**Current state:** `tracker.update(az_deg, el_deg, confidence, class_label)` handles one target. Called from `pipeline._process_cnn()` line 405 with single peak. [VERIFIED: tracker.py, pipeline.py]

**New interface:**
```python
def update_multi(
    self,
    peaks: list[PeakDetection],
    confidence: float,
    class_label: str = "drone",
    mounting: MountingOrientation = MountingOrientation.VERTICAL_Y_UP,
) -> list[TrackedTarget]:
    """Associate peaks to existing targets, create new targets for unmatched peaks."""
```

**Algorithm (D-03):**
1. Get all active (non-lost) targets
2. Build cost matrix: angular distance between each target and each peak
3. Greedy nearest-neighbor: for each target, find closest unmatched peak within threshold
4. Matched peaks: update target bearing, pan/tilt, confidence, last_seen
5. Unmatched peaks: create new target with UUID
6. Unmatched targets: do NOT immediately lose -- TTL handles this via `tick()`

**Angular distance:**
```python
def angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
    """Euclidean angular distance in degrees (sufficient for small angles)."""
    return math.sqrt((az1 - az2) ** 2 + (el1 - el2) ** 2)
```

**Association threshold:** Use `bf_min_separation_deg / 2` (7.5 degrees) as the max association distance. Since peaks are guaranteed >= 15 degrees apart by `multi_peak.py`, a threshold of half that ensures unambiguous association. [ASSUMED: 7.5 degree threshold is appropriate -- could be configurable]

[VERIFIED: multi_peak.py enforces min_separation_deg=15.0 between peaks]

### Pattern 3: Direction Smoothing (D-05 -- Claude's Discretion)

**Recommendation: Lightweight EMA with configurable alpha, defaulting to raw (alpha=1.0).**

**Reasoning:**
- At the beamforming chunk rate (~6-7 Hz, 150ms chunks at 48kHz), the update rate is already modest [VERIFIED: config.py chunk_seconds=0.15]
- Parabolic interpolation (BF-12) already reduces grid quantization noise [VERIFIED: interpolation.py]
- Downstream PTZ servos have their own smoothing/PID control -- over-smoothing in software adds latency
- Default alpha=1.0 means "no smoothing" (raw pass-through) -- operators can lower alpha to e.g. 0.7 for smoother tracking

**Implementation:**
```python
@dataclass
class TrackedTarget:
    # ... existing fields ...
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    _smooth_pan: float = 0.0  # internal EMA state
    _smooth_tilt: float = 0.0
```

Add `doa_smoothing_alpha` to `AcousticSettings` (default 1.0 = no smoothing).

[ASSUMED: alpha=1.0 default is appropriate for ~6-7Hz update rate with downstream PTZ smoothing]

### Pattern 4: Schema Enrichment (DIR-01, D-06)

**What:** Add `pan_deg` and `tilt_deg` to three schema touchpoints.

**TargetEvent (schema.py):**
```python
class TargetEvent(BaseModel):
    # ... existing fields ...
    pan_deg: float = 0.0   # World pan degrees (D-08, D-09)
    tilt_deg: float = 0.0  # World tilt degrees (D-08, D-09)
```
Using default 0.0 preserves backward compatibility -- old consumers that don't read these fields are unaffected. [VERIFIED: schema.py uses Pydantic BaseModel with model_dump()]

**TrackedTarget (tracker.py):** Add `pan_deg: float = 0.0` and `tilt_deg: float = 0.0` fields.

**get_target_states() output:** Add `pan_deg` and `tilt_deg` to the returned dicts.

**TargetState (api/models.py):** Add `pan_deg: float = 0.0` and `tilt_deg: float = 0.0`.

[VERIFIED: all schema locations identified from codebase inspection]

### Pattern 5: Configurable /ws/targets Rate (DIR-02, D-07)

**Current state:** `/ws/targets` hard-codes `await asyncio.sleep(0.5)` for 2 Hz. [VERIFIED: websocket.py line 142]

**Change:** Read rate from settings, clamp to valid range.

```python
# In config.py
ws_targets_hz: float = 2.0  # Default 2 Hz, max ~6-7 Hz (chunk rate)

# In websocket.py ws_targets()
settings = websocket.app.state.settings
interval = 1.0 / max(0.5, min(settings.ws_targets_hz, 10.0))  # clamp 0.5-10 Hz
await asyncio.sleep(interval)
```

[VERIFIED: websocket.py pattern for accessing settings via websocket.app.state.settings]

### Anti-Patterns to Avoid
- **Great circle distance for small angles:** Euclidean angular distance is fine for angles < 90 degrees. Haversine adds complexity for zero benefit in this domain. [ASSUMED]
- **Kalman filter for tracking:** Overkill for a 4x4 planar array at 6-7Hz. Simple nearest-neighbor with TTL is sufficient per D-03. Save Kalman for multi-array triangulation (v2 requirement).
- **New WebSocket endpoints:** D-06 explicitly says enrich existing endpoints. Do NOT create `/ws/doa` or similar.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Coordinate transform | Full rotation matrix library | Simple if/else on mounting enum | Only 2 mounting orientations; array convention already matches world convention for vertical_y_up |
| Target association | Hungarian algorithm | Greedy nearest-neighbor | With 15-degree min separation and max 5 peaks, greedy is O(n*m) with n,m <= 5. Hungarian is overkill. |
| Bearing smoothing | Custom Kalman filter | Simple EMA (single multiply) | Update rate is 6-7Hz, downstream PTZ has its own PID |
| WebSocket protocol | Custom binary frame format | JSON via existing EventBroadcaster | Direction data is a few floats per target -- JSON overhead is negligible at 2-7Hz |

## Common Pitfalls

### Pitfall 1: Sign Convention Mismatch
**What goes wrong:** Pan/tilt signs are inverted, causing PTZ to track in the wrong direction.
**Why it happens:** Different sign conventions between array frame, math convention, and PTZ convention.
**How to avoid:** D-09 is explicit: pan positive = right (looking from behind), tilt positive = up. Write a unit test: source at array az=+30 must produce pan=+30 (rightward). Source at array el=+10 must produce tilt=+10 (upward).
**Warning signs:** PTZ tracks opposite direction from actual target; bearings appear mirrored.

### Pitfall 2: Stale Target Association After Loss
**What goes wrong:** A lost target's ID gets reassigned to a new detection at a different bearing, creating a bearing jump.
**Why it happens:** Old targets linger in the dict after being marked lost, and nearest-neighbor might match a new peak to a lost target.
**How to avoid:** Only consider active (non-lost) targets during association. The `_get_active_target()` pattern already exists. [VERIFIED: tracker.py line 133]
**Warning signs:** Target IDs reappear after being reported lost; bearings jump discontinuously.

### Pitfall 3: Pipeline Thread / Tracker Thread Safety
**What goes wrong:** Race conditions when pipeline thread writes targets while WebSocket reads them.
**Why it happens:** `_process_cnn` runs in the pipeline background thread; `/ws/targets` reads `get_target_states()` from the async event loop.
**How to avoid:** `get_target_states()` returns a list of dicts (snapshot) -- this is already safe because the list is built fresh each call. The `EventBroadcaster` already handles thread safety via `call_soon_threadsafe`. Multi-target update must not leave the targets dict in an inconsistent intermediate state -- do atomic batch updates.
**Warning signs:** Intermittent missing targets, duplicate events, crash in dict iteration.

### Pitfall 4: Forgetting to Pass Peaks from Pipeline to Tracker
**What goes wrong:** Pipeline produces multi-peak list but tracker still gets single-peak input.
**Why it happens:** `pipeline._process_cnn()` line 405 calls `tracker.update(result.az_deg, result.el_deg, ...)` with CNN result bearing, not beamforming peaks.
**How to avoid:** Change `_process_cnn` to pass `self.latest_peaks` to the new `tracker.update_multi()` method when CNN state is CONFIRMED.
**Warning signs:** Only one target ever tracked despite multiple peaks visible on heatmap.

### Pitfall 5: Hardcoded Sleep in /ws/targets
**What goes wrong:** Changing the setting has no effect because sleep value is hardcoded.
**Why it happens:** Easy to add the setting but forget to wire it into the WebSocket loop.
**How to avoid:** Read setting inside the loop (not just at connection time) so hot-reload works.
**Warning signs:** UI update rate doesn't change when setting is modified.

## Code Examples

### Coordinate Transform Module
```python
# src/acoustic/tracking/doa.py
# Source: derived from geometry.py convention + D-01/D-08/D-09

from __future__ import annotations
from enum import Enum

class MountingOrientation(str, Enum):
    VERTICAL_Y_UP = "vertical_y_up"
    HORIZONTAL = "horizontal"

def array_to_world(
    az_deg: float,
    el_deg: float,
    mounting: MountingOrientation = MountingOrientation.VERTICAL_Y_UP,
) -> tuple[float, float]:
    """Convert array-frame (az, el) to world-frame (pan, tilt).

    Convention (D-08, D-09):
      pan=0, tilt=0 = broadside (front of array, horizontal plane)
      pan positive  = right (looking from behind array)
      tilt positive = above horizontal
    """
    if mounting == MountingOrientation.VERTICAL_Y_UP:
        # Array az: from y-axis broadside, positive = rightward (x+)
        # Array el: from xy-plane, positive = upward (z+)
        # Both already match world convention.
        return az_deg, el_deg
    elif mounting == MountingOrientation.HORIZONTAL:
        return az_deg, el_deg
    raise ValueError(f"Unknown mounting: {mounting}")
```

### Multi-Target Association
```python
# In tracker.py -- update_multi method skeleton
# Source: D-03, D-04 decisions

import math

def _angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
    return math.sqrt((az1 - az2) ** 2 + (el1 - el2) ** 2)

def update_multi(
    self,
    peaks: list[PeakDetection],
    confidence: float,
    class_label: str = "drone",
) -> list[TrackedTarget]:
    active = self.get_active_targets()
    matched_peak_indices: set[int] = set()
    updated: list[TrackedTarget] = []

    # Greedy nearest-neighbor: for each target, find closest peak
    for target in active:
        best_idx = -1
        best_dist = float("inf")
        for i, peak in enumerate(peaks):
            if i in matched_peak_indices:
                continue
            dist = _angular_distance(target.az_deg, target.el_deg, peak.az_deg, peak.el_deg)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist < self._association_threshold:
            matched_peak_indices.add(best_idx)
            pk = peaks[best_idx]
            # Update target bearing + pan/tilt
            target.az_deg = pk.az_deg
            target.el_deg = pk.el_deg
            target.pan_deg, target.tilt_deg = array_to_world(pk.az_deg, pk.el_deg, self._mounting)
            target.confidence = confidence
            target.last_seen = time.monotonic()
            self._emit(EventType.UPDATE, target)
            updated.append(target)

    # Unmatched peaks become new targets
    for i, peak in enumerate(peaks):
        if i not in matched_peak_indices:
            pan, tilt = array_to_world(peak.az_deg, peak.el_deg, self._mounting)
            target = TrackedTarget(
                id=str(uuid.uuid4()),
                class_label=class_label,
                az_deg=peak.az_deg,
                el_deg=peak.el_deg,
                pan_deg=pan,
                tilt_deg=tilt,
                confidence=confidence,
            )
            self._targets[target.id] = target
            self._emit(EventType.NEW, target)
            updated.append(target)

    return updated
```

### Settings Additions
```python
# In config.py -- new fields
mounting_orientation: str = "vertical_y_up"
ws_targets_hz: float = 2.0
doa_smoothing_alpha: float = 1.0  # 1.0 = no smoothing
doa_association_threshold_deg: float = 7.5  # half of bf_min_separation_deg
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-target tracker | Multi-target with association | Phase 18 (this phase) | Enables tracking multiple drones simultaneously |
| az_deg/el_deg only | pan_deg/tilt_deg added | Phase 18 (this phase) | PTZ-ready output with mounting-aware transform |
| Hardcoded /ws/targets 2Hz | Configurable rate via setting | Phase 18 (this phase) | Operators tune responsiveness vs traffic |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Identity transform is correct for vertical_y_up mounting (array az/el already match world pan/tilt conventions) | Architecture Pattern 1 | Pan/tilt output would be wrong -- need physical validation with known sound source |
| A2 | 7.5 degree association threshold (half of min_separation) is appropriate | Architecture Pattern 2 | Too tight: lose targets on small movements. Too loose: swap target IDs. Made configurable to mitigate. |
| A3 | EMA alpha=1.0 (no smoothing) is the right default for ~6-7Hz with downstream PTZ PID | Architecture Pattern 3 | PTZ may jitter if beamforming peaks are noisy; operator can lower alpha |
| A4 | Euclidean angular distance is sufficient (no need for great circle) | Anti-Patterns | For angles within +/-90 az and +/-45 el, error is < 1%. Not a real risk. |

## Open Questions

1. **Physical broadside validation (DOA-02)**
   - What we know: The geometry.py convention says az=0, el=0 = broadside = (0,1,0)
   - What's unclear: Whether the physical array, when mounted vertically, actually produces az=0, el=0 for a source directly in front
   - Recommendation: Add an integration test with a known synthetic signal at broadside; defer physical validation to deployment

2. **Backward compatibility of tracker.update()**
   - What we know: `_process_cnn` calls `tracker.update()` with single peak
   - What's unclear: Whether to keep the old single-peak `update()` alongside new `update_multi()`, or replace it
   - Recommendation: Keep `update()` for backward compatibility, add `update_multi()` as the new primary method. Pipeline code switches to `update_multi()`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/unit/test_tracker.py tests/unit/test_target_schema.py tests/unit/test_events_ws.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOA-01 | Pan/tilt calculated from beamforming peak per source | unit | `python -m pytest tests/unit/test_doa.py -x` | Wave 0 |
| DOA-02 | Vertical mounting transform: broadside -> pan=0, tilt=0 | unit | `python -m pytest tests/unit/test_doa.py::test_broadside_identity -x` | Wave 0 |
| DOA-03 | Multi-target persistent direction tracking | unit | `python -m pytest tests/unit/test_tracker.py -x` | Exists (needs extension) |
| DIR-01 | /ws/events broadcasts pan_deg, tilt_deg in events | unit | `python -m pytest tests/unit/test_events_ws.py tests/unit/test_target_schema.py -x` | Exists (needs extension) |
| DIR-02 | /ws/targets configurable rate with pan/tilt fields | unit+integration | `python -m pytest tests/unit/test_tracker.py::TestTrackerState -x` | Exists (needs extension) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/test_doa.py tests/unit/test_tracker.py tests/unit/test_target_schema.py tests/unit/test_events_ws.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_doa.py` -- covers DOA-01, DOA-02 (coordinate transform unit tests)
- [ ] Extend `tests/unit/test_tracker.py` -- covers DOA-03 (multi-target association tests)
- [ ] Extend `tests/unit/test_target_schema.py` -- covers DIR-01 (pan_deg/tilt_deg in schema)
- [ ] Extend `tests/unit/test_events_ws.py` -- covers DIR-01 (pan_deg/tilt_deg in broadcast events)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A -- internal service |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A -- no new endpoints |
| V5 Input Validation | yes | Pydantic models validate all input; mounting_orientation via Enum; ws_targets_hz clamped to valid range |
| V6 Cryptography | no | N/A |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| WebSocket message flooding | DoS | Existing asyncio.Queue maxsize=256 with drop semantics [VERIFIED: events.py line 37] |
| Invalid mounting_orientation setting | Tampering | Enum validation in Pydantic; ValueError on unknown values |
| Extreme ws_targets_hz value | DoS | Clamp to 0.5-10 Hz range in WebSocket loop |

## Sources

### Primary (HIGH confidence)
- `src/acoustic/beamforming/geometry.py` -- coordinate convention, mic positions, steering vectors
- `src/acoustic/tracking/tracker.py` -- current single-target tracker implementation
- `src/acoustic/tracking/schema.py` -- TargetEvent Pydantic schema
- `src/acoustic/tracking/events.py` -- EventBroadcaster thread-safe fan-out
- `src/acoustic/api/websocket.py` -- all WebSocket endpoints, /ws/targets 2Hz polling
- `src/acoustic/api/models.py` -- TargetState REST model
- `src/acoustic/pipeline.py` -- process_chunk, _process_cnn integration points
- `src/acoustic/config.py` -- AcousticSettings with env var pattern
- `src/acoustic/beamforming/multi_peak.py` -- multi-peak detection, PeakDetection type
- `src/acoustic/beamforming/interpolation.py` -- parabolic sub-grid refinement

### Secondary (MEDIUM confidence)
- Phase 18 CONTEXT.md -- all locked decisions (D-01 through D-10)
- Existing test files -- test_tracker.py, test_target_schema.py, test_events_ws.py

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing
- Architecture: HIGH -- straightforward transforms and schema additions; all integration points identified
- Coordinate transform: MEDIUM -- identity transform logic is sound based on geometry.py analysis, but needs physical validation (A1)
- Pitfalls: HIGH -- identified from direct codebase inspection

**Research date:** 2026-04-10
**Valid until:** 2026-05-10 (stable domain, no external dependency changes)
