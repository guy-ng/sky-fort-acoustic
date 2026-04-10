# Phase 18: Direction of Arrival and WebSocket Broadcasting - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Each detected source gets accurate pan/tilt degrees calculated from beamforming peaks, with a coordinate transform for vertical UMA-16v2 mounting. Per-target direction tracking persists and updates bearing as sources move. Direction data is broadcast to WebSocket subscribers through enriched existing endpoints.

</domain>

<decisions>
## Implementation Decisions

### Vertical Mounting Transform
- **D-01:** UMA-16v2 is mounted vertically with y-axis pointing up. The coordinate transform must map: array y-axis -> world elevation, array x-axis -> world azimuth. Broadside (array face forward) = horizontal look direction.
- **D-02:** Mounting orientation is configurable via settings (e.g., `mounting_orientation` setting with values like `vertical_y_up`, `horizontal`). Default is `vertical_y_up`. This allows future installations with different orientations without code changes.

### Multi-target Tracking
- **D-03:** Multi-peak to multi-target association uses nearest-neighbor by angular distance. For each existing target, find the closest peak. Unmatched peaks become new targets. This works well given the 15-degree minimum separation already enforced by `multi_peak.py`.
- **D-04:** The current single-target `TargetTracker` must be upgraded to support multiple simultaneous targets with independent lifecycles (create, update, lose per target).

### Direction Smoothing
- **D-05:** Claude's Discretion — choose the best smoothing approach (EMA vs raw) based on update rate and downstream PTZ needs. Document the choice and make it tunable if appropriate.

### Direction Broadcast Strategy
- **D-06:** Enrich existing endpoints — add `pan_deg` and `tilt_deg` fields to the `TargetEvent` schema and `/ws/targets` payload. No new WebSocket endpoints. Direction data flows through `/ws/events` (lifecycle events) and `/ws/targets` (periodic state).
- **D-07:** The `/ws/targets` update rate becomes configurable via settings. Default stays at 2Hz, adjustable up to the beamforming chunk rate (~6-7Hz). This lets operators tune the tradeoff between responsiveness and traffic.

### Pan/Tilt Output Convention
- **D-08:** pan=0, tilt=0 = directly in front of the array (broadside), at the horizontal plane. Pan is azimuth from broadside, tilt is elevation from horizontal.
- **D-09:** Sign convention: pan positive = target to the right (looking at array from behind), tilt positive = target above horizontal. Standard PTZ convention.
- **D-10:** A source at physical broadside center must produce pan=0, tilt=0 in the output — this is the validation criterion for DOA-02.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Beamforming & Geometry
- `src/acoustic/beamforming/geometry.py` — Mic positions and steering vector convention (az from y-axis broadside, el from xy-plane)
- `src/acoustic/beamforming/multi_peak.py` — Multi-peak detection with greedy angular separation (BF-13)
- `src/acoustic/beamforming/interpolation.py` — Parabolic sub-grid interpolation for sub-degree accuracy (BF-12)
- `src/acoustic/beamforming/srp_phat.py` — SRP-PHAT engine producing spatial power maps

### Tracking & Events
- `src/acoustic/tracking/tracker.py` — Current single-target tracker (must be upgraded to multi-target)
- `src/acoustic/tracking/schema.py` — TargetEvent schema (already has az_deg, el_deg; needs pan_deg, tilt_deg)
- `src/acoustic/tracking/events.py` — EventBroadcaster fan-out pattern (thread-safe)

### WebSocket & API
- `src/acoustic/api/websocket.py` — All WebSocket endpoints including /ws/events and /ws/targets
- `src/acoustic/pipeline.py` — BeamformingPipeline.process_chunk and _process_cnn wiring

### Requirements
- `.planning/REQUIREMENTS.md` §v4.0 — DOA-01, DOA-02, DOA-03, DIR-01, DIR-02

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `multi_peak.py:detect_multi_peak()` — Returns `list[PeakDetection]` with `az_deg`, `el_deg`, `power`, `threshold`. Direct input for DOA calculation.
- `interpolation.py:parabolic_interpolation_2d()` — Already refines peak positions to sub-degree. Called during beamforming.
- `EventBroadcaster` — Thread-safe fan-out broadcaster, already wired to `/ws/events`. Adding `pan_deg`/`tilt_deg` to `TargetEvent` flows through automatically.
- `TrackedTarget` dataclass — Already has `az_deg`, `el_deg`, `speed_mps`. Add `pan_deg`, `tilt_deg` fields.

### Established Patterns
- Pipeline thread calls `tracker.update()` with peak data, tracker emits events via broadcaster
- WebSocket endpoints poll pipeline properties (`latest_targets`, `latest_map`) at fixed intervals
- Settings are centralized in `acoustic/config.py` with env var support

### Integration Points
- `pipeline.py:_process_cnn()` calls `tracker.update()` with single peak — must change to pass multiple peaks
- `tracker.get_target_states()` returns dicts consumed by `/ws/targets` — add pan_deg/tilt_deg to output
- `geometry.py:build_steering_vectors_2d()` defines the coordinate convention — transform layer sits between this and the output

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches within the decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 18-direction-of-arrival-and-websocket-broadcasting*
*Context gathered: 2026-04-10*
