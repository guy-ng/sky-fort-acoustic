# Phase 2: REST API and Live Monitoring UI - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a web-based monitoring dashboard that shows a live beamforming heatmap and target state. This includes: FastAPI REST endpoints for on-demand beamforming map and target list, WebSocket endpoints for real-time streaming, a React web UI (Vite + TypeScript + Tailwind 4) with live heatmap and target overlay, and a multi-stage Dockerfile that builds the frontend alongside the Python backend. Target data is placeholder in this phase — real CNN/tracking comes in Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Heatmap Visualization
- **D-01:** Canvas 2D rendering for the beamforming heatmap. Draw colored pixels directly on an HTML canvas element. Enables client-side flexibility for overlays, hover interactions, and marker rendering.
- **D-02:** Jet/Rainbow color scheme (blue-cyan-green-yellow-red). Classic radar/sonar look, high contrast for spatial power maps.
- **D-03:** Update rate at Claude's discretion. Pipeline produces frames at ~6.7 fps (150ms chunks); Claude decides whether to match or throttle based on performance.

### Target Overlay and Details
- **D-04:** Target markers rendered on the heatmap at azimuth/elevation positions, with a compact horizontal strip below the heatmap showing target cards. Good for multiple simultaneous targets while keeping the heatmap dominant.
- **D-05:** Target cards show full state: ID, class label, speed, bearing (azimuth/elevation), and confidence indicator. Layout is built complete even though Phase 2 uses placeholder data — Phase 3 swaps in real values.

### Data Transport
- **D-06:** WebSocket sends beamforming maps as binary ArrayBuffer (raw float32 bytes). ~64KB per frame, fast to decode into typed arrays on the client. No JSON overhead.
- **D-07:** Separate WebSocket endpoints: `/ws/heatmap` for binary beamforming map data, `/ws/targets` for JSON target state updates. Clean separation, clients subscribe to what they need.

### UI Layout
- **D-08:** Dashboard layout with header, heatmap in a card/panel, sidebar with pipeline health/stats, and target strip below the heatmap. Operational monitoring feel with system info visible.
- **D-09:** Fixed viewport — no scrolling. Everything fits in one screen, components resize to fill available space. Classic always-on monitoring dashboard.

### Claude's Discretion
- Heatmap update rate / throttling strategy (D-03)
- React component structure and state management approach
- Sidebar health/stats content and update frequency
- Placeholder target data generation approach
- REST endpoint response format details (image vs JSON for beamforming map)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Backend Code
- `src/acoustic/main.py` — FastAPI app with lifespan, health endpoint. `app.state.pipeline` exposes `latest_map` (NumPy 2D array) and `latest_peak` (PeakDetection). Extend this with REST and WebSocket endpoints.
- `src/acoustic/pipeline.py` — BeamformingPipeline class. `latest_map` is the 2D SRP-PHAT output (azimuth x elevation). `latest_peak` has azimuth/elevation in degrees.
- `src/acoustic/types.py` — PeakDetection dataclass with azimuth, elevation, power fields.
- `src/acoustic/config.py` — AcousticSettings with grid resolution, frequency band, and other config.

### Beamforming Geometry
- `src/acoustic/beamforming/geometry.py` — UMA-16v2 mic positions (4x4 URA, 42mm spacing). Defines the coordinate system used by the spatial map.

### POC Reference
- `POC-code/PT520/PTZ/radar_gui_all_mics_fast_drone.py` — Original beamforming visualization patterns (matplotlib-based). Reference for what the heatmap should represent, not how to render it.

### Project Configuration
- `CLAUDE.md` &sect;Technology Stack — Pinned versions: React 19, Vite 8, Tailwind 4, TypeScript ~5.9, Recharts >=2.15, TanStack Query ^5. Must match sky-fort-dashboard patterns.
- `CLAUDE.md` &sect;Docker Considerations — Multi-stage build: Stage 1 Node.js (build React), Stage 2 Python (copy built frontend, serve via FastAPI).

### Phase 1 Context
- `.planning/phases/01-audio-capture-beamforming-and-infrastructure/01-CONTEXT.md` — Ring buffer architecture, audio pipeline decisions, Docker base image choices.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FastAPI app` (main.py) — Already has lifespan managing capture + pipeline. Add routers/endpoints to this app.
- `BeamformingPipeline` (pipeline.py) — `latest_map` and `latest_peak` are the data sources for REST and WebSocket endpoints.
- `AcousticSettings` (config.py) — Extend with any new config (WebSocket ports, UI settings).

### Established Patterns
- Backend uses background threads for audio capture and beamforming (not asyncio tasks).
- `app.state` holds runtime objects (settings, device_info, capture, pipeline).
- Config via pydantic Settings with environment variable overrides.

### Integration Points
- REST endpoints read from `app.state.pipeline.latest_map` and `app.state.pipeline.latest_peak`.
- WebSocket endpoints need a way to push new frames — either poll `latest_map` or add a callback/event mechanism to the pipeline.
- React frontend will be built by Vite and served as static files by FastAPI (multi-stage Docker build).
- No existing frontend — this phase scaffolds the entire React app.

</code_context>

<specifics>
## Specific Ideas

- Beamforming map is 181x91 (azimuth +/-90 at 1deg, elevation +/-45 at 1deg) — ~16K float32 values per frame.
- Binary WebSocket frame size: ~64KB (16,471 * 4 bytes). At 6.7 fps that's ~430KB/s — manageable.
- Target strip below heatmap should accommodate 1-5 simultaneous targets (typical drone detection scenario).
- Dashboard sidebar shows pipeline health similar to existing `/health` endpoint data: device status, pipeline running, overflow count, frame timing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-rest-api-and-live-monitoring-ui*
*Context gathered: 2026-03-31*
