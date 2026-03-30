# Phase 2: REST API and Live Monitoring UI - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-31
**Phase:** 02-rest-api-and-live-monitoring-ui
**Areas discussed:** Heatmap visualization, Target overlay and details, Data transport format, UI layout and dashboard structure

---

## Heatmap Visualization

### Rendering approach

| Option | Description | Selected |
|--------|-------------|----------|
| Canvas 2D | Draw colored pixels on HTML canvas. Fast for dense grids, easy overlays. | ✓ |
| Server-rendered image | Backend converts NumPy to PNG/JPEG, sends over WebSocket. Simplest client. | |
| You decide | Claude picks based on performance and flexibility. | |

**User's choice:** Canvas 2D
**Notes:** Chosen for client-side flexibility — overlays, hover interactions, marker rendering.

### Color scheme

| Option | Description | Selected |
|--------|-------------|----------|
| Jet/Rainbow | Classic radar/sonar (blue-cyan-green-yellow-red). High contrast. | ✓ |
| Viridis | Perceptually uniform, colorblind-friendly. More modern/scientific. | |
| Inferno/Hot | Dark-to-bright (black-red-yellow-white). Dramatic look. | |
| You decide | Claude picks what fits sky-fort-dashboard theme. | |

**User's choice:** Jet/Rainbow
**Notes:** Classic radar look, appropriate for the domain.

### Update rate

| Option | Description | Selected |
|--------|-------------|----------|
| Match pipeline (~6.7 fps) | Every frame rendered. True real-time. | |
| Throttled (2-3 fps) | Smoother on weak clients, saves bandwidth. | |
| You decide | Claude picks based on responsiveness vs waste. | ✓ |

**User's choice:** You decide
**Notes:** Claude's discretion on update rate.

---

## Target Overlay and Details

### Target display approach

| Option | Description | Selected |
|--------|-------------|----------|
| Markers + side panel | Markers on heatmap, collapsible side panel with full details. | |
| Markers + hover/click popover | Markers on heatmap, tooltip/popover on interaction. No panel. | |
| Markers + bottom strip | Markers on heatmap, compact horizontal strip with target cards below. | ✓ |
| You decide | Claude picks what works best for the layout. | |

**User's choice:** Markers + bottom strip
**Notes:** Keeps heatmap prominent while showing multiple targets at a glance.

### Target card content

| Option | Description | Selected |
|--------|-------------|----------|
| Full state | ID, class, speed, bearing, confidence. Everything available. | ✓ |
| Compact essentials | ID, class, bearing only. Extras added in Phase 3. | |
| You decide | Claude picks given Phase 2 is placeholder data. | |

**User's choice:** Full state
**Notes:** Build the complete layout now; Phase 3 just swaps in real data.

---

## Data Transport Format

### Beamforming map transport

| Option | Description | Selected |
|--------|-------------|----------|
| Binary ArrayBuffer | Raw float32 bytes over WebSocket binary frames. ~64KB/frame. | ✓ |
| JSON grid | Nested JSON arrays. ~200-300KB/frame, human-readable. | |
| Base64 in JSON | Binary data in JSON envelope with metadata. Middle ground. | |
| You decide | Claude picks best tradeoff. | |

**User's choice:** Binary ArrayBuffer
**Notes:** Compact and fast for real-time at this data rate.

### Target state transport

| Option | Description | Selected |
|--------|-------------|----------|
| Same WebSocket, multiplexed | Binary for heatmap, JSON text for targets. One connection. | |
| Separate WebSockets | /ws/heatmap for binary, /ws/targets for JSON. Clean separation. | ✓ |
| You decide | Claude picks simplest approach. | |

**User's choice:** Separate WebSockets
**Notes:** Clean separation, clients subscribe to what they need.

---

## UI Layout and Dashboard Structure

### Page structure

| Option | Description | Selected |
|--------|-------------|----------|
| Heatmap dominant + minimal chrome | Full-width heatmap, thin header, target strip at bottom. | |
| Dashboard layout | Header, heatmap in card, sidebar with health/stats, target strip below. | ✓ |
| You decide | Claude picks based on sky-fort-dashboard patterns. | |

**User's choice:** Dashboard layout
**Notes:** Operational feel with system info visible.

### Scrolling behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed viewport | No scrolling. Everything fits one screen, components resize. | ✓ |
| Scrollable | Above-fold content with room to scroll for future panels. | |
| You decide | Claude picks for a monitoring tool. | |

**User's choice:** Fixed viewport
**Notes:** Classic always-on monitoring dashboard.

---

## Claude's Discretion

- Heatmap update rate / throttling strategy
- React component structure and state management
- Sidebar health/stats content
- Placeholder target data generation
- REST endpoint response format details

## Deferred Ideas

None — discussion stayed within phase scope.
