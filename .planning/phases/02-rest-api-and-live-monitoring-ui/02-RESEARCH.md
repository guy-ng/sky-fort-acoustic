# Phase 2: REST API and Live Monitoring UI - Research

**Researched:** 2026-03-31
**Domain:** FastAPI REST/WebSocket + React frontend + Docker multi-stage build
**Confidence:** HIGH

## Summary

This phase adds REST endpoints and WebSocket streaming to the existing FastAPI app (from Phase 1), scaffolds a complete React web UI with live beamforming heatmap visualization, and upgrades the Dockerfile to a multi-stage build that compiles the frontend and serves it from the Python container.

The backend work is straightforward -- FastAPI natively supports WebSocket endpoints with binary data (`send_bytes`), and the existing `app.state.pipeline.latest_map` provides the data source. The frontend is a greenfield React 19 + Vite 8 + Tailwind 4 app that must match the sky-fort-dashboard's design language (dark HUD theme, Panel components, Inter/JetBrains Mono fonts, material symbols). The heatmap renders on a Canvas 2D element, receiving binary float32 ArrayBuffer frames over WebSocket at up to 6.7 fps.

**Primary recommendation:** Use FastAPI routers (APIRouter) to organize REST and WebSocket endpoints into a dedicated `api/` module, serve the built React frontend via `StaticFiles` mount with SPA fallback, and scaffold the frontend using the exact same Vite/Tailwind/TypeScript configuration as sky-fort-dashboard.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Canvas 2D rendering for the beamforming heatmap. Draw colored pixels directly on an HTML canvas element.
- **D-02:** Jet/Rainbow color scheme (blue-cyan-green-yellow-red). Classic radar/sonar look.
- **D-03:** Update rate at Claude's discretion. Pipeline produces frames at ~6.7 fps (150ms chunks).
- **D-04:** Target markers rendered on the heatmap at azimuth/elevation positions, with a compact horizontal strip below the heatmap showing target cards.
- **D-05:** Target cards show full state: ID, class label, speed, bearing (azimuth/elevation), and confidence indicator. Layout built complete with placeholder data.
- **D-06:** WebSocket sends beamforming maps as binary ArrayBuffer (raw float32 bytes). ~64KB per frame.
- **D-07:** Separate WebSocket endpoints: `/ws/heatmap` for binary beamforming map data, `/ws/targets` for JSON target state updates.
- **D-08:** Dashboard layout with header, heatmap in a card/panel, sidebar with pipeline health/stats, and target strip below the heatmap.
- **D-09:** Fixed viewport -- no scrolling. Everything fits in one screen.

### Claude's Discretion
- Heatmap update rate / throttling strategy (D-03)
- React component structure and state management approach
- Sidebar health/stats content and update frequency
- Placeholder target data generation approach
- REST endpoint response format details (image vs JSON for beamforming map)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | REST endpoint serves current beamforming map (image or JSON grid) | FastAPI route reading `app.state.pipeline.latest_map` (NumPy 2D array). Return as JSON grid with metadata or PNG via StreamingResponse. |
| API-02 | REST endpoint serves list of active targets with current state | FastAPI route returning placeholder target list. Data model defined now, real data comes Phase 3. |
| API-03 | WebSocket endpoint streams beamforming map updates in real time | FastAPI `@app.websocket("/ws/heatmap")` sending `latest_map.tobytes()` as binary frames. |
| UI-01 | React app displays live beamforming heatmap updated via WebSocket | Canvas 2D rendering, binary WebSocket consumer, jet colormap, Vite + TypeScript + Tailwind 4. |
| UI-02 | Web UI shows active target overlay on heatmap | Canvas overlay layer rendering markers at azimuth/elevation positions from `/ws/targets`. |
| UI-03 | Web UI displays target details panel | Horizontal target card strip below heatmap with ID, class, speed, bearing, confidence. |
| UI-08 | Web UI consistent with sky-fort-dashboard styling | Same React 19, Tailwind 4, dark HUD theme, Panel/Header components, Inter + JetBrains Mono fonts. |
| INF-02 | Dockerfile uses multi-stage build (Python backend + React frontend) | Stage 1: Node 22 builds React app. Stage 2: Python 3.11-slim copies dist/, serves via FastAPI StaticFiles. |

</phase_requirements>

## Standard Stack

### Backend (additions to existing)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.135 | REST + WebSocket | Already installed. Native WebSocket support via Starlette. |
| Starlette StaticFiles | (bundled) | Serve built React app | Built into Starlette, zero additional deps. |

### Frontend (new -- greenfield scaffold)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | ^19.2.4 | UI framework | Matches sky-fort-dashboard. |
| react-dom | ^19.2.4 | DOM rendering | Paired with React. |
| Vite | ^8.0.3 | Build tool | Matches sky-fort-dashboard. |
| @vitejs/plugin-react | ^6.0.1 | React HMR/JSX | Matches sky-fort-dashboard. |
| TypeScript | ~5.9 | Type safety | Matches sky-fort-dashboard (~5.9.3 latest). |
| Tailwind CSS | ^4.2.2 | Styling | Matches sky-fort-dashboard. |
| @tailwindcss/vite | ^4.2.2 | Vite plugin for Tailwind 4 | Matches sky-fort-dashboard. |
| @tanstack/react-query | ^5.95.2 | Server state / REST polling | Matches sky-fort-dashboard. Used for health/targets REST polling. |
| @fontsource/inter | ^5.2.8 | Body font | Matches sky-fort-dashboard. |
| @fontsource/jetbrains-mono | ^5.2.8 | Mono font | Matches sky-fort-dashboard. |
| material-symbols | ^0.42.3 | Icons | Matches sky-fort-dashboard. |

### Dev Dependencies (frontend)
| Library | Version | Purpose |
|---------|---------|---------|
| @types/react | ^19 | React types |
| @types/react-dom | ^19 | React DOM types |
| eslint | ^9 | Linting |
| @eslint/js | ^9 | ESLint config |
| eslint-plugin-react-hooks | ^7 | Hooks linting |
| eslint-plugin-react-refresh | ^0.5 | Fast refresh safety |
| typescript-eslint | ^8 | TS ESLint rules |

### Not Using (from CLAUDE.md stack)
| Library | Reason |
|---------|--------|
| Recharts | Not needed for heatmap -- Canvas 2D is the locked decision (D-01). No chart-type visualizations in this phase. |

**Installation (frontend scaffold):**
```bash
# From project root
npm create vite@latest web -- --template react-ts
cd web
npm install @tailwindcss/vite @tanstack/react-query @fontsource/inter @fontsource/jetbrains-mono material-symbols
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  acoustic/
    api/                    # NEW: REST and WebSocket endpoints
      __init__.py
      routes.py             # REST endpoints (beamforming map, targets)
      websocket.py          # WebSocket endpoints (/ws/heatmap, /ws/targets)
      models.py             # Pydantic response models
    main.py                 # Existing -- mount routers + StaticFiles
    pipeline.py             # Existing -- add frame event notification
    config.py               # Existing -- add WebSocket config
    types.py                # Existing -- add Target dataclass
    ...
web/                        # NEW: React frontend (Vite project)
  src/
    main.tsx
    App.tsx
    index.css               # Tailwind + HUD theme
    components/
      layout/
        DashboardLayout.tsx  # Grid layout (header, heatmap, sidebar, targets)
        Header.tsx           # Top bar with service name + status
        Panel.tsx            # Reusable panel container (matches dashboard)
        Sidebar.tsx          # Pipeline health/stats
      heatmap/
        HeatmapCanvas.tsx    # Canvas 2D beamforming visualization
        TargetOverlay.tsx    # Target markers on canvas
        ColorScale.tsx       # Jet colormap legend
      targets/
        TargetStrip.tsx      # Horizontal strip of target cards
        TargetCard.tsx       # Individual target details
    hooks/
      useHeatmapSocket.ts   # WebSocket connection for binary heatmap data
      useTargetSocket.ts    # WebSocket connection for JSON target updates
      useHealth.ts          # TanStack Query hook for /health polling
    utils/
      colormap.ts           # Jet colormap: value -> RGB
      types.ts              # Shared TypeScript types
    api/
      client.ts             # REST API client functions
  vite.config.ts
  tsconfig.json
  package.json
```

### Pattern 1: FastAPI WebSocket Binary Streaming
**What:** Push binary float32 beamforming map frames to connected clients
**When to use:** Real-time data where JSON overhead is unacceptable
**Example:**
```python
# Source: FastAPI docs + Starlette WebSocket API
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

async def heatmap_ws(websocket: WebSocket, pipeline):
    """Stream beamforming map as binary float32 frames."""
    await websocket.accept()
    last_map_id = None
    try:
        while True:
            current_map = pipeline.latest_map
            if current_map is not None and id(current_map) != last_map_id:
                last_map_id = id(current_map)
                # Send raw float32 bytes -- client knows shape from config
                await websocket.send_bytes(current_map.astype(np.float32).tobytes())
            await asyncio.sleep(0.05)  # ~20 Hz poll, throttled by map update rate
    except WebSocketDisconnect:
        pass
```

### Pattern 2: SPA Static Files with Fallback
**What:** Serve built React app from FastAPI, with index.html fallback for client-side routing
**When to use:** Single container deployment
**Example:**
```python
# Source: FastAPI StaticFiles docs + community pattern
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

# Mount static assets (JS, CSS, images)
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes (SPA fallback)."""
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
```

### Pattern 3: Canvas 2D Heatmap from Binary WebSocket
**What:** Receive binary float32 array, render as colored heatmap on canvas
**When to use:** Real-time spatial visualization
**Example:**
```typescript
// Receive binary frame and render on canvas
function renderHeatmap(
  ctx: CanvasRenderingContext2D,
  buffer: ArrayBuffer,
  width: number,  // azimuth grid points (181)
  height: number, // elevation grid points (91)
) {
  const values = new Float32Array(buffer);
  const imageData = ctx.createImageData(width, height);

  // Find min/max for normalization
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  const range = max - min || 1;

  for (let i = 0; i < values.length; i++) {
    const normalized = (values[i] - min) / range;
    const [r, g, b] = jetColormap(normalized); // 0..1 -> RGB
    imageData.data[i * 4] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}
```

### Pattern 4: Dashboard Layout (matching sky-fort-dashboard)
**What:** CSS Grid full-viewport layout with named areas
**When to use:** Fixed monitoring dashboards
**Example:**
```typescript
// Matches sky-fort-dashboard DashboardGrid pattern
<div
  className="h-screen w-screen bg-hud-bg p-1 gap-1"
  style={{
    display: 'grid',
    gridTemplateAreas: `
      "header    header    header"
      "heatmap   heatmap   sidebar"
      "targets   targets   sidebar"
    `,
    gridTemplateColumns: '1fr 1fr 300px',
    gridTemplateRows: 'auto 1fr minmax(120px, 20vh)',
  }}
>
```

### Anti-Patterns to Avoid
- **Polling REST for heatmap data:** WebSocket is the locked decision. REST endpoint is for on-demand snapshots only.
- **JSON encoding the heatmap map:** D-06 locks binary ArrayBuffer. JSON would add 3-4x overhead for float arrays.
- **Using Recharts for the heatmap:** D-01 locks Canvas 2D. Recharts is for standard charts, not custom spatial maps.
- **Blocking the event loop in WebSocket handler:** Use `asyncio.sleep()` for the poll loop, never `time.sleep()`.
- **Creating a separate server for the frontend:** INF-02 requires single container with multi-stage Docker build.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SPA routing fallback | Custom middleware | StaticFiles + catch-all route | Well-understood pattern, handles edge cases |
| Jet colormap | Custom math | Standard jet formula (4-segment piecewise linear) | The formula is well-known but easy to get wrong at segment boundaries |
| WebSocket reconnection | Custom retry logic | Simple reconnect hook with exponential backoff | Client-side reconnect is deceptively tricky (race conditions, stale closures) |
| Response models | Raw dicts | Pydantic BaseModel | Type safety, auto-docs, serialization |

## Common Pitfalls

### Pitfall 1: WebSocket Sending to Disconnected Client
**What goes wrong:** `send_bytes()` raises `WebSocketDisconnect` or `RuntimeError` if the client has disconnected between the check and the send.
**Why it happens:** Network disconnects are asynchronous. The server doesn't know the client is gone until a send fails.
**How to avoid:** Wrap the send loop in try/except for both `WebSocketDisconnect` and `RuntimeError`. Clean up any connection tracking on disconnect.
**Warning signs:** Server logs showing unhandled exceptions in WebSocket handlers.

### Pitfall 2: NumPy Array Race Condition
**What goes wrong:** Pipeline thread writes `latest_map` while WebSocket handler reads it. The ndarray reference swap is atomic (Python GIL), but if the pipeline writes to the same array in-place, data corruption occurs.
**Why it happens:** `latest_map` is reassigned in the pipeline thread.
**How to avoid:** The current pipeline code reassigns `self.latest_map = srp_map` (new array each time), which is safe -- the GIL ensures the reference swap is atomic. Do NOT change this to in-place writes. In the WebSocket handler, grab a local reference: `current_map = pipeline.latest_map`.
**Warning signs:** Garbled heatmap frames, visual tearing.

### Pitfall 3: Canvas ImageData Row Order
**What goes wrong:** The heatmap appears flipped vertically or with azimuth/elevation axes swapped.
**Why it happens:** Canvas ImageData is row-major (left-to-right, top-to-bottom). NumPy array indexing may differ from the expected visual layout. The SRP-PHAT output is (azimuth, elevation) but canvas expects (x=column, y=row).
**How to avoid:** Define the mapping explicitly: azimuth maps to X axis (columns), elevation maps to Y axis (rows, inverted so positive elevation is up). The binary frame should be sent in row-major order matching canvas expectations.
**Warning signs:** Heatmap peak appears in wrong position relative to known source direction.

### Pitfall 4: Vite Dev Server Proxy vs Production
**What goes wrong:** WebSocket connections work in dev but fail in production (or vice versa).
**Why it happens:** In dev, Vite runs on port 5173 and needs a proxy to reach FastAPI on 8000. In production, everything is served from FastAPI on one port.
**How to avoid:** Configure Vite proxy in dev (`vite.config.ts`), and use relative WebSocket URLs in the app (`ws://${window.location.host}/ws/heatmap`). This works in both modes.
**Warning signs:** `WebSocket connection failed` errors in browser console.

### Pitfall 5: Static Files Mount Order
**What goes wrong:** The SPA catch-all route intercepts API requests, or API routes shadow static files.
**Why it happens:** FastAPI evaluates routes in registration order. A catch-all `/{path:path}` registered before API routes will match `/api/map`.
**How to avoid:** Register API routes first, then mount static files last. Use an `/api` prefix for all REST endpoints to avoid collisions.
**Warning signs:** 404 on API endpoints, or HTML returned instead of JSON.

### Pitfall 6: Tailwind 4 Configuration Differences
**What goes wrong:** Tailwind classes don't apply, custom theme colors missing.
**Why it happens:** Tailwind 4 uses `@theme` directive in CSS instead of `tailwind.config.js`. The `@tailwindcss/vite` plugin replaces the PostCSS plugin approach.
**How to avoid:** Copy the `index.css` pattern from sky-fort-dashboard exactly: `@import "tailwindcss"`, then `@theme { ... }` block with HUD colors.
**Warning signs:** Unstyled components, missing custom color utilities.

## Code Examples

### Jet Colormap Implementation
```typescript
// Standard jet colormap: maps 0..1 to RGB
export function jetColormap(t: number): [number, number, number] {
  // Clamp to [0, 1]
  t = Math.max(0, Math.min(1, t));

  let r: number, g: number, b: number;

  if (t < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + t * 4;
  } else if (t < 0.375) {
    r = 0;
    g = (t - 0.125) * 4;
    b = 1;
  } else if (t < 0.625) {
    r = (t - 0.375) * 4;
    g = 1;
    b = 1 - (t - 0.375) * 4;
  } else if (t < 0.875) {
    r = 1;
    g = 1 - (t - 0.625) * 4;
    b = 0;
  } else {
    r = 1 - (t - 0.875) * 4;
    g = 0;
    b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
```

### WebSocket Reconnect Hook
```typescript
// Custom hook for binary WebSocket with auto-reconnect
import { useEffect, useRef, useCallback } from 'react';

export function useBinaryWebSocket(
  path: string,
  onMessage: (data: ArrayBuffer) => void,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number>();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${path}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        onMessage(event.data);
      }
    };

    ws.onclose = () => {
      // Reconnect after 2 seconds
      reconnectTimer.current = window.setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, [path, onMessage]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);
}
```

### Vite Config with API Proxy
```typescript
// web/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/health': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

### Multi-Stage Dockerfile
```dockerfile
# Stage 1: Build React frontend
FROM node:22-slim AS frontend-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: Python runtime with built frontend
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2-dev libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY --from=frontend-build /web/dist /app/web/dist

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "acoustic.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pydantic Response Models
```python
from pydantic import BaseModel

class TargetState(BaseModel):
    """A detected target with current state."""
    id: str
    class_label: str
    speed_mps: float | None
    az_deg: float
    el_deg: float
    confidence: float

class BeamformingMapResponse(BaseModel):
    """Beamforming map as JSON grid with metadata."""
    az_min: float
    az_max: float
    el_min: float
    el_max: float
    az_resolution: float
    el_resolution: float
    width: int   # azimuth grid points
    height: int  # elevation grid points
    data: list[list[float]]  # 2D grid [elevation][azimuth]
    peak: dict | None  # {az_deg, el_deg, power} if detected
```

### FastAPI WebSocket Test Pattern
```python
# Using Starlette TestClient for WebSocket testing
from starlette.testclient import TestClient

def test_heatmap_websocket(running_app):
    """Test binary heatmap WebSocket streams data."""
    client = TestClient(running_app)
    with client.websocket_connect("/ws/heatmap") as ws:
        data = ws.receive_bytes()
        # Verify it's a valid float32 array of expected size
        values = np.frombuffer(data, dtype=np.float32)
        expected_size = 181 * 91  # az_range * el_range grid
        assert len(values) == expected_size
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| tailwind.config.js | @theme in CSS (Tailwind 4) | 2025 | Config is in CSS, not JS. Use @tailwindcss/vite plugin. |
| PostCSS plugin | @tailwindcss/vite | Tailwind 4 (2025) | Simpler setup, no postcss.config.js needed. |
| React 18 createRoot | React 19 (no change to createRoot) | 2024 | Minor -- same API, new concurrent features. |
| Vite 5/6 | Vite 8 | 2025 | Faster builds, ESM improvements. Same config API. |
| TestClient sync | httpx AsyncClient | 2024+ | Prefer AsyncClient for async endpoint testing, but TestClient still works for WebSocket. |

## Open Questions

1. **Heatmap frame header/metadata**
   - What we know: Binary frames are raw float32 bytes. Client needs to know grid dimensions to interpret.
   - What's unclear: Should dimensions be sent once on connection (as initial JSON message) or assumed from config endpoint?
   - Recommendation: Send a JSON handshake message on WebSocket connect with `{width, height, az_min, az_max, el_min, el_max}`, then switch to binary frames. This avoids hardcoding dimensions on the client.

2. **WebSocket push vs poll model**
   - What we know: Pipeline runs in a background thread, producing new maps at ~6.7 fps. WebSocket handler runs in asyncio.
   - What's unclear: Whether to add an asyncio.Event/callback to the pipeline or poll `latest_map` in an async loop.
   - Recommendation: Poll with `asyncio.sleep(0.05)` (20 Hz poll rate). Simple, no cross-thread synchronization needed beyond the GIL-atomic reference swap. The effective frame rate is limited by the pipeline's ~150ms chunk interval anyway.

3. **Placeholder target data strategy**
   - What we know: Real CNN classification comes in Phase 3. This phase needs fake targets for UI development.
   - What's unclear: Generate from peak detection data or use completely synthetic data.
   - Recommendation: Generate one placeholder target from `latest_peak` when a peak is detected: assign a fixed UUID, "unknown" class, null speed, and the peak's azimuth/elevation. This tests the real data flow while clearly marking the data as placeholder.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js | Frontend build | Yes | v25.5.0 | -- |
| npm | Frontend deps | Yes | 11.8.0 | -- |
| Python 3.11+ | Backend | Yes (in .venv) | 3.11+ | -- |
| Docker | Multi-stage build | Platform-dependent | -- | Build frontend separately, copy dist/ manually |

**Missing dependencies with no fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio 0.24+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | GET /api/map returns beamforming map JSON | integration | `pytest tests/integration/test_api.py::test_map_endpoint -x` | No -- Wave 0 |
| API-02 | GET /api/targets returns target list JSON | integration | `pytest tests/integration/test_api.py::test_targets_endpoint -x` | No -- Wave 0 |
| API-03 | WS /ws/heatmap streams binary frames | integration | `pytest tests/integration/test_websocket.py::test_heatmap_ws -x` | No -- Wave 0 |
| UI-01 | React app renders heatmap canvas | manual | Browser check -- Canvas rendering not automatable in pytest | N/A |
| UI-02 | Target overlay renders on heatmap | manual | Browser check | N/A |
| UI-03 | Target strip shows cards | manual | Browser check | N/A |
| UI-08 | Styling matches dashboard | manual | Visual comparison | N/A |
| INF-02 | Docker multi-stage build succeeds | integration | `docker build -t sky-fort-acoustic . && docker run --rm sky-fort-acoustic python -c "from acoustic.main import app"` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/integration/test_api.py` -- covers API-01, API-02
- [ ] `tests/integration/test_websocket.py` -- covers API-03
- [ ] Frontend tests not covered by pytest (manual browser verification for UI-01, UI-02, UI-03, UI-08)

## Sources

### Primary (HIGH confidence)
- [FastAPI WebSocket docs](https://fastapi.tiangolo.com/advanced/websockets/) -- WebSocket endpoint patterns, send_bytes API
- [FastAPI Static Files docs](https://fastapi.tiangolo.com/tutorial/static-files/) -- StaticFiles mount pattern
- sky-fort-dashboard source code (`/Users/guyelisha/Projects/sky-fort-dashboard/`) -- exact package versions, Vite config, Tailwind 4 theme, component patterns (Panel, Header, DashboardGrid)
- Existing codebase: `src/acoustic/main.py`, `pipeline.py`, `config.py`, `types.py` -- current architecture and integration points

### Secondary (MEDIUM confidence)
- [FastAPI + React Docker discussion](https://github.com/fastapi/fastapi/discussions/5134) -- multi-stage build patterns
- [FastAPI Docker deployment docs](https://fastapi.tiangolo.com/deployment/docker/) -- official Dockerfile recommendations
- [FastAPI WebSocket testing patterns](https://www.getorchestra.io/guides/fast-api-testing-websockets-a-detailed-tutorial-with-python-code-examples) -- TestClient websocket_connect
- [Streaming Architecture 2026](https://jetbi.com/blog/streaming-architecture-2026-beyond-websockets/) -- binary vs JSON WebSocket patterns

### Tertiary (LOW confidence)
- npm registry version checks (verified 2026-03-31, may change)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified against npm registry and sky-fort-dashboard source
- Architecture: HIGH -- patterns taken from existing codebase + established FastAPI/React conventions
- Pitfalls: HIGH -- drawn from documented common issues and project-specific concerns (threading, array order)

**Research date:** 2026-03-31
**Valid until:** 2026-04-30 (stable libraries, no fast-moving dependencies)
