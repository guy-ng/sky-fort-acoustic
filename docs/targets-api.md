# Targets API & WebSocket

Reference for consuming acoustic target data from the Sky Fort Acoustic Service.

Base URL: `http://<host>:<port>` (default `8000`).

There are two ways to consume target data:

1. **REST** — poll `GET /api/targets` for current state (UI / debugging).
2. **WebSocket** — subscribe to streams for real-time updates.
   - `/ws/targets` — current target state snapshots at ~2 Hz (for UI).
   - `/ws/events` — detection lifecycle events (`new`, `update`, `lost`) as they occur (for downstream consumers).

## Coordinate system

Each angular field is an **`AngleDeg` object** with `val`, `min`, `max` (all in degrees):

```json
{ "val": 12.5, "min": 10.0, "max": 15.0 }
```

- `val` — centroid bearing.
- `min` / `max` — angular extent of the target (bounding bracket).

Conventions:
- `az` — azimuth. Allowed range `[-az_range, +az_range]` (default `±90`). `0` = array boresight, positive = right.
- `el` — elevation. Allowed range `[-el_range, +el_range]` (default `±45`). `0` = horizontal, positive = up.
- `speed_mps` — target speed in m/s. `null` until Doppler (TRK-02 / D-07) is implemented.
- `confidence` — classifier confidence in `[0.0, 1.0]`.
- `prob` — drone-class probability in `[0.0, 1.0]`.
- `timestamp` — monotonic seconds (event wall-clock is not implied).

---

## REST: `GET /api/targets`

Returns the current list of active targets.

**Response** `200 OK` — `application/json`, array of `TargetState`:

```json
[
  {
    "id": "c4a4f3b2-1e1b-4f37-9f9b-9b83f5d2a1de",
    "class_label": "drone",
    "speed_mps": null,
    "az": { "val": 12.5, "min": 10.0, "max": 15.0 },
    "el": { "val": 3.0,  "min": 1.0,  "max": 5.0  },
    "confidence": 0.94,
    "prob": 0.91
  }
]
```

| Field         | Type              | Range          | Notes                                           |
|---------------|-------------------|----------------|-------------------------------------------------|
| `id`          | string (UUID)     | —              | Stable per target until lost.                   |
| `class_label` | string            | —              | e.g. `drone`, `background`, `unknown`.          |
| `speed_mps`   | number \| null    | `≥ 0.0`        | `null` until Doppler lands.                     |
| `az`          | `AngleDeg`        | `[-90, +90]`   | Azimuth `{val,min,max}`. `0` = boresight.       |
| `el`          | `AngleDeg`        | `[-45, +45]`   | Elevation `{val,min,max}`. `0` = horizontal.    |
| `confidence`  | number            | `[0.0, 1.0]`   | Classifier confidence.                          |
| `prob`        | number            | `[0.0, 1.0]`   | Drone-class probability for this target.        |

Where `AngleDeg = { val: number, min: number, max: number }` and each component is bounded by the field's range above.

Empty array when no targets are active.

---

## REST: `GET /api/map`

Returns the current beamforming map as a JSON grid. Useful for debugging target bearings against the underlying SRP-PHAT power surface.

**Response** `200 OK`:

```json
{
  "az_min": -90.0, "az_max": 90.0,
  "el_min": -45.0, "el_max": 45.0,
  "az_resolution": 1.0, "el_resolution": 1.0,
  "width": 181, "height": 91,
  "data": [[...], ...],
  "peak": { "az_deg": 12.5, "el_deg": 3.0, "power": 0.87 }
}
```

`data` is row-major `[elevation][azimuth]`. `peak` is `null` when no peak was detected.

**`503 Service Unavailable`** — `{"detail": "Pipeline not ready"}` while the pipeline is still warming up.

---

## WebSocket: `/ws/targets`

UI-facing stream: current target state + pipeline health at ~2 Hz.

### Normal frame

```json
{
  "targets": [
    {
      "id": "c4a4f3b2-1e1b-4f37-9f9b-9b83f5d2a1de",
      "class_label": "drone",
      "speed_mps": null,
      "az": { "val": 12.5, "min": 10.0, "max": 15.0 },
      "el": { "val": 3.0,  "min": 1.0,  "max": 5.0  },
      "confidence": 0.94,
      "prob": 0.91
    }
  ],
  "drone_probability": 0.91,
  "detection_state": "confirmed"
}
```

- `drone_probability` — latest CNN probability (`null` if no inference yet).
- `detection_state` — hysteresis state machine (`idle`, `candidate`, `confirmed`, …) or `null`.

### Device disconnect / reconnect

```json
{ "type": "device_disconnected", "scanning": true,
  "targets": [], "drone_probability": null, "detection_state": null }
```

```json
{ "type": "device_reconnected" }
```

Clients should clear target overlays on `device_disconnected` and resume reading normal frames after `device_reconnected`.

---

## WebSocket: `/ws/events`

Lifecycle event stream intended for downstream systems (ZMQ bridges, camera slew, logging). Each message is a single `TargetEvent`.

### Message schema

```json
{
  "event": "new",
  "target_id": "c4a4f3b2-1e1b-4f37-9f9b-9b83f5d2a1de",
  "class_label": "drone",
  "confidence": 0.94,
  "az": { "val": 12.5, "min": 10.0, "max": 15.0 },
  "el": { "val": 3.0,  "min": 1.0,  "max": 5.0  },
  "speed_mps": null,
  "prob": 0.91,
  "timestamp": 184523.912
}
```

| Field         | Type              | Range          | Notes                                                  |
|---------------|-------------------|----------------|--------------------------------------------------------|
| `event`       | `"new"` \| `"update"` \| `"lost"` | — | Lifecycle event type.                                  |
| `target_id`   | string (UUID)     | —              | Same ID across `new` → `update`* → `lost`.             |
| `class_label` | string            | —              | Drone class or `background`.                           |
| `confidence`  | number            | `[0.0, 1.0]`   | Classifier confidence.                                 |
| `az`          | `AngleDeg`        | `[-90, +90]`   | Azimuth `{val,min,max}` at event time.                 |
| `el`          | `AngleDeg`        | `[-45, +45]`   | Elevation `{val,min,max}` at event time.               |
| `speed_mps`   | number \| null    | `≥ 0.0`        | Doppler speed or `null`.                               |
| `prob`        | number            | `[0.0, 1.0]`   | Drone-class probability for this target.               |
| `timestamp`   | number            | —              | Monotonic seconds.                                     |

### Event semantics

- `new` — first confirmed detection of a target. Emit once per `target_id`.
- `update` — periodic state refresh for a live target (bearing, confidence, later: speed).
- `lost` — target dropped (timeout / exit threshold). `target_id` is retired.

### Connection errors

If the event broadcaster is not available (CNN disabled / not initialized), the server closes the socket with:

- Code `1011`
- Reason `Event broadcasting not available`

Clients should reconnect with backoff when the pipeline is restarted.

---

## Usage

### Python (websockets)

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/events") as ws:
        async for raw in ws:
            ev = json.loads(raw)
            print(ev["event"], ev["target_id"], ev["az"]["val"], ev["el"]["val"])

asyncio.run(main())
```

### JavaScript (browser)

```js
const ws = new WebSocket("ws://localhost:8000/ws/targets");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "device_disconnected") { /* clear overlay */ return; }
  if (msg.type === "device_reconnected")  { return; }
  renderTargets(msg.targets);
};
```

---

## Related requirements

- `TRK-01` — unique UUID per target.
- `TRK-03` / `TRK-04` — initial + periodic events on `/ws/events`.
- `TRK-05` — JSON schema with `new` / `update` / `lost` event types.
- `API-02` — REST list of active targets.
- `DIR-01` / `DIR-02` — bearing + pan/tilt broadcast (in progress).

## Source

- Schema: `src/acoustic/tracking/schema.py`
- REST: `src/acoustic/api/routes.py`
- WebSocket: `src/acoustic/api/websocket.py`
- Response models: `src/acoustic/api/models.py`
