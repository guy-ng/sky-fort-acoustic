# Sky Fort Acoustic Service

Real-time acoustic drone detection and tracking microservice using a UMA-16v2 16-channel microphone array.

## Running Locally

### Backend

```bash
cd /path/to/sky-fort-acoustic
.venv/bin/uvicorn acoustic.main:app --reload --host 0.0.0.0 --port 8000 --app-dir src
```

Runs on **http://localhost:8000**. The `--app-dir src` flag is required because the `acoustic` package lives under `src/`.

### Frontend

In a separate terminal:

```bash
cd web
npm run dev
```

Runs on **http://localhost:5173**.

### Run Both in the Background

Useful for quick iteration when you don't want two terminals open:

```bash
.venv/bin/uvicorn acoustic.main:app --reload --host 0.0.0.0 --port 8000 --app-dir src > /tmp/sfa-backend.log 2>&1 &
(cd web && npm run dev > /tmp/sfa-frontend.log 2>&1 &)
```

Tail logs with `tail -f /tmp/sfa-backend.log` / `tail -f /tmp/sfa-frontend.log`.

### Restarting the Backend

```bash
# Stop whatever is on :8000, then start a fresh detached backend
lsof -tiTCP:8000 -sTCP:LISTEN | xargs -r kill -9
.venv/bin/uvicorn acoustic.main:app --reload --host 0.0.0.0 --port 8000 --app-dir src > /tmp/sfa-backend.log 2>&1 &
disown 2>/dev/null || true

# Verify it came up (give it ~5s to import torch etc.)
sleep 5 && curl -s http://localhost:8000/health
```

### Stopping Services

```bash
# Find anything bound to the two ports
lsof -iTCP:8000 -sTCP:LISTEN -nP
lsof -iTCP:5173 -sTCP:LISTEN -nP

# Kill by PID (prefer SIGTERM; fall back to SIGKILL only if needed)
kill <pid>            # graceful
kill -9 <pid>         # force

# One-liner: kill whatever is listening on both dev ports
lsof -tiTCP:8000 -sTCP:LISTEN | xargs -r kill -9
lsof -tiTCP:5173 -sTCP:LISTEN | xargs -r kill -9
```

Note: `uvicorn --reload` spawns a reloader parent and a worker child — both will appear in `lsof`. Killing the parent takes the worker with it.

### Hardware Fallback

If the UMA-16v2 is not connected, the backend auto-falls back to any available input device (ReSpeaker 4 Mic, built-in mic, etc.) and replicates channels to keep the pipeline shape consistent. Beamforming output is meaningless on non-UMA hardware but CNN classification still runs. Check `/health` — `device_name` reports the active device.

## API

- `GET /health` — service health and pipeline status
- `WS /ws/targets` — live target tracking events
- `WS /ws/events` — raw detection events
