# Sky Fort Acoustic Service

Real-time acoustic drone detection and tracking microservice using a UMA-16v2 16-channel microphone array.

## Running Locally

### Backend

```bash
cd /path/to/sky-fort-acoustic
source .venv/bin/activate
python -m uvicorn acoustic.main:app --reload --host 0.0.0.0 --port 8000
```

Or without activating the venv:

```bash
.venv/bin/uvicorn acoustic.main:app --reload --host 0.0.0.0 --port 8000
```

Runs on **http://localhost:8000**

### Frontend

In a separate terminal:

```bash
cd web
npm run dev
```

Runs on **http://localhost:5173**

### Simulated Mode (no hardware)

If the UMA-16v2 mic is not connected, the backend automatically falls back to simulated audio. To force it explicitly:

```bash
AUDIO_SOURCE=simulated python -m uvicorn acoustic.main:app --reload
```

## API

- `GET /health` — service health and pipeline status
- `WS /ws/targets` — live target tracking events
- `WS /ws/events` — raw detection events
