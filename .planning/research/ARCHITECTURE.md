# Architecture Patterns

**Domain:** Acoustic drone detection and tracking microservice
**Researched:** 2026-03-29

## Recommended Architecture

A **pipeline architecture** with four processing stages feeding a shared state bus. Each stage runs in its own thread (or async task), communicating via thread-safe queues and shared state objects. The service exposes two output interfaces: ZeroMQ PUB for downstream machine consumers and REST/WebSocket for the web UI.

```
+------------------+     +------------------+     +------------------+
|  Audio Capture   | --> |  Beamforming     | --> |  Target Tracker  |
|  (sounddevice)   |     |  (acoular)       |     |  (state machine) |
|  16ch @ 48kHz    |     |  spatial map     |     |  ID, bearing,    |
|                  |     |  peak detection  |     |  speed, class    |
+------------------+     +--+------------+--+     +--------+---------+
                            |            |                 |
                            v            |                 v
                    +-------+------+     |        +--------+---------+
                    | CNN Classifier|    |        |  Event Publisher  |
                    | (TF/Keras)   |     |        |  (ZeroMQ PUB)    |
                    | mel-spec +   |     |        |  detection events |
                    | drone prob   |     |        |  periodic updates |
                    +--------------+     |        +------------------+
                                         |
                                         v
                              +----------+---------+
                              |  REST API (FastAPI) |
                              |  /beamforming-map   |
                              |  /targets           |
                              |  /recordings        |
                              |  /training          |
                              |  WebSocket /ws/live  |
                              +----------+---------+
                                         |
                                         v
                              +----------+---------+
                              |  React Web UI       |
                              |  (Vite+TS+Tailwind) |
                              |  served as static   |
                              +--------------------+
```

### Component Boundaries

| Component | Responsibility | Inputs | Outputs | Thread Model |
|-----------|---------------|--------|---------|--------------|
| **Audio Capture** | Acquire raw 16-channel PCM from UMA-16v2 via sounddevice | USB audio device | Raw PCM chunks (80ms @ 48kHz = 3840 samples x 16ch) | Dedicated thread with blocking read |
| **Beamforming Engine** | Compute spatial sound map, find peak direction | Raw PCM chunks | Beamforming map (2D grid), peak (x,y), azimuth, L_max dB | Same thread as capture (tight loop, as in POC) |
| **CNN Classifier** | Classify audio as drone/not-drone, output probability | Resampled mono audio segment (2s @ 16kHz) | Drone probability (0-1), drone class label | Separate thread (GPU/CPU inference is blocking) |
| **Target Tracker** | Maintain target state machine: detection, tracking, loss | Beamforming peak + CNN probability | Target ID, class, pan/tilt degrees, Doppler speed, state | Runs in main processing loop after beamforming |
| **Doppler Estimator** | Estimate radial speed from frequency shift over time | Sequential beamforming peaks + audio spectra | Speed estimate (m/s) | Inline computation in tracker |
| **ZeroMQ Publisher** | Publish detection and tracking events to downstream | Target state changes | JSON messages on topic-filtered PUB socket | Dedicated async task or thread |
| **REST API** | Serve beamforming map, target info, recording CRUD, training trigger | HTTP requests | JSON responses, WebSocket streams | FastAPI async event loop |
| **Recording Manager** | Record raw 16-ch audio to disk, manage metadata | Audio capture stream + user commands | WAV files + metadata JSON | Writes in capture thread, metadata via API |
| **Training Pipeline** | Train/retrain CNN model from labeled recordings | Labeled WAV files + metadata | Updated model file (.h5/.keras) | Background process (CPU/GPU intensive) |
| **Web UI** | Live monitoring, recording controls, training UI | REST API + WebSocket | User interactions | Separate build artifact, served as static files |

### Data Flow

**Real-time detection path (latency-critical):**

```
USB Mic (48kHz, 16ch)
  |
  v
[Audio Capture Thread]
  |-- raw PCM chunk (80ms)
  |
  +---> [Beamforming] (acoular: TimeSamples -> PowerSpectra -> BeamformerBase)
  |       |-- spatial map (101x101 grid)
  |       |-- peak location (ix, iy)
  |       |-- azimuth (degrees)
  |       |-- L_max (dB)
  |       |
  |       +---> [Target Tracker]
  |               |-- pan/tilt calculation from peak
  |               |-- Doppler estimation from peak shift over time
  |               |-- state machine: IDLE -> DETECTED -> TRACKING -> LOST
  |               |
  |               +---> [ZeroMQ PUB] detection/update events
  |               +---> [Shared State] for REST API to read
  |
  +---> [Ring Buffer] (last 2s of mono audio)
          |
          +---> [CNN Classifier Thread] (every 0.5s)
                  |-- resample 48kHz -> 16kHz
                  |-- mel spectrogram (64 mels, 128 frames)
                  |-- model.predict() -> drone probability
                  |-- feeds into Target Tracker state
```

**Recording path:**

```
[Audio Capture Thread] ---(raw 16ch PCM)---> [WAV Writer] ---> disk
[REST API] ---(start/stop/metadata)---> [Recording Manager]
[Web UI] ---(labels, notes)---> [REST API] ---> [Metadata Store]
```

**Training path:**

```
[Web UI] ---(trigger training)---> [REST API] ---> [Training Pipeline]
[Training Pipeline]:
  1. Load labeled recordings from disk
  2. Extract mel spectrograms
  3. Train CNN (TensorFlow/Keras)
  4. Save model to disk
  5. Hot-reload model in CNN Classifier thread
```

**Playback/simulation path:**

```
[Web UI] ---(select recording)---> [REST API] ---> [Playback Engine]
[Playback Engine]:
  1. Read WAV file
  2. Feed chunks to Beamforming Engine (same pipeline as live)
  3. Results flow through Target Tracker -> ZeroMQ + API as normal
```

## Patterns to Follow

### Pattern 1: Threaded Pipeline with Shared State

**What:** Each processing stage runs in its own thread. Communication happens through thread-safe shared state objects (with locks) and queues. This matches the POC's proven `SharedState` pattern.

**When:** Always -- this is the core architecture.

**Why:** Audio capture is blocking I/O. CNN inference is CPU-bound. The REST API is async I/O. These workloads naturally separate into threads. The POC already validates this approach works at 48kHz/16ch.

```python
import threading
from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class SharedState:
    # Audio capture -> Beamforming (same thread, direct)
    # Beamforming -> API/ZMQ (shared state with locks)
    map_lock: threading.Lock = field(default_factory=threading.Lock)
    beamforming_map: np.ndarray | None = None
    peak_azimuth_deg: float = 0.0
    peak_elevation_deg: float = 0.0
    l_max_db: float = -100.0

    # CNN -> Tracker (shared state with lock)
    cnn_lock: threading.Lock = field(default_factory=threading.Lock)
    drone_probability: float = 0.0
    drone_class: str = ""

    # Tracker -> ZMQ/API (shared state with lock)
    target_lock: threading.Lock = field(default_factory=threading.Lock)
    targets: dict = field(default_factory=dict)

    # Control signals
    stop: threading.Event = field(default_factory=threading.Event)
```

### Pattern 2: Topic-Based ZeroMQ Events

**What:** Use ZeroMQ PUB/SUB with topic prefixes for different event types. Subscribers filter by topic.

**When:** All outbound event publishing.

```python
import zmq
import json

# Publisher side
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://*:5556")

# Detection event (new target)
pub.send_multipart([
    b"detection",
    json.dumps({
        "target_id": "T-001",
        "drone_class": "DJI_Mavic",
        "confidence": 0.92,
        "pan_deg": 45.2,
        "tilt_deg": 12.1,
        "timestamp": 1711720000.123
    }).encode()
])

# Periodic update (existing target)
pub.send_multipart([
    b"tracking",
    json.dumps({
        "target_id": "T-001",
        "speed_mps": 8.3,
        "pan_deg": 46.1,
        "tilt_deg": 11.8,
        "timestamp": 1711720000.223
    }).encode()
])

# Target lost
pub.send_multipart([
    b"lost",
    json.dumps({
        "target_id": "T-001",
        "timestamp": 1711720002.5
    }).encode()
])
```

### Pattern 3: FastAPI with Background Thread Bridge

**What:** FastAPI runs in its own async event loop. A bridge reads from SharedState and can push to WebSocket clients. Background threads handle the audio pipeline.

**When:** REST API + WebSocket serving.

```python
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start audio pipeline threads
    pipeline = AudioPipeline(state)
    pipeline.start()
    yield
    # Shutdown
    state.stop.set()
    pipeline.join(timeout=5.0)

app = FastAPI(lifespan=lifespan)

@app.get("/api/beamforming-map")
async def get_beamforming_map():
    with state.map_lock:
        if state.beamforming_map is None:
            return {"map": None}
        return {
            "map": state.beamforming_map.tolist(),
            "peak_azimuth": state.peak_azimuth_deg,
            "l_max_db": state.l_max_db
        }

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    while True:
        with state.target_lock:
            data = serialize_targets(state.targets)
        await ws.send_json(data)
        await asyncio.sleep(0.1)  # 10 Hz update
```

### Pattern 4: Single-Container Multi-Process via Supervisor or Entrypoint

**What:** Single Docker container runs Python backend (FastAPI + audio pipeline threads) plus serves pre-built React static files. No need for a separate frontend container.

**When:** Docker deployment.

```dockerfile
FROM python:3.11-slim
# Install system deps for audio (ALSA/PulseAudio), build tools
RUN apt-get update && apt-get install -y libportaudio2 libsndfile1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ /app/backend/
COPY frontend/dist/ /app/frontend/dist/
WORKDIR /app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

FastAPI serves the React build as static files via `StaticFiles` mount.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Processing Loop

**What:** Putting audio capture, beamforming, CNN inference, and HTTP serving all in a single thread/loop (like the POC does partially).

**Why bad:** CNN inference (50-200ms) blocks the audio capture loop, causing buffer overruns and dropped frames. The POC already solves this with a separate `SoundCNNWorker` thread -- the new service must maintain this separation.

**Instead:** Keep CNN in its own thread with a queue. Audio capture + beamforming can share a thread (beamforming is fast, ~5-10ms per chunk). REST API runs in async event loop.

### Anti-Pattern 2: Passing Full Audio Arrays via ZeroMQ

**What:** Publishing raw 16-channel audio data over ZeroMQ to downstream consumers.

**Why bad:** 48kHz x 16ch x 4 bytes = 3 MB/s. ZeroMQ PUB/SUB is for events and metadata, not bulk audio streaming.

**Instead:** Publish only detection events and tracking updates (small JSON payloads). If raw audio is needed downstream, use a different mechanism (shared filesystem, dedicated stream).

### Anti-Pattern 3: REST Polling for Live Data

**What:** Having the web UI poll REST endpoints at high frequency for live beamforming map and target updates.

**Why bad:** HTTP overhead, latency, unnecessary load. Beamforming map is ~40KB per update at 12 Hz = 480 KB/s of redundant HTTP framing.

**Instead:** Use WebSocket for live streaming data to the UI. REST for CRUD operations (recordings, metadata, training). The beamforming map and target updates flow over WebSocket.

### Anti-Pattern 4: Training in the Main Process

**What:** Running CNN training in the same Python process as the real-time pipeline.

**Why bad:** Training consumes all CPU/GPU, starving the real-time audio pipeline. GIL contention with NumPy operations.

**Instead:** Spawn training as a subprocess. The API triggers it, monitors progress via file or IPC, and hot-reloads the model when training completes.

## Component Build Order (Dependencies)

The architecture has clear dependency chains that dictate build order:

```
Phase 1: Audio Capture + Beamforming (foundation)
   |
   +-- No downstream deps. This is the core that everything reads from.
   |   Extract from POC: sounddevice capture, acoular beamforming, mic geometry.
   |   Output: SharedState with beamforming map, peak, azimuth, L_max.
   |
Phase 2: REST API + Static File Serving (output layer)
   |
   +-- Depends on: Phase 1 (reads SharedState for /beamforming-map endpoint)
   |   FastAPI app, beamforming map endpoint, health check.
   |   Serve React static files.
   |
Phase 3: Web UI - Live Monitoring (visualization)
   |
   +-- Depends on: Phase 2 (consumes REST API / WebSocket)
   |   React app: beamforming heatmap, target list, connection status.
   |   WebSocket for real-time updates.
   |
Phase 4: CNN Classifier (detection intelligence)
   |
   +-- Depends on: Phase 1 (reads audio from ring buffer)
   |   Extract from POC: SoundCNNWorker, mel spectrogram, model loading.
   |   Feeds drone probability into SharedState.
   |
Phase 5: Target Tracker + ZeroMQ Publisher (state machine + output)
   |
   +-- Depends on: Phase 1 (beamforming), Phase 4 (CNN classification)
   |   State machine: IDLE -> DETECTED -> TRACKING -> LOST
   |   Doppler speed estimation from peak shift.
   |   Pan/tilt degree calculation.
   |   ZeroMQ PUB with topic-filtered events.
   |
Phase 6: Recording + Playback (data capture)
   |
   +-- Depends on: Phase 1 (audio stream), Phase 2 (API endpoints)
   |   Record raw 16ch WAV. Metadata attachment via API.
   |   Playback: feed recorded WAV through same pipeline as live.
   |
Phase 7: CNN Training Pipeline (model lifecycle)
   |
   +-- Depends on: Phase 4 (model format), Phase 6 (labeled recordings)
   |   Training subprocess. Hot-reload trained model.
   |
Phase 8: Docker Packaging (deployment)
   |
   +-- Depends on: All phases (containerize the whole service)
       USB passthrough for UMA-16v2. Single container.
```

**Why this order:**

1. **Audio + Beamforming first** because every other component depends on it. Cannot test anything without audio input.
2. **REST API second** because it provides the interface for both humans (web UI) and integration testing.
3. **Web UI third** because it enables visual validation of beamforming output -- critical for tuning.
4. **CNN fourth** because it requires a working audio pipeline to feed it, and visual confirmation helps validate results.
5. **Tracker + ZeroMQ fifth** because it synthesizes beamforming + CNN into the actual product output.
6. **Recording sixth** because it's a side-channel off the existing pipeline, and recordings are needed for training.
7. **Training seventh** because it needs labeled recordings to exist.
8. **Docker last** because containerization should wrap a working service, not a partial one.

## Scalability Considerations

| Concern | Single Unit (v1) | Multi-Unit (future) | Notes |
|---------|------------------|---------------------|-------|
| Audio throughput | 48kHz x 16ch = 3 MB/s raw PCM. Easily handled by one thread. | N/A (one mic array per service instance) | Bottleneck is beamforming compute, not I/O |
| Beamforming latency | ~5-10ms per 80ms chunk on modern CPU. Well within real-time. | N/A | acoular uses NumPy + optional Numba JIT |
| CNN inference | ~50-200ms per 2s segment. Runs every 0.5s in separate thread. | N/A | TensorFlow CPU is adequate; GPU optional |
| ZeroMQ throughput | ~20 events/sec (detection + tracking). Trivial for ZMQ. | Multiple subscribers, still trivial | ZMQ PUB fans out automatically |
| WebSocket clients | 1-5 monitoring UIs. Fine with FastAPI async. | 10-50 would still be fine | Beamforming map is largest payload (~40KB) |
| Recording storage | ~3 MB/s raw. 10 min recording = ~1.8 GB. Disk-bound. | External storage mount | Docker volume for recordings |
| Training time | Minutes to hours depending on dataset size. Runs as subprocess. | Could offload to separate GPU machine | Out of scope for v1 |

## Key Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Audio framework | `sounddevice` + `acoular` | Proven in POC. acoular is purpose-built for mic array beamforming. No reason to switch. |
| Web framework | FastAPI | Async support for WebSocket + REST. Better performance than Flask (used in POC). Type-safe with Pydantic. |
| CNN framework | TensorFlow/Keras | POC uses `.h5` model format. Training pipeline already exists. Switching to PyTorch would require model retraining and format migration with no benefit. |
| Event transport | ZeroMQ (pyzmq) | Project requirement. PUB/SUB is simple, brokerless, low-latency. |
| Frontend | React + Vite + TypeScript + Tailwind CSS | Consistent with sky-fort-dashboard. Same tooling, same patterns. |
| Container | Single Docker container | Project requirement. FastAPI serves both API and static React build. |

## Sources

- POC codebase analysis: `POC-code/scripts/POC_Recorder.py` (beamforming pipeline, CNN worker, shared state pattern)
- POC codebase analysis: `POC-code/scripts/unified_drone_collection_web_gui.py` (process management, multi-service orchestration)
- [Acoular - Acoustic testing and source mapping software](https://www.acoular.org/) - mic array beamforming framework
- [Acoular GitHub](https://github.com/acoular/acoular) - pipeline architecture with generator objects
- [Pyroomacoustics](https://pyroomacoustics.readthedocs.io/) - alternative framework (not recommended, acoular is already proven in POC)
- [ZeroMQ PUB/SUB patterns](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html) - topic-based message filtering
- [FastAPI ZMQ Integration](https://www.restack.io/p/fastapi-answer-zmq-integration) - async ZMQ + FastAPI patterns
- [Acoustic drone detection via ML](https://www.researchgate.net/publication/366717010_Acoustic_Based_Drone_Detection_Via_Machine_Learning) - CNN + mel-spectrogram approach validation
- [Drone detection with beamforming + HMM](https://www.researchgate.net/publication/338472717_Classification_positioning_and_tracking_of_drones_by_HMM_using_acoustic_circular_microphone_array_beamforming) - beamforming + classification pipeline reference
- [Development of Acoustic System for UAV Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506852/) - system architecture reference
- sky-fort-dashboard `package.json` - React 19 + Vite 8 + Tailwind 4 + TypeScript 5.9 stack reference
