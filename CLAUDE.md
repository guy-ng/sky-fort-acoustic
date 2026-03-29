<!-- GSD:project-start source:PROJECT.md -->
## Project

**Sky Fort Acoustic Service**

A standalone, Dockerized Python microservice that performs real-time acoustic drone detection and tracking using a UMA-16v2 (16-channel) microphone array. It replaces the scattered POC code with a clean, single-responsibility service that does beamforming, CNN-based drone classification, Doppler speed estimation, and publishes tracking events over ZeroMQ — plus a web interface for live monitoring, recording, and model training.

**Core Value:** Reliably detect and classify drones acoustically in real time, publishing target events (ID, class, speed, bearing) over ZeroMQ so downstream systems can act on them.

### Constraints

- **Hardware**: UMA-16v2 mic array must be accessible from Docker (USB passthrough)
- **Runtime**: Python backend, single Docker container
- **Web UI**: React app (Vite + TypeScript + Tailwind CSS) — consistent with sky-fort-dashboard
- **Messaging**: ZeroMQ for event publishing (PUB/SUB pattern)
- **Real-time**: Audio processing must keep up with 48kHz 16-channel stream
- **Deployment**: Must run independently — no dependency on other POC components
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Core Framework
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | >=3.11 | Runtime | f-strings, TaskGroup, performance improvements. 3.11+ has 25% faster execution than 3.10. Match PyTorch 2.11 compatibility (requires >=3.10). | HIGH |
| FastAPI | >=0.135 | REST API + WebSocket | Async-native (ASGI), native WebSocket support for live beamforming map streaming, auto-generated OpenAPI docs, 5-10x faster than Flask for concurrent requests. The POC uses Flask but this service needs async for simultaneous audio streaming + REST + WebSocket. | HIGH |
| Uvicorn | >=0.34 | ASGI server | Standard production server for FastAPI. Use with `--workers` for multi-process if needed. | HIGH |
### Audio Capture & Processing
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| sounddevice | >=0.5.1 | 16-channel audio capture | Already proven in POC for UMA-16v2. PortAudio backend, supports ALSA device names (`hw:X,0`), callback-based streaming for real-time. No viable alternative for multi-channel USB audio in Python. | HIGH |
| soundfile | >=0.13.1 | WAV read/write | libsndfile-backed, handles 16-channel WAV natively with NumPy arrays. Used for recording and playback. | HIGH |
| NumPy | >=1.26,<3 | DSP core | All beamforming math (FFT, cross-correlation, steering vectors). POC already uses it. Keep `<3` for now -- NumPy 2.x broke some downstream libs in 2024 but is stable in 2025+. Can relax to `>=2.0` once acoular confirms compatibility. | HIGH |
| SciPy | >=1.14 | Signal processing | Filters, windowing, spectral analysis. `scipy.signal` for bandpass filters on drone frequency band (100-2000 Hz). | HIGH |
### Beamforming
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Custom SRP-PHAT | N/A | Primary beamforming | The POC already implements SRP-PHAT with GCC-PHAT in pure NumPy (~180 lines). This is fast enough for 16 channels at 150ms chunks and gives full control over the algorithm. Keep this, don't add a dependency for what works. | HIGH |
| acoular | >=25.10 | Optional/reference | Acoular is the gold standard Python beamforming library (Numba-accelerated, supports streaming via SoundDeviceSamplesGenerator). However, its pipeline architecture (lazy evaluation, caching) is designed for offline analysis, not tight real-time loops. Use it as a reference/validation tool, not as the core engine. The POC's custom SRP-PHAT is simpler and sufficient for a 4x4 array. | MEDIUM |
### Machine Learning
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | >=2.11.0 | CNN training + inference | Dominates audio ML research (85% of papers). Better debug experience than TensorFlow (eager mode). `torch.compile()` for production speedup. The POC listed TensorFlow but the research community has moved to PyTorch for audio classification. | HIGH |
| torchaudio | >=2.11.0 | Audio transforms | `MelSpectrogram`, `MFCC`, `Spectrogram` transforms as GPU-accelerated layers. Converts raw audio to CNN-ready features in the training pipeline. Matches PyTorch version. | HIGH |
### Messaging
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pyzmq | >=27.1.0 | ZeroMQ PUB/SUB | Project requirement. Lightweight, brokerless, low-latency. PUB/SUB pattern for detection events + periodic target updates. No broker to deploy or maintain. | HIGH |
### Frontend (Web UI)
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| React | ^19 | UI framework | Matches sky-fort-dashboard. React 19 with concurrent features. | HIGH |
| Vite | ^8 | Build tool | Matches sky-fort-dashboard (v8). Fast HMR, ESM-native. | HIGH |
| TypeScript | ~5.9 | Type safety | Matches sky-fort-dashboard. | HIGH |
| Tailwind CSS | ^4 | Styling | Matches sky-fort-dashboard (v4 with `@tailwindcss/vite`). | HIGH |
| Recharts | >=2.15 | Charts/visualization | Lightweight, React-native charting for beamforming heatmap, signal levels, Doppler plots. Simpler than D3 for dashboard charts. | MEDIUM |
| TanStack Query | ^5 | Server state | Matches sky-fort-dashboard. Handles REST polling, caching, refetching for beamforming map endpoint. | HIGH |
### Infrastructure
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Docker | N/A | Containerization | Project requirement. Single container with Python backend + built React frontend served by FastAPI. | HIGH |
| ALSA | N/A | Audio device layer | Required for USB mic array access in Docker via `--device /dev/snd`. No PulseAudio needed -- direct ALSA access is simpler and lower latency. | HIGH |
### Dev & Testing
| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=8.0 | Testing | Standard Python testing. Use with `pytest-asyncio` for FastAPI route tests. | HIGH |
| pytest-asyncio | >=0.24 | Async test support | Required for testing async FastAPI endpoints. | HIGH |
| httpx | >=0.28 | HTTP test client | FastAPI's recommended test client (`AsyncClient`). | HIGH |
| Ruff | >=0.9 | Linting + formatting | Replaces flake8 + black + isort. Single tool, 100x faster. | HIGH |
| mypy | >=1.14 | Type checking | Catch type errors in DSP code where numpy dtype mismatches cause silent bugs. | MEDIUM |
## Alternatives Considered
| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Web framework | FastAPI | Flask | Flask is sync-only (WSGI). This service needs async WebSocket for live beamforming map + concurrent REST. POC used Flask but that was a prototype. |
| Web framework | FastAPI | Litestar | Less ecosystem, fewer tutorials, smaller community. FastAPI is the standard. |
| ML framework | PyTorch | TensorFlow | Research community moved to PyTorch. TF's eager mode is second-class. Maintaining two ML frameworks is wasteful. |
| ML framework | PyTorch | ONNX Runtime only | Need training pipeline, not just inference. PyTorch for training, can export to ONNX later for optimized inference if needed. |
| Beamforming | Custom NumPy | Acoular | Acoular's pipeline model adds complexity for a simple 4x4 SRP-PHAT. POC's 180-line implementation works. |
| Beamforming | Custom NumPy | pyroomacoustics | Focused on simulation, not real-time processing. Good for testing but not production. |
| Audio capture | sounddevice | PyAudio | PyAudio is unmaintained, has installation issues. sounddevice wraps PortAudio cleanly and is proven in POC. |
| Messaging | ZeroMQ | MQTT | Project specifies ZeroMQ. MQTT adds broker dependency. ZMQ is lower latency for local pub/sub. |
| Messaging | ZeroMQ | Redis Pub/Sub | Adds Redis server dependency. ZMQ is brokerless -- simpler for embedded/edge deployment. |
| Charts | Recharts | D3.js | D3 is lower-level, more code for standard charts. Recharts wraps it for React. |
| Charts | Recharts | visx | More flexible but more boilerplate. Recharts sufficient for dashboards. |
| Linting | Ruff | flake8+black+isort | Ruff replaces all three in one tool, 100x faster. No reason to use the old stack. |
## Python Dependencies
# Core runtime
# ML (install separately -- large)
# Dev dependencies
## Frontend Dependencies
# Scaffold (match sky-fort-dashboard)
# Core
# Dev
## Docker Considerations
# Key Docker run flags for USB mic array access:
# docker run --device /dev/snd -v /dev/bus/usb:/dev/bus/usb --privileged ...
#
# The container needs:
# - ALSA libraries (libasound2-dev)
# - PortAudio (libportaudio2)
# - No PulseAudio needed
#
# Multi-stage build:
# Stage 1: Node.js -- build React frontend
# Stage 2: Python -- copy built frontend, install Python deps, serve via FastAPI
## Version Pinning Strategy
- `fastapi>=0.135,<1.0` (pre-1.0, minor can break)
- `torch>=2.11,<2.12` (ML frameworks break between minors)
- `numpy>=1.26,<3` (conservative until ecosystem catches up)
- `sounddevice>=0.5,<1.0`
## Sources
- [Acoular 25.10 Documentation](https://www.acoular.org/)
- [Acoular PyPI](https://pypi.org/project/acoular/)
- [FastAPI PyPI](https://pypi.org/project/fastapi/)
- [PyTorch Releases](https://github.com/pytorch/pytorch/releases)
- [PyZMQ GitHub](https://github.com/zeromq/pyzmq)
- [sounddevice Documentation](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)
- [soundfile Documentation](https://python-soundfile.readthedocs.io/)
- [torchaudio MelSpectrogram](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html)
- [Drone detection CNN research](https://arxiv.org/html/2406.18624v2) - Robust Low-Cost Drone Detection Using CNNs
- [AUDRON Framework](https://arxiv.org/pdf/2512.20407) - Deep Learning for drone audio classification
- POC source: `POC-code/PT520/PTZ/radar_gui_all_mics_fast_drone.py` (SRP-PHAT reference implementation)
- sky-fort-dashboard `package.json` (frontend stack reference)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
