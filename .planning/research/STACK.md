# Technology Stack

**Project:** Sky Fort Acoustic Service
**Researched:** 2026-03-29

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

**Decision: Custom over Acoular.** Acoular adds complexity (Numba compilation, caching layer, pipeline graph) without benefit for a 4x4 array with a single known geometry. The POC's approach is 180 lines of NumPy that runs in <50ms per chunk. If we later need CLEAN-SC, DAMAS, or other advanced algorithms, Acoular can be added as an optional dependency.

### Machine Learning

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | >=2.11.0 | CNN training + inference | Dominates audio ML research (85% of papers). Better debug experience than TensorFlow (eager mode). `torch.compile()` for production speedup. The POC listed TensorFlow but the research community has moved to PyTorch for audio classification. | HIGH |
| torchaudio | >=2.11.0 | Audio transforms | `MelSpectrogram`, `MFCC`, `Spectrogram` transforms as GPU-accelerated layers. Converts raw audio to CNN-ready features in the training pipeline. Matches PyTorch version. | HIGH |

**Why NOT TensorFlow:** The POC had TensorFlow in requirements but the acoustic drone detection research community has converged on PyTorch. EfficientNet and ResNet variants for mel-spectrogram classification are trivially available in `torchvision`. TensorFlow adds a second ML framework to maintain with no compensating benefit for this use case.

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

**Consistency note:** The existing sky-fort-dashboard uses React 19, Vite 8, Tailwind 4, TanStack Query 5. This service's UI MUST match that stack exactly to share components and maintain team familiarity.

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

```bash
# Core runtime
pip install \
  fastapi>=0.135 \
  uvicorn[standard]>=0.34 \
  sounddevice>=0.5.1 \
  soundfile>=0.13.1 \
  numpy>=1.26 \
  scipy>=1.14 \
  pyzmq>=27.1.0 \
  pydantic>=2.10

# ML (install separately -- large)
pip install \
  torch>=2.11.0 \
  torchaudio>=2.11.0

# Dev dependencies
pip install \
  pytest>=8.0 \
  pytest-asyncio>=0.24 \
  httpx>=0.28 \
  ruff>=0.9 \
  mypy>=1.14
```

## Frontend Dependencies

```bash
# Scaffold (match sky-fort-dashboard)
pnpm create vite acoustic-ui --template react-ts

# Core
pnpm add react react-dom tailwindcss @tailwindcss/vite \
  @tanstack/react-query recharts

# Dev
pnpm add -D typescript @types/react @types/react-dom \
  @vitejs/plugin-react eslint vite
```

## Docker Considerations

```dockerfile
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
```

## Version Pinning Strategy

Pin MAJOR.MINOR in requirements, allow PATCH float:
- `fastapi>=0.135,<1.0` (pre-1.0, minor can break)
- `torch>=2.11,<2.12` (ML frameworks break between minors)
- `numpy>=1.26,<3` (conservative until ecosystem catches up)
- `sounddevice>=0.5,<1.0`

Use a `requirements.txt` for reproducible builds with exact pins generated by `pip freeze`. Use `pyproject.toml` for the "loose" version specs.

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
