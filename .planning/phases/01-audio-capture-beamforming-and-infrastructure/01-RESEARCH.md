# Phase 1: Audio Capture, Beamforming, and Infrastructure - Research

**Researched:** 2026-03-29
**Domain:** Real-time multi-channel audio capture, SRP-PHAT beamforming, Docker/ALSA infrastructure
**Confidence:** HIGH

## Summary

Phase 1 builds the foundational service: a Docker container that captures 16-channel audio from the UMA-16v2 mic array (or a simulated source), runs SRP-PHAT beamforming to produce a 2D spatial map, detects the strongest peak (azimuth + elevation), and exposes a FastAPI health endpoint. The core technical challenges are (1) callback-based audio streaming with a lock-free ring buffer to decouple capture from processing, (2) extending the POC's 1D azimuth-only SRP-PHAT to a 2D azimuth+elevation grid, and (3) Docker ALSA passthrough for USB audio devices.

The POC provides a solid 1D SRP-PHAT implementation (~180 lines of NumPy) that needs refactoring into clean modules and extending to 2D. The `sounddevice.InputStream` callback API is proven for multi-channel USB audio. The main pitfall is that Docker on macOS cannot pass through USB audio devices -- the simulated audio source (D-04) is essential for development, with real hardware testing only on Linux.

**Primary recommendation:** Port the POC's SRP-PHAT to clean modules, extend to 2D steering vectors, wrap in a callback-based InputStream + ring buffer architecture, and validate with synthetic audio from the simulator before hardware testing.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Callback-based `sounddevice.InputStream` in a dedicated thread writes chunks into a lock-free ring buffer. Beamforming consumer reads from the ring buffer asynchronously. This decouples capture from processing so neither blocks the other.
- **D-02:** Chunk size stays at POC's 150ms (7200 samples at 48kHz) -- proven compromise between latency and low-frequency content for the 100-2000 Hz drone band.
- **D-03:** Ring buffer sized for ~2 seconds of audio (approximately 14 chunks) to absorb processing jitter without dropping frames.
- **D-04:** A simulated audio source provides synthetic 16-channel audio (sine waves with configurable direction-of-arrival) when no UMA-16v2 is detected. Enabled automatically on device absence or via environment variable (`AUDIO_SOURCE=simulated`).
- **D-05:** Test fixtures use short pre-recorded WAV snippets (from `audio-data/`) for deterministic beamforming validation.
- **D-06:** Spatial map is a 2D NumPy array (azimuth x elevation grid). Resolution and angular range configurable, defaulting to 1 degree steps over +/-90 degrees azimuth, +/-45 degrees elevation.
- **D-07:** Peak detection returns azimuth and elevation in degrees (pan/tilt), matching the coordinate system downstream consumers expect.
- **D-08:** Adaptive noise threshold uses percentile-based calibration (per BF-04) -- beamforming peak must exceed the Nth percentile of the map by a configurable margin to count as a detection.
- **D-09:** Base image: `python:3.11-slim` with ALSA libraries (`libasound2-dev`, `libportaudio2`) installed. No PulseAudio.
- **D-10:** USB passthrough via `--device /dev/snd` and `-v /dev/bus/usb:/dev/bus/usb`. Document `--privileged` as fallback if device mapping fails.
- **D-11:** Single Dockerfile, no multi-stage yet (frontend comes in Phase 2). Multi-stage build added when React UI is introduced.

### Claude's Discretion
- Exact ring buffer implementation (stdlib `queue`, custom numpy circular buffer, etc.)
- Project directory structure and module layout
- Logging framework choice and verbosity levels
- Test structure (unit vs integration split)
- FastAPI app skeleton scope in Phase 1 (minimal health endpoint only, or full app structure)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUD-01 | Service captures real-time 16-channel audio from UMA-16v2 at 48kHz using callback-based streaming (not blocking) | sounddevice.InputStream callback API verified (v0.5.5). POC proves UMA-16v2 works with sounddevice via ALSA device names. D-01 locks callback architecture. |
| AUD-02 | Audio capture runs continuously in a dedicated thread with a ring buffer for downstream consumers | D-01/D-03 lock ring buffer architecture. Research recommends NumPy circular buffer over queue.Queue for zero-copy performance. |
| AUD-03 | Service detects and reports UMA-16v2 device presence/absence at startup and during operation | sounddevice.query_devices() returns device list with name, channels, hostapi. Match against "UMA16v2" string. Metadata JSON confirms real device name: "UMA16v2: USB Audio (hw:2,0)". |
| BF-01 | Service produces a beamforming spatial map (SRP-PHAT) from 16-channel audio in real time | POC's srp_phat_1d_fast() ported and extended to 2D. D-06 locks output format as 2D NumPy array. |
| BF-02 | Beamforming frequency band is configurable at runtime (default 100-2000 Hz for drone detection) | POC already implements band_mask via DRONE_FMIN/DRONE_FMAX. Expose as env vars FREQ_MIN/FREQ_MAX. |
| BF-03 | Service calculates peak azimuth and elevation from beamforming map | D-07 locks output as degrees. np.unravel_index(np.argmax(srp_map)) on 2D grid gives (az_idx, el_idx). |
| BF-04 | Service applies adaptive noise threshold (percentile-based calibration with configurable margin) | D-08 locks percentile-based approach. np.percentile(srp_map, N) + margin. |
| INF-01 | Service runs in a single Docker container with USB passthrough for UMA-16v2 | D-09/D-10 lock Docker approach. ALSA libs in python:3.11-slim. --device /dev/snd for Linux. |
| INF-02 | Dockerfile uses multi-stage build | D-11 explicitly defers multi-stage to Phase 2. Single-stage Dockerfile for Phase 1. **Note: INF-02 is partially addressed** -- the Dockerfile exists but is single-stage. |
| INF-03 | Service configurable via environment variables | Pydantic Settings (BaseSettings) for typed env var parsing with defaults. Standard FastAPI pattern. |
| INF-04 | Service includes health check endpoint reporting device status and pipeline state | FastAPI GET /health endpoint returning JSON with device_detected, pipeline_running, last_frame_time. Docker HEALTHCHECK uses curl. |
</phase_requirements>

## Standard Stack

### Core (Phase 1 only)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11 | Runtime (Docker image) | Pinned in CLAUDE.md. 3.11-slim base image per D-09. |
| FastAPI | 0.135.2 | REST API (health endpoint) | Latest stable. Async-native, lifespan events for startup/shutdown. |
| Uvicorn | 0.42.0 | ASGI server | Latest stable. Standard production server for FastAPI. |
| sounddevice | 0.5.5 | 16-channel audio capture | Latest stable. PortAudio backend, callback-based InputStream, proven with UMA-16v2 in POC. |
| soundfile | 0.13.1 | WAV read/write | Latest stable. For test fixtures (reading pre-recorded WAVs). |
| NumPy | >=1.26,<3 | DSP core (FFT, beamforming math) | Per CLAUDE.md pinning strategy. POC uses NumPy for all DSP. |
| SciPy | >=1.14 | Signal processing (bandpass filters) | Per CLAUDE.md. scipy.signal for optional filtering. |
| pydantic-settings | >=2.7 | Env var configuration | Standard FastAPI pattern for typed settings with BaseSettings. |

### Dev & Testing (Phase 1)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.0.2 | Test runner | All tests |
| pytest-asyncio | 1.3.0 | Async test support | FastAPI endpoint tests |
| httpx | 0.28.1 | HTTP test client | FastAPI test client (AsyncClient) |
| Ruff | 0.15.8 | Linting + formatting | Pre-commit, CI |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| NumPy circular buffer | queue.Queue | Queue adds GIL contention and copy overhead. NumPy pre-allocated circular buffer with atomic index is faster for fixed-size audio chunks. |
| pydantic-settings | python-dotenv + os.environ | pydantic-settings gives type validation, defaults, and nesting. Worth the dependency for a FastAPI project. |
| stdlib logging | structlog | structlog adds structured JSON logging but is an extra dependency. stdlib logging is sufficient for Phase 1; can upgrade later. |

**Installation (requirements.txt):**
```
fastapi>=0.135,<1.0
uvicorn[standard]>=0.42,<1.0
sounddevice>=0.5,<1.0
soundfile>=0.13,<1.0
numpy>=1.26,<3
scipy>=1.14
pydantic-settings>=2.7
```

**Dev requirements (requirements-dev.txt):**
```
pytest>=9.0
pytest-asyncio>=1.3
httpx>=0.28
ruff>=0.15
```

## Architecture Patterns

### Recommended Project Structure
```
sky-fort-acoustic/
├── src/
│   └── acoustic/
│       ├── __init__.py
│       ├── main.py              # FastAPI app, lifespan, health endpoint
│       ├── config.py            # Pydantic Settings (env vars)
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── capture.py       # InputStream callback, ring buffer
│       │   ├── device.py        # Device detection, UMA-16v2 matching
│       │   └── simulator.py     # Synthetic 16-ch audio source (D-04)
│       ├── beamforming/
│       │   ├── __init__.py
│       │   ├── srp_phat.py      # 2D SRP-PHAT engine (ported from POC)
│       │   ├── gcc_phat.py      # GCC-PHAT cross-correlation
│       │   ├── geometry.py      # Mic positions, steering vectors
│       │   └── peak.py          # Peak detection + noise threshold (BF-03, BF-04)
│       └── types.py             # Shared dataclasses/TypedDicts
├── tests/
│   ├── conftest.py              # Shared fixtures (synthetic audio, mic positions)
│   ├── unit/
│   │   ├── test_ring_buffer.py
│   │   ├── test_srp_phat.py
│   │   ├── test_gcc_phat.py
│   │   ├── test_geometry.py
│   │   ├── test_peak.py
│   │   └── test_config.py
│   └── integration/
│       ├── test_capture_pipeline.py
│       └── test_health.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .dockerignore
```

### Pattern 1: Callback-Based Audio Capture with Ring Buffer
**What:** sounddevice.InputStream invokes a callback on a PortAudio thread. The callback copies audio data into a pre-allocated NumPy circular buffer using atomic index updates. A consumer coroutine polls the buffer for new chunks.
**When to use:** Always -- this is the locked architecture (D-01).
**Example:**
```python
# Source: sounddevice docs + D-01/D-03 decisions
import numpy as np
import sounddevice as sd
import threading

class AudioRingBuffer:
    """Lock-free ring buffer for fixed-size audio chunks."""

    def __init__(self, num_chunks: int, chunk_samples: int, num_channels: int):
        self.buffer = np.zeros((num_chunks, chunk_samples, num_channels), dtype=np.float32)
        self.num_chunks = num_chunks
        self.write_idx = 0  # Only written by producer (callback thread)
        self.read_idx = 0   # Only written by consumer
        self._overflow_count = 0

    def write(self, data: np.ndarray) -> bool:
        """Write a chunk. Called from PortAudio callback thread."""
        next_idx = (self.write_idx + 1) % self.num_chunks
        if next_idx == self.read_idx:
            self._overflow_count += 1
            return False  # Buffer full
        self.buffer[self.write_idx] = data
        self.write_idx = next_idx
        return True

    def read(self) -> np.ndarray | None:
        """Read a chunk. Called from consumer thread/coroutine."""
        if self.read_idx == self.write_idx:
            return None  # Buffer empty
        data = self.buffer[self.read_idx].copy()
        self.read_idx = (self.read_idx + 1) % self.num_chunks
        return data

class AudioCapture:
    """Manages sounddevice InputStream with ring buffer."""

    def __init__(self, device: str | int | None, fs: int, channels: int,
                 chunk_samples: int, ring_chunks: int = 14):
        self.ring = AudioRingBuffer(ring_chunks, chunk_samples, channels)
        self.stream = sd.InputStream(
            device=device,
            samplerate=fs,
            channels=channels,
            dtype='float32',
            blocksize=chunk_samples,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        """PortAudio callback -- runs on audio thread, must be fast."""
        if status:
            # Log xruns (buffer overflows/underflows)
            pass
        self.ring.write(indata)  # No allocation, no GIL contention

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
```

### Pattern 2: 2D SRP-PHAT Extension
**What:** Extend POC's 1D azimuth scan to 2D (azimuth x elevation) by creating steering vectors for all (az, el) combinations.
**When to use:** Core beamforming engine (BF-01, BF-03).
**Example:**
```python
# Source: POC radar_gui_all_mics_fast_drone.py + 2D extension
import numpy as np
import itertools

def build_steering_vectors_2d(az_grid_deg, el_grid_deg):
    """Build unit direction vectors for 2D (azimuth, elevation) grid.

    az_grid_deg: 1D array of azimuth angles in degrees
    el_grid_deg: 1D array of elevation angles in degrees

    Returns: (n_az * n_el, 3) array of unit vectors
    """
    az_rad = np.deg2rad(az_grid_deg)
    el_rad = np.deg2rad(el_grid_deg)

    # Meshgrid: all combinations
    az_mesh, el_mesh = np.meshgrid(az_rad, el_rad, indexing='ij')
    az_flat = az_mesh.ravel()
    el_flat = el_mesh.ravel()

    # Spherical to Cartesian (azimuth from y-axis, elevation from xy-plane)
    dirs = np.stack([
        np.sin(az_flat) * np.cos(el_flat),   # x
        np.cos(az_flat) * np.cos(el_flat),   # y
        np.sin(el_flat),                      # z
    ], axis=1)

    return dirs  # (n_directions, 3)

def srp_phat_2d(signals, mic_positions, fs, c, az_grid_deg, el_grid_deg,
                fmin=100.0, fmax=2000.0):
    """2D SRP-PHAT beamforming.

    signals: (n_mics, n_samples)
    mic_positions: (3, n_mics)
    Returns: (n_az, n_el) power map
    """
    n_mics, n_samples = signals.shape
    dirs = build_steering_vectors_2d(az_grid_deg, el_grid_deg)
    n_dirs = dirs.shape[0]

    # FFT once per mic (same as POC)
    X, nfft, max_shift, band_mask = prepare_fft(signals, fs, fmin, fmax)

    srp = np.zeros(n_dirs, dtype=np.float64)
    pairs = list(itertools.combinations(range(n_mics), 2))

    for m, n in pairs:
        cc = gcc_phat_from_fft(X[m], X[n], nfft, max_shift, band_mask)
        delta_p = mic_positions[:, m] - mic_positions[:, n]
        tdoa_pred = dirs @ delta_p / c
        shift_pred = np.round(tdoa_pred * fs).astype(int)
        shift_pred = np.clip(shift_pred, -max_shift + 1, max_shift - 1)
        srp += cc[shift_pred + max_shift]

    return srp.reshape(len(az_grid_deg), len(el_grid_deg))
```

### Pattern 3: Pydantic Settings for Configuration
**What:** Use pydantic-settings BaseSettings for typed env var configuration with defaults.
**When to use:** INF-03 configuration requirement.
**Example:**
```python
from pydantic_settings import BaseSettings

class AcousticSettings(BaseSettings):
    # Audio
    audio_device: str | None = None  # None = auto-detect
    audio_source: str = "hardware"   # "hardware" or "simulated"
    sample_rate: int = 48000
    num_channels: int = 16
    chunk_seconds: float = 0.15

    # Beamforming
    freq_min: float = 100.0
    freq_max: float = 2000.0
    az_range: float = 90.0           # +/- degrees
    el_range: float = 45.0           # +/- degrees
    az_resolution: float = 1.0       # degrees per step
    el_resolution: float = 1.0       # degrees per step
    noise_percentile: float = 95.0   # BF-04 threshold percentile
    noise_margin: float = 1.5        # BF-04 margin multiplier

    # Service
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "ACOUSTIC_"}
```

### Pattern 4: FastAPI Lifespan for Audio Pipeline
**What:** Use FastAPI lifespan context manager to start/stop the audio capture and beamforming pipeline.
**When to use:** INF-04 health check, clean startup/shutdown.
**Example:**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: detect device, start capture
    settings = AcousticSettings()
    device_info = detect_uma16v2()
    capture = AudioCapture(...)
    capture.start()
    app.state.capture = capture
    app.state.device_info = device_info
    yield
    # Shutdown: stop capture
    capture.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device_detected": app.state.device_info is not None,
        "pipeline_running": app.state.capture.stream.active,
        "overflow_count": app.state.capture.ring._overflow_count,
    }
```

### Anti-Patterns to Avoid
- **Blocking audio capture in the main thread:** The POC uses `sd.rec(blocking=True)` in a loop. This blocks the event loop and prevents concurrent REST/WebSocket serving. Use callback-based InputStream per D-01.
- **Allocating memory in the PortAudio callback:** The callback runs on a real-time audio thread. Any allocation, logging, or GIL acquisition risks xruns. Only do array copy into pre-allocated buffer.
- **Per-channel WAV files for capture:** The POC records per-channel WAVs. For beamforming, capture all 16 channels interleaved in a single stream for efficiency.
- **Hardcoded device names:** The POC hardcodes `hw:3,0`. Use auto-detection by scanning `sd.query_devices()` for "UMA16v2" string, with env var override.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Env var parsing + validation | Manual os.environ parsing | pydantic-settings BaseSettings | Type coercion, defaults, validation, nested config, env prefix -- all free |
| Audio device enumeration | Manual /dev/snd scanning | sounddevice.query_devices() | Cross-platform, returns structured device info with channel counts |
| FFT computation | Custom DFT | numpy.fft.rfft | NumPy's FFTPACK/pocketfft is orders of magnitude faster than any handwritten DFT |
| ASGI server | Custom TCP server | Uvicorn | Production-grade, handles signals, graceful shutdown, workers |

**Key insight:** The beamforming math (SRP-PHAT, GCC-PHAT) IS worth hand-rolling because the POC proves it works in ~180 lines of NumPy and gives full control over the algorithm. Everything else should use established libraries.

## Common Pitfalls

### Pitfall 1: PortAudio Callback Thread Safety
**What goes wrong:** Calling Python functions that acquire the GIL (logging, print, list.append) in the PortAudio callback causes audio glitches (xruns) and dropped frames.
**Why it happens:** The PortAudio callback runs on a real-time audio thread. The GIL and memory allocation are not real-time safe.
**How to avoid:** Only do NumPy array copy (memcpy) in the callback. Set a flag or increment a counter for errors. Log xruns from the consumer thread, not the callback.
**Warning signs:** `status.input_overflow` or `status.input_underflow` in callback status flags.

### Pitfall 2: Docker on macOS Cannot Pass Through USB Audio
**What goes wrong:** `--device /dev/snd` does not work on macOS because Docker Desktop runs in a Linux VM without USB passthrough to host audio devices.
**Why it happens:** Docker on macOS uses a HyperKit/Apple Hypervisor VM. USB devices are not mapped through.
**How to avoid:** Always implement and test with the simulated audio source (D-04) on macOS. Real hardware testing requires Linux. The Dockerfile is correct for Linux deployment.
**Warning signs:** `sounddevice.query_devices()` returns empty list or no UMA-16v2 device inside the container on macOS.

### Pitfall 3: 2D SRP-PHAT Performance with Fine Resolution
**What goes wrong:** 1-degree resolution over +/-90 az x +/-45 el = 181 x 91 = 16,471 directions. With 120 mic pairs (C(16,2)), each frame evaluates ~2M GCC-PHAT lookups. This can exceed the 150ms budget.
**Why it happens:** The 2D grid is much larger than the POC's 61-point 1D grid.
**How to avoid:** Start with coarser resolution (2-3 degree steps) for real-time operation. Profile on target hardware. Vectorize the inner loop (the POC already uses vectorized NumPy indexing for shift lookups). Consider a two-pass approach: coarse scan then fine-scan around detected peaks.
**Warning signs:** Beamforming processing time exceeds chunk interval (150ms), causing ring buffer to fill up.

### Pitfall 4: Elevation Ambiguity with Planar Array
**What goes wrong:** A planar 4x4 array has limited elevation resolution due to its small aperture in the z-direction (it has zero aperture in z -- all mics are coplanar).
**Why it happens:** A planar array can resolve azimuth well but has front/back ambiguity and poor elevation discrimination because there is no z-axis baseline.
**How to avoid:** Document this limitation. The elevation axis in the spatial map will show less distinct peaks than azimuth. The primary useful output is azimuth. Elevation may show broad lobes rather than sharp peaks. This is a physics limitation, not a software bug.
**Warning signs:** Elevation peaks are much broader than azimuth peaks in the spatial map.

### Pitfall 5: NumPy 2.x Compatibility
**What goes wrong:** NumPy 2.0+ changed some C API contracts and deprecated aliases. Some libraries that depend on NumPy's C API may break.
**Why it happens:** NumPy 2.0 was a major breaking release.
**How to avoid:** Pin `numpy>=1.26,<3` per CLAUDE.md. The POC's pure-NumPy code works fine with NumPy 2.x. The risk is from downstream libraries (SciPy, soundfile). Both SciPy 1.14+ and soundfile 0.13+ support NumPy 2.x.
**Warning signs:** ImportError or DeprecationWarning at import time.

## Code Examples

### UMA-16v2 Device Detection
```python
# Source: POC device handling + audio-data metadata JSON
import sounddevice as sd

def detect_uma16v2() -> dict | None:
    """Find UMA-16v2 device in available audio devices.

    Returns device info dict or None if not found.
    Matches against known device name from real recordings:
    "UMA16v2: USB Audio (hw:X,0)"
    """
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if "UMA16v2" in dev["name"] or "uma16v2" in dev["name"].lower():
            if dev["max_input_channels"] >= 16:
                return {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
    return None
```

### Mic Position Geometry (direct port from POC)
```python
# Source: POC radar_gui_all_mics_fast_drone.py lines 26-72
import numpy as np

SPACING = 0.042  # meters, 42mm between adjacent mics
NUM_CHANNELS = 16

def build_mic_positions() -> np.ndarray:
    """Build UMA-16v2 mic positions as (3, 16) array.

    Channel-to-position mapping from mechanical drawing (top view):
        Row 0 (top):    MIC8   MIC7   MIC10  MIC9
        Row 1:          MIC6   MIC5   MIC12  MIC11
        Row 2:          MIC4   MIC3   MIC14  MIC13
        Row 3 (bottom): MIC2   MIC1   MIC16  MIC15
    """
    mic_rc = {
        8: (0,0), 7: (0,1), 10: (0,2), 9: (0,3),
        6: (1,0), 5: (1,1), 12: (1,2), 11: (1,3),
        4: (2,0), 3: (2,1), 14: (2,2), 13: (2,3),
        2: (3,0), 1: (3,1), 16: (3,2), 15: (3,3),
    }
    xs = np.array([-1.5, -0.5, 0.5, 1.5]) * SPACING
    ys = np.array([+1.5, +0.5, -0.5, -1.5]) * SPACING

    positions = np.zeros((3, NUM_CHANNELS))
    for ch in range(NUM_CHANNELS):
        row, col = mic_rc[ch + 1]
        positions[0, ch] = xs[col]
        positions[1, ch] = ys[row]
        # positions[2, ch] = 0.0  (planar array)

    return positions
```

### Simulated Audio Source (D-04)
```python
# Synthetic 16-channel audio with configurable direction-of-arrival
import numpy as np

def generate_simulated_chunk(
    mic_positions: np.ndarray,  # (3, n_mics)
    fs: int,
    chunk_samples: int,
    source_az_deg: float,
    source_el_deg: float = 0.0,
    freq: float = 500.0,
    c: float = 343.0,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Generate synthetic 16-channel audio from a point source.

    Returns: (chunk_samples, n_mics) array matching sounddevice format.
    """
    n_mics = mic_positions.shape[1]
    az_rad = np.deg2rad(source_az_deg)
    el_rad = np.deg2rad(source_el_deg)

    # Source direction unit vector
    d = np.array([
        np.sin(az_rad) * np.cos(el_rad),
        np.cos(az_rad) * np.cos(el_rad),
        np.sin(el_rad),
    ])

    # Time delays per mic (positive = arrives later)
    delays = mic_positions.T @ d / c  # (n_mics,)

    t = np.arange(chunk_samples) / fs  # (samples,)
    signal = np.sin(2 * np.pi * freq * (t[:, None] - delays[None, :]))

    # Add noise
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power) * np.random.randn(chunk_samples, n_mics)

    return (signal + noise).astype(np.float32)
```

### Percentile-Based Noise Threshold (BF-04)
```python
# Source: D-08 decision
import numpy as np

def detect_peak_with_threshold(
    srp_map: np.ndarray,       # (n_az, n_el) beamforming power map
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    percentile: float = 95.0,
    margin: float = 1.5,
) -> dict | None:
    """Find strongest peak above adaptive noise threshold.

    Returns dict with az_deg, el_deg, power, threshold or None if no detection.
    """
    threshold = np.percentile(srp_map, percentile) * margin
    max_val = np.max(srp_map)

    if max_val < threshold:
        return None  # No detection above noise floor

    az_idx, el_idx = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    return {
        "az_deg": float(az_grid_deg[az_idx]),
        "el_deg": float(el_grid_deg[el_idx]),
        "power": float(max_val),
        "threshold": float(threshold),
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sd.rec() blocking | sd.InputStream callback | Always available | Non-blocking, real-time safe, decouples capture from processing |
| Flask sync | FastAPI async + lifespan | 2020+ | Lifespan replaces on_event("startup"), cleaner resource management |
| @app.on_event("startup") | lifespan context manager | FastAPI 0.93+ | on_event is deprecated, lifespan is the current pattern |
| python-dotenv + os.environ | pydantic-settings | 2023+ | Typed, validated, IDE-friendly config |
| flake8 + black + isort | Ruff | 2023+ | Single tool, 100x faster, drop-in replacement |

**Deprecated/outdated:**
- `@app.on_event("startup")` / `@app.on_event("shutdown")` -- use lifespan context manager instead
- `sd.rec()` with blocking=True for real-time -- use InputStream callback

## Open Questions

1. **UMA-16v2 Channel Mapping Verification**
   - What we know: POC defines a channel-to-position mapping based on mechanical drawing. STATE.md flags this needs empirical verification (tap test).
   - What's unclear: Whether USB channel order exactly matches the documented MIC numbering.
   - Recommendation: Use the POC mapping as-is for Phase 1. The simulated audio source bypasses this issue. Flag for hardware validation when device is available.

2. **2D SRP-PHAT Performance Budget**
   - What we know: POC's 1D scan (61 angles, 120 pairs) runs well within 150ms. 2D grid at 1-degree resolution is ~270x more directions.
   - What's unclear: Exact timing on target hardware (RPi, x86 edge device, etc.).
   - Recommendation: Start with 3-degree resolution (61 x 31 = 1,891 directions), profile, then refine. Keep resolution configurable per D-06.

3. **Ring Buffer Consumer Pattern**
   - What we know: D-01 says beamforming reads "asynchronously" from ring buffer.
   - What's unclear: Whether consumer should poll, use threading.Event, or use asyncio integration.
   - Recommendation: Use a threading.Event that the callback sets after each write. Consumer thread waits on event, processes, then clears. This avoids busy-polling and CPU waste.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Docker base image | N/A (Docker pulls) | 3.11-slim | -- |
| Docker | INF-01 | Yes | 28.3.2 | -- |
| ALSA / /dev/snd | AUD-01 (hardware) | No (macOS host) | -- | Simulated audio source (D-04) |
| UMA-16v2 USB | AUD-01, AUD-03 | No (dev machine) | -- | Simulated audio source (D-04) |
| PortAudio | sounddevice dependency | In Docker image | -- | Installed via libportaudio2 in Dockerfile |

**Missing dependencies with no fallback:**
- None -- the simulated audio source (D-04) makes all hardware dependencies optional for development.

**Missing dependencies with fallback:**
- ALSA + UMA-16v2: Not available on macOS dev machine. Use `AUDIO_SOURCE=simulated` for all development and testing. Real hardware testing requires Linux host with USB mic array.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 + pytest-asyncio 1.3.0 |
| Config file | pyproject.toml [tool.pytest.ini_options] -- Wave 0 |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ -x -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUD-01 | Callback-based 16-ch capture at 48kHz | unit (simulated) | `pytest tests/unit/test_capture.py -x` | Wave 0 |
| AUD-02 | Ring buffer write/read without drops | unit | `pytest tests/unit/test_ring_buffer.py -x` | Wave 0 |
| AUD-03 | Device detection reports UMA-16v2 presence | unit (mocked) | `pytest tests/unit/test_device.py -x` | Wave 0 |
| BF-01 | SRP-PHAT produces spatial map from 16-ch audio | unit | `pytest tests/unit/test_srp_phat.py -x` | Wave 0 |
| BF-02 | Frequency band configuration changes filtering | unit | `pytest tests/unit/test_srp_phat.py::test_freq_band -x` | Wave 0 |
| BF-03 | Peak detection returns correct az/el degrees | unit | `pytest tests/unit/test_peak.py -x` | Wave 0 |
| BF-04 | Percentile noise threshold filters false detections | unit | `pytest tests/unit/test_peak.py::test_noise_threshold -x` | Wave 0 |
| INF-01 | Docker container builds and starts | integration | `docker build -t acoustic . && docker run --rm acoustic python -c "import acoustic"` | Wave 0 |
| INF-02 | Dockerfile exists (single-stage for Phase 1) | manual | Visual inspection | Wave 0 |
| INF-03 | Env vars configure device, ports, freq band | unit | `pytest tests/unit/test_config.py -x` | Wave 0 |
| INF-04 | Health endpoint returns device + pipeline status | integration | `pytest tests/integration/test_health.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x -q`
- **Per wave merge:** `pytest tests/ -x -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `pyproject.toml` -- project config with [tool.pytest.ini_options] (asyncio_mode = "auto")
- [ ] `tests/conftest.py` -- shared fixtures (mic positions, synthetic audio generator, mock device list)
- [ ] `tests/unit/test_ring_buffer.py` -- AUD-02
- [ ] `tests/unit/test_srp_phat.py` -- BF-01, BF-02
- [ ] `tests/unit/test_peak.py` -- BF-03, BF-04
- [ ] `tests/unit/test_config.py` -- INF-03
- [ ] `tests/unit/test_device.py` -- AUD-03
- [ ] `tests/unit/test_capture.py` -- AUD-01
- [ ] `tests/integration/test_health.py` -- INF-04
- [ ] Framework install: `pip install pytest pytest-asyncio httpx`

## Project Constraints (from CLAUDE.md)

- **Python >=3.11**: Use `python:3.11-slim` base Docker image
- **FastAPI >=0.135**: Async-native, use lifespan pattern (not deprecated on_event)
- **sounddevice >=0.5.1**: Proven with UMA-16v2 in POC
- **NumPy >=1.26,<3**: Conservative pinning per CLAUDE.md version strategy
- **SciPy >=1.14**: For signal processing utilities
- **Ruff >=0.9**: Linting + formatting (replaces flake8+black+isort)
- **pytest >=8.0 + pytest-asyncio**: Testing framework
- **Custom SRP-PHAT over Acoular**: POC's 180-line implementation is simpler for 4x4 array
- **Docker single container**: With ALSA libs, no PulseAudio
- **ZeroMQ not in Phase 1**: Messaging comes in Phase 3
- **No web UI in Phase 1**: React frontend comes in Phase 2
- **GSD Workflow**: Use GSD commands for file changes, do not make direct repo edits outside workflow

## Sources

### Primary (HIGH confidence)
- POC `radar_gui_all_mics_fast_drone.py` -- SRP-PHAT algorithm, mic geometry, audio constants (directly read)
- `audio-data/data/background/*.json` -- Real UMA-16v2 device metadata (directly read)
- PyPI version checks -- fastapi 0.135.2, sounddevice 0.5.5, uvicorn 0.42.0, numpy 2.4.4, scipy 1.17.1, pytest 9.0.2 (verified via `pip3 index versions`)
- CONTEXT.md decisions D-01 through D-11 (directly read)

### Secondary (MEDIUM confidence)
- [sounddevice API docs](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html) -- InputStream callback signature, threading model
- [Pyroomacoustics SRP-PHAT](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.srp.html) -- Reference for 2D SRP-PHAT with azimuth+elevation
- [PySDR 2D Beamforming](https://pysdr.org/content/2d_beamforming.html) -- 2D beamforming with planar arrays
- [Docker ALSA sound access](https://github.com/mviereck/x11docker/wiki/Container-sound:-ALSA-or-Pulseaudio) -- Docker --device /dev/snd pattern
- [FastAPI health check patterns](https://www.index.dev/blog/how-to-implement-health-check-in-python) -- Lifespan + health endpoint

### Tertiary (LOW confidence)
- [Docker for Mac audio limitation](https://github.com/docker/for-mac/issues/6789) -- macOS cannot pass USB audio to containers (community issue, matches known architecture)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All versions verified against PyPI, libraries proven in POC
- Architecture: HIGH -- Callback + ring buffer pattern is well-established, POC SRP-PHAT is working code, 2D extension is straightforward math
- Pitfalls: HIGH -- Callback thread safety is well-documented, macOS Docker limitation confirmed by multiple sources, elevation ambiguity is physics

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (stable domain, 30 days)
