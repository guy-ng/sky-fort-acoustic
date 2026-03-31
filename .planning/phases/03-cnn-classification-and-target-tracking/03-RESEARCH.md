# Phase 3: CNN Classification and Target Tracking - Research

**Researched:** 2026-03-31
**Domain:** ML inference (ONNX Runtime), audio preprocessing (mel-spectrogram), state machines, ZeroMQ PUB/SUB
**Confidence:** HIGH

## Summary

Phase 3 replaces the placeholder target system with real CNN-based drone detection, a hysteresis state machine for detection stability, UUID-based target tracking, and ZeroMQ event publishing. The core technical challenge is replicating the POC's mel-spectrogram preprocessing pipeline exactly so the converted ONNX model produces valid predictions.

The POC uses TensorFlow/Keras for inference and librosa for mel-spectrogram computation. The user has decided to convert the .h5 model to ONNX format and use `onnxruntime` for inference (D-02), which eliminates the heavy TensorFlow dependency in production. The mel-spectrogram pipeline must use librosa (or an exact numeric equivalent) to match the training-time preprocessing. The CNNWorker pattern from the POC (background thread with maxsize=1 queue) maps cleanly to the existing `BeamformingPipeline` threading architecture.

ZeroMQ PUB/SUB via pyzmq is straightforward. The hysteresis state machine from the POC (NO_DRONE / CANDIDATE / CONFIRMED with enter=0.80 / exit=0.40 thresholds) provides a proven starting point. The main integration work involves wiring the CNN worker into the pipeline loop, replacing `placeholder_target_from_peak()`, and adding the ZMQ publisher to the FastAPI lifespan.

**Primary recommendation:** Use librosa for mel-spectrogram preprocessing (exact POC match), onnxruntime for inference, port the POC's 3-state hysteresis machine, and use pyzmq's synchronous PUB socket in a dedicated background thread.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Binary drone/not-drone detection only. Single sigmoid output producing drone probability. Drone type classification (CLS-02) deferred to ms-2.
- **D-02:** Use the POC's existing trained model (`uav_melspec_cnn.h5`) converted to ONNX format. Inference via `onnxruntime` (lightweight ~50MB, no TF dependency in production). One-time conversion using `tf2onnx`.
- **D-03:** Mel-spectrogram pipeline replicates POC exactly: resample to 16kHz, 2-second segments, 64 mels, n_fft=1024, hop_length=256, pad/trim to 128 frames, mean/std normalization. Input shape: (1, 128, 64, 1).
- **D-04:** CNN inference gated on beamforming peak detection. Only run the model when SRP-PHAT detects a peak above the noise threshold. Saves compute -- CNN doesn't run on silence. Matches POC's ENERGY+MODEL gating pattern.
- **D-05:** Hysteresis state machine prevents detection flickering (CLS-03). Enter/exit thresholds and confirmation hit counts are implementation details for Claude to decide.
- **D-06:** Target created with UUID on first confirmed detection. Target persists for 5 seconds after last detection signal before being marked as lost.
- **D-07:** Doppler speed estimation (TRK-02) deferred to ms-2. `speed_mps` field remains `null` in this phase.
- **D-08:** Single topic `acoustic/targets` for all events. Messages include an `event` field with values: `new`, `update`, `lost`.
- **D-09:** JSON message schema carries: event type, target ID, bearing (az/el degrees), drone probability/confidence. Speed field present but null until ms-2.
- **D-10:** PUB/SUB pattern. Publisher binds to configurable endpoint (env var). Subscribers connect and filter by topic prefix.

### Claude's Discretion
- Hysteresis thresholds and confirmation hit counts (D-05 details)
- CNN worker threading model (background thread with queue, matching POC's CNNWorker pattern)
- ZeroMQ publish frequency for update events
- How the CNN inference integrates with the existing `BeamformingPipeline` class
- Multi-target handling when beamforming detects multiple peaks simultaneously
- Web UI updates to show real detection data instead of placeholder targets

### Deferred Ideas (OUT OF SCOPE)
- **Drone type classification (CLS-02)** -- Multi-class deferred to milestone 2
- **Doppler speed estimation (TRK-02)** -- Deferred to milestone 2
- **CNN training pipeline** -- Phase 5 as planned. Phase 3 ships pre-trained model only.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLS-01 | CNN inference on audio segments for drone/not-drone classification | ONNX Runtime inference session with mel-spectrogram preprocessing pipeline (D-02, D-03) |
| CLS-02 | Multi-class drone type classification | DEFERRED to ms-2 (D-01, D-11). Binary only in this phase. |
| CLS-03 | Hysteresis state machine with enter/exit thresholds | POC's 3-state machine (NO_DRONE/CANDIDATE/CONFIRMED) with P_DRONE_ENTER=0.80, P_DRONE_EXIT=0.40 |
| CLS-04 | Load CNN model from configurable file path at startup | `ACOUSTIC_CNN_MODEL_PATH` env var in AcousticSettings, loaded during FastAPI lifespan |
| TRK-01 | UUID target ID assigned on first detection, maintained until lost | UUID4 on CANDIDATE->CONFIRMED transition, 5s TTL after last signal (D-06) |
| TRK-02 | Doppler speed estimation | DEFERRED to ms-2 (D-07, D-12). `speed_mps=null` in all messages. |
| TRK-03 | ZeroMQ detection event with target ID and drone class | ZMQ PUB on `acoustic/targets` topic, `event: "new"` on first confirmed detection |
| TRK-04 | Periodic ZeroMQ update events with speed, bearing per target | ZMQ `event: "update"` published periodically while target is active |
| TRK-05 | ZeroMQ PUB/SUB with JSON message schema | pyzmq PUB socket, JSON-serialized messages with defined schema (D-08, D-09, D-10) |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnxruntime | >=1.20,<2.0 | CNN inference | Locked decision D-02. Lightweight (~50MB), no TF dependency. Already installed (1.20.1). Latest is 1.24.4. |
| librosa | >=0.11.0 | Mel-spectrogram computation | POC uses librosa for make_melspec(). Must match exactly for model compatibility (D-03). |
| pyzmq | >=27.1.0 | ZeroMQ PUB/SUB | Project requirement. Already installed (27.1.0). Brokerless, low-latency. |
| numpy | >=1.26,<3 | Array operations | Already in stack. Audio resampling, normalization. |
| scipy | >=1.14 | Signal resampling | Already in stack. `scipy.signal.resample_poly` for 48kHz->16kHz. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tf2onnx | >=1.16 | Model conversion | One-time dev dependency. Convert .h5 to .onnx. Not needed at runtime. |
| tensorflow | >=2.15 | Model conversion | One-time dev dependency for tf2onnx. Not shipped in production container. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| librosa | scipy.signal + manual mel filterbank | Risk of numeric mismatch with POC training. librosa adds ~20MB but guarantees exact preprocessing. |
| librosa | torchaudio.transforms.MelSpectrogram | Would require PyTorch runtime (~2GB). Overkill for inference-only phase. |
| onnxruntime | PyTorch + torch.load | PyTorch is ~2GB. ONNX Runtime is ~50MB. Model was trained in TF/Keras anyway. |

**Installation (runtime):**
```bash
pip install "onnxruntime>=1.20,<2.0" "librosa>=0.11.0" "pyzmq>=27.1.0"
```

**Installation (model conversion, dev only):**
```bash
pip install "tf2onnx>=1.16" "tensorflow>=2.15"
python -m tf2onnx.convert --keras uav_melspec_cnn.h5 --output uav_melspec_cnn.onnx
```

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/
  classification/
    __init__.py
    inference.py      # OnnxDroneClassifier — loads model, runs mel-spec + ONNX inference
    preprocessing.py  # make_melspec(), pad_or_trim(), norm_spec() — exact POC port
    state_machine.py  # DetectionStateMachine — hysteresis logic
  tracking/
    __init__.py
    tracker.py        # TargetTracker — UUID assignment, TTL, multi-target management
    publisher.py      # ZmqPublisher — PUB socket, JSON serialization
    schema.py         # TargetEvent Pydantic model (message schema)
  config.py           # Extended AcousticSettings with CNN + ZMQ fields
  pipeline.py         # Modified to integrate CNN worker
  types.py            # Extended with detection-related types
```

### Pattern 1: CNN Worker Thread (from POC's CNNWorker)
**What:** Background thread with maxsize=1 queue for non-blocking CNN inference
**When to use:** Always -- CNN inference takes 10-50ms, must not block the beamforming loop
**Example:**
```python
# Port of POC's CNNWorker pattern
import queue
import threading
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    drone_probability: float
    timestamp: float
    az_deg: float
    el_deg: float

class CNNWorker:
    def __init__(self, classifier: OnnxDroneClassifier):
        self._classifier = classifier
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: ClassificationResult | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def push(self, mono_audio: np.ndarray, az_deg: float, el_deg: float) -> None:
        """Non-blocking push. Drops if queue full (latest-only semantics)."""
        try:
            self._queue.put_nowait((mono_audio, az_deg, el_deg))
        except queue.Full:
            pass

    def get_latest(self) -> ClassificationResult | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                mono_audio, az, el = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            prob = self._classifier.predict(mono_audio)
            with self._lock:
                self._latest = ClassificationResult(
                    drone_probability=prob,
                    timestamp=time.monotonic(),
                    az_deg=az, el_deg=el,
                )
```

### Pattern 2: Hysteresis State Machine (from POC)
**What:** 3-state machine preventing detection flickering
**When to use:** Every CNN result is fed through this before creating/updating targets
**Example:**
```python
from enum import Enum

class DetectionState(str, Enum):
    NO_DRONE = "NO_DRONE"
    CANDIDATE = "DRONE_CANDIDATE"
    CONFIRMED = "DRONE_CONFIRMED"

class DetectionStateMachine:
    def __init__(
        self,
        enter_threshold: float = 0.80,
        exit_threshold: float = 0.40,
        confirm_hits: int = 2,      # consecutive hits to go CANDIDATE -> CONFIRMED
    ):
        self.state = DetectionState.NO_DRONE
        self._enter = enter_threshold
        self._exit = exit_threshold
        self._confirm_hits = confirm_hits
        self._hit_count = 0

    def update(self, drone_probability: float) -> DetectionState:
        if self.state == DetectionState.NO_DRONE:
            if drone_probability >= self._enter:
                self.state = DetectionState.CANDIDATE
                self._hit_count = 1
        elif self.state == DetectionState.CANDIDATE:
            if drone_probability >= self._enter:
                self._hit_count += 1
                if self._hit_count >= self._confirm_hits:
                    self.state = DetectionState.CONFIRMED
            elif drone_probability <= self._exit:
                self.state = DetectionState.NO_DRONE
                self._hit_count = 0
        elif self.state == DetectionState.CONFIRMED:
            if drone_probability <= self._exit:
                self.state = DetectionState.CANDIDATE
                self._hit_count = 0
        return self.state
```

### Pattern 3: ZeroMQ Publisher
**What:** PUB socket bound to configurable endpoint, sends JSON messages on topic
**When to use:** On every state transition (new/lost) and periodically for active targets (update)
**Example:**
```python
import json
import zmq

class ZmqPublisher:
    def __init__(self, endpoint: str = "tcp://*:5556"):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(endpoint)

    def publish(self, event: dict) -> None:
        topic = b"acoustic/targets"
        payload = json.dumps(event).encode("utf-8")
        self._sock.send_multipart([topic, payload])

    def close(self) -> None:
        self._sock.close()
```

### Pattern 4: Pipeline Integration
**What:** Modify the beamforming pipeline loop to gate CNN inference on peak detection
**When to use:** In `_run_loop()` -- when `process_chunk()` returns a PeakDetection, feed audio to CNN worker
**Example:**
```python
# In pipeline._run_loop():
peak = self.process_chunk(chunk)
if peak is not None:
    # Mix 16-channel to mono, collect 2s segment
    mono = chunk.mean(axis=1)  # (samples,) float32
    self._mono_ring.extend(mono)
    if len(self._mono_ring) >= self._cnn_segment_samples:
        segment = np.array(self._mono_ring, dtype=np.float32)
        self._cnn_worker.push(segment, peak.az_deg, peak.el_deg)
```

### Anti-Patterns to Avoid
- **Running CNN in the beamforming thread:** CNN inference (10-50ms) blocks the 150ms chunk processing loop. Always use a separate worker thread.
- **Using TensorFlow at runtime:** The model was trained in TF but should be served via ONNX Runtime. Do NOT import tensorflow in production code.
- **Custom mel-spectrogram implementation:** Any numeric deviation from librosa's implementation will produce wrong predictions. Use librosa even though it adds a dependency.
- **Blocking ZMQ publish:** Use `zmq.NOBLOCK` flag or publish from a non-critical path. A slow subscriber must not block the detection loop.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mel-spectrogram | Custom FFT + mel filterbank | `librosa.feature.melspectrogram()` | Numeric equivalence with POC training is critical. Even small differences in windowing or mel filter shape will degrade model accuracy. |
| ONNX inference | Custom model loading | `onnxruntime.InferenceSession` | Handles graph optimization, memory management, threading. 3 lines of code. |
| Audio resampling | Manual interpolation | `scipy.signal.resample_poly()` or `librosa.resample()` | Anti-aliasing, polyphase filtering. POC's `fast_resample()` already uses this. |
| ZeroMQ messaging | Raw TCP sockets | `pyzmq` | Connection management, topic filtering, message framing all handled. |
| Model conversion | Manual weight export | `tf2onnx.convert.from_keras()` | Handles all TF/Keras op mappings, opset versioning. One command. |

**Key insight:** The mel-spectrogram pipeline is the most critical "don't hand-roll" item. The CNN model learned features from librosa's specific implementation of mel filterbanks. Any alternative must produce bit-identical output or model accuracy degrades.

## Common Pitfalls

### Pitfall 1: Mel-Spectrogram Numeric Mismatch
**What goes wrong:** Model produces random/degraded predictions despite correct architecture
**Why it happens:** Training used librosa with specific parameters; inference uses different FFT windowing, mel filter shape, or normalization
**How to avoid:** Use exact same librosa call: `melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=64, power=2.0)` followed by `power_to_db(S, ref=np.max)` then transpose `.T`
**Warning signs:** Inference always returns ~0.5 (random), or always >0.99 (saturated)

### Pitfall 2: ONNX Input Shape Mismatch
**What goes wrong:** ONNX Runtime raises shape error or produces garbage
**Why it happens:** Keras model expects (batch, height, width, channels) = (1, 128, 64, 1) in NHWC format. Common mistake is omitting the channel dimension or using wrong axis order.
**How to avoid:** Verify with `session.get_inputs()[0].shape` after loading. POC line 354: `X = spec.astype(np.float32)[None, :, :, None]`
**Warning signs:** Runtime shape errors, or silently wrong predictions

### Pitfall 3: Resampling Before Mono Mixdown
**What goes wrong:** Incorrect audio fed to CNN
**Why it happens:** 16-channel audio at 48kHz needs: (1) mix to mono, (2) resample to 16kHz. Order matters for anti-aliasing.
**How to avoid:** Mix to mono first (mean of all channels), then resample to 16kHz. POC mixes to mono before resampling.
**Warning signs:** Aliasing artifacts in spectrogram

### Pitfall 4: ZMQ Slow Subscriber Blocks Publisher
**What goes wrong:** Publisher's send buffer fills up, blocking the main thread
**Why it happens:** ZMQ PUB socket has a high-water mark (default 1000). If subscriber is slow, messages queue up.
**How to avoid:** Set `socket.setsockopt(zmq.SNDHWM, 100)` and use `zmq.NOBLOCK` flag on send. Drop messages rather than block.
**Warning signs:** Pipeline thread hangs intermittently

### Pitfall 5: Target Flickering Despite Hysteresis
**What goes wrong:** Target IDs change rapidly, UI shows flickering markers
**Why it happens:** State machine exits CONFIRMED too quickly, new UUID assigned on re-entry
**How to avoid:** Keep the 5-second TTL (D-06) -- once a target is CONFIRMED, it stays alive for 5s after last signal. Don't create a new UUID until the old target is fully lost.
**Warning signs:** Multiple UUID values in rapid succession for same physical drone

### Pitfall 6: librosa Import Time
**What goes wrong:** Service startup takes 3-5 seconds longer than expected
**Why it happens:** librosa imports numba on first use, which triggers JIT compilation
**How to avoid:** Import librosa during startup (in lifespan), not on first inference. Accept the one-time cost. Set `NUMBA_CACHE_DIR` for container caching.
**Warning signs:** First CNN inference takes significantly longer than subsequent ones

## Code Examples

### Mel-Spectrogram Preprocessing (exact POC port)
```python
# Source: POC-code/PT520/PTZ/uma16_master_live_with_polar.py lines 313-356
import librosa
import numpy as np
from scipy.signal import resample_poly
from math import gcd

SR_CNN = 16000
CNN_SEGMENT_SECONDS = 2.0
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_FRAMES = 128

def fast_resample(y: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample audio using polyphase method (matches POC)."""
    if fs_in == fs_out:
        return y
    g = gcd(int(fs_in), int(fs_out))
    up = fs_out // g
    down = fs_in // g
    return resample_poly(y, up, down).astype(np.float32)

def make_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel-spectrogram exactly matching POC."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T  # (frames, n_mels)

def pad_or_trim(spec: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Pad or trim spectrogram to fixed frame count."""
    frames, n_mels = spec.shape
    if frames < max_frames:
        spec = np.pad(spec, ((0, max_frames - frames), (0, 0)), mode="constant")
    elif frames > max_frames:
        spec = spec[:max_frames, :]
    return spec

def norm_spec(spec: np.ndarray) -> np.ndarray:
    """Mean/std normalization matching POC."""
    m = float(np.mean(spec))
    s = float(np.std(spec)) + 1e-8
    return (spec - m) / s

def preprocess_for_cnn(mono_audio: np.ndarray, fs_in: int) -> np.ndarray:
    """Full preprocessing pipeline: resample -> melspec -> pad -> normalize -> reshape.

    Returns: (1, 128, 64, 1) float32 array ready for ONNX inference.
    """
    y16 = fast_resample(mono_audio, fs_in, SR_CNN)
    target_len = int(SR_CNN * CNN_SEGMENT_SECONDS)
    if y16.size > target_len:
        y16 = y16[-target_len:]  # take last 2 seconds
    elif y16.size < target_len:
        y16 = np.pad(y16, (target_len - y16.size, 0), mode="constant")

    spec = make_melspec(y16, SR_CNN)
    spec = pad_or_trim(spec, MAX_FRAMES)
    spec = norm_spec(spec)
    return spec.astype(np.float32)[None, :, :, None]  # (1, 128, 64, 1)
```

### ONNX Runtime Inference
```python
# Source: https://onnxruntime.ai/docs/get-started/with-python.html
import onnxruntime as ort

class OnnxDroneClassifier:
    def __init__(self, model_path: str):
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        # Verify expected shape
        expected = self._session.get_inputs()[0].shape
        assert expected == [1, 128, 64, 1] or expected == [None, 128, 64, 1], (
            f"Model input shape mismatch: {expected}"
        )

    def predict(self, preprocessed: np.ndarray) -> float:
        """Run inference. Returns drone probability 0.0-1.0."""
        outputs = self._session.run(None, {self._input_name: preprocessed})
        return float(outputs[0][0, 0])  # sigmoid output
```

### ZeroMQ Message Schema
```python
# Source: CONTEXT.md D-08, D-09
{
    "event": "new",        # "new" | "update" | "lost"
    "target_id": "uuid-string",
    "class_label": "drone",  # "drone" | "background"
    "confidence": 0.92,
    "az_deg": 45.3,
    "el_deg": 12.1,
    "speed_mps": null,     # null until ms-2 (D-07)
    "timestamp": 1711900000.123
}
```

### Model Conversion (one-time, dev only)
```bash
# Source: https://github.com/onnx/tensorflow-onnx
pip install tf2onnx tensorflow
python -m tf2onnx.convert --keras uav_melspec_cnn.h5 --output models/uav_melspec_cnn.onnx --opset 13
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TF Keras .h5 inference | ONNX Runtime inference | Decided in CONTEXT.md D-02 | ~50MB vs ~2GB dependency. 2-5x faster inference on CPU. |
| TensorFlow model.predict() | ort.InferenceSession.run() | N/A | Different API but same result. Single function call. |
| librosa 0.10.x | librosa 0.11.0 | 2024 | Minor API changes. `melspectrogram()` API unchanged. |
| pyzmq 26.x | pyzmq 27.1.0 | 2024 | Stable API. No breaking changes for basic PUB/SUB. |

**Deprecated/outdated:**
- `keras2onnx` package: frozen at TF 2.3, use `tf2onnx` instead
- `tensorflow-onnx` < 1.16: use latest for best opset support

## Open Questions

1. **POC Model File Location**
   - What we know: Model path in POC is `/home/skyfortubuntu/Skyfort/Cursor/acoular/models/uav_melspec_cnn.h5`
   - What's unclear: Whether this file is accessible from the dev machine, or needs to be copied into the repo
   - Recommendation: Include a conversion script in the repo. Ship a placeholder/dummy ONNX model for testing. Document where to obtain the real .h5 file.

2. **Multi-Peak Handling**
   - What we know: Beamforming can detect multiple peaks (multiple drones)
   - What's unclear: Current `detect_peak_with_threshold()` returns a single PeakDetection
   - Recommendation: For Phase 3, support single-target tracking only. Multi-target requires modifying the beamforming peak detector to return multiple peaks, which is a separate concern. Document this as a future enhancement.

3. **CNN Inference Latency on Target Hardware**
   - What we know: ONNX Runtime on CPU typically does small CNN inference in 5-50ms
   - What's unclear: Actual latency on the deployment hardware (likely ARM or low-power x86)
   - Recommendation: Measure and log inference time. The 2-second CNN segment window plus the worker thread pattern means latency is not critical -- we have a 2-second budget.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| onnxruntime | CNN inference (CLS-01) | Yes | 1.20.1 | -- |
| pyzmq | Event publishing (TRK-03-05) | Yes | 27.1.0 | -- |
| numpy | Preprocessing | Yes | 1.26.4 | -- |
| scipy | Audio resampling | Yes | 1.15.2 | -- |
| librosa | Mel-spectrogram (D-03) | No | -- | Must install. pip install librosa>=0.11.0 |
| tf2onnx | Model conversion (one-time) | No | -- | Install in dev only. Not needed at runtime. |
| tensorflow | Model conversion (one-time) | No | -- | Install in dev only. Not needed at runtime. |

**Missing dependencies with no fallback:**
- librosa: Must be installed for exact POC mel-spectrogram matching. Add to requirements.

**Missing dependencies with fallback:**
- tf2onnx + tensorflow: Only needed once for model conversion. Can run conversion on any machine with TF installed, then commit the .onnx file.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLS-01 | CNN inference produces drone probability from audio | unit | `pytest tests/unit/test_inference.py -x` | Wave 0 |
| CLS-03 | Hysteresis state machine transitions correctly | unit | `pytest tests/unit/test_state_machine.py -x` | Wave 0 |
| CLS-04 | Model loads from configurable path | unit | `pytest tests/unit/test_inference.py::test_model_load -x` | Wave 0 |
| TRK-01 | UUID assigned on confirmed detection, TTL on lost | unit | `pytest tests/unit/test_tracker.py -x` | Wave 0 |
| TRK-03 | ZMQ publishes "new" event on detection | integration | `pytest tests/integration/test_zmq_publisher.py -x` | Wave 0 |
| TRK-04 | ZMQ publishes periodic "update" events | integration | `pytest tests/integration/test_zmq_publisher.py::test_update_events -x` | Wave 0 |
| TRK-05 | ZMQ messages follow JSON schema | unit | `pytest tests/unit/test_target_schema.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_preprocessing.py` -- covers mel-spectrogram pipeline (CLS-01)
- [ ] `tests/unit/test_inference.py` -- covers ONNX model loading and inference (CLS-01, CLS-04)
- [ ] `tests/unit/test_state_machine.py` -- covers hysteresis transitions (CLS-03)
- [ ] `tests/unit/test_tracker.py` -- covers UUID assignment, TTL, lost detection (TRK-01)
- [ ] `tests/unit/test_target_schema.py` -- covers ZMQ message schema validation (TRK-05)
- [ ] `tests/integration/test_zmq_publisher.py` -- covers ZMQ publish/subscribe (TRK-03, TRK-04)
- [ ] `tests/fixtures/dummy_model.onnx` -- small ONNX model for testing without real model file
- [ ] Framework install: `pip install librosa>=0.11.0` -- not currently installed

## Sources

### Primary (HIGH confidence)
- POC source: `POC-code/PT520/PTZ/uma16_master_live_with_polar.py` -- CNN pipeline (lines 313-403), state machine (lines 405-454), detection parameters (lines 60-107)
- Existing codebase: `src/acoustic/pipeline.py`, `src/acoustic/types.py`, `src/acoustic/api/routes.py`, `src/acoustic/api/websocket.py`, `src/acoustic/config.py`, `src/acoustic/main.py`
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html) -- InferenceSession API
- [ONNX Runtime Getting Started](https://onnxruntime.ai/docs/get-started/with-python.html) -- Usage patterns

### Secondary (MEDIUM confidence)
- [tf2onnx GitHub](https://github.com/onnx/tensorflow-onnx) -- Model conversion from Keras .h5 to ONNX
- [ONNX Runtime TF Getting Started](https://onnxruntime.ai/docs/tutorials/tf-get-started.html) -- Conversion tutorial
- [ZeroMQ PUB/SUB examples](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html) -- pyzmq patterns
- [librosa melspectrogram docs](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html) -- API reference

### Tertiary (LOW confidence)
- None -- all findings verified against primary sources or codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are locked decisions from CONTEXT.md or already in the project
- Architecture: HIGH -- POC provides proven patterns (CNNWorker, state machine), existing codebase has clear integration points
- Pitfalls: HIGH -- most pitfalls identified from POC code analysis and ONNX Runtime documentation
- Preprocessing: HIGH -- exact parameters documented in POC source code (line 313-356)

**Research date:** 2026-03-31
**Valid until:** 2026-04-30 (stable domain, locked technology choices)
