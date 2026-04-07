# Phase 21: Build Raspberry Pi 4 edge drone-detection app — Research

**Researched:** 2026-04-07
**Domain:** Edge ML deployment — ONNX Runtime on ARM64 Linux, single-mic audio capture, GPIO, systemd packaging
**Confidence:** HIGH on code-side patterns / MEDIUM on Pi-specific runtime facts (onnxruntime wheel stability, inference latency — require on-device validation)

## Summary

Phase 21 ships a standalone Raspberry Pi 4 edge app that runs `efficientat_mn10_v6.pt` (converted to ONNX on the host) against a single USB mic, with a hysteresis state machine driving a single GPIO LED and an always-on rotating JSONL detection log. The app is deliberately **not** part of the main Docker service — it is a new top-level `apps/rpi-edge/` tree with minimal dependencies, vendored preprocessing, bare-venv systemd install, and a localhost HTTP `/health` + `/status` endpoint.

The hardest correctness risk is **preprocessing drift**: the Pi must produce byte-identical mel spectrograms to what the training pipeline produced, because the ONNX model only sees mel features after the conversion boundary. The cleanest answer — and the one D-04 already commits to — is vendoring `preprocess.py` + `mel_banks_128_1024_32k.pt` from `src/acoustic/classification/efficientat/` into `apps/rpi-edge/` and locking a CI drift test.

The second-hardest risk is **onnxruntime ARM64 wheel behavior on Pi 4**: official `manylinux_2_27_aarch64` wheels exist on PyPI (verified up through 1.24.4 in March 2026 [CITED]), but there are known "illegal instruction" reports on Pi 4 with some recent versions when the wheel requires CPU features the Pi 4's Cortex-A72 doesn't support. The planner must pin a known-working version and leave a fallback path documented in the install script.

**Primary recommendation:** Use **onnxruntime 1.18.x** as the pinned Pi version (last line conservatively known-safe for generic armv8-a), `gpiozero` with the default `lgpio` backend (not `RPi.GPIO`), `scipy.signal.resample_poly(2, 3)` for 48→32 kHz (no new dep), stdlib `http.server` for the 2-endpoint JSON API (no FastAPI on Pi), `PyYAML` + `argparse` for config, and `logging.handlers.RotatingFileHandler` on a dedicated detection logger instance. Conversion on the host uses `torch.onnx.export(opset=17, dynamo=False, dynamic_axes={'input': {0: 'batch'}})` with a fixed time axis, followed by `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, op_types_to_quantize=['MatMul','Gemm'])`. A sanity-validation step runs both the FP32 and int8 ONNX artifacts against the PyTorch reference on a held-out sample set and asserts top-1 agreement ≥ 99% (FP32) and ≥ 97% (int8); failures abort the write.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Audio Input & Windowing**
- D-01: Input device is a single mono USB microphone (any class-compliant USB mic). No UMA-16, no HATs.
- D-02: Capture at 48 kHz (USB device default) and resample to 32 kHz on-Pi before feature extraction.
- D-03: Inference window length and hop are configurable. Default: 1.0 s window, 0.5 s hop (50% overlap) ⇒ ~2 inferences/sec.
- D-04: Preprocessing parity enforced by vendoring `preprocess.py` and `mel_banks_128_1024_32k.pt` into `apps/rpi-edge/`. CI drift test compares vendored copies against `src/acoustic/classification/efficientat/` and fails on divergence.

**Model Conversion & Runtime**
- D-05: Target runtime on Pi is ONNX Runtime (onnxruntime arm64). No PyTorch on the Pi.
- D-06: Conversion is host-side via `scripts/convert_efficientat_to_onnx.py`, exporting BOTH FP32 and dynamically-quantized int8 ONNX artifacts from `efficientat_mn10_v6.pt`.
- D-07: Both artifacts are committed alongside the `.pt` (or published as release assets — decide in planning). Pi app prefers int8 with FP32 fallback.
- D-08: Conversion script must include a sanity-validation step: top-1 agreement with PyTorch reference within tolerance before writing output. Fails loudly on excessive int8 drift.

**Detection Params & Config**
- D-09: YAML file + CLI overrides. Default path: `apps/rpi-edge/config.yaml` or `/etc/skyfort-edge/config.yaml` — install script chooses; decide in planning.
- D-10: No hot-reload. Config read once at startup; `systemctl restart` applies changes.
- D-11: Configurable param groups: thresholds/hysteresis, timing/smoothing, audio/GPIO hardware, model/runtime.
- D-12: Hysteresis state-machine pattern from CLS-03 is ported (not imported) into the edge app.

**GPIO LED & Audio Alarm**
- D-13: Single LED, configurable GPIO pin. off = idle, on = detect.
- D-14: LED latches on hysteresis rising edge, stays on ≥ `min_on_seconds` after last positive frame.
- D-15: SIGTERM-safe: signal handler drives all pins low and releases GPIO cleanly.
- D-16: No buzzer/relay in v1 — code structured for future pin addition.
- D-17: Optional audio alarm via bundled `alert.wav` through Pi default audio sink (ALSA).
- D-18: Audio alarm disabled by default. Plays once per latch cycle on rising edge — no looping.
- D-19: If no audio device present or playback fails, warns and continues silently. Never blocks detection.

**Detection Log & Observability**
- D-20: Always-on rotating JSONL detection log. One JSON record per latched detection, minimum: ISO timestamp, predicted class, score, latch duration, optional mel/score stats.
- D-21: Detection log CANNOT be disabled via config. General app log level can be lowered; detection log always writes.
- D-22: Log file path configurable. Default: `apps/rpi-edge/var/detections.jsonl` or `/var/lib/skyfort-edge/detections.jsonl` — decide in planning. Size-based rotation (e.g. 10 MB × N files, both configurable).
- D-23: General app logging → journald via systemd. User monitors via `ssh` + `journalctl -u skyfort-edge -f`. README must document.
- D-24: Minimal HTTP `/health` + `/status` on localhost only. `/health`: model loaded + audio stream alive. `/status`: last inference time, last detection time, current LED state, log file path. No auth, no web UI.

**App Packaging & Lifecycle**
- D-25: Edge app lives in new top-level dir `apps/rpi-edge/`. Vendors minimum code; does NOT `import` from main service package.
- D-26: Runs as systemd service under bare Python venv on Pi (no Docker). Auto-start on boot, restart on failure, journald logs.
- D-27: Install via `scripts/install_edge_rpi.sh`: creates venv, installs pinned deps, copies systemd unit, enables service. README documents SSH install + config tweaks.
- D-28: Python deps kept minimal: onnxruntime, numpy, scipy or soxr, sounddevice or soundfile, PyYAML, gpiozero or RPi.GPIO. No PyTorch, no FastAPI if stdlib `http.server` suffices.

### Claude's Discretion
- `RPi.GPIO` vs `gpiozero` — pick lightest/cleanest.
- `soxr` vs `scipy.signal.resample_poly` — pick lightest/cleanest.
- stdlib `http.server` vs FastAPI — pick lightest/cleanest.
- Internal module layout inside `apps/rpi-edge/`.
- Default log file paths and rotation sizes — propose sane defaults, surface as config keys.
- Exact systemd unit content and install script ordering.
- ONNX export details (opset, input shape fixed vs dynamic).
- int8 quantization approach (dynamic vs static calibration) — start with dynamic; escalate if accuracy guard trips.
- CI/test coverage choices for drift test (D-04) and conversion sanity check (D-08).

### Deferred Ideas (OUT OF SCOPE)
- Bearing / DOA estimation on Pi (no array).
- Multi-mic or UMA-16 on Pi.
- Model hot-swap or remote model push.
- Cloud reporting / remote detection streaming.
- OTA updates or SD-image provisioning.
- Web dashboard on Pi (`/status` is a localhost JSON endpoint only).
- Buzzer / relay GPIO output.
- Hot-reload of config file.
- Static int8 calibration.
</user_constraints>

<phase_requirements>
## Phase Requirements

Phase 21 has no formal REQ-IDs in `.planning/REQUIREMENTS.md` — the v1 traceability table only tracks the service-side CLS-* and AUD-* requirements, which are all marked Complete. This phase is a **parallel edge deployment** of the Phase 3 / Phase 14 classifier behavior, not a reinterpretation of those requirements. The 28 D-XX decisions in `21-CONTEXT.md` are the authoritative requirement set for this phase.

Mapping to existing REQ-IDs (for context only — these requirements were satisfied by the main service, but this phase replicates their behavior on the Pi edge):

| REQ-ID | Behavior | Phase 21 Manifestation | Research Support |
|--------|----------|-------------------------|-------------------|
| CLS-01 | CNN inference on audio segments | ONNX Runtime inference on 1s mel windows from single USB mic | §Standard Stack (onnxruntime), §Code Examples |
| CLS-03 | Hysteresis state machine (enter/exit + confirm hits) | Ported state machine drives GPIO LED latching | D-12, §Architecture Patterns |
| CLS-04 | Loads CNN model from configurable path at startup | YAML `model.onnx_path` + CLI override, no hot-reload | D-09, D-10 |
| AUD-01 | Real-time audio capture (callback-based, not blocking) | `sounddevice.InputStream` callback at 48 kHz mono | §Code Examples |
| AUD-02 | Continuous capture with ring buffer for downstream consumers | numpy ring buffer between audio callback and inference worker | §Architecture Patterns |
| AUD-03 | Device presence/absence detection at startup and runtime | `sd.query_devices()` on start; `/health` reports stream alive | D-24 |

**Phase-native requirements** (D-XX in CONTEXT.md): 28 decisions covering audio in/windowing, model conversion & runtime, detection params & config, GPIO/audio alarm, detection log & observability, and app packaging & lifecycle. The planner should treat each D-XX as an individual testable requirement.
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **No direct edits outside GSD workflow.** This research feeds `/gsd-plan-phase 21`.
- **Match sky-fort-dashboard styling** — N/A for this phase (no web UI; `/status` is JSON only).
- **Python ≥ 3.11** — applies on host (for conversion script); on Pi, pin to **Python 3.11** specifically to match the onnxruntime wheel tag and the main-service Python version, even though Raspberry Pi OS Bookworm also ships 3.11 by default.
- **Minimal deps on Pi** (D-28) — do not drag the main-service stack into `apps/rpi-edge/`. No FastAPI, no torch, no torchaudio, no librosa, no acoular, no pyzmq.
- **ZeroMQ is a project requirement** for the main service — **not** for this phase. The edge app is self-contained; Phase 21 does NOT publish ZMQ events. (If cross-service publishing is ever wanted later, it's a separate phase.)
- **GSD Workflow Enforcement** — all file edits must go through a GSD command. Research is the first step; planning follows.

## Standard Stack

### Core (Pi runtime)

| Library | Version | Purpose | Why Standard | Confidence |
|---------|---------|---------|--------------|------------|
| Python | 3.11.x | Pi runtime | Matches Raspberry Pi OS Bookworm default; matches onnxruntime `cp311` wheel tag. | HIGH [VERIFIED: Raspberry Pi OS Bookworm docs] |
| onnxruntime | **1.18.1** (pinned) | ONNX inference | Official `manylinux_2_27_aarch64` wheels on PyPI. 1.18.1 is conservatively known-good on Cortex-A72 (Pi 4). Newer versions (1.21+) have open reports of "illegal instruction" on Pi 4 — pin to 1.18.1 and defer upgrades to a targeted phase. [CITED: github.com/microsoft/onnxruntime/issues/24112] | MEDIUM [CITED — requires on-device verification] |
| numpy | ≥ 1.26, < 3 | Array ops, ring buffer, resample | Matches main service; pre-installed via pip wheel on aarch64. | HIGH [ASSUMED — matches main service] |
| scipy | ≥ 1.14 | `scipy.signal.resample_poly(2, 3)` for 48→32 kHz | Polyphase FIR resampling, lower latency than soxr's FFT-based approach; no new C dep. 48000/32000 = 3/2 ⇒ `resample_poly(x, 2, 3)`. | HIGH [CITED: scipy.signal.resample_poly docs] |
| sounddevice | ≥ 0.5.1 | USB mic capture (callback stream) | Already proven in main service (AUD-01). PortAudio backend supports ALSA `hw:X,0` device strings. | HIGH [VERIFIED: main service code + POC] |
| PyYAML | ≥ 6.0 | Config file parsing | Stdlib-adjacent, single C dep, no alternatives worth considering for a config file. | HIGH [ASSUMED — ecosystem default] |
| gpiozero | ≥ 2.0 | GPIO LED driver | Official Raspberry Pi recommended library; auto-selects lgpio backend on Bookworm; SIGTERM-safe cleanup via `Device.close()`. RPi.GPIO is explicitly NOT supported on Bookworm/Pi 5 by Raspberry Pi Ltd. [CITED: Raspberry Pi forums] | HIGH [CITED] |
| lgpio | ≥ 0.2 | gpiozero backend (transitive) | Implicit dep of gpiozero on Bookworm; included here only to pin it for reproducibility. | HIGH [CITED] |

### Host-side conversion (not shipped to Pi)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| torch | ≥ 2.11 (match main service) | Load `.pt`, export to ONNX | Must match the version the checkpoint was trained under — Phase 14/15 training stack. |
| torchaudio | ≥ 2.11 | Only transitively required by `preprocess.py` for `FrequencyMasking`/`TimeMasking` (no-op at eval) | Used by the `AugmentMelSTFT` module during model construction. |
| onnx | ≥ 1.17 | ONNX IR validation | `onnx.checker.check_model()` after export. |
| onnxruntime | ≥ 1.18 (matches Pi) | Run exported ONNX for sanity check | Host-side validation of exported artifact against PyTorch reference. |
| onnxruntime-tools (quantization) | bundled with onnxruntime | `quantize_dynamic()` | Dynamic int8 quantization — the D-06 path. |

**Installation (Pi):**

```bash
# Pinned for reproducibility — the install script should use exact versions
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    numpy==1.26.4 \
    scipy==1.14.1 \
    sounddevice==0.5.1 \
    PyYAML==6.0.2 \
    onnxruntime==1.18.1 \
    gpiozero==2.0.1 \
    lgpio==0.2.2.0
```

**System packages (Pi, via apt):**
```bash
sudo apt-get install -y libportaudio2 libasound2 python3.11-venv
# libportaudio2: sounddevice runtime dep
# libasound2: ALSA runtime (usually already present)
# python3.11-venv: bare venv support
```

**Version verification:** Before finalizing `requirements.txt`, the planner MUST run `npm view`-equivalent checks against PyPI for each version pin. Training-data pins are almost certainly stale. Run:
```bash
pip index versions onnxruntime  # confirm 1.18.1 still available for cp311 aarch64
pip index versions gpiozero
```

### Alternatives Considered

| Instead of | Could Use | Tradeoff | Why Rejected |
|------------|-----------|----------|--------------|
| scipy `resample_poly` | `python-soxr` | Higher quality (FFT bandlimited), but **~1000 sample latency** on HQ setting; extra C dep | 48→32 is a clean 3:2 ratio; polyphase is indistinguishable in quality for this use case and lower latency. [CITED: python-soxr docs] |
| gpiozero | `RPi.GPIO` | Older library, huge community base | Raspberry Pi Ltd explicitly state "never supported" on Pi 5 / Bookworm; pre-installed but broken. gpiozero is the official path. [CITED: Raspberry Pi forums] |
| stdlib `http.server` | FastAPI + uvicorn | Schema validation, async, auto-docs | Brings uvicorn + starlette + pydantic + typing-extensions — roughly 15 MB of deps for 2 endpoints that return static JSON. Stdlib is enough for D-24's scope. |
| PyYAML + argparse | `click` + `pydantic-settings` | Nicer DX, validation | Adds 3 deps for a 4-group config file. Argparse is stdlib; PyYAML is the single new dep. |
| onnxruntime 1.21+ | onnxruntime 1.18.1 | Latest features, perf improvements | Open "illegal instruction" reports on Pi 4 in 2026; conservative pin until someone proves a newer version works on A72. [CITED: github.com/microsoft/onnxruntime/issues/24112] |
| dynamic int8 quantization | static (calibration-based) int8 | Higher accuracy retention, requires calibration dataset | D-08 says start with dynamic, escalate only on accuracy failure — honor that. |

## Architecture Patterns

### Recommended Project Structure

```
apps/rpi-edge/
├── README.md                    # install + SSH monitoring workflow (D-23/D-27)
├── config.yaml                  # default config (D-09)
├── pyproject.toml               # pinned deps + metadata
├── requirements.txt             # mirror of pyproject for install script
├── systemd/
│   └── skyfort-edge.service     # systemd unit (D-26)
├── assets/
│   └── alert.wav                # bundled audio alarm (D-17)
├── models/                      # ONNX artifacts copied here at install time
│   ├── efficientat_mn10_v6_fp32.onnx
│   └── efficientat_mn10_v6_int8.onnx
├── vendored/                    # byte-identical copies from main service (D-04)
│   ├── preprocess.py
│   ├── mel_banks_128_1024_32k.pt
│   └── .VENDOR_SOURCE           # records source paths + hash for drift test
├── skyfort_edge/
│   ├── __init__.py
│   ├── __main__.py              # entry point: python -m skyfort_edge
│   ├── app.py                   # top-level lifecycle: start/stop/signal handling
│   ├── config.py                # YAML loader + CLI override merge
│   ├── audio.py                 # sounddevice stream + resample + ring buffer
│   ├── inference.py             # ONNX Runtime session wrapper, prefers int8 w/ FP32 fallback
│   ├── hysteresis.py            # ported state machine from CLS-03
│   ├── gpio_driver.py           # gpiozero LED with SIGTERM cleanup
│   ├── alarm.py                 # optional alert.wav playback
│   ├── detection_log.py         # RotatingFileHandler on a dedicated logger
│   ├── http_status.py           # stdlib http.server /health + /status
│   └── preprocess_adapter.py    # imports vendored/preprocess.py and adapts to ORT input
├── scripts/
│   └── install_edge_rpi.sh      # shell installer (venv + systemd enable)
└── tests/
    ├── test_hysteresis.py
    ├── test_config_loader.py
    ├── test_detection_log.py
    ├── test_gpio_driver_mock.py  # uses gpiozero MockFactory
    ├── test_http_status.py
    └── test_preprocess_drift.py  # CI: vendored == main service
```

### Pattern 1: Callback-based Audio Capture + Ring Buffer + Inference Worker Thread

**What:** `sounddevice.InputStream` callback pushes 48 kHz mono frames into a numpy ring buffer. A separate inference worker thread pops `window_samples` (1 s = 48000 samples) every `hop_seconds` (0.5 s), resamples to 32 kHz, runs preprocessing + ORT inference, feeds hysteresis, and writes detections.

**When to use:** This IS the standard pattern — it's what the main service already does for UMA-16 (AUD-01, AUD-02). Mirror it, single-channel.

**Code sketch:**
```python
# audio.py (source: adapted from src/acoustic/capture/*.py main service pattern)
import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from threading import Lock

class AudioCapture:
    def __init__(self, device: str | int, sr_in: int = 48000, sr_out: int = 32000,
                 buffer_seconds: float = 3.0):
        self.sr_in, self.sr_out = sr_in, sr_out
        self.buffer = np.zeros(int(sr_in * buffer_seconds), dtype=np.float32)
        self.write_idx = 0
        self.lock = Lock()
        self.stream = sd.InputStream(
            device=device, samplerate=sr_in, channels=1, dtype='float32',
            blocksize=int(sr_in * 0.05),  # 50 ms block
            callback=self._callback,
        )
    def _callback(self, indata, frames, time_info, status):
        # NO LOGGING, NO ALLOCATIONS — audio thread discipline
        with self.lock:
            n = frames
            end = self.write_idx + n
            if end <= len(self.buffer):
                self.buffer[self.write_idx:end] = indata[:, 0]
            else:
                split = len(self.buffer) - self.write_idx
                self.buffer[self.write_idx:] = indata[:split, 0]
                self.buffer[:n - split] = indata[split:, 0]
            self.write_idx = end % len(self.buffer)

    def read_window(self, window_samples: int) -> np.ndarray:
        # returns last window_samples samples (48 kHz)
        with self.lock:
            idx = self.write_idx
        # naive read — for ring correctness use np.concatenate
        ...

    def resample_to_32k(self, x_48k: np.ndarray) -> np.ndarray:
        return resample_poly(x_48k, up=2, down=3).astype(np.float32)
```

### Pattern 2: Hysteresis State Machine (port, not import)

**What:** Finite state machine with states `IDLE`, `RISING`, `LATCHED`, `FALLING`. Configurable `enter_threshold`, `exit_threshold`, `confirm_hits`, `release_hits`. LED is ON iff state ∈ {RISING (post-confirm), LATCHED, FALLING}. D-12 says port from main service.

**When to use:** Always. This is the D-14 rising-edge latch + `min_on_seconds` enforcement point.

```python
# hysteresis.py (port of src/acoustic/classification/state_machine.py)
from dataclasses import dataclass, field
from enum import Enum
import time

class State(Enum):
    IDLE = "idle"
    LATCHED = "latched"

@dataclass
class HysteresisConfig:
    enter_threshold: float = 0.8
    exit_threshold: float = 0.5
    confirm_hits: int = 3
    release_hits: int = 5
    min_on_seconds: float = 2.0

@dataclass
class Hysteresis:
    cfg: HysteresisConfig
    state: State = State.IDLE
    _enter_count: int = 0
    _exit_count: int = 0
    _latched_at: float = 0.0
    def update(self, score: float, now: float) -> tuple[State, bool]:
        """Returns (new_state, rising_edge_flag)."""
        rising_edge = False
        if self.state == State.IDLE:
            if score >= self.cfg.enter_threshold:
                self._enter_count += 1
                if self._enter_count >= self.cfg.confirm_hits:
                    self.state = State.LATCHED
                    self._latched_at = now
                    self._enter_count = 0
                    rising_edge = True
            else:
                self._enter_count = 0
        else:  # LATCHED
            if score < self.cfg.exit_threshold:
                self._exit_count += 1
                if (self._exit_count >= self.cfg.release_hits
                    and now - self._latched_at >= self.cfg.min_on_seconds):
                    self.state = State.IDLE
                    self._exit_count = 0
            else:
                self._exit_count = 0
        return self.state, rising_edge
```

### Pattern 3: Dedicated Detection Logger (not root logger)

**What:** A **separate** `logging.Logger` instance named `skyfort_edge.detections` with `propagate=False` and a single `RotatingFileHandler`. This logger ignores the root `logging.level` entirely — D-21 requires it cannot be disabled.

```python
# detection_log.py
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # record.msg is a dict, emitted as JSON line
        return json.dumps(record.msg, separators=(",", ":"), default=str)

def build_detection_logger(path: Path, max_bytes: int, backup_count: int) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("skyfort_edge.detections")
    logger.setLevel(logging.INFO)       # fixed — not user-configurable
    logger.propagate = False            # do NOT leak to root / journald
    # Idempotent handler install
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        h = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    return logger

# Usage on a rising edge:
# detection_logger.info({
#     "ts": datetime.now(timezone.utc).isoformat(),
#     "class": "drone",
#     "score": 0.94,
#     "latch_seconds": 0.0,  # written on release, see below
#     ...
# })
```

**Subtlety:** D-20 says "ISO timestamp, predicted class, score, latch duration". `latch_duration` is only knowable on release, so you want TWO records per detection or a single record on release. Recommendation: **single record on latch release** with full duration — simpler downstream, matches what the user called "every latched detection".

### Pattern 4: GPIO LED with SIGTERM-safe cleanup (D-15)

```python
# gpio_driver.py
import signal
from gpiozero import LED, Device
# On non-Pi dev machines: Device.pin_factory = MockFactory()

class LedDriver:
    def __init__(self, pin: int):
        self._led = LED(pin, initial_value=False)
        signal.signal(signal.SIGTERM, self._cleanup)
        signal.signal(signal.SIGINT, self._cleanup)
    def on(self): self._led.on()
    def off(self): self._led.off()
    def _cleanup(self, signum, frame):
        try:
            self._led.off()
            self._led.close()
        finally:
            # re-raise as default so systemd sees normal exit
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
```

**Note:** `gpiozero` registers its own atexit handler that calls `Device.close_all()`, which is normally enough. The explicit SIGTERM handler is belt-and-suspenders for D-15's "no stuck-on LED after restart" requirement.

### Pattern 5: stdlib http.server for /health + /status

```python
# http_status.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Thread

def make_server(state_provider, host="127.0.0.1", port=8088) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass  # silence stdout
        def do_GET(self):
            if self.path == "/health":
                body = {"model_loaded": state_provider.model_loaded(),
                        "audio_alive": state_provider.audio_alive()}
                code = 200 if all(body.values()) else 503
            elif self.path == "/status":
                body = state_provider.status_snapshot()
                code = 200
            else:
                self.send_response(404); self.end_headers(); return
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    return HTTPServer((host, port), Handler)
```

Run in a daemon thread: `Thread(target=server.serve_forever, daemon=True).start()`.

### Anti-Patterns to Avoid

- **Importing from main service package** — D-25 explicitly forbids. The vendored files live under `apps/rpi-edge/vendored/` and are `sys.path`-local.
- **Doing anything in the audio callback except `np.copyto` + buffer-index update** — matches the main service discipline. No logging, no inference trigger, no locks held across Python code that could allocate.
- **Root-logger detection writes** — would be disabled by D-21 users lowering general log level. Use a dedicated logger.
- **Passing mel tensors through ONNX as the model input** — the vendored `AugmentMelSTFT` is a `torch.nn.Module`, and exporting the `MN` model alone means the ONNX input is the mel spectrogram, not raw audio. The Pi must run `preprocess.py` on CPU (with torch) before ORT inference. **This is load-bearing.** See "Common Pitfalls".
- **Using `RPi.GPIO`** — "never supported" on Pi 5 / Bookworm per Raspberry Pi Ltd.
- **Blocking on audio alarm playback** — D-19. Use a fire-and-forget thread or `sounddevice.play()` with non-blocking mode.
- **Assuming the USB mic always gets the same ALSA card ID** — use the device *name* match (`sd.query_devices()`) not raw index.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mel spectrogram extraction | Custom FFT + mel pipeline | Vendored `AugmentMelSTFT` from `src/acoustic/classification/efficientat/preprocess.py` | Exact parity with training is the whole point. The training-side code uses a precomputed mel filterbank file (`mel_banks_128_1024_32k.pt`) specifically to eliminate torchaudio version drift. Copy both files byte-identical. |
| Hysteresis state machine | New FSM | Port from main service | D-12 mandates it. Re-inventing would break the Phase 3 tested semantics. |
| ONNX int8 quantization | Custom weight scaling | `onnxruntime.quantization.quantize_dynamic` | Handles QLinear op replacement correctly; supports MatMul/Gemm/Conv. |
| Log rotation | Custom file roll | `logging.handlers.RotatingFileHandler` | stdlib, battle-tested, atomic rename. |
| YAML → CLI merge | Custom merge logic | Two-step: parse YAML → argparse `set_defaults(**yaml_dict)` → `parser.parse_args()` | 8 lines of code, stdlib-only. |
| 48→32 kHz resample | Custom FIR | `scipy.signal.resample_poly(x, 2, 3)` | Polyphase, low latency, already a dep. |
| GPIO cleanup on signal | Manual pin low + release | `gpiozero.Device.close_all()` + atexit | gpiozero handles it; add SIGTERM redirect for durability. |
| HTTP server | Socket loop | stdlib `http.server` | 2 endpoints, localhost-only. |
| Audio playback | Custom ALSA open | `sounddevice.play()` or `simpleaudio` | sounddevice is already a dep. |

**Key insight:** **The preprocessing boundary is sacred.** Training-time mels are produced by a specific `torch.stft` + preemphasis + log + normalization pipeline with a specific precomputed filterbank tensor. Any drift — even a float rounding difference — will silently degrade accuracy. The vendoring-plus-CI-drift-test approach (D-04) is the correct answer; do NOT try to "optimize" preprocessing on the Pi by rewriting it in numpy.

## Runtime State Inventory

*(Phase 21 is a new edge application, not a rename/refactor. Runtime state inventory is minimal.)*

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | None — phase introduces a new `detections.jsonl` and creates `apps/rpi-edge/` fresh. | None. |
| Live service config | None — edge app is independent. | None. |
| OS-registered state | New: systemd unit `skyfort-edge.service` will be registered by `install_edge_rpi.sh`. Planner must design uninstall path (`systemctl disable --now && rm /etc/systemd/system/skyfort-edge.service && systemctl daemon-reload`). | Install script must provide uninstall. |
| Secrets/env vars | None. No secrets — localhost-only HTTP, no auth, no remote. | None. |
| Build artifacts | New: ONNX artifacts under `models/efficientat_mn10_v6_{fp32,int8}.onnx`. Consider whether these belong in git LFS or release assets (D-07 defers to planning — recommend **release assets** to keep git repo lean; the `.pt` source is already tracked). | Planning decision. |

**Nothing found in categories:** Stored data, live service config, secrets — verified by reading CONTEXT.md and confirming the phase creates fresh assets only.

## Common Pitfalls

### Pitfall 1: ONNX export boundary — mel preprocessing is NOT in the graph

**What goes wrong:** Naive `torch.onnx.export(classifier, raw_audio_tensor, ...)` tries to trace the whole `EfficientATClassifier`, which includes `AugmentMelSTFT` with `torch.stft`, `FrequencyMasking`, `TimeMasking`. `torch.stft` is exportable in recent opsets but the masking transforms create dynamic control flow that complicates tracing, and running mels in ORT on the Pi duplicates work that's cheaper in numpy/torch-cpu anyway.

**Why it happens:** Conceptually the user thinks "convert the classifier end-to-end." But the training code deliberately separated preprocessing (`preprocess.py`) from the model (`model.py`) to enable exactly this split.

**How to avoid:** Export **only `MN` (the model from `model.py`)** with input shape `(batch=1, 1, n_mels=128, time=100)`. On the Pi, run the vendored `AugmentMelSTFT` on CPU (it's a `nn.Module`; it only needs `torch` at runtime — or, better, **convert it to a pure numpy function once and vendor that**). Actually, since D-28 forbids PyTorch on Pi, you need a **numpy reimplementation of `AugmentMelSTFT`** that produces byte-identical output, OR you install `torch` CPU-only on Pi just for preprocessing.

**DECISION NEEDED (flag for planner):** Two options:
1. **Install torch CPU-only on Pi** — biggest single dep (~200 MB), but the vendored `preprocess.py` works unchanged → automatic drift immunity.
2. **Reimplement `AugmentMelSTFT` in pure numpy** inside `apps/rpi-edge/vendored/preprocess_numpy.py` — small dep footprint, but requires numerical parity tests and a second drift test comparing numpy output to torch output within `atol=1e-5`.

**Recommendation:** Go with **Option 2 (numpy reimpl)** because D-28 says "minimal deps" and "no PyTorch" explicitly. The numpy reimpl is maybe 60 lines (preemphasis conv, `np.fft.rfft`, `np.matmul` with loaded mel_basis, log, normalize) and the drift test verifies it against a golden set generated offline by the torch version. **But:** this breaks the simple vendoring story in D-04. The planner should raise this as an ambiguity with the user.

**Warning signs:** Top-1 agreement in the D-08 sanity check is < 99%; score distribution on a held-out drone set is shifted vs the training-time distribution.

### Pitfall 2: onnxruntime illegal instruction on Pi 4

**What goes wrong:** `import onnxruntime` on Pi 4 with a recent manylinux_2_27 wheel crashes with "illegal hardware instruction" because the wheel was compiled with `-march` flags assuming CPU features the A72 lacks.

**Why it happens:** Generic aarch64 wheels target a baseline that sometimes drifts upward across releases. Pi 4's Cortex-A72 is older (ARMv8.0-A), while newer SoCs the wheel targets may include ARMv8.2-A features.

**How to avoid:** **Pin `onnxruntime==1.18.1`** (last line known to work on A72 in community reports through early 2026). Document in the install script a fallback to 1.16.3 if 1.18.1 fails. Test on actual Pi 4 hardware during verification, not only on a dev x86/arm64 machine.

**Warning signs:** `import onnxruntime` exits with `Illegal instruction (core dumped)`. There's no Python traceback because it crashes in native code.

### Pitfall 3: USB mic card ID drift

**What goes wrong:** A USB mic shows up as `card 1` today, but after a reboot (or unrelated USB reorder) it's `card 2`. Config file says `hw:1,0` → audio stream fails to open.

**How to avoid:** Select by **device name substring**, not index:
```python
def find_input_device(name_substring: str) -> int:
    for i, dev in enumerate(sd.query_devices()):
        if name_substring.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            return i
    raise RuntimeError(f"No input device matching '{name_substring}'")
```
Config should support both `device_name` (preferred) and `device_index` (override).

**Warning signs:** `/health` reports `audio_alive: false` intermittently after reboots.

### Pitfall 4: Dynamic int8 quantization accuracy drop on Conv-heavy models

**What goes wrong:** EfficientAT MN10 is mostly `Conv2d` + `InvertedResidual` blocks. `quantize_dynamic` by default quantizes `MatMul` and `Gemm` — **it does NOT quantize `Conv` with dynamic quantization** (because dynamic Conv int8 support is limited in ORT). Result: int8 "quantization" is mostly a no-op because most of the compute is in Conv layers. You get a smaller model file but minimal speedup.

**Why it happens:** Dynamic quantization only works well for MatMul/Gemm-heavy models (transformers, LSTMs). CNNs need static (calibration-based) quantization for meaningful int8 Conv replacement.

**How to avoid:** Be honest in the sanity check: the int8 artifact may only be slightly smaller than FP32 and approximately the same speed. D-07 says "prefer int8 with FP32 fallback" — the fallback path may become the default in practice. If Pi 4 inference latency is unacceptable with FP32, escalate to **static int8 with calibration** (D-08 "fails loudly" path) in a follow-up phase.

**Recommendation for sanity check tolerance:**
- FP32 vs PyTorch: top-1 agreement ≥ 99.5%, max score delta ≤ 1e-3.
- int8 vs PyTorch: top-1 agreement ≥ 97%, max score delta ≤ 0.05.

**Warning signs:** int8 model file size is ≥ 90% of FP32 size (expected: ~40% for good quantization); inference latency is identical between FP32 and int8.

### Pitfall 5: `RotatingFileHandler` + systemd + /var/lib

**What goes wrong:** systemd service runs as non-root user; `/var/lib/skyfort-edge/` doesn't exist or isn't writable; `RotatingFileHandler` crashes at handler-install time and the service fails to start.

**How to avoid:** Install script creates the dir with correct ownership. systemd unit uses `StateDirectory=skyfort-edge` (systemd auto-creates `/var/lib/skyfort-edge` with service user ownership). Reference unit:

```ini
[Unit]
Description=Sky Fort Acoustic Edge Detector
After=sound.target network.target

[Service]
Type=simple
User=skyfort
Group=audio
SupplementaryGroups=gpio
WorkingDirectory=/opt/skyfort-edge
ExecStart=/opt/skyfort-edge/.venv/bin/python -m skyfort_edge --config /etc/skyfort-edge/config.yaml
Restart=on-failure
RestartSec=5
StateDirectory=skyfort-edge
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Warning signs:** `systemctl status skyfort-edge` shows `Failed to start` with `PermissionError` in journal.

### Pitfall 6: LED latched but log record written on wrong edge

**What goes wrong:** D-20 says "every latched detection" writes a record. If you write on the **rising** edge, you don't know `latch_duration` yet. If you write on the **falling** edge, the record appears after the detection is already over — slightly surprising for live monitoring.

**How to avoid:** Emit **one record per completed latch cycle, on release**, with full `latch_duration`. For live monitoring immediacy, rely on `/status` (`last_detection_time`, `current_led_state`) and journald (which gets the general log line on rising edge). This separates durable history (JSONL) from live state (HTTP + journald).

## Code Examples

### ONNX export (host-side)

```python
# scripts/convert_efficientat_to_onnx.py
# Source: torch.onnx docs + EfficientAT model signature
import torch
import onnx
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

from acoustic.classification.efficientat.model import get_model
from acoustic.classification.efficientat.config import EfficientATMelConfig

CKPT = Path("models/efficientat_mn10_v6.pt")
FP32_OUT = Path("models/efficientat_mn10_v6_fp32.onnx")
INT8_OUT = Path("models/efficientat_mn10_v6_int8.onnx")

def build_model() -> torch.nn.Module:
    model = get_model(
        num_classes=1,             # binary drone head (match training)
        width_mult=1.0,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=100,
    )
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    # MN.forward returns (logits, features); wrap to return logits only.
    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x): return self.inner(x)[0]
    return LogitsOnly(model)

def export_fp32(model: torch.nn.Module, out: Path):
    cfg = EfficientATMelConfig()
    dummy = torch.randn(1, 1, cfg.n_mels, cfg.input_dim_t)  # (B, C, F, T)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,                     # supported by onnxruntime 1.18
        dynamo=False,                          # legacy exporter, stable for CNNs
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(out))

def quantize_int8(fp32: Path, out: Path):
    quantize_dynamic(
        str(fp32), str(out),
        weight_type=QuantType.QInt8,
        # NOTE: quantize_dynamic for CNN will only quantize MatMul/Gemm,
        # NOT Conv. Accept this — D-08 sanity check will catch accuracy drift.
    )

def sanity_check(pt_model, onnx_path: Path, samples: list[torch.Tensor],
                 min_top1_agreement: float, max_score_delta: float):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    agree = 0
    max_delta = 0.0
    for s in samples:
        with torch.no_grad():
            pt_logits = pt_model(s).squeeze().item()
        ort_logits = sess.run(None, {"mel": s.numpy()})[0].squeeze().item()
        pt_pred = pt_logits > 0
        ort_pred = ort_logits > 0
        if pt_pred == ort_pred:
            agree += 1
        max_delta = max(max_delta, abs(pt_logits - ort_logits))
    top1 = agree / len(samples)
    if top1 < min_top1_agreement:
        raise RuntimeError(f"Top-1 agreement {top1:.3f} < {min_top1_agreement}")
    if max_delta > max_score_delta:
        raise RuntimeError(f"Max score delta {max_delta:.3e} > {max_score_delta}")
```

**Sources:** [PyTorch 2.11 torch.onnx docs](https://docs.pytorch.org/docs/stable/onnx_export.html), [onnxruntime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

### Inference session with int8 preference + FP32 fallback (Pi runtime)

```python
# inference.py
import onnxruntime as ort
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def load_session(int8_path: Path, fp32_path: Path, num_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if int8_path.exists():
        try:
            sess = ort.InferenceSession(str(int8_path), so, providers=providers)
            log.info("Loaded int8 ONNX: %s", int8_path)
            return sess
        except Exception as e:
            log.warning("int8 load failed (%s) — falling back to FP32", e)
    sess = ort.InferenceSession(str(fp32_path), so, providers=providers)
    log.info("Loaded FP32 ONNX: %s", fp32_path)
    return sess
```

### YAML config + CLI override

```python
# config.py
import argparse, yaml
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Config:
    # ... all D-11 fields ...
    score_threshold: float = 0.8
    enter_threshold: float = 0.8
    exit_threshold: float = 0.5
    confirm_hits: int = 3
    release_hits: int = 5
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    device_name: str = "USB"
    led_pin: int = 17
    min_on_seconds: float = 2.0
    onnx_path_int8: Path = Path("models/efficientat_mn10_v6_int8.onnx")
    onnx_path_fp32: Path = Path("models/efficientat_mn10_v6_fp32.onnx")
    prefer_int8: bool = True
    num_threads: int = 2
    http_port: int = 8088
    log_path: Path = Path("/var/lib/skyfort-edge/detections.jsonl")
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5
    alarm_enabled: bool = False
    alarm_wav: Path = Path("assets/alert.wav")
    log_level: str = "INFO"

def load(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    # add --score-threshold, --led-pin, etc. for every Config field
    for field in Config.__dataclass_fields__.values():
        p.add_argument(f"--{field.name.replace('_','-')}", default=None)
    args = p.parse_args(argv)
    # 1. start from dataclass defaults
    cfg_dict = asdict(Config())
    # 2. overlay YAML
    if args.config.exists():
        with args.config.open() as f:
            cfg_dict.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    # 3. overlay CLI (only fields explicitly set)
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg_dict[k] = v
    return Config(**cfg_dict)
```

### Drift test (CI)

```python
# tests/integration/test_rpi_edge_preprocess_drift.py
import hashlib
from pathlib import Path

REPO = Path(__file__).parents[2]
MAIN = REPO / "src/acoustic/classification/efficientat"
EDGE = REPO / "apps/rpi-edge/vendored"

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def test_preprocess_vendored_byte_identical():
    assert _sha(EDGE / "preprocess.py") == _sha(MAIN / "preprocess.py"), (
        "apps/rpi-edge/vendored/preprocess.py has drifted from "
        "src/acoustic/classification/efficientat/preprocess.py. "
        "Re-vendor with scripts/vendor_rpi_edge_assets.sh."
    )

def test_mel_banks_vendored_byte_identical():
    assert _sha(EDGE / "mel_banks_128_1024_32k.pt") == _sha(MAIN / "mel_banks_128_1024_32k.pt")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `RPi.GPIO` | `gpiozero` with `lgpio` backend | Bookworm release (2023) | RPi.GPIO "never supported" on Pi 5 / Bookworm per RP Ltd. Pi 4 still works but is the old path. |
| `torch.onnx.export(dynamo=False)` | `torch.onnx.export(dynamo=True)` | PyTorch 2.5+ | `dynamo=True` is the new exporter for opset ≥ 18. For CNNs on opset 17 (onnxruntime 1.18 compat), `dynamo=False` remains the stable choice. [CITED: pytorch docs] |
| Static int8 with calibration | Dynamic int8 (first try) | Practical default per D-08 | Dynamic is simpler to implement but less effective for Conv-heavy models. |
| FastAPI for everything | stdlib `http.server` for 2-endpoint localhost JSON | This phase | Dep weight tradeoff. |

**Deprecated/outdated in this phase's context:**
- `RPi.GPIO`: Deprecated for new code by Raspberry Pi Ltd.
- `onnxruntime ≤ 1.16` on Pi 4: Older wheels work but lack newer ops; don't downgrade unless 1.18.1 breaks.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `onnxruntime 1.18.1` runs cleanly on Pi 4 Cortex-A72. | Standard Stack | MEDIUM — if it crashes with "illegal instruction", plan must provide a 1.16.x or community wheel fallback. **Action: verify on real Pi 4 in Wave 0.** |
| A2 | `quantize_dynamic` on MN10 produces a valid ONNX artifact, even if it mostly no-ops on Conv layers. | Pitfall 4 | LOW — dynamic quant is well-tested; only the benefit is in doubt, not correctness. |
| A3 | The trained `efficientat_mn10_v6.pt` checkpoint uses `num_classes=1` (binary head) with `head_type="mlp"` matching the `MN` constructor signature. | Code Examples / Export | HIGH — if the checkpoint was trained with a different head (e.g. multi-class 4-way), the export `LogitsOnly` wrapper and sanity check need different shapes. **Action: planner confirms with `torch.load(ckpt); print(state_dict.keys())` in Wave 0.** |
| A4 | Pi 4 inference latency for MN10 FP32 at 1 s input is < 500 ms (i.e., can keep up with 2 Hz hop). | Validation Architecture | HIGH — if inference is slower than 500 ms, the hop budget is violated and windows queue up. **Action: benchmark on real Pi 4; if too slow, consider Option 2 for preprocessing separately, or static int8.** |
| A5 | `gpiozero` LED cleanup via atexit + explicit SIGTERM handler is sufficient to guarantee LED is off after crash, including SIGKILL. | Pattern 4 / D-15 | MEDIUM — SIGKILL bypasses Python signal handlers. A true "LED off after any crash" guarantee would need a hardware pulldown resistor. Document this as an accepted limitation. |
| A6 | The `MN` model from `model.py` can be exported via tracing without graph-break issues from the `return_fmaps` branch and the `.squeeze()` calls. | Code Examples | MEDIUM — some squeeze patterns break ONNX export; may need to wrap the model to force the `return_fmaps=False` path and fix output shapes. **Action: do a dry-run export in Wave 0 and catch any failures early.** |
| A7 | `preprocess.py` can be replaced with a pure-numpy implementation producing byte-identical output, OR torch CPU is installed on Pi. | Pitfall 1 / DECISION NEEDED | HIGH — this is the unresolved fork in the plan. **Action: planner raises with user during plan-check, or chooses a side with explicit rationale.** |
| A8 | `onnxruntime.quantization` module is bundled with the standard `onnxruntime` package (not a separate `onnxruntime-tools`). | Standard Stack | HIGH — as of onnxruntime ≥ 1.6, it is bundled. [CITED: onnxruntime docs] |

## Open Questions

1. **Preprocessing on Pi: torch CPU install vs. pure-numpy reimpl.**
   - What we know: D-28 says "no PyTorch on Pi"; D-04 says vendor `preprocess.py` byte-identical.
   - What's unclear: These two are in tension — the vendored file imports `torch` and `torchaudio`.
   - Recommendation: **Pure numpy reimpl** inside `apps/rpi-edge/vendored/preprocess_numpy.py`, with a parity test in CI that runs the torch version and the numpy version against a fixed input and asserts `np.allclose(rtol=1e-5, atol=1e-6)`. This keeps D-28 intact at the cost of a second vendoring relationship. Raise with user.

2. **Install path: `/opt/skyfort-edge/` vs `/home/<user>/skyfort-edge/`.**
   - What we know: D-27 says install script creates venv, copies systemd unit. D-09 mentions `/etc/skyfort-edge/config.yaml` as an option.
   - What's unclear: FHS discipline vs. user-space simplicity.
   - Recommendation: `/opt/skyfort-edge/` for app, `/etc/skyfort-edge/config.yaml` for config, `/var/lib/skyfort-edge/detections.jsonl` for log. Standard FHS, matches systemd `StateDirectory=`.

3. **ONNX artifact storage: git vs. release assets.**
   - What we know: D-07 says "committed alongside `.pt` (or published as release assets — decide in planning)".
   - What's unclear: Git LFS policy for the project.
   - Recommendation: **Commit the `.pt` (already done), commit both `.onnx` files as artifacts only if they're < 20 MB each** (MN10 is ~18 MB FP32 per Phase 14 context → ~18 MB int8 dynamic → acceptable). Otherwise use GitHub release assets + install-script download.

4. **Checkpoint metadata: does `efficientat_mn10_v6.pt` have a `num_classes` / `head_type` record inside it?**
   - What we know: Phase 14 vendored EfficientAT and trained MN10; v6 presumably is a later iteration from Phase 15/20.
   - What's unclear: The exact head shape.
   - Recommendation: Planner opens the checkpoint in Wave 0 and records the shape in the PLAN. Ties to A3.

5. **http.server port collision.**
   - What we know: No specific port in D-24.
   - Recommendation: Default `127.0.0.1:8088`, configurable.

## Environment Availability

| Dependency | Required By | Available (host) | Version | Fallback |
|------------|------------|------------------|---------|----------|
| python3.11 | host conversion script | ✓ (assumed, matches main service) | 3.11.x | — |
| torch | host conversion script | ✓ (main service dep) | ≥ 2.11 | — |
| onnx | host: `check_model` | to install | ≥ 1.17 | skip check |
| onnxruntime | host: sanity check + quantize | to install | ≥ 1.18 | — |
| Raspberry Pi 4 | phase target | ✗ (no Pi in CI) | — | **Manual verification step** — planner must design a real-hardware verification checkpoint. Host-only tests can cover everything except onnxruntime-on-A72 behavior and actual GPIO/audio. |
| USB mic | Pi runtime | ✗ (not available in CI) | — | gpiozero `MockFactory` + pre-recorded WAV injection |

**Missing with no fallback:**
- Real Pi 4 hardware for final verification. **The plan must include a manual "on-device smoke test" step** — at minimum: install, start service, play known drone WAV through room speaker, verify LED and JSONL entry.

**Missing with fallback:**
- GPIO on dev machine: use `gpiozero.Device.pin_factory = MockFactory()` in all GPIO tests.
- USB mic on dev machine: use `sounddevice` file-backed playback or a file-input adapter for the audio capture module.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥ 8.0 (matches main repo) |
| Config file | `pyproject.toml` (inherits from repo root) |
| Quick run command | `pytest apps/rpi-edge/tests/ -x --timeout=30` |
| Full suite command | `pytest apps/rpi-edge/tests/ tests/integration/test_rpi_edge_preprocess_drift.py -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-02 | Resample 48→32 kHz preserves spectral content | unit | `pytest apps/rpi-edge/tests/test_audio.py::test_resample_48_to_32 -x` | ❌ Wave 0 |
| D-03 | Hop scheduling produces 2 Hz inferences | unit | `pytest apps/rpi-edge/tests/test_scheduling.py::test_window_hop -x` | ❌ Wave 0 |
| D-04 | Vendored preprocess byte-identical to main | integration | `pytest tests/integration/test_rpi_edge_preprocess_drift.py -x` | ❌ Wave 0 |
| D-06 | Conversion script produces both FP32 and int8 ONNX | integration | `pytest tests/integration/test_convert_efficientat_to_onnx.py -x` | ❌ Wave 0 |
| D-08 | Sanity check raises on excessive drift | unit | `pytest tests/integration/test_convert_efficientat_to_onnx.py::test_sanity_check_fails_on_bad_model -x` | ❌ Wave 0 |
| D-12 | Hysteresis enter/exit + confirm/release | unit | `pytest apps/rpi-edge/tests/test_hysteresis.py -x` | ❌ Wave 0 |
| D-14 | LED latches ≥ min_on_seconds | unit | `pytest apps/rpi-edge/tests/test_gpio_driver_mock.py::test_min_on_seconds -x` | ❌ Wave 0 |
| D-15 | SIGTERM drives LED low | unit | `pytest apps/rpi-edge/tests/test_gpio_driver_mock.py::test_sigterm_cleanup -x` | ❌ Wave 0 |
| D-17/18/19 | Audio alarm plays once per latch, degrades silently | unit | `pytest apps/rpi-edge/tests/test_alarm.py -x` | ❌ Wave 0 |
| D-20/21 | Detection log always writes, regardless of app log level | unit | `pytest apps/rpi-edge/tests/test_detection_log.py::test_log_writes_despite_root_silenced -x` | ❌ Wave 0 |
| D-22 | Log rotation at configured size | unit | `pytest apps/rpi-edge/tests/test_detection_log.py::test_rotation -x` | ❌ Wave 0 |
| D-24 | /health and /status return correct JSON | unit | `pytest apps/rpi-edge/tests/test_http_status.py -x` | ❌ Wave 0 |
| D-09/D-11 | YAML loads + CLI overrides | unit | `pytest apps/rpi-edge/tests/test_config_loader.py -x` | ❌ Wave 0 |
| D-26 | systemd unit file parses (syntax) | unit | `pytest apps/rpi-edge/tests/test_systemd_unit.py -x` (uses `systemd-analyze verify` or a regex-level check) | ❌ Wave 0 |
| N/A (integration) | End-to-end: WAV file → hysteresis → mock LED + JSONL record | integration | `pytest apps/rpi-edge/tests/test_e2e_mock.py -x` | ❌ Wave 0 |
| A4 | Host latency proxy: inference on MN10 ONNX < 200 ms per window on x86 dev machine | integration (host) | `pytest apps/rpi-edge/tests/test_latency_host.py -x` | ❌ Wave 0 |
| Manual | On-device smoke test (real Pi 4, real USB mic, real LED) | manual-only | N/A — checklist in README | — |

### Independent Validation Signals (Nyquist)

Beyond unit tests, four independent signals prove correctness:

1. **Conversion sanity check (built into `convert_efficientat_to_onnx.py`).** FP32 top-1 agreement ≥ 99.5%, max score delta ≤ 1e-3 vs PyTorch reference. int8 top-1 agreement ≥ 97%, max score delta ≤ 0.05. Script exits non-zero on failure — CI blocks the artifact write. Pass threshold: both must pass for the artifact to be committed.

2. **Preprocess drift test (CI, byte-level).** SHA-256 compare of vendored vs main-service `preprocess.py` and `mel_banks_128_1024_32k.pt`. **If pure-numpy reimpl path is chosen (A7), add a second test:** feed 10 golden audio fixtures through both the torch version and the numpy reimpl, assert `np.allclose(atol=1e-6, rtol=1e-5)`.

3. **End-to-end mock-LED test (host-side).** Load a known-drone WAV + a known-silence WAV, run through the full app with `gpiozero.MockFactory`, assert:
   - Drone WAV → at least one rising-edge LED.on() call.
   - Silence WAV → no rising-edge.
   - Each rising edge produces exactly one JSONL record after min_on_seconds.
   - LED.off() called after min_on_seconds + release hits worth of silence.

4. **Host-side latency budget proxy (pre-Pi).** On x86 dev machine, assert MN10 ONNX FP32 inference on 1 sample is < 50 ms. This is a pessimistic proxy; Pi 4 will be ~3-5x slower. **Pass threshold:** < 50 ms on dev, < 500 ms on Pi 4 (verified manually). Failure → escalate to static int8 quantization or reduce `input_dim_t`.

### Sampling Rate

- **Per task commit:** `pytest apps/rpi-edge/tests/ -x --timeout=30` (~5 s)
- **Per wave merge:** `pytest apps/rpi-edge/tests/ tests/integration/test_rpi_edge_preprocess_drift.py tests/integration/test_convert_efficientat_to_onnx.py -v`
- **Phase gate:** Full suite + manual on-device smoke test checklist signed off

### Wave 0 Gaps

- [ ] `apps/rpi-edge/pyproject.toml` — pinned deps + pytest config
- [ ] `apps/rpi-edge/tests/conftest.py` — shared fixtures: mock GPIO factory, tmp config dir, golden WAV loader
- [ ] `apps/rpi-edge/tests/fixtures/` — small golden audio clips (drone + silence, ~1 s each) for end-to-end tests
- [ ] `apps/rpi-edge/vendored/.VENDOR_SOURCE` — records source paths + SHA for drift test
- [ ] `tests/integration/test_rpi_edge_preprocess_drift.py` — CI drift test (sits in main repo tests/)
- [ ] `tests/integration/test_convert_efficientat_to_onnx.py` — host-side conversion + sanity check test
- [ ] **Wave 0 checkpoint: `torch.load(models/efficientat_mn10_v6.pt)` and record `num_classes` + head shape** (resolves A3, A6)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | localhost-only binding, no network exposure (D-24) |
| V3 Session Management | no | no sessions |
| V4 Access Control | partial | systemd `User=skyfort` (non-root); file perms on `/var/lib/skyfort-edge/` |
| V5 Input Validation | yes | YAML config schema validation (dataclass types); HTTP path whitelist (`/health`, `/status` only) |
| V6 Cryptography | no | no secrets, no TLS needed (localhost HTTP) |
| V7 Error Handling | yes | never leak file paths or stack traces in HTTP responses |
| V10 Malicious Code | yes | `torch.load(weights_only=True)` in conversion script (prevent pickle RCE via tampered checkpoint) |
| V12 File Operations | yes | `RotatingFileHandler` path must be pre-created with correct perms; no user input into file paths |

### Known Threat Patterns for this stack

| # | Threat | STRIDE | Mitigation |
|---|--------|--------|------------|
| T1 | HTTP `/status` leaks file paths / internal state to a local attacker who gained shell on the Pi | Information Disclosure | `/status` responses contain only non-sensitive fields: timestamps, LED state, `log_path` as a string. No user data, no audio samples, no model weights. Localhost bind means no network attacker. |
| T2 | HTTP endpoint accessible from LAN because of bind-address misconfiguration | Elevation of Privilege / I.D. | Hardcode bind to `127.0.0.1` unless config override. Add a test: `test_http_binds_localhost_only`. Document in README that exposing the endpoint to LAN requires adding auth (explicitly out of scope). |
| T3 | Malicious model file (tampered `.pt` or `.onnx`) executes code via pickle | RCE (Tampering) | Host-side conversion uses `torch.load(..., weights_only=True)`. Pi uses ONNX Runtime, which does NOT use pickle — ONNX is a protobuf format, safe against pickle attacks. Config must reject arbitrary file paths from YAML if the service runs as `skyfort` user who only has read access to `/opt/skyfort-edge/models/`. |
| T4 | Detection log grows unbounded, exhausts disk → service down | DoS | `RotatingFileHandler` with configured `max_bytes × backup_count`. Default 10 MB × 5 = 50 MB cap. Document in README. |
| T5 | Systemd restart loop after repeated crashes masks underlying issue / wears SD card | DoS / Operational | `RestartSec=5` + `StartLimitBurst=5` + `StartLimitIntervalSec=60` in systemd unit. After 5 restarts in 60 s, stop and surface in journald. |
| T6 | Audio capture callback allocates / logs → real-time priority inversion drops frames | DoS (availability) | Audio callback discipline: no logging, no allocations, only `np.copyto` + index update. Enforced by code review + pattern from main service. |
| T7 | GPIO left asserted after crash → hardware left in unexpected state (LED on forever, or worse with future relay expansion) | Tampering / Safety | gpiozero atexit + explicit SIGTERM handler. Document accepted SIGKILL limitation in README. |

**Top 5 threats for planner to include in `<threat_model>`:** T1, T2, T3, T5, T7.

## Sources

### Primary (HIGH confidence)
- `src/acoustic/classification/efficientat/preprocess.py` — training-time mel preprocessing (read in full)
- `src/acoustic/classification/efficientat/model.py` — MN model architecture + `get_model()` signature (read in full)
- `src/acoustic/classification/efficientat/config.py` — `EfficientATMelConfig` constants (read in full)
- `src/acoustic/classification/efficientat/classifier.py` — reference inference wrapper (read in full)
- `.planning/phases/21-.../21-CONTEXT.md` — 28 D-XX decisions (authoritative requirements)
- `.planning/REQUIREMENTS.md` — CLS-01..04, AUD-01..03 (context only)
- `.planning/ROADMAP.md` — Phase 21 slot + dependencies
- `./CLAUDE.md` — project stack + GSD workflow enforcement
- [ONNX Runtime Quantization Docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — `quantize_dynamic` API, QInt8 weight_type, MatMul/Gemm op support
- [PyTorch torch.onnx docs](https://docs.pytorch.org/docs/stable/onnx_export.html) — opset_version, dynamic_axes, dynamo=True vs False

### Secondary (MEDIUM confidence)
- [onnxruntime PyPI](https://pypi.org/project/onnxruntime/) — aarch64 wheel availability (cp311, manylinux_2_27)
- [gpiozero/gpiozero issue #1166 "lgpio pin factory broken on RPi5"](https://github.com/gpiozero/gpiozero/issues/1166) — lgpio is default backend on Bookworm
- [Raspberry Pi Forums: "Does RPi.GPIO work in Bookworm??"](https://forums.raspberrypi.com/viewtopic.php?t=372507) — "never supported" statement from RP Ltd
- [python-soxr docs](https://python-soxr.readthedocs.io/en/latest/soxr.html) — HQ resampler latency ~1000 output samples
- [scipy.signal.resample_poly docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html) — polyphase up/down args

### Tertiary (LOW confidence — flag for on-device validation)
- [onnxruntime issue #24112: "fails to load on Raspberry Pi 4"](https://github.com/microsoft/onnxruntime/issues/24112) — illegal instruction reports on Pi 4, multiple versions; motivates the 1.18.1 pin
- Raspberry Pi forums: USB microphone `hw:X,0` / card-id drift reports (multiple threads)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH for Python/scipy/sounddevice/PyYAML (proven in main service); MEDIUM for onnxruntime 1.18.1 on Pi 4 (needs on-device verification); HIGH for gpiozero (official Raspberry Pi recommendation)
- Architecture: HIGH — patterns are straightforward ports of main-service patterns with stdlib substitutions
- Pitfalls: HIGH on preprocessing boundary (this is the largest silent-failure risk); MEDIUM on onnxruntime wheel stability (requires empirical validation); HIGH on dynamic-int8-Conv gotcha (well-documented by onnxruntime team)
- Validation: HIGH on the test plan structure; MEDIUM on the exact latency thresholds (they are informed guesses — must be recalibrated after first on-device run)

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (30 days — most facts are about stable libraries; the onnxruntime-on-Pi-4 situation may change faster and should be re-checked if the plan slips)

## RESEARCH COMPLETE

Research complete. Key unresolved items for the planner to raise during plan-check:

1. **A7 / Pitfall 1:** Preprocessing on Pi — torch CPU install vs pure-numpy reimpl. The CONTEXT.md D-04 ("vendor `preprocess.py` byte-identical") and D-28 ("no PyTorch on Pi") are in direct tension. Planner must pick a side or escalate to user. Recommendation: pure-numpy reimpl with a second parity test.
2. **A3 / A6:** Wave 0 must open `models/efficientat_mn10_v6.pt` and record the exact head shape (`num_classes`, `head_type`) before the conversion script is written. The example code assumes `num_classes=1, head_type="mlp"` which may not be correct for v6.
3. **A1:** onnxruntime 1.18.1 must be verified on real Pi 4 hardware in Wave 0 (or at latest, before the conversion script is considered correct). Fallback wheel path must be documented in the install script.
4. **Artifact storage (Open Question 3):** Git commit vs release assets — recommend committing (both files together ≤ ~40 MB) unless the project has a size policy.
5. **On-device smoke test:** The plan MUST include a manual real-hardware verification step; host-only tests cannot cover the full stack.

The planner can now create PLAN.md files.
