# Phase 21: Build Raspberry Pi 4 edge drone-detection app - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 21-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 21-build-raspberry-pi-4-edge-drone-detection-app-using-efficien
**Areas discussed:** Audio input & windowing, Model conversion & runtime, Detection params & config, GPIO LED alarm behavior, App packaging & lifecycle, Detection log

---

## Audio input & windowing

| Option | Description | Selected |
|--------|-------------|----------|
| Single mono USB mic | Class-compliant USB mic, matches mono training modality, minimal power/CPU | ✓ |
| UMA-16v2 (same as main service) | Reuse 16-ch array, downmix/select ch 0, overkill for alarm | |
| Respeaker / Pi HAT | I2S/HAT mic, fixed pinout, adds board dependency | |

| Option | Description | Selected |
|--------|-------------|----------|
| 32 kHz native, 1.0 s / 0.5 s | Matches training mel banks exactly | |
| 48 kHz capture → resample to 32 kHz | Matches USB mic defaults, on-Pi resample | ✓ |
| 32 kHz capture, 1.0 s / 1.0 s (no overlap) | Cheapest CPU, slower reaction | |

| Option | Description | Selected |
|--------|-------------|----------|
| 1.0 s window, 0.5 s hop | 50% overlap, ~2 inferences/sec | (default) |
| 1.0 s window, 1.0 s hop | No overlap, 1 inference/sec | |
| 1.0 s window, 0.25 s hop | 75% overlap, 4 inferences/sec, risk of CPU strain | |

**User's choice:** Single mono USB mic; 48 kHz → resample to 32 kHz; window/hop configurable with 50% default.
**Notes:** "it needs to be configurable with default to 50%".

---

## Model conversion & runtime

| Option | Description | Selected |
|--------|-------------|----------|
| ONNX Runtime | Mature ARM support, XNNPACK EP, no PyTorch on Pi | ✓ |
| TorchScript JIT | Simpler export, requires PyTorch on Pi (~200 MB) | |
| PyTorch eager | No conversion, likely too slow for 2 Hz real-time | |

| Option | Description | Selected |
|--------|-------------|----------|
| Host-side FP32 only | Commit .onnx alongside .pt, quantization later | |
| Host-side FP32 + dynamic int8 | Export both, prefer int8 on Pi with FP32 fallback | ✓ |
| On-Pi conversion at first boot | Needs PyTorch on Pi, slower cold start | |

**User's choice:** ONNX Runtime on Pi; host-side conversion to FP32 + dynamic int8.

---

## Detection params & config

| Option | Description | Selected |
|--------|-------------|----------|
| YAML file + CLI overrides | Standard for edge services, editable without rebuild | ✓ |
| TOML file + CLI overrides | Nicer typing, less common in repo | |
| Env vars only (12-factor) | Simple for systemd/Docker, ugly for many params | |

| Option | Description | Selected |
|--------|-------------|----------|
| Thresholds & hysteresis | score_threshold, enter/exit, confirm_hits, release_hits, per-class thresholds, class allowlist | ✓ |
| Timing & smoothing | window/hop/smoothing/cooldown | ✓ |
| Audio & GPIO hardware | input device, SR, LED/buzzer pins, latch duration | ✓ |
| Model & runtime | onnx path, int8/fp32 preference, num_threads, execution provider, log level | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| No hot-reload, restart to apply | Simpler, no file watcher | ✓ |
| Yes, watch file and reload on change | More complex, more edge cases | |

**User's choice:** YAML + CLI; all four parameter groups configurable; no hot-reload.

---

## GPIO LED alarm behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Single LED off=idle, on=detect | Simplest wiring | ✓ |
| Single LED off=idle, blink=detect | More eye-catching, needs timer thread | |
| Two LEDs (armed/detect) | Nicer operator feedback | |
| Three LEDs (power/armed/detect) | Most informative, most pins | |

| Option | Description | Selected |
|--------|-------------|----------|
| Hysteresis + configurable min-on time | Prevents flicker, configurable dwell | ✓ |
| Instant follow (LED == hysteresis state) | Can flicker | |
| Fixed latch duration per detection | Ignores confirmation logic | |

| Option | Description | Selected |
|--------|-------------|----------|
| LED only for v1 | Minimal scope, structure leaves room for buzzer | ✓ |
| LED + optional buzzer pin (disabled by default) | Small extra code | |

| Option | Description | Selected |
|--------|-------------|----------|
| SIGTERM handler drives pins low + releases GPIO | Prevents stuck-on LED | ✓ |
| Rely on Pi reboot to clear state | Simpler code | |

**User's choice:** Single LED (off/on); hysteresis + min-on; LED only for GPIO v1; SIGTERM-safe cleanup.
**Notes:** User added: "we will also add audio if speaker is connected" — captured as separate optional audio channel.

### Optional audio alarm (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Bundled WAV via default audio sink, config-enabled | Simple, config-gated, skip silently if no device | ✓ |
| System beep / PWM tone | No WAV file, less flexible | |

| Option | Description | Selected |
|--------|-------------|----------|
| Once per latch cycle, on rising edge | Clean, least annoying | ✓ |
| Loop every N seconds while latched | More aggressive alerting | |

---

## App packaging & lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| Subpackage src/acoustic/edge/ | Reuses main package, couples deployment | |
| Separate top-level dir apps/rpi-edge/ | Clean Pi install footprint, some duplication | ✓ |
| Separate repo | Pure separation, out of scope this phase | |

| Option | Description | Selected |
|--------|-------------|----------|
| systemd service + bare Python venv | Standard Pi pattern, no Docker overhead | ✓ |
| Docker on Pi | Consistent with main service, heavier | |
| Plain Python script, manual start | Fine for dev, not deployed | |

| Option | Description | Selected |
|--------|-------------|----------|
| Install script + README | scripts/install_edge_rpi.sh automates venv + systemd | ✓ |
| README only, manual steps | Lightest scope | |
| Ansible / cloud-init image | Overkill | |

| Option | Description | Selected |
|--------|-------------|----------|
| Copy preprocess.py + mel_banks into apps/rpi-edge/ | Fully decoupled, drift test | ✓ |
| Import from src/acoustic/classification/efficientat | Zero duplication but couples deployment | |
| Bake preprocessing into exported ONNX | Cleanest but torchaudio→ONNX compat work | |

| Option | Description | Selected |
|--------|-------------|----------|
| Journald logs + LED only | Simplest | |
| Minimal HTTP /health + /status endpoint | Remote monitoring, adds web dep | |

**User's choice:** apps/rpi-edge/ top-level dir; systemd + venv; install script + README; vendor preprocessing; **both** — HTTP /health + /status AND journald logs tailable via SSH.
**Notes:** User wrote "booth + option to monitor log with ssh on the rp" — interpreted as both surfaces enabled.

---

## Detection log (added after review)

| Option | Description | Selected |
|--------|-------------|----------|
| Rotating JSONL file | Machine-readable, append-only, configurable path | ✓ |
| Rotating CSV file | Human-readable in Excel, rigid schema | |
| SQLite database | Queryable, overkill for append-only alarm log | |

**User's choice:** Rotating JSONL, always on, cannot be disabled by config even when general log is silenced.
**Notes:** User said "need to keep log of all the detections (the full log can be turn off in settings but not this one)".

---
