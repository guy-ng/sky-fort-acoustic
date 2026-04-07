# Phase 21: Build Raspberry Pi 4 edge drone-detection app - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a standalone Raspberry Pi 4 edge application that:

1. Loads a converted `efficientat_mn10_v6.pt` model and runs continuous audio classification from a single mono USB microphone.
2. Applies a hysteresis-based drone detection policy with fully configurable params (thresholds, timing, hardware, runtime).
3. Drives a single GPIO LED alarm (off=idle, on=detect) with configurable min-on latch and SIGTERM-safe cleanup.
4. Optionally plays a bundled alert WAV through the default audio sink (disabled by default).
5. Persists every detection to an always-on rotating JSONL log that cannot be disabled by config.
6. Ships with a host-side model conversion script producing ONNX FP32 + dynamic-int8 artifacts committed alongside the `.pt`.
7. Installs via a shell script + systemd unit, runs in a bare Python venv on the Pi, and exposes a minimal HTTP `/health` + `/status` endpoint on localhost alongside journald logging.

**Not in scope (belongs to other phases):** DOA/bearing on Pi, multi-mic/array input, range estimation, model hot-swap, remote config push, OTA updates, cloud reporting, Pi provisioning via image/Ansible.

</domain>

<decisions>
## Implementation Decisions

### Audio Input & Windowing
- **D-01:** Input device is a **single mono USB microphone** (any class-compliant USB mic). No UMA-16, no HATs.
- **D-02:** Capture at **48 kHz** (USB device default) and **resample to 32 kHz** on-Pi before feature extraction. This matches the training mel filterbank (`mel_banks_128_1024_32k.pt`) without forcing device-specific capture rates.
- **D-03:** Inference window length and hop are **configurable**. Default: **1.0 s window, 0.5 s hop (50% overlap)** ⇒ ~2 inferences/sec.
- **D-04:** Preprocessing parity is enforced by **vendoring** `preprocess.py` and `mel_banks_128_1024_32k.pt` into `apps/rpi-edge/` (see D-25). A drift test in the main repo compares vendored copies against `src/acoustic/classification/efficientat/` and fails CI if they diverge.

### Model Conversion & Runtime
- **D-05:** Target runtime on Pi is **ONNX Runtime (onnxruntime arm64)**. No PyTorch on the Pi.
- **D-06:** Conversion is **host-side** via a script (e.g. `scripts/convert_efficientat_to_onnx.py`) that exports **both FP32 and dynamically-quantized int8 ONNX** artifacts from `efficientat_mn10_v6.pt`.
- **D-07:** Both artifacts are **committed** alongside the `.pt` (or published as release assets — decide in planning). Pi app **prefers int8** with **FP32 fallback** if int8 load fails or accuracy guard trips.
- **D-08:** Conversion script must include a **sanity validation step**: run the exported ONNX against a small held-out sample and assert top-1 agreement with the PyTorch reference within tolerance before writing the output file. Fails loudly if int8 drift is excessive.

### Detection Params & Config
- **D-09:** Config surface is a **YAML file + CLI overrides**. File default path: `apps/rpi-edge/config.yaml` (repo-local) or `/etc/skyfort-edge/config.yaml` (installed); the install script picks one — decide in planning.
- **D-10:** **No hot-reload.** Config is read once at startup; `systemctl restart skyfort-edge` applies changes.
- **D-11:** Configurable parameter groups:
  - **Thresholds & hysteresis**: `score_threshold`, enter/exit thresholds, `confirm_hits`, `release_hits`, per-class thresholds, class allowlist.
  - **Timing & smoothing**: `window_seconds`, `hop_seconds`, smoothing window / EMA length, post-release cooldown.
  - **Audio & GPIO hardware**: input device name or index, capture sample rate, LED pin, (optional) alarm audio device, latch `min_on_seconds`.
  - **Model & runtime**: ONNX path, int8/fp32 preference, `num_threads`, ORT execution provider, general log level.
- **D-12:** Detection hysteresis reuses the state-machine pattern from CLS-03 (enter/exit thresholds + confirm/release hit counts). Pi-app should port the logic, not import the service pipeline.

### GPIO LED & Audio Alarm
- **D-13:** **Single LED** on a configurable GPIO pin. States: **off = idle, on = detect**.
- **D-14:** LED latches on the **hysteresis rising edge** and stays on for at least `min_on_seconds` (configurable) after the last positive frame before releasing. Prevents flicker.
- **D-15:** LED driver is SIGTERM-safe: a signal handler **drives all pins low and releases the GPIO** cleanly on shutdown, crash paths, and Ctrl-C. No stuck-on LED after a restart.
- **D-16:** **No buzzer / relay pin** in v1. Code is structured so adding an output pin later is a config addition, not a refactor.
- **D-17:** **Optional audio alarm** is supported as a *separate* channel from GPIO: a **bundled `alert.wav`** played through the Pi's default audio sink (ALSA) via `sounddevice` or equivalent.
- **D-18:** Audio alarm is **disabled by default** in config. When enabled, it plays **once per latch cycle, on the rising edge** — no looping, no replay until LED releases and re-latches.
- **D-19:** If no audio device is present or playback fails, the audio alarm **logs a warning and continues silently**. It never blocks detection or crashes the service.

### Detection Log & Observability
- **D-20:** **Always-on rotating JSONL detection log.** Every latched detection writes one JSON record per line with at minimum: ISO timestamp, predicted class, score, latch duration, optional mel/score stats.
- **D-21:** The detection log **cannot be disabled via config**. The general/verbose application log level *can* be lowered or silenced, but the detection log always writes.
- **D-22:** Log file path is configurable (default: `apps/rpi-edge/var/detections.jsonl` or `/var/lib/skyfort-edge/detections.jsonl` — decide in planning). Rotation is size-based (e.g. 10 MB per file, N files retained, both configurable).
- **D-23:** General application logging goes to **journald** via systemd; the user monitors the Pi via `ssh` + `journalctl -u skyfort-edge -f`. The README must document this workflow.
- **D-24:** A **minimal HTTP `/health` + `/status` endpoint** is bound to `localhost` only. `/health` reports model loaded + audio stream alive; `/status` reports last inference time, last detection time, current LED state, and log file path. No auth (localhost-only), no web UI.

### App Packaging & Lifecycle
- **D-25:** The edge app lives in a **new top-level directory: `apps/rpi-edge/`** — separate from `src/acoustic/`. It vendors the minimum code it needs (D-04) and does *not* `import` from the main service package.
- **D-26:** On the Pi the app runs as a **systemd service** under a **bare Python venv** (no Docker on the Pi). The systemd unit auto-starts on boot, restarts on failure, and logs to journald.
- **D-27:** **Install automation** ships as `scripts/install_edge_rpi.sh`: creates the venv, installs pinned deps (including `onnxruntime` arm64 wheel and `RPi.GPIO` or `gpiozero`), copies the systemd unit, and enables the service. A README documents SSH install, monitoring, and config tweaks.
- **D-28:** Python dependencies on Pi are kept **minimal**: `onnxruntime`, `numpy`, `scipy` or `soxr` (resample), `sounddevice` or `soundfile`, `PyYAML`, `gpiozero` (or `RPi.GPIO`). No PyTorch, no FastAPI if a stdlib `http.server` suffices (decide in planning based on `/status` richness).

### Claude's Discretion
- Exact dependency choices between equivalent alternatives (`RPi.GPIO` vs `gpiozero`, `soxr` vs `scipy.signal.resample_poly`, stdlib `http.server` vs FastAPI) — pick what keeps the Pi install lightest and cleanest.
- Internal module layout inside `apps/rpi-edge/` (how to split detector, GPIO, audio alarm, logger, config loader).
- Default log file paths and rotation sizes — propose sane defaults, surface as config keys.
- Exact systemd unit content and install script ordering.
- ONNX export details (opset, input shape fixed vs dynamic) — planner/researcher decides based on EfficientAT export docs.
- int8 quantization approach (dynamic vs static calibration) — start with dynamic; escalate to static only if accuracy guard trips.
- CI/test coverage choices for the drift test (D-04) and conversion sanity check (D-08).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training-side Reference (source of truth for preprocessing)
- `src/acoustic/classification/efficientat/preprocess.py` — Training-side mel preprocessing. The Pi app's vendored copy must stay byte-identical for the drift test.
- `src/acoustic/classification/efficientat/mel_banks_128_1024_32k.pt` — Precomputed mel filterbanks (128 mels, 1024 FFT, 32 kHz). Vendored into the edge app.
- `src/acoustic/classification/efficientat/model.py` — EfficientAT MN10 architecture definition. Needed by the host-side conversion script.
- `src/acoustic/classification/efficientat/config.py` — Training-side model/preprocess config constants. Conversion script must load these to construct the model before `torch.onnx.export`.
- `src/acoustic/classification/efficientat/classifier.py` — Reference inference wrapper; the Pi app's hysteresis + scoring logic should match the state-machine semantics used in the service (CLS-03).
- `models/efficientat_mn10_v6.pt` — The target checkpoint to convert.

### Project / Phase Constraints
- `.planning/PROJECT.md` — Project context, constraints, current milestone.
- `.planning/REQUIREMENTS.md` — CLS-01..CLS-04 (classification & hysteresis pattern), AUD-01..AUD-03 (audio capture patterns to mirror where useful).
- `.planning/ROADMAP.md` — Phase 21 entry and its Phase 20 dependency.

### Prior Decisions to Respect
- `.planning/phases/07-research-cnn-and-inference-integration/07-CONTEXT.md` — EfficientAT selection + integration decisions.
- `.planning/phases/08-pytorch-training-pipeline/08-CONTEXT.md` — Training-side preprocessing config that drives `mel_banks_128_1024_32k.pt`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/acoustic/classification/efficientat/preprocess.py` + `mel_banks_128_1024_32k.pt` — vendored into `apps/rpi-edge/` to guarantee training/inference parity.
- `src/acoustic/classification/efficientat/model.py` + `config.py` — used by the host-side conversion script to load weights before ONNX export.
- Hysteresis state-machine pattern from the live service (CLS-03) — ported (not imported) into the edge app.

### Established Patterns
- The main service is Dockerized, FastAPI, UMA-16, and heavyweight. The Pi edge app is intentionally the opposite: bare-metal systemd, single mic, ONNX Runtime, minimal deps. Do not try to share runtime code — share only preprocessing (vendored) and model weights (converted).
- Configuration in prior phases leans on dataclasses + Python config objects; the Pi app introduces YAML + CLI, which is new for this repo.

### Integration Points
- New top-level directory: `apps/rpi-edge/`.
- New host-side script: `scripts/convert_efficientat_to_onnx.py`.
- New install script: `scripts/install_edge_rpi.sh`.
- New CI test: preprocess/mel-bank drift test between vendored copies and `src/acoustic/classification/efficientat/`.

### Creative Options (enabled by existing architecture)
- Because training preprocessing lives in a single `preprocess.py` + mel-bank tensor file, vendoring is clean: two files to copy, one drift test to maintain.
- Because the `.pt` checkpoint is already committed in `models/`, host-side conversion is fully reproducible from the repo.

</code_context>

<specifics>
## Specific Ideas

- Target checkpoint is explicitly `models/efficientat_mn10_v6.pt`. Earlier versions (`_v2..v5`) are not in scope.
- Single LED semantics are deliberately blunt: off = idle, on = latched detection. No multi-color, no state panel.
- Optional speaker alarm is a late-added requirement; it is an *independent* output channel from GPIO and must degrade silently if no audio device is present.
- Detection JSONL log is a *durability* requirement, not a debugging one — the user explicitly distinguished it from general logs and said "the full log can be turned off in settings but not this one".

</specifics>

<deferred>
## Deferred Ideas

- Bearing / DOA estimation on Pi (no array).
- Multi-mic or UMA-16 on Pi.
- Model hot-swap or remote model push.
- Cloud reporting / remote detection streaming.
- OTA updates or SD-image provisioning.
- Web dashboard on Pi (current `/status` is a localhost JSON endpoint only).
- Buzzer / relay GPIO output (structure leaves room, but v1 ships LED + optional speaker only).
- Hot-reload of config file.
- Static int8 calibration (start with dynamic; upgrade only if accuracy guard trips).

</deferred>

---

*Phase: 21-build-raspberry-pi-4-edge-drone-detection-app-using-efficien*
*Context gathered: 2026-04-07*
