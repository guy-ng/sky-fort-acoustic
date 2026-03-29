# Project Research Summary

**Project:** Sky Fort Acoustic Service
**Domain:** Real-time acoustic drone detection and tracking microservice
**Researched:** 2026-03-29
**Confidence:** HIGH

## Executive Summary

Sky Fort Acoustic is a real-time microservice that captures 16-channel audio from a miniDSP UMA-16v2 microphone array, performs SRP-PHAT beamforming to localize sound sources, classifies them as drones using a CNN on mel-spectrograms, and publishes detection events over ZeroMQ for downstream consumption. This is a well-studied domain with proven patterns: the pipeline of beamforming followed by ML classification is the industry standard used by Fraunhofer IDMT, Squarehead Discovair, and multiple academic systems. A working POC already validates the core approach -- the task is to refactor it from a monolithic prototype into a clean, containerized microservice with a React monitoring UI.

The recommended approach is a threaded pipeline architecture using FastAPI (async REST + WebSocket), custom NumPy-based SRP-PHAT beamforming (not Acoular -- the POC's 180-line implementation is simpler and sufficient for a 4x4 array), PyTorch for CNN training and inference (replacing the POC's TensorFlow), and a React 19 + Vite 8 frontend matching the existing sky-fort-dashboard. The service runs as a single Docker container with USB audio passthrough. Audio capture uses a callback-based `sounddevice.InputStream` feeding a ring buffer, with beamforming and CNN inference decoupled into separate threads to avoid the GIL bottleneck demonstrated in the POC's Python-loop-over-pairs pattern.

The key risks are: (1) USB audio device instability in Docker requiring udev rules and health-check reconnection logic, (2) buffer overflows from the POC's blocking `sd.rec()` pattern which must be replaced with callback-based capture from day one, (3) spatial aliasing above 4 kHz from the array's 42mm spacing requiring hard frequency limits, and (4) CNN overfitting to recording environment requiring session-level data splits. All four risks have clear mitigations identified in research.

## Key Findings

### Recommended Stack

The stack splits into a Python backend for audio processing and ML, and a TypeScript/React frontend matching sky-fort-dashboard conventions. See [STACK.md](./STACK.md) for full rationale and version pins.

**Core technologies:**
- **Python 3.11+ / FastAPI**: Async-native web framework with WebSocket support. Replaces Flask from the POC to handle concurrent audio streaming + REST + WebSocket
- **sounddevice + NumPy + SciPy**: Proven audio capture (UMA-16v2 via PortAudio/ALSA) and DSP core. All beamforming math runs in NumPy
- **Custom SRP-PHAT (NumPy)**: The POC's 180-line beamforming implementation is preferred over Acoular. Acoular's pipeline model adds complexity (Numba compilation, caching, lazy evaluation) without benefit for a 4x4 array with known geometry
- **PyTorch + torchaudio**: CNN training and inference for mel-spectrogram drone classification. Replaces TensorFlow -- the acoustic ML research community has converged on PyTorch
- **ZeroMQ (pyzmq)**: Brokerless PUB/SUB for detection events. Project requirement
- **React 19 + Vite 8 + Tailwind 4 + TanStack Query 5**: Must match sky-fort-dashboard exactly for component sharing

**Critical stack decision -- PyTorch over TensorFlow:** The ARCHITECTURE.md references TensorFlow/Keras (matching the POC), but STACK.md makes a strong case for PyTorch: 85% of audio ML research uses it, better debugging (eager mode), and `torch.compile()` for production. This means the POC's `.h5` model cannot be reused directly -- retraining is required. This is acceptable because the training pipeline is being rebuilt anyway, and carrying forward TensorFlow adds a second ML framework with no benefit.

**Critical stack decision -- Custom SRP-PHAT over Acoular:** The ARCHITECTURE.md references Acoular for beamforming, but STACK.md correctly identifies that Acoular's pipeline architecture (lazy evaluation, streaming generators, caching) is designed for offline analysis, not tight real-time loops. The POC's custom implementation runs in under 50ms per chunk with full control. Acoular can serve as a reference/validation tool if needed later.

### Expected Features

The critical path is: Audio Capture -> Beamforming -> Peak Detection -> CNN Classification -> ZeroMQ Events. Everything else hangs off this spine. See [FEATURES.md](./FEATURES.md) for the full feature landscape and dependency graph.

**Must have (table stakes):**
- Real-time 16-channel audio capture at 48kHz
- SRP-PHAT beamforming with spatial sound map
- Peak detection with azimuth/elevation output
- Adaptive noise gate with percentile-based calibration
- Binary drone/not-drone CNN classification (mel-spectrogram)
- Target ID assignment with lifecycle state machine
- ZeroMQ PUB/SUB event publishing (detection, tracking, lost)
- REST API for beamforming map and target state
- Web UI with live beamforming heatmap and target overlay
- Docker deployment with USB audio passthrough

**Should have (differentiators):**
- Multi-class drone type identification (7+ drone types already in dataset)
- Recording raw 16-channel audio with metadata tagging
- Pipeline replay from recordings (turns field data into test cases)
- In-service CNN training pipeline with model versioning
- Hysteresis-based detection state machine (enter/exit thresholds)
- Configurable frequency band focus

**Defer indefinitely:**
- Doppler speed estimation -- the UMA-16v2's small aperture (126mm) likely cannot deliver reliable frequency resolution for acoustic Doppler. Requires feasibility validation before committing
- PTZ camera control, visual/YOLO detection, RF fusion, multi-array support, range estimation (all out of scope per anti-features analysis)

### Architecture Approach

A threaded pipeline architecture with four processing stages communicating via thread-safe shared state and queues. FastAPI runs in the async event loop, audio capture and beamforming share a dedicated thread (beamforming is fast at 5-10ms), CNN inference runs in a separate thread to avoid blocking, and ZeroMQ publishing runs as an async task. The React frontend is built separately and served as static files by FastAPI. See [ARCHITECTURE.md](./ARCHITECTURE.md) for full data flow diagrams.

**Major components:**
1. **Audio Capture** -- callback-based `sounddevice.InputStream`, writes to ring buffer, dedicated thread
2. **Beamforming Engine** -- custom SRP-PHAT on raw PCM chunks, produces spatial map + peak direction
3. **CNN Classifier** -- separate thread, reads 2s segments from ring buffer, outputs drone probability
4. **Target Tracker** -- state machine (IDLE -> DETECTED -> TRACKING -> LOST), assigns IDs, computes bearing
5. **ZeroMQ Publisher** -- topic-based PUB socket (detection/tracking/lost events)
6. **REST API + WebSocket** -- FastAPI serving beamforming map, targets, recording CRUD, live WebSocket stream
7. **Recording Manager** -- raw 16-ch WAV capture + structured metadata (HDF5 or SQLite)
8. **Training Pipeline** -- subprocess for CNN training from labeled recordings, model hot-reload
9. **Web UI** -- React app with live heatmap, target overlay, recording controls

### Critical Pitfalls

See [PITFALLS.md](./PITFALLS.md) for the complete list of 14 pitfalls with detection and prevention strategies.

1. **USB audio device instability in Docker** -- use udev rules for stable device paths, `--device /dev/snd` + `--group-add audio` (not `--privileged`), health-check loop that verifies device presence, graceful reconnection logic
2. **Buffer overflows from blocking capture** -- replace POC's `sd.rec(blocking=True)` with callback-based `sounddevice.InputStream` from day one. This is an architectural decision that cannot be retrofitted
3. **Spatial aliasing above 4 kHz** -- hard-code frequency band limits (100-2000 Hz) with configuration guard rejecting bands above 4000 Hz. Document array resolution limitations
4. **CNN overfitting to recording environment** -- split data by recording session (not random sample), include diverse ambient noise conditions, report per-session accuracy
5. **Python GIL blocking the pipeline** -- vectorize the 120 microphone-pair loop into batched NumPy operations, keep CNN inference in separate thread/process, set thread count env vars (OPENBLAS_NUM_THREADS, NUMBA_NUM_THREADS)

## Implications for Roadmap

Based on combined research, the following 6-phase structure is recommended. This ordering is driven by the dependency chain identified in FEATURES.md and the "build order" analysis in ARCHITECTURE.md, with phase boundaries aligned to avoid the critical pitfalls.

### Phase 1: Audio Capture and Beamforming Core
**Rationale:** Everything depends on reliable audio input. The capture architecture (callback vs blocking) is an irreversible decision that affects all downstream components. Beamforming is tightly coupled to capture.
**Delivers:** Working audio capture from UMA-16v2 with callback-based InputStream, SRP-PHAT beamforming producing spatial map and peak direction, SharedState pattern for inter-thread communication, Docker container with USB audio passthrough and multi-stage Dockerfile.
**Addresses:** Real-time 16-ch capture, beamforming spatial map, peak detection with azimuth/elevation, noise gate, Docker deployment
**Avoids:** USB device instability (P1), buffer overflow (P2), GIL blocking (P5), thread oversubscription (P11), channel mapping errors (P12)

### Phase 2: REST API and Web UI
**Rationale:** Visual feedback on beamforming output is essential for tuning and validating Phase 1. The API layer is needed by both the web UI and integration tests.
**Delivers:** FastAPI REST endpoints (beamforming map, health check), WebSocket for live streaming, React web UI with beamforming heatmap display, static file serving from FastAPI.
**Addresses:** REST API for beamforming map, Web UI live beamforming heatmap, active target overlay (placeholder until tracker exists)
**Avoids:** Serialization bottleneck (P8 -- use binary WebSocket), WebSocket connection leaks (P13), REST polling anti-pattern

### Phase 3: CNN Classification and Target Tracking
**Rationale:** With audio capture proven and visual validation available, add the detection intelligence. CNN and tracker are tightly coupled (tracker consumes CNN probability).
**Delivers:** PyTorch CNN classifier (binary drone/not-drone), mel-spectrogram feature extraction via torchaudio, target tracker state machine with ID assignment and lifecycle, ZeroMQ PUB/SUB event publishing, hysteresis-based detection (enter/exit thresholds).
**Addresses:** CNN classification, target ID assignment, ZeroMQ events, hysteresis state machine
**Avoids:** Spatial aliasing feeding bad data to CNN (P3 -- solved in Phase 1), slow subscriber memory explosion (P6 -- set HWM at socket creation)

### Phase 4: Recording and Playback
**Rationale:** Recording infrastructure must exist before the training pipeline. Field recordings become both test fixtures and training data. This phase is independent of detection quality.
**Delivers:** Raw 16-ch WAV recording triggered via API, structured metadata storage (session ID, labels, conditions), pipeline replay from recorded WAV files, JSONL or HDF5 metadata index.
**Addresses:** Recording with pipeline replay, metadata tagging
**Avoids:** Metadata loss (P9 -- structured format from the start), environment overfitting groundwork (P4 -- session-level metadata enables proper splits later)

### Phase 5: CNN Training Pipeline
**Rationale:** Depends on labeled recordings from Phase 4 and the model architecture from Phase 3. This enables the model to improve with deployment-specific data.
**Delivers:** Training UI in web app, dataset management with session-level splits, model versioning, validation reporting, subprocess-based training (not in main process), model hot-reload.
**Addresses:** In-service CNN training pipeline, multi-class drone type identification, configurable frequency band, background noise class
**Avoids:** Training in main process starving real-time pipeline (ARCHITECTURE.md anti-pattern 4), overfitting (P4 -- session splits, per-session accuracy reporting)

### Phase 6: Hardening and Production Readiness
**Rationale:** Polish phase after all features are functional. Focus on reliability, monitoring, and deployment concerns.
**Delivers:** Device health monitoring and auto-reconnection, wind noise detection for outdoor deployment, performance profiling and optimization, comprehensive integration tests, production Docker configuration.
**Addresses:** Remaining edge cases from pitfalls research
**Avoids:** Wind noise saturation (P14), USB re-enumeration in production (P1 hardening)

### Phase Ordering Rationale

- **Audio before everything:** The entire system is meaningless without reliable 16-channel capture. The callback-vs-blocking architecture decision in Phase 1 is irreversible and affects all downstream phases
- **API + UI before CNN:** Visual validation of beamforming output is critical for tuning array parameters and verifying the frequency band limits. Without the UI, Phase 3 development is blind
- **CNN + Tracker together:** These are tightly coupled -- the tracker consumes CNN probability to make state transitions. Building one without the other produces untestable code
- **Recording before training:** The training pipeline consumes labeled recordings. The metadata schema designed in Phase 4 directly determines what the training pipeline in Phase 5 can do
- **Hardening last:** Production reliability concerns (device reconnection, wind noise) are best addressed after the full pipeline works end-to-end

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Callback-based sounddevice capture with UMA-16v2 needs prototype validation. The POC uses blocking capture; the callback pattern has not been proven with this specific hardware
- **Phase 3:** PyTorch model architecture selection (EfficientNet vs ResNet vs custom) needs benchmarking on target hardware. Also, ZeroMQ message schema design needs coordination with downstream consumers
- **Phase 5:** Training pipeline UX patterns (dataset splits, augmentation config, validation reporting) are not well-covered by the POC

Phases with standard patterns (skip research):
- **Phase 2:** FastAPI + React + WebSocket is thoroughly documented. Standard web stack with no domain-specific unknowns
- **Phase 4:** WAV recording + metadata management is straightforward file I/O. Well-understood patterns
- **Phase 6:** Docker hardening and health checks are standard DevOps practices

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies are well-established. POC validates core choices (sounddevice, NumPy beamforming). PyTorch over TensorFlow is a deliberate upgrade with clear rationale. Frontend matches existing dashboard. |
| Features | HIGH | Feature landscape derived from commercial systems (Discovair, AARTOS, AirGuard), academic literature, and the working POC. Table stakes are clear. Anti-features are well-justified. |
| Architecture | HIGH | Pipeline-with-shared-state pattern is proven in the POC. Thread separation of audio/beamforming/CNN/API follows established real-time audio processing patterns. Build order follows clear dependency chains. |
| Pitfalls | HIGH | Critical pitfalls (USB instability, buffer overflow, spatial aliasing, GIL) are confirmed by POC code analysis, hardware specs, and published issues. Prevention strategies are concrete and actionable. |

**Overall confidence:** HIGH

### Gaps to Address

- **Doppler speed estimation feasibility:** The UMA-16v2's 126mm aperture and 42mm spacing may not support reliable acoustic Doppler extraction. At typical drone speeds, Doppler shift is 0.3-12 Hz, requiring long FFT windows that conflict with real-time latency. Recommend a focused feasibility study during Phase 3 before committing to this feature
- **PyTorch model migration:** The POC has a trained TensorFlow/Keras model (`.h5` format). Switching to PyTorch requires retraining from scratch. Verify that the existing labeled dataset is sufficient and properly structured before Phase 3
- **UMA-16v2 channel mapping verification:** The POC's microphone-to-channel mapping is annotated with "we assume." This must be empirically verified (tap test) before any beamforming work in Phase 1
- **Callback-based capture with UMA-16v2:** The POC uses blocking `sd.rec()`. The callback-based `InputStream` approach has not been tested with this specific USB device. Prototype early in Phase 1
- **Downstream ZeroMQ consumers:** The message schema and topic structure need coordination with whatever services subscribe to detection events. This interface contract should be defined during Phase 3 planning

## Sources

### Primary (HIGH confidence)
- POC codebase: `POC-code/PT520/PTZ/radar_gui_all_mics_fast_drone.py` -- SRP-PHAT implementation, SharedState pattern, CNN worker threading
- POC codebase: `POC-code/scripts/POC_Recorder.py` -- recording pipeline, metadata schema
- [miniDSP UMA-16 v2 specs](https://www.minidsp.com/products/usb-audio-interface/uma-16-microphone-array) -- hardware constraints (42mm spacing, 16ch, 48kHz)
- [sounddevice documentation](https://python-sounddevice.readthedocs.io/) -- InputStream callback API, buffer management
- [FastAPI documentation](https://fastapi.tiangolo.com/) -- async patterns, WebSocket, lifespan events
- [PyTorch/torchaudio documentation](https://docs.pytorch.org/audio/) -- MelSpectrogram transforms, model training
- sky-fort-dashboard `package.json` -- frontend stack reference (React 19, Vite 8, Tailwind 4)

### Secondary (MEDIUM confidence)
- [Squarehead Discovair G2+](https://www.sqhead.com/drone-detection) -- commercial feature benchmarking
- [Fraunhofer IDMT acoustic detection](https://www.idmt.fraunhofer.de/en/Press_and_Media/press_releases/2025/acoustic-drone-detection.html) -- research validation
- [AUDRON framework](https://arxiv.org/html/2512.20407) -- deep learning drone classification patterns
- [DSIAC acoustic drone detection primer](https://dsiac.dtic.mil/primers/what-is-an-acoustic-drone-detection-system/) -- capabilities and limitations overview
- [Acoular GitHub issue #187](https://github.com/acoular/acoular/issues/187) -- thread oversubscription documentation
- [ZeroMQ Guide Chapter 5](https://zguide.zeromq.org/docs/chapter5/) -- slow subscriber patterns

### Tertiary (LOW confidence)
- Doppler speed estimation from single planar array -- limited academic literature for acoustic (vs radar) micro-Doppler. Feasibility unconfirmed for UMA-16v2's small aperture
- Multi-class drone type classification accuracy -- research uses larger arrays and controlled conditions. Real-world accuracy with 4x4 array at field distances is uncertain

---
*Research completed: 2026-03-29*
*Ready for roadmap: yes*
