# Phase 3: CNN Classification and Target Tracking - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver binary drone/not-drone detection using the POC's trained CNN model (converted to ONNX), a target tracking state machine with UUID-based ID persistence, and ZeroMQ PUB/SUB event publishing. This phase does NOT include drone type classification (multi-class) or Doppler speed estimation — both are deferred to milestone 2.

</domain>

<decisions>
## Implementation Decisions

### CNN Architecture & Inference
- **D-01:** Binary drone/not-drone detection only. Single sigmoid output producing drone probability. Drone type classification (CLS-02) deferred to ms-2.
- **D-02:** Use the POC's existing trained model (`uav_melspec_cnn.h5`) converted to ONNX format. Inference via `onnxruntime` (lightweight ~50MB, no TF dependency in production). One-time conversion using `tf2onnx`.
- **D-03:** Mel-spectrogram pipeline replicates POC exactly: resample to 16kHz, 2-second segments, 64 mels, n_fft=1024, hop_length=256, pad/trim to 128 frames, mean/std normalization. Input shape: (1, 128, 64, 1).
- **D-04:** CNN inference gated on beamforming peak detection. Only run the model when SRP-PHAT detects a peak above the noise threshold. Saves compute — CNN doesn't run on silence. Matches POC's ENERGY+MODEL gating pattern.

### Target Tracking State Machine
- **D-05:** Hysteresis state machine prevents detection flickering (CLS-03). Enter/exit thresholds and confirmation hit counts are implementation details for Claude to decide.
- **D-06:** Target created with UUID on first confirmed detection. Target persists for 5 seconds after last detection signal before being marked as lost.
- **D-07:** Doppler speed estimation (TRK-02) deferred to ms-2. `speed_mps` field remains `null` in this phase.

### Event Publishing (WebSocket, not ZeroMQ)
- **D-08:** ~~ZeroMQ~~ Dedicated `/ws/events` WebSocket endpoint for external event consumers. Separate from the existing `/ws/targets` (UI heatmap/target state). Messages include an `event` field with values: `new`, `update`, `lost`.
- **D-09:** JSON message schema carries: event type, target ID, bearing (az/el degrees), drone probability/confidence. Speed field present but null until ms-2.
- **D-10:** ~~PUB/SUB pattern via pyzmq~~ WebSocket broadcast pattern. All connected `/ws/events` clients receive all events. No topic filtering needed — single event stream. Removes `pyzmq` dependency entirely.

### Scope Deferrals to Milestone 2
- **D-11:** Drone type classification (CLS-02: multi-class — 5-inch, Mavic, Matrice, EvoMax, FlyCart, etc.) deferred to ms-2.
- **D-12:** Doppler speed estimation (TRK-02) deferred to ms-2. UMA-16v2 aperture feasibility needs validation.

### Claude's Discretion
- Hysteresis thresholds and confirmation hit counts (D-05 details)
- CNN worker threading model (background thread with queue, matching POC's CNNWorker pattern)
- ZeroMQ publish frequency for update events
- How the CNN inference integrates with the existing `BeamformingPipeline` class
- Multi-target handling when beamforming detects multiple peaks simultaneously
- Web UI updates to show real detection data instead of placeholder targets

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### POC CNN Implementation
- `POC-code/PT520/PTZ/uma16_master_live_with_polar.py` — CNN inference pipeline: `predict_drone_prob()` (line 341), `make_melspec()` (line 313), `CNNWorker` class (line 359), mel-spectrogram params (SR_CNN=16000, CNN_SEGMENT_SECONDS=2.0, n_fft=1024, hop_length=256, n_mels=64). Port the preprocessing pipeline exactly.
- `POC-code/PT520/PTZ/uma16_master_live_with_polar.py` lines 760-870 — Main detection loop showing ENERGY+MODEL gating pattern, beamforming peak threshold check before CNN inference.

### Existing Service Code
- `src/acoustic/pipeline.py` — `BeamformingPipeline` class. `process_chunk()` returns `PeakDetection | None`. CNN inference hooks into this — when peak detected, feed audio to CNN.
- `src/acoustic/types.py` — `placeholder_target_from_peak()` is the explicit Phase 3 swap point. Replace with real CNN-backed detection.
- `src/acoustic/api/models.py` — `TargetState` Pydantic model already has all fields (id, class_label, speed_mps, az_deg, el_deg, confidence). Wire to real data.
- `src/acoustic/api/websocket.py` — `/ws/targets` endpoint already streams JSON target updates. Swap placeholder data for real tracking data.
- `src/acoustic/config.py` — `AcousticSettings` needs CNN and ZMQ config additions (model path, ZMQ endpoint, detection thresholds).

### Training Data
- `audio-data/data/` — Labeled recordings: `drone/` (13 subcategories by type/distance), `background/` (~490 files), `other/`. `index.jsonl` has metadata (label, distance, device, sample rate, duration, path).
- POC model file: `/home/skyfortubuntu/Skyfort/Cursor/acoular/models/uav_melspec_cnn.h5` (needs to be located/converted to ONNX).

### Requirements
- `REQUIREMENTS.md` — CLS-01, CLS-03, CLS-04, TRK-01, TRK-03, TRK-04, TRK-05 are in scope. CLS-02 and TRK-02 deferred to ms-2.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BeamformingPipeline.process_chunk()` — Returns `PeakDetection` with az/el/power. CNN inference gates on this.
- `AudioRingBuffer` — Ring buffer provides raw 16-channel chunks. CNN needs mono mix-down + resample to 16kHz.
- `TargetState` Pydantic model — Already has all fields, just needs real data.
- `/ws/targets` WebSocket — Already streams target state to UI, just swapping placeholder for real data.
- `placeholder_target_from_peak()` — Explicit swap point for Phase 3.

### Established Patterns
- Background thread with daemon=True for pipeline processing (see `BeamformingPipeline._run_loop`)
- POC's `CNNWorker` uses queue.Queue(maxsize=1) for non-blocking inference — same pattern fits here
- Config via `AcousticSettings` (Pydantic BaseSettings with env var support)
- FastAPI lifespan for startup/shutdown of background workers

### Integration Points
- Pipeline: CNN worker receives audio when beamforming peak is detected
- Targets: Real `TargetState` objects replace `placeholder_target_from_peak()`
- WebSocket: `/ws/targets` streams real tracking data instead of placeholder
- REST: `/api/targets` returns real target list
- Config: New env vars for model path, ZMQ endpoint, detection thresholds
- WebSocket: New `/ws/events` endpoint for external event consumers (separate from `/ws/targets`)

</code_context>

<specifics>
## Specific Ideas

- User explicitly wants to reuse the POC's trained model as-is — convert .h5 to .onnx, don't retrain
- Mel-spectrogram preprocessing must exactly match POC (librosa params) to ensure model compatibility
- Training script deferred to Phase 5 — this phase ships inference only
- `class_label` will be "drone" or "background" (binary), not drone subtypes

</specifics>

<deferred>
## Deferred Ideas

- **Drone type classification (CLS-02)** — Multi-class (5-inch, Mavic, Matrice, EvoMax, FlyCart, etc.) deferred to milestone 2
- **Doppler speed estimation (TRK-02)** — Deferred to milestone 2. UMA-16v2 aperture feasibility uncertain.
- **CNN training pipeline** — Phase 5 as planned. Phase 3 ships pre-trained model only.

</deferred>

---

*Phase: 03-cnn-classification-and-target-tracking*
*Context gathered: 2026-03-31*
