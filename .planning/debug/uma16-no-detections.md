---
status: awaiting_human_verify
trigger: "Pipeline runs against the live UMA-16 mic array but produces zero detection events."
created: 2026-04-06T00:00:00Z
updated: 2026-04-06T00:00:00Z
---

## Current Focus

hypothesis: Silent failure chain: UMA-16 ambient RMS (~9e-5) is below CNNWorker silence_threshold (1e-3), so CNN always reports prob=0.0, state machine never leaves NO_DRONE, demand-driven BF gate never opens, no detections ever fire. This is a "dead-on-arrival" default — even with a drone present, the gain path can't lift a -80 dBFS ambient to -60 dBFS unless the user explicitly starts a detection session with gain, and even gain=3 only reaches 2.7e-4 (still below 1e-3).
test: Verify by (1) confirming live RMS with real capture, (2) tracing gate logic, (3) checking whether any path bypasses the silence gate
expecting: Confirmed — live RMS is 9e-5, silence threshold is 1e-3, ratio 11x too quiet
next_action: Verify by reading CNNWorker silence gate, then propose fix

## Symptoms

expected: When UMA-16 is plugged in and pipeline starts, system picks up signal on all 16 channels, runs SRP-PHAT + CNN classifier, and publishes detection events (UI + ZMQ).
actual: Pipeline starts cleanly and runs, but no detections produced from live UMA-16 capture. UI/event stream shows nothing.
errors: No errors in backend logs — silent failure. Logs clean, pipeline reports running, nothing downstream fires.
reproduction: Start service against live UMA-16 USB capture path. No detections appear.
started: First end-to-end test — has never worked with real array. Recent phase 17-03 wired real SRP-PHAT + demand-driven gate (prime suspect).

## Eliminated

## Evidence

- timestamp: 2026-04-06T00:00:00Z
  checked: Device enumeration and 1s live capture from UMA16v2 (device 3) at 48kHz, 16ch, float32, blocksize 7200 (0.15s) — same params as service
  found: Capture works end-to-end. 6 chunks received, shape (43200, 16), all 16 channels non-zero. Per-channel RMS ~9.8e-5 to 1.2e-4. Mono (mean across channels) RMS = 9.0e-5. Peak abs = 1.17e-3. In dBFS: -81 dBFS mono, -80 dBFS per-ch.
  implication: Hardware capture path is healthy. Signal is real but very quiet (typical for MEMS array in ambient room). Issue is NOT "no data flowing" — it's "data is below every downstream threshold."

- timestamp: 2026-04-06T00:00:00Z
  checked: src/acoustic/classification/worker.py CNNWorker._loop silence gate (lines 133-145)
  found: Hard-coded `silence_threshold=0.001` (default). If `rms < 0.001`, worker writes ClassificationResult(drone_probability=0.0) WITHOUT calling preprocessor/classifier and `continue`s.
  implication: With ambient mono RMS = 9e-5, every push is gated as silence → prob=0.0 → state machine stays NO_DRONE forever.

- timestamp: 2026-04-06T00:00:00Z
  checked: src/acoustic/pipeline.py process_chunk demand-driven gate (lines 157-176) and _process_cnn gain application (lines 317-352)
  found: (1) Gate initializes `_last_bf_active_time = 0.0`. Since `now` is time.monotonic() (large), `now - 0.0 >> bf_holdoff (5s)`, so `bf_should_run = False` unless state machine is CONFIRMED. Beamforming returns empty peaks and zeroed map on every chunk. (2) Gain is applied ONLY when a detection session is active (`session is not None`). Default session gain is 3.0, but 9e-5 * 3 = 2.7e-4, still 3.7x below silence threshold. Without a detection session, no gain at all — gain=1.0 path means mono stays at 9e-5.
  implication: Two interlocking problems:
    (a) **Gain gap**: Even WITH a detection session (gain=3), ambient is still below the silence threshold. User needs gain ≈ 15-30 to get ambient above 1e-3.
    (b) **Gate deadlock**: Demand-driven BF gate is entirely downstream of CNN state machine. If CNN is dormant (no model loaded — default `models/uav_melspec_cnn.onnx` does NOT exist, only efficientat_*.pt and research_cnn_trained.pt exist) OR silence-gated (current case), the state machine never updates and BF never runs. There is no way to observe a beamforming map at rest to validate the geometry/filter path.

- timestamp: 2026-04-06T00:00:00Z
  checked: Default cnn_model_path = "models/uav_melspec_cnn.onnx" vs actual files in models/
  found: File does not exist. models/ contains efficientat_mn10*.pt, research_cnn_trained.pt, mn5/10.pt, ast-drone-detection/. No uav_melspec_cnn.onnx.
  implication: Without explicit env var override, classifier loads in DORMANT mode at startup (main.py line 362-367). CNNWorker._loop returns early at line 153 (`if self._classifier is None: continue`) — NEVER writes a ClassificationResult. `_latest` stays None. State machine never updates. Gate never opens. No beamforming, no detections, no CNN probability reported to UI. COMPLETELY SILENT.

## Resolution

root_cause: |
  The CNNWorker silence gate (`silence_threshold=0.001` hard-coded) rejects every audio chunk from the live UMA-16 array because the MEMS mic ambient noise floor is ~8e-5 RMS (−82 dBFS mono) — more than 10x below the threshold. Chain of silent failures that result in zero detections:

  1. UMA-16 produces ambient mono RMS ≈ 8e-5 (per-channel ≈ 1e-4). This is REAL signal, not silence — confirmed by direct sounddevice capture and per-channel RMS measurements on all 16 channels.
  2. CNNWorker._loop checks `rms < 0.001` first and writes `ClassificationResult(drone_probability=0.0)` without running preprocessing or classification. Every single chunk is treated as silence.
  3. DetectionStateMachine never receives a probability ≥ enter_threshold (0.80), so state stays NO_DRONE forever.
  4. BeamformingPipeline.process_chunk (phase 17-03 demand-driven gate) only opens when `state == CONFIRMED`. With `_last_bf_active_time = 0.0` at init and state never reaching CONFIRMED, `bf_should_run` is always False. SRP-PHAT never runs on live data, `latest_peaks` is always [], no events are published.
  5. Even with an explicit detection session started from the Pipeline tab (gain=3.0 default), 8e-5 * 3 = 2.4e-4, still 4x below the silence threshold. The Pipeline tab would need gain ≥ 15 (extreme) to overcome the gate.
  6. Additionally, the default `cnn_model_path = "models/uav_melspec_cnn.onnx"` does not exist — the classifier boots in dormant mode and CNNWorker._loop never even reaches the silence gate (it returns early on `classifier is None`). User must manually start a detection session via the Pipeline tab to load a real model — but even then, the silence gate blocks everything.

  Core design flaw: the silence threshold was tuned for audio sources producing ~−40 dBFS nominal signal, but UMA-16v2 MEMS mics produce ~−80 dBFS ambient. The threshold is 20 dB too aggressive for this hardware. Not a Phase 17-03 regression per se — the silence gate has been there since Phase 06/07 — but the BF demand-driven gate added in 17-03 means the silence gate now blocks EVERYTHING downstream, including beamforming visibility. Before 17-03, BF ran on every chunk regardless of CNN state, so the user would at least see a beamforming map; now the map is also gated.

fix: |
  1. Wire `silence_threshold` through AcousticSettings (`cnn_silence_threshold`, default `1.0e-5`).
  2. Pass it into `CNNWorker(...)` from main.py lifespan.
  3. New default `1e-5` is below MEMS ambient (8e-5) but still catches truly dead signal (all zeros from a disconnected/muted stream), preserving the original intent of the silence gate.

verification: |
  1. Run a fresh capture of 1s UMA-16 audio (ambient, no drone) and confirm mono RMS ≈ 8e-5 passes the new 1e-5 gate.
  2. Manually instantiate CNNWorker with the new threshold, push real captured audio, and confirm it no longer short-circuits to prob=0.0 (instead hits the "no preprocessor" / "no classifier" path when dormant, or runs inference if a model is loaded).
  3. Confirm tests still pass.
  4. Require user to confirm end-to-end: start service → open Pipeline tab → load an efficientat model → play drone audio → observe CNN probability rising and detection events firing.

files_changed:
  - src/acoustic/config.py  # add cnn_silence_threshold setting (default 1e-5)
  - src/acoustic/main.py    # pass settings.cnn_silence_threshold into CNNWorker
self_verified:
  - AcousticSettings loads cnn_silence_threshold=1e-5 correctly
  - Real UMA-16 ambient capture (mono RMS ≈ 1.3e-4) passes the new 1e-5 gate
  - CNNWorker with a fake classifier now reaches inference path on ambient audio (returns real prob, not 0.0)
  - CNNWorker with old 1e-3 threshold confirmed to still gate ambient to 0.0 (regression baseline)
  - 42/42 unit tests pass in test_worker.py + test_config.py
  - 27/27 pipeline tests pass including phase 17-03 integration tests
still_needs_user_verification:
  - Start the service against the live UMA-16 and observe that the pipeline is now able to produce detections when a drone sound is played.
  - Specifically: open the Pipeline tab, load an efficientat model (e.g. efficientat_mn10_v6.pt), start a detection session, and confirm CNN drone probability updates in real time. Play drone audio and confirm the beamforming map populates and detection events fire.
