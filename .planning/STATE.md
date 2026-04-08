---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: MVP
status: executing
stopped_at: Session resumed, routing to /gsd-execute-phase 21 (final plan 21-08)
last_updated: "2026-04-08T12:59:08.513Z"
last_activity: 2026-04-08 -- Phase 23 planning complete
progress:
  total_phases: 19
  completed_phases: 11
  total_plans: 58
  completed_plans: 42
  percent: 72
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-03)

**Core value:** Reliably detect and classify drones acoustically in real time, publishing target events over ZeroMQ so downstream systems can act on them.
**Current focus:** Phase 20.1 — acquire-noise-corpora-esc50-urbansound8k-fsd50k-subset-and-a

## Current Position

Phase: 20.1 (acquire-noise-corpora-esc50-urbansound8k-fsd50k-subset-and-a) — EXECUTING
Plan: 1 of 3
Status: Ready to execute
Last activity: 2026-04-08 -- Phase 23 planning complete

## Performance Metrics

**Velocity:**

- Total plans completed: 8
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 17 | 3 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01 P01 | 8min | 2 tasks | 20 files |
| Phase 01 P02 | 9min | 2 tasks | 11 files |
| Phase 02 P01 | 7min | 3 tasks | 11 files |
| Phase 02 P02 | 8min | 2 tasks | 28 files |
| Phase 03 P01 | 6min | 1 tasks | 9 files |
| Phase 03 P02 | 5m19s | 1 tasks | 7 files |
| Phase 03 P03 | 12min | 2 tasks | 9 files |
| Phase 11 P02 | 10min | 2 tasks | 7 files |
| Phase 13 P01 | 3m40s | 1 tasks | 2 files |
| Phase 13 P02 | 4m01s | 2 tasks | 5 files |
| Phase 14 P01 | 19min | 2 tasks | 13 files |
| Phase 14 P02 | 23min | 2 tasks | 8 files |
| Phase 15 P02 | 7m17s | 2 tasks | 6 files |

## Accumulated Context

### Roadmap Evolution

- v3.0 milestone added (2026-04-03): DADS-Powered Detection Upgrade (Phases 13-16)
- Phase 13 added: DADS Dataset Integration and Training Data Pipeline
- Phase 14 added: EfficientAT Model Architecture with AudioSet Transfer Learning
- Phase 15 added: Advanced Training Enhancements - Focal Loss, Noise Augmentation, Balanced Sampling
- Phase 16 added: Edge Export Pipeline - ONNX TensorRT TFLite Quantization
- v4.0 milestone added (2026-04-04): Research-Based Beamforming & Direction Calculation (Phases 17-19)
- Phase 17 added: Beamforming Engine Upgrade and Pipeline Integration (BF-10 through BF-15)
- Phase 18 added: Direction of Arrival and WebSocket Broadcasting (DOA-01 through DOA-03, DIR-01, DIR-02)
- Phase 19 added: Functional Beamforming Visualization (VIZ-01, VIZ-02)
- Source: Research-validated beamforming parameters, spatial aliasing analysis for UMA-16v2
- Phase 20 added: Retrain v7 with wide gain + room-IR augmentation, Vertex remote training, 60% overlap windows, expanded BG noise negatives
- Phase 20.1 inserted after Phase 20 (2026-04-07): Acquire noise corpora ESC50 / UrbanSound8K / FSD50K subset and add host preflight test (URGENT — unblocks Phase 20 Plan 20-05 Task 2 Vertex submission)
- Phase 21 added (2026-04-07): Build Raspberry Pi 4 edge drone-detection app using efficientat_mn10_v6.pt with configurable detection params, GPIO LED alarm, and model conversion
- Phase 22 added (2026-04-08): EfficientAT v8 retrain with fixed train/serve window contract (1.0s @ 32kHz, 50% overlap), 2026-04-08 field recordings, Vertex L4 us-east1. Driven by v7 regression post-mortem in .planning/debug/efficientat-v7-regression-vs-v6.md (root cause: window_samples literal 0.5s in trainer mismatched 1.0s inference contract).
- Phase 23 added (2026-04-08): Evaluate AUDRON multi-branch hybrid architecture (MFCC 1D-CNN + STFT 2D-CNN + BiLSTM-Attention + Autoencoder fusion) as alternative/complement to EfficientAT classifier line. Research-first phase — deliverable is adopt/reject/hybrid decision, NOT a trained model. Source: AmygdalaAI-India Lab framework, claims 98.51% accuracy / 300m+ detection. Must NOT disrupt phase 22 v8 retrain or phase 20.1 noise corpora.

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyTorch over TensorFlow for CNN (research recommendation -- existing .h5 model cannot be reused, retraining required)
- Custom SRP-PHAT over Acoular (POC's 180-line implementation is simpler and sufficient for 4x4 array)
- Callback-based sounddevice.InputStream over blocking sd.rec() (irreversible architecture decision)
- [Phase 01]: Ring buffer uses one-slot-reserved circular pattern for full/empty disambiguation
- [Phase 01]: AudioCapture callback does only np.copyto + monotonic timestamp -- no logging in audio thread
- [Phase 01]: Elevation test relaxed for planar array -- UMA-16v2 has zero z-baseline, poor elevation discrimination is physics, not a bug
- [Phase 01]: Frequency band test uses variance comparison due to GCC-PHAT magnitude normalization
- [Phase 02]: Map data transposed to [elevation][azimuth] row-major for canvas rendering
- [Phase 02]: WebSocket heatmap uses JSON handshake then binary float32 frames
- [Phase 02]: Heatmap 20 Hz poll, targets 2 Hz matching data change rates
- [Phase 02]: Pre-built 256-entry colormap LUT for O(1) heatmap pixel mapping
- [Phase 02]: useImperativeHandle pattern on HeatmapCanvas to avoid React re-renders per frame
- [Phase 03]: Used librosa for mel-spectrogram to match POC parameters exactly
- [Phase 03]: ONNX Runtime for inference (lighter than PyTorch, model-agnostic)
- [Phase 03]: Binary drone/not-drone only -- CLS-02 multi-class deferred to milestone 2
- [Phase 03]: WebSocket broadcast via asyncio.Queue, not ZeroMQ PUB/SUB (per D-10)
- [Phase 03]: Single-target tracking for Phase 3; multi-target is future enhancement
- [Phase 03]: speed_mps always None -- Doppler deferred to milestone 2 (per D-07)
- [Phase 03]: CNNWorker uses single-slot queue with drop semantics for non-blocking inference
- [Phase 03]: Fixed EventBroadcaster to use call_soon_threadsafe for thread-safe async delivery
- [Phase 11]: Evaluator refactored with evaluate_classifier and evaluate_ensemble, keeping backward-compatible evaluate(model_path)
- [Phase 11]: Ensemble factory in lifespan runs before single-model with fallback on failure
- [Phase 13]: Builder pattern for ParquetDataset to avoid 3x shard scanning for train/val/test splits
- [Phase 13]: 44-byte WAV header skip for in-memory Parquet audio decoding (no soundfile dep for this path)
- [Phase 13]: TrainingRunner branches on presence of train-*.parquet files in dads_path, not config flag
- [Phase 13]: Parquet created as post-processing after WAV write, keeping streaming WAV writer unchanged
- [Phase 14]: Vendored EfficientAT source as package (not pip) -- model code under efficientat/ with registry auto-registration
- [Phase 14]: Added channel unsqueeze between AugmentMelSTFT 3D output and Conv2D 4D input for EfficientAT inference
- [Phase 14]: Runner dispatch in TrainingManager._run() via config.model_type -- simpler than manager subclass
- [Phase 14]: EarlyStopping shared globally across all 3 EfficientAT training stages for cross-stage convergence
- [Phase 15]: All training uses logits mode -- both FocalLoss and BCEWithLogitsLoss expect raw logits
- [Phase 15]: TorchScript export with logits_mode=False for inference-ready sigmoid output
- [Phase 15]: ComposedAugmentation class over closure for DataLoader num_workers picklability

### Pending Todos

None yet.

### Blockers/Concerns

- **[ACTIVE 2026-04-07] Phase 20 blocked on noise-corpora data acquisition.** The committed `Dockerfile.vertex-base` and `build_env_vars_v7()` env vars assume ESC50, UrbanSound8K, and an FSD50K subset live under `data/noise/`. On disk, only `data/noise/fsd50k_subset/.gitkeep` exists — Plan 20-00 created placeholder dirs only. Building and submitting v7 as-is would silently train against ONE noise source (UMA-16 outdoor_quiet ambient pool, 31 min) instead of FOUR — gutting the whole point of Phase 20 and almost certainly failing the locked promotion gate (DADS≥0.95 / real_TPR≥0.80 / real_FPR≤0.05). See `.planning/phases/20-.../20-05-SUMMARY.md` "BLOCKED ON DATA ACQUISITION" section. **Resolution path:** create a new phase to acquire the three noise corpora, add a host-side preflight test that fails if any noise dir is empty, then resume `/gsd-execute-phase 20 --wave 4`.
- Doppler speed estimation feasibility uncertain (UMA-16v2 aperture may be too small -- validate during Phase 3)
- UMA-16v2 channel mapping needs empirical verification (tap test) before beamforming work
- Callback-based capture not yet proven with UMA-16v2 (prototype early in Phase 1)
- Beamforming is currently stubbed in pipeline.py -- Phase 17 must re-integrate the real SRP-PHAT engine

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260331-mok | Device disconnect overlay on heatmap | 2026-03-31 | c020721 | [260331-mok-device-state-and-reconnect-ws-disconnect](./quick/260331-mok-device-state-and-reconnect-ws-disconnect/) |
| 260331-myc | Fix device disconnect/reconnect recovery | 2026-03-31 | 2b25331 | [260331-myc-fix-device-disconnect-reconnect-backend-](./quick/260331-myc-fix-device-disconnect-reconnect-backend-/) |
| 260401-0fb | Beamforming map dB normalization + origin suppression | 2026-04-01 | caeaabd | [260401-0fb-beamforming-map-focus-on-drone-data-copy](./quick/260401-0fb-beamforming-map-focus-on-drone-data-copy/) |
| 260407-jx8 | Fix WindowedHFDroneDataset assertion on non-uniform DADS clips | 2026-04-07 | 7ea498a | [260407-jx8-fix-windowedhfdronedataset-assertion-on-](./quick/260407-jx8-fix-windowedhfdronedataset-assertion-on-/) |
| 260407-ls3 | Fix WindowedHFDroneDataset to return 32 kHz raw audio for EfficientAT v7 | 2026-04-07 | f007e91 | [260407-ls3-fix-windowedhfdronedataset-to-return-32-](./quick/260407-ls3-fix-windowedhfdronedataset-to-return-32-/) |
| 260407-nir | Fix WeightedRandomSampler to cover all windows in sliding-window path | 2026-04-07 | 9809d3f | [260407-nir-fix-weightedrandomsampler-to-use-dataset](./quick/260407-nir-fix-weightedrandomsampler-to-use-dataset/) |

## Session Continuity

Last session: 2026-04-08
Stopped at: Session resumed, routing to /gsd-execute-phase 21 (final plan 21-08)
Resume file: .planning/phases/21-build-raspberry-pi-4-edge-drone-detection-app-using-efficien/21-08-PLAN.md
Note: STATE.md header is stale — Phase 20.1 is fully complete (3/3 plans), Phase 21 is 7/8 complete with only 21-08 (systemd unit + install script + README) remaining.
