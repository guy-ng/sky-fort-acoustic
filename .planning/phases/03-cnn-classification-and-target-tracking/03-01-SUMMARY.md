---
phase: 03-cnn-classification-and-target-tracking
plan: 01
subsystem: classification
tags: [pytorch, onnx, mel-spectrogram, cnn, librosa, scipy, hysteresis]

# Dependency graph
requires:
  - phase: 01-audio-capture-beamforming-infrastructure
    provides: "AcousticSettings config, audio capture pipeline, types"
provides:
  - "Mel-spectrogram preprocessing pipeline (preprocess_for_cnn)"
  - "ONNX Runtime drone classifier (OnnxDroneClassifier)"
  - "Hysteresis detection state machine (DetectionStateMachine)"
  - "CNN config fields (model path, thresholds, TTL)"
  - "Dummy ONNX model test fixture"
affects: [03-02-target-tracker, 03-03-zmq-publishing]

# Tech tracking
tech-stack:
  added: [librosa, onnxruntime, onnx]
  patterns: [mel-spectrogram-preprocessing, onnx-inference, hysteresis-state-machine]

key-files:
  created:
    - src/acoustic/classification/__init__.py
    - src/acoustic/classification/preprocessing.py
    - src/acoustic/classification/inference.py
    - src/acoustic/classification/state_machine.py
    - tests/fixtures/dummy_model.onnx
    - tests/unit/test_preprocessing.py
    - tests/unit/test_inference.py
    - tests/unit/test_state_machine.py
  modified:
    - src/acoustic/config.py

key-decisions:
  - "Used librosa for mel-spectrogram to match POC parameters exactly (16kHz, 64 mels, n_fft=1024, hop=256)"
  - "ONNX Runtime over PyTorch for inference — lighter runtime, model-agnostic"
  - "3-state hysteresis (NO_DRONE/CANDIDATE/CONFIRMED) with configurable thresholds (0.80 enter, 0.40 exit)"
  - "Binary drone/not-drone only — CLS-02 multi-class deferred to milestone 2 per D-01"

patterns-established:
  - "POC constant porting: SR_CNN=16000, N_FFT=1024, HOP_LENGTH=256, N_MELS=64, MAX_FRAMES=128"
  - "NHWC tensor format (1, 128, 64, 1) for ONNX model input"
  - "State machine pattern with enter/exit thresholds and confirm_hits for flicker prevention"

requirements-completed: [CLS-01, CLS-02, CLS-03, CLS-04]

# Metrics
duration: 6min
completed: 2026-03-31
---

# Phase 3 Plan 01: CNN Classification Package Summary

**ONNX-based binary drone classifier with mel-spectrogram preprocessing matching POC parameters and 3-state hysteresis detection state machine**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-31T16:25:59Z
- **Completed:** 2026-03-31T16:31:34Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 9

## Accomplishments
- Mel-spectrogram preprocessing pipeline porting exact POC parameters (16kHz resample, 64 mels, 1024 FFT, 256 hop, 128 frames)
- ONNX Runtime classifier with configurable model path and CPU-only inference
- 3-state hysteresis state machine preventing detection flickering with enter/exit thresholds
- Config extended with CNN fields (model path, thresholds, confirm hits, target TTL)
- Full TDD with 20 passing unit tests

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `5412a1b` (test)
2. **Task 1 GREEN: Classification implementation** - `2dc6584` (feat)

## Files Created/Modified
- `src/acoustic/classification/__init__.py` - Package marker
- `src/acoustic/classification/preprocessing.py` - Mel-spectrogram pipeline (fast_resample, make_melspec, pad_or_trim, norm_spec, preprocess_for_cnn)
- `src/acoustic/classification/inference.py` - OnnxDroneClassifier with ONNX Runtime
- `src/acoustic/classification/state_machine.py` - DetectionStateMachine with hysteresis
- `src/acoustic/config.py` - Added cnn_model_path, cnn_enter_threshold, cnn_exit_threshold, cnn_confirm_hits, cnn_target_ttl
- `tests/fixtures/dummy_model.onnx` - Minimal ONNX model for testing (ReduceMean + Sigmoid)
- `tests/unit/test_preprocessing.py` - 7 tests for preprocessing functions
- `tests/unit/test_inference.py` - 4 tests for ONNX classifier
- `tests/unit/test_state_machine.py` - 9 tests for hysteresis state machine

## Decisions Made
- Used librosa for mel-spectrogram computation to match POC parameters exactly
- ONNX Runtime for inference (lighter than full PyTorch, model-agnostic)
- 3-state hysteresis with 0.80 enter / 0.40 exit thresholds and 2 confirm hits
- Binary classification only (drone/not-drone) — multi-class deferred per D-01

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functions are fully implemented with real logic.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Classification package ready for Plan 03-02 (target tracker) to wire into live pipeline
- OnnxDroneClassifier accepts preprocessed audio and returns drone probability
- DetectionStateMachine manages detection lifecycle
- Config fields available for environment variable override

## Self-Check: PASSED

- All 9 files exist
- Both commits found (5412a1b, 2dc6584)
- All acceptance criteria verified (grep counts all >= 1)
- 20/20 unit tests passing
- Zero tensorflow references in src/acoustic/

---
*Phase: 03-cnn-classification-and-target-tracking*
*Completed: 2026-03-31*
