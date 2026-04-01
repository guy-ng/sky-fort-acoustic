---
phase: 06-preprocessing-parity-foundation
plan: 01
subsystem: classification
tags: [melconfig, dataclass, protocol, onnx-removal, pytorch, torchaudio, fixtures]

# Dependency graph
requires: []
provides:
  - MelConfig frozen dataclass with research-validated preprocessing constants
  - Classifier and Preprocessor runtime-checkable protocols
  - Reference mel-spectrogram fixture for parity testing
  - ONNX dead code removed from src/
affects: [06-02, 07-cnn-architecture, 08-training-pipeline]

# Tech tracking
tech-stack:
  added: [torch, torchaudio]
  patterns: [frozen-dataclass-config, runtime-checkable-protocol]

key-files:
  created:
    - src/acoustic/classification/config.py
    - src/acoustic/classification/protocols.py
    - scripts/generate_reference_fixtures.py
    - tests/fixtures/reference_melspec_440hz.npy
    - tests/unit/test_mel_config.py
    - tests/unit/test_protocols.py
  modified:
    - src/acoustic/classification/__init__.py
    - src/acoustic/classification/worker.py
    - src/acoustic/config.py
    - src/acoustic/main.py
    - requirements.txt

key-decisions:
  - "Removed ONNX inference entirely (not wrapped) per D-02 -- dead code deletion"
  - "CNNWorker accepts optional preprocessor/classifier to stay dormant until Phase 7"
  - "librosa used only in scripts/ as dev dependency, not in runtime requirements.txt"

patterns-established:
  - "Frozen dataclass for immutable research constants (MelConfig)"
  - "Runtime-checkable Protocol for classifier/preprocessor contracts"

requirements-completed: [PRE-01, PRE-02]

# Metrics
duration: 5min
completed: 2026-04-01
---

# Phase 06 Plan 01: Preprocessing Foundation Summary

**MelConfig frozen dataclass with 7 research constants, Classifier/Preprocessor protocols, ONNX removal, and librosa-generated reference fixture for parity testing**

## Performance

- **Duration:** 5 min
- **Completed:** 2026-04-01
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Created MelConfig frozen dataclass matching all research constants (16kHz, 1024 FFT, 256 hop, 64 mels, 128 frames, 0.5s, 80dB)
- Created Classifier and Preprocessor runtime-checkable protocols for clean model swaps
- Removed all ONNX dead code (inference.py, test_inference.py, dummy_model.onnx)
- Generated reference mel-spectrogram fixture (128, 64) float32 from 440Hz sine wave via librosa
- Updated main.py to use dormant CNNWorker (preprocessor=None, classifier=None)

## Task Commits

1. **Task 1: MelConfig, protocols, ONNX removal** - `1efa038` (feat)
2. **Task 2: Reference fixture generation** - `1824907` (feat)

## Files Created/Modified
- `src/acoustic/classification/config.py` - MelConfig frozen dataclass with research constants
- `src/acoustic/classification/protocols.py` - Classifier and Preprocessor runtime-checkable protocols
- `src/acoustic/classification/__init__.py` - Updated exports to MelConfig, Classifier, Preprocessor
- `src/acoustic/classification/worker.py` - Optional preprocessor/classifier constructor args
- `src/acoustic/config.py` - Changed cnn_model_path default from .onnx to .pt
- `src/acoustic/main.py` - Removed OnnxDroneClassifier, dormant CNNWorker
- `requirements.txt` - Added torch>=2.11 and torchaudio>=2.11
- `scripts/generate_reference_fixtures.py` - One-time librosa reference tensor generator
- `tests/fixtures/reference_melspec_440hz.npy` - Reference fixture (128, 64) float32
- `tests/unit/test_mel_config.py` - MelConfig default values and frozen tests
- `tests/unit/test_protocols.py` - Protocol isinstance satisfaction tests

## Decisions Made
- Removed ONNX inference entirely rather than wrapping it (per D-02 -- dead code)
- CNNWorker constructor changed to accept optional preprocessor/classifier kwargs
- librosa kept as dev-only dependency in scripts/, not added to requirements.txt

## Deviations from Plan

None.

## Issues Encountered

None.

---
*Phase: 06-preprocessing-parity-foundation*
*Completed: 2026-04-01*
