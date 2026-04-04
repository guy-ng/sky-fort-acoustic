---
phase: 15-advanced-training-enhancements-focal-loss-noise-augmentation-balanced-sampling
plan: 01
subsystem: training
tags: [focal-loss, audiomentations, noise-augmentation, data-augmentation, pytorch]

requires:
  - phase: 14-efficientat-model-architecture-with-audioset-transfer-learning
    provides: TrainingConfig with EfficientAT fields, training infrastructure

provides:
  - FocalLoss nn.Module wrapping torchvision sigmoid_focal_loss
  - build_loss_function factory (focal/bce/bce_weighted selection)
  - BackgroundNoiseMixer for SNR-controlled noise mixing
  - AudiomentationsAugmentation for PitchShift+TimeStretch+Gain
  - ComposedAugmentation for picklable augmentation chaining
  - Extended TrainingConfig with loss, noise, and audiomentations fields

affects: [15-02, training-loop-integration]

tech-stack:
  added: [audiomentations>=0.43, torchvision.ops.sigmoid_focal_loss]
  patterns: [factory-pattern-for-loss-selection, lazy-cache-for-noise-files, picklable-augmentation-composition]

key-files:
  created:
    - src/acoustic/training/losses.py
    - tests/unit/test_focal_loss.py
    - tests/unit/test_noise_augmentation.py
    - tests/unit/test_audiomentations_aug.py
  modified:
    - src/acoustic/training/augmentation.py
    - src/acoustic/training/config.py
    - requirements.txt

key-decisions:
  - "FocalLoss wraps torchvision sigmoid_focal_loss rather than custom implementation"
  - "BackgroundNoiseMixer uses lazy warm_cache pattern for deferred audio loading"
  - "AudiomentationsAugmentation replaces WaveformAugmentation as primary waveform aug"
  - "ComposedAugmentation is picklable for DataLoader num_workers > 0"
  - "noise_augmentation_enabled defaults to False (requires noise dataset download)"

patterns-established:
  - "Factory pattern: build_loss_function selects loss by config string"
  - "Lazy cache: BackgroundNoiseMixer defers audio loading until warm_cache()"
  - "Composition: ComposedAugmentation chains augmentations, picklable for multiprocessing"

requirements-completed: [TRN-10, TRN-11, TRN-12]

duration: 7min
completed: 2026-04-04
---

# Phase 15 Plan 01: Focal Loss, Noise Augmentation, and Audiomentations Summary

**Focal loss module, background noise mixer, audiomentations-based waveform augmentation, and ComposedAugmentation utility with 32 passing unit tests**

## What Was Built

### Task 1: FocalLoss Module and Config Extensions
- Created `src/acoustic/training/losses.py` with `FocalLoss` nn.Module wrapping `torchvision.ops.sigmoid_focal_loss` (alpha/gamma configurable)
- Added `build_loss_function` factory: selects focal/bce/bce_weighted by config string
- Extended `TrainingConfig` with 15 new fields for loss function, noise augmentation, and audiomentations parameters
- Added `audiomentations>=0.43,<1.0` to requirements.txt
- 11 unit tests covering loss behavior, BCE equivalence at gamma=0, gradient flow, factory, and config defaults

### Task 2: BackgroundNoiseMixer, AudiomentationsAugmentation, ComposedAugmentation
- Added `BackgroundNoiseMixer` to augmentation.py: loads WAV files, resamples via torchaudio, caches in memory, mixes at random SNR with probability p, clips output to [-1, 1]
- Added `AudiomentationsAugmentation`: wraps audiomentations Compose pipeline with PitchShift + TimeStretch + Gain
- Added `ComposedAugmentation`: chains multiple augmentations sequentially, picklable for DataLoader num_workers > 0
- Preserved existing `WaveformAugmentation` and `SpecAugment` for backward compatibility
- 13 unit tests covering noise mixing, resampling, probability control, audiomentations output, pickling, and composition chaining

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 2525c90 | FocalLoss module, config extensions, and unit tests |
| 2 | 1b02b4e | BackgroundNoiseMixer, AudiomentationsAugmentation, ComposedAugmentation |

## Verification

All 32 tests pass across 4 test files:
- `test_focal_loss.py`: 11 passed
- `test_noise_augmentation.py`: 6 passed
- `test_audiomentations_aug.py`: 7 passed
- `test_augmentation.py`: 8 passed (existing, unchanged)

Import verification: `FocalLoss`, `build_loss_function`, `BackgroundNoiseMixer`, `AudiomentationsAugmentation`, `ComposedAugmentation` all importable.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all modules are fully implemented with real logic.

## Self-Check: PASSED

All 7 files found. Both commits (2525c90, 1b02b4e) verified.
