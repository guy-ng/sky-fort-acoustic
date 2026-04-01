---
phase: 08-pytorch-training-pipeline
plan: 01
subsystem: training-data-pipeline
tags: [pytorch, dataset, augmentation, mel-spectrogram, training]
dependency_graph:
  requires: [classification/config.py, classification/preprocessing.py]
  provides: [training/config.py, training/augmentation.py, training/dataset.py, mel_spectrogram_from_segment]
  affects: [classification/preprocessing.py]
tech_stack:
  added: [torchaudio.transforms.TimeMasking, torchaudio.transforms.FrequencyMasking, soundfile, WeightedRandomSampler]
  patterns: [TDD, module-level-cache, lazy-IO, random-segment-extraction]
key_files:
  created:
    - src/acoustic/training/__init__.py
    - src/acoustic/training/config.py
    - src/acoustic/training/augmentation.py
    - src/acoustic/training/dataset.py
    - tests/unit/test_augmentation.py
    - tests/unit/test_training_dataset.py
  modified:
    - src/acoustic/classification/preprocessing.py
decisions:
  - Module-level MelSpectrogram cache keyed by frozen MelConfig to avoid per-call reconstruction
  - mel_spectrogram_from_segment extracted as public utility for shared use between inference and training
metrics:
  duration: 5min
  completed: 2026-04-01
  tasks: 2
  files: 7
  tests_added: 20
---

# Phase 08 Plan 01: Training Data Pipeline Summary

TrainingConfig, augmentation classes, DroneAudioDataset with lazy WAV loading and random 0.5s segment extraction, plus a shared mel_spectrogram_from_segment utility extracted from ResearchPreprocessor.

## What Was Built

### Task 1: TrainingConfig and Augmentation Modules
- **TrainingConfig** (Pydantic BaseSettings): All hyperparameters loadable from `ACOUSTIC_TRAINING_*` env vars with research-validated defaults (lr=1e-3, batch_size=32, max_epochs=50, patience=5, augmentation_enabled=True)
- **WaveformAugmentation**: Gaussian noise injection at random SNR (10-40dB) + random gain (+/-6dB), applied to raw audio before mel conversion
- **SpecAugment**: Time masking (up to 20 frames) and frequency masking (up to 8 mel bins) via torchaudio, with proper shape transposition for (1, time, freq) layout

### Task 2: Shared Mel Utility and DroneAudioDataset
- **mel_spectrogram_from_segment()**: Public utility extracted from ResearchPreprocessor internals, produces identical output (verified by parity test). Uses module-level cache for MelSpectrogram transform keyed by frozen MelConfig
- **collect_wav_files()**: Scans data_root subdirectories matching label_map keys, returns parallel (paths, labels) lists
- **DroneAudioDataset**: PyTorch Dataset with lazy sf.read, random 0.5s segment extraction per epoch, optional waveform + spectrogram augmentation pipeline, zero-pads short audio
- **build_weighted_sampler()**: Creates WeightedRandomSampler with inverse-frequency class balancing

## Test Results

20 tests passing:
- 8 augmentation tests (config defaults, env overrides, noise injection, gain variation, masking shape/zeros/passthrough)
- 12 dataset tests (mel shape/range/parity, collect scanning/labeling/skipping, dataset len/shape/randomness/no-aug/short-pad, sampler)

Existing preprocessing tests (7) still pass — no regressions.

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 539f187 | TrainingConfig, WaveformAugmentation, SpecAugment + 8 tests |
| 2 | 91e8e26 | mel_spectrogram_from_segment, DroneAudioDataset, collect_wav_files, build_weighted_sampler + 12 tests |

## Known Stubs

None — all data paths are fully wired.
