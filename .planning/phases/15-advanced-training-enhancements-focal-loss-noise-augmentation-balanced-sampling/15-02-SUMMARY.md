---
phase: 15-advanced-training-enhancements-focal-loss-noise-augmentation-balanced-sampling
plan: 02
subsystem: training
tags: [focal-loss, augmentation, balanced-sampling, training-loop]
dependency_graph:
  requires: [15-01]
  provides: [training-runner-focal-loss, training-runner-audiomentations, training-runner-noise-mixer]
  affects: [src/acoustic/training/trainer.py, src/acoustic/classification/research_cnn.py]
tech_stack:
  added: []
  patterns: [logits-mode-training, composed-augmentation-pipeline, picklable-augmentation]
key_files:
  created:
    - tests/unit/test_training_enhancements_integration.py
  modified:
    - src/acoustic/classification/research_cnn.py
    - src/acoustic/training/dataset.py
    - src/acoustic/training/parquet_dataset.py
    - src/acoustic/training/trainer.py
    - tests/unit/test_research_cnn.py
decisions:
  - "All loss functions use logits mode (FocalLoss and BCEWithLogitsLoss both expect logits)"
  - "TorchScript export uses logits_mode=False to include sigmoid for inference compatibility"
  - "ComposedAugmentation chosen over closure for DataLoader num_workers picklability"
metrics:
  duration: 7m17s
  completed: 2026-04-04
  tasks: 2
  files: 6
---

# Phase 15 Plan 02: Training Loop Integration Summary

Wire focal loss, audiomentations, and background noise mixer into TrainingRunner with logits-mode ResearchCNN and picklable composed augmentation pipeline.

## Tasks Completed

### Task 1: Update ResearchCNN for logits mode and widen dataset augmentation types
- **Commit:** 1fa463f
- Added `logits_mode: bool = False` parameter to ResearchCNN
- Sigmoid moved from inside `self.classifier` Sequential to separate `self._sigmoid` applied conditionally in forward()
- State dict keys unchanged (backward compatible with existing checkpoints)
- DroneAudioDataset and ParquetDataset now accept `Callable[[np.ndarray], np.ndarray]` instead of `WaveformAugmentation`
- Removed unused `WaveformAugmentation` import from dataset modules (kept `SpecAugment`)

### Task 2: Wire focal loss, augmentations, and balanced sampling into TrainingRunner (TDD)
- **Commits:** 9ada6b7 (tests), e02ca29 (implementation)
- Replaced `nn.BCELoss()` with `build_loss_function()` factory (focal loss default)
- Model instantiated with `ResearchCNN(logits_mode=True)` during training
- Validation loop applies `torch.sigmoid(output) >= 0.5` for correct thresholding
- Audiomentations augmentation used when `cfg.use_audiomentations=True`, legacy WaveformAugmentation otherwise
- BackgroundNoiseMixer created and warm_cache() called when `cfg.noise_augmentation_enabled=True`
- Augmentation pipeline composed via picklable `ComposedAugmentation` class
- Class distribution logged before weighted sampler construction
- TorchScript export uses `ResearchCNN(logits_mode=False)` for inference-ready model with sigmoid
- 10 integration tests covering all wiring points

## Decisions Made

1. **All training uses logits mode** -- both FocalLoss and BCEWithLogitsLoss expect raw logits, so ResearchCNN always runs with `logits_mode=True` during training. This simplifies the criterion wiring.
2. **TorchScript export with sigmoid** -- exported model uses `logits_mode=False` so downstream inference consumers get probability outputs without needing to apply sigmoid themselves.
3. **ComposedAugmentation over closure** -- picklable class required for DataLoader `num_workers > 0`. Lambda/closure composition would break multiprocess data loading.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing test_research_cnn.py architecture test**
- **Found during:** Task 2 verification
- **Issue:** `test_architecture_matches_spec` expected sigmoid inside `model.classifier` Sequential, but sigmoid was moved to `model._sigmoid`
- **Fix:** Updated assertion to check `isinstance(model._sigmoid, nn.Sigmoid)` instead of scanning classifier Sequential
- **Files modified:** tests/unit/test_research_cnn.py
- **Commit:** e02ca29

## Test Results

- 10 new integration tests: ALL PASSING
- 37 existing tests (research_cnn, dataset, parquet_dataset): ALL PASSING
- 9 training checkpoint tests: ALL PASSING
- Total: 47+ tests verified

## Known Stubs

None -- all wiring is functional with no placeholder data.

## Self-Check: PASSED

- All 6 files verified present
- All 3 commits verified (1fa463f, 9ada6b7, e02ca29)
