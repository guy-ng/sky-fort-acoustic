---
phase: 13-dads-dataset-integration-and-training-data-pipeline
plan: 02
subsystem: training-data
tags: [parquet, training-pipeline, recording, integration]
dependency_graph:
  requires: [parquet-dataset, split-indices, training-runner, recording-session]
  provides: [parquet-training-integration, parquet-recording-output]
  affects: [training-pipeline, recording-pipeline]
tech_stack:
  added: []
  patterns: [branch-on-data-source, post-processing-parquet]
key_files:
  created:
    - tests/unit/test_parquet_training_integration.py
  modified:
    - src/acoustic/training/config.py
    - src/acoustic/training/trainer.py
    - src/acoustic/recording/recorder.py
    - src/acoustic/recording/manager.py
decisions:
  - "TrainingRunner branches on presence of train-*.parquet files in dads_path, not on config flag"
  - "Parquet created as post-processing after WAV write, keeping streaming WAV writer unchanged"
  - "drone label maps to 1, all others to 0 in field recording Parquet output"
metrics:
  duration: "4m01s"
  completed: "2026-04-03"
  tasks_completed: 2
  tasks_total: 2
  test_count: 8
  files_created: 1
  files_modified: 4
---

# Phase 13 Plan 02: Wire ParquetDataset into Training and Recording Pipelines Summary

TrainingRunner auto-selects ParquetDataset when DADS shards exist in configurable dads_path (default "data/", env ACOUSTIC_TRAINING_DADS_PATH), falls back to WAV-based DroneAudioDataset otherwise; field recordings now emit DADS-compatible Parquet alongside WAV on labeling.

## Task Results

### Task 1: Extend TrainingConfig and wire ParquetDataset into TrainingRunner

**Status:** Complete
**Commit:** 58f0060
**Files modified:** `src/acoustic/training/config.py`, `src/acoustic/training/trainer.py`

Implemented:
- Added `dads_path: str = "data/"` to TrainingConfig (auto-configurable via ACOUSTIC_TRAINING_DADS_PATH env var)
- TrainingRunner.run() checks for train-*.parquet files in dads_path directory
- When shards found: uses ParquetDatasetBuilder + split_indices(seed=42) for 70/15/15 split
- When no shards: falls back to existing collect_wav_files + DroneAudioDataset path (unchanged)
- No changes needed in manager.py (dads_path flows through automatically via Pydantic BaseSettings)

### Task 2: Update field recording pipeline to save Parquet and add integration tests

**Status:** Complete
**Commit:** f602c8d
**Files modified:** `src/acoustic/recording/recorder.py`, `src/acoustic/recording/manager.py`
**Files created:** `tests/unit/test_parquet_training_integration.py`

Implemented:
- RecordingSession.to_parquet(label) reads completed WAV, wraps in DADS schema {audio: {bytes, path}, label}
- RecordingSession.path property for WAV path access
- RecordingManager.label_recording() creates .parquet alongside .wav after move
- Label mapping: "drone" -> 1, everything else -> 0

**Tests (8/8 passing):**
- `TestTrainingConfigDadsPath`: default value, env var override
- `TestTrainerParquetBranch`: Parquet path with synthetic shards, WAV fallback on missing dir
- `TestRecordingSessionToParquet`: WAV-to-Parquet conversion, path property
- `TestLabelRecordingCreatesParquet`: label_recording creates Parquet, background label maps to 0

## Deviations from Plan

None - plan executed exactly as written.

## Verification

```
PYTHONPATH=src python -m pytest tests/unit/test_parquet_dataset.py tests/unit/test_parquet_training_integration.py -x -v  # 25 passed in 1.96s
PYTHONPATH=src python -c "from acoustic.training.config import TrainingConfig; c = TrainingConfig(); print(f'dads_path={c.dads_path}')"  # dads_path=data/
PYTHONPATH=src python -c "from acoustic.training.trainer import TrainingRunner; from acoustic.training.parquet_dataset import ParquetDataset; print('all imports OK')"
PYTHONPATH=src python -c "from acoustic.recording.recorder import RecordingSession; print(hasattr(RecordingSession, 'to_parquet'))"  # True
```

## Known Stubs

None - all data paths are wired to real Parquet I/O and training pipeline integration.

## Self-Check: PASSED

- [x] src/acoustic/training/config.py exists (modified)
- [x] src/acoustic/training/trainer.py exists (modified)
- [x] src/acoustic/recording/recorder.py exists (modified)
- [x] src/acoustic/recording/manager.py exists (modified)
- [x] tests/unit/test_parquet_training_integration.py exists (created)
- [x] 13-02-SUMMARY.md exists
- [x] Commit 58f0060 found (Task 1: config + trainer)
- [x] Commit f602c8d found (Task 2: recording + tests)
