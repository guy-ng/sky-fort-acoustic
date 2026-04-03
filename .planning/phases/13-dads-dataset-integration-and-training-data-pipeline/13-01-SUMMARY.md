---
phase: 13-dads-dataset-integration-and-training-data-pipeline
plan: 01
subsystem: training-data
tags: [parquet, dataset, dads, mel-spectrogram, pytorch]
dependency_graph:
  requires: [classification-config, preprocessing, augmentation]
  provides: [parquet-dataset, split-indices]
  affects: [training-pipeline]
tech_stack:
  added: [pyarrow]
  patterns: [builder-factory, bisect-shard-lookup, tdd]
key_files:
  created:
    - src/acoustic/training/parquet_dataset.py
    - tests/unit/test_parquet_dataset.py
  modified: []
decisions:
  - "Builder pattern to avoid 3x shard scanning for train/val/test splits"
  - "44-byte WAV header skip for in-memory decoding (no soundfile dependency for Parquet path)"
  - "bisect.bisect_right for O(log n) shard lookup instead of linear scan"
metrics:
  duration: "3m40s"
  completed: "2026-04-03"
  tasks_completed: 1
  tasks_total: 1
  test_count: 17
  files_created: 2
  files_modified: 0
---

# Phase 13 Plan 01: ParquetDataset for DADS Integration Summary

ParquetDataset loads 180K DADS audio samples from Parquet shards, decodes WAV bytes in-memory, extracts random 0.5s segments, and returns (1,128,64) mel-spectrogram tensors with float32 labels via bisect-based shard lookup and deterministic 70/15/15 splitting.

## Task Results

### Task 1: Create ParquetDataset class with shard scanning, WAV decoding, and split logic (TDD)

**Status:** Complete
**Commits:** 9e8d3ad (RED: failing tests), ae22f72 (GREEN: implementation)
**Files created:** `src/acoustic/training/parquet_dataset.py`, `tests/unit/test_parquet_dataset.py`

Implemented:
- `split_indices()`: Deterministic shuffled 70/15/15 splitting with configurable seed
- `decode_wav_bytes()`: Skip 44-byte WAV header, int16-to-float32 normalization
- `ParquetDataset`: Full `torch.utils.data.Dataset` implementation with lazy Parquet I/O
- `ParquetDatasetBuilder`: Factory that scans shards once and builds datasets for each split
- `_locate()`: `bisect.bisect_right`-based O(log n) global-to-shard index mapping
- Short audio zero-padding, optional waveform/spectrogram augmentation hooks

**Tests (17/17 passing):**
- `TestSplitIndices`: sizes, proportions, no-overlap, determinism, seed variation, small edge case
- `TestDecodeWavBytes`: float32 dtype, value range [-1,1], correct length
- `TestParquetDataset`: total_rows, len, getitem shape/dtype, labels property, _locate at boundaries, short audio padding

## Deviations from Plan

None - plan executed exactly as written.

## Verification

```
python -m pytest tests/unit/test_parquet_dataset.py -x -v  # 17 passed in 0.79s
python -c "from acoustic.training.parquet_dataset import ParquetDataset, split_indices; print('imports OK')"
```

## Known Stubs

None - all data paths are wired to real Parquet I/O via pyarrow.

## Self-Check: PASSED

- [x] src/acoustic/training/parquet_dataset.py exists
- [x] tests/unit/test_parquet_dataset.py exists
- [x] 13-01-SUMMARY.md exists
- [x] Commit 9e8d3ad found (RED: failing tests)
- [x] Commit ae22f72 found (GREEN: implementation)
