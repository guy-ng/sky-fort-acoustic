---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 06
subsystem: efficientat-v8-training-pipeline
tags: [efficientat, v8, concat-dataset, fine-tune, field-recordings, vertex-ai, preflight]

# Dependency graph
requires:
  - phase: 22
    plan: 03
    provides: WindowedHFDroneDataset with per_file_lengths and post_resample_norm
  - phase: 22
    plan: 04
    provides: preflight_v8_data.py with HOLDOUT_FILES and preflight_field_recordings
provides:
  - src/acoustic/training/efficientat_trainer.py v8 training path (preflight + ConcatDataset + fine-tune-from-v6)
  - scripts/vertex_submit.py additive v8 submission (submit_v8_job + build_env_vars_v8 + GCP_REGION_V8)
  - Dockerfile.vertex-base with field recordings + v6 checkpoint baked in
affects: [22-07, 22-08, 22-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ConcatDataset for heterogeneous data sources
    - _FieldRecordingDataset HF-compatible wrapper for in-memory audio
    - additive versioned submission paths (v7 untouched)
    - per_file_lengths sliding window for multi-second recordings

key-files:
  created: []
  modified:
    - src/acoustic/training/efficientat_trainer.py
    - src/acoustic/training/config.py
    - scripts/vertex_submit.py
    - Dockerfile.vertex-base
    - tests/integration/test_dataset_cardinality.py

decisions:
  - "_FieldRecordingDataset wrapper encodes audio as WAV bytes on-the-fly to match WindowedHFDroneDataset's decode_wav_bytes path -- avoids forking the dataset class"
  - "build_env_vars_v8 starts from build_env_vars_v7 baseline then overrides v8-specific keys -- ensures all Phase 20 augmentation config carries forward"
  - "Docker image :v2 build deferred to main repo -- worktree lacks noise corpora and model files"

metrics:
  duration: 10m
  tasks_completed: 3
  tasks_total: 3
  files_modified: 5
  completed: "2026-04-12T21:21:36Z"
---

# Phase 22 Plan 06: v8 Training Pipeline Wiring Summary

**ConcatDataset(DADS + field) with fine-tune-from-v6, data preflight call, additive Vertex v8 submission path, and Dockerfile :v2 with field recordings baked in**

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Wire ConcatDataset + fine-tune-from-v6 + preflight call | 3b86812 | efficientat_trainer.py, config.py, test_dataset_cardinality.py |
| 2 | Add submit_v8_job + build_env_vars_v8 + GCP_REGION_V8 | 54e8c15 | scripts/vertex_submit.py |
| 3 | Dockerfile.vertex-base :v2 with field recordings | a0a7b51 | Dockerfile.vertex-base |

## What Was Built

### Task 1 -- Trainer v8 Wiring (efficientat_trainer.py + config.py)

**New TrainingConfig fields** (all env-var-overridable via ACOUSTIC_TRAINING_* prefix):
- `finetune_from_trained: bool = False` -- load binary-head checkpoint instead of AudioSet pretrained
- `include_field_recordings: bool = False` -- concat 2026-04-08 field data with DADS
- `run_data_preflight: bool = False` -- call preflight_field_recordings before DataLoader construction
- `field_drone_dir: str` / `field_background_dir: str` -- paths to field WAVs
- `output_path: str` -- override checkpoint_path for v8 saves

**Preflight hook** -- at top of `run()`, before any DataLoader construction:
```python
if cfg.run_data_preflight:
    from scripts.preflight_v8_data import preflight_field_recordings
    preflight_manifest = preflight_field_recordings(...)
```

**_build_field_dataset method** -- constructs a `WindowedHFDroneDataset` for multi-second field recordings using `per_file_lengths` (Plan 22-03 feature). Each file gets its correct sliding-window count. 50% overlap at 1.0s windows = `hop_samples=8000`. Uses `_FieldRecordingDataset` wrapper to present list-of-dicts as HF-compatible column/row access.

**ConcatDataset wiring** -- in the `use_phase20_path` block:
```python
if cfg.include_field_recordings:
    field_train_ds = self._build_field_dataset(cfg, post_norm)
    train_ds = ConcatDataset([dads_train_ds, field_train_ds])
    train_ds.labels = list(dads._labels_cache) + list(field._labels_cache)
```

**Fine-tune-from-v6** -- after binary head replacement, loads trained checkpoint with `strict=False`, asserts classifier weights are present.

### Task 2 -- Vertex v8 Submission Path (scripts/vertex_submit.py)

All additive -- v7 functions untouched:

- `GCP_REGION_V8 = "us-east1"` (user-locked)
- `build_env_vars_v8(output_dir)` -- starts from v7 baseline, overrides with v8 flags
- `submit_v8_job(image_uri)` -- L4 quota check, us-east1, g2-standard-8, sync=False
- `--version v8` CLI route with default image URI pointing to `:v2` tag
- `GCR_BASE_IMAGE_V2` constant for the Phase 22 Docker tag

### Task 3 -- Dockerfile.vertex-base :v2

Added COPY steps after existing noise data:
```dockerfile
COPY data/field/drone/20260408_*.wav /app/data/field/drone/
COPY data/field/background/20260408_*.wav /app/data/field/background/
COPY models/efficientat_mn10_v6.pt /app/models/efficientat_mn10_v6.pt
```

**Image build/push deferred** -- the worktree lacks the noise corpora (3-4 GB) and model files required for a complete Docker build. The Dockerfile changes are committed and correct. User must build and push from the main repo:
```bash
docker build -f Dockerfile.vertex-base \
  -t us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v2 .
docker push us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v2
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] _FieldRecordingDataset wrapper for HF-compatible access**
- **Found during:** Task 1
- **Issue:** WindowedHFDroneDataset does `hf_dataset["label"]` for column access, which fails on a plain list of dicts.
- **Fix:** Created `_FieldRecordingDataset` class that wraps list-of-dicts with both column access (`dataset["label"]`) and row access (`dataset[idx]`). Row access encodes audio as WAV bytes on-the-fly so `decode_wav_bytes` in `__getitem__` works transparently.
- **Files modified:** src/acoustic/training/efficientat_trainer.py
- **Commit:** 3b86812

## User Action Required

- **Build and push Dockerfile.vertex-base:v2** from the main repo (requires noise corpora + model files on disk)
- Do NOT submit the Vertex job -- that is Plan 08

## Self-Check: PASSED

All files exist. All commits found.
