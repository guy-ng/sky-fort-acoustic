---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 08
subsystem: training-infrastructure
tags: [vertex-ai, gcp, l4, training, docker, efficientat]

requires:
  - phase: 22
    plan: 06
    provides: training pipeline wiring
  - phase: 22
    plan: 07
    provides: eval harness and promotion gate
provides:
  - models/efficientat_mn10_v8.pt checkpoint (16.2 MB)
  - sha256 sidecar
  - TRAINING_RUN.md with full job log
affects: [22-09]

tech-stack:
  added: []
  patterns: [CustomJob API for Vertex submission, cross-region Artifact Registry pull]

key-files:
  created:
    - models/efficientat_mn10_v8.pt
    - models/efficientat_mn10_v8.sha256
  modified:
    - Dockerfile.vertex-base (g++ for pyroomacoustics)
    - Dockerfile.vertex (preflight script inclusion)
    - scripts/vertex_submit.py (staging_bucket, east1 GCS bucket)
    - scripts/vertex_train.py (finetune_from_trained skip GCS download)
    - models/MODEL_PROVENANCE.md (v8 row)

key-decisions:
  - "Used CustomJob API directly — CustomContainerTrainingJob.run(sync=False) silently fails"
  - "GCS output bucket must be co-located with Vertex region (sky-fort-acoustic-east1)"
  - "vertex_train.py must not overwrite PRETRAINED_WEIGHTS when fine-tuning from v6"

metrics:
  duration: ~4 hours (including 3 failed attempts)
  completed: "2026-04-13T08:04:30Z"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 8
  vertex_job_id: "projects/859551133057/locations/us-east1/customJobs/56444872520892416"
  training_duration_min: 212.1
---

# Phase 22 Plan 08: Vertex Training Run Summary

**One-liner:** v8 trained successfully on L4 us-east1 in 212 min (fine-tuned from v6), val_acc=0.994 / F1=0.996. Three infrastructure issues fixed en route: missing g++, wrong GCS bucket region, pretrained weights path clobbering.

## What Was Done

### Task 1: Preflight (COMPLETE)
- Local data preflight OK (9 drone, 3 bg, 5 holdout excluded)
- L4 quota confirmed in us-east1
- Docker base image v2 built and pushed (required g++ fix for pyroomacoustics)
- v6 sha256 recorded

### Task 2: Submit + Monitor (COMPLETE)
Three failed attempts before success:
1. **Attempt 1:** Used base image (no entrypoint) — container exited immediately
2. **Attempt 2:** `ModuleNotFoundError: No module named 'scripts'` — preflight_v8_data.py not in image
3. **Attempt 3:** `RuntimeError: size mismatch for classifier.5.weight` — vertex_train.py clobbered PRETRAINED_WEIGHTS with mn10_as.pt (527-class AudioSet) instead of v6 (binary)
4. **Attempt 4:** Success — 212 min, val_acc=0.994

### Task 3: Download + Verify (COMPLETE)
- Checkpoint downloaded from GCS (16.2 MB)
- sha256: 02839a1d102fe7ca3116739160d7d9c97e9a025d73dbe7d6cb9afd147a877071
- MODEL_PROVENANCE.md updated

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | 7dfcd61 | fix(22-08): add g++ to Dockerfile, fix staging_bucket + GCS region |
| 2 | 7de0d9f | fix(22-08): add preflight_v8_data.py to Dockerfile.vertex |
| 2 | 264f1cd | fix(22-08): don't overwrite pretrained_weights when fine-tuning from v6 |
| 3 | a813e43 | feat(22-08): download v8 checkpoint from Vertex, record training results |

## Self-Check: PASS

Checkpoint exists, sha256 sidecar committed, TRAINING_RUN.md populated, MODEL_PROVENANCE.md updated. No window contract warnings in training logs.
