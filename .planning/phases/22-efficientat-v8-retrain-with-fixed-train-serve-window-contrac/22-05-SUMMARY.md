---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 05
subsystem: data-investigation
tags: [kaggle, drone-audio, dataset-review, license-review, droneaudiodataset]

# Dependency graph
requires:
  - phase: 22
    plan: 01
    provides: test scaffolds and model provenance
provides:
  - Investigation script for probing DroneAudioDataset metadata
  - KAGGLE_DATASET_INVESTIGATION.md with technical findings and preliminary REJECT recommendation
affects: [22-06, 22-07, 22-08]

# Tech tracking
tech-stack:
  added: []
  patterns: [GitHub API metadata probing, sparse git clone for audio property sampling]

key-files:
  created:
    - scripts/investigate_kaggle_drone_dataset.py
    - .planning/phases/22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac/KAGGLE_DATASET_INVESTIGATION.md
  modified: []

# Decisions
key-decisions:
  - "Preliminary REJECT recommendation for DroneAudioDataset -- no license, indoor recordings, ESC-50 negative overlap"

# Metrics
metrics:
  duration: 5m28s
  completed: "2026-04-12T19:59:25Z"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 2
---

# Phase 22 Plan 05: Kaggle DroneAudioDataset Investigation Summary

**One-liner:** GitHub API + audio probe of saraalemadi/DroneAudioDataset reveals no license, indoor-only 16kHz mono clips, and ESC-50 negative overlap -- preliminary REJECT pending human decision checkpoint.

## What Was Done

### Task 1: Technical Investigation (COMPLETE)

Created `scripts/investigate_kaggle_drone_dataset.py` that probes the DroneAudioDataset via:
1. GitHub API for repo metadata (license, size, description)
2. Raw README fetch (first 200 lines)
3. LICENSE file fetch (returned 404 -- no license)
4. Git tree enumeration (23,408 audio files across 7 directories)
5. Sparse clone + soundfile.info probe on sample files

**Key findings:**

| Property | Value |
|---|---|
| License | **NONE** (no LICENSE file, API returns null) |
| Size | ~281 MB, 23,408 files |
| Format | WAV / PCM_16 |
| Sample rate | 16,000 Hz |
| Channels | 1 (mono) |
| Duration | ~0.9s - 1.02s per clip |
| Recording | **Indoor** environment |
| Drone models | DJI Bebop, DJI Membo (Mambo) |
| Negative class | ESC-50 subset + Speech Commands white noise + silence |

**Preliminary recommendation: REJECT** for three reasons:
1. No license (legal blocker)
2. Indoor recording environment (poor operational match for outdoor UMA-16 detector)
3. ESC-50 negative class overlap with existing noise corpora

Full report: `KAGGLE_DATASET_INVESTIGATION.md`

### Task 2: Human Decision Checkpoint (COMPLETE)

**Decision:** REJECT — decided 2026-04-13 by user.

Rationale: No license, indoor-only recordings are a poor match for outdoor UMA-16v2 detection. Phase 22 stays focused on the v7 window-contract fix and field data integration. No further action required.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Audio probe adapted for large file count**
- **Found during:** Task 1
- **Issue:** The script's MAX_FILES_FOR_CLONE threshold was 50; the repo has 23,408 files. Automatic clone was skipped.
- **Fix:** Performed manual sparse checkout of 7 specific files (2 unknown, 5 bebop) to get audio properties without downloading the full 281MB repo.
- **Files modified:** None (manual probe supplemented script output)

## Commits

| Task | Hash | Message |
|---|---|---|
| 1 | d932fe1 | feat(22-05): investigate Kaggle DroneAudioDataset for v8 training integration |

## Self-Check: PASS

All tasks complete. Investigation produced, checkpoint resolved (REJECT). No dataset ingested — no downstream impact on plans 22-06 through 22-09.
