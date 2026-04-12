---
phase: 22
plan: 04
subsystem: data-integrity
tags: [preflight, holdout, data-validation, sha256]
dependency_graph:
  requires: [22-02]
  provides: [preflight_v8_data, holdout-manifest, HOLDOUT_FILES]
  affects: [22-06, 22-07]
tech_stack:
  added: []
  patterns: [frozen-holdout-in-code, sha256-tamper-detection, fail-fast-preflight]
key_files:
  created:
    - scripts/preflight_v8_data.py
    - data/eval/uma16_real_v8/manifest.json
    - data/eval/uma16_real_v8/.gitkeep
  modified:
    - tests/integration/test_data_integrity_preflight.py
    - tests/integration/test_holdout_split.py
    - .gitignore
decisions:
  - "Holdout frozen as HOLDOUT_FILES frozenset in code, not config — reviewer sees any change"
  - "Manifest lives at data/eval/uma16_real_v8/manifest.json with source_path references, no WAV duplication"
  - "Added .gitignore negation pattern for data/eval/uma16_real_v8/ to track manifest while keeping data/ ignored"
metrics:
  duration_seconds: 623
  completed: "2026-04-12T22:57:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 3
---

# Phase 22 Plan 04: Data Integrity Preflight and Holdout Manifest Summary

Data integrity preflight script with frozen 5-file holdout split, sha256 manifest, and fail-fast validation for SR/channels/NaN/cardinality before any Vertex training job runs.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Preflight script with frozen HOLDOUT_FILES + CLI | 909862e | scripts/preflight_v8_data.py, tests/integration/test_data_integrity_preflight.py, tests/integration/test_holdout_split.py |
| 2 | Holdout manifest with sha256 hashes | 698dcb3 | data/eval/uma16_real_v8/manifest.json, data/eval/uma16_real_v8/.gitkeep, .gitignore |

## What Was Built

### scripts/preflight_v8_data.py

- **HOLDOUT_FILES** frozenset: 5 files (4 drone sub-classes + 1 background), frozen in code
- **preflight_field_recordings()**: validates all 20260408_*.wav files — checks SR=16000, mono, no NaN/Inf, non-empty; excludes holdout; asserts cardinality 9 drone + 3 background
- **preflight_holdout()**: validates holdout files separately with trimmed-file sentinel check (61.4s range)
- **CLI**: `python scripts/preflight_v8_data.py` exits 0 on success, 1 on failure
- Exports: `HOLDOUT_FILES`, `preflight_field_recordings`, `preflight_holdout`, `main`

### data/eval/uma16_real_v8/manifest.json

- 5 holdout files with real sha256 hashes for tamper detection
- Promotion gate thresholds: real_TPR >= 0.80, real_FPR <= 0.05
- source_path references to data/field/ (no WAV duplication)
- Isolation invariant documented: enforced by HOLDOUT_FILES in code

### Test Updates

- Removed `pytest.mark.xfail` from test_data_integrity_preflight.py (3 tests now pass)
- Removed try/except xfail pattern from test_holdout_split.py (2 tests now pass directly)
- All 5 integration tests green

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .gitignore blocking eval manifest from git tracking**
- **Found during:** Task 2
- **Issue:** `data/` in .gitignore blocked `data/eval/uma16_real_v8/manifest.json` from being tracked
- **Fix:** Added negation patterns (`data/*` + `!data/eval/` cascade) to allow tracking manifest while keeping all other data/ contents ignored
- **Files modified:** .gitignore
- **Commit:** 698dcb3

## Verification Results

- `python scripts/preflight_v8_data.py` exits 0: "OK -- training: 9 drone + 3 bg | holdout: 4 drone + 1 bg"
- 5 integration tests pass (test_data_integrity_preflight: 3, test_holdout_split: 2)
- Manifest JSON valid, 5 entries, all sha256 are 64-char hex digests
- Preflight runs in <1s on local dev (well under 5s requirement)

## Self-Check: PASSED

All files exist. All commits found.
