---
phase: 17-beamforming-engine-upgrade-and-pipeline-integration
plan: 02
subsystem: beamforming
tags: [mcra, noise-estimation, multi-peak, signal-detection]
dependency_graph:
  requires: []
  provides: [MCRANoiseEstimator, detect_multi_peak, bf-config-fields]
  affects: [pipeline-integration-17-03]
tech_stack:
  added: []
  patterns: [adaptive-noise-estimation, greedy-peak-selection, global-median-signal-presence]
key_files:
  created:
    - src/acoustic/beamforming/mcra.py
    - src/acoustic/beamforming/multi_peak.py
    - tests/unit/test_mcra.py
    - tests/unit/test_multi_peak.py
  modified:
    - src/acoustic/config.py
decisions:
  - Used global median of smoothed power map as secondary signal presence indicator in MCRA, supplementing classic S/S_min ratio. This handles persistent signals present from initialization that classic MCRA cannot distinguish.
  - When signal is detected, noise adapts toward global median rather than raw power, preventing noise floor from rising to signal level.
  - Multi-peak uses Euclidean angular distance in (az, el) space for separation constraint rather than great-circle distance — sufficient for the narrow scan range.
metrics:
  duration_seconds: 1223
  completed: "2026-04-05T21:33:49Z"
  tasks_completed: 2
  tasks_total: 2
  test_count: 13
  files_created: 4
  files_modified: 1
---

# Phase 17 Plan 02: MCRA Noise Estimator and Multi-Peak Detection Summary

MCRA adaptive noise floor with global-median signal presence detection and greedy multi-peak extraction with angular separation constraint.

## Task Results

### Task 1: MCRA Noise Estimator module, config extensions, and unit tests

**Commits:** `e8da3b5` (RED), `7b80c62` (GREEN)

Created `MCRANoiseEstimator` class that tracks an adaptive noise floor for SRP-PHAT spatial power maps. The algorithm uses smoothed power spectrum tracking with minimum-controlled recursive averaging. A key enhancement over classic MCRA: a secondary signal presence check using the global median of the smoothed map catches persistent signals that are present from initialization (where S/S_min ratio stays at 1.0 and cannot detect them). When signal is present, noise adapts toward the global median rather than the raw power, preventing the noise estimate from rising to the signal level.

Config fields added to `AcousticSettings`:
- BF-13: `bf_min_separation_deg=15.0`, `bf_max_peaks=5`, `bf_peak_threshold=3.0`
- BF-14: `bf_mcra_alpha_s=0.8`, `bf_mcra_alpha_d=0.95`, `bf_mcra_delta=5.0`, `bf_mcra_min_window=50`
- BF-16: `bf_holdoff_seconds=5.0`

7 unit tests covering initialization, convergence, signal preservation, reset, min-window tracking, signal presence detection, and config defaults.

### Task 2: Multi-peak detection module and unit tests

**Commits:** `552208d` (RED), `29087d6` (GREEN)

Created `detect_multi_peak` function that finds multiple simultaneous drone sources in the SRP map. Uses noise-floor-relative thresholding and a greedy angular separation algorithm: candidates are sorted by power descending, then each is accepted only if it meets the minimum angular distance from all previously accepted peaks. Respects configurable `max_peaks` limit.

6 unit tests covering well-separated peaks, close peaks (suppression), below-threshold (empty), max_peaks limit, PeakDetection value correctness, and zero-separation mode.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MCRA signal presence detection for persistent signals**
- **Found during:** Task 1
- **Issue:** Classic MCRA S/S_min ratio cannot detect signals present from frame 1 (S_min equals S, ratio stays 1.0). Persistent drone signals would have their noise estimate converge to the signal level.
- **Fix:** Added secondary signal presence check using global median of the smoothed power map. When signal is present, noise adapts toward global median instead of raw power.
- **Files modified:** `src/acoustic/beamforming/mcra.py`
- **Commit:** `7b80c62`

## Verification Results

```
PYTHONPATH=src python -m pytest tests/unit/test_mcra.py tests/unit/test_multi_peak.py -q
13 passed in 0.29s

PYTHONPATH=src python -m pytest tests/unit/test_config.py tests/unit/test_peak.py -q
33 passed in 0.98s
```

All 13 new tests pass. All existing related tests unaffected.

## Known Stubs

None - all modules are fully implemented with real algorithms.

## Self-Check: PASSED

All 4 created files verified on disk. All 4 commit hashes verified in git log.
