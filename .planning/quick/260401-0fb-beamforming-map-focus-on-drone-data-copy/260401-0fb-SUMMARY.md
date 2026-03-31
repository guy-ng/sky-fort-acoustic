---
phase: quick
plan: 260401-0fb
subsystem: beamforming-visualization
tags: [beamforming, heatmap, normalization, dB, origin-suppression]
dependency_graph:
  requires: []
  provides: [normalized-heatmap-data, origin-suppression]
  affects: [pipeline, peak-detection, heatmap-canvas]
tech_stack:
  added: []
  patterns: [dB-normalization, squared-contrast, origin-artifact-suppression]
key_files:
  created: []
  modified:
    - src/acoustic/config.py
    - src/acoustic/pipeline.py
    - src/acoustic/beamforming/peak.py
    - web/src/components/heatmap/HeatmapCanvas.tsx
    - tests/unit/test_peak.py
decisions:
  - "Origin suppression uses angular distance (3.5 deg) instead of POC's Cartesian distance (0.06m at z=1m focal plane) -- equivalent result, simpler math"
  - "Pre-compute origin distance mesh in __init__ to avoid per-frame meshgrid allocation"
metrics:
  duration: 5min
  completed: 2026-04-01
  tasks: 2
  files: 5
---

# Quick Task 260401-0fb: Beamforming Map Focus on Drone Data (POC Copy) Summary

**One-liner:** dB conversion + top-10-dB masking + origin suppression in backend, squared contrast on frontend -- heatmap now shows drone hotspots instead of uniform noise.

## What Changed

### Task 1: Config settings and dB/masking in pipeline (7505d3e)

Added `mask_threshold_db` (10.0) and `ignore_origin_deg` (3.5) to `AcousticSettings`. Implemented `BeamformingPipeline._normalize_map()` which converts raw SRP power to dB, suppresses the broadside (0,0) artifact, and masks everything below the top 10 dB into a [0,1] normalized range. `latest_map` now stores this normalized data for WebSocket broadcast, while peak detection continues to operate on raw SRP values. Origin distance mesh is pre-computed in `__init__` for efficiency.

### Task 2: Origin suppression in peak detection + frontend squared normalization (caeaabd)

Added `ignore_origin_deg` parameter to `detect_peak_with_threshold()` -- peaks within the specified angular radius of (0,0) are zeroed before threshold/max logic runs. Pipeline now passes this setting through. Frontend `HeatmapCanvas` was updated to remove min/max normalization (backend now sends [0,1] data) and apply squared contrast (`v * v`) matching the POC's `alpha**2` behavior. Added `test_origin_suppressed` test validating that origin peaks are skipped in favor of secondary peaks.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 | 7505d3e | feat(quick-0fb): add dB normalization and origin suppression to beamforming map |
| 2 | caeaabd | feat(quick-0fb): add origin suppression to peak detection and squared normalization to frontend |

## Verification

- Task 1 inline verification: config fields exist with correct defaults, normalize_map produces [0,1] output, origin is suppressed
- Task 2: all 5 peak detection tests pass including new origin suppression test
- Full suite: 130/130 tests pass (1 pre-existing failure in test_health_simulated_mode -- unrelated to this change)

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED
