---
phase: 17-beamforming-engine-upgrade-and-pipeline-integration
plan: 01
subsystem: beamforming
tags: [dsp, bandpass, interpolation, config, srp-phat]
dependency_graph:
  requires: []
  provides: [BandpassFilter, parabolic_interpolation_2d, bf_config_fields]
  affects: [src/acoustic/config.py, src/acoustic/beamforming/]
tech_stack:
  added: [scipy.signal.butter, scipy.signal.sosfilt]
  patterns: [streaming-filter-state, parabolic-interpolation, sos-format]
key_files:
  created:
    - src/acoustic/beamforming/bandpass.py
    - src/acoustic/beamforming/interpolation.py
    - tests/unit/test_bandpass.py
    - tests/unit/test_interpolation.py
  modified:
    - src/acoustic/config.py
decisions:
  - "Used 10000 Hz test tone instead of 6000 Hz for above-passband test -- order-4 Butterworth at 6kHz/4kHz cutoff only achieves ~18 dB attenuation, insufficient for >20 dB threshold"
metrics:
  duration: 10m15s
  completed: 2026-04-05
  tasks_completed: 2
  tasks_total: 2
  files_created: 4
  files_modified: 1
---

# Phase 17 Plan 01: Bandpass Pre-Filter and Parabolic Interpolation Summary

Streaming Butterworth bandpass filter (500-4000 Hz, SOS format) and parabolic sub-grid interpolation for sub-degree DOA, plus all Phase 17 config extensions for beamforming parameters.

## What Was Built

### Task 1: BandpassFilter Module + Config Extensions
- **BandpassFilter** class in `src/acoustic/beamforming/bandpass.py`: streaming per-channel Butterworth bandpass filter using SOS format for numerical stability
- Per-channel state tracking via `sosfilt_zi` -- continuous filtering across audio chunks without transient artifacts
- `apply(signals)` auto-initializes state on first call or channel count change
- `reset(n_channels)` clears state for fresh start
- All 12 `bf_*` config fields added to `AcousticSettings`: frequency band (500-4000 Hz), filter order (4), multi-peak detection params (separation, max peaks, threshold), MCRA noise estimation params (alpha_s, alpha_d, delta, min_window), and holdoff seconds
- 9 unit tests covering SOS shape, passband/stopband attenuation, streaming state, reset, multi-channel, and config defaults

### Task 2: Parabolic Interpolation Module
- **parabolic_interpolation_2d** function in `src/acoustic/beamforming/interpolation.py`: refines SRP map peak from grid-quantized to sub-degree accuracy
- Independent azimuth and elevation refinement with configurable grid spacing
- Boundary-safe: falls back to grid-quantized value at edges (no out-of-bounds access)
- Handles degenerate case (equal neighbors) by returning grid value
- 5 unit tests including analytical verification (delta = 1/6 degree for known input values)

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 | fe7c70b | feat(17-01): add BandpassFilter module and all Phase 17 config extensions |
| 2 | b8ce8ea | feat(17-01): add parabolic interpolation module for sub-degree DOA accuracy |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test frequency adjusted for above-passband attenuation test**
- **Found during:** Task 1
- **Issue:** Plan specified 6000 Hz test tone with >20 dB attenuation expectation, but order-4 Butterworth with 4000 Hz cutoff only achieves ~17.9 dB at 6000 Hz (1.5x cutoff ratio is too close)
- **Fix:** Changed test frequency to 10000 Hz (2.5x cutoff) which achieves >20 dB attenuation as required
- **Files modified:** tests/unit/test_bandpass.py
- **Commit:** fe7c70b

## Verification

- `pytest tests/unit/test_bandpass.py` -- 9 passed
- `pytest tests/unit/test_interpolation.py` -- 5 passed
- `pytest tests/unit/` -- 115 passed (all existing tests unaffected)

## Known Stubs

None -- all modules are fully implemented with real DSP logic.

## Self-Check: PASSED

- All 5 created/modified files found on disk
- Both commits (fe7c70b, b8ce8ea) verified in git log
