# Quick Task: Stub Beamforming — Remove SRP-PHAT

**Task ID:** quick-260401-948
**Created:** 2026-04-01
**Status:** DONE

## Goal
Stop beamforming when no target is detected. Remove beamforming computation from the pipeline and replace with stubs. Commit all changes.

## Changes
- `src/acoustic/pipeline.py`: Replaced `process_chunk()` SRP-PHAT computation with a stub that returns a zero map and no peak. Removed imports of `build_mic_positions`, `detect_peak_with_threshold`, `srp_phat_2d`. Kept beamforming module files intact for future re-integration.

## Verification
- Import check: OK
- Stub produces correct shape (181, 91) zero map
- All 101 unit tests pass
- Integration test assertions compatible with stub (None peak, correct shape)
