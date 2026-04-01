---
phase: 06-preprocessing-parity-foundation
plan: 02
subsystem: classification
tags: [torchaudio, preprocessor, parity, protocol-injection, mel-spectrogram]

# Dependency graph
requires: [06-01]
provides:
  - ResearchPreprocessor implementing Preprocessor protocol with torchaudio
  - Numerical parity with librosa reference within atol=1e-4
  - CNNWorker tested with protocol injection
  - Pipeline using 0.5s segments for research-standard CNN input
affects: [07-cnn-architecture, 08-training-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [per-spectrogram-ref-max-db, torchaudio-mel-spectrogram]

key-files:
  created:
    - tests/unit/test_parity.py
    - tests/unit/test_worker.py
  modified:
    - src/acoustic/classification/preprocessing.py
    - src/acoustic/pipeline.py

key-decisions:
  - "Used norm='slaney' in torchaudio MelSpectrogram to match librosa.filters.mel default"
  - "Custom _power_to_db with per-spectrogram ref=max instead of AmplitudeToDB (matches librosa.power_to_db(ref=np.max))"
  - "Pipeline segment changed from 2.0s to 0.5s per research standard"

patterns-established:
  - "Per-spectrogram max-reference dB normalization for librosa parity"

requirements-completed: [PRE-03, PRE-04]

# Metrics
duration: 7min
completed: 2026-04-01
---

# Phase 06 Plan 02: Preprocessing Parity Implementation Summary

**torchaudio ResearchPreprocessor with per-spectrogram ref=max dB normalization achieving librosa parity within atol=1e-4, plus protocol-tested CNNWorker and 0.5s pipeline segments**

## Performance

- **Duration:** 7 min
- **Completed:** 2026-04-01
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced entire librosa-based preprocessing.py with torchaudio ResearchPreprocessor
- Achieved numerical parity within atol=1e-4 against librosa reference fixture (128, 64)
- Custom _power_to_db function matches librosa.power_to_db(ref=np.max) behavior
- Pipeline segment duration changed from 2.0s to 0.5s matching research constants
- Added comprehensive test_worker.py verifying protocol injection and source cleanliness
- All 118 unit tests pass with zero regressions

## Task Commits

1. **Task 1: ResearchPreprocessor + parity tests** - `c95d5db` (feat)
2. **Task 2: Pipeline 0.5s segments + worker tests** - `fcde32e` (feat)

## Files Created/Modified
- `src/acoustic/classification/preprocessing.py` - Complete rewrite: torchaudio ResearchPreprocessor
- `src/acoustic/pipeline.py` - CNN segment duration 2.0s -> 0.5s
- `tests/unit/test_preprocessing.py` - Complete rewrite: ResearchPreprocessor tests (7 tests)
- `tests/unit/test_parity.py` - Numerical parity against librosa reference (3 tests)
- `tests/unit/test_worker.py` - CNNWorker constructor and protocol injection (6 tests)

## Decisions Made
- Used `norm="slaney"` in torchaudio MelSpectrogram (matches librosa.filters.mel default, not torchaudio default of None)
- Implemented custom `_power_to_db` instead of `AmplitudeToDB` because AmplitudeToDB uses fixed ref=1.0 while librosa uses per-spectrogram ref=np.max
- Adjusted silence test from "values near 0" to "valid range" since ref=max normalization maps silence to 1.0 (mathematically correct)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mel filterbank norm parameter**
- **Found during:** Task 1
- **Issue:** Plan specified `norm="slaney"` matching librosa default, but also listed `norm=None` in one place. Investigation confirmed librosa.filters.mel defaults to `norm="slaney"`, not `None`.
- **Fix:** Used `norm="slaney"` in torchaudio MelSpectrogram
- **Files modified:** src/acoustic/classification/preprocessing.py

**2. [Rule 1 - Bug] Replaced AmplitudeToDB with custom _power_to_db**
- **Found during:** Task 1
- **Issue:** torchaudio's AmplitudeToDB uses fixed ref=1.0 while librosa.power_to_db(ref=np.max) uses per-spectrogram maximum as reference, causing 24.7% element mismatch
- **Fix:** Custom _power_to_db function implementing per-spectrogram max reference
- **Files modified:** src/acoustic/classification/preprocessing.py

**3. [Rule 1 - Bug] Fixed silence test expectation**
- **Found during:** Task 1
- **Issue:** Plan expected silence to produce values near 0.0, but with ref=max normalization, silence (all equal tiny values) correctly normalizes to 1.0 (0 dB relative to max = max normalized value)
- **Fix:** Changed test to verify valid output range instead of near-zero values
- **Files modified:** tests/unit/test_preprocessing.py

## Issues Encountered

None beyond the deviations above (all auto-fixed).

## Known Stubs

None.

---
*Phase: 06-preprocessing-parity-foundation*
*Completed: 2026-04-01*
