---
phase: 11-late-fusion-ensemble-conditional
plan: 01
subsystem: classification
tags: [ensemble, weighted-voting, pydantic, model-registry, pytorch]

requires:
  - phase: 07-research-cnn-and-inference-integration
    provides: ResearchCNN and ResearchClassifier, Classifier protocol
provides:
  - EnsembleClassifier with weighted soft voting over N models
  - ModelRegistry for pluggable architecture resolution
  - EnsembleConfig/ModelEntry Pydantic models for JSON config parsing
  - ensemble_config_path setting in AcousticSettings
affects: [11-02-PLAN, live-pipeline-integration, evaluation-harness]

tech-stack:
  added: []
  patterns: [model-registry-pattern, weighted-soft-voting, protocol-based-classification]

key-files:
  created:
    - src/acoustic/classification/ensemble.py
    - tests/unit/test_ensemble.py
  modified:
    - src/acoustic/config.py

key-decisions:
  - "Module-level dict registry for model loaders -- simple, extensible, no framework needed"
  - "Weights normalized at construction time (static), not per-predict call"
  - "Live mode hard cap at 3 models for latency control"

patterns-established:
  - "Model registry: register_model(type_name, loader_fn) -> load_model(type_name, path)"
  - "Ensemble config via JSON file pointed to by env var ACOUSTIC_ENSEMBLE_CONFIG_PATH"

requirements-completed: [ENS-01, ENS-02]

duration: 4min
completed: 2026-04-02
---

# Phase 11 Plan 01: Core Ensemble Module Summary

**EnsembleClassifier with F1-weighted soft voting, model registry for architecture resolution, and JSON config parsing via Pydantic**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-02T17:31:43Z
- **Completed:** 2026-04-02T17:35:45Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- EnsembleClassifier combines N Classifier predictions via weighted probability averaging, satisfying the Classifier protocol
- Model registry maps type strings to loader functions with "research_cnn" pre-registered
- EnsembleConfig/ModelEntry Pydantic models parse and validate JSON config files
- Live mode enforces hard cap of 3 models; offline mode has no limit
- AcousticSettings extended with ensemble_config_path field (env var ACOUSTIC_ENSEMBLE_CONFIG_PATH)
- 12 unit tests covering all behaviors, all 113 existing tests still passing

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for ensemble module** - `fbfeb69` (test)
2. **Task 1 (GREEN): Implement ensemble classifier** - `950f915` (feat)
3. **Task 2: Add ensemble_config_path to AcousticSettings** - `469c57c` (feat)

## Files Created/Modified
- `src/acoustic/classification/ensemble.py` - EnsembleClassifier, ModelRegistry, EnsembleConfig, ModelEntry
- `src/acoustic/classification/protocols.py` - Classifier protocol (dependency, copied from main)
- `src/acoustic/classification/research_cnn.py` - ResearchCNN/ResearchClassifier (dependency, copied from main)
- `src/acoustic/config.py` - Added ensemble_config_path field
- `tests/unit/test_ensemble.py` - 12 unit tests for all ensemble behaviors

## Decisions Made
- Module-level dict registry for model loaders -- simple and extensible without framework overhead
- Weights normalized at construction time (static per D-05), not recomputed per predict call
- Live mode hard cap at 3 models (per D-02) for latency control in real-time pipeline

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Copied protocols.py and research_cnn.py into worktree**
- **Found during:** Task 1 (ensemble module implementation)
- **Issue:** Worktree missing files from phases 7-10 that ensemble.py depends on (Classifier protocol, ResearchCNN/ResearchClassifier)
- **Fix:** Copied protocols.py and research_cnn.py from main repo to worktree classification dir
- **Files modified:** src/acoustic/classification/protocols.py, src/acoustic/classification/research_cnn.py
- **Verification:** Tests import and run successfully
- **Committed in:** 950f915 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for parallel worktree execution. No scope creep.

## Issues Encountered
- Editable pip install points to main repo src/, not worktree src/. Used PYTHONPATH override for test verification.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ensemble module ready for Plan 02 to wire into evaluation harness and live pipeline
- Model registry extensible for future architecture types
- Config parsing ready for JSON ensemble config files

## Self-Check: PASSED

All 5 files verified present. All 3 commit hashes verified in git log.

---
*Phase: 11-late-fusion-ensemble-conditional*
*Completed: 2026-04-02*
