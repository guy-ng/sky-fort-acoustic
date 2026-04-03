---
phase: 14-efficientat-model-architecture-with-audioset-transfer-learning
plan: 01
subsystem: classification
tags: [efficientat, mobilenetv3, audioset, transfer-learning, pytorch, model-registry]

requires:
  - phase: 03-cnn-classification
    provides: Classifier protocol, model registry, ensemble system, CNNWorker pipeline
provides:
  - Vendored EfficientAT MobileNetV3 mn10 architecture (~4.88M params)
  - EfficientATClassifier wrapper implementing Classifier protocol
  - Model registry entry "efficientat_mn10"
  - AugmentMelSTFT preprocessing (32kHz, 128 mels, preemphasis + log + normalization)
  - EfficientATMelConfig dataclass for preprocessing parameters
  - cnn_model_type config field for model selection at startup
  - Pretrained weight download script (mn10_as_mAP_471.pt from GitHub releases)
affects: [14-02, 15-advanced-training-enhancements, 16-edge-export-pipeline]

tech-stack:
  added: [torchvision (ConvNormActivation for MobileNetV3 blocks)]
  patterns: [vendored-model-package, channel-unsqueeze-for-mel-to-conv2d]

key-files:
  created:
    - src/acoustic/classification/efficientat/__init__.py
    - src/acoustic/classification/efficientat/model.py
    - src/acoustic/classification/efficientat/inverted_residual.py
    - src/acoustic/classification/efficientat/preprocess.py
    - src/acoustic/classification/efficientat/classifier.py
    - src/acoustic/classification/efficientat/config.py
    - src/acoustic/classification/efficientat/utils.py
    - src/acoustic/classification/efficientat/attention_pooling.py
    - scripts/download_pretrained.py
    - tests/unit/test_efficientat.py
  modified:
    - src/acoustic/config.py
    - src/acoustic/main.py
    - tests/unit/test_config.py

key-decisions:
  - "Vendored 4 EfficientAT source files (model, block_types, preprocess, attention_pooling, utils) instead of pip install -- EfficientAT is not a PyPI package"
  - "Added channel unsqueeze(1) between AugmentMelSTFT output and MN model input -- mel outputs 3D (batch, mels, time) but Conv2D needs 4D (batch, 1, mels, time)"
  - "Removed pretrained download from model.py -- handled externally by scripts/download_pretrained.py"
  - "Stripped helpers/utils.py side effects (CSV loading) -- only vendored NAME_TO_WIDTH map and math utilities"

patterns-established:
  - "Vendored model package pattern: all model code under src/acoustic/classification/<model_name>/ with __init__.py registering in ensemble registry"
  - "Classifier wrapper pattern: each model type has its own Classifier wrapper handling preprocessing internally"

requirements-completed: [MDL-10, MDL-12]

duration: 19min
completed: 2026-04-04
---

# Phase 14 Plan 01: EfficientAT mn10 Model Architecture Summary

**Vendored EfficientAT MobileNetV3 mn10 (~4.88M params) with AudioSet preprocessing, Classifier protocol wrapper, and registry-based model selection at startup**

## Performance

- **Duration:** 19 min
- **Started:** 2026-04-04T03:09:15Z
- **Completed:** 2026-04-04T03:28:14Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Vendored EfficientAT MN model architecture (15 inverted residual blocks, squeeze-excitation, MLP head) from fschmid56/EfficientAT
- Created EfficientATClassifier satisfying Classifier protocol with internal AugmentMelSTFT preprocessing
- Registered "efficientat_mn10" in model registry, startup factory now uses registry-based loading
- Added ACOUSTIC_CNN_MODEL_TYPE config field for model type selection via environment variable
- Created pretrained weight download script for mn10_as_mAP_471.pt (18MB)
- All 39 tests pass (11 new EfficientAT tests + 28 existing tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Vendor EfficientAT model code, create classifier wrapper, config, registry, and download script** - `1401ab5` (test: TDD RED), `8df2cdd` (feat: TDD GREEN)
2. **Task 2: Extend startup classifier factory to use cnn_model_type config** - `9386828` (feat)

## Files Created/Modified
- `src/acoustic/classification/efficientat/__init__.py` - Registry registration, _load_efficientat_mn10 loader
- `src/acoustic/classification/efficientat/model.py` - Vendored MN class and get_model() factory
- `src/acoustic/classification/efficientat/inverted_residual.py` - InvertedResidual, InvertedResidualConfig, ConcurrentSEBlock
- `src/acoustic/classification/efficientat/preprocess.py` - AugmentMelSTFT (preemphasis + mel + log + normalization)
- `src/acoustic/classification/efficientat/classifier.py` - EfficientATClassifier wrapping MN model
- `src/acoustic/classification/efficientat/config.py` - EfficientATMelConfig dataclass (32kHz, 128 mels)
- `src/acoustic/classification/efficientat/utils.py` - make_divisible, cnn_out_size, collapse_dim utilities
- `src/acoustic/classification/efficientat/attention_pooling.py` - MultiHeadAttentionPooling head option
- `scripts/download_pretrained.py` - Downloads mn10_as_mAP_471.pt from GitHub releases
- `src/acoustic/config.py` - Added cnn_model_type field
- `src/acoustic/main.py` - Replaced hardcoded ResearchCNN with registry load_model()
- `tests/unit/test_efficientat.py` - 11 tests covering model, protocol, registry, config
- `tests/unit/test_config.py` - 2 new tests for cnn_model_type config field

## Decisions Made
- Vendored EfficientAT source as a package rather than pip install (not a PyPI package, need import control)
- Added channel unsqueeze between mel output and model input (AugmentMelSTFT outputs 3D but Conv2D needs 4D)
- Stripped side effects from helpers/utils.py (CSV loading at import time) -- only kept math utilities
- Removed pretrained auto-download from model.py in favor of explicit download script

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mel-to-model dimension mismatch**
- **Found during:** Task 1 (test_output_shape_binary)
- **Issue:** AugmentMelSTFT outputs (batch, n_mels, time) 3D tensor but MN model's first Conv2D layer expects (batch, 1, n_mels, time) 4D input
- **Fix:** Added `mel.unsqueeze(1)` in EfficientATClassifier.predict() and test
- **Files modified:** src/acoustic/classification/efficientat/classifier.py, tests/unit/test_efficientat.py
- **Verification:** test_output_shape_binary and test_predict_returns_float both pass
- **Committed in:** 8df2cdd

**2. [Rule 3 - Blocking] Adapted EfficientAT repo structure (no inverted_residual.py standalone)**
- **Found during:** Task 1 (vendoring)
- **Issue:** Plan referenced `models/mn/inverted_residual.py` but actual repo has `models/mn/block_types.py` containing both InvertedResidual and SE blocks, plus separate `utils.py` and `attention_pooling.py`
- **Fix:** Vendored the actual files (block_types.py -> inverted_residual.py, plus utils.py and attention_pooling.py)
- **Files modified:** Created all 5 vendored files with corrected relative imports
- **Verification:** All 11 tests pass
- **Committed in:** 8df2cdd

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correct operation. No scope creep.

## Issues Encountered
None beyond the documented deviations.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired.

## Next Phase Readiness
- EfficientAT model architecture ready for Plan 02 (three-stage transfer learning trainer)
- Pretrained weights can be downloaded via `python scripts/download_pretrained.py`
- Model selectable at startup via `ACOUSTIC_CNN_MODEL_TYPE=efficientat_mn10`

---
*Phase: 14-efficientat-model-architecture-with-audioset-transfer-learning*
*Completed: 2026-04-04*
