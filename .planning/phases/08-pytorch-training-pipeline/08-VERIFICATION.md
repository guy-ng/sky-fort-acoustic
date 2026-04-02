---
phase: 08-pytorch-training-pipeline
verified: 2026-04-02T00:00:00Z
status: passed
score: 4/4 success criteria verified
re_verification: true
re_verification_meta:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "Training produces a model checkpoint (.pt) and exports to a deployable format on completion — TorchScript .jit export added in Plan 03"
    - "Training runs as a background thread with resource isolation (os.nice, thread limits) — torch.set_num_threads(2) + torch.set_num_interop_threads(1) added in Plan 03"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Verify training does not degrade live beamforming in a running container"
    expected: "Beamforming map continues to update within 150ms intervals while training is running concurrently"
    why_human: "Cannot programmatically verify latency impact without running both the beamforming pipeline and training simultaneously in a real container"
---

# Phase 8: PyTorch Training Pipeline Verification Report

**Phase Goal:** Users can train a research CNN model from labeled WAV files with the training process isolated from live detection
**Verified:** 2026-04-02
**Status:** passed
**Re-verification:** Yes — after gap closure from Plans 03 (TorchScript export, thread limits, confusion matrix)

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| #   | Truth   | Status     | Evidence       |
| --- | ------- | ---------- | -------------- |
| 1   | Training pipeline loads WAV files lazily, extracts random 0.5s segments, and trains with Adam optimizer, BCE loss, and early stopping | ✓ VERIFIED | `DroneAudioDataset.__getitem__` uses `sf.read` lazy I/O + `random.randint` segment extraction; `TrainingRunner.run()` uses `torch.optim.Adam`, `nn.BCELoss`, `EarlyStopping`; 43/43 tests pass |
| 2   | Training runs as a background thread with resource isolation (os.nice, thread limits) and does not degrade live detection latency below the 150ms beamforming deadline | ✓ VERIFIED (partial — automated) | `os.nice(10)` at `manager.py:123`; `torch.set_num_threads(2)` at `manager.py:128`; `torch.set_num_interop_threads(1)` at `manager.py:130`; daemon thread confirmed; `test_thread_limits_applied` passes; live-latency portion requires human verification (see below) |
| 3   | Training produces a model checkpoint (.pt) and exports to a deployable format on completion | ✓ VERIFIED | `.pt` state_dict saved via `torch.save(model.state_dict(), ...)` at `trainer.py:208`; TorchScript `.jit` export via `torch.jit.script(export_model)` at `trainer.py:242`; `test_torchscript_export_created` and `test_torchscript_loadable` pass |
| 4   | Training data augmentation (SpecAugment time/frequency masking and waveform augmentation) is applied during training and can be toggled via config | ✓ VERIFIED | `WaveformAugmentation` (noise + gain) and `SpecAugment` (TimeMasking/FrequencyMasking via torchaudio) fully implemented; `augmentation_enabled` flag in `TrainingConfig` controls both; val set never augmented; 8 augmentation tests pass |

**Score:** 4/4 success criteria verified (live-latency interaction requires human confirmation — flagged below)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/acoustic/training/__init__.py` | Package init | ✓ VERIFIED | Exists |
| `src/acoustic/training/config.py` | TrainingConfig Pydantic BaseSettings | ✓ VERIFIED | `class TrainingConfig(BaseSettings)` with all hyperparameters; `env_prefix="ACOUSTIC_TRAINING_"`; all defaults correct |
| `src/acoustic/training/augmentation.py` | WaveformAugmentation and SpecAugment | ✓ VERIFIED | Both classes substantive: noise injection at random SNR, gain scaling, TimeMasking/FrequencyMasking via torchaudio with correct (1, time, freq) transpose |
| `src/acoustic/training/dataset.py` | DroneAudioDataset, collect_wav_files, build_weighted_sampler | ✓ VERIFIED | All three exported; lazy `sf.read`; `random.randint` segment extraction; zero-pad for short audio; WeightedRandomSampler inverse-frequency |
| `src/acoustic/training/trainer.py` | EarlyStopping and TrainingRunner with TorchScript export and confusion matrix | ✓ VERIFIED | Both classes present; Adam, BCELoss, ReduceLROnPlateau, stop_event check, checkpoint save, TorchScript export at line 242, TP/FP/TN/FN confusion matrix at lines 183+226 |
| `src/acoustic/training/manager.py` | TrainingManager with thread lifecycle, progress, concurrency guard, thread limits | ✓ VERIFIED | TrainingStatus enum, TrainingProgress dataclass (incl. tp/fp/tn/fn), TrainingManager with daemon thread + Lock + os.nice(10) + torch.set_num_threads(2) |
| `tests/unit/test_augmentation.py` | Augmentation unit tests | ✓ VERIFIED | 8 tests passing |
| `tests/unit/test_training_dataset.py` | Dataset unit tests | ✓ VERIFIED | 12 tests passing |
| `tests/unit/test_training_checkpoint.py` | EarlyStopping + checkpoint save/load tests | ✓ VERIFIED | 9 tests passing |
| `tests/unit/test_training_manager.py` | Manager lifecycle tests | ✓ VERIFIED | 8 tests passing |
| `tests/unit/test_training_export.py` | TorchScript export, confusion matrix, thread limits tests | ✓ VERIFIED | 5 tests passing (new in Plan 03 gap closure) |
| `tests/integration/test_training_smoke.py` | End-to-end training smoke test | ✓ VERIFIED | 1 test passing — full training cycle on synthetic data; status=COMPLETED, checkpoint loadable, output in [0,1] |

**Modified artifact:**
| `src/acoustic/classification/preprocessing.py` | mel_spectrogram_from_segment utility added | ✓ VERIFIED | `def mel_spectrogram_from_segment` at line 38; module-level cache via `_get_mel_transform`; 7 existing preprocessing tests unbroken |

---

### Key Link Verification

**Plan 01 key links:**

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `training/dataset.py` | `classification/config.py` | `from acoustic.classification.config import MelConfig` | ✓ WIRED | Line 13 of dataset.py |
| `training/dataset.py` | `classification/preprocessing.py` | `from acoustic.classification.preprocessing import mel_spectrogram_from_segment` | ✓ WIRED | Line 14 of dataset.py |
| `training/dataset.py` | `training/augmentation.py` | `from acoustic.training.augmentation import SpecAugment, WaveformAugmentation` | ✓ WIRED | Line 15 of dataset.py |

**Plan 02 key links:**

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `training/trainer.py` | `classification/research_cnn.py` | `from acoustic.classification.research_cnn import ResearchCNN` | ✓ WIRED | Line 20 of trainer.py |
| `training/trainer.py` | `training/dataset.py` | `from acoustic.training.dataset import DroneAudioDataset, build_weighted_sampler, collect_wav_files` | ✓ WIRED | Line 23 of trainer.py |
| `training/manager.py` | `training/trainer.py` | `from acoustic.training.trainer import TrainingRunner` | ✓ WIRED | Line 20 of manager.py |
| `tests/integration/test_training_smoke.py` | `training/manager.py` | `from acoustic.training.manager import TrainingManager` | ✓ WIRED | Line 37 of smoke test |

**Plan 03 key links:**

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `training/trainer.py` | `classification/research_cnn.py` | `torch.jit.script(export_model)` after loading best state_dict | ✓ WIRED | Line 242 of trainer.py — `export_model = ResearchCNN()`, `export_model.load_state_dict(...)`, `scripted = torch.jit.script(export_model)` |
| `training/manager.py` | `torch` | `torch.set_num_threads(2)` before runner.run() | ✓ WIRED | Line 128 of manager.py |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `DroneAudioDataset.__getitem__` | `audio` | `sf.read(self._paths[idx])` | Yes — reads from actual WAV files on disk | ✓ FLOWING |
| `TrainingRunner.run()` | `all_paths, all_labels` | `collect_wav_files(cfg.data_root, cfg.label_map)` | Yes — scans real filesystem | ✓ FLOWING |
| `TrainingManager._on_progress` | `self._progress` | callback from `TrainingRunner` per epoch | Yes — real epoch metrics written under Lock including tp/fp/tn/fn | ✓ FLOWING |
| `EarlyStopping.step()` | `best_loss` | `avg_val_loss` computed from real val batches | Yes — computed from model output vs. labels | ✓ FLOWING |
| `trainer.py` TorchScript export | `jit_path` | loads best state_dict from `ckpt_path` into fresh `ResearchCNN`, runs `torch.jit.script` | Yes — reads real checkpoint, produces real .jit file | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| All 43 Phase 08 tests pass | `.venv/bin/pytest tests/unit/test_augmentation.py tests/unit/test_training_dataset.py tests/unit/test_training_checkpoint.py tests/unit/test_training_manager.py tests/unit/test_training_export.py tests/integration/test_training_smoke.py -v` | 43 passed in 4.91s | ✓ PASS |
| 187 unit tests pass (no regressions) | `.venv/bin/pytest tests/unit/ -q` | 187 passed in 11.01s | ✓ PASS |
| TorchScript export code present | `grep "torch.jit.script" src/acoustic/training/trainer.py` | Line 242 match | ✓ PASS |
| Thread limits present | `grep "torch.set_num_threads" src/acoustic/training/manager.py` | Line 128 match | ✓ PASS |
| Confusion matrix in progress | `grep "tp.*fp.*tn.*fn" src/acoustic/training/trainer.py` | Lines 183, 226 match | ✓ PASS |
| TorchScript + confusion matrix tests | `.venv/bin/pytest tests/unit/test_training_export.py -v` | 5 passed | ✓ PASS |

**Note:** TorchScript (`torch.jit.script`) generates DeprecationWarnings on Python 3.14+ ("not supported in Python 3.14+ and may break. Please switch to `torch.compile` or `torch.export`"). Tests pass today. This is a forward-compatibility warning for when Python 3.14 support becomes more constrained in future PyTorch releases. Not a current blocker.

---

### Requirements Coverage

The PLANS for Phase 8 claim TRN-01, TRN-02, TRN-03, TRN-04. These IDs are defined in REQUIREMENTS.md as:

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| TRN-01 | 08-01, 08-02 | Web UI provides interface to select labeled recordings as training dataset | ? DEFERRED | Phase 8 delivers only the backend training engine. Training data directory path is configurable via `ACOUSTIC_TRAINING_DATA_ROOT` env var but there is no Web UI or REST API endpoint. TRN-01 is a UI requirement deferred to Phase 9 (REST API) or Phase 10 (field data collection UI). The plans over-claim this requirement ID. |
| TRN-02 | 08-02, 08-03 | Service runs CNN training as a background subprocess (does not block live detection) | ✓ SATISFIED | `TrainingManager` runs `TrainingRunner` in `daemon=True` thread with `os.nice(10)`, `torch.set_num_threads(2)`, and `torch.set_num_interop_threads(1)`. Training is non-blocking. Live-latency verification requires human testing. |
| TRN-03 | 08-02, 08-03 | Training produces a new model file with validation metrics (accuracy, confusion matrix) | ✓ SATISFIED | `.pt` checkpoint saved. `val_acc` tracked per epoch in `TrainingProgress`. Confusion matrix (TP/FP/TN/FN) computed per validation epoch and included in progress callback and `TrainingProgress` dataclass. `test_confusion_matrix_in_progress` verifies. |
| TRN-04 | 08-01, 08-02 | Web UI displays training progress and results | ? DEFERRED | Phase 8 delivers no Web UI. `TrainingProgress` dataclass is thread-safe and ready for Phase 9's REST/WebSocket API consumption. Plans over-claim this requirement ID. |

**Requirements note:** REQUIREMENTS.md traceability table maps TRN-01/02/03/04 to Phase 5. Phase 5 is unimplemented. Phase 8 is the first phase to implement the training backend. TRN-02 and TRN-03 are now satisfied by Phase 8. TRN-01 and TRN-04 (UI requirements) remain deferred pending Phase 9/10.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `src/acoustic/training/manager.py` | 130-132 | `torch.set_num_interop_threads(1)` wrapped in `try/except RuntimeError` — silently ignored if already set | ℹ️ Info | Intentional: this call can only succeed once per process. Wrapping is correct behavior documented in SUMMARY. No correctness impact. |
| `src/acoustic/training/trainer.py` | 235-246 | TorchScript export failure silently logged and swallowed — non-fatal | ℹ️ Info | Intentional: state_dict checkpoint remains available. `.jit` file is an additional deployable artifact. Failure mode is documented. |
| `src/acoustic/training/trainer.py` | `torch.jit.script` | DeprecationWarning on Python 3.14+ — torch.jit.script not supported in Python 3.14+ | ⚠️ Warning | Tests pass today. When PyTorch drops Python 3.14 TorchScript support (currently "may break"), this will fail silently due to the try/except wrapper. Forward-compatibility risk for Phase 9+. Consider migrating to `torch.export.export()` before Phase 9 ships. |

---

### Human Verification Required

#### 1. Verify live detection latency during concurrent training

**Test:** Start the full service with a running audio pipeline (or simulator), then trigger `TrainingManager.start()` with a real or synthetic dataset. Monitor beamforming map update frequency from the `/api/beamforming-map` endpoint.
**Expected:** Beamforming map continues to update within 150ms intervals while training is running. Audio callback does not drop frames. `torch.set_num_threads(2)` should limit PyTorch to 2 intra-op threads, leaving cores available for the beamforming pipeline.
**Why human:** Cannot programmatically test the latency interaction between the beamforming pipeline (separate worker thread with 150ms chunks) and the training thread without running both simultaneously in a container with real-time audio input.

---

### Re-verification Summary

This is a re-verification after Plan 03 gap closure. Both gaps identified in the initial verification are now closed:

**Gap 1 — TorchScript export (previously FAILED):** CLOSED.
`trainer.py:235-246` adds a TorchScript export step after saving the best checkpoint. `torch.jit.script(export_model)` produces a `.jit` file at `{checkpoint_path}.jit`. `test_torchscript_export_created` and `test_torchscript_loadable` verify the file exists and produces valid (1, 1) output with Sigmoid preserved. ROADMAP SC3 satisfied.

**Gap 2 — Thread limits (previously PARTIAL):** CLOSED.
`manager.py:128-133` calls `torch.set_num_threads(2)` and `torch.set_num_interop_threads(1)` in `_run()` before creating `TrainingRunner`. The interop call is wrapped in `try/except RuntimeError` (can only be called once per process — this is correct). `test_thread_limits_applied` verifies both calls are made with expected values. ROADMAP SC2 satisfied for automated verification. Live-latency verification remains human-only.

**Gap 3 — Confusion matrix (previously BLOCKED for TRN-03):** CLOSED.
`trainer.py:183-200` accumulates TP/FP/TN/FN per validation epoch. `trainer.py:226` includes them in the progress callback dict. `manager.py:47-50` adds four fields to `TrainingProgress`. `manager.py:160-163` reads them in `_on_progress()`. `test_confusion_matrix_in_progress` verifies. TRN-03 satisfied.

**No regressions:** 187 unit tests pass (up from 182 at initial verification, adding 5 new tests in test_training_export.py).

---

_Verified: 2026-04-02_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — initial gaps closed by Plan 03_
