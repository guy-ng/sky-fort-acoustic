---
phase: 20
plan: 07
subsystem: training
tags: [trainer, correctness, specaugment, focal-loss, save-gate, transfer-learning]
wave: 3
requires:
  - 20-00
  - 20-04
provides:
  - trainer-correctness-fixes-D30-D33
affects:
  - src/acoustic/training/efficientat_trainer.py
  - src/acoustic/training/config.py
  - tests/unit/test_efficientat_training.py
tech_stack_added: []
tech_stack_patterns:
  - "build_loss_function(cfg) factory wiring"
  - "behavioral checkpoint save gate"
  - "narrow-scope stage-1 unfreezing"
key_files_created:
  - tests/unit/training/test_specaug_scaling.py
  - tests/unit/training/test_trainer_loss_factory.py
  - tests/unit/training/test_save_gate.py
  - tests/unit/training/test_stage1_unfreeze_scope.py
key_files_modified:
  - src/acoustic/training/efficientat_trainer.py
  - src/acoustic/training/config.py
  - tests/unit/test_efficientat_training.py
decisions:
  - "SpecAugment widths default to freqm=8 / timem=10 -- proportional to ~100-frame input"
  - "save_gate_min_accuracy=0.55 default; threshold<=0 disables gate (for smoke tests)"
  - "Stage 1 only unfreezes classifier[-1]; stage 2 unfreezes full classifier"
  - "build_loss_function wired with cfg.loss_function/focal_alpha/focal_gamma/bce_pos_weight"
metrics:
  duration_seconds: 615
  completed: 2026-04-06T19:17:42Z
  tasks_completed: 4
  files_modified: 3
  files_created: 4
requirements_satisfied:
  - D-30
  - D-31
  - D-32
  - D-33
---

# Phase 20 Plan 07: Trainer Correctness Fixes Summary

Applied the four trainer-correctness fixes (D-30..D-33) from the training-collapse debug session to EfficientAT trainer and config so Phase 20 augmentation (Plan 20-04) can actually converge to a non-degenerate model.

## One-liner

Config-driven SpecAugment + loss factory + behavioral save gate + narrow stage-1 unfreeze, closing all four root causes (PRIMARY-A, PRIMARY-C, CONTRIBUTING-D, CONTRIBUTING-F) of the v3/v5/v6 constant-output collapse.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 RED | failing test for SpecAugment config scaling | 567afe7 | tests/unit/training/test_specaug_scaling.py |
| 1 GREEN | scale SpecAugment masks to actual input dim (D-30) | 3d04c2a | src/acoustic/training/{config,efficientat_trainer}.py, tests/unit/training/test_specaug_scaling.py |
| 2 RED | failing test for trainer loss factory wiring | 1d949d6 | tests/unit/training/test_trainer_loss_factory.py |
| 2 GREEN | wire build_loss_function (D-31) | 3b0208a | src/acoustic/training/efficientat_trainer.py |
| 3 RED | failing tests for save gate | 4ee0ae6 | tests/unit/training/test_save_gate.py |
| 3 GREEN | behavioral save gate (D-32) | df68a4b | src/acoustic/training/efficientat_trainer.py, tests/unit/test_efficientat_training.py |
| 4 RED | failing tests for narrowed stage 1 unfreezing | 992cf9b | tests/unit/training/test_stage1_unfreeze_scope.py |
| 4 GREEN | narrow stage 1 to final head only (D-33) | 8225062 | src/acoustic/training/efficientat_trainer.py, tests/unit/test_efficientat_training.py |

## D-30: SpecAugment scaled to actual input

- Added `TrainingConfig.specaug_freq_mask=8` and `specaug_time_mask=10` (pydantic fields with descriptions).
- Replaced hard-coded `freqm=48, timem=192` in `mel_train` construction with `cfg.specaug_freq_mask` / `cfg.specaug_time_mask`.
- `mel_eval` remains untouched at `freqm=0, timem=0`.
- Legacy 48/192 values would mask up to 100% of the time axis on ~100-frame EfficientAT inputs — the empirically confirmed PRIMARY-A root cause of training collapse.

## D-31: Loss factory wired

- Added `from acoustic.training.losses import build_loss_function` import.
- Replaced `criterion = nn.BCEWithLogitsLoss()` with:
  ```python
  criterion = build_loss_function(
      loss_type=cfg.loss_function,
      focal_alpha=cfg.focal_alpha,
      focal_gamma=cfg.focal_gamma,
      bce_pos_weight=cfg.bce_pos_weight,
  ).to(device)
  ```
- `cfg.loss_function` (default `"focal"`) now actually takes effect; operators can switch to `"bce"` or `"bce_weighted"` via env var.
- `FocalLoss`, `BCEWithLogitsLoss`, and `BCEWithLogitsLoss(pos_weight=…)` all present-callable signatures `(logits, target) -> scalar`, so no call-site changes needed.

## D-32: Behavioral save gate

- Added `TrainingConfig.save_gate_min_accuracy=0.55`.
- In the per-epoch checkpoint block: when `early_stopping.step(avg_val_loss)` reports an improvement, compute `gate_acc = (tp+tn)/total` and require `min(tp, tn) > 0 AND gate_acc >= cfg.save_gate_min_accuracy` before writing the checkpoint.
- If the gate blocks a save, log:
  ```
  WARNING save gate blocked: tp=X tn=Y val_acc=Z (degenerate output, threshold=T)
  ```
- When `save_gate_min_accuracy <= 0`, the gate is fully disabled — this is how the synthetic smoke test and unit tests opt out (see `test_efficientat_training.py::TestTrainingLoopSmoke` and `test_save_gate.py::test_save_gate_allows_save_when_threshold_zero`).
- Early stopping still tracks val_loss as a side guard; the gate is AND-ed with the improvement check, it is not a control loop.

## D-33: Narrow stage 1 unfreezing

- `_setup_stage1` now freezes every param then only unfreezes `model.classifier[-1]` (the fresh `Linear(1280, 1)` binary head).
- `_setup_stage2` was extended to also unfreeze the whole classifier (not just `features[-3:]`) since stage 1 no longer does it.
- `_setup_stage3` unchanged — still unfreezes everything.
- Prevents the pretrained `Linear(1280, 1280)` head from being clobbered by Adam@1e-3 on masked/noisy inputs during stage 1.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] AugmentMelSTFT attribute access in test**
- **Found during:** Task 1 RED→GREEN verification
- **Issue:** Initial test asserted `mel.freqm == 8`, but `AugmentMelSTFT` wraps freqm/timem as `torchaudio.transforms.FrequencyMasking`/`TimeMasking` modules whose config lives in `.mask_param`.
- **Fix:** Test now asserts `mel.freqm.mask_param == 8` and `mel.timem.mask_param == 10`.
- **Files:** `tests/unit/training/test_specaug_scaling.py`
- **Commit:** 3d04c2a

**2. [Rule 3 - Blocker] Smoke test blocked by save gate defaults**
- **Found during:** Task 3 GREEN verification
- **Issue:** The existing `test_efficientat_training.py::TestTrainingLoopSmoke` trains on 10 synthetic random-noise samples. Under the new default `save_gate_min_accuracy=0.55`, the gate correctly refuses to save any checkpoint (model cannot hit 55% on random labels), and the smoke test's `assert result is not None` failed.
- **Fix:** (a) Added explicit `save_gate_min_accuracy=0.0` to the smoke test fixture. (b) Added an early-return short-circuit in the gate: when `save_gate_min_accuracy <= 0.0`, the gate is fully disabled (no `min(tp,tn) > 0` check either). This lets operators explicitly disable the gate and keeps the smoke test exercising the full training loop.
- **Files:** `tests/unit/test_efficientat_training.py`, `src/acoustic/training/efficientat_trainer.py`
- **Commit:** df68a4b

**3. [Rule 1 - Bug] Stage 2 no longer unfreezes classifier after D-33**
- **Found during:** Task 4 implementation
- **Issue:** `_setup_stage2` relied on `_setup_stage1` having unfrozen the full classifier. After narrowing stage 1 to only the final head, stage 2 would have left `classifier[0..4]` frozen, never fine-tuning them.
- **Fix:** `_setup_stage2` now also unfreezes `model.classifier.parameters()` in addition to `features[-3:]`.
- **Files:** `src/acoustic/training/efficientat_trainer.py`
- **Commit:** 8225062

**4. [Rule 1 - Bug] test_stage1_freeze assertion updated**
- **Found during:** Task 4 implementation
- **Issue:** The existing `test_efficientat_training.py::TestStage1Freeze::test_stage1_freeze` asserted that the entire classifier was unfrozen in stage 1. After D-33 this is no longer true.
- **Fix:** Test now asserts only `classifier[-1]` (final head) is trainable and that earlier classifier layers (e.g. `Linear(1280, 1280)`) stay frozen.
- **Files:** `tests/unit/test_efficientat_training.py`
- **Commit:** 8225062

## Deferred Issues (Out of Scope)

The following unit-test failures exist on the worktree base and are NOT caused by this plan. They belong to other in-flight Wave 3 plans:

- `tests/unit/test_sliding_window_dataset.py` — 5 failures, all `TypeError: WindowedHFDroneDataset.__init__() got an unexpected keyword argument 'file_lengths'`. These are Plan 20-03 RED stubs from Wave 0; the `WindowedHFDroneDataset` full implementation is owned by Plan 20-03.
- `tests/unit/test_vertex_submit_phase20.py` — 4 `ImportError`s. These are Plan 20-05 (vertex-submit) RED stubs; owned by Plan 20-05.

All training-trainer tests (`tests/unit/training/` + `tests/unit/test_efficientat_training.py`) — 26 total — pass.

## Verification

- `pytest tests/unit/training/test_specaug_scaling.py tests/unit/training/test_trainer_loss_factory.py tests/unit/training/test_save_gate.py tests/unit/training/test_stage1_unfreeze_scope.py tests/unit/test_efficientat_training.py -q` → **24 passed**
- `pytest tests/unit/training/ tests/unit/test_efficientat_training.py -q` → **26 passed** (full training suite incl. augmentation ordering from 20-04)
- Grep acceptance checks:
  - D-30: `freqm=cfg.specaug_freq_mask` present, `freqm=48|timem=192` absent ✓
  - D-31: `build_loss_function(` present, `nn.BCEWithLogitsLoss()` absent ✓
  - D-32: `save_gate_min_accuracy` and `save gate blocked` present ✓
  - D-33: `model.classifier[-1]` unfreeze present; `model.classifier.parameters()` in stage 1 absent ✓

## Threat Flags

None. All surface changes (new config fields, new loss factory call, new save gate, narrower unfreeze scope) are covered by the plan's `<threat_model>` section. No new network endpoints, auth paths, file access patterns, or trust boundaries introduced.

## Self-Check: PASSED

- `src/acoustic/training/efficientat_trainer.py` — present (modified) ✓
- `src/acoustic/training/config.py` — present (modified) ✓
- `tests/unit/test_efficientat_training.py` — present (modified) ✓
- `tests/unit/training/test_specaug_scaling.py` — present (created) ✓
- `tests/unit/training/test_trainer_loss_factory.py` — present (created) ✓
- `tests/unit/training/test_save_gate.py` — present (created) ✓
- `tests/unit/training/test_stage1_unfreeze_scope.py` — present (created) ✓
- Commits 567afe7, 3d04c2a, 1d949d6, 3b0208a, 4ee0ae6, df68a4b, 992cf9b, 8225062 — all present in `git log` ✓
