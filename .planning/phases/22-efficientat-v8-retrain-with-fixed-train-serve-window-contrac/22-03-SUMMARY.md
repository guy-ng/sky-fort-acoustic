---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 03
subsystem: efficientat-train-serve-guardrails
tags: [efficientat, v8, guardrails, rms-parity, length-contract, warn-on-mismatch]

# Dependency graph
requires:
  - phase: 22
    plan: 02
    provides: window_contract module (EFFICIENTAT_SEGMENT_SAMPLES + source_window_samples)
provides:
  - src/acoustic/training/hf_dataset.py — fail-loud 32000-sample contract assert, post_resample_norm hook, per_file_lengths for multi-second clips
  - src/acoustic/classification/efficientat/classifier.py — one-shot length-mismatch WARN (never raises)
  - src/acoustic/training/efficientat_trainer.py — RmsNormalize moved out of 16 kHz chain into post_resample_norm (32 kHz train/serve parity)
  - 3 previously-xfailed Wave 0 guardrail tests now green (17 asserts total)
affects: [22-04, 22-05, 22-06, 22-07, 22-08, 22-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - fail-loud dataset length assertion
    - one-shot per-instance WARN via self._warned_mismatch flag
    - post-resample normalize hook (sample-rate domain parity)
    - per_file_lengths generalization for heterogeneous clip durations

key-files:
  created:
    - (none — all modifications to existing files)
  modified:
    - src/acoustic/training/hf_dataset.py
    - src/acoustic/classification/efficientat/classifier.py
    - src/acoustic/training/efficientat_trainer.py
    - tests/unit/training/test_windowed_dataset_length.py
    - tests/unit/classification/test_efficientat_predict_warn.py
    - tests/unit/training/test_rmsnormalize_parity.py
    - tests/unit/training/test_rms_normalize_augmentation.py  (D-34 contract update)
    - tests/unit/training/test_trainer_augmentation_order.py  (D-34 contract update)

decisions:
  - Classifier WARN is a signal, NOT a gate — predict() still runs the model on out-of-domain input so live detection keeps flowing; the warning makes drift operator-visible.
  - D-34 "RMS last in chain" contract is preserved semantically, not literally — RMS is still the last waveform-domain op, but it now lives on the dataset (32 kHz domain) rather than the 16 kHz ComposedAugmentation chain.
  - ``_LazyEfficientATDataset`` intentionally left untouched — Plan 22-06 will decide DADS vs field-data wiring (ConcatDataset).
  - Eval augmentation now returns ``None`` when noise is disabled (previously returned a single-stage ``RmsNormalize`` chain). The dataset's ``post_resample_norm`` handles the RMS step in both cases.
  - One-shot WARN flag is per-instance (``self._warned_mismatch``), not module-global, so test isolation works and separate classifiers each get their own signal.

metrics:
  duration: 20m
  tasks_completed: 3
  files_modified: 8
  tests_added_or_unxfailed: 17
  completed: 2026-04-09
---

# Phase 22 Plan 03: Dataset + Classifier + Trainer Guardrails Summary

## One-liner

Three redundant v7-regression guardrails landed: dataset length assertion (REQ-22-W2), classifier one-shot length-mismatch WARN (REQ-22-W3), and RmsNormalize moved to 32 kHz post-resample domain for train/serve parity (REQ-22-W4).

## What was built

### Task 1 — `WindowedHFDroneDataset` hardening (`src/acoustic/training/hf_dataset.py`)

Three changes, all surgical and backward-compatible:

1. **Fail-loud length contract assertion in `__getitem__`.** After the 16 → 32 kHz resample, an `assert segment_t.shape[-1] == EFFICIENTAT_SEGMENT_SAMPLES` fires on the first bad item. Assertion message references the "v7 train/serve mismatch signature" so an operator reading the traceback immediately understands what went wrong. This would have caught v7 at the first `__getitem__` call instead of letting it silently train a doomed model for a full 3-stage run.

2. **`post_resample_norm: Callable | None = None` kwarg.** When passed, runs on the 32 kHz tensor `numpy() → normalize → from_numpy()`. Used by the trainer to push `RmsNormalize` into the post-resample domain (Task 3). Default `None` preserves legacy DADS behavior exactly — verified by regression tests.

3. **`per_file_lengths: list[int] | None = None` kwarg.** Generalizes the dataset to heterogeneous clip lengths. When provided, each `(file_idx, clip_len)` pair gets its own per-file sliding-window count: `num_w = max(1, 1 + (clip_len - window_samples) // hop_samples)`. The pad/truncate block in `__getitem__` now honors the per-file clip length via a `{file_idx: clip_len}` lookup. This unblocks Plan 06 (field recording ingest) without touching the DADS uniform path.

Implementation detail: `_labels_cache` stays one entry per WINDOW in both paths, so `_build_weighted_sampler` sees the right item count.

### Task 2 — `EfficientATClassifier.predict` one-shot WARN (`src/acoustic/classification/efficientat/classifier.py`)

Added:
- Module-level `_logger = logging.getLogger(__name__)`
- `self._warned_mismatch = False` in `__init__`
- After `x = x.unsqueeze(0)` but BEFORE `self._mel(x)`, a check:
  ```python
  actual = int(x.shape[-1])
  if actual != EFFICIENTAT_SEGMENT_SAMPLES and not self._warned_mismatch:
      _logger.warning("EfficientAT input length %d != expected %d ... "
                      "this is the v7 regression signature ...", ...)
      self._warned_mismatch = True
  ```
- Crucially: the model STILL runs on mismatched input. No short-circuit, no raise. The WARN is an operator signal, not a gate — killing a live detection pipeline over a soft-drift would be strictly worse than running out-of-domain.
- One-shot per instance (not global) so separate classifiers each log independently and test isolation works.

### Task 3 — `RmsNormalize` moved post-resample (`src/acoustic/training/efficientat_trainer.py`)

`_build_train_augmentation`:
- Dropped the final `augs.append(RmsNormalize(...))` line. The returned `ComposedAugmentation` now ends at `BackgroundNoiseMixer` (or earlier, depending on enabled stages).

`_build_eval_augmentation`:
- Also dropped `RmsNormalize`. When noise mixing is disabled, returns `None` instead of a single-stage `RmsNormalize` chain.

`run()`:
- Both `train_ds` and `val_ds` constructed with a shared `post_norm = RmsNormalize(target=cfg.rms_normalize_target)` passed as `post_resample_norm=post_norm`. This lands every training sample at the configured RMS target in the 32 kHz domain, matching `RawAudioPreprocessor.process` at inference exactly.

Parity test (`test_train_serve_rms_parity_within_1e4`) verifies `max|train_out - serve_out| < 1e-4` on the same synthetic input when both paths run their respective `_rms_normalize` helper.

## Verification

```
$ pytest tests/unit/training/test_windowed_dataset_length.py \
         tests/unit/classification/test_efficientat_predict_warn.py \
         tests/unit/training/test_rmsnormalize_parity.py -v
=========================== 17 passed in 11.54s ===========================

$ pytest tests/unit/training/ tests/unit/classification/ tests/unit/test_efficientat.py -q
=========================== 67 passed in 19.13s ===========================
```

All 3 previously-xfailed Wave 0 guardrail test files are now un-xfailed and green.
Broader training + classification + efficientat unit sweeps (67 tests) all pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking Issue] Updated D-34 tests for new post-resample contract**
- **Found during:** Task 3 (after running the broader test sweep)
- **Issue:** Pre-existing tests enforced the OLD D-34 invariant ("RmsNormalize is the last element of the ComposedAugmentation chain"). Plan 03 explicitly inverts this contract — RMS now runs on the dataset (32 kHz) not the aug chain (16 kHz). Three tests broke:
  - `test_rms_normalize_augmentation.py::test_train_chain_ends_with_rms_normalize`
  - `test_rms_normalize_augmentation.py::test_eval_chain_ends_with_rms_normalize`
  - `test_rms_normalize_augmentation.py::test_eval_chain_has_rms_normalize_even_without_noise`
  - `test_trainer_augmentation_order.py::test_train_chain_order`
  - `test_trainer_augmentation_order.py::test_eval_chain_excludes_rir`
- **Fix:** Rewrote these tests to enforce the NEW contract: train chain must NOT contain RmsNormalize; eval chain returns `None` when noise disabled; trainer wires `RmsNormalize` as `post_resample_norm` on the dataset (verified end-to-end by constructing the dataset and checking output RMS ≈ 0.1).
- **Files modified:** `tests/unit/training/test_rms_normalize_augmentation.py`, `tests/unit/training/test_trainer_augmentation_order.py`
- **Commit:** `5362170`

No other deviations. All 3 plan tasks landed exactly as described, and the plan's own success criteria ("3 previously-xfailed Wave 0 tests turn green") is satisfied.

## Threat Flags

No new threat surface introduced. Plan 03 strengthens existing mitigations in the phase's `<threat_model>`:
- `T-22-01 (length contract)`: dataset-level fail-loud assertion now active
- `T-22-03 (silent regression)`: classifier WARN now active in live pipeline
- `T-22-01 sub (train/serve amplitude skew)`: eliminated via 32 kHz RMS parity

## Commits

| Task | Commit | Scope |
|------|--------|-------|
| Task 1 — WindowedHFDroneDataset hardening | `533b88b` | fail-loud assert, post_resample_norm, per_file_lengths |
| Task 2 — classifier WARN | `47e41e0` | one-shot length-mismatch WARN (never raises) |
| Task 3 — RmsNormalize post-resample | `5362170` | trainer wiring + D-34 test updates |

## Self-Check: PASSED
- Task 1 commit `533b88b`: FOUND
- Task 2 commit `47e41e0`: FOUND
- Task 3 commit `5362170`: FOUND
- `src/acoustic/training/hf_dataset.py`: FOUND (modified)
- `src/acoustic/classification/efficientat/classifier.py`: FOUND (modified)
- `src/acoustic/training/efficientat_trainer.py`: FOUND (modified)
- 17/17 Wave 0 guardrail tests: PASSED
- 67/67 broader unit sweep: PASSED
