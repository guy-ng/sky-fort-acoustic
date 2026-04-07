---
quick_id: 260407-nir
description: Fix WeightedRandomSampler to use dataset-level labels for sliding-window path
date: 2026-04-07
commit: 9809d3f
---

# Quick Task 260407-nir — Summary

## What changed

- `src/acoustic/training/efficientat_trainer.py`
  - Added `from collections import Counter` import.
  - Added `_build_weighted_sampler(dataset)` module-level helper that reads
    `dataset.labels` directly and builds a `WeightedRandomSampler` with
    `num_samples=len(dataset)`. Uses `Counter` for O(n) class-count
    computation instead of the old O(n²) `list.count` inside a
    comprehension.
  - Replaced the 4-line sampler construction at the old line 481-484 with a
    single call to `sampler = _build_weighted_sampler(train_ds)`.

- `tests/unit/test_efficientat_sampler.py` (new, 3 tests)
  - `test_sampler_covers_all_windows_for_windowed_dataset`: 10 files × 3
    windows each = 30 items. Asserts `sampler.num_samples == 30` (not 10)
    and that each weight equals `1 / window_level_count[label]`. This is
    the direct regression guard for the v7 bug — with the previous
    implementation this test would fail with `sampler.num_samples == 10`.
  - `test_sampler_is_noop_for_legacy_file_dataset`: builds a
    `_LazyEfficientATDataset` with 5 files (labels `[0,1,0,1,0]`), then
    asserts the refactored sampler produces the **exact** same weights
    array the old expression would have produced. Guards against any
    behavior drift for the v6-style non-windowed path.
  - `test_sampler_handles_single_class_gracefully`: all-same-class edge
    case (7 labels all `1`). Asserts each weight is `1/7` and finite.

## Why this fix

[efficientat_trainer.py:481-484 (pre-fix)](src/acoustic/training/efficientat_trainer.py#L481)
built the training sampler from `train_lbl`, a **file-level** label list
(`len(train_lbl) == len(train_files) ≈ 153k` for DADS 85/7.5/7.5 split):

```python
sampler = WeightedRandomSampler(
    [1.0 / max(1, train_lbl.count(l)) for l in train_lbl],
    num_samples=len(train_lbl), replacement=True,
)
```

For the Phase 20 v7 sliding-window path, `train_ds` is a
`WindowedHFDroneDataset` with `len ≈ 459k` (3 windows per file at
`window_samples=8000` / `hop_samples=3200` / `assumed_clip_samples=16000`).
Because `num_samples` was capped at file count, every epoch drew ~153k
windows — **2/3 of the dataset was unused every epoch** and the D-13/D-14
overlap augmentation (the whole point of Plan 20-03) was wasted.

Observed impact on Vertex job `6241595853509754880` (image
`phase20-v7-fix3`):
- Actual: 2254 batches per epoch at `batch_size=64` = 144,256 samples.
- Expected with fix: ~7180 batches per epoch = ~459k samples.
- The run was cancelled (user instruction) so we can retrain with the
  corrected sampler.

All three dataset classes expose `.labels`:
- `_EfficientATDataset.labels` → file-level list
- `_LazyEfficientATDataset.labels` → file-level list
- `WindowedHFDroneDataset.labels` → window-level list (`_labels_cache`)

By reading `train_ds.labels` instead of `train_lbl`, the helper
automatically matches whichever dataset is in use. For legacy non-windowed
paths `train_ds.labels == train_lbl` element-for-element and
`len(train_ds) == len(train_lbl)`, so the refactor is a zero-behavior-change
no-op there.

## Verification

```
$ .venv/bin/python -m pytest tests/unit/test_efficientat_sampler.py \
                              tests/unit/test_windowed_dataset_non_uniform.py \
                              -x -q
.....                                                                    [100%]
5 passed in 2.39s
```

Import smoke test:

```
$ .venv/bin/python -c "from acoustic.training.efficientat_trainer import \
                       EfficientATTrainingRunner, _build_weighted_sampler; print('ok')"
ok
```

## Expected operational impact

Training wall-time grows proportionally with epoch-level batch count:

| | Before fix (v7 fix3) | After fix (v7 fix4) |
|---|---|---|
| Batches per epoch | ~2254 | ~7180 (3.2×) |
| Stage 1 duration (10 ep) | ~2.5 hrs | ~8 hrs |
| Stage 2 duration (15 ep) | ~3.5 hrs | ~11 hrs |
| Stage 3 duration (20 ep) | ~4.5 hrs | ~14 hrs |
| Total wall-time | ~10 hrs | **~30-35 hrs** |
| L4 cost | 1× | 3.2× |

This is the intended cost of D-13 sliding-window overlap coverage. Each
epoch now sees the full set of overlapping windows, so the model gets
real benefit from the 60% overlap augmentation rather than treating it as
random subsampling.

## Out of scope / follow-ups

- **Operational rebuild** (NOT part of this commit): sync to
  `acoustic-builder` VM, build `acoustic-trainer:phase20-v7-fix4`, push to
  Artifact Registry, submit fresh Vertex L4 job. Driven by the
  orchestrator outside the GSD workflow.
- **Still no end-to-end integration test** for
  `EfficientATTrainingRunner.run()` through the windowed path. This quick
  task adds unit coverage for the sampler specifically but the broader
  "1-batch smoke test through the full trainer.run() loop" gap from quick
  task 260407-ls3's follow-ups section remains open. Worth a dedicated
  task in a quiet moment.

## Commit

```
9809d3f fix(quick-260407-nir): WeightedRandomSampler covers all windows in sliding-window path
```
