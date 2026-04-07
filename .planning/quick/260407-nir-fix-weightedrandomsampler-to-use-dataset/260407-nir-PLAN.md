---
quick_id: 260407-nir
description: Fix WeightedRandomSampler to use dataset-level labels for sliding-window path
date: 2026-04-07
mode: quick
---

# Quick Task 260407-nir: WeightedRandomSampler must cover all windows for Phase 20 v7

## Problem

[efficientat_trainer.py:481-484](src/acoustic/training/efficientat_trainer.py#L481) builds
the training sampler from `train_lbl`, which is a **file-level** label list
(length ≈ `len(train_files)` ≈ 153k for DADS train split at 85/7.5/7.5):

```python
sampler = WeightedRandomSampler(
    [1.0 / max(1, train_lbl.count(l)) for l in train_lbl],
    num_samples=len(train_lbl), replacement=True,
)
```

For the Phase 20 v7 sliding-window path, `train_ds` is a
`WindowedHFDroneDataset` with `len(train_ds) ≈ 459k` (3 windows per file at
`window_samples=8000`, `hop_samples=3200`, `assumed_clip_samples=16000`).
Because `num_samples` is capped at the file count, every epoch draws ~153k
windows out of the 459k available — **2/3 of the sliding-window dataset is
unused every epoch**, and the D-13/D-14 overlap augmentation is effectively
wasted.

This was observed on Vertex job `6241595853509754880` (image `phase20-v7-fix3`):
each epoch ran 2254 batches at `batch_size=64` = 144,256 samples, not the
~7200 batches we'd expect at 3× window coverage. The run was cancelled at
quick task `260407-s2` (user instruction) so we can retrain with the fix.

The bug also has a minor secondary cost: `train_lbl.count(l)` inside a list
comprehension is O(n²) at 153k labels — not huge (a few seconds) but
gratuitous when `collections.Counter` is right there.

## Goal

Every epoch must draw `len(train_ds)` samples from the training dataset, with
inverse-frequency class weighting computed over the **same** label list that
indexes the dataset (window-level for `WindowedHFDroneDataset`, file-level
for `_LazyEfficientATDataset` and `_EfficientATDataset`). No other behavior
should change.

## Approach

All three training dataset classes expose a `.labels` property that returns
the list of labels indexable with `__getitem__`:

- [`_EfficientATDataset.labels`](src/acoustic/training/efficientat_trainer.py#L82) → `self._labels` (file-level)
- [`_LazyEfficientATDataset.labels`](src/acoustic/training/efficientat_trainer.py#L135) → `self._labels` (file-level)
- [`WindowedHFDroneDataset.labels`](src/acoustic/training/hf_dataset.py#L320) → `self._labels_cache` (window-level)

Refactor the sampler construction to read labels directly from `train_ds.labels`
so it automatically matches `len(train_ds)` for every path. Use
`collections.Counter` for O(n) class-count computation.

```python
from collections import Counter
# ...
dataset_labels = train_ds.labels
label_counts = Counter(dataset_labels)
weights = [1.0 / max(1, label_counts[l]) for l in dataset_labels]
sampler = WeightedRandomSampler(
    weights,
    num_samples=len(train_ds),
    replacement=True,
)
```

Legacy non-windowed paths keep working unchanged:
- For `_EfficientATDataset` and `_LazyEfficientATDataset`, `train_ds.labels ==
  train_lbl`, so `len(train_ds.labels) == len(train_lbl)` and the weights
  array is identical element-for-element. `num_samples=len(train_ds)` equals
  `len(train_lbl)` in those paths. Zero behavior change.
- For `WindowedHFDroneDataset`, weights are now per-window (not per-file) and
  `num_samples` grows from 153k to 459k. Each epoch covers the full sliding
  window set — the whole point of D-13.

## must_haves

- `efficientat_trainer.py` sampler is built from `train_ds.labels` and
  `num_samples=len(train_ds)`.
- `collections.Counter` replaces the O(n²) `list.count()` inside the weights
  comprehension.
- Legacy non-windowed paths (`_EfficientATDataset`, `_LazyEfficientATDataset`)
  produce an identical sampler (same weights per index, same num_samples)
  — verified by a unit test.
- Windowed path (`WindowedHFDroneDataset`) produces a sampler with
  `num_samples == len(train_ds)` — verified by a unit test.
- `python -c "from acoustic.training.efficientat_trainer import EfficientATTrainingRunner"` still imports cleanly.
- Existing `tests/unit/test_windowed_dataset_non_uniform.py` continues to pass.

## Tasks

### Task 1: Refactor sampler construction in `efficientat_trainer.py`

- **files**: `src/acoustic/training/efficientat_trainer.py`
- **action**:
  - Add `from collections import Counter` near the top of the file
    (alongside existing stdlib imports — `logging`, `os`, `random`,
    `threading`).
  - In the sampler block at lines 481-484, replace the list comprehension and
    the `num_samples=len(train_lbl)` literal with:
    ```python
    # Use dataset-level labels so the sampler covers every __getitem__ index.
    # Windowed path: len(train_ds) is window-count, which is >> file-count.
    # Legacy paths: len(train_ds) == len(train_lbl), so this is a no-op.
    dataset_labels = train_ds.labels
    label_counts = Counter(dataset_labels)
    sampler = WeightedRandomSampler(
        [1.0 / max(1, label_counts[l]) for l in dataset_labels],
        num_samples=len(train_ds),
        replacement=True,
    )
    ```
  - Leave `train_lbl` in place as a file-level local — it's used elsewhere
    (dataset construction logging, etc.).
- **verify**:
  - `python -c "from acoustic.training.efficientat_trainer import EfficientATTrainingRunner; print('ok')"`
- **done**: Sampler construction reads from `train_ds.labels` and `num_samples=len(train_ds)`.

### Task 2: Unit test for sampler correctness across all three dataset paths

- **files**: `tests/unit/test_efficientat_sampler.py` (new)
- **action**: Create a test file with two tests:
  1. **`test_sampler_covers_all_windows_for_windowed_dataset`**:
     - Build a fake HF dataset with e.g. 10 files using the same monkeypatch
       pattern as [test_windowed_dataset_non_uniform.py](tests/unit/test_windowed_dataset_non_uniform.py)
       (module-level `decode_wav_bytes` patch returning 16000-sample numpy arrays).
     - Construct `WindowedHFDroneDataset` with `file_indices=[0..9]`,
       `window_samples=8000`, `hop_samples=3200` → expect `len == 30`.
     - Reproduce the sampler construction from the patched
       `efficientat_trainer.py` (or factor it out into a helper and call it
       — see task design note below).
     - Assert `sampler.num_samples == 30`.
     - Assert `len(sampler.weights) == 30`.
     - Assert that each weight is `1.0 / label_counts[label]` for the
       matching window.
  2. **`test_sampler_is_noop_for_legacy_file_dataset`**:
     - Build a `_LazyEfficientATDataset` or mock with a known labels list
       (5 files, labels `[0,1,0,1,0]`).
     - Reproduce the sampler construction.
     - Assert `sampler.num_samples == 5`.
     - Assert the weights equal what the old
       `[1.0 / max(1, train_lbl.count(l)) for l in train_lbl]` expression
       would have produced — guarding against any drift in the refactor.

  **Design note**: Rather than pasting the sampler-construction block into
  the test (which would drift), extract a tiny private helper
  `_build_weighted_sampler(dataset)` from `efficientat_trainer.py` and call
  it from both the trainer and the test. Keep the helper private (`_`).

  Update Task 1 accordingly: add the helper and call it from the trainer.
- **verify**: `python -m pytest tests/unit/test_efficientat_sampler.py -x -q` passes (2/2).
- **done**: Both tests green, helper extracted.

### Task 3: Regression check — existing windowed-dataset test still passes

- **files**: (no new files)
- **action**: Run `python -m pytest tests/unit/test_windowed_dataset_non_uniform.py tests/unit/test_efficientat_sampler.py -x -q` and confirm all tests green.
- **verify**: 4/4 passing (2 pre-existing + 2 new).
- **done**: No regressions.

## Notes

- **Operational follow-up (NOT part of this quick task)**: rebuild
  `acoustic-trainer:phase20-v7-fix4` on the builder VM, push to Artifact
  Registry, resubmit a fresh Vertex L4 job. The orchestrator handles this
  outside the GSD workflow. Expected wall-time impact: each epoch becomes
  ~3× longer (~42 min vs ~14 min) because the sampler now draws 3× more
  windows. Total wall-time for 10+15+20 epochs grows from ~8-10 hrs to
  ~24-30 hrs on L4. This is the intended cost of D-13 overlap coverage.
- The cancelled fix3 run (`6241595853509754880`) was transitioning into
  stage 2 with val_acc=0.970 when stopped. That checkpoint is not needed;
  v7 will be retrained from scratch with the correct sampling.
- `test_sliding_window_dataset.py` was deleted in the previous quick task
  (593a65c). No stale stubs to worry about.
