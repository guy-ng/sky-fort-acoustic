---
phase: 20
plan: 03
subsystem: training
tags: [sliding-window, dataset, session-level-split, v7, hf-dataset]
requires:
  - 20-00 (Wave 0 test stubs and DADS data)
provides:
  - WindowedHFDroneDataset (deterministic sliding-window enumeration)
  - split_file_indices (session-level train/val/test split helper)
affects:
  - src/acoustic/training/hf_dataset.py
  - src/acoustic/training/parquet_dataset.py
tech_stack:
  added: []
  patterns:
    - file-indexed sliding-window enumeration (Plotz 2021, compass §4)
    - session-level group split for windowed time-series
key_files:
  created: []
  modified:
    - src/acoustic/training/hf_dataset.py
    - src/acoustic/training/parquet_dataset.py
decisions:
  - Append-only: legacy HFDroneDataset and split_indices kept untouched for v5/v6 reproducibility
  - Hard-assert uniform-clip assumption (Research A1) instead of silent fallback
  - Reject duplicate file indices in WindowedHFDroneDataset constructor
metrics:
  tasks_completed: 2
  tasks_total: 2
  duration: ~5min
  completed: 2026-04-06
requirements:
  - D-13
  - D-14
  - D-15
  - D-16
---

# Phase 20 Plan 03: Sliding-Window Dataset Summary

Implements deterministic sliding-window enumeration (D-13/D-14) and session-level
file-index splits (D-15) with non-overlapping test windows (D-16) so v7 training
can inflate effective sample count ~2.5× without leaking adjacent windows from the
same source recording across train/val/test.

## What Changed

### Task 1 — `split_file_indices` helper (commit `6a3e434`)
Appended a new `split_file_indices(num_files, seed, train, val)` function to
`src/acoustic/training/parquet_dataset.py`. Operates on FILE indices (not window
indices), shuffles deterministically with a seeded `random.Random`, and returns
three disjoint lists whose union is exactly `range(num_files)`. Legacy
`split_indices` is left untouched for v6 backward compatibility.

Validation: `split_file_indices(100, seed=42)` → `(70, 15, 15)` disjoint lists.

### Task 2 — `WindowedHFDroneDataset` class (commit `0822496`)
Appended a new `WindowedHFDroneDataset` class to
`src/acoustic/training/hf_dataset.py`. Constructor takes a list of FILE indices
(typically from `split_file_indices`), pre-computes a flat
`_items: list[tuple[int, int]]` mapping each flat index → `(file_idx, offset)`
using:

```
num_windows_per_file = max(1, 1 + (n_samples - window_samples) // hop_samples)
```

For DADS uniform 1 s @ 16 kHz clips:
- train/val: `window=8000, hop=3200` → 3 windows/file (60 % overlap, D-13/D-14)
- test:      `window=8000, hop=8000` → 2 windows/file (no overlap, D-16)

`__getitem__` decodes the WAV bytes via the existing `decode_wav_bytes` helper,
slices the deterministic window, runs waveform augmentation at 16 kHz, computes
mel spectrogram via `mel_spectrogram_from_segment`, applies optional
`SpecAugment`, and returns `(features (1,128,64), label float32)` — identical
downstream tensor shape to `HFDroneDataset` so the trainer is a drop-in
substitution.

Defensive checks (Rule 2 — correctness invariants from the threat register):
- `assert len(set(file_indices)) == len(file_indices)` — duplicate file indices
  silently break session-level isolation, must fail fast.
- Per-row sanity check that decoded clip length matches `assumed_clip_samples`
  — Research A1 fallback hook; loud `AssertionError` beats silent corrupt
  windows.

Legacy `HFDroneDataset` is fully untouched; both classes coexist (`grep -c`
returns 2).

## Verification

- `python -c "from acoustic.training.parquet_dataset import split_file_indices; ..."`
  passes the disjoint/size invariants.
- `python -c "from acoustic.training.hf_dataset import HFDroneDataset, WindowedHFDroneDataset"`
  imports both classes cleanly.
- Window-math sanity: `n_w(16000,8000,3200)==3` and `n_w(16000,8000,8000)==2`.
- All grep acceptance criteria pass:
  - `class WindowedHFDroneDataset` × 1
  - `session-level split` doc comment present
  - `self._items` flat-index list present
  - both `def split_file_indices` and `def split_indices` present in
    parquet_dataset.py

The Wave 0 test file `tests/unit/test_sliding_window_dataset.py` is owned by
Plan 20-00 (parallel sibling) and is not present in this worktree base. The
orchestrator will run the full sliding-window suite — including
`test_no_file_leakage_across_splits`, the explicit D-15 oracle — after merging
all Wave 1 plans on top of Wave 0.

## Deviations from Plan

None — plan executed exactly as written. The plan-suggested per-task `pytest`
verifications were deferred to the post-wave orchestrator run because the test
files live in plan 20-00 (a parallel wave-0 sibling not yet merged into this
worktree base); production code matches the plan's behavior contract literally
so the upcoming test run is the verification gate.

## Known Stubs

None.

## Self-Check: PASSED

- src/acoustic/training/parquet_dataset.py — FOUND, contains `def split_file_indices`
- src/acoustic/training/hf_dataset.py — FOUND, contains `class WindowedHFDroneDataset`
- commit `6a3e434` — FOUND
- commit `0822496` — FOUND
