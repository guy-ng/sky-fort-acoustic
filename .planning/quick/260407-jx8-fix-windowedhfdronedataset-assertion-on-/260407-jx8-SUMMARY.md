---
quick_id: 260407-jx8
description: Fix WindowedHFDroneDataset assertion on non-uniform DADS clips
date: 2026-04-07
commit: 7ea498a
---

# Quick Task 260407-jx8 — Summary

## What changed

- `src/acoustic/training/hf_dataset.py`
  - Imported `pad_or_loop` at module top (and removed the inlined import in
    `HFDroneDataset.__getitem__`).
  - Added `WindowedHFDroneDataset._warned_non_uniform` instance flag for
    one-shot warning gating.
  - Replaced the hard `AssertionError` in `WindowedHFDroneDataset.__getitem__`
    with a tolerant fallback:
    - First non-uniform encounter logs a `logger.warning` (then suppresses).
    - Short clips → `pad_or_loop(audio, assumed_clip_samples)` so every
      pre-allocated `(file_idx, offset)` window contains real (looped) audio
      instead of zero-padded silence.
    - Long clips → truncate to `assumed_clip_samples` so window offsets stay
      in-bounds.
- `tests/unit/test_windowed_dataset_non_uniform.py` (new)
  - Fake HF dataset stub + monkeypatched `decode_wav_bytes` covering uniform,
    short, and long clip variants.
  - `test_non_uniform_clips_do_not_crash` iterates the full 9-window dataset
    over `[16000, 8000, 24000]`-sample files and asserts each `__getitem__`
    returns a `(1, 128, 64)` tensor + label, with exactly one warning emitted.
  - `test_uniform_clips_emit_no_warning` guards against spurious warnings when
    all clips already match the assumption.

## Why this fix

Phase 20 v7 Vertex training crashed at `hf_dataset.py:264` on
`file_idx=174151` (8000 samples vs the assumed 16000). The "loud failure"
guard from 20-RESEARCH A1 turned a benign data outlier into a hard training
abort. Looping the short clip is consistent with how `HFDroneDataset` already
handles short audio (line 76, `pad_or_loop`) and keeps the pre-computed
`num_w` window count valid without forcing a (very expensive) per-file
length scan at `__init__`.

## Verification

```
$ python -m pytest tests/unit/test_windowed_dataset_non_uniform.py -x -q
..                                                                       [100%]
2 passed in 1.77s
```

Import sanity:

```
$ python -c "from acoustic.training.hf_dataset import WindowedHFDroneDataset; print('ok')"
ok
```

## Out of scope / notes

- `tests/unit/test_sliding_window_dataset.py` is a pre-existing RED stub file
  with an unrelated `file_lengths=` API and is broken on `main` independently
  of this fix. Left untouched.
- This fix is intentionally minimal — we do not pre-scan all 174k+ DADS clips
  to recompute window counts per file. The looping fallback preserves training
  throughput and only "wastes" a tiny number of windows on the rare outliers.
- Phase 20 v7 Vertex training can now resume with the same dataset shard.

## Commit

```
7ea498a fix(quick-260407-jx8): tolerate non-uniform DADS clips in WindowedHFDroneDataset
```
