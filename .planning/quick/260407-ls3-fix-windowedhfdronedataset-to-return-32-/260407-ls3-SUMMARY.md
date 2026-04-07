---
quick_id: 260407-ls3
description: Fix WindowedHFDroneDataset to return 32 kHz raw audio for EfficientAT v7
date: 2026-04-07
commit: f007e91
---

# Quick Task 260407-ls3 — Summary

## What changed

- `src/acoustic/training/hf_dataset.py`
  - Added `import torchaudio.functional as F_audio` and module-level
    `_SOURCE_SR = 16000` / `_TARGET_SR = 32000` constants (mirroring the names
    in [efficientat_trainer.py](src/acoustic/training/efficientat_trainer.py#L42)).
  - `WindowedHFDroneDataset.__init__` signature: dropped `mel_config: MelConfig | None`
    and `spec_aug: SpecAugment | None` parameters and the corresponding
    `self._mel_config` / `self._spec_aug` instance fields. Existing call sites
    in [efficientat_trainer.py:439-452](src/acoustic/training/efficientat_trainer.py#L439)
    don't pass either argument, so the change is API-safe.
  - `WindowedHFDroneDataset.__getitem__` no longer computes a mel spectrogram.
    After the existing slice + non-uniform-clip handling + waveform_aug, it now
    converts the segment to a `torch.Tensor`, calls
    `F_audio.resample(segment_t, _SOURCE_SR, _TARGET_SR)`, and returns the
    1D float32 tensor of length 16000 (= 0.5 s × 32 kHz).
  - Updated class docstring to state the new contract and reference this
    quick task. Also added a "Window math" note clarifying that the slice
    happens at 16 kHz and resampling doubles the length.
  - Updated `__getitem__` docstring from "Return one (mel, label)" to
    "Return one (raw_waveform_32k, label)".

- `tests/unit/test_windowed_dataset_non_uniform.py`
  - `test_non_uniform_clips_do_not_crash`: replaced the old
    `assert features.shape == (1, 128, 64)` mel-shape check with assertions
    on a 1D float32 tensor of length 16000, plus an `isfinite` sanity check.
    Renamed the local `features` variable to `waveform` for clarity.
  - `test_uniform_clips_emit_no_warning`: untouched (only iterates).
  - Both tests still construct `WindowedHFDroneDataset` without passing
    `mel_config`/`spec_aug`, so no signature updates were needed there.

## Why this fix

Phase 20 v7 Vertex job `8506941650449203200` (image
`acoustic-trainer:phase20-v7-fix2`) crashed at 12:27:39 UTC, ~30 s into stage
1, with:

```
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d,
but got input of size: [64, 1, 1, 128, 64]
File "/app/src/acoustic/classification/efficientat/preprocess.py", line 84
File "/app/src/acoustic/training/efficientat_trainer.py" → mel = mel_train(batch_wav)
```

The shape `[64, 1, 1, 128, 64]` is a batched mel spectrogram, not raw audio.
`WindowedHFDroneDataset.__getitem__` (Plan 20-03) was returning `(1, 128, 64)`
mel features by mirroring the legacy `HFDroneDataset` contract, but the
EfficientAT trainer (Plan 20-04) feeds dataloader output through its OWN
`AugmentMelSTFT` (which handles mel + spec-aug per-batch on device). The
trainer's `mel_train(batch_wav)` therefore expected raw waveform `(B, T)` and
got a 4D mel instead — double-mel.

The contrast with the working path is stark:

| | `_LazyEfficientATDataset` (works) | `WindowedHFDroneDataset` (broken) |
|---|---|---|
| Decodes 16 kHz audio | ✓ | ✓ |
| Resamples to 32 kHz | ✓ ([line 120](src/acoustic/training/efficientat_trainer.py#L120)) | ✗ |
| Returns raw waveform | ✓ (1D) | ✗ — `(1, 128, 64)` mel |
| Compatible with `mel_train(batch_wav)` | ✓ | ✗ |

This fix makes `WindowedHFDroneDataset` adhere to the same contract as
`_LazyEfficientATDataset`, so both lazy paths feed the trainer the same shape.

## Verification

```
$ .venv/bin/python -m pytest tests/unit/test_windowed_dataset_non_uniform.py -x -q
..                                                                       [100%]
2 passed in 1.77s
```

Module imports cleanly:

```
$ .venv/bin/python -c "from acoustic.training.hf_dataset import WindowedHFDroneDataset; \
                       from acoustic.training.efficientat_trainer import EfficientATTrainingRunner; \
                       print('imports ok')"
imports ok
```

The Phase 20 v7 trainer still constructs `WindowedHFDroneDataset` with only
`hf_dataset`, `file_indices`, `window_samples`, `hop_samples`, and
`waveform_aug` ([efficientat_trainer.py:439-452](src/acoustic/training/efficientat_trainer.py#L439))
— exactly the parameters that survived the signature trim.

## Out of scope / follow-ups

- **`tests/unit/test_sliding_window_dataset.py` is still a broken RED stub.**
  It uses an unrelated `file_lengths=` constructor API that has never matched
  the real class. Plan 20-03 landed it as a stub and never wired it up. It
  is intentionally untouched here — its mere existence in the repo is
  misleading and should be addressed separately (delete it or rewrite it
  against the real API). Recommend a follow-up `/gsd-quick` to rip it out
  or convert it into a real test.
- **No end-to-end integration test for `EfficientATTrainingRunner` over the
  windowed path.** Plans 20-03 and 20-04 both shipped without one, which is
  why this double-mel bug reached production. A future quick task should
  add an in-memory HF dataset stub + 1-batch smoke test for
  `EfficientATTrainingRunner.run()` along the windowed path. Would have
  caught this bug at PR time.
- **Operational follow-up (NOT this commit):** sync the patch to the
  `acoustic-builder` GCP VM, build `acoustic-trainer:phase20-v7-fix3`, push
  to Artifact Registry, resubmit Vertex job. Driven outside the GSD workflow
  by the orchestrator.

## Commit

```
f007e91 fix(quick-260407-ls3): WindowedHFDroneDataset returns 32 kHz raw audio for EfficientAT
```
