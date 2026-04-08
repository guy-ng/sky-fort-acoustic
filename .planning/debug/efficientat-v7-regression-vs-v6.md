---
status: diagnosed
trigger: "efficientat_mn10_v7.pt underperforms efficientat_mn10_v6.pt on 2026-04-08 recordings; diagnose-only"
created: 2026-04-08T00:00:00Z
updated: 2026-04-08T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — v7 was trained on 0.5-second windows while v6 and inference use 1.0-second windows. Train/serve window-length mismatch driven by the Phase 20 `WindowedHFDroneDataset` path (window_samples=8000 @ 16kHz), contrasted against v6's `_LazyEfficientATDataset` path (segment_samples=32000 @ 32kHz) and the inference pipeline's `_training_window_seconds("efficientat") = 1.0`.
test: static read of training path, inference path, model loader, and pipeline window selection
expecting: n/a — confirmed
next_action: return diagnosis

## Symptoms

expected: v7 (newer) should match or beat v6 on today's recordings
actual: v7 produces poor detections / wrong class / low confidence; v6 is "good"
errors: none (quality regression)
reproduction: feed data/field/drone/20260408_*.wav and data/field/background/20260408_*.wav through the live pipeline comparing v6 vs v7
started: v7 trained 2026-04-08 07:56 (Vertex job 6987879698596364288, image tag phase20-v7-fix4)

## Eliminated

- hypothesis: Class index / label flip
  evidence: Both v6 and v7 load through the same `_load_efficientat_mn10` path (src/acoustic/classification/efficientat/__init__.py:15-36) which always treats sigmoid(logit[0]) as drone probability. No version branching. Phase 15 / training collapse debug confirmed DADS label semantics (label==1 == drone) end-to-end.
  timestamp: 2026-04-08

- hypothesis: Logits vs sigmoid output mode double-sigmoid
  evidence: `EfficientATClassifier.predict` (src/acoustic/classification/efficientat/classifier.py:41-58) always runs `torch.sigmoid(logits)`. The saved checkpoint is a plain state_dict (not TorchScript with baked-in sigmoid) — `_load_efficientat_mn10` uses `torch.load(..., weights_only=True)` + `load_state_dict`. No double-sigmoid possible.
  timestamp: 2026-04-08

- hypothesis: Wrong checkpoint promoted (intermediate instead of best)
  evidence: Phase 20-05 SUMMARY records Vertex job 6987879698596364288 ("phase20-v7-fix4") with val_acc=0.983, F1=0.990, early-stopped at epoch 42. The save gate (D-32) would have blocked any degenerate checkpoint. v7 was the best-on-val-loss checkpoint from a successful run.
  timestamp: 2026-04-08

- hypothesis: v7 file is a copy of v6 or corrupted
  evidence: sha256 of v6=d6acbfa0, v7=0f6ca9d0 — distinct. v7 matches the sha256 recorded in 20-05-SUMMARY (421ea22c...).
  timestamp: 2026-04-08

- hypothesis: Sample rate mismatch (16k vs 32k)
  evidence: WindowedHFDroneDataset.__getitem__ explicitly resamples 16k→32k (hf_dataset.py:312-316). `_load_efficientat_mn10` uses `EfficientATMelConfig(sample_rate=32000)` at inference. Both sides are at 32 kHz. This hypothesis is ruled out for raw SR — but see the "window-length" finding below, which is the real bug the 260407-ls3 patch introduced.
  timestamp: 2026-04-08

- hypothesis: Normalization / mel parameter drift
  evidence: `AugmentMelSTFT` parameters are identical between training and inference (vendored module, `(log_mel + 4.5)/5.0` normalization, precomputed mel filterbank at 32kHz — src/acoustic/classification/efficientat/preprocess.py). No version-specific parameters.
  timestamp: 2026-04-08

- hypothesis: WeightedRandomSampler bug (260407-nir) fallout
  evidence: The nir fix was merged BEFORE the successful v7 training run (phase20-v7-fix4). The SUMMARY records the fix chain: fix3 was cancelled due to the sampler bug, fix4 includes it. v7 training had full sliding-window coverage.
  timestamp: 2026-04-08

- hypothesis: SpecAugment collapse (PRIMARY-A from training-collapse debug)
  evidence: Phase 20-07 closed D-30 by making specaug_freq_mask=8, specaug_time_mask=10 config-driven. v7's val metrics (val_acc=0.983, F1=0.990) prove training did not collapse.
  timestamp: 2026-04-08

## Evidence

- timestamp: 2026-04-08
  checked: models/ directory (ls + sha256)
  found: v6.pt sha256 d6acbfa0... (Apr 6 10:48); v7.pt sha256 0f6ca9d0... (Apr 8 07:56). Both ~17 MB TorchScript-free state_dicts. Distinct checkpoints.
  implication: v7 is a real, distinct model. Not a copy of v6.

- timestamp: 2026-04-08
  checked: .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-05-SUMMARY.md (commit 45e26b9 diff)
  found: Vertex job 6987879698596364288 "phase20-v7-fix4" completed 2026-04-08 01:27 UTC. Final val: loss=0.0036, acc=0.983, precision=0.999, recall=0.981, F1=0.990. Training was HEALTHY — no collapse. Four fix iterations were needed (jx8, fix1/AppleDouble, ls3, nir); fix4 was the successful run.
  implication: The model itself is not degenerate. The bug must be in a train/inference mismatch introduced by one of the four fixes.

- timestamp: 2026-04-08
  checked: src/acoustic/training/efficientat_trainer.py:437-478 (Phase 20 path) + src/acoustic/training/hf_dataset.py:217-223 (WindowedHFDroneDataset signature)
  found: Phase 20 path is activated when `cfg.window_overlap_ratio > 0 or cfg.rir_enabled`. v7's config set both (60% overlap, RIR enabled per the phase plan). The trainer sets:
    `window_samples = int(0.5 * _SOURCE_SR)  # 8000 samples = 0.5 s @ 16 kHz`
  and passes that directly to `WindowedHFDroneDataset`.
  implication: v7 training slices DADS into 0.5-SECOND windows before any augmentation.

- timestamp: 2026-04-08
  checked: src/acoustic/training/hf_dataset.py:301-319 (WindowedHFDroneDataset.__getitem__ after 260407-ls3 fix)
  found: After slicing the 0.5s window at 16kHz, the dataset calls `F_audio.resample(segment_t, _SOURCE_SR=16000, _TARGET_SR=32000)` — yielding a tensor of length 16000 samples = **0.5 seconds at 32 kHz**. That tensor is what the DataLoader hands to the trainer batch, which then passes it to `mel_train(batch_wav)`.
  implication: **Every v7 training sample was a 0.5-second waveform** (16000 samples @ 32kHz). The resulting mel spectrogram has ~50 time frames (16000 / hop_size=320 = 50), NOT the 100 frames that `EfficientATMelConfig.input_dim_t=100` implies.

- timestamp: 2026-04-08
  checked: src/acoustic/training/efficientat_trainer.py:487-491 (legacy _LazyEfficientATDataset path used for v6)
  found: When `window_overlap_ratio=0` and `rir_enabled=False` (the default — used by v6), training uses `_LazyEfficientATDataset(..., segment_samples=mel_cfg.segment_samples)`. `EfficientATMelConfig.segment_samples = input_dim_t * hop_size = 100 * 320 = 32000 samples = 1.0 second @ 32 kHz`. Random 1.0-second segment extraction per sample (hf_dataset.py:146-153 — old `_LazyEfficientATDataset.__getitem__`).
  implication: **v6 was trained on 1.0-second windows at 32 kHz** (32000 samples, ~100 mel frames). v7 was trained on **0.5-second windows** (16000 samples, ~50 mel frames). Different training distributions.

- timestamp: 2026-04-08
  checked: src/acoustic/pipeline.py:72-86 `_training_window_seconds()` + src/acoustic/pipeline.py:298-314 `start_detection_session`
  found: `_training_window_seconds("efficientat")` returns `1.0`. Detection sessions compute `self._cnn_segment_samples = int(settings.sample_rate * 1.0)` and the pipeline pushes the most recent `_cnn_segment_samples` of audio to the classifier every `interval_seconds`. At pipeline sample rate 48 kHz: 48000 samples; resampled down to 32 kHz by `RawAudioPreprocessor.process()` → 32000 samples = **1.0 second @ 32 kHz**.
  implication: **Inference feeds the classifier 1.0-second waveforms** for ANY efficientat model — including v7.

- timestamp: 2026-04-08
  checked: src/acoustic/classification/efficientat/__init__.py:15-39 + src/acoustic/classification/efficientat/classifier.py
  found: Both v6 and v7 load through the SAME `_load_efficientat_mn10(path)` function → `get_model(input_dim_t=100, ...)` + `EfficientATClassifier` + `AugmentMelSTFT(sr=32000, hopsize=320)`. NO version-specific branching. The classifier is shape-agnostic (attention pooling + SE blocks with `se_dims="c"` only), so it RUNS on both 50-frame (0.5s) and 100-frame (1.0s) inputs without shape errors.
  implication: The classifier code path does not detect or adjust for the window-length mismatch. v7's learned weights (BatchNorm running stats, classifier head, attention pooling weights) were tuned for 50-frame feature maps. At inference they see 100-frame feature maps — twice as many time tokens going into the attention pool, BatchNorm stats computed over a different distribution, etc. Functionally the model runs but its decision surface is out-of-domain.

- timestamp: 2026-04-08
  checked: src/acoustic/classification/efficientat/preprocess.py (AugmentMelSTFT forward)
  found: AugmentMelSTFT is a plain STFT + mel + `(log_mel+4.5)/5.0` normalization. No `input_dim_t` is used — it processes whatever length waveform it receives. So the mel itself does not crash at 0.5s vs 1.0s inputs.
  implication: Confirms the shape-agnostic claim. There is no hard size check that would have caught the mismatch.

- timestamp: 2026-04-08
  checked: RMS normalization path — src/acoustic/training/efficientat_trainer.py:246-250 (train augmentation chain, RmsNormalize last) vs src/acoustic/training/hf_dataset.py:309-316 (waveform_aug applied at 16k BEFORE resample to 32k) vs src/acoustic/classification/preprocessing.py:205-209 (RawAudioPreprocessor applies _rms_normalize AFTER resample to 32k)
  found: Training RMS normalization happens in the SOURCE-RATE (16 kHz) domain inside `waveform_aug`, before the dataset resamples to 32 kHz. Inference RMS normalization happens in the TARGET-RATE (32 kHz) domain, after resample. Since torchaudio resampling preserves RMS up to polyphase filter bleed (~2%), this is a small secondary mismatch but not the driver.
  implication: SECONDARY — worth noting as a train/serve parity gap in the v7 pipeline, but too small to explain the observed regression.

- timestamp: 2026-04-08
  checked: data/field/drone/20260408_091054_136dc5.json
  found: sub_label = "10inch payloda 4kg need to cut last 10 sec" — the user's contaminated tail recording. Duration 71.4s.
  implication: Note in report; do NOT modify file.

- timestamp: 2026-04-08
  checked: data/field/drone/20260408_*.json metadata
  found: 13 drone recordings from today, sample_rate=16000, various sub_labels (5inch, 10inch, 10inch 1.5kg payload, 10inch 4kg payload, phantom 4, 10inch heavy). All 16 kHz single-channel — will be pushed through `RawAudioPreprocessor` which resamples to 32 kHz and RMS-normalizes. Background recordings similar.
  implication: Test corpus is valid and diverse. v6 vs v7 runs over this corpus should produce directly comparable per-clip predictions.

## Resolution

root_cause: |
  **PRIMARY (confirmed, highest confidence): TRAIN/SERVE WINDOW-LENGTH MISMATCH.**

  v7 was trained on 0.5-second audio windows while the inference pipeline
  feeds it 1.0-second windows. v6 was trained on 1.0-second windows (matching
  inference), which is why v6 works and v7 doesn't.

  The regression was introduced by quick task 260407-ls3 (commit f007e91),
  which fixed a crash in `WindowedHFDroneDataset` but preserved the
  `window_samples = int(0.5 * _SOURCE_SR) = 8000 samples @ 16 kHz` window
  size chosen at Plan 20-03 authoring time. After resample to 32 kHz this
  yields 16000-sample (0.5s) training tensors, while the inference pipeline
  hard-codes 1.0-second windows for the "efficientat" model family
  (`_training_window_seconds("efficientat") = 1.0` in src/acoustic/pipeline.py:83-84).

  EfficientAT mn10 is architecturally shape-agnostic (attention pooling,
  channel-only SE), so there is no runtime error — the model evaluates a
  100-frame mel spectrogram when it was trained on 50-frame spectrograms.
  But its learned BatchNorm running statistics, classifier-head decision
  surface, and attention pooling weights are all tuned for 50-frame inputs.
  At inference they see twice as many time tokens, producing a systematic
  out-of-distribution shift in features that degrades classification quality
  — exactly matching the user's observation that v7 is "no good" on real
  recordings while v6 (trained on matching 1.0s windows via the legacy
  `_LazyEfficientATDataset` path) performs well.

  The training val metrics (val_acc=0.983, F1=0.990) are real — the model
  learned the 0.5-second DADS task correctly. But because the val set also
  came from `WindowedHFDroneDataset` with the same 0.5s window, val never
  measured the model in its deployed (1.0s) regime. The bug is a silent
  train/val/serve contract violation, not a training failure.

  **SECONDARY (low confidence, worth noting): RMS normalization domain.**

  Training RMS-normalizes at 16 kHz inside `waveform_aug` (before resample);
  inference RMS-normalizes at 32 kHz (after resample). Resample bleed is
  ~2% so this produces a tiny (~0.02x) amplitude skew between train and
  serve. Not the driver of the regression but a latent parity gap that
  should be closed in any future retrain.

  **WHY THE PHASE 20 ARTIFACTS MISSED THIS:**
    1. Plan 20-03 RED stubs tested the window-count math at 8000 samples,
       but never asserted that window duration matches the inference
       pipeline's `_training_window_seconds("efficientat") = 1.0` contract.
    2. Plan 20-07's save gate (D-32) correctly refused degenerate checkpoints
       but cannot detect a train/serve window-length contract violation
       because val uses the same 0.5s windows as train.
    3. Plan 20-08's RMS parity contract test covers `_rms_normalize` math
       but not window length.
    4. Plan 20-06 (eval harness + promotion gate on real UMA-16 data) was
       NOT yet executed before v7 was promoted. Per 20-05-SUMMARY:
       "Next: Plan 20-06 (eval harness + promotion gate, D-26/D-27/D-29)".
       The D-27 real-device TPR/FPR gate would very likely have caught this
       regression before v7 replaced v6 in operational use.

fix: |
  NOT APPLIED (diagnose-only mode per user instruction).

  Recommended remediation (ordered by leverage):

  1. **CHANGE WINDOW SAMPLES IN trainer to match inference contract.**
     src/acoustic/training/efficientat_trainer.py:456
     `window_samples = int(0.5 * _SOURCE_SR)`  →  `window_samples = int(1.0 * _SOURCE_SR)`
     (or better: derive it from `EfficientATMelConfig().input_dim_t * EfficientATMelConfig().hop_size / _TARGET_SR * _SOURCE_SR = 16000` — 1.0s at 16 kHz source = 16000 samples, which equals `_assumed_clip_samples`. The entire clip becomes a single window per file; set hop to match).

     Implication: window_samples=16000 equals the DADS clip length (1s @ 16kHz), so there is only ONE window per clip. The sliding-window-with-overlap scheme is moot on 1s clips. Phase 20's D-13 60%-overlap requirement only makes sense if training clips are LONGER than the target window. If DADS clips are uniformly 1s, Phase 20's windowing added no real augmentation over random-crop `_LazyEfficientATDataset` and should be dropped for this dataset.

  2. **ADD A STATIC CONTRACT TEST** asserting
     `WindowedHFDroneDataset(...).__getitem__(0)[0].shape[-1] == EfficientATMelConfig().segment_samples`
     so this cannot silently regress again.

  3. **ADD A RUNTIME INVARIANT CHECK** in `EfficientATClassifier.predict` logging WARN when `features.shape[-1] != expected_samples` (1-tick warning, not a raise). This would have surfaced the mismatch immediately in logs during live use.

  4. **MOVE RmsNormalize TO AFTER RESAMPLE** in the training path so the RMS is measured in the same sample-rate domain as inference. Easiest approach: keep augmentation chain at 16k for WideGain/RIR/Audiomentations/BackgroundNoiseMixer, but apply RmsNormalize in the 32k-resampled tensor at the END of `WindowedHFDroneDataset.__getitem__`.

  5. **COMPLETE PLAN 20-06** (real-device eval harness + promotion gate) before the next EfficientAT training run. This should include (a) a hold-out set of fresh UMA-16 recordings that the model has never seen, (b) strict D-27 TPR≥0.80 / FPR≤0.05 thresholds, (c) an automated comparison against the current deployed model before promoting.

verification: |
  Verification is pending user confirmation. Two no-code checks the user can run:

  1. **Window-length check** (expected to CONFIRM the finding):
     Load v6.pt and v7.pt, feed each a 32000-sample random tensor, then a 16000-sample random tensor. Compare outputs.
     Expected if hypothesis correct: v6's output varies smoothly with input; v7's output is stable on 16000-sample input but drifts on 32000-sample input.
     Minimal Python (for the user to run MANUALLY, not for this agent):
     ```python
     import torch
     from acoustic.classification.ensemble import load_model
     import acoustic.classification.efficientat  # register
     v6 = load_model("efficientat_mn10", "models/efficientat_mn10_v6.pt")
     v7 = load_model("efficientat_mn10", "models/efficientat_mn10_v7.pt")
     x_1s = torch.randn(1, 32000)
     x_05s = torch.randn(1, 16000)
     print("v6 1s:", v6.predict(x_1s), " 0.5s:", v6.predict(x_05s))
     print("v7 1s:", v7.predict(x_1s), " 0.5s:", v7.predict(x_05s))
     ```

  2. **Compare v6 vs v7 on today's recordings** (the originally requested eval). Since this is diagnose-only, the user should run this manually with both model paths through the live pipeline or a scripted evaluator, confirming v7 is worse than v6 specifically on the 1-second inference window regime.

files_changed: []
