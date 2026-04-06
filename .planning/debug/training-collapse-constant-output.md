---
status: verified
trigger: "training-collapse-constant-output: EfficientAT trainer producing degenerate constant-output checkpoints across v2/v3/v5/v6"
created: 2026-04-06T00:00:00Z
updated: 2026-04-06T00:00:00Z
---

## Current Focus

hypothesis: PRIMARY ROOT CAUSE = trainer ignores `loss_function` config and hard-codes `nn.BCEWithLogitsLoss()` AND `_setup_stage1` zeroes-out the entire MLP head (including the still-pretrained `Linear(1280, last_channel)` and the brand-new `Linear(last_channel, 1)`) by freezing then unfreezing the WHOLE classifier — but the new head is randomly initialized and the only learning signal goes through it. CONTRIBUTING = (a) AdamW vs Adam, (b) input gain mismatch between train (no gain) and inference (~500x gain via cnn_input_gain), (c) val_loss-based early-stop with no acc tracking lets degenerate "predict majority class" win.
test: All evidence collected via static read of trainer, model, classifier, config, preprocess, hf_dataset
expecting: Multiple root causes; inversion explained by cause #4 (gain/scale mismatch shifting decision boundary at inference)
next_action: Return diagnosis to user — DO NOT FIX

## Symptoms

expected: Bimodal probability outputs on real audio (high for drone, low for background) with reasonable separation, similar to AudioSet-pretrained baseline after fine-tuning.
actual: Constant-output collapse across runs - v3 outputs ~0, v5/v6 output ~1, v2 random, local model has -0.22 separation BUT inverted (drone < background).
errors: None - silent failure, training "completes" successfully.
reproduction: Run EfficientAT trainer on HF DADS dataset -> evaluate checkpoint with live pipeline -> useless outputs.
started: All recent runs (v2-v6 + local). Live pipeline ruled out (mn10.pt baseline produces sane outputs).

## Eliminated

- hypothesis: Live inference pipeline bug
  evidence: Feeding AudioSet-pretrained mn10.pt baseline through same pipeline produces sane outputs
  timestamp: pre-investigation

## Evidence

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py line 357
  found: `criterion = nn.BCEWithLogitsLoss()` is hard-coded. The factory `build_loss_function()` and config fields `loss_function`, `focal_alpha`, `focal_gamma`, `bce_pos_weight` are completely ignored.
  implication: PRIMARY-1. Even though config defaults to `loss_function="focal"`, EfficientAT runs always use plain BCE. This would not by itself cause collapse, but it does mean the user cannot tune or remediate the loss without code changes — and no `pos_weight` is passed regardless of class balance.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py lines 340-343 + 137-142
  found: After `model = get_model(num_classes=527)` and loading `mn10_as.pt` pretrained weights, the trainer does `model.classifier[-1] = nn.Linear(in_features, 1)`. The full MLP head is `[AdaptiveAvgPool2d, Flatten, Linear(1280, last_channel=1280), Hardswish, Dropout, Linear(last_channel, 527)]`. Replacing only `[-1]` keeps `Linear(1280, 1280)` from pretrained weights but creates a fresh randomly-initialized `Linear(1280, 1)`. Then `_setup_stage1` unfreezes `model.classifier.parameters()` — i.e. it ALSO unfreezes the pretrained `Linear(1280, 1280)`. So in stage 1 the trainer is fine-tuning ~1.6M params (1280×1280 + 1280) of the head while everything before is frozen.
  implication: PRIMARY-2. Stage 1 with `lr=1e-3` on a 1.6M-parameter randomly-mixed head is large. With Kaiming init on Linear (the model's `__init__` re-inits with `nn.init.normal_(m.weight, 0, 0.01)` for ALL Linear modules — see model.py:150 — which means **the pretrained head weights get clobbered when the MN class re-runs init at construction**). NOTE: `get_model()` constructs `MN(...)` which runs the `for m in self.modules()` init loop AFTER classifier is built — this re-initializes ALL Linear/Conv2d weights, INCLUDING the classifier head. THEN `model.load_state_dict(state_dict)` is called and reloads pretrained values. So this part is OK as long as load_state_dict succeeds with the unmodified num_classes=527 head BEFORE the head replacement. ✓ Verified: trainer loads state_dict at line 331 BEFORE replacing head at line 342. So the body weights ARE pretrained correctly.

- timestamp: 2026-04-06
  checked: src/acoustic/classification/efficientat/preprocess.py forward()
  found: AugmentMelSTFT uses normalization `(log_mel + 4.5) / 5.0` — a hard-coded statistical shift derived from AudioSet at standard mic levels. This normalization is shared by training (`mel_train`) and inference (`mel_eval`).
  implication: SUPPORTING. Normalization itself is consistent.

- timestamp: 2026-04-06
  checked: src/acoustic/classification/efficientat/classifier.py vs src/acoustic/classification/preprocessing.py RawAudioPreprocessor (lines 104-161)
  found: Inference path = `RawAudioPreprocessor(target_sr=32000, input_gain=cnn_input_gain).process(audio, sr)` produces a waveform that has been **multiplied by `cnn_input_gain`** (default ~500x for UMA-16v2). Then `EfficientATClassifier.predict(features)` runs `AugmentMelSTFT(features)` — i.e. mel of (raw_audio * 500). Training path = waveforms come from `decode_wav_bytes` (int16 PCM ÷ 32768 → range [-1, 1]) at training-data RMS levels (~0.05 per the docstring), then `mel_train(batch_wav)` — NO 500x gain.
  implication: PRIMARY-3 / EXPLAINS INVERSION. Training distribution: raw audio in [-1, 1] with RMS ~0.05. Inference distribution: raw audio multiplied by 500 (or whatever cnn_input_gain is set to in the live pipeline). After log_mel, this becomes a +log(500²) ≈ +12.4 nat ≈ +5.4 in normalized space — i.e. the entire spectrogram is shifted upward by ~5 units of normalized magnitude. The model has NEVER seen inputs that "loud" during training; the head's decision boundary lives in a totally different region. Outputs become near-constant (saturated sigmoid) and which side they saturate to depends on tiny biases that were never trained for that regime. The "local" model showing **inverted** separation (-0.22) is consistent with this: a model that learned drone>background on training-RMS audio, when fed +12dB-loud audio at inference, can have its "is the input loud and structured" features overpower the learned drone-vs-background features so background (which has more broadband content) lights up more than drone (often a tonal, lower-energy signature). The training docstring explicitly says: "EfficientAT training data sits around ~0.05 RMS, a ~500x gap" — meaning RawAudioPreprocessor's 500x gain was calibrated to match training data, but TRAINING ITSELF does not apply this gain, so the calibration target is wrong: it puts inference data at the SAME RMS as training data, which is correct, BUT only if training used the SAME ~0.05 RMS data. **Need to verify the actual RMS of decoded WAVs from the HF DADS dataset** — if HF data is normalized to peak/quieter than 0.05 RMS, the train/inference mismatch returns. Either way, this sample-rate gain mismatch is the highest-confidence explanation for inversion.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py lines 386-498 (val loop + early stopping)
  found: Validation runs every epoch and computes accuracy + confusion matrix. Early stopping uses `EarlyStopping(patience=cfg.patience)` keyed on `avg_val_loss` — saves checkpoint only when val_loss improves. **But no minimum-accuracy gate**: a model that collapses to "predict majority class" will achieve a stable val_loss equal to `-log(p_majority)` and CAN be saved as the "best" checkpoint if that loss is monotonically improving even slightly. With WeightedRandomSampler making training balanced, but val NOT being balanced (uses raw labels in order), `avg_val_loss` is dominated by the majority class.
  implication: SUPPORTING-1. Val loop runs but its quality gate is too weak. A constant-output model with `prob ≈ p_majority` would still produce reasonable val_loss and be saved. The fact that confusion matrix (tp/fp/tn/fn) is sent to the progress callback but NOT used as a gate means the trainer happily saves degenerate checkpoints.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py lines 298-301 (WeightedRandomSampler)
  found: `WeightedRandomSampler([1.0 / max(1, train_lbl.count(l)) for l in train_lbl], num_samples=len(train_lbl), replacement=True)` — class-balanced sampling. CRITICAL ISSUE: `train_lbl.count(l)` is called inside a list comprehension over each sample, which is O(N²). For DADS (~58k rows × 70% train ≈ 40k), that's ~1.6 billion ops just to build the sampler. This is slow but functionally correct — sampling IS balanced.
  implication: NOT root cause but a perf bug. Class imbalance is being correctly addressed at the sampling level. Combined with hard-coded BCE (no pos_weight), this is fine because the sampler does the balancing.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py lines 386-414 (train loop) + AugmentMelSTFT freqm=48, timem=192
  found: SpecAugment is aggressive: `freqm=48` masks 48 of 128 mel bins (37.5%), `timem=192` masks 192 of ~100 time frames (i.e. up to 100% — since `segment_samples / hopsize = 32000/320 = 100` time frames). `torchaudio.transforms.TimeMasking(time_mask_param=192)` will draw a mask length uniformly in [0, min(192, 100)] = [0, 100]. So with worst-case draw, 100% of the time axis can be masked.
  implication: PRIMARY-4. With ~100 time frames per sample, `timem=192` means the model can see a fully time-masked spectrogram on a non-trivial fraction of training batches. When BOTH freq and time masks hit hard, the input becomes pure noise/zeros — the model literally cannot see the drone. This would push the model toward outputting whatever minimizes BCE on a "blind" input, i.e. the marginal class probability — which is CONSTANT. Combined with the WeightedRandomSampler making the marginal 0.5, the model collapses to logit ≈ 0 (sigmoid 0.5) on training, then any tiny bias drift during stage 2/3 fine-tuning pushes it to 0 or 1 on inference. **This is the most likely explanation for the constant-output collapse pattern across versions** — different runs collapse to different constants (≈0 for v3, ≈1 for v5/v6) depending on which way the bias drifted in the last stage.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py line 374 + stage1_lr=1e-3 default
  found: Optimizer is `torch.optim.Adam` (no weight decay, no AdamW). Stage 1 head-only LR = 1e-3, applied to a 1.6M-param head fine-tuning a `Linear(1280, 1)` from scratch. With aggressive SpecAugment hiding signal, Adam at 1e-3 will rapidly push the head toward whatever minimizes loss on the noisy/masked inputs — i.e. toward the marginal.
  implication: SUPPORTING-2. LR is reasonable in normal conditions but pathological when combined with freqm=48/timem=192. The combo is what kills it.

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py lines 280-285 (lazy HF path)
  found: The HF lazy dataset path passes `_LazyEfficientATDataset` to `WeightedRandomSampler`, but the `train_lbl` list used to build the sampler is `[hf_builder.all_labels[i] for i in train_indices]` — list of int. This is consistent. The dataset path returns (waveform, label_float32). No augmentation is applied in the HF path: no `WaveformAugmentation`, no `BackgroundNoiseMixer`, no `AudiomentationsAugmentation`. The trainer ignores cfg.augmentation_enabled, cfg.use_audiomentations, cfg.noise_augmentation_enabled, cfg.wave_*, etc.
  implication: SUPPORTING-3. The trainer uses ONLY SpecAugment (via AugmentMelSTFT freqm/timem) and ignores all the configured waveform augmentations. So config flags about audiomentations / noise mixing have no effect on EfficientAT training. This is a separate latent bug but not the cause of collapse.

- timestamp: 2026-04-06 (empirical)
  checked: scripts/verify_specaug_collapse.py — 512 synthetic 1-second waveforms through AugmentMelSTFT with trainer params vs proposed
  found: Legacy (freqm=48, timem=192) produces >50% time-axis collapse on 75.2% of samples and >90% collapse on 52.0%. Proposed (freqm=8, timem=10) produces 0% collapse on both thresholds.
  implication: PRIMARY-A empirically CONFIRMED. The trainer shows the model near-blank spectrograms on the majority of batches, which fully explains convergence to marginal class probability (constant output).

- timestamp: 2026-04-06 (empirical)
  checked: scripts/verify_rms_domain_mismatch.py — 10 HF DADS via HFDatasetBuilder.load_raw_waveforms + 10 UMA-16 captures in data/test_samples/ through RawAudioPreprocessor(input_gain=500.0)
  found: DADS raw RMS mean=0.1777 (drone clips ~0.25, no-drone ~0.002–0.08, peak-normalized vs not). Live UMA-16 raw RMS mean=0.01856. Live post-gain RMS mean=9.277. Ratio live_post / DADS = 52.21x. Normalized log-mel mean: DADS=0.542, LIVE=1.645, Δ=+1.103.
  implication: PRIMARY-B empirically CONFIRMED. Inference domain shifts features by +1.1 in normalized log-mel — far outside anything seen during training. Bonus finding: DADS drone clips are peak-normalized short clips while DADS no-drone clips are much longer unnormalized recordings → label-correlated amplitude shortcut that interacts pathologically with cnn_input_gain at inference (loud → drone at training, everything is loud at inference → predict drone always).

- timestamp: 2026-04-06
  checked: src/acoustic/training/efficientat_trainer.py line 471
  found: `torch.save(model.state_dict(), str(ckpt_path))` saves the FULL state_dict including the body. On load, `_load_efficientat_mn10` builds `get_model(num_classes=1, ...)` then `model.load_state_dict(state_dict)`. Shapes match because both sides have classifier.5 = Linear(1280, 1). ✓ Round-trip is correct.
  implication: NOT a root cause.

## Empirical Verification

### PRIMARY-A — SpecAugment collapse on 100-frame inputs (CONFIRMED)

- **Date:** 2026-04-06
- **Script:** scripts/verify_specaug_collapse.py
- **Method:** Constructed `AugmentMelSTFT` with trainer params (freqm=48, timem=192)
  vs proposed (freqm=8, timem=10). Fed 512 synthetic 1-second waveforms at
  `EfficientATMelConfig().segment_samples` (32000 samples, input_dim_t=100).
  Measured fraction of time-frames with per-frame stddev ≈ 0 (masked-out frames).
- **Numbers:**
  - Legacy (freqm=48, timem=192) on 100-frame inputs:
    - mean fraction frames masked = **0.743**  (74.3%)
    - max  fraction frames masked = **1.000**  (entire time axis masked is a real event)
    - P(>50% time-frames masked)  = **0.752**
    - P(>90% time-frames masked)  = **0.520**
  - Proposed (freqm=8, timem=10) on same inputs:
    - mean fraction frames masked = 0.044
    - max  fraction frames masked = 0.090
    - P(>50% time-frames masked)  = 0.000
    - P(>90% time-frames masked)  = 0.000
- **Verdict:** **CONFIRMED**. 75% of training batches have >50% of the time axis
  zeroed out; 52% have >90% zeroed out. The model sees mostly-blank spectrograms
  on the majority of its training steps, which trivially explains convergence to
  the marginal class probability (constant output). Proposed params drop the
  collapse rate to 0% of samples.

### PRIMARY-B — Train/inference RMS domain mismatch (CONFIRMED)

- **Date:** 2026-04-06
- **Script:** scripts/verify_rms_domain_mismatch.py
- **Method:** Loaded 10 HF DADS samples via `HFDatasetBuilder.load_raw_waveforms`
  (same path as the trainer). Loaded 10 UMA-16 captures from
  `data/test_samples/{background,drone}_*.wav` (16 kHz single-channel raw
  recordings with RMS ~1e-3 matching the mic docstring). Passed live samples
  through `RawAudioPreprocessor(target_sr=32000, input_gain=settings.cnn_input_gain=500.0)`
  exactly as `main.py:314` does. Computed RMS of raw DADS, raw live, post-gain
  live, and the mean normalized log-mel output of `AugmentMelSTFT(freqm=0, timem=0)`
  on both.
- **Numbers:**
  - DADS raw RMS: min=0.00204, mean=0.1777, max=0.2836
    - (drone clips ~0.21–0.28, background clips ~0.002–0.08 — DADS drones are
      peak-normalized short clips; DADS background is quieter longer recordings)
  - Live UMA-16 raw RMS: min=0.00107, mean=0.01856, max=0.0640
  - Live UMA-16 post-gain RMS (× 500): min=0.534, mean=**9.277**, max=32.02
  - **Ratio = live_post_gain_rms / dads_rms = 9.277 / 0.1777 = 52.21x**
  - Normalized log-mel mean: DADS=**0.542**, LIVE=**1.645**, Δ=**+1.103**
- **Verdict:** **CONFIRMED**. The live pipeline feeds the model inputs ~52x
  louder (RMS) than the training distribution mean, corresponding to a +1.1
  shift in normalized log-mel space — i.e. the entire spectrogram sits in a
  region of feature space the model never saw during training. The 500x
  cnn_input_gain was calibrated against a false target (DADS drone RMS was
  assumed to be ~0.05, actual mean ~0.18 with huge within-class variance), so
  even if the model had learned proper features it would see out-of-domain
  inputs at runtime.
- **Additional finding (incidental, not part of verification scope):** DADS
  drone clips (RMS ~0.25) are peak-normalized short 0.5s clips while DADS
  no-drone clips (RMS ~0.002–0.08) are much longer unnormalized recordings.
  This introduces a **label-correlated amplitude shortcut**: the model can
  trivially learn "if it's loud, it's drone" from DADS alone, then at inference
  where cnn_input_gain pushes everything to the loud regime, it predicts drone
  everywhere — which is consistent with v5/v6 collapsing to ~1. Worth noting
  for Phase 20's eval harness (Plan 20-06) and data-augmentation design.

## Resolution

root_cause: |
  MULTIPLE COMPOUNDING ROOT CAUSES, in order of likely impact on the constant-output collapse:

  PRIMARY-A (most likely cause of CONSTANT OUTPUT): Aggressive SpecAugment fully masks the input.
    src/acoustic/training/efficientat_trainer.py line 350
    `mel_train = AugmentMelSTFT(..., freqm=48, timem=192)`
    With segment_samples=32000 and hop=320, each training sample has only ~100 time frames.
    timem=192 means TimeMasking can mask the entire time axis. freqm=48 masks 37.5% of mel bins.
    On a significant fraction of batches the model sees fully or near-fully masked spectrograms,
    so it learns to predict the marginal class probability (0.5 → either side after stage 2/3 drift).
    This is a copy-paste of the AudioSet hyperparameters (where input_dim_t=1000), but the trainer
    feeds 1-second segments where input_dim_t=100 — 10x shorter, making timem=192 catastrophic.

  PRIMARY-B (most likely cause of INVERTED LABELS in the local model): Train/inference input-gain mismatch.
    src/acoustic/classification/preprocessing.py RawAudioPreprocessor multiplies live audio by `cnn_input_gain`
    (~500x by default per the docstring) BEFORE handing it to AugmentMelSTFT.
    src/acoustic/training/efficientat_trainer.py applies NO such gain — training data goes through
    AugmentMelSTFT at its natural [-1, 1] range from `decode_wav_bytes`.
    Result: at inference, log-mel values are shifted by ~+5 in normalized units. The trained head
    has never seen this region; its decision boundary becomes meaningless and saturated outputs
    can flip (background > drone) for any model that didn't learn a strongly invariant feature.
    The local model's -0.22 separation with INVERTED polarity is the fingerprint of this mismatch:
    if the model had genuinely learned wrong labels, the constant-output runs (v3/v5/v6) would be
    consistently inverted too — but they're constant, not inverted. Inversion only shows up in the
    one model that escaped collapse, exactly because that one had real (but wrong-domain) features
    to invert. This strongly implicates a domain-shift / scaling issue, not a label flip.

  PRIMARY-C (force-multiplier for both above): Hard-coded BCEWithLogitsLoss with no pos_weight, no focal.
    src/acoustic/training/efficientat_trainer.py:357
    `criterion = nn.BCEWithLogitsLoss()` — completely ignores config (`loss_function`, `focal_alpha`,
    `focal_gamma`, `bce_pos_weight`). The factory `build_loss_function()` exists in losses.py but
    is never called. With WeightedRandomSampler this is *survivable* in a healthy regime, but it
    removes the gradient sharpening that focal loss provides on hard examples — and "everything is
    masked to zero" is the ultimate easy example, so the model converges fast to the marginal.

  CONTRIBUTING-D: Validation early-stopping has no behavioral gate.
    src/acoustic/training/efficientat_trainer.py:469-473
    Best checkpoint is selected on `avg_val_loss` only. A constant-output model achieves a flat
    `-log(p_majority)` val_loss which IS monotonically saved as "improved" if it edges down by
    even 1e-6. Confusion matrix (tp/fp/tn/fn) is computed and sent to the progress callback BUT
    NEVER USED as a save/stop criterion. So degenerate checkpoints get saved and ship.

  CONTRIBUTING-E: Configured waveform augmentations are silently dropped on the EfficientAT path.
    src/acoustic/training/efficientat_trainer.py never wires `WaveformAugmentation`,
    `AudiomentationsAugmentation`, or `BackgroundNoiseMixer` into either `_EfficientATDataset` or
    `_LazyEfficientATDataset`. Only SpecAugment runs. So the model never sees noise variation,
    pitch shifts, or gain perturbations during training, making it fragile to the live pipeline's
    exact RMS/spectral envelope. Compounds PRIMARY-B.

  CONTRIBUTING-F: Stage 1 unfreezes the entire MLP head including a brand-new randomly-initialized
    `Linear(1280, 1)` and runs Adam at lr=1e-3 over 1.6M params on top of catastrophically-masked
    inputs. Combined with PRIMARY-A this drives rapid head collapse before stage 2/3 ever start.

  HYPOTHESES RULED OUT (with evidence):
    - "DADS HF labels are flipped between drone/background": refuted by
      src/acoustic/api/test_pipeline_routes.py:100 (`label_str = "drone" if label_val == 1 else "background"`)
      which confirms the dataset's label==1 means drone, AND the trainer maps target=label directly
      without inversion. Labels are correct end-to-end.
    - "Class imbalance with no pos_weight": partially refuted — pos_weight is missing, but
      WeightedRandomSampler IS used and IS class-balanced, so the training distribution is ~50/50.
      Imbalance is not the driver here.
    - "Stage 3 LR too high": refuted — stage3_lr default = 1e-5 with cosine schedule on full model,
      this is a conservative value. Stage 3 doesn't crash a healthy model; it just doesn't fix
      what's already collapsed in stage 1.
    - "Validation never runs": refuted — val loop at lines 432-466 runs every epoch and reports
      val_loss, val_acc, and confusion matrix.
    - "Pretrained weights silently overwritten by re-init": refuted — model construction order is
      `MN.__init__` (which re-inits all Linear/Conv2d) → `load_state_dict` → head replacement.
      Body weights ARE the pretrained ones at training start.

fix: |
  NOT APPLIED (diagnose-only mode). Recommended order of remediation:
    1. Fix PRIMARY-A first: change `mel_train` to `freqm=8, timem=20` (or scale to ~10% of dim).
       This alone may unblock training on the existing setup.
    2. Then verify PRIMARY-B by measuring RMS of decoded HF DADS waveforms vs the live pipeline
       output WITH cnn_input_gain applied. Reconcile: either drop cnn_input_gain to 1.0 and
       calibrate during training, OR apply the same gain during training so domains match.
    3. Wire `build_loss_function()` into the EfficientAT trainer (PRIMARY-C). At minimum allow
       focal loss; ideally pass `bce_pos_weight` from config.
    4. Add an accuracy/F1 gate to checkpoint saving (CONTRIBUTING-D): refuse to save if
       `min(tp, tn) == 0` or if val accuracy is below e.g. 0.55.
    5. Wire waveform augmentations into the EfficientAT dataset path (CONTRIBUTING-E).

verification: |
  Empirical checks run 2026-04-06 (see "Empirical Verification" section above):
    - PRIMARY-A CONFIRMED via scripts/verify_specaug_collapse.py
      (75.2% of batches have >50% time-axis masked; 52% have >90% masked)
    - PRIMARY-B CONFIRMED via scripts/verify_rms_domain_mismatch.py
      (52.2x RMS gap between live post-gain and DADS training;
       +1.1 shift in normalized log-mel mean)
    - Bonus: DADS drone clips are peak-normalized while DADS no-drone clips are
      ~100x quieter, creating a label-correlated amplitude shortcut that amplifies
      PRIMARY-B's effect at inference.
  Pending user confirmation before applying fixes.
files_changed: []
