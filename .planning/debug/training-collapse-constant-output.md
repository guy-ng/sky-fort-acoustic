---
status: investigating
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

## Resolution

root_cause:
fix:
verification:
files_changed: []
