---
phase: 22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac
plan: 09
subsystem: evaluation
tags: [promotion-gate, d-27, uma16, real-device, eval]

requires:
  - phase: 22
    plan: 08
    provides: v8 checkpoint
provides:
  - metrics_v8.json with D-27 gate results
  - Promotion decision: REJECTED — v6 stays operational
affects: []

key-decisions:
  - "v8 REJECTED: real_TPR=0.034 (need >=0.80). v6 stays as operational model."
  - "Root cause: fine-tuning from v6 on DADS-dominated dataset causes v8 to overfit to DADS and lose v6's real-device detection ability"
  - "Eval harness was missing RMS normalization — fixed, but didn't resolve the core domain gap"

metrics:
  completed: "2026-04-13"
  tasks_completed: 1
  tasks_total: 3
---

# Phase 22 Plan 09: Promotion Gate Summary

**One-liner:** v8 REJECTED — real_TPR=0.034 (gate requires >=0.80). v6 remains operational. Fine-tuning degraded real-device performance despite 0.994 DADS val_acc.

## D-27 Gate Result

| Metric | Required | v8 Actual | v6 Comparison |
|--------|----------|-----------|---------------|
| real_TPR | >= 0.80 | **0.034** | ~0.60 |
| real_FPR | <= 0.05 | 0.0 | ~0.0 |
| DADS val_acc | — | 0.994 | ~0.98 |

**Decision: NOT PROMOTED. v6 stays.**

## Root Cause Analysis

### The paradox: higher val_acc, worse real performance

v8 achieves val_acc=0.994 on DADS (vs v6's ~0.98) but TPR=0.034 on real UMA-16 audio (vs v6's ~0.60). This is a textbook **distribution overfitting** pattern:

1. **Training data imbalance:** DADS has 180,320 samples. Field recordings contributed only ~12 ConcatDataset entries (7 drone + 3 bg for training, 5 holdout). The field data is <0.01% of the training set.

2. **Fine-tuning from v6 overwrote real-device features:** v6 was trained from AudioSet pretrained weights, which gave it broad audio feature representations. Fine-tuning on DADS-dominated data specialized v8 toward DADS acoustic signatures and away from v6's generalizable features.

3. **Segment-level comparison (10in drone, RMS-normalized):**

| Segment | v8 prob | v6 prob |
|---------|---------|---------|
| 6 | 0.59 | **0.99** |
| 7 | 0.56 | **0.99** |
| 8 | 0.62 | **1.00** |
| 9 | 0.72 | **1.00** |

v6 is confidently detecting drones (0.99+) where v8 is uncertain (0.56-0.72).

### Secondary finding: eval harness missing RMS normalization

The eval harness (`uma16_eval.py`) was feeding raw audio to `classifier.predict()` without the D-34 RMS normalization (target=0.1) that both training and live inference apply. Fixed, but the effect was marginal (TPR: 0.031 → 0.034) — the core issue is the domain gap, not preprocessing.

## Recommendations for v9

1. **Reweight field data dramatically** — either oversample field recordings 100x+ or undersample DADS to match field data volume
2. **Freeze backbone, train only classifier head** — prevents overwriting v6's feature representations
3. **Collect more field data** — 7 training files is insufficient to represent outdoor UMA-16 drone signatures
4. **Use v6 as the starting point again** but with a learning rate 10-100x lower to preserve its generalizable features
5. **Add real-device TPR as a training metric** — run eval harness after each epoch and early-stop on real_TPR, not just DADS val_loss

## Commits

| Hash | Message |
|------|---------|
| 3e2159f | fix(22-09): fix promote script loader + add RMS norm to eval harness |
