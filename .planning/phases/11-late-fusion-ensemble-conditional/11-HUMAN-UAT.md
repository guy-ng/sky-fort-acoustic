---
status: partial
phase: 11-late-fusion-ensemble-conditional
source: [11-VERIFICATION.md]
started: 2026-04-02T00:00:00Z
updated: 2026-04-02T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Ensemble Live Pipeline
expected: Start the service with a real ensemble config (2+ trained .pt checkpoints). Startup log shows "Loaded ensemble with N models" and detection events flow from the weighted ensemble. Service starts in ensemble mode with no errors; detection events use weighted voting output.
result: [pending]

### 2. Measurable Accuracy Improvement (Success Criterion 3)
expected: Run `POST /api/eval/run` with `ensemble_config_path` pointing to an ensemble of trained models on a labeled test dataset. Ensemble F1 >= best single-model F1, with `per_model_results` in the response showing individual model metrics for comparison.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
