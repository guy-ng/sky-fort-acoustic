---
phase: 20
slug: retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `pytest tests/unit/training/ -x -q` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~60 seconds (unit), ~180 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/training/ -x -q`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

*Populated by planner after plans are drafted. Each plan task must map to a row.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 20-XX-XX | XX | N | D-XX | — | N/A | unit | `pytest tests/unit/training/test_xxx.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/training/test_wide_gain_augmentation.py` — stubs for D-01..D-04
- [ ] `tests/unit/training/test_room_ir_augmentation.py` — stubs for D-05..D-08
- [ ] `tests/unit/training/test_background_noise_mixer_uma16.py` — stubs for D-09..D-12
- [ ] `tests/unit/training/test_sliding_window_dataset.py` — stubs for D-13..D-16 (including session-level split leakage test)
- [ ] `tests/unit/training/test_training_config_phase20.py` — stubs for new config fields (wide_gain_db, rir_enabled, rir_probability, window_hop_ratio, **specaug_freq_mask, specaug_time_mask, save_gate_min_accuracy** per D-30/D-32)
- [ ] `tests/unit/training/test_specaug_scaling.py` — stub for D-30 (mel_train uses cfg.specaug_freq_mask / specaug_time_mask, not legacy 48/192)
- [ ] `tests/unit/training/test_trainer_loss_factory.py` — stub for D-31 (criterion = build_loss_function(cfg); focal selected when configured)
- [ ] `tests/unit/training/test_save_gate.py` — stub for D-32 (refuse-to-save when min(tp,tn)==0 or val_acc < threshold)
- [ ] `tests/unit/training/test_stage1_unfreeze_scope.py` — stub for D-33 (Stage 1 unfreezes only final binary head)
- [ ] `tests/unit/test_rms_normalize.py` — stub for D-34 helper (numpy + torch, idempotent, silence-safe)
- [ ] `tests/unit/test_raw_audio_preprocessor.py` — stub for D-34 inference path (RMS within 1e-3 of 0.1 across input range 0.001–10.0)
- [ ] `tests/unit/training/test_rms_normalize_augmentation.py` — stub for D-34 trainer wiring (RmsNormalize is LAST in train + eval chains)
- [ ] `tests/integration/test_rms_contract_train_inference.py` — stub for D-34 end-to-end parity gate
- [ ] `tests/integration/test_vertex_dockerfile_copy.py` — stub asserting Dockerfile.vertex copies new data dirs (D-24)
- [ ] `tests/conftest.py` — shared fixtures (tiny RIR, synthetic 16 kHz waveform, temp noise dir)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| UMA-16 ambient collection ≥30 min | D-09 | Requires physical hardware capture in 4 conditions | Record indoor quiet, indoor HVAC, outdoor quiet, outdoor wind to `data/field/uma16_ambient/` — verify with `python -c "import soundfile; ..."` sum duration ≥1800s |
| Real-UMA-16 eval set ≥20 min with labels | D-27 | Requires physical drone-present capture + hand-labeling | Capture drone flight + ambient, label segments in `labels.json`, verify duration via script |
| Vertex training job succeeds | D-21..D-25 | Requires GCP billing + remote run | Submit via `scripts/vertex_submit.py` v7 job, verify `best_model.pt` in GCS bucket |
| D-26 DADS test accuracy ≥95% | D-26 | Requires trained v7 checkpoint + DADS test split | Run `python -m acoustic.training.evaluate --model models/efficientat_mn10_v7.pt --split test` |
| D-27 real-capture TPR ≥0.80, FPR ≤0.05 | D-27 | Requires trained v7 + real eval set | Run extended Phase 9 eval harness on `data/eval/uma16_real/` |
| D-29 promotion | D-29 | Gated behind D-26 AND D-27 | Manual `cp models/efficientat_mn10_v7.pt models/efficientat_mn10.pt` after gates pass |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
