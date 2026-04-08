---
phase: 22
slug: efficientat-v8-retrain-with-fixed-train-serve-window-contrac
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 22 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `22-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (+ pytest-asyncio) |
| **Config file** | `pyproject.toml` (see `[tool.pytest.ini_options]`) |
| **Quick run command** | `pytest tests/unit/training tests/unit/classification/efficientat -x -q` |
| **Full suite command** | `pytest tests/ -x -q --ignore=tests/integration/test_training_smoke.py` |
| **Estimated runtime** | ~30 seconds (quick) / ~3 minutes (full, sans Vertex e2e) |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green AND eval harness produces a metrics JSON
- **Max feedback latency:** 60 seconds for unit/integration; Vertex training and eval harness are out-of-band

---

## Per-Task Verification Map

> Populated by the planner during plan generation. Each task with code impact MUST have an `<automated>` verify command pointing to a test file enumerated below, OR depend on a Wave 0 task that creates the test.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | TBD | — | TBD | TBD | TBD | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

These test files must exist (or be created in Wave 0) before any other wave runs:

- [ ] `tests/unit/classification/efficientat/test_window_contract.py` — assert `EFFICIENTAT_WINDOW_SECONDS == 1.0`, `EFFICIENTAT_SEGMENT_SAMPLES == 32000`, and `source_window_samples(16000) == 16000`. Pins the v7 regression vector.
- [ ] `tests/unit/training/test_windowed_dataset_length.py` — instantiate `WindowedHFDroneDataset` with a synthetic 1-second clip; assert `__getitem__` returns a tensor of length `EfficientATMelConfig().segment_samples`. Loud-fail on any drift.
- [ ] `tests/unit/classification/test_efficientat_predict_warn.py` — feed `EfficientATClassifier.predict` a tensor whose last dim ≠ `segment_samples`; assert a WARN is emitted (caplog) and the call does not raise.
- [ ] `tests/unit/training/test_rmsnormalize_parity.py` — load a fixture WAV, run train preprocessing path and inference preprocessing path, assert max abs amplitude difference < 1e-4 (post-resample parity).
- [ ] `tests/integration/test_data_integrity_preflight.py` — enumerate `data/field/{drone,background}/20260408_*.wav`, soundfile.info each, assert `samplerate == 16000`, `channels == 1`, no NaN after decode, total file count == 17 (13 drone + 4 background).
- [ ] `tests/integration/test_dataset_cardinality.py` — build the v8 dataset, assert it contains every preflight file (no silent drops), assert label balance matches manifest.
- [ ] `tests/integration/test_holdout_split.py` — assert holdout split is deterministic (same seed → same split) and no file appears in both train and eval.
- [ ] `tests/e2e/test_eval_harness.py` — run the new `promote_if_gates_pass` against a tiny synthetic UMA-16 fixture set; assert it emits `metrics.json` with `real_TPR` and `real_FPR` keys and respects the D-27 thresholds.

*Wave 0 also installs/verifies: pytest-asyncio (already declared), `soundfile`, `torch>=2.11`, fixture WAVs under `tests/fixtures/efficientat_v8/`.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vertex L4 training run completes in us-east1 | Promotion path | Requires GCP credentials and quota — cannot run in CI | `python scripts/vertex_submit.py --version v8 --region us-east1` then monitor job logs; success = checkpoint uploaded to GCS |
| Real-device hold-out TPR ≥ 0.80 / FPR ≤ 0.05 | D-27 promotion gate | Requires UMA-16 hardware recordings | Run `python scripts/promote_efficientat.py --version v8 --eval-set data/eval/uma16_real_v8/` and read `metrics.json` |
| v8 replaces v6 in operational pipeline without regression | Promotion completion | Requires live pipeline integration test on hardware | Swap model symlink, run a recorded UMA-16 session, compare detection events to v6 baseline |
| Operator visually inspects preflight report for the trimmed `20260408_091054_136dc5.wav` clip | Data integrity | Trimmed file is sentinel for the broader integrity story | Run preflight script, eyeball the duration column shows ~61.4 s, not 71.4 s |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (8 test files above)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for non-Vertex tasks
- [ ] `nyquist_compliant: true` set in frontmatter after planner populates the per-task table

**Approval:** pending
