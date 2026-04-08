---
phase: 22
slug: efficientat-v8-retrain-with-fixed-train-serve-window-contrac
status: draft
nyquist_compliant: true
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
| 22-01-T1 | 01 | 0 | REQ-22-W1/W2/W3/W4 | T-22-02 | test scaffolds exist | unit | `pytest tests/unit/classification/efficientat/test_window_contract.py tests/unit/training/test_windowed_dataset_length.py tests/unit/classification/test_efficientat_predict_warn.py tests/unit/training/test_rmsnormalize_parity.py --collect-only -q` | ✅ Plan 01 creates | ⬜ pending |
| 22-01-T2 | 01 | 0 | REQ-22-D3/G1 | T-22-05 | integration scaffolds | integration | `pytest tests/integration/test_data_integrity_preflight.py tests/integration/test_dataset_cardinality.py tests/integration/test_holdout_split.py tests/e2e/test_eval_harness.py --collect-only -q` | ✅ Plan 01 creates | ⬜ pending |
| 22-01-T3 | 01 | 0 | — | T-22-02 | provenance frozen | script | `test -f models/MODEL_PROVENANCE.md && grep -q sha256 models/MODEL_PROVENANCE.md` | n/a | ⬜ pending |
| 22-02-T1 | 02 | 1 | REQ-22-W1 | T-22-01 | contract module exists + self-check | unit | `pytest tests/unit/classification/efficientat/test_window_contract.py -x -q` | ✅ | ⬜ pending |
| 22-02-T2 | 02 | 1 | REQ-22-W1 | T-22-01 | trainer literal eliminated | grep | `! grep -n 'int(0.5 \* _SOURCE_SR)' src/acoustic/training/efficientat_trainer.py && grep -q window_contract src/acoustic/training/efficientat_trainer.py` | n/a | ⬜ pending |
| 22-02-T3 | 02 | 1 | REQ-22-W1 | T-22-01 | pipeline + hf_dataset imports | grep | `grep -q EFFICIENTAT_WINDOW_SECONDS src/acoustic/pipeline.py && grep -q window_samples:.*16000 src/acoustic/training/hf_dataset.py` | n/a | ⬜ pending |
| 22-03-T1 | 03 | 2 | REQ-22-W2 | T-22-01 | dataset length assertion | unit | `pytest tests/unit/training/test_windowed_dataset_length.py -x -q` | ✅ | ⬜ pending |
| 22-03-T2 | 03 | 2 | REQ-22-W3 | T-22-03 | classifier WARN one-shot | unit | `pytest tests/unit/classification/test_efficientat_predict_warn.py -x -q` | ✅ | ⬜ pending |
| 22-03-T3 | 03 | 2 | REQ-22-W4 | T-22-01 | RMS parity post-resample | unit | `pytest tests/unit/training/test_rmsnormalize_parity.py -x -q` | ✅ | ⬜ pending |
| 22-04-T1 | 04 | 3 | REQ-22-D3 | T-22-01/05 | preflight + holdout frozen | integration | `python scripts/preflight_v8_data.py && pytest tests/integration/test_data_integrity_preflight.py tests/integration/test_holdout_split.py -x -q` | ✅ | ⬜ pending |
| 22-04-T2 | 04 | 3 | REQ-22-D2 | T-22-02 | holdout manifest sha256 | script | `python -c "import json;m=json.load(open('data/eval/uma16_real_v8/manifest.json'));assert len(m['files'])==5 and all(len(f['sha256'])==64 for f in m['files'])"` | n/a | ⬜ pending |
| 22-05-T1 | 05 | 3 | REQ-22-D1 | T-22-01/legal | Kaggle investigation report | script | `test -f .planning/phases/22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac/KAGGLE_DATASET_INVESTIGATION.md` | n/a | ⬜ pending |
| 22-05-T2 | 05 | 3 | REQ-22-D1 | T-22-legal | human decision | checkpoint | manual (decision) | n/a | ⬜ pending |
| 22-06-T1 | 06 | 4 | REQ-22-D1/D2 | T-22-01 | ConcatDataset + finetune | integration | `pytest tests/integration/test_dataset_cardinality.py -x -q && grep -q ConcatDataset src/acoustic/training/efficientat_trainer.py` | ✅ | ⬜ pending |
| 22-06-T2 | 06 | 4 | — | T-22-04 | additive v8 submit path | grep | `grep -q submit_v8_job scripts/vertex_submit.py && grep -q submit_v7_job scripts/vertex_submit.py` | n/a | ⬜ pending |
| 22-06-T3 | 06 | 4 | — | T-22-04 | base image v2 with field data | grep | `grep -q 20260408 Dockerfile.vertex-base` | n/a | ⬜ pending |
| 22-07-T1 | 07 | 5 | REQ-22-G1 | T-22-02 | eval harness + promotion module | e2e | `pytest tests/e2e/test_eval_harness.py -x -q` | ✅ | ⬜ pending |
| 22-07-T2 | 07 | 5 | REQ-22-G1 | T-22-03 | promote_efficientat CLI exists | script | `python scripts/promote_efficientat.py --help >/dev/null` | n/a | ⬜ pending |
| 22-08-T1 | 08 | 6 | — | T-22-04 | quota + preflight | script | `python scripts/preflight_v8_data.py` | n/a | ⬜ pending |
| 22-08-T2 | 08 | 6 | REQ-22-D1 | T-22-04 | Vertex training run | manual | Vertex job SUCCEEDED | n/a | ⬜ pending |
| 22-08-T3 | 08 | 6 | — | T-22-02 | v8 sha sidecar + smoke | script | `test -f models/efficientat_mn10_v8.pt && test -f models/efficientat_mn10_v8.sha256` | n/a | ⬜ pending |
| 22-09-T1 | 09 | 7 | REQ-22-G1 | T-22-03 | metrics_v8.json | script | `python -c "import json;m=json.load(open('.planning/phases/22-efficientat-v8-retrain-with-fixed-train-serve-window-contrac/metrics_v8.json'));assert 'real_TPR' in m and 'real_FPR' in m"` | n/a | ⬜ pending |
| 22-09-T2 | 09 | 7 | REQ-22-G1 | T-22-03 | human approve | checkpoint | manual | n/a | ⬜ pending |
| 22-09-T3 | 09 | 7 | REQ-22-G2 | T-22-02 | operational swap | script | `[ "$(shasum -a 256 models/efficientat_mn10.pt | awk '{print $1}')" = "$(shasum -a 256 models/efficientat_mn10_v8.pt | awk '{print $1}')" ]` | n/a | ⬜ pending |


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
