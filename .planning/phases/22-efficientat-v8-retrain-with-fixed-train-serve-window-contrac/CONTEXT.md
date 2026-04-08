# Phase 22 — EfficientAT v8 Retrain — Pre-Plan Context

> **Read this before running `/gsd-plan-phase 22`.** It captures user-locked constraints, the v7 post-mortem inputs, and acceptance criteria so the planner does not re-derive them.

## Origin

Driven by the v7 model regression diagnosed on 2026-04-08:
- **Root cause report:** [`.planning/debug/efficientat-v7-regression-vs-v6.md`](../../debug/efficientat-v7-regression-vs-v6.md)
- **Single-line summary:** v7 was trained on 0.5s windows; the inference pipeline feeds it 1.0s windows. EfficientAT mn10 is shape-agnostic so no crash, but BatchNorm/attention pooling/classifier head are tuned for 50 mel frames and at inference see 100. v7 looks healthy in val (val_acc=0.983) and silently regresses in production. The bug is a single literal at [`src/acoustic/training/efficientat_trainer.py:456`](../../../src/acoustic/training/efficientat_trainer.py#L456): `window_samples = int(0.5 * _SOURCE_SR)`.

## Goal

Train `models/efficientat_mn10_v8.pt` that:
1. Beats v6 on a real-device UMA-16 hold-out (TPR ≥ 0.80, FPR ≤ 0.05)
2. Cannot silently regress like v7 — train/serve contract enforced by code, not convention
3. Incorporates the 2026-04-08 field recordings (drone + background)

## User-Locked Constraints

**Do not negotiate these in planning. Plan around them.**

| Constraint | Value | Why |
|---|---|---|
| Training window length | **1.0 second** @ 32 kHz = 32000 samples | Must equal `EfficientATMelConfig().segment_samples` (the inference contract). Anything else reproduces the v7 bug. |
| Sliding-window overlap | **50%** (`window_overlap_ratio=0.5`) | User-specified. Note that DADS clips are uniformly 1.0s, so 50% overlap on 1.0s windows of 1.0s clips yields 1 window per clip — meaningful overlap only happens on the new multi-second field recordings. |
| Cloud region | **`us-east1`** | User-specified. |
| Cloud accelerator | **NVIDIA L4** | User-specified. Note: Phase 20 used L4 — region may differ; verify L4 quota in us-east1 before submitting. |
| New training data | All `data/field/drone/20260408_*.wav` and `data/field/background/20260408_*.wav` | User-specified. |
| `20260408_091054_136dc5.wav` ("10inch payload 4kg") | **Already trimmed to 61.4s** (was 71.4s, contaminated tail removed). Backup at `.bak`. | Use as-is. Do not re-trim. |
| Data integrity preflight | **Required** — assert every new recording is transferred, decoded, sample-rate-correct, and reaches the training loop with the right label | User: "make sure all the data is transfer correctly" |

## Carry Over From Phase 20 (Keep — They Fix Real Bugs)

These were validated by the training-collapse debug (`.planning/debug/training-collapse-constant-output.md`) and must NOT be reverted in v8:
- Wide-gain augmentation
- Room IR augmentation
- BG noise negatives from ESC50 / UrbanSound8K / FSD50K subset (Phase 20.1 corpora at `data/noise/`)
- Focal loss
- Save gate D-32 (logits-mode parity check)
- Narrow Stage 1 schedule

## Must-Fix Items From v7 Post-Mortem

1. **Window-length literal** — derive `window_samples` in `efficientat_trainer.py:456` from `EfficientATMelConfig().segment_samples`, not the 0.5 literal.
2. **Single source of truth for window length** — share `_training_window_seconds` (currently in [`src/acoustic/pipeline.py:72-86`](../../../src/acoustic/pipeline.py#L72-L86)) with the trainer. Move to a shared config module or import from pipeline. Rationale: the trainer and the pipeline cannot drift again.
3. **Length assertion in `WindowedHFDroneDataset`** — `__getitem__` must assert returned tensor length equals `EfficientATMelConfig().segment_samples`. Fail loudly at the first bad item.
4. **Runtime length WARN in inference** — `EfficientATClassifier.predict` should log a WARN (not raise) when `features.shape[-1] != EfficientATMelConfig().segment_samples`. This would have surfaced the v7 bug in the first inference log.
5. **RmsNormalize domain parity** — training normalizes at 16 kHz pre-resample ([`src/acoustic/training/hf_dataset.py:309-316`](../../../src/acoustic/training/hf_dataset.py#L309-L316)); inference normalizes at 32 kHz post-resample ([`src/acoustic/classification/preprocessing.py:205-209`](../../../src/acoustic/classification/preprocessing.py#L205-L209)). Move training to post-resample. ~2% amplitude skew today; latent parity gap that will eventually bite.

## Promotion Gate (Hard Stop)

**v8 must not replace v6 in operational use until:**
1. **Plan 20-06 (eval harness + D-27 real-device promotion gate) is executed.** It already exists at [`.planning/phases/20-.../20-06-eval-harness-and-promotion-gate-PLAN.md`](../20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-06-eval-harness-and-promotion-gate-PLAN.md). It was deferred during Phase 20 — must be done first now.
2. v8 passes D-27 on a UMA-16 hold-out: **real_TPR ≥ 0.80** and **real_FPR ≤ 0.05**.
3. Hold-out includes 2026-04-08 recordings that were NOT used in training (no double-dipping).

## Hold-out Discipline

The 2026-04-08 recordings are precious — they're the only real-device UMA-16 data we have and the only thing that proves v8 isn't lying. Split policy:
- Pick a deterministic train/eval split (e.g., 70/30 by file hash, or hold out specific files like the 4kg/10in clip)
- Document the split in the phase plan so it's reproducible
- Real-device eval gate runs ONLY on hold-out files

## Open Questions for the Planner (resolve in plan, not constraints)

These are NOT user-locked — the planner should propose answers in PLAN.md:

1. **Fine-tune from v6 vs train from EfficientAT AudioSet checkpoint?** v6 is the proven baseline; fine-tuning from it is lower-risk and faster. Training from scratch matches Phase 20's protocol but takes longer. Recommend: **fine-tune from v6** for v8, since the only changes vs v6 are (a) the window-length fix (which trainer code, not weights) and (b) new field data.
2. **Sliding-window sampler retirement for 1s clips?** With 1.0s windows and 1.0s DADS clips, sliding adds nothing. Either (a) keep `WindowedHFDroneDataset` for the multi-second field recordings only and route DADS through `_LazyEfficientATDataset`, or (b) delete `WindowedHFDroneDataset` entirely if all data is ≤1s. Decide in plan.
3. **Class balance with new field data** — how many drone vs background clips does 2026-04-08 add? Does the WeightedRandomSampler need re-weighting?
4. **Vertex base image rebuild** — Phase 20 rebuilt `Dockerfile.vertex-base`. Does the v8 trainer require a rebuild or can we reuse the Phase 20 image?

## Data Inventory (Snapshot, 2026-04-08)

Run during planning to populate exact counts; here's what's known:
- Drone: `data/field/drone/20260408_*.wav` (multiple files; one trimmed)
- Background: `data/field/background/20260408_*.wav`
- DADS corpus: unchanged from Phase 20 (1.0s clips, 16 kHz)
- BG noise corpora: ESC50, UrbanSound8K, FSD50K subset under `data/noise/` (Phase 20.1 — already on disk)

## Reference Artifacts

- Root cause: [`.planning/debug/efficientat-v7-regression-vs-v6.md`](../../debug/efficientat-v7-regression-vs-v6.md)
- Phase 20 summary: [`.planning/phases/20-.../20-05-SUMMARY.md`](../20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-05-SUMMARY.md)
- Phase 20-06 deferred plan: [`.planning/phases/20-.../20-06-eval-harness-and-promotion-gate-PLAN.md`](../20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-06-eval-harness-and-promotion-gate-PLAN.md)
- Trainer (must-fix line): [`src/acoustic/training/efficientat_trainer.py:456`](../../../src/acoustic/training/efficientat_trainer.py#L456)
- Inference contract: [`src/acoustic/pipeline.py:72-86`](../../../src/acoustic/pipeline.py#L72-L86), [`src/acoustic/classification/efficientat/config.py:24-29`](../../../src/acoustic/classification/efficientat/config.py#L24-L29)
