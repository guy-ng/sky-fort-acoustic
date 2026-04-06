# Phase 20: Retrain v7 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote
**Mode:** auto (`--auto`)
**Areas discussed:** Gain Augmentation, Room IR, UMA-16 Ambient Negatives, Window Overlap, BG Noise Set, Vertex Config, Eval Metric

Gray areas were auto-resolved with recommended defaults because the user selected auto mode at ~78% context usage.

---

## Gain Augmentation (Wide Range)

| Option | Description | Selected |
|--------|-------------|----------|
| ±6 dB (current) | Keep Phase 15 baseline. Does not bridge the ~60 dB gap between DADS (~ −25 dBFS) and UMA-16 ambient (~ −82 dBFS). | |
| ±20 dB | Moderate, covers most of the gap. | |
| ±40 dB (uniform) + keep ±6 dB audiomentations jitter | Phase description explicit. Uniform dist over full range. Clip after. | ✓ |
| ±40 dB log-uniform | Emphasizes smaller gains. Rejected — evidence shows distribution should be flat across the full deployment range. | |

**Auto-selected:** ±40 dB uniform, stacked before audiomentations ±6 dB jitter.
**Rationale:** Phase goal is explicit; debug evidence (uma16-no-detections.md) proves the gap exists.

---

## Room Impulse Response

| Option | Description | Selected |
|--------|-------------|----------|
| pyroomacoustics (procedural) | Zero data download, parameterized, reproducible. | ✓ |
| MIT IR Survey (recorded) | Real IRs but requires download + licensing review. | |
| OpenAIR (recorded) | Free, but smaller collection. | |
| BUT Reverb Database | Academic, large, but download-bound. | |
| No RIR | Ignore phase requirement. | |

**Auto-selected:** pyroomacoustics procedural RIRs with random room params.
**Rationale:** Fastest to integrate, no external download, matches "procedural over dataset" preference implied by debug-to-fix cycle speed.

---

## UMA-16 Ambient Negatives

| Option | Description | Selected |
|--------|-------------|----------|
| Collect ≥30 min real UMA-16 ambient, mix via BackgroundNoiseMixer + pure negatives | Direct deployment match. | ✓ |
| Synthesize UMA-16 ambient from DC mic noise models | Cheap but unrealistic. | |
| Skip — rely on BG noise from ESC-50/UrbanSound8K | Keeps Phase 15 baseline. Doesn't address deployment gap. | |

**Auto-selected:** Real capture ≥30 min + mixer integration + 10% pure-negative batch share.

---

## 60% Overlap Windows

| Option | Description | Selected |
|--------|-------------|----------|
| Random 0.5s segment (current) | 1 random view per file per epoch. | |
| 60% overlap sliding windows (hop 0.2s), session-split preserved | Phase-specified. ~5x more training samples per epoch. | ✓ |
| 50% overlap | Less dense, simpler arithmetic. | |
| 75% overlap | Denser, higher compute + correlation. | |

**Auto-selected:** 60% overlap with mandatory file-level session split.
**Rationale:** Phase description explicit; session-level split is a hard invariant per compass doc §4.

---

## BG Noise Set Expansion

| Option | Description | Selected |
|--------|-------------|----------|
| ESC-50 + UrbanSound8K only (Phase 15) | Current baseline. | |
| + FSD50K subset (wind/rain/traffic/fan/engine/bird) + DroneAudioSet negatives + UMA-16 ambient | Full compass doc §1 integration. | ✓ |
| + AudioSet full | Huge, requires YouTube scraping. | |

**Auto-selected:** FSD50K subset + DroneAudioSet negatives + UMA-16 ambient (stacked on Phase 15 baseline).

---

## Vertex Training Config

| Option | Description | Selected |
|--------|-------------|----------|
| Local training | Phase rules out local training. | |
| Vertex T4, 40 epochs | Cheapest GPU. | |
| Vertex L4, g2-standard-8, three-stage recipe (10+15+20), focal loss | Faster, better memory headroom for batch_size=64. | ✓ |
| Vertex A100 | Overkill for mn10 4.5M params. | |

**Auto-selected:** L4 / g2-standard-8 / three-stage / focal / batch_size=64.

---

## Success Metric / Eval

| Option | Description | Selected |
|--------|-------------|----------|
| Only DADS test accuracy | Doesn't address deployment generalization. | |
| DADS ≥95% + real UMA-16 eval set TPR ≥ 0.80 @ FPR ≤ 0.05 | Locked, objective, matches phase goal. | ✓ |
| Subjective "sounds better" | Not measurable. | |

**Auto-selected:** Quantified dual metric (DADS regression + real-capture TPR/FPR).

---

## Claude's Discretion

- RIR caching strategy (precompute pool vs per-sample generation)
- Exact FSD50K class-slug → filename mapping
- DataLoader worker count on Vertex L4
- `torchaudio.functional.fftconvolve` vs `scipy.signal.fftconvolve` for RIR application
- Docker layer caching strategy for new data directories
- Room parameter sampling distribution (uniform vs log-uniform per dimension)

## Deferred Ideas

- Real recorded RIR datasets (MIT IR Survey, BUT Reverb, OpenAIR) — procedural used instead
- Architectural changes (ResNet-Mamba hybrid from compass §3.E)
- ONNX/TFLite export (Phase 16)
- Vertex hyperparameter sweep
- Online learning from deployed captures
