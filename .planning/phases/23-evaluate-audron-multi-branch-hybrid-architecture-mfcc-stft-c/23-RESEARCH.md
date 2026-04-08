# Phase 23: Evaluate AUDRON Multi-Branch Hybrid Architecture — Research

**Researched:** 2026-04-08
**Domain:** Acoustic drone classification — alternative architecture evaluation
**Research type:** Decision-output (Adopt / Reject / Hybrid), not implementation plan
**Overall confidence:** HIGH (paper + repos verified; comparison conclusions well-grounded)

---

## Executive Recommendation

**REJECT as classifier replacement. HYBRID-RESEARCH-TRACK permissible only under strict conditions. Confidence: HIGH.**

Rationale (5 sentences):

1. AUDRON is a real, verifiable paper (arXiv 2512.20407, AmygdalaAI-India Lab, IEEE INDICON 2025) with a coherent 4-branch architecture, but its headline 98.51% number comes from **binary classification on a heavily imbalanced indoor close-mic dataset (DroneAudioDataset: 1,066 drone vs 8,297 noise)**, which is not the regime we operate in. [VERIFIED: arxiv.org/html/2512.20407v1]
2. The 300m+ detection range attributed to AUDRON in CONTEXT.md is **not in the paper** — AUDRON reports no range claim at all. 300m is a generic ceiling from the acoustic-detection literature (ESC-50/YAMNet-era benchmarks) that got cross-wired during phase intake. [VERIFIED: paper text + WebSearch]
3. The "SudarshanChakra production-grade variant" referenced in CONTEXT.md (`kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN`) is a **plain 4-block Mel-Spec CNN (~500K params, 95.23% on the same indoor dataset) with no AUDRON affiliation** — the CONTEXT.md claim of a lineage is wrong. [VERIFIED: github.com/kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN]
4. Our EfficientAT mn10 v6/v8 line is already the correct architectural choice: ~4.5M params, AudioSet-pretrained, proven on UMA-16 field recordings, with an ONNX int8 edge path already built for RPi4 (Phase 21). AUDRON's 4-branch fusion (no reported param count, no reported latency, no edge deployment, no pretrained weights, no public repo) is strictly worse on every axis we care about — **except ensemble diversity**, which is its only possibly-useful contribution.
5. The only defensible use is as a **parallel research track after Phase 22 v8 lands**, to evaluate whether adding a second diverse model to an ensemble improves real-device TPR/FPR on our UMA-16 hold-out — and even that is gated on independently reimplementing AUDRON since no reference weights exist.

**What to do instead:** Ship Phase 22 (v8 retrain) and Phase 21 (RPi4 edge app) as-is. Do not spawn AUDRON training phases now. Revisit only if v8 fails the real-device gate (TPR ≥ 0.80 / FPR ≤ 0.05) AND the failure mode looks like representation diversity (e.g., v8 confident on background classes that a temporal/attention branch would catch).

---

## Source Verification

| Source | Status | URL | What was confirmed |
|---|---|---|---|
| AUDRON paper | VERIFIED | https://arxiv.org/abs/2512.20407 ; https://arxiv.org/html/2512.20407v1 | Title, authors (Rajdeep Chatterjee, Sudip Chakrabarty, Trishaani Acharjee, Deepanjali Mishra), AmygdalaAI-India Lab affiliation, IEEE INDICON 2025 venue, 4-branch architecture, 98.51% binary / 97.11% multiclass accuracy, dataset composition, AdamW/lr=0.001/batch=16/50 epochs on NVIDIA T4 Kaggle — all match CONTEXT.md. |
| AmygdalaAI-India Lab | VERIFIED | https://amygdalaaiindia.github.io/ | Referenced inside paper header. Volunteer research group. |
| SudarshanChakra repo | VERIFIED (but MISATTRIBUTED in CONTEXT.md) | https://github.com/kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN | Repo exists. Described as "production-grade". **However: it is a single 4-conv-block CNN (~500K params), NOT the AUDRON 4-branch fusion. No reference to AUDRON or AmygdalaAI. Uses DroneAudioDataset auto-ingested. Reports 95.23% accuracy / 96.97% recall on test.** Created ~Nov 24 2025. Author `kbhujbal` has no public link to the AUDRON authors. |
| DroneAudioDataset | VERIFIED | https://github.com/saraalemadi/DroneAudioDataset | Repo exists. **Indoor propeller recordings**, explicitly stated. License/drone-types/SR/duration/mic-distance **NOT documented in README** — only the associated conference paper ("Audio Based Drone Detection and Identification using Deep Learning", Al-Emadi et al.) would have details. Known weakness: close-mic indoor → poor transfer to open-air beamformer single-channel. |
| AUDRON 300m+ range claim | ❌ NOT FOUND | — | The AUDRON paper contains **no range claim**. CONTEXT.md's "300m+ detection" attribution is incorrect. WebSearch confirms 300m is a generic acoustic-detection ceiling cited in unrelated surveys (Fraunhofer IDMT, MDPI Electronics UAV acoustic surveys, Robin Radar blog). |
| AUDRON official code release | ❌ NOT FOUND | — | No GitHub link in the paper. No AmygdalaAI-India repo hosting the 4-branch fusion model. Paper does not ship weights, config, or training code. Reimplementation would be **from-scratch from paper text**. |
| 98.51% independent reproduction | ❌ NOT FOUND | — | No third-party reimplementation located. Single-paper, single-run, single-lab result. Stated as `98.51% ± 0.09` across folds in the paper — internal cross-validation, not independent reproduction. |

**Honest summary of source health:** The paper exists and is internally consistent. The "production-grade variant" claim in CONTEXT.md is factually wrong — SudarshanChakra is an unrelated simpler CNN that just happens to use the same dataset. There is no public AUDRON code. All numbers are from a single paper on a single indoor dataset.

---

## Comparison Table: AUDRON vs EfficientAT v6/v8

| Axis | EfficientAT mn10 (v6 baseline / v8 in flight) | AUDRON (paper) | Winner |
|---|---|---|---|
| **Headline accuracy** | v6: operational on UMA-16 field data; v7 val_acc=0.983, F1=0.990 (collapsed in production due to window bug); v8 gate = real_TPR≥0.80 / real_FPR≤0.05 on UMA-16 hold-out | 98.51% binary / 97.11% multiclass — **indoor DroneAudioDataset cross-val only** | **Tie on paper, EfficientAT on what matters (real-device)** |
| **Evaluation regime** | Real UMA-16 field recordings (2026-04-08) + DADS + outdoor ambient | DroneAudioDataset indoor close-mic + ESC-50 noise + synthetic harmonics | **EfficientAT** — matches deployment |
| **Parameter count** | ~4.5M params (mn10) | **Not reported in paper**. Rough estimate: 4 branches + BiLSTM + AE → easily 5-15M+ depending on STFT-CNN filter counts. Larger than mn10. | **EfficientAT** (known, bounded) |
| **Model size on disk** | ~17 MB TorchScript-free state_dict; int8 ONNX even smaller (Phase 21 21-03 produces FP32 + int8) | Not reported. Likely 20-60 MB. | **EfficientAT** |
| **Inference latency** | v6 runs real-time in Docker microservice; ONNX int8 path for RPi4 validated in Phase 21 | **Not reported at all**. 4 parallel branches + BiLSTM attention — BiLSTM is sequential and edge-hostile | **EfficientAT** |
| **RPi4 edge fit (Phase 21)** | ✅ Already running. ONNX int8 artifact shipped. Phase 21 is ~7/8 complete. | ❌ No edge deployment demonstrated. Paper says "future work: low-power version for edge devices". BiLSTM on RPi4 CPU is painful. 4 parallel branches multiply RAM + cache pressure. | **EfficientAT — not close** |
| **Real-time 48kHz × 16-ch fit** | Fits: beamformer outputs single-channel, v8 trains on 1.0s @ 32kHz windows (matches inference contract after Phase 22 fix) | Paper uses ~3s windows (48,000 samples, sample rate implied ~16 kHz). Not 32kHz, not 1.0s. **Retraining at our 1.0s @ 32kHz would invalidate paper numbers.** | **EfficientAT** |
| **Open-air range validity** | v8 being trained with room-IR + wide-gain + field recordings (Phase 20.1 noise corpora) — designed for open-air generalization | **Indoor close-mic only.** Paper makes no open-air claim. CONTEXT.md 300m+ attribution is wrong. | **EfficientAT** |
| **Pretrained weights** | AudioSet pretrained via EfficientAT mn10 (Phase 14) | **None.** No weights released. | **EfficientAT** |
| **Training cost (1 run)** | Phase 20 v7/v8 on Vertex L4 us-east1 — hours per run | Paper: 235-310 min on free-tier T4 Kaggle for binary task | AUDRON cheaper per run; neither is limiting |
| **Training dataset fit** | DADS (180K files, 60.9h) + field UMA-16 + Phase 20.1 noise corpora | DroneAudioDataset (indoor, 175 unique drone files supplementation) — **1 order of magnitude smaller drone pool** | **EfficientAT** |
| **Reimplementation risk** | Vendored and operational | From-scratch reimplementation from paper prose. STFT-CNN filter counts not disclosed. Fusion hyperparameters not disclosed. Likely ≥2 person-weeks before first training run. | **EfficientAT — 0 risk vs high risk** |
| **Ensemble diversity contribution** | Single mel-based CNN family — ensemble of 2× EfficientAT is low-diversity | MFCC + STFT + BiLSTM + AE fusion *is* genuinely diverse from mel-only CNN — this is the ONE axis AUDRON wins on | **AUDRON (niche)** |

**Bottom line:** AUDRON loses on every operational axis. Its only possibly-useful contribution is representation diversity for ensembling — and we don't need ensembling unless v8 fails the gate.

---

## Research Questions Answered

### Q1. Reproducibility — Is AUDRON a real, accessible framework?

- **Paper:** YES. arXiv 2512.20407, presented at IEEE INDICON 2025, AmygdalaAI-India Lab authors Chatterjee / Chakrabarty / Acharjee / Mishra. [VERIFIED]
- **Code:** NO. No official repo. No weights. No training config files. Reimplementation is from-paper-prose only. [VERIFIED by absence]
- **SudarshanChakra relationship:** **FALSE claim in CONTEXT.md.** The `kbhujbal/SudarshanChakra` repo is a simple single-CNN (~500K params) with no AUDRON affiliation. It uses the same DroneAudioDataset, which is likely the source of the confusion. [VERIFIED]
- **98.51% independent reproduction:** NO. Single paper, single lab, internal cross-val only. Not reproduced independently. [VERIFIED by absence]

**Confidence: HIGH.** The paper is real; the reimplementation burden and lack of code are real; the SudarshanChakra misattribution is real.

### Q2. Comparison vs EfficientAT v7/v8

See Comparison Table above. Summary:
- **Accuracy:** AUDRON 98.51% on indoor cross-val is apples-to-oranges with our UMA-16 real-device gate. EfficientAT v8's gate is `real_TPR ≥ 0.80 / real_FPR ≤ 0.05` on held-out UMA-16 field recordings — a much harder and more operationally meaningful bar.
- **Latency:** Not reported for AUDRON. EfficientAT v6 ONNX int8 already validated on RPi4.
- **Model size:** AUDRON not reported (estimate 5-15M+ params); EfficientAT mn10 ~4.5M.
- **Real-time 48kHz×16ch:** EfficientAT already ships in the Docker microservice. AUDRON would require porting ~3s @ ~16kHz → 1s @ 32kHz training and breaking the paper numbers.
- **RPi4 edge fit:** EfficientAT has ONNX int8 path shipped (Phase 21). AUDRON has no edge story — paper lists it as "future work". BiLSTM + 4 parallel branches is edge-hostile.

**Confidence: HIGH.**

### Q3. Dataset access — DroneAudioDataset

- **Repo:** https://github.com/saraalemadi/DroneAudioDataset — exists. [VERIFIED]
- **License:** Not specified in README. Associated paper is Al-Emadi et al., Qatar University (HBKU). Likely academic-use-only; would need to check the paper before any shipping. [PARTIAL]
- **Recording conditions:** **Indoor propeller recordings.** Stated verbatim in README. No mic distance, drone type, or SR documented in README. [VERIFIED]
- **Fit for our pipeline:** **POOR.** Our beamformer outputs single-channel audio captured outdoors from a UMA-16v2 array at ranges of meters to tens of meters. DroneAudioDataset's indoor close-mic propeller noise has:
  - Different SNR (indoor ~0-5m → very high; outdoor → low)
  - Different spectral character (no wind, no outdoor reverb, no distant propagation attenuation)
  - Different noise floor (indoor HVAC vs outdoor traffic/wildlife/wind)
  - No payload variation (vs our 4kg 10-inch payload recordings)
- **Overlap with our existing data:** Essentially none. Our DADS corpus (Phase 13) is already larger, outdoor-biased, and covers more drone types.

**Verdict:** DroneAudioDataset is not a useful addition to our training pool. AUDRON's entire evaluation regime is based on a dataset that does not match our deployment. **Confidence: HIGH.**

### Q4. Range claim validity — Does 300m+ hold open-air?

**The AUDRON paper does not make a 300m+ claim.** This was misattributed in CONTEXT.md. [VERIFIED by paper text review]

What we actually know about acoustic drone detection ranges (from the broader literature):
- Practical ceiling: 200-300m for typical small quadcopters in low-wind conditions; up to 500m for advanced YAMNet-based detectors on medium drones [CITED: MDPI Electronics 2024 UAV acoustic survey, Fraunhofer IDMT 2025 press release]
- Severe degradation above wind 5 m/s
- Background noise 50-80 dB masks drone signatures entirely beyond 200-300m in urban environments

**Implication:** Even if AUDRON claimed 300m (it doesn't), that would be a generic achievable-under-ideal-conditions number, not an architecture-specific advantage. Range is dominated by SNR and propagation physics, not model architecture — a better model extracts a few more dB of margin, not an order of magnitude of range.

**Confidence: HIGH.**

### Q5. Phase 20.1 overlap

Phase 20.1 acquires: ESC-50, UrbanSound8K, FSD50K 6-class subset.
AUDRON uses: DroneAudioDataset + ESC-50 + Speech Commands + synthetic harmonics + silence.

| Corpus | Phase 20.1 (v8 training) | AUDRON paper | Overlap action |
|---|---|---|---|
| ESC-50 | ✅ In use as BG noise negatives | ✅ In use as BG noise | **Redundant — no action needed**, we already have it. |
| UrbanSound8K | ✅ In use as BG noise negatives | ❌ Not used | Complementary; we have it, AUDRON doesn't. |
| FSD50K subset | ✅ In use as BG noise negatives | ❌ Not used | Complementary. |
| Speech Commands | ❌ Not in our pool | ✅ Used as "white noise" | **Gap — marginal value**. Speech Commands is not a relevant negative for drone detection; it's a low-value addition. Skip. |
| Synthetic harmonic clips | ❌ Not in our pool | ✅ Core of method (99.92% acc on synth) | **Gap — possibly valuable.** The formula `x_c(t) = (Σ A_k sin(2π f_k t)) · M(t) + η + ξ` with base 75 Hz for quadcopters is cheap to implement and could augment drone-positive class diversity. **Flag for Phase 22 / v8+1 consideration** — can be added without adopting AUDRON. |
| Silence clips | ✅ Implicit (ambient_quiet pool) | ✅ Used for class balancing | Equivalent. |
| DroneAudioDataset | ❌ Not in our pool | ✅ Primary drone source | Skip (indoor, poor fit — see Q3). |

**Verdict:** **No redundancy. Phase 20.1 is not disrupted. The only possibly-useful AUDRON-derived idea is synthetic harmonic drone augmentation**, which is an independent technique that can be adopted without adopting the AUDRON architecture. **Confidence: HIGH.**

### Q6. Training feasibility on our hardware

Paper: 235-310 minutes on NVIDIA T4 (Kaggle free tier) for binary task. T4 ≈ 8 TFLOPs FP16, 16GB VRAM.
Our baseline: Phase 20 used Vertex L4 us-east1 for v7; Phase 22 locked to L4 us-east1 for v8. L4 ≈ 30 TFLOPs FP16, 24GB VRAM — roughly 4× faster than T4 on vision workloads, more on mixed precision.

**Estimate for a hypothetical AUDRON retraining run on our data at our window contract (1.0s @ 32kHz):**
- T4-equivalent: ~4-5 hours (paper's 5-hour augmented binary run, but on smaller dataset)
- L4: ~1-2 hours if compute-bound
- DADS is ~180K files vs AUDRON's ~9K training samples → dataset is ~20× larger → **actual runtime 20-40 hours on L4 per experiment**, dominated by I/O and augmentation, not raw compute
- BiLSTM branch adds significant sequential compute even on L4

**Cost:** L4 spot on us-east1 ~$0.30-0.60/hr → $6-24 per run. Manageable, not limiting.

**Blocker is not compute, it's engineering time:** reimplementing the 4-branch fusion from paper prose (no code released), fighting unreported STFT-CNN filter counts, tuning the fusion head, porting from ~3s @ ~16kHz to 1s @ 32kHz (invalidating paper numbers), and wiring into our vendored training harness. Estimate **2-4 person-weeks before the first successful training run**, vs Phase 22 v8 which is a one-line literal fix on a proven harness.

**Confidence: HIGH on direction (training is cheap, reimplementation is expensive); MEDIUM on exact hours.**

### Q7. Integration fit — swap, ensemble, or parallel research?

Three options:

| Option | Viability | Conditions |
|---|---|---|
| **Swap EfficientAT for AUDRON** | ❌ REJECT | No pretrained weights, no code, larger model, no edge story, no real-device validation, no open-air validation. Would throw away Phase 14-22 work. |
| **Ensemble: EfficientAT + AUDRON** | ⚠️ CONDITIONAL | Only defensible if (a) v8 fails the real-device gate AND (b) the failure mode is representation diversity (not window contract, label noise, or data paucity). Ensembling a broken model doesn't fix it. |
| **Parallel research track** | ✅ PERMISSIBLE POST-v8 | After v8 lands and meets the gate, AUDRON can be reimplemented in a research branch (not main pipeline) to compare on the same UMA-16 hold-out. Outcome informs whether to invest in ensembling in milestone v2. |

**Recommended option: parallel research track, gated on v8 success. Do not start now.**

**Confidence: HIGH.**

---

## Architecture Patterns (if hybridizing in future)

Documented for reference only; not prescriptive for this phase's deliverable.

### Multi-branch fusion pattern
- Each branch is a feature extractor producing a fixed-dim embedding
- Concatenate embeddings → single dense fusion head with Dropout + BatchNorm → classifier
- Key principle: **branches must be genuinely diverse** (different transforms, different inductive biases). Two mel-spec CNNs ensembled = minimal diversity gain.

### AUDRON's branch choices (as pattern catalog)
| Branch | Transform | Model | Inductive bias captured |
|---|---|---|---|
| MFCC | 13 MFCCs | 2× Conv1d → 128-d | Compact spectral texture (speech-recognition heritage) |
| STFT-CNN | STFT spectrogram | 4× Conv2d blocks | Hierarchical 2D spectro-temporal patterns |
| RNN | Raw waveform chunks | BiLSTM + attention | Long-range temporal dependencies |
| Autoencoder | 48,000-sample waveform | Encoder → 160-d | Self-supervised / reconstruction-based representation |

### Standard fusion head (from paper)
```
concat([mfcc_emb_128, stft_cnn_emb, lstm_emb, ae_emb_160])  # → 736-d
→ Linear(736 → hidden)
→ BatchNorm1d
→ Dropout
→ Linear(hidden → num_classes)
```

### Anti-patterns to avoid if reimplementing
- **Don't** train branches independently then freeze and train only the head — end-to-end is standard for modern fusion and typically wins.
- **Don't** use the paper's ~3s @ ~16kHz input if our pipeline is 1s @ 32kHz — the entire feature extraction stack must be re-tuned. Retraining at our window contract makes AUDRON's 98.51% number irrelevant.
- **Don't** hand-roll the BiLSTM attention block — use `torch.nn.MultiheadAttention` or HuggingFace `TransformerEncoder`.

---

## Standard Stack (if reimplementing in a research track)

All already in project per CLAUDE.md — no new dependencies needed.

| Library | Version | Use |
|---|---|---|
| PyTorch | ≥2.11.0 | Training + inference |
| torchaudio | ≥2.11.0 | `MFCC`, `Spectrogram`, `MelSpectrogram` transforms as nn.Modules — replaces librosa [VERIFIED: docs.pytorch.org/audio] |
| NumPy | ≥1.26,<3 | DSP glue |
| SciPy | ≥1.14 | Bandpass filters, windowing |

**Do NOT add new libraries.** The AUDRON paper uses nothing exotic.

### `torchaudio` transforms that replace hand-rolled code
```python
# MFCC branch input — don't hand-roll, use torchaudio
import torchaudio.transforms as T
mfcc = T.MFCC(sample_rate=32000, n_mfcc=13, melkwargs={"n_fft": 1024, "hop_length": 320, "n_mels": 64})
# STFT-CNN branch input
stft = T.Spectrogram(n_fft=1024, hop_length=320, power=2.0)
```

Source: https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html [CITED]

---

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---|---|---|---|
| MFCC extraction | Custom DCT-on-log-mel | `torchaudio.transforms.MFCC` | Differentiable, GPU-accelerated, well-tested, matches librosa numerics |
| STFT spectrogram | Custom `np.fft` + windowing loop | `torchaudio.transforms.Spectrogram` | Same reasons; also handles windowing correctly |
| BiLSTM + attention | Hand-wiring `nn.LSTM` output into attention | `nn.LSTM(bidirectional=True)` + `nn.MultiheadAttention` | Standard, batched, CUDA-optimized |
| Autoencoder | Custom encoder/decoder from scratch | `nn.Conv1d` blocks or HuggingFace `AutoModel` from audio-SSL zoo (wav2vec2, HuBERT, BEATs) | If you want a self-supervised audio embedding, **use a pretrained SSL model rather than training a 160-d AE from scratch** — BEATs / wav2vec2-base give much richer embeddings for free |
| Synthetic drone audio | Ad-hoc scripts | `torch.sin` + amplitude envelope; reproduce AUDRON formula `Σ A_k sin(2π f_k t) · M(t) + η + ξ` | Cheap; AUDRON's formula is 10 lines |
| Focal loss | Custom implementation | Project already has it (Phase 15) | Reuse |
| Class balancing | Custom sampler | Project already uses `WeightedRandomSampler` (fixed in 260407-nir) | Reuse |

**Key insight:** If we ever reimplement AUDRON, the 4-branch structure is 90% off-the-shelf torchaudio + torch.nn. The only bespoke code is the fusion head. Do not let a reimplementation become a DSP rewrite.

---

## Common Pitfalls

### Pitfall 1: Indoor-vs-outdoor training/deployment mismatch
**What goes wrong:** Model trained on DroneAudioDataset (indoor close-mic propeller noise) gets 98% cross-val, drops to ~50-60% on UMA-16 outdoor field recordings.
**Root cause:** Indoor recordings have no outdoor reverb, no wind, no distance attenuation, no traffic/ambient background at the levels a field microphone sees. The spectral statistics are completely different.
**Prevention:** Evaluate on real UMA-16 field data (the 2026-04-08 recordings) as the only trustworthy gate. Phase 22's D-27 promotion gate already enforces this for v8.
**Warning sign:** Excellent val accuracy, poor production accuracy — **exactly the v7 regression pattern**, but for a different reason.

### Pitfall 2: Reported accuracy on imbalanced binary is misleading
**What goes wrong:** AUDRON's binary task is 1,066 drone vs 8,297 noise. A dumb "always predict noise" classifier gets 88.6% accuracy. 98.51% looks less impressive in context.
**Prevention:** Always report precision/recall/F1 separately. AUDRON does this (recall 0.9840). The number is legitimate but should not be compared to balanced-class accuracies.
**Warning sign:** Single accuracy number without class distribution or confusion matrix.

### Pitfall 3: No public code → reimplementation drift
**What goes wrong:** Paper describes "4 Conv2d blocks" for STFT-CNN without filter counts, strides, or kernel sizes. Reimplementer guesses; model behavior diverges from paper.
**Prevention:** If reimplementing, treat paper numbers as **unachievable ceilings**, not targets. Validate against our own baseline (EfficientAT v8) on our own hold-out.
**Warning sign:** Spending days chasing the 98.51% number on indoor data instead of running the real-device gate.

### Pitfall 4: BiLSTM on edge
**What goes wrong:** BiLSTM is sequential — cannot be parallelized across timesteps. On RPi4 CPU, inference latency scales linearly with sequence length and is 10-100× slower than Conv-only models for the same receptive field.
**Prevention:** If ensembling AUDRON with EfficientAT for edge, AUDRON must run on the Docker host, not the RPi4. Phase 21 RPi4 deployment stays EfficientAT-only.
**Warning sign:** "Let's just port AUDRON to RPi4" — don't.

### Pitfall 5: Synthetic-only training bias
**What goes wrong:** AUDRON reports 99.92% on synthetic data. Synthetic harmonic waveforms are easy — the model learns "is there a stable harmonic stack" not "is this a drone". Transferring to real drones that have motor vibration, wind modulation, and background clutter drops performance hard.
**Prevention:** Synthetic data is useful as **augmentation**, never as the primary training pool.
**Warning sign:** High synthetic accuracy used to justify deployment readiness.

### Pitfall 6: "Production-grade variant" confusion
**What goes wrong:** CONTEXT.md attributes the SudarshanChakra repo as a production AUDRON variant. It isn't. If a reader trusts the misattribution and clones SudarshanChakra expecting AUDRON, they get a completely different architecture (simple 4-block CNN) with no bearing on the research question.
**Prevention:** Always verify claimed lineage before trusting it.
**Warning sign:** "Production-grade" claims with no commit log, contributor overlap, or paper citation linking to the upstream work.

---

## Dataset Analysis

### Phase 20.1 corpora vs AUDRON datasets — gap table

| Dataset | Our pool | AUDRON | Recording regime | Action |
|---|---|---|---|---|
| DADS (180K, 60.9h) | ✅ Phase 13 | ❌ | Mixed, outdoor-biased | Keep as primary drone source |
| UMA-16 field (2026-04-08) | ✅ Phase 22 | ❌ | Real-device outdoor | Keep as hold-out gate |
| ESC-50 | ✅ Phase 20.1 | ✅ | Environmental sounds, 5s clips | Already have. No action. |
| UrbanSound8K | ✅ Phase 20.1 | ❌ | Urban ambient | Keep. Better than Speech Commands. |
| FSD50K 6-class subset | ✅ Phase 20.1 | ❌ | FreeSound crowd-sourced | Keep. Complementary. |
| DroneAudioDataset (Al-Emadi) | ❌ | ✅ (primary) | **Indoor close-mic propeller** | **Skip** — poor regime fit. Our DADS is larger and more relevant. |
| Speech Commands | ❌ | ✅ (as "white noise") | Voice commands | **Skip** — no relevance to drone detection. |
| Synthetic harmonics | ❌ | ✅ (260+80 samples) | Generated | **Consider adding as augmentation** — cheap to implement; does not require AUDRON adoption. Could go into Phase 22 v8 or a v8+1 task. |
| Silence clips | ✅ (implicit) | ✅ | — | Equivalent. |

### Synthetic harmonic augmentation — decoupled recommendation

Even if we reject AUDRON wholesale, the synthetic harmonic generator is a standalone technique worth evaluating. Formula:

```
x_c(t) = (Σ_{k=1..K} A_k · sin(2π f_k t)) · M(t) + η(t) + ξ(t)
# f_k = k × 75 Hz (quadcopter base)
# M(t) = amplitude modulation envelope (blade-passing frequency)
# η(t) = background noise
# ξ(t) = sensor noise
```

**Cost:** ~30 lines of NumPy, runs during dataloading.
**Potential value:** Class-balance the positive class without recording more flights. Adds harmonic-structure prior.
**Risk:** Pitfall 5 (synthetic bias) if overweighted.
**Recommendation:** Park as a candidate augmentation for a post-v8 improvement phase. Do not block v8.

---

## Runtime State Inventory

**Not applicable.** Phase 23 is a research/decision phase. No code changes, no stored data, no config changes, no services touched.

| Category | Result |
|---|---|
| Stored data | None — research only |
| Live service config | None — research only |
| OS-registered state | None — research only |
| Secrets/env vars | None — research only |
| Build artifacts | None — research only |

---

## Environment Availability

**Not applicable for this phase.** Deliverable is a decision document, not code. No external tools probed.

Follow-on phases (if any) would depend on standard project stack already verified in prior phases:
- PyTorch ≥2.11, torchaudio ≥2.11 — present
- Vertex L4 us-east1 — verified in Phase 20 and Phase 22 planning
- DroneAudioDataset download — would require new probe if ever adopted (Kaggle or GitHub-hosted, a few hundred MB)

---

## Validation Architecture

**Not applicable for this phase.** Research output is a decision document. No tests, no runtime behavior to validate.

If a follow-on reimplementation phase is ever spawned, it would reuse the existing project test infrastructure (pytest, the Phase 20-06 eval harness, and the D-27 real-device promotion gate).

---

## Security Domain

**Not applicable for this phase.** No user data, no network endpoints, no credentials, no code changes. Pure research and documentation.

The general project security posture (Dockerized microservice, ZeroMQ PUB/SUB, UMA-16v2 USB passthrough) is unchanged by this research and is documented elsewhere.

---

## Code Examples

Reference patterns if AUDRON is ever reimplemented in a research branch. None of this runs in the main pipeline.

### Minimal 4-branch fusion skeleton
```python
# Source: adapted from arxiv.org/html/2512.20407v1 paper prose
import torch
import torch.nn as nn
import torchaudio.transforms as T

class MFCCBranch(nn.Module):
    def __init__(self, sr=32000):
        super().__init__()
        self.mfcc = T.MFCC(sample_rate=sr, n_mfcc=13,
                           melkwargs={"n_fft": 1024, "hop_length": 320, "n_mels": 64})
        self.conv = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, wav):                 # wav: (B, T)
        x = self.mfcc(wav)                   # (B, 13, frames)
        return self.conv(x).squeeze(-1)      # (B, 128)

class STFTCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = T.Spectrogram(n_fft=1024, hop_length=320, power=2.0)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, wav):
        x = self.spec(wav).unsqueeze(1)      # (B, 1, F, T)
        return self.cnn(x).flatten(1)        # (B, 256)

class RNNBranch(nn.Module):
    def __init__(self, sr=32000):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sr, n_mels=64, hop_length=320)
        self.lstm = nn.LSTM(64, 96, bidirectional=True, batch_first=True)
        self.attn = nn.MultiheadAttention(192, num_heads=4, batch_first=True)
    def forward(self, wav):
        x = self.mel(wav).transpose(1, 2)    # (B, T, 64)
        x, _ = self.lstm(x)                  # (B, T, 192)
        x, _ = self.attn(x, x, x)
        return x.mean(dim=1)                 # (B, 192)

class AEBranch(nn.Module):
    """Use a pretrained SSL model instead of a custom AE — see Don't Hand-Roll."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=4, padding=4), nn.ReLU(),
            nn.Conv1d(64, 160, 9, stride=4, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, wav):
        return self.enc(wav.unsqueeze(1)).squeeze(-1)   # (B, 160)

class AUDRONFusion(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mfcc = MFCCBranch()
        self.stft = STFTCNNBranch()
        self.rnn  = RNNBranch()
        self.ae   = AEBranch()
        # 128 + 256 + 192 + 160 = 736 (happens to match paper)
        self.head = nn.Sequential(
            nn.Linear(736, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, wav):
        z = torch.cat([self.mfcc(wav), self.stft(wav), self.rnn(wav), self.ae(wav)], dim=-1)
        return self.head(z)
```

**Reality check on this skeleton:** The exact filter counts are guesses — the paper does not disclose them. Any reimplementation that tries to hit 98.51% on DroneAudioDataset will burn days tuning these. Don't do that. Tune to our UMA-16 hold-out instead.

---

## Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|---|---|---|
| A1 | AUDRON's binary task is class-imbalanced (1,066 vs 8,297) and 98.51% should be interpreted with that in mind | Executive Rec, Comparison Table | LOW — numbers quoted verbatim from paper; interpretation is standard ML hygiene |
| A2 | BiLSTM is edge-hostile on RPi4 compared to Conv-only mn10 | Comparison Table, Pitfall 4 | LOW — well-known ARM CPU behavior; not specific to RPi4 |
| A3 | Reimplementation effort is 2-4 person-weeks before first successful training run | Q6, Executive Rec | MEDIUM — actual effort depends on how closely the reimplementer hews to the paper; could be 1 week if synthetic-only validation is accepted |
| A4 | AUDRON total param count is 5-15M+ | Comparison Table | MEDIUM — genuinely unknown; paper doesn't report. Estimate based on 4 parallel branches + BiLSTM. Could be smaller if STFT-CNN filter counts are tiny. Does not change the recommendation. |
| A5 | L4 us-east1 cost of ~$0.30-0.60/hr spot | Q6 | LOW — market rates; verified in Phase 20/22 planning |
| A6 | EfficientAT v8 will pass the real-device gate | Executive Rec (conditional on this) | MEDIUM — Phase 22 is in flight; if v8 fails the gate the recommendation "reject AUDRON now, revisit if v8 fails" still holds, it just reactivates sooner |

**Claims flagged `[ASSUMED]` in prose:** None. All factual claims are either verified against the paper/repo or explicitly labeled as estimates in this Assumptions Log.

---

## Open Questions

1. **Did Phase 22 v8 pass the D-27 real-device gate?**
   - What we know: Phase 22 is in flight, planning complete, execution ongoing.
   - What's unclear: v8 training outcome.
   - Recommendation: This research's recommendation (reject AUDRON now) is stable regardless. Revisit only if v8 fails.

2. **Would the synthetic harmonic augmentation actually help v8+1?**
   - What we know: Cheap to implement; AUDRON reports 99.92% on synth (meaningless alone) but the technique is standard data aug.
   - What's unclear: Whether our existing wide-gain + room-IR + field-data pipeline already saturates the augmentation benefit curve.
   - Recommendation: Park as a candidate quick task after v8 ships. Do not block anything on it.

3. **Is DroneAudioDataset licensed for our use?**
   - What we know: Repo public; README does not specify a license.
   - What's unclear: Default GitHub terms (view-only, no redistribution). Associated conference paper would have author intent.
   - Recommendation: Only resolve this question if we actually adopt the dataset — which the recommendation says we shouldn't.

---

## Follow-On Phases

**Recommended decision: REJECT now, parallel research track permissible later.**

### If v8 passes the gate (expected path)
**No follow-on phases.** Ship v8 (Phase 22) and the RPi4 edge app (Phase 21). Archive this research as the "we looked at AUDRON and here's why it's not now" record.

### If v8 fails the gate
Spawn a diagnostic phase first, not an AUDRON phase:
1. **Phase 24 (hypothetical): v8 failure diagnosis.** Determine whether failure is architecture-limited, data-limited, or pipeline-limited. Do not jump to architecture replacement.
2. **Phase 25 (conditional on Phase 24 finding architecture-limit):** Reimplement AUDRON in a research branch, train on our data, evaluate on the same UMA-16 hold-out. Gate: does it beat v8's failing metric?
3. **Phase 26 (conditional on Phase 25 showing diversity value):** Ensemble EfficientAT + AUDRON on the Docker host. RPi4 stays EfficientAT-only.

### Conditions to revisit AUDRON (if we reject now)
- v8 fails the real-device gate AND Phase 24 diagnosis points at representation limits
- A public AUDRON code release with pretrained weights appears
- An independent third-party reproduction validates the 98.51% on a more relevant dataset
- Ensemble research elsewhere shows strong gains from combining mel-CNN + multi-branch fusion for drone detection specifically

### Standalone opportunity (decoupled from AUDRON)
**Quick task candidate (post-v8):** Add synthetic harmonic drone augmentation (AUDRON's Eq. 1 formula) as a dataloader augmentation in the EfficientAT trainer. ~1 day work. Does NOT require adopting AUDRON. Flagged here so the idea is not lost.

---

## State of the Art

| Old approach | Current approach | When | Impact |
|---|---|---|---|
| Custom 3-layer CNN on mel-spec (our Phase 3) | EfficientAT mn10 AudioSet pretrained (our Phase 14+) | 2026-04 | Our existing upgrade path — this is where we are |
| Mel-CNN only | Multi-branch fusion (MFCC+STFT+BiLSTM+AE) | AUDRON INDICON 2025 | New architectural proposal; not yet reproduced independently |
| Supervised-only SSL representations | Pretrained audio SSL (BEATs, wav2vec2, HuBERT) as embedding source | 2024-2025 | If we ever wanted an "autoencoder branch", a pretrained SSL model is strictly better than training a 160-d AE from scratch |
| Indoor propeller datasets only | Real-device + field-collected + augmented datasets | 2024+ | Our approach (Phase 20.1 noise corpora + 2026-04-08 field recordings + DADS) — AUDRON's dataset choice is backward-looking |

**Deprecated / outdated for our use case:**
- **Training on indoor-only datasets and expecting open-air performance** — deprecated by the broader literature (MDPI 2024, Fraunhofer IDMT 2025). The field has moved to outdoor-validated models.
- **Custom autoencoder branches** — superseded by pretrained audio SSL models (BEATs, wav2vec2). If we wanted the "self-supervised representation" angle, use those.

---

## Sources

### Primary (HIGH confidence)
- **AUDRON paper (full HTML):** https://arxiv.org/html/2512.20407v1 — architecture, training config, accuracy/precision/recall/F1 numbers, dataset composition, training times, author affiliations
- **AUDRON paper (arXiv landing):** https://arxiv.org/abs/2512.20407 — metadata, INDICON 2025 venue
- **AmygdalaAI-India Lab:** https://amygdalaaiindia.github.io/ — paper author affiliation
- **SudarshanChakra repo:** https://github.com/kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN — actual content (simple 4-block CNN, ~500K params, 95.23% acc), NOT an AUDRON variant
- **DroneAudioDataset repo:** https://github.com/saraalemadi/DroneAudioDataset — confirms indoor-propeller recording regime
- **Project phase 22 CONTEXT.md:** `.planning/phases/22-.../CONTEXT.md` — EfficientAT v8 training regime, D-27 real-device gate
- **v7 regression post-mortem:** `.planning/debug/efficientat-v7-regression-vs-v6.md` — EfficientAT baseline metrics and window contract fix
- **Project CLAUDE.md:** root — locked tech stack (PyTorch ≥2.11, torchaudio ≥2.11, no new deps)

### Secondary (MEDIUM confidence)
- **MDPI Electronics UAV acoustic detection survey:** https://www.mdpi.com/2079-9292/13/3/643 — 200-500m practical range limits for acoustic drone detection
- **Fraunhofer IDMT 2025 press release:** https://www.idmt.fraunhofer.de/en/Press_and_Media/press_releases/2025/acoustic-drone-detection.html — reliable acoustic drone detection state-of-practice
- **torchaudio MFCC docs:** https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html — transform API reference
- **torchaudio MelSpectrogram docs:** https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html

### Tertiary (LOW confidence / flagged)
- **300m range claim attribution to AUDRON in CONTEXT.md:** NOT FOUND IN PAPER. Discard.
- **SudarshanChakra = AUDRON production variant claim in CONTEXT.md:** CONTRADICTED BY REPO. Discard.

---

## Metadata

**Confidence breakdown:**
- Source verification: HIGH — paper, lab, repos all located and inspected
- SudarshanChakra misattribution: HIGH — verified by reading the repo directly
- 300m range claim falsification: HIGH — verified by full paper text extraction
- AUDRON vs EfficientAT comparison: HIGH on direction, MEDIUM on exact param counts (AUDRON count not reported)
- Reimplementation effort estimate: MEDIUM — depends on team and how strictly paper-faithful
- Reject-now recommendation: HIGH — holds under all reasonable uncertainty about A3/A4

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (one month — re-check if AUDRON authors release code or weights; re-check if independent reproduction appears)

**Open-item watchlist for auto-invalidation:**
- Public code release at an AmygdalaAI-India GitHub org
- Third-party reproduction on a non-indoor dataset
- EfficientAT v8 failing the Phase 22 real-device gate
