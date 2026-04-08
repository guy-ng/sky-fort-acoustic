# Phase 23 — Evaluate AUDRON Multi-Branch Hybrid Architecture

## Origin

User-proposed alternative training approach. Source: AmygdalaAI-India Lab AUDRON
(AUdio-based Drone Recognition Network) framework. Phase entered the roadmap as
**research-first** — the deliverable is an adopt/reject/hybrid decision, NOT a
trained model. Implementation phases (if any) are spawned downstream.

## Approach Under Evaluation

Multi-branch hybrid deep learning that fuses 4 parallel feature extractors into
a 736-dim fusion head:

| Branch | Method | Output |
|--------|--------|--------|
| MFCC | 13 MFCCs → 2× 1D-CNN | 128-dim spectral texture |
| STFT-CNN | STFT spectrograms → 4× 2D-CNN blocks | hierarchical spectro-temporal patterns |
| RNN | Bidirectional LSTM + Attention | long-range temporal dependencies |
| Autoencoder | Self-supervised encoder, 48k samples in | 160-dim latent |

**Fusion head:** concat (128 + STFT-CNN out + LSTM out + 160) → 736-dim dense
layer with Dropout + BatchNorm → binary classifier.

## Claimed Training Configuration

- Datasets: DroneAudioDataset (indoor propellers) + ESC-50 (ambient noise) +
  Speech Commands (white noise) + synthetic harmonic waveforms + silence clips
- Synthetic generation formula:
  `x_c(t) = (Σ_{k=1..K_c} A_k · sin(2π f_k t)) · M(t) + η(t) + ξ(t)`
  where `f_k` is harmonic frequency (base ≈ 75 Hz for quadcopters),
  `M(t)` is amplitude modulation, `η(t)` is background noise, `ξ(t)` is sensor noise
- Optimizer: AdamW, lr=0.001
- Batch size: 16
- Epochs: up to 50
- LR scheduler: ReduceLROnPlateau, patience=5 (val accuracy)
- Hardware: NVIDIA T4 (Kaggle/Colab)
- Class balancing via custom silence clips
- Claimed accuracy: 98.51%
- Claimed range: 300m+ detection
- Related repo: SudarshanChakra (production-grade variant)

## Research Questions (must answer all)

1. **Reproducibility:** Is AUDRON a real, accessible framework? Locate the
   AmygdalaAI-India Lab repo or paper. Verify SudarshanChakra relationship.
   Are the 98.51% accuracy and 300m+ claims independently reproducible?
2. **Comparison vs EfficientAT v7/v8:** accuracy, latency, model size, real-time
   suitability on 48 kHz × 16-ch UMA-16v2 stream, Raspberry Pi 4 edge fit
   (phase 21 target).
3. **Dataset access:** DroneAudioDataset license, availability, fit for our
   beamformer single-channel output (vs raw indoor propeller recordings).
4. **Range claim validity:** Does 300m+ hold open-air, or only for indoor
   propeller close-mic recordings?
5. **Phase 20.1 overlap:** ESC-50, UrbanSound8K, FSD50K acquisition is in
   flight — what overlaps, what's redundant, what's complementary?
6. **Training feasibility on our hardware:** NVIDIA T4 baseline vs available
   compute. Time/cost estimate for 50 epochs.
7. **Integration fit:** Does this swap the EfficientAT classifier, run in
   ensemble, or live as a parallel research track?

## Constraints (do not violate)

- **In-flight work:** Phase 22 (EfficientAT v8 retrain) and phase 20.1
  (noise corpora acquisition) must NOT be disrupted.
- **Real-time:** 48 kHz × 16-ch pipeline must keep up.
- **Edge target:** Raspberry Pi 4 (phase 21).
- **Deployment:** Dockerized Python microservice.
- **ML stack:** PyTorch ≥ 2.11 (per CLAUDE.md tech stack).

## Success Criteria

1. `RESEARCH.md` produced with:
   - Adopt / Reject / Hybrid recommendation
   - Confidence level (HIGH/MEDIUM/LOW) with evidence
   - Comparison table: AUDRON vs EfficientAT v8 across accuracy, latency,
     model size, edge fit, training cost
2. If **adopt** or **hybrid** → list of follow-on phases needed
3. If **reject** → documented reasons + conditions under which we'd revisit
4. Verification of every cited source (no hallucinated repos/papers)

## Out of Scope

- Actually training AUDRON (deferred to follow-on phase if adopted)
- Replacing EfficientAT v8 retrain in-flight
- Hardware procurement
