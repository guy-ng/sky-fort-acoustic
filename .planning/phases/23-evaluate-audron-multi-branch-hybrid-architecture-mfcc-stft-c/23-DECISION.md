# Phase 23 Decision: AUDRON Multi-Branch Hybrid Architecture

**Decision:** REJECT as classifier replacement. Parallel research track permissible post-v8 gate.
**Confidence:** HIGH
**Date:** 2026-04-08
**Evidence:** 23-RESEARCH.md (full analysis)

## Decision Summary

AUDRON is a real, verifiable paper (arXiv 2512.20407, AmygdalaAI-India Lab, IEEE INDICON 2025) with a coherent 4-branch architecture (MFCC 1D-CNN + STFT 2D-CNN + BiLSTM-Attention + Autoencoder fusion), but its headline 98.51% accuracy comes from binary classification on a heavily imbalanced indoor close-mic dataset (DroneAudioDataset: 1,066 drone vs 8,297 noise), which does not match our outdoor UMA-16v2 deployment regime. There is no public code, no pretrained weights, and no edge deployment story -- reimplementation would be from-scratch from paper prose, estimated at 2-4 person-weeks before a first training run. Our EfficientAT mn10 v6/v8 line is already the correct architectural choice: ~4.5M params, AudioSet-pretrained, proven on UMA-16 field recordings, with an ONNX int8 edge path already built for RPi4 (Phase 21). AUDRON loses on every operational axis except ensemble diversity, which is its only possibly-useful contribution -- and we do not need ensembling unless v8 fails the real-device gate. The two CONTEXT.md claims (300m+ detection range and SudarshanChakra as a production AUDRON variant) are both factually incorrect.

## Comparison Snapshot

| Axis | Winner | Key Reason |
|---|---|---|
| Headline accuracy | Tie on paper; EfficientAT on what matters | AUDRON 98.51% is indoor cross-val only; EfficientAT v8 gate is real-device TPR/FPR |
| Evaluation regime | EfficientAT | Trained/tested on real UMA-16 field recordings vs indoor close-mic |
| Parameter count | EfficientAT | ~4.5M known vs AUDRON unreported (estimated 5-15M+) |
| Model size on disk | EfficientAT | ~17 MB state_dict vs AUDRON estimated 20-60 MB |
| Inference latency | EfficientAT | Real-time in Docker + ONNX int8 RPi4 vs AUDRON not reported; BiLSTM is sequential |
| RPi4 edge fit | EfficientAT (not close) | ONNX int8 shipped; AUDRON has no edge story, BiLSTM is edge-hostile |
| Real-time 48kHz x 16-ch fit | EfficientAT | 1.0s @ 32kHz contract matches; AUDRON uses ~3s @ ~16kHz, retraining invalidates paper numbers |
| Open-air range validity | EfficientAT | v8 trains with room-IR + field recordings; AUDRON is indoor-only, no range claim |
| Pretrained weights | EfficientAT | AudioSet pretrained available; AUDRON has none |
| Training cost (1 run) | Neither limiting | AUDRON cheaper per run on T4; neither is a bottleneck |
| Training dataset fit | EfficientAT | DADS 180K files / 60.9h vs DroneAudioDataset ~1K drone files indoor |
| Reimplementation risk | EfficientAT (0 risk vs high) | Vendored and operational vs from-scratch reimplementation from paper prose |
| Ensemble diversity contribution | AUDRON (niche) | Multi-branch fusion is genuinely diverse from mel-only CNN family |

## CONTEXT.md Claims Corrected

| Claim | Status | Correction |
|---|---|---|
| 300m+ detection range | FALSE | AUDRON paper makes no range claim. 300m is from unrelated acoustic detection surveys (MDPI Electronics 2024, Fraunhofer IDMT 2025). |
| SudarshanChakra = production AUDRON variant | FALSE | SudarshanChakra is a simple 4-block CNN (~500K params, 95.23% accuracy), no AUDRON affiliation. Author `kbhujbal` has no public link to the AUDRON authors. Uses the same DroneAudioDataset, which is likely the source of the confusion. |

## Reject Reasons

1. No public code or pretrained weights exist -- reimplementation from paper prose alone is estimated at 2-4 person-weeks before a first training run.
2. The 98.51% accuracy was evaluated on an indoor close-mic dataset (DroneAudioDataset) that does not transfer to our outdoor UMA-16v2 beamformer regime.
3. AUDRON has no edge deployment story -- the paper lists low-power edge devices as "future work", and the BiLSTM branch is fundamentally edge-hostile on RPi4 CPU.
4. EfficientAT mn10 v6/v8 already wins on every operational axis (latency, model size, edge fit, pretrained weights, real-device validation, reimplementation risk).
5. The only axis AUDRON wins on (ensemble diversity) is not needed unless v8 fails the real-device gate, which has not happened.

## Revisit Conditions

- v8 fails the real-device gate (TPR >= 0.80 / FPR <= 0.05) AND Phase 24 diagnosis points at representation limits (not window contract, label noise, or data paucity)
- A public AUDRON code release with pretrained weights appears (watch AmygdalaAI-India GitHub org)
- An independent third-party reproduction validates the 98.51% on a more relevant (non-indoor) dataset
- Ensemble research elsewhere shows strong gains from combining mel-CNN + multi-branch fusion for drone detection specifically

## Decoupled Opportunity

**Synthetic harmonic drone augmentation** -- standalone technique from AUDRON's Eq. 1 formula:

```
x_c(t) = (sum_{k=1..K} A_k * sin(2*pi*f_k*t)) * M(t) + eta(t) + xi(t)
# f_k = k * 75 Hz (quadcopter base), M(t) = amplitude modulation, eta = background noise, xi = sensor noise
```

~30 lines of NumPy, ~1 day work. Does NOT require adopting AUDRON. Adds harmonic-structure prior for class-balancing the drone-positive class. Risk: synthetic bias if overweighted (see RESEARCH.md Pitfall 5). Candidate quick task post-v8. See RESEARCH.md "Synthetic harmonic augmentation -- decoupled recommendation" section.

## Source Verification Status

| URL | Status | Checked |
|---|---|---|
| https://arxiv.org/abs/2512.20407 | OK (200) | 2026-04-08 |
| https://arxiv.org/html/2512.20407v1 | OK (200) | 2026-04-08 |
| https://amygdalaaiindia.github.io/ | OK (200) | 2026-04-08 |
| https://github.com/kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN | OK (200) | 2026-04-08 |
| https://github.com/saraalemadi/DroneAudioDataset | OK (200) | 2026-04-08 |
| https://www.mdpi.com/2079-9292/13/3/643 | OK (403 bot-protection, page exists) | 2026-04-08 |
| https://www.idmt.fraunhofer.de/en/Press_and_Media/press_releases/2025/acoustic-drone-detection.html | OK (200) | 2026-04-08 |
| https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html | OK (200) | 2026-04-08 |
| https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html | OK (200) | 2026-04-08 |

All 9 sources are accessible. The MDPI URL returns HTTP 403 due to automated-request bot protection but the page is confirmed to exist (the journal article is publicly listed and accessible via browser).

## Follow-On Phases

**If v8 passes gate (expected):** No follow-on phases. Archive this research as the "we evaluated AUDRON and here is why it is not needed now" record.

**If v8 fails gate:** Phase 24 (v8 failure diagnosis -- determine whether failure is architecture-limited, data-limited, or pipeline-limited) --> Phase 25 (conditional AUDRON reimplement in research branch, train on our data, evaluate on same UMA-16 hold-out) --> Phase 26 (conditional ensemble: EfficientAT + AUDRON on Docker host, RPi4 stays EfficientAT-only). See RESEARCH.md "Follow-On Phases" section.
