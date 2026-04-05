# Acoustic drone detection with CNNs for edge deployment

**A MobileNet-class CNN trained on mel spectrograms, pretrained on AudioSet, and quantized for TensorRT/TFLite delivers the best accuracy-to-latency ratio for real-time drone detection on Jetson and Raspberry Pi hardware.** Binary detection accuracy above 95% is achievable with public data alone, using a model under 5M parameters that infers in under 30ms on a Jetson Nano. The critical path to a working system involves aggregating the DADS and DroneAudioSet datasets (~84 hours combined), extracting 64-band log-mel spectrograms, fine-tuning an EfficientAT MobileNetV3 backbone, applying INT8/FP16 quantization, and deploying a sliding-window inference loop. This report covers every component of that pipeline in depth—from raw audio to edge alert—with architecture benchmarks, training recipes, and deployment specifics.

---

## 1. Public datasets for acoustic drone detection

The field suffers from dataset fragmentation, but several high-quality public resources now exist. The largest and most practical are listed below with full details.

### Primary drone-specific datasets

**DADS (Drone Audio Detection Samples)** is the largest aggregated public dataset, hosted on HuggingFace (`geronimobasso/drone-audio-detection-samples`). It contains **180,320 files totaling ~60.9 hours**—163,591 drone clips (27.1 hours) and 16,729 no-drone clips (33.8 hours). All files are WAV, PCM 16-bit, mono, 16 kHz. It aggregates audio from 6 drone sources and 4 non-drone sources under an MIT license. This is the recommended starting point for binary detection.

**DroneAudioSet** (Gupta et al., 2025; `augmented-human-lab/DroneAudioSet-code`) provides **23.5 hours** of systematically controlled recordings with SNR ranging from −57.2 to −2.5 dB. It captures multiple drone types at varying throttle settings, microphone configurations, and acoustic environments. The controlled recording conditions make it ideal for evaluating model robustness across distances and noise levels.

**DroneAudioDataset** (Al-Emadi et al., 2019; `saraalemadi/DroneAudioDataset`) is the most widely cited drone audio dataset. It covers Parrot Bebop and Parrot Mambo recorded indoors, augmented with ESC-50 environmental sounds at SNR levels from −32 to −3 dB. Despite covering only two drone models, it established the benchmark for CNN/RNN/CRNN comparison.

**Wang et al. 32-class dataset** (2025; `mackenzie-jane/drone-visualization`) offers **3,200 five-second clips** across 32 drone categories, with raw audio, spectrograms, and MFCC visualizations. Recording conditions include both indoor (drone lab) and outdoor (rooftop with wind, birdsong, traffic). An interactive web tool accompanies the dataset.

**Multi-Sensor Drone Detection Dataset** (Svanström et al., 2021; `DroneDetectionThesis/Drone-detection-dataset`) includes IR, visible, and audio data with labels for Drone, Helicopter, and Background. It was produced in collaboration with the Swedish Armed Forces, recorded at distances up to 200m.

Several additional datasets cover specialized use cases: **DREGON** (INRIA, 2018) provides 8-channel 44.1 kHz recordings from a UAV-embedded microphone array for sound source localization; **DroneNoise Database** (University of Salford, 2024) captures sUAS overflight noise under field conditions in Scotland; **UaVirBASE** (2025) offers 8-channel 96 kHz recordings for localization; and the **drone authentication dataset** (University of Glasgow, 2022) captures individual drone acoustic fingerprints.

### Background noise and pretraining datasets

| Dataset | Size | Classes | Use in drone detection |
|---------|------|---------|----------------------|
| **ESC-50** | 2,000 clips, 2.78 hours, 44.1 kHz | 50 environmental sound classes | Negative class augmentation, background noise mixing |
| **UrbanSound8K** | 8,732 clips (≤4s each) | 10 urban sound classes | Urban background noise augmentation |
| **Google AudioSet** | ~1.8M clips, 632 classes, 48 kHz | Aircraft/vehicle classes included | Pretraining corpus for transfer learning (VGGish, PANNs, YAMNet) |
| **TUT Acoustic Scenes** | Multiple DCASE challenge sets | 10-15 scene types | Scene-aware augmentation |

### Drone types and coverage gaps

Nearly all public datasets focus on **multirotor (quadcopter) drones**—DJI Mavic, Phantom, Parrot Bebop/Mambo, and various racing/toy quads. Fixed-wing UAV data is essentially absent from public repositories. The Wang 32-class dataset has the broadest model coverage, spanning small toys to larger Class I UAVs. **No public dataset adequately covers fixed-wing, hybrid VTOL, or large military-class UAVs.** For a production system, collecting supplementary field recordings of target drone types is strongly recommended.

### Data augmentation strategies for small datasets

The most impactful augmentation for drone detection is **background noise injection**: mixing clean drone recordings with ESC-50/UrbanSound8K audio at SNR levels from −10 to +20 dB. Al-Emadi et al. demonstrated that this approach, combined with GAN-generated synthetic drone audio, significantly improves generalization to unseen drone types. Additional effective augmentations include pitch shifting (±3 semitones, simulating different RPMs), time stretching (0.85×–1.15×, simulating approach speeds), and distance attenuation simulation with frequency-dependent rolloff. In the spectrogram domain, SpecAugment (frequency/time masking) provides consistent 3–5% accuracy gains, and SpecMix (CutMix-style blending) outperforms both Mixup and SpecAugment individually on acoustic scene benchmarks.

---

## 2. From raw audio to model-ready features

### Audio preprocessing pipeline

**Sample rate:** 16 kHz is sufficient and recommended. The critical drone propeller content exists below 8 kHz (Nyquist at 16 kHz), and the DADS dataset is already standardized at this rate. Higher rates (22.05–44.1 kHz) capture broadband high-frequency components but double computational cost with marginal detection benefit.

**Drone frequency characteristics:** The Blade Passing Frequency (BPF) is calculated as `(RPM ÷ 60) × number_of_blades`. A DJI Mavic at 5000 RPM with 2-bladed props produces a BPF of ~166 Hz. Harmonics extend to 6+ kHz, with the strongest energy concentrated in the **100–1,100 Hz band**. Motor noise contributes in the 600–6,000 Hz range at 22+ dB above the broadband floor. A bandpass filter at **80–8,000 Hz** captures all relevant content while rejecting low-frequency wind rumble and high-frequency sensor noise.

**Noise reduction:** Spectral gating via the `noisereduce` library is the most practical approach—it operates in either stationary mode (fixed noise profile) or non-stationary adaptive mode. Wiener filtering achieves better theoretical performance (20–25 dB improvement) but requires known noise statistics. For a real-time edge system, a simple 4th-order Butterworth bandpass (80–8,000 Hz) combined with RMS normalization provides a good cost-benefit balance.

**Windowing for inference:** Use a 25ms Hann window with 10ms hop for spectrogram computation within each analysis frame. The analysis frame itself should be **1–2 seconds** with a **250–500ms hop** (75–50% overlap) for the sliding-window inference loop.

### Feature extraction comparison for drone acoustics

| Feature | Drone detection performance | Parameters | Edge cost | Recommended? |
|---------|---------------------------|------------|-----------|-------------|
| **Log-Mel spectrogram** | Excellent—preserves full spectral detail for CNNs | 64–128 mel bands, n_fft=2048, hop=512 | Moderate | **Yes—primary choice** |
| **MFCC** | Very good—most widely used; 13 coefficients sufficient for detection | 13–30 MFCCs, 40 mel filters | Low | Yes—lightweight alternative |
| **STFT spectrogram** | Good with SVM; 98.97% detection, 1.28% FAR (Seo et al.) | n_fft=2048, linear scale | Higher | Situational |
| **GTCC (Gammatone)** | Potentially superior—99.9% with SVM (Salman et al.) | 13–20 coefficients | Low | Worth testing |
| Raw waveform | Acceptable—no preprocessing needed | Direct input | Highest model cost | Only with specialized 1D CNN |
| CQT | Good frequency resolution at low frequencies | Variable Q | High | Overkill for binary detection |

**The recommended approach is 64-band log-mel spectrograms.** This balances spectral richness (CNNs can exploit full 2D patterns including harmonic stacks), computational efficiency, and compatibility with pretrained models (PANNs, YAMNet, EfficientAT all expect mel spectrogram input). Multiple studies confirm that **feature-level fusion** of MFCC + mel spectrograms outperforms either alone—the AUDRON framework achieved **98.51%** binary accuracy with a fused approach—but for edge deployment the added complexity rarely justifies the marginal gain.

One important modification from recent literature (MDPI Drones, 2025): the standard Mel scale is optimized for human speech perception (65–1,100 Hz emphasis). Drone harmonics span a broader, flatter range. A **frequency-band filterbank** with uniform density across 80–8,000 Hz may outperform standard Mel for drone-specific tasks.

---

## 3. Five architectures compared head-to-head

### A. 2D CNN on mel spectrograms: the proven workhorse

This is the dominant paradigm. Convert audio to a mel spectrogram image, then classify with a standard image CNN. Three tiers exist:

**Transfer learning from AudioSet** is the most practical starting point. PANNs (Pretrained Audio Neural Networks) provide CNN6, CNN10, and CNN14 backbones pretrained on 5,000 hours of AudioSet. CNN14 achieves **mAP 0.431** on AudioSet with ~81M parameters—too heavy for edge directly, but fine-tunable with excellent feature extraction. The **MobileNetV1/V2 variants in PANNs** offer mAP 0.383–0.389 with only **4.1–4.8M parameters** and 2.8–3.6G MACs, making them directly deployable on all Jetson devices and RPi 4. YAMNet (Google's MobileNetV1-based classifier, 3.7M params) is available as a ready-made TFLite model.

**EfficientAT** (`fschmid56/EfficientAT`) represents the current state of the art for efficient audio classification. It uses **MobileNetV3 trained via knowledge distillation from a PaSST transformer ensemble**. The mn10 variant achieves **mAP 0.47+ on AudioSet with ~4.5M parameters**—outperforming CNN14 (81M params) while being 18× smaller. The mn01 variant drops below 1M parameters for MCU-class deployment. This is the **single best architecture recommendation** for edge acoustic drone detection.

**Custom lightweight CNNs** (2–4 conv layers + FC head) work well for binary detection. Al-Emadi's ~70K-parameter CNN achieved >90% accuracy. The SudarshanChakra implementation (~500K params) reports **95.23% accuracy, 96.97% recall** on drone detection. These tiny models infer in under 10ms on any edge device.

### B. 1D CNN on raw waveform: simpler pipeline, slightly lower accuracy

1D CNNs process audio samples directly, eliminating spectrogram computation. PANN's Res1dNet31 achieves mAP 0.365 on AudioSet—significantly below the 2D CNN14 (0.431). On ESC-10, the gap narrows to ~0.2% (80.4% vs 80.2%). The key advantage is **lower end-to-end latency** since no spectrogram precomputation is needed. AM-MobileNet1D (adapted MobileNetV2 with 1D depthwise separable convolutions) runs **7× faster than SincNet** with only **11.6 MB** on disk. For drone detection specifically, the AUDRON framework uses a 1D CNN branch on MFCCs as part of its multi-branch fusion.

**Verdict:** 1D CNNs are viable for ultra-low-latency requirements but sacrifice 2–5% accuracy versus 2D approaches. Not recommended as the primary architecture unless spectrogram computation is a bottleneck.

### C. CRNN (CNN + RNN): better temporal modeling, worse edge fit

CRNNs add LSTM or GRU layers after CNN feature extraction to capture temporal dynamics across frames. Several findings from the literature:

Al-Emadi et al. found **CNN outperformed CRNN**, which outperformed standalone RNN. However, GRU-ANET (2025) reported GRU with attention achieving ~99% detection accuracy, outperforming all baselines including CNN. The key insight is that **GRU is preferred over LSTM** for this task—fewer parameters, less overfitting on small datasets, and comparable performance.

For edge deployment, CRNNs present significant challenges: RNN layers cannot be parallelized (sequential computation), hidden states consume memory across timesteps, and **RNN quantization is less mature** in TensorRT and TFLite than CNN quantization. A pure CNN with a sufficiently large receptive field (via stacked convolutions or attention) captures most of the useful temporal context without these drawbacks.

**Verdict:** Use CRNN only if temporal dynamics across multiple seconds are critical (e.g., tracking approach/departure patterns). For binary detection, CNN-only or CNN+attention is preferred for edge.

### D. Lightweight transformers: powerful but challenging on edge

The **Audio Spectrogram Transformer (AST)** achieves **mAP 0.485 on AudioSet** and **95.6% on ESC-50** with 86.2M parameters—too large for direct edge deployment. However, a fine-tuned tiny-AST achieves **96.73% accuracy** on military audio classification running in **<200ms on an RPi 5** with only 16.5% of parameters active.

The practical solution is **knowledge distillation from transformers to CNNs**, which is exactly what EfficientAT implements. The PaSST transformer teacher trains a MobileNetV3 student that inherits transformer-level accuracy at CNN-level efficiency. The **Conformer architecture** (convolution + self-attention) is described as the "most significant algorithmic development in acoustic classification" for drone detection, capturing both local harmonic patterns and long-range temporal dependencies.

**Verdict:** Do not deploy full transformers on edge. Instead, use transformer-to-CNN distillation (EfficientAT) to get transformer-grade accuracy in an edge-deployable CNN.

### E. Domain-specific approaches targeting drone harmonics

Drone propeller sound is fundamentally a **harmonic stack**—a fundamental frequency plus integer multiples. Several architectures exploit this:

**Harmonic-Percussive Source Separation (HPSS)** separates the harmonic drone rotor content from transient/percussive sounds before classification. The DCASE 2017 runner-up used this preprocessing to achieve 91.7% accuracy. **AECM-Net** (2025) fuses MFCC and Gammatone features with adaptive attention and multi-scale convolutions, achieving **94.50% accuracy** at medium/long distances. **AUDRON** (2025) uses four parallel branches—MFCC-1D CNN, STFT-2D CNN, BiLSTM, and Autoencoder—fused at the feature level for **98.51% binary detection** and **97.11% multiclass identification**.

The **ResNet-Mamba hybrid** (2025) combines ResNet with a multi-level state space model, achieving **F1=98.2%, accuracy=99.1%** under low-SNR conditions—the best reported performance in challenging noise environments.

### Architecture summary with edge benchmarks

| Architecture | Params | Size | Accuracy (drone) | Jetson Nano latency | RPi 4 latency | Recommendation |
|---|---|---|---|---|---|---|
| EfficientAT mn10 | ~4.5M | ~18 MB | ~95–97% (fine-tuned) | ~30–50ms | ~150ms | **Best overall** |
| EfficientAT mn01 | <1M | ~2 MB | ~90–93% | <15ms | ~50ms | Best for ultra-constrained |
| MobileNetV2 (PANN) | ~4.1M | ~16 MB | ~93–95% | ~29ms (TRT) | ~100–200ms | Strong alternative |
| Custom tiny CNN | ~70K–500K | <2 MB | ~90–95% | <10ms | ~30–50ms | Simplest to deploy |
| CNN14 (PANN) | ~81M | ~320 MB | ~95–97% | ~200ms+ | Not feasible | Only with pruning/quantization |
| CRNN (3 conv + 2 GRU) | ~75K | <1 MB | ~92% | Variable | ~50–100ms | Only if temporal context needed |
| AST-base | ~86.2M | ~340 MB | ~96% | >500ms | Not feasible | Not recommended for edge |

---

## 4. Training strategy for maximum detection performance

### Data splitting: session-level grouping is non-negotiable

**Never split audio data at the frame level.** When a 30-second drone recording is segmented into overlapping 1-second windows, consecutive frames share substantial content. Random frame-level splitting creates data leakage that inflates metrics by 10–20% (Plötz, 2021; Kapoor & Narayanan, 2023). Instead, use **group-based splitting** where all segments from the same recording session go into the same split. In scikit-learn, use `GroupKFold` or `GroupShuffleSplit` with recording file ID as the group variable. The recommended ratio is **70/15/15** for train/validation/test, stratified by class. For small datasets, **grouped 5-fold cross-validation** provides more reliable estimates.

For maximum rigor, consider **leave-one-environment-out** validation: train on indoor/lab recordings, test on outdoor field data. This evaluates the generalization that actually matters for deployment.

### Class imbalance: focal loss with balanced sampling

Drone sounds are rare relative to background noise in continuous monitoring. The recommended approach layers three strategies:

1. **Class-balanced sampling** within each batch, targeting ~50/50 drone/no-drone ratio
2. **Focal Loss** with γ=2.0: `FL(pt) = -αt(1 - pt)^γ · log(pt)`, which down-weights well-classified examples and focuses training on hard cases. **Time-Balanced Focal Loss** (Park & Elhilali, ICASSP 2022) further accounts for event duration variability
3. As a simpler baseline, **Weighted BCE** with `pos_weight = num_negative / num_positive` in PyTorch's `BCEWithLogitsLoss`

Standard SMOTE is **not recommended** for spectrogram data—it creates unrealistic interpolations in time-frequency space.

### Augmentation pipeline (ordered by impact)

```
Waveform-domain (applied with probability p):
  1. AddBackgroundNoise(ESC-50 + UrbanSound8K, SNR: -10 to +20 dB, p=0.7)  ← most critical
  2. PitchShift(±3 semitones, p=0.3)
  3. TimeStretch(0.85–1.15×, p=0.3)
  4. AddGaussianNoise(amplitude: 0.001–0.01, p=0.3)
  5. Gain(-6 to +6 dB, p=0.3)

Spectrogram-domain:
  6. FrequencyMasking(F=20, num_masks=2, p=0.5)
  7. TimeMasking(T=50, num_masks=2, p=0.5)
  8. Mixup(α=0.3, p=0.3)
```

Use `audiomentations` for waveform augmentations (CPU) or `torch-audiomentations` for GPU-accelerated training. SpecAugment masks are applied via `torchaudio.transforms.FrequencyMasking` and `TimeMasking`. GAN-based synthetic drone audio generation (Al-Emadi, 2021) further improves generalization to unseen drone types when real data is limited.

### Transfer learning: the three-stage unfreezing recipe

Start from an AudioSet-pretrained backbone (EfficientAT mn10 or PANNs MobileNetV2):

1. **Stage 1** (5–10 epochs): Freeze backbone, train only the new classification head. Learning rate: **1e-3**, AdamW optimizer
2. **Stage 2** (5–10 epochs): Unfreeze last 2–3 convolutional blocks. Learning rate: **1e-4**
3. **Stage 3** (5–10 epochs): Unfreeze all layers. Learning rate: **1e-5**

Use **cosine annealing** (`CosineAnnealingLR`, T_max=total_epochs, eta_min=1e-6) as the scheduler. Early stopping on validation F1-score with patience=10. With PANNs/YAMNet embeddings as frozen feature extractors, as few as **50–100 drone samples** can yield reasonable detection performance. Full fine-tuning benefits from 200+ diverse drone clips.

### Hyperparameter summary

| Parameter | Recommended value | Notes |
|-----------|------------------|-------|
| Optimizer | AdamW | weight_decay=0.01–0.05 |
| Learning rate | 1e-4 to 3e-4 (fine-tuning) | 1e-3 for training from scratch |
| LR scheduler | Cosine annealing | With 5–10% linear warmup for transformers |
| Batch size | 32–64 | Smaller (16–32) if GPU memory constrained |
| Max epochs | 50 (fine-tuning), 200 (scratch) | Early stopping patience=10 on val F1 |
| Loss | Focal Loss (γ=2.0, α=0.25) | Fallback: Weighted BCE |

### Framework choice: PyTorch for training, export for deployment

**PyTorch is the clear recommendation for training.** PANNs, EfficientAT, and AST are all PyTorch-native. The `torchaudio` library provides GPU-accelerated mel spectrogram computation and SpecAugment transforms. For deployment: export via **ONNX → TensorRT** for Jetson, or use Google's **ai-edge-torch** library for direct **PyTorch → TFLite** conversion for Raspberry Pi. This eliminates the need to choose between frameworks—train in PyTorch, deploy everywhere.

---

## 5. Optimizing inference for edge hardware

### Quantization: INT8 cuts size 4×, costs <1% accuracy

**Post-Training Quantization (PTQ)** is the simplest path: collect 500–1,000 representative mel spectrogram samples, run TensorRT's `IInt8EntropyCalibratorV2` calibration, and generate an INT8 engine. Typical accuracy loss for audio CNNs is **1–3%** with PTQ. **Quantization-Aware Training (QAT)** inserts fake-quantization nodes during training and recovers nearly all accuracy, but adds implementation complexity.

On Jetson, **FP16 inference is essentially free**—Maxwell (Nano) and Volta (Xavier NX) GPUs have native FP16 support. FP16 provides ~3× speedup over FP32 on Nano. **INT8 requires Volta or newer** (Xavier NX, Orin)—the Jetson Nano lacks INT8 Tensor Core support. On Raspberry Pi, **TFLite INT8** static quantization reduces model size ~4× and accelerates CPU inference significantly via ARM NEON optimizations.

Research on audio CNN compression (Mou & Milanova, 2024) demonstrates that a combined pruning + quantization pipeline achieves **97.2% size reduction and 97.3% FLOP reduction** while maintaining competitive accuracy on ESC-50 and UrbanSound8K.

### TensorRT on Jetson: the canonical deployment path

The conversion pipeline is: **PyTorch → ONNX → TensorRT engine**.

```bash
# Step 1: Export PyTorch model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=13)

# Step 2: Build TensorRT engine (on target Jetson device)
trtexec --onnx=model.onnx --fp16 --workspace=1024 --saveEngine=model_fp16.engine

# For INT8 on Xavier NX/Orin:
trtexec --onnx=model.onnx --int8 --calib=calibration_cache --saveEngine=model_int8.engine
```

TensorRT performs layer fusion (Conv+BN+ReLU → single kernel), precision calibration, and kernel auto-tuning. For audio spectrogram CNNs, expect **2.5–5× speedup** over vanilla PyTorch inference. A MobileNetV2-based audio classifier achieves **~10–15ms inference on Jetson Nano with TensorRT FP16**, and **~2–5ms on Xavier NX**.

### TFLite on Raspberry Pi: lightweight and proven

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen  # for INT8
tflite_model = converter.convert()
```

YAMNet's TFLite model processes ~10 classifications/second on RPi 4 and ~25/second on RPi 5. A custom small CNN (<2M params, INT8) achieves **15–30ms per inference on RPi 4** and **6–15ms on RPi 5**. For additional acceleration, the **Hailo-8L AI Kit** ($65, 13 TOPS at ~2W) or **Coral USB Accelerator** (4 TOPS) dramatically boost throughput.

### Real-time streaming: ring buffer with sliding window

The deployment architecture for continuous monitoring:

```
I2S/USB Microphone → Audio Capture (16kHz mono, ALSA/sounddevice)
    → Ring Buffer (2-second capacity)
        → Extract 1s window every 250ms
            → Mel Spectrogram (64 bands, librosa/torchaudio)
                → Quantized CNN (TensorRT/TFLite)
                    → Temporal Smoothing (majority vote, last 4–6 predictions)
                        → Alert (GPIO, MQTT, HTTP)
```

**Latency budget:** Audio acquisition ~25–100ms + feature extraction ~5–15ms + model inference 5–50ms + post-processing ~10–50ms. With a 250ms hop and 4 predictions for confirmed detection, total response time is **~1–1.5 seconds** from drone sound onset to alert. This is adequate for security applications—a drone at 10 m/s covers only 15m in 1.5 seconds.

For audio capture, **I2S MEMS microphones** (Adafruit SPH0645LM4H) provide the lowest latency (~1–5ms) but require GPIO wiring. USB microphones are simpler. For direction-of-arrival estimation, the **ReSpeaker 4-mic array** HAT for RPi enables coarse drone localization.

### Hardware selection for new designs

| Device | AI Performance | Power | Price | Best for |
|--------|---------------|-------|-------|----------|
| **Jetson Orin Nano** | 40–67 TOPS | 7–25W | ~$199–249 | Best overall: massive headroom, INT8, current |
| **RPi 5 + Hailo-8L** | ~13 TOPS | ~8–10W | ~$145 | Best budget: good ecosystem, low power |
| Jetson Nano | 0.47 TFLOPS | 5–10W | ~$99 (legacy) | Prototyping only; no INT8, EOL |
| RPi 5 (no accelerator) | ~5 TOPS (CPU) | 3–7W | $80 | Viable with lightweight models only |
| RPi 4 | ~2 TOPS (CPU) | 3–6W | $75 | Not recommended without accelerator |

The **Jetson Orin Nano** is the strongest recommendation for new designs—it provides 40+ TOPS with full INT8 TensorRT support at 7W minimum power, in the same SOM form factor as previous Jetsons. For budget-constrained or battery-powered deployments, the **RPi 5 + Hailo-8L** combination offers 13 TOPS at excellent power efficiency.

---

## 6. Evaluation metrics: optimizing for operational reality

### Metric hierarchy for drone detection

**False positive rate is the single most critical operational metric.** Frid et al. (EUSIPCO 2020) explicitly warn: "regular false alarms will encourage security staff to ignore UAV detection output." Seo et al. achieved a **false alarm rate of just 1.28%** with STFT+CNN—a strong benchmark to target. The standard metric suite includes:

- **Recall (sensitivity)** ≥95%: missing a real drone is unacceptable in security applications. The SudarshanChakra system achieves **96.97% recall**.
- **False positive rate** <5%: operational credibility requires few false alarms. Use confidence thresholding (start at 0.5, lower to 0.4 for defense-biased systems) plus temporal smoothing.
- **F1-score**: balances precision and recall; typical range **0.93–0.98** for well-trained models.
- **ROC-AUC** >0.95: measures discrimination ability across all thresholds.
- **Detection latency**: time from drone sound onset to confirmed alert. Target **<2 seconds**.

### Confusion matrix analysis patterns

The most common failure modes are: helicopters misclassified as drones (similar rotor harmonics), lawnmowers/leaf blowers triggering false positives (overlapping frequency content), and wind gusts causing intermittent false detections. Temporal smoothing (majority vote over 4–6 consecutive predictions) dramatically reduces transient false positives. In production, maintain a **rolling confusion matrix** and retrain periodically as new failure modes are identified.

For reporting, always evaluate on **per-recording** metrics (not per-frame, which inflates scores due to temporal correlation). Report results at multiple operating points on the ROC curve, specifically at the FPR=1% and FPR=5% thresholds.

---

## 7. Recommended end-to-end pipeline

Based on all evidence reviewed, here is the recommended practical pipeline for a binary acoustic drone detection system using public datasets, targeting edge deployment.

### System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (Desktop GPU)               │
│                                                                  │
│  DADS + DroneAudioSet datasets                                  │
│       ↓                                                          │
│  Preprocessing: resample 16kHz → bandpass 80–8kHz → normalize   │
│       ↓                                                          │
│  Augmentation: noise mixing, pitch shift, SpecAugment, Mixup    │
│       ↓                                                          │
│  Feature extraction: 64-band log-mel spectrogram                │
│       (n_fft=1024, hop=320, fmin=80, fmax=8000)                │
│       ↓                                                          │
│  Model: EfficientAT mn10 (MobileNetV3, AudioSet-pretrained)     │
│       ↓                                                          │
│  Training: AdamW, Focal Loss, cosine annealing, 50 epochs       │
│       ↓                                                          │
│  Export: PyTorch → ONNX → TensorRT FP16/INT8 or TFLite INT8    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               INFERENCE PIPELINE (Edge Device)                   │
│                                                                  │
│  I2S/USB Microphone (16kHz, mono)                               │
│       ↓                                                          │
│  Ring Buffer (2s capacity, 250ms hop)                           │
│       ↓                                                          │
│  Bandpass filter → RMS normalize → Mel spectrogram              │
│       ↓                                                          │
│  TensorRT/TFLite inference (<30ms)                              │
│       ↓                                                          │
│  Temporal smoothing (majority vote, 4 consecutive frames)       │
│       ↓                                                          │
│  Alert: GPIO trigger / MQTT message / HTTP webhook              │
└─────────────────────────────────────────────────────────────────┘
```

### Library stack

| Purpose | Library | Why |
|---------|---------|-----|
| Audio I/O | `soundfile` + `torchaudio` | Lightweight loading; torchaudio for GPU transforms |
| Feature analysis | `librosa` | Rich feature extraction for exploration and evaluation |
| Mel spectrograms (training) | `torchaudio.transforms.MelSpectrogram` | GPU-accelerated, differentiable |
| Waveform augmentation | `audiomentations` or `torch-audiomentations` | Background noise mixing, pitch shift |
| Spectrogram augmentation | `torchaudio.transforms.FrequencyMasking/TimeMasking` | SpecAugment |
| Model backbone | EfficientAT (`fschmid56/EfficientAT`) | Best accuracy/efficiency tradeoff |
| Training framework | PyTorch 2.x + torchaudio | Research standard, best model ecosystem |
| Noise reduction | `noisereduce` | Spectral gating for preprocessing |
| Edge inference (Jetson) | TensorRT via `trtexec` | 2.5–5× speedup, FP16/INT8 |
| Edge inference (RPi) | TFLite via `tflite-runtime` | Lightweight, INT8 optimized for ARM |
| Audio capture (edge) | `sounddevice` or ALSA | Low-latency real-time capture |

### Performance targets

| Metric | Target | Achievable with this pipeline |
|--------|--------|-------------------------------|
| Binary detection accuracy | >95% | Yes—literature shows 95–99% range |
| Recall | >95% | Yes—tune confidence threshold |
| False positive rate | <5% | Yes—with temporal smoothing |
| Inference latency (Jetson Orin) | <10ms | Yes—TensorRT INT8 |
| Inference latency (Jetson Nano) | <30ms | Yes—TensorRT FP16 |
| Inference latency (RPi 5 + Hailo) | <30ms | Yes—TFLite INT8 + accelerator |
| Detection response time | <1.5s | Yes—1s window + 250ms hop + 4-vote smoothing |
| Model size (quantized) | <5 MB | Yes—mn10 INT8 ~4.5 MB |
| Power consumption | 5–15W | Yes—depends on platform and mode |

### Key implementation notes

**Start simple, then optimize.** Begin with YAMNet (available as a ready TFLite model) as a frozen feature extractor with a 2-layer classifier trained on DADS. This gives a working baseline in hours. Then graduate to EfficientAT mn10 fine-tuning for production-grade accuracy. Only pursue multi-branch fusion (AUDRON-style) or custom architectures if the simpler approach fails to meet accuracy requirements.

**Field calibration matters.** Collect 15–30 minutes of ambient audio from the actual deployment location. Use this for noise profile estimation, background noise augmentation during fine-tuning, and setting the detection confidence threshold. Acoustic environments vary enormously—a model trained on lab data will underperform without domain adaptation via noise mixing.

**Detection range is the fundamental constraint.** Acoustic detection reliability degrades significantly beyond **150m**, and wind noise above 54 dB severely impacts accuracy. For a complete drone interception platform, acoustic detection should be treated as the **close-range early warning** component, complemented by RF detection or radar for longer ranges.

---

## Conclusion

The acoustic drone detection field has matured rapidly. The convergence of large aggregated datasets (DADS at 60+ hours), efficient pretrained audio models (EfficientAT), and capable edge hardware (Jetson Orin Nano at 40 TOPS for $199) means a production-viable system is achievable using only public resources. The dominant design pattern—**log-mel spectrogram → AudioSet-pretrained MobileNetV3 → TensorRT quantization → sliding-window inference**—delivers 95%+ accuracy at sub-30ms latency on modest hardware. 

Three non-obvious insights from this analysis deserve emphasis. First, **background noise augmentation matters more than model architecture**—mixing drone audio with diverse environmental noise at realistic SNR levels consistently produces larger accuracy gains than switching from a simple CNN to a complex multi-branch fusion model. Second, **transformer-to-CNN knowledge distillation** (the EfficientAT approach) has obsoleted the accuracy-efficiency tradeoff that previously forced choosing between large accurate models and small fast ones. Third, **temporal smoothing in post-processing** is often the difference between a demo and a deployable system—majority voting over 4–6 consecutive predictions reduces false positives by an order of magnitude with negligible added latency. For a system that is part of a larger drone interception platform, prioritize high recall with temporal smoothing over raw precision, and plan for multi-modal fusion (acoustic + RF + visual) to extend detection range beyond the ~150m acoustic ceiling.