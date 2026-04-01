# Feature Research: Classification Pipeline Migration

**Domain:** Acoustic UAV classification pipeline migration (v2.0)
**Researched:** 2026-04-01
**Confidence:** HIGH

## Context

This feature analysis covers migrating the Acoustic-UAV-Identification research pipeline into the existing sky-fort-acoustic microservice. The service already has: real-time 16-channel capture, ONNX inference with EfficientNet-B0, a 3-state hysteresis state machine, single-target tracking, WebSocket streaming, and a React dashboard.

The research pipeline uses TensorFlow/Keras with a 3-layer CNN (32->64->128 filters, BatchNorm, GlobalAvgPool, dense 128, dropout 0.3, sigmoid output), mel-spectrogram features (64 mels, 128 frames, 16kHz, n_fft=1024, hop=256), segment aggregation (p_max, p_mean, p_agg), and a late fusion ensemble of 10 CRNN models with accuracy-weighted soft voting.

**Key observation:** The existing service preprocessing (`src/acoustic/classification/preprocessing.py`) already uses **identical mel-spectrogram parameters** to the research CNN (64 mels, 128 frames, 16kHz, n_fft=1024, hop=256). The only differences are:
1. The service resizes to 224x224 for EfficientNet-B0 (research CNN takes raw 128x64)
2. The service uses zero-mean/unit-std normalization (research uses (dB+80)/80 clip to 0..1)
3. The service does single-frame inference (research aggregates across segments)

This means the preprocessing migration is a parameter swap, not a rewrite.

## Feature Landscape

### Table Stakes

These features must ship for the migration to be considered complete. Without them, the new classification is worse than what exists.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| **Research CNN architecture in PyTorch** | The whole point of the migration. 3-layer CNN (Conv2D 32/64/128 + BN + MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, sigmoid). Replaces EfficientNet-B0 ONNX. | Low | None | Architecture is simple -- ~15 lines of PyTorch. Already proven in research. Export to ONNX for inference keeps existing OnnxDroneClassifier interface. |
| **Research preprocessing params** | Model is trained on specific normalization: `(S_db + 80) / 80` clipped to [0,1]. Current service uses z-score normalization. Wrong normalization = garbage predictions. | Low | CNN architecture | Change `norm_spec()` in preprocessing.py. Remove EfficientNet 224x224 resize. Input becomes (1, 1, 128, 64) instead of (1, 3, 224, 224). |
| **Segment aggregation for file-level decisions** | Research pipeline slices audio into 0.5s segments with 50% overlap, runs CNN on each, then aggregates. Three metrics: p_max (any segment positive), p_mean (average confidence), p_agg = 1 - product(1 - p_i) (at-least-one probability). The real-time inference uses weighted combo: `0.7 * p_max + 0.3 * p_mean`. | Med | CNN architecture, preprocessing | The existing service sends 2s audio to CNN. Need to split into 0.5s chunks internally and aggregate. This replaces the single-shot inference with a more robust multi-segment approach. |
| **PyTorch training pipeline** | Users need to retrain on their own UMA-16 recordings. Research pipeline uses TF with tf.data lazy loading, stratified train/val split, early stopping (patience 8), LR reduction, CSV logging. Must port to PyTorch. | Med | CNN architecture, preprocessing, data collection UI | Core training loop: DataLoader with mel-spec transform, Adam 1e-3, BCE loss, early stopping, best-model checkpoint. The research code is ~300 lines including data loading. |
| **Model evaluation harness** | After training, need to assess quality before deploying. Research pipeline provides: confusion matrix (TP/FN/TN/FP), accuracy/precision/recall/F1, distribution stats (percentiles of p_agg/p_max/p_mean for each class), per-file verbose output. | Med | CNN architecture, segment aggregation | Builds on `eval_folder_with_strong_cnn.py`. Evaluates on held-out test folders. Critical for knowing whether a retrained model is safe to deploy. |
| **ONNX export from PyTorch** | Inference must run via ONNX Runtime (existing infrastructure). Training in PyTorch, serving in ONNX. Standard pattern -- `torch.onnx.export()` with dummy input. | Low | PyTorch training pipeline | Existing `OnnxDroneClassifier` and `CNNWorker` stay mostly intact. Only input shape changes from (1,3,224,224) to (1,1,128,64). |
| **REST API for training/evaluation** | Training and evaluation must be triggerable from the web UI. Need endpoints: POST /train (start training job), GET /train/status (progress), POST /evaluate (run eval), GET /evaluate/results. | Med | PyTorch training, evaluation harness | Long-running training must be async (background task). WebSocket progress updates for the UI. |

### Differentiators

Features that add significant value beyond a basic migration. Not strictly required for parity with the research pipeline, but high ROI.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Late fusion ensemble (soft voting)** | Research uses 10 independently-trained CRNNs with accuracy-weighted soft voting. Reduces variance and improves robustness -- ensemble accuracy consistently beats any single model. In the research code, weights are `(1/N) + (model_acc - avg_acc)`, normalizing individual model contributions by relative performance. | High | Training pipeline, evaluation | **Defer to post-MVP.** The research implementation is brittle (10 hardcoded scripts, exec()-based execution, hardcoded file paths). The concept is sound but needs a clean N-model abstraction. Training 10 models takes 10x compute. Start with single best model, add ensemble later. |
| **UMA-16 dataset collection via web UI** | Current training data comes from separate recording sessions. Integrating collection into the service means: record labeled segments from live mic array, attach metadata (drone type, distance, weather, location), auto-organize into train/test splits. | Med | Existing recording infrastructure | The service already has WebSocket audio streaming. Add: label selection, segment marking, metadata form, auto-save to organized directory structure matching training expectations (data/train/uav/, data/train/background/). |
| **Configurable aggregation strategy** | Research pipeline hardcodes `0.7 * p_max + 0.3 * p_mean` for real-time. Different environments may need different strategies. Expose aggregation weights and threshold as runtime config. | Low | Segment aggregation | Simple config values. Low effort, meaningful tuning knob for field deployment. The research eval harness already computes all three metrics (p_max, p_mean, p_agg) so operators can compare. |
| **Training data augmentation** | Research pipeline does none. Adding time-shift, noise injection, gain variation, and pitch shift during training would improve generalization significantly. Standard practice in audio ML. | Med | Training pipeline | Use torchaudio transforms. Low risk, high reward for field robustness. SpecAugment (frequency/time masking) is also proven for mel-spectrogram inputs. |
| **A/B model comparison** | Load two models simultaneously, run both on same audio, compare predictions in UI. Critical for safe model upgrades -- verify new model outperforms old before swapping. | Med | Evaluation harness, ONNX inference | Could use existing CNNWorker pattern with two instances. Dashboard shows side-by-side probabilities. |
| **Training progress dashboard** | Real-time training curves (loss, accuracy, val_loss, val_accuracy) in the web UI via WebSocket. Research code logs to CSV and TensorBoard. Port to live WebSocket updates. | Med | Training pipeline, WebSocket infrastructure | More useful than TensorBoard for non-ML-engineer operators. Show epoch progress, early stopping countdown, best metric so far. |

### Anti-Features

Features that seem natural to include but would hurt the project. Explicitly do NOT build these.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **10-model ensemble as initial architecture** | Research paper uses it for accuracy. | 10x training time, 10x inference compute, 10x model storage. The research implementation uses `exec()` on 10 separate Python files with hardcoded paths -- a code smell indicating it was never designed for production. Real-time inference of 10 models on a single Docker container competing with beamforming for CPU is not viable without significant optimization. | Start with single best CNN. Add 2-3 model ensemble later with proper abstraction. Measure single-model accuracy first -- it may be sufficient. |
| **TensorFlow/Keras preservation** | Research code is all TF/Keras. "Just keep it." | The service stack is PyTorch (per STACK.md). Running both TF and PyTorch in one container doubles image size (~2GB each) and creates dependency conflicts. TF is heavier at runtime. The research CNN is 3 layers -- porting to PyTorch is trivial. | Port CNN to PyTorch (~30 lines). Use `torch.onnx.export()` for inference. Keep ONNX Runtime for serving. |
| **librosa as runtime dependency** | Research preprocessing uses librosa everywhere. | librosa pulls in numba, llvmlite, soundfile, and more. Heavy dependency chain. The mel-spectrogram computation is just `np.abs(stft) @ mel_filterbank` -- can be done with torchaudio.transforms.MelSpectrogram or even raw numpy+scipy. librosa is convenient for research, overkill for a fixed pipeline. | Use torchaudio.transforms.MelSpectrogram for training. For ONNX inference preprocessing, use scipy or numpy directly (existing code already does this). |
| **MFCC features** | Research repo has `inference.py` using MFCC (n_mfcc=40) and a separate mel-based pipeline. Two feature extraction paths. | The strong CNN (`train_strong_cnn.py`) uses mel-spectrograms, not MFCCs. The MFCC-based `inference.py` appears to be an older/alternative pipeline. MFCCs discard phase information. Mel-spectrograms are strictly more informative. The research community has moved to mel-spectrograms for audio classification. | Use mel-spectrograms only. Do not port the MFCC path. |
| **Research repo's preprocessing script** | `Mel_Preprocess_and_Feature_Extract.py` extracts all features to a giant JSON file. | Materializes entire dataset as JSON (all mel-spectrograms in memory). Does not scale -- a few thousand 10-second files at 90 mels would be gigabytes of JSON. The strong CNN pipeline already uses lazy loading (tf.data with numpy_function). | Use PyTorch DataLoader with on-the-fly mel-spectrogram computation. Never materialize full dataset to disk as JSON. |
| **22kHz sample rate from preprocessing script** | `Mel_Preprocess_and_Feature_Extract.py` uses 22050 Hz. | The strong CNN and real-time inference both use 16kHz. The 22050 Hz is from an older pipeline. Drone audio content is primarily 100-4000 Hz; 16kHz (Nyquist 8kHz) captures everything relevant with lower compute. | Use 16kHz throughout. Already standardized in the service. |
| **Hard voting** | Research repo implements both hard and soft voting. | Hard voting discards confidence information. A model that is 51% confident of "drone" gets the same vote as one that is 99% confident. Soft voting strictly dominates hard voting for binary classification. | If ensemble is built, use soft voting only. |

## Feature Dependencies

```
Research CNN Architecture (PyTorch)
  |
  +---> Research Preprocessing Params
  |       |
  |       +---> Segment Aggregation
  |               |
  |               +---> Configurable Aggregation Strategy
  |
  +---> PyTorch Training Pipeline
  |       |
  |       +---> ONNX Export
  |       |       |
  |       |       +---> Hot-swap model in running service
  |       |
  |       +---> Training Data Augmentation
  |       |
  |       +---> Training REST API
  |       |       |
  |       |       +---> Training Progress Dashboard (WebSocket)
  |       |
  |       +---> Model Evaluation Harness
  |               |
  |               +---> Evaluation REST API
  |               |
  |               +---> A/B Model Comparison
  |
  +---> Late Fusion Ensemble (post-MVP)

UMA-16 Data Collection UI (independent -- can start anytime)
  |
  +---> Label management
  +---> Metadata attachment
  +---> Auto train/test split
```

## MVP Definition

The minimum viable migration replaces EfficientNet-B0 with the research CNN and adds segment aggregation. Everything else is incremental.

**MVP features (in dependency order):**

1. **Research CNN in PyTorch + ONNX export** -- New model architecture, export to ONNX, drop-in replacement for existing model file. Existing OnnxDroneClassifier interface barely changes (input shape only).

2. **Research preprocessing params** -- Switch normalization from z-score to (dB+80)/80, remove 224x224 resize. ~20 lines changed in `preprocessing.py`.

3. **Segment aggregation** -- Split 2s audio into overlapping 0.5s segments, run CNN on each, aggregate with configurable weights. Modifies `CNNWorker._loop()` or adds an aggregation layer.

4. **PyTorch training pipeline** -- Port the training loop from TF to PyTorch. DataLoader, Adam, BCE, early stopping, checkpoint saving, ONNX export on completion.

5. **Model evaluation harness** -- Confusion matrix, precision/recall/F1, distribution stats. CLI-first, REST API later.

6. **Training + evaluation REST API** -- Async training endpoint, progress via WebSocket, evaluation endpoint returning metrics.

**Defer:**
- Late fusion ensemble: High complexity, unclear necessity. Measure single-model accuracy first.
- Data collection UI: Valuable but independent. Can ship in a later phase.
- Training data augmentation: Easy to add to training pipeline later. Not blocking.
- A/B model comparison: Nice-to-have after evaluation harness exists.
- Training progress dashboard: REST polling is fine initially.

## Feature Prioritization Matrix

| Feature | Impact | Complexity | Risk | Priority |
|---------|--------|------------|------|----------|
| Research CNN in PyTorch + ONNX | Critical -- entire migration depends on it | Low | Low -- architecture is simple, well-proven | P0 |
| Research preprocessing params | Critical -- wrong params = wrong predictions | Low | Low -- parameter change only | P0 |
| Segment aggregation | High -- major accuracy improvement over single-frame | Med | Low -- math is straightforward | P0 |
| PyTorch training pipeline | High -- enables model retraining | Med | Med -- data loading, training loop, callbacks | P1 |
| ONNX export from training | High -- connects training to serving | Low | Low -- standard PyTorch API | P1 |
| Model evaluation harness | High -- safety gate before model deployment | Med | Low -- metrics computation is standard | P1 |
| Training + eval REST API | Med -- enables UI integration | Med | Low -- standard FastAPI patterns | P2 |
| Configurable aggregation | Med -- field tuning knob | Low | Low -- config values | P2 |
| Data collection UI | Med -- enables in-field dataset building | Med | Low -- extends existing recording | P2 |
| Training data augmentation | Med -- improves generalization | Med | Low -- standard torchaudio transforms | P2 |
| Training progress dashboard | Low -- convenience for operators | Med | Low -- WebSocket already exists | P3 |
| A/B model comparison | Low -- model upgrade safety | Med | Low -- dual inference workers | P3 |
| Late fusion ensemble | Low (initially) -- uncertain ROI | High | Med -- compute budget unknown | P3 |

## Key Technical Notes

### Normalization Mismatch is the Biggest Risk
The research CNN was trained with `(S_db + 80) / 80` normalization clipped to [0, 1]. The existing service uses zero-mean unit-variance. If you train a new model with research normalization but forget to update the inference preprocessing (or vice versa), predictions will be random noise. This must be tested as a unit test: same audio file through training preprocessing and inference preprocessing must produce identical tensors.

### Segment Aggregation Changes Detection Latency
Currently the service sends 2s of audio to the CNN and gets one probability back. With segment aggregation, the same 2s produces 7 overlapping 0.5s segments (at 250ms hop), each scored independently. The aggregation step adds ~negligible compute (max/mean/product of 7 floats) but the multi-segment inference takes ~7x the single-segment time. However, each segment is tiny (128x64 vs 224x224 = 97% fewer pixels), so net inference time may actually decrease.

### ONNX Export Compatibility
The research CNN uses BatchNorm, which has train/eval mode behavior. Must call `model.eval()` before `torch.onnx.export()`. GlobalAveragePooling2D exports cleanly to ONNX. No custom ops needed.

### Training Compute Budget
The research trains for 60 epochs with early stopping (patience 8). On a CPU Docker container, training on a few thousand 0.5s segments should complete in minutes, not hours. No GPU required for this model size. The research ensemble (10 models x 60 epochs) is what makes training expensive -- single model is fast.

## Sources

- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` -- Research CNN architecture and training pipeline (TensorFlow/Keras)
- `Acoustic-UAV-Identification-main-main/eval_folder_with_strong_cnn.py` -- Segment aggregation (p_max, p_mean, p_agg) and evaluation metrics
- `Acoustic-UAV-Identification-main-main/mic_realtime_inference.py` -- Real-time inference with weighted aggregation (0.7*p_max + 0.3*p_mean) and three-state classification
- `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/Performance_Soft_Voting_Calcs.py` -- Accuracy-weighted soft voting implementation
- `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/Voting_Models_Train_and_Individual_Test.py` -- 10-model ensemble training orchestration (exec-based)
- `Acoustic-UAV-Identification-main-main/1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py` -- Older preprocessing pipeline (22kHz, JSON materialization)
- `Acoustic-UAV-Identification-main-main/inference.py` -- MFCC-based inference (older pipeline, not the strong CNN)
- `src/acoustic/classification/preprocessing.py` -- Existing service preprocessing (already matches research mel params)
- `src/acoustic/classification/worker.py` -- Existing CNN worker thread with drop-semantics queue
- `src/acoustic/classification/inference.py` -- Existing ONNX Runtime inference wrapper
- `src/acoustic/pipeline.py` -- Existing beamforming pipeline with CNN integration
