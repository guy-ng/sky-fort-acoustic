# Domain Pitfalls

**Domain:** Acoustic drone detection and tracking microservice (UMA-16v2, Docker, real-time)
**Researched:** 2026-04-01
**Confidence:** HIGH (v1 pitfalls) + HIGH (v2 migration pitfalls, based on direct source code analysis)

---

## v1 Infrastructure Pitfalls (from initial research)

These pitfalls from the original service research remain relevant and are not repeated in detail here. See git history for full descriptions.

| # | Pitfall | Phase | Status |
|---|---------|-------|--------|
| 1 | USB Audio Device Instability in Docker | Phase 1 | Addressed |
| 2 | Buffer Overflows and Xruns in 16-Channel Capture | Phase 1 | Addressed |
| 3 | Spatial Aliasing from 42mm Grid Spacing | Phase 2 | Addressed |
| 4 | CNN Overfitting to Recording Environment | Phase 4 | Still relevant for v2 training |
| 5 | Python GIL Blocking the Processing Pipeline | Phase 1-2 | Addressed |
| 6 | ZeroMQ PUB/SUB Slow Subscriber | Phase 3 | Addressed |
| 7 | Doppler Estimation Confusion from Multi-Rotor Harmonics | Phase 3 | Still relevant |
| 8 | Beamforming Map Serialization Bottleneck | Phase 5 | Addressed |
| 9 | Recording File Format and Metadata Loss | Phase 3 | Still relevant for v2 data collection |
| 10 | Single Docker Container Becoming Unmanageable | Phase 1/5 | Addressed |
| 11 | Numba/OpenBLAS Thread Oversubscription | Phase 1 | Addressed |
| 12 | Microphone Channel Mapping Errors | Phase 1-2 | Addressed |
| 13 | WebSocket Connection Leaks in React UI | Phase 5 | Addressed |
| 14 | Wind Noise Saturation in Outdoor Deployment | Phase 2-3 | Still relevant |

---

## v2 Critical Pitfalls: TF-to-PyTorch Migration

Mistakes that cause silent accuracy loss, production failures, or force rewrites of the classification pipeline.

---

### Pitfall 15: Three Incompatible Normalization Schemes

**What goes wrong:** The model produces random/degraded predictions in production because training and inference normalize mel-spectrograms differently. The accuracy you measured offline evaporates when deployed.

**Why it happens:** The existing codebase contains THREE different normalization approaches, and the research code itself is internally inconsistent between training and inference:

| Location | Normalization | Output Range |
|----------|--------------|-------------|
| `train_strong_cnn.py` (research training) | `(S_db + 80) / 80`, clip [0, 1] | [0.0, 1.0] |
| `run_strong_inference.py` (research inference) | zero-mean, unit-variance | unbounded |
| `eval_folder_with_strong_cnn.py` (research eval) | `(S_db + 80) / 80`, clip [0, 1] | [0.0, 1.0] |
| `preprocessing.py` (current service) | zero-mean, unit-variance | unbounded |
| `Mel_Preprocess_and_Feature_Extract.py` (original research) | raw `power_to_db`, no norm | [-80, 0] approx |

The research training uses fixed-range normalization `(S_db + 80) / 80` while its own inference script (`run_strong_inference.py`) uses zero-mean/unit-variance. This is already a bug in the research code. If you port the inference normalization rather than the training normalization, the model will underperform silently.

**How to avoid:**
- The canonical normalization is whatever was used during training. Port `train_strong_cnn.py`'s `(S_db + 80) / 80` normalization, NOT the inference script's z-score normalization.
- Write a "preprocessing parity test": generate a mel-spec from a known WAV file using the original TF training code, then generate one using the new PyTorch code. Assert they are numerically identical (`np.allclose` with atol=1e-4).
- Create a single `MelConfig` dataclass that holds all parameters (n_fft, hop_length, n_mels, norm_method, norm_params) and is shared between training and inference -- no duplicate constants across files.

**Warning signs:**
- Validation accuracy during PyTorch training is significantly different from the TF benchmark
- Model performs well on evaluation harness but poorly on live audio
- Confidence scores are consistently lower or higher than expected

**Phase to address:** First phase of migration -- preprocessing must be validated before any model work begins.

---

### Pitfall 16: Segment Duration Mismatch Between Training and Live Inference

**What goes wrong:** The CNN was trained on 0.5-second chunks (`CHUNK_SECONDS = 0.5` in `train_strong_cnn.py`) but the live inference system feeds it 2.0-second segments (`CNN_SEGMENT_SECONDS = 2.0` in current `preprocessing.py`; `SNIPPET_DURATION = 2.0` in `mic_realtime_inference.py`). The mel-spectrogram dimensions happen to be padded/trimmed to the same shape (128, 64), masking the fact that the temporal resolution is completely different.

**Why it happens:** The research evolved in stages. Training uses 0.5s random crops from longer files. The research inference script uses 2.0s segments split with 1.0s hop for aggregation. The current service copies the 2.0s approach. All three produce (128, 64) spectrograms through pad/trim, so the shape matches but the content semantics diverge:
- 0.5s at 16kHz with hop=256 produces ~31 frames, padded to 128 (75% zero-padding)
- 2.0s at 16kHz with hop=256 produces ~125 frames, nearly fills 128 (2% zero-padding)

A model trained on mostly-padded spectrograms will see very different feature distributions when fed nearly-full spectrograms. The model may have learned to rely on the padding pattern as a feature.

**How to avoid:**
- Decide ONE segment duration and use it everywhere. Two valid approaches:
  - (A) Keep 0.5s training chunks and implement sliding-window aggregation (p_max, p_mean, p_agg) over multiple 0.5s segments in the live pipeline -- this matches `eval_folder_with_strong_cnn.py`'s approach.
  - (B) Retrain on 2.0s segments and use a single-pass classification without aggregation.
- Approach (A) is safer because it matches the proven evaluation pipeline and does not require retraining.
- Document the chosen segment duration as a system invariant in the `MelConfig`.

**Warning signs:**
- Model accuracy on evaluation harness differs significantly from training val accuracy
- Suspiciously high amount of zero-padding visible when debugging spectrograms
- Model over-predicts "drone" on long audio segments (because it never sees this much actual content during training)

**Phase to address:** Must be resolved BEFORE model porting -- this determines the training data pipeline design.

---

### Pitfall 17: Pad-or-Trim Asymmetry (Center Crop vs Truncate)

**What goes wrong:** When a spectrogram exceeds `MAX_FRAMES`, the training code center-crops:
```python
# train_strong_cnn.py line 63
start = (t - max_frames) // 2
return spec[start:start + max_frames, :]
```
While the current service and research inference both truncate from the front:
```python
# preprocessing.py line 53 / run_strong_inference.py line 72
return spec[:max_frames]
```

This means the model sees different temporal windows of the same audio depending on whether it's training or serving.

**Why it happens:** Copied code was "close enough" -- both produce the right shape. Nobody noticed the 3-line difference in behavior.

**How to avoid:**
- Use a single, shared `pad_or_trim` function with explicit behavior documented.
- Given training uses center-crop, inference must also center-crop. But note: for the 0.5s training chunks, `MAX_FRAMES=128` is larger than the actual frame count (~31), so trimming never triggers during training. It only triggers during the 2.0s inference path. This means the center-crop logic in training is dead code for its actual use case -- making the mismatch less dangerous in practice but still a code smell.
- The real fix: if using 0.5s segments consistently (per Pitfall 16 resolution), trimming never triggers and this pitfall becomes moot.

**Warning signs:** Subtle accuracy degradation that is hard to diagnose. Spectrogram visual inspection shows different content for the same audio.

**Phase to address:** Preprocessing parity phase, alongside Pitfall 15.

---

### Pitfall 18: PyTorch Channel Ordering (NCHW) vs TensorFlow (NHWC)

**What goes wrong:** The TF model expects input shape `(batch, time, mels, 1)` -- NHWC format. PyTorch Conv2d expects `(batch, 1, time, mels)` -- NCHW format. If you transpose incorrectly or forget to transpose, the model silently processes garbage because the dimensions happen to be permutable without runtime errors (128 and 64 are valid spatial dims either way).

**Why it happens:** This is the single most common TF-to-PyTorch migration mistake. The shapes don't crash -- they produce wrong results silently because Conv2d will happily convolve over whatever spatial dimensions you give it.

**How to avoid:**
- When building the PyTorch model, define `(batch, 1, MAX_FRAMES, N_MELS)` as the canonical input shape -- i.e., `(N, 1, 128, 64)`.
- Add an explicit shape assertion at the model's `forward()` entry:
  ```python
  assert x.shape[1:] == (1, 128, 64), f"Expected (1, 128, 64), got {x.shape[1:]}"
  ```
- When porting weights (if attempting direct weight transfer rather than retraining): Conv2d kernels in TF are `(H, W, C_in, C_out)` and in PyTorch are `(C_out, C_in, H, W)`. BatchNorm params have the same layout in both frameworks.
- Strongly recommend RETRAINING in PyTorch rather than porting weights -- the architecture is simple enough (3 conv layers, ~200K params) that retraining takes minutes to hours, while manual weight surgery is error-prone.

**Warning signs:**
- Model output is near-random (~50% accuracy on binary task)
- Trained model gives suspiciously uniform confidence across all inputs
- Model outputs all-same predictions regardless of input content

**Phase to address:** Model architecture porting phase.

---

### Pitfall 19: BatchNorm Behavior Difference Between Training and Inference

**What goes wrong:** PyTorch BatchNorm uses running statistics during `model.eval()` but per-batch statistics during `model.train()`. If you forget to call `model.eval()` before inference, or if you accidentally leave the model in training mode, BatchNorm uses batch statistics from whatever single sample is being processed -- producing noisy, unstable predictions.

**Why it happens:** In TF/Keras, `model.predict()` automatically uses inference mode. In PyTorch, you must explicitly switch with `model.eval()` and optionally wrap in `torch.no_grad()`. The CNN has 3 BatchNorm layers, so this affects every forward pass.

**How to avoid:**
- Wrap inference in a context manager that guarantees eval mode:
  ```python
  @torch.no_grad()
  def predict(self, x: torch.Tensor) -> float:
      self.model.eval()
      return self.model(x)
  ```
- Add a unit test that runs the same input twice in eval mode and asserts identical output. (BatchNorm in training mode produces different results each time for batch_size=1.)
- Never expose `model.train()` outside the training loop.

**Warning signs:**
- Inference results fluctuate for identical inputs
- Single-sample inference gives different results than batch inference
- Results are "approximately right" but have unexplained noise

**Phase to address:** Model inference integration phase.

---

### Pitfall 20: Training Pipeline Starves Real-Time Inference of CPU/Memory

**What goes wrong:** Starting a PyTorch training job in the same Docker container as the live audio pipeline causes the beamforming loop to miss deadlines. The 48kHz 16-channel stream must be consumed at ~150ms intervals; training's data loading, augmentation, and backpropagation compete for CPU and memory, causing audio buffer overruns and dropped frames.

**Why it happens:** PyTorch DataLoader uses worker processes that compete for CPU cores. A training batch of 32 mel-spectrograms with librosa processing is CPU-intensive. The Docker container likely has limited CPU allocation. The audio capture ring buffer has finite depth.

**How to avoid:**
- Training MUST run with reduced priority: `os.nice(10)` for the training process, and limit DataLoader `num_workers` to 1-2.
- Implement a "training throttle" that pauses training batches when audio buffer occupancy exceeds 70%.
- Better: run training as a separate process (not thread) with explicit CPU affinity, leaving core 0 for the audio pipeline.
- Best: defer training to non-operational windows. The web UI should show a "Training will degrade detection performance" warning and let the operator choose.
- Set `torch.set_num_threads(2)` for the training process to limit BLAS thread pool.

**Warning signs:**
- Audio buffer overrun warnings in logs during training
- Beamforming map update rate drops below expected frequency
- Increased false negatives during training (missed detections because inference is slow)

**Phase to address:** Training pipeline integration phase -- must be designed with resource isolation from the start.

---

### Pitfall 21: librosa Version Skew Between Research and Service

**What goes wrong:** `librosa.feature.melspectrogram` default behavior changed across versions. The research repo uses Python 3.10 with an older librosa (visible in its `.venv`). The service targets Python 3.11+ with `librosa>=0.10`. Subtle numerical differences include `center` padding behavior, FFT backend changes, and `power_to_db` `ref` parameter handling.

**Why it happens:** librosa 0.10+ switched from `audioread` to `soundfile` as default backend. Mel filter bank computation had subtle numerical changes between versions. The `ref` parameter in `power_to_db` interacts with normalization -- `ref=np.max` gives per-sample normalization while `ref=1.0` gives absolute dB values. Both codebases use `ref=np.max`, but the resulting range depends on the input signal.

**How to avoid:**
- Replace librosa mel computation with `torchaudio.transforms.MelSpectrogram` for the PyTorch pipeline. This eliminates the librosa dependency entirely from the inference path and ensures training/inference use identical transforms.
- If keeping librosa for training data preparation, write a numerical equivalence test between `librosa.feature.melspectrogram` and `torchaudio.transforms.MelSpectrogram` with your exact parameters.
- Pin exact librosa version in both training and inference requirements if using librosa at all.
- Note: `torchaudio.transforms.MelSpectrogram` uses `power=2.0` by default (matching the research code) but its `center` default is `True` (matching librosa). The key difference is the mel filter bank implementation -- they are slightly different between librosa and torchaudio. For a retrained model, this doesn't matter as long as you use the same library for both training and inference.

**Warning signs:**
- Tiny but consistent accuracy drop after upgrading librosa
- Mel-spectrogram visual comparison shows slight differences at high/low frequencies

**Phase to address:** Preprocessing parity phase.

---

## v2 Technical Debt Patterns

### Debt 1: Hardcoded Magic Numbers Across Files

The research codebase duplicates constants (`FS=16000`, `N_FFT=1024`, `HOP_LENGTH=256`, `N_MELS=64`, `MAX_FRAMES=128`) in at minimum 5 files: `train_strong_cnn.py`, `run_strong_inference.py`, `eval_folder_with_strong_cnn.py`, `mic_realtime_inference.py`, and the current service's `preprocessing.py`. Any change requires updating all files.

**Prevention:** Single `MelConfig` dataclass, imported everywhere. This is the first thing to build in the migration.

### Debt 2: ONNX-to-PyTorch Migration Leaves Dead Code

The current service has `OnnxDroneClassifier` with EfficientNet-specific preprocessing (224x224 resize via `scipy.ndimage.zoom`, 3-channel repeat via `np.stack([spec_resized] * 3)`). When replacing with PyTorch CNN, this code becomes dead weight. If not cleanly removed, future developers will be confused about which path is active.

**Prevention:** Define a `DroneClassifier` protocol/ABC with a `predict(preprocessed) -> float` interface. Swap implementations cleanly. Remove EfficientNet-specific code (224x224 resize, 3-channel stack, ONNX Runtime dependency) once the new model is validated. Do not leave both paths "just in case."

### Debt 3: The Research Ensemble is Untranslatable as-Is

The late fusion code (`Voting_Models_Train_and_Individual_Test.py`) uses `exec()` to run 10 separate Python files sequentially, storing results as JSON files on disk. Each of the 10 model trainers is a separate file with nearly identical code. This pattern is fundamentally incompatible with real-time service architecture and must be redesigned from scratch, not "ported."

**Prevention:** Design ensemble as a `List[nn.Module]` held by an `EnsembleClassifier` that:
- Runs all models in a single loop or batched forward pass
- Combines predictions via configurable strategy (soft vote, hard vote, weighted)
- Computes weights from validation accuracy stored in model metadata, not from JSON files
- Supports adding/removing models without code changes

---

## v2 Integration Gotchas

### Gotcha 1: State Machine Expects Single Probability, Aggregation Returns Three

The current `DetectionStateMachine.update()` takes a single `drone_probability: float`. The research pipeline produces three values: `p_max`, `p_mean`, and `p_agg` (where `p_agg = 1 - prod(1 - p_i)` across segments). Additionally, the research inference uses a fourth composite: `p_drone = 0.7 * p_max + 0.3 * p_mean`.

**Resolution:** Feed `p_agg` to the state machine as it is the most principled metric (probability that at least one segment contains a drone). BUT: the state machine thresholds (`enter=0.80`, `exit=0.40`) were calibrated for the old EfficientNet model's output distribution. They MUST be re-calibrated for the new CNN's output distribution using the evaluation harness.

### Gotcha 2: CNNWorker Queue-of-1 Drops Segments Needed for Aggregation

The current `CNNWorker` uses a maxsize=1 queue with drop semantics -- only the latest audio segment is kept. But segment aggregation (p_max, p_mean, p_agg) requires accumulating predictions across MULTIPLE overlapping segments of a single audio window. The drop-latest-wins design is fundamentally incompatible with aggregation.

**Resolution:** Two valid approaches:
- (A) Buffer the last N segments internally in the worker and run aggregation across the buffer.
- (B) Keep a ring buffer of the last K classification probabilities (not audio segments) and compute p_max/p_mean/p_agg over that window of results.

Option (B) is more natural for real-time: each inference cycle produces one probability, and the aggregation window slides forward. This maps cleanly onto the existing single-slot queue design -- the queue stays at maxsize=1 but results accumulate in a separate ring buffer.

### Gotcha 3: EfficientNet Removal Changes the Preprocessing Shape Contract

The current preprocessing outputs `(1, 3, 224, 224)` for EfficientNet. The new CNN expects `(1, 1, 128, 64)`. Every caller of `preprocess_for_cnn()` must be updated. Specifically:
- Remove the `scipy.ndimage.zoom` resize to 224x224
- Remove the 3-channel repeat (`np.stack([spec_resized] * 3)`)
- Keep the silence gate (`SILENCE_RMS_THRESHOLD`)
- Update return shape documentation and type hints

### Gotcha 4: Mixed Framework Dependencies Bloat Docker Image

Adding PyTorch (~2GB) to the Docker container while retaining ONNX Runtime creates a bloated image. If both remain during a transition period, the image could exceed 4GB, impacting deployment time on edge devices.

**Resolution:** Plan a clean cutover. Do not ship both frameworks simultaneously unless actively A/B testing. The migration plan should include an explicit "remove ONNX" step.

---

## v2 Performance Traps

### Trap 1: librosa in the Inference Hot Path

The current service and all research code use `librosa.feature.melspectrogram` at inference time. librosa is NumPy-based and single-threaded. For real-time inference at every beamforming cycle (~150ms):

| Approach | Time for 2s audio at 16kHz | Notes |
|----------|---------------------------|-------|
| librosa (NumPy) | ~15ms | Current approach |
| torchaudio (PyTorch CPU) | ~3ms | 5x faster, shares framework with model |
| torchaudio (as nn.Module) | ~2ms | Can be part of compiled model graph |

Replace librosa with `torchaudio.transforms.MelSpectrogram` as part of the migration. This eliminates one dependency and lets the preprocessing be JIT-compiled alongside the model via `torch.compile()`.

### Trap 2: Ensemble Multiplies Inference Latency

Running N models in an ensemble multiplies inference time by N. The research uses 10 models. At ~20ms per model inference on CPU, that is 200ms per classification -- exceeding the 150ms beamforming cycle time.

**Mitigation options (in order of preference):**
1. Use 3 models instead of 10 (diminishing returns beyond 3-5 for binary classification)
2. Run ensemble at a lower rate than beamforming (e.g., classify every 500ms, beamform every 150ms)
3. Batch all model inputs together if models share architecture (single batched forward pass)
4. Accept that ensemble is an offline evaluation tool, not a real-time feature

### Trap 3: torch.compile() First-Inference Cold Start

`torch.compile()` triggers JIT compilation on the first forward pass, which can take 10-30 seconds. If the service starts, receives audio immediately, and tries to classify, the first few beamforming cycles will timeout or produce no classification results.

**Mitigation:** Run a warmup inference with dummy data during service startup, before the audio capture begins. Log the compilation time. Structure startup as: load model -> warmup -> start audio capture.

---

## "Looks Done But Isn't" Checklist

These items will appear complete (tests pass, shapes match, code runs) but hide subtle bugs:

- [ ] **Normalization matches training exactly** -- not just "a normalization is applied" but the SAME normalization with the SAME constants. Test: save a spectrogram from TF training code, reproduce it numerically in the new pipeline. The research training uses `(S_db + 80) / 80` clipped to [0,1], while the research inference uses z-score. The TRAINING normalization is canonical.

- [ ] **Segment duration used in live inference matches what the model was trained on** -- shape matching is necessary but not sufficient. 0.5s padded to 128 frames and 2.0s trimmed to 128 frames produce the same tensor shape but completely different feature distributions.

- [ ] **BatchNorm running statistics were computed** -- a freshly initialized PyTorch model has `running_mean=0` and `running_var=1`. After training, these must be populated. If you export/load a model without the running stats, inference produces garbage. Validate by checking `model.bn1.running_mean` is not all zeros.

- [ ] **model.eval() is called before every inference** -- passes all tests in a training context (where train mode is expected) but produces noisy results in production.

- [ ] **Aggregation thresholds are re-calibrated** -- old thresholds (`enter=0.80`, `exit=0.40`) were tuned for EfficientNet's output distribution. The new 3-layer CNN has different confidence calibration. Running the evaluation harness and computing optimal thresholds is not optional.

- [ ] **Ensemble weights sum to 1.0** -- the research soft voting code computes weights as `(1/10) + (accuracy_i - mean_accuracy)`. These only sum to 1.0 in exact arithmetic. Floating point drift can cause predictions slightly > 1.0 or < 0.0. Always normalize: `weights = weights / weights.sum()`.

- [ ] **Random seed doesn't leak between training and inference** -- if the training pipeline sets `np.random.seed(42)` globally, it affects the audio processing pipeline. Use `np.random.default_rng(seed)` with local generators.

- [ ] **Training data augmentation doesn't run during inference** -- if you add augmentations (pitch shift, noise injection, time stretch) to the preprocessing module, ensure they are gated by a `training: bool` flag and never activate during live inference.

- [ ] **CNNWorker aggregation window is bounded** -- if implementing a sliding window of classification results for p_agg, ensure old results are evicted. The formula `p_agg = 1 - prod(1 - p_i)` monotonically increases toward 1.0 as more segments accumulate. Without a window bound, p_agg will always be near 1.0 after a few minutes, regardless of input.

- [ ] **Recording during inference doesn't cause frame drops** -- writing 16-channel 48kHz audio to disk while simultaneously processing it causes I/O contention. Use a separate I/O thread with a bounded write queue for recording.

- [ ] **ONNX Runtime is fully removed** -- not just unused, but removed from requirements.txt and Docker image. Otherwise the image carries ~500MB of dead weight.

---

## Pitfall-to-Phase Mapping (v2 Migration)

| Phase | Pitfalls to Address | Priority | Notes |
|-------|-------------------|----------|-------|
| **1. Preprocessing Parity** | #15 (normalization), #16 (segment duration), #17 (pad/trim), #21 (librosa version), Debt #1 (MelConfig) | CRITICAL -- first | Everything else depends on correct preprocessing. Build `MelConfig`, write parity tests. |
| **2. Model Architecture Port** | #18 (NCHW vs NHWC), #19 (BatchNorm eval), Debt #2 (remove ONNX), Gotcha #3 (shape contract) | CRITICAL -- second | Need correct preprocessing to validate model. Retrain in PyTorch, don't port weights. |
| **3. Segment Aggregation** | Gotcha #1 (state machine interface), Gotcha #2 (worker queue redesign), Checklist (bounded window) | HIGH -- third | Builds on working model. Redesign CNNWorker to support result accumulation. |
| **4. Evaluation Harness** | Checklist (threshold re-calibration) | HIGH -- parallel with 2/3 | Needed to validate each preceding phase. Port eval_folder logic. |
| **5. Training Pipeline** | #20 (resource starvation), Trap #3 (torch.compile warmup), Checklist (seed isolation, augmentation gating) | MEDIUM -- fourth | Training can degrade the live system. Design with resource isolation. |
| **6. Late Fusion Ensemble** | Debt #3 (redesign from scratch), Trap #2 (latency multiplication), Checklist (weight normalization) | LOW -- fifth | Most complex, least urgent. Single model is sufficient for MVP. |
| **7. Data Collection** | Checklist (recording during inference), Gotcha #4 (Docker size), v1 Pitfall #9 (metadata) | MEDIUM -- independent | Low coupling to model work. Can be parallel with other phases. |

---

## Sources

- Direct source code analysis of both codebases (all files listed below were read and compared):
  - `src/acoustic/classification/preprocessing.py` -- current service preprocessing, z-score norm, 2.0s segments, front-truncate
  - `src/acoustic/classification/inference.py` -- current ONNX inference, EfficientNet (1, 3, 224, 224) input
  - `src/acoustic/classification/worker.py` -- current CNN worker thread, maxsize=1 queue with drop semantics
  - `src/acoustic/classification/state_machine.py` -- current detection state machine, single probability input
  - `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` -- research training, (S_db+80)/80 norm, 0.5s chunks, center-crop
  - `Acoustic-UAV-Identification-main-main/run_strong_inference.py` -- research inference, z-score norm, 2.0s segments, front-truncate
  - `Acoustic-UAV-Identification-main-main/eval_folder_with_strong_cnn.py` -- research evaluation, (S_db+80)/80 norm, 0.5s segments, p_agg aggregation
  - `Acoustic-UAV-Identification-main-main/mic_realtime_inference.py` -- research real-time, z-score norm, 2.0s snippets
  - `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/Performance_Soft_Voting_Calcs.py` -- ensemble soft voting with accuracy-based weights
  - `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/Voting_Models_Train_and_Individual_Test.py` -- ensemble training via exec() of 10 separate files
  - `Acoustic-UAV-Identification-main-main/1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py` -- original preprocessing, 22050Hz, 90 mels, no normalization
- PyTorch documentation on BatchNorm behavior (training vs eval mode) -- HIGH confidence
- torchaudio MelSpectrogram documentation -- HIGH confidence (per STACK.md sources)
- General TF-to-PyTorch migration patterns -- HIGH confidence (well-documented in PyTorch migration guides)
