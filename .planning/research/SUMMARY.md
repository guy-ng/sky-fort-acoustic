# Project Research Summary

**Project:** Sky Fort Acoustic Service
**Domain:** Acoustic UAV classification pipeline migration (TF/Keras to PyTorch)
**Researched:** 2026-04-01
**Confidence:** HIGH

## Executive Summary

The Sky Fort Acoustic Service already has a working real-time audio processing pipeline (16-channel UMA-16v2 capture, SRP-PHAT beamforming, ONNX EfficientNet-B0 inference, 3-state hysteresis, ZeroMQ publishing, React dashboard). The v2 migration replaces the EfficientNet-B0 ONNX classifier with a proven 3-layer research CNN (~95K parameters) trained in PyTorch, while keeping all surrounding infrastructure intact. The critical insight from research is that the existing preprocessing already uses the correct mel-spectrogram parameters (64 mels, 128 frames, 16kHz, n_fft=1024, hop=256) — the migration is a normalization and inference architecture swap, not a pipeline rewrite. New capabilities (training pipeline, evaluation harness, dataset collection) attach to the existing FastAPI service as independent packages.

The recommended approach is strict layered migration: build preprocessing parity first (single shared `MelConfig`, numerical equivalence tests), port the model architecture second, wire segment aggregation into the existing `CNNWorker` third, then add the training and evaluation pipelines as background capabilities. The Classifier/Preprocessor protocol pattern decouples the worker from any specific model, enabling clean swaps between the legacy ONNX path, the new PyTorch CNN, and future ensemble models without touching pipeline or state machine code. Stack additions are minimal: `torch + torchaudio 2.11.x` (version-locked), `scikit-learn 1.6.x`. Remove `onnxruntime`, `onnx`, and `librosa` from production requirements.

The overriding risk across all phases is silent accuracy degradation from normalization or segment duration mismatch. The research codebase is internally inconsistent — the training script uses `(S_db + 80) / 80` normalization while the inference script uses z-score. The training normalization is canonical. Additionally, the segment duration must be resolved early: the CNN was trained on 0.5s chunks with heavy zero-padding, but the live system feeds it 2.0s segments. These two decisions must be locked in before any model work begins and enforced via automated parity tests throughout the migration.

## Key Findings

### Recommended Stack

Stack additions are minimal and targeted. The existing service stack (FastAPI, sounddevice, NumPy, SciPy, pyzmq) is validated and unchanged. New dependencies are only for the classification migration. Use `--index-url https://download.pytorch.org/whl/cpu` in Docker to avoid shipping CUDA libraries (~2GB savings). The research CNN at ~95K parameters trains in minutes on CPU — no GPU, no PyTorch Lightning, no torchvision, no MLflow required. A 50-line custom training loop with manual early stopping is the correct level of abstraction for this architecture.

**Core technologies:**
- `torch + torchaudio 2.11.x`: CNN definition, training, inference, and mel-spectrogram transforms — replaces TF/Keras + librosa + ONNX Runtime in one framework. torchaudio version must match torch exactly (shared C++ extensions).
- `scikit-learn 1.6.x`: Post-training evaluation metrics (confusion matrix, F1, precision/recall) — stateless, no GPU, standard reporting. Preferred over torchmetrics (overkill for binary classification, no distributed training needed).
- `audiomentations 0.39+`: Optional waveform augmentation for training (noise injection, time stretch, pitch shift). Mature, NumPy-based. Prefer over torch-audiomentations (which has known multiprocessing memory leaks and is labeled "early development stage").
- `torchaudio.transforms.TimeMasking / FrequencyMasking`: SpecAugment for mel-spectrogram augmentation — built into torchaudio, no extra dependency.

**What to remove:** `onnxruntime`, `onnx`, `librosa`. librosa pulls in numba + llvmlite (~150MB) and is not differentiable. torchaudio mel-spectrograms are 5x faster, differentiable, and eliminating the librosa dependency simplifies the Docker image.

**Critical version note:** torchaudio and librosa mel-spectrograms differ by default. Set `norm="slaney"` and `mel_scale="slaney"` in torchaudio to match librosa's defaults. Validate numerically during porting — max absolute difference should be < 1e-4.

### Expected Features

The migration is complete when six features ship together as an integrated unit:

**Must have (table stakes — migration is incomplete without these):**
- Research CNN architecture in PyTorch + ONNX export — replaces EfficientNet-B0; `(N, 1, 128, 64)` input shape, ~15 lines of PyTorch code, already proven in research
- Research preprocessing params — switches normalization from z-score to `(S_db+80)/80` clipped to `[0,1]`; removes 224x224 resize; ~20 lines changed in `preprocessing.py`
- Segment aggregation (p_max, p_mean, p_agg) — splits 2s audio into overlapping 0.5s segments; `0.7*p_max + 0.3*p_mean` feeds the existing state machine unchanged
- PyTorch training pipeline — DataLoader, Adam, BCE, early stopping, checkpoint saving, ONNX export on completion
- Model evaluation harness — confusion matrix, precision/recall/F1, distribution stats, per-file output; safety gate before deploying any trained model
- Training + evaluation REST API — async endpoints, WebSocket progress updates

**Should have (high ROI, add in same sprint or shortly after):**
- Configurable aggregation strategy (expose `w_max`, `w_mean`, threshold as env vars)
- Training data augmentation (SpecAugment via torchaudio + waveform augmentation via audiomentations)
- Dataset collection UI (record labeled clips from live UMA-16 via web interface)

**Defer (v2+ after single-model accuracy is measured):**
- Late fusion ensemble (10-model architecture) — research implementation uses `exec()` on 10 separate scripts; untranslatable to production; requires clean redesign; ROI unknown until single-model F1 is measured
- A/B model comparison dashboard — useful after evaluation harness is stable
- Training progress WebSocket — REST polling is sufficient initially

### Architecture Approach

The architecture follows a Strategy pattern at the classifier boundary. A `Classifier` protocol replaces the hardcoded `OnnxDroneClassifier`; all new classifiers implement it. The `CNNWorker` is modified to accept injected `Classifier`, `Preprocessor`, and `SegmentAggregator` instances. A factory in `main.py` selects the classifier at startup based on config. The `DetectionStateMachine` and `TargetTracker` are untouched — they receive the same single probability value, now computed by the aggregator instead of a single ONNX call. Three new packages attach to the existing `src/acoustic/` tree (`training/`, `evaluation/`, `collection/`), each independently triggerable via FastAPI routes and isolated from the real-time audio path.

**Major components:**
1. `classification/protocol.py` — `Classifier` and `Preprocessor` protocols; foundation for all changes; build first
2. `classification/research_model.py` — `ResearchCNN` PyTorch `nn.Module`; 3 layers (Conv2d 32/64/128 + BN + MaxPool), GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid; input `(N, 1, 128, 64)`
3. `classification/research_preprocessing.py` — mel-spec pipeline with `(S_db+80)/80` normalization, 0.5s segments, `(1, 1, 128, 64)` output; shares `MelConfig` constants with training
4. `classification/aggregation.py` — `SegmentAggregator` producing p_max/p_mean/p_agg from per-segment probabilities; bounded ring buffer to prevent p_agg monotonic drift; pure NumPy
5. `classification/worker.py` (modified) — injects protocol implementations, splits chunks into 0.5s segments, accumulates predictions in bounded ring buffer
6. `training/` package — `dataset.py` (PyTorch Dataset), `trainer.py` (background thread training loop), `export.py` (torch.onnx.export)
7. `evaluation/` package — `harness.py` (port of eval_folder logic), `metrics.py` (sklearn-backed metrics + distribution stats)
8. `collection/` package — `recorder.py` (label + save from live UMA-16), `metadata.py` (Pydantic schema for recording metadata)
9. `classification/ensemble.py` — `EnsembleClassifier` wrapping N classifiers via the protocol; deferred post-MVP

### Critical Pitfalls

1. **Three incompatible normalization schemes (Pitfall 15)** — The research training uses `(S_db+80)/80` but the research inference script uses z-score, and the current service also uses z-score. Port the TRAINING normalization only. Write a numerical parity test (TF output vs PyTorch output on same WAV file, `np.allclose(atol=1e-4)`) before any model work. Create a single `MelConfig` dataclass shared between training and inference — no duplicate constants.

2. **Segment duration mismatch (Pitfall 16)** — CNN was trained on 0.5s chunks (heavily zero-padded to 128 frames); live inference currently feeds 2.0s segments (nearly fills 128 frames). These produce different feature distributions despite identical tensor shapes. Adopt 0.5s segments everywhere with sliding-window aggregation — this matches the proven evaluation pipeline (`eval_folder_with_strong_cnn.py`).

3. **BatchNorm train/eval mode (Pitfall 19)** — PyTorch requires explicit `model.eval()` before inference; TF/Keras handles this automatically. Missing `model.eval()` causes BatchNorm to use per-batch statistics during single-sample inference, producing unstable results. Always call `model.eval()` and wrap in `@torch.no_grad()`. Validate with a unit test: same input twice must produce identical output.

4. **Training starves real-time inference (Pitfall 20)** — PyTorch training on CPU competes with beamforming pipeline for cores and memory. Training must run with `os.nice(10)`, `torch.set_num_threads(2)`, `DataLoader num_workers=1`. Show a warning in the UI ("Training may degrade detection performance"). Never trigger training from the inference path.

5. **CNNWorker queue-of-1 incompatible with aggregation (Gotcha 2)** — The existing maxsize=1 drop-latest queue cannot accumulate probabilities across overlapping segments. Implement a bounded ring buffer of the last K classification probabilities alongside the existing queue; compute p_max/p_mean/p_agg over that window. Bound the window (K=10) to prevent p_agg from monotonically approaching 1.0 over time.

## Implications for Roadmap

The migration has clear dependency ordering driven by the research findings. Preprocessing correctness is the invariant that all downstream phases depend on. Model architecture and aggregation are tightly coupled and form the second phase. Training and evaluation are independent of the live inference path and can run in parallel. The existing service is untouched in all phases until Phase 3 (worker integration).

### Phase 1: Preprocessing Parity Foundation

**Rationale:** Everything downstream — model training, inference, evaluation — depends on the mel-spectrogram computation being correct and consistent. This phase has no external dependencies, produces verifiable artifacts (unit tests), and unblocks all subsequent work. Must address normalization mismatch, segment duration decision, and pad-or-trim behavior before any model work begins.

**Delivers:**
- `MelConfig` dataclass with all shared constants (SR=16000, N_FFT=1024, HOP_LENGTH=256, N_MELS=64, MAX_FRAMES=128, normalization strategy)
- `classification/protocol.py` — Classifier and Preprocessor protocols
- `classification/research_preprocessing.py` — mel-spec pipeline with `(S_db+80)/80` normalization, 0.5s segments, `(1, 1, 128, 64)` output
- Numerical parity test suite: same WAV file through TF training code and new PyTorch preprocessor must produce tensors within `atol=1e-4`

**Avoids:** Pitfalls 15 (normalization mismatch), 16 (segment duration mismatch), 17 (pad-or-trim asymmetry), 21 (librosa version skew), Debt 1 (magic number duplication)

### Phase 2: Research CNN Architecture + Inference Integration

**Rationale:** With preprocessing validated, the model architecture can be built and tested against known-good inputs. The Classifier protocol from Phase 1 allows wiring the new model into the existing worker without breaking the live pipeline. ONNX export is part of this phase — the existing `OnnxDroneClassifier` interface remains the integration seam.

**Delivers:**
- `classification/research_model.py` — `ResearchCNN` PyTorch `nn.Module` (3 layers, ~95K params) with shape assertion in `forward()`
- `classification/research_classifier.py` — implements `Classifier` protocol, loads `.pt` or `.onnx`, enforces `model.eval()` + `torch.no_grad()`
- `classification/aggregation.py` — `SegmentAggregator` with configurable `w_max`/`w_mean` and bounded ring buffer
- Modified `classification/worker.py` — accepts injected `Classifier` + `Preprocessor` + `SegmentAggregator`; splits chunks into 0.5s segments
- Updated `config.py` — `cnn_classifier_type`, `segment_seconds`, `segment_hop_seconds`, `agg_w_max/w_mean`
- Updated `main.py` — classifier factory based on config

**Avoids:** Pitfalls 18 (NCHW/NHWC), 19 (BatchNorm eval mode), Gotcha 2 (queue redesign), Gotcha 3 (preprocessing shape contract)

**Note:** This phase uses a randomly-initialized model for integration testing. The integration milestone is that the pipeline processes audio end-to-end with the new classifier without crashing or regressing beamforming. A real trained model comes from Phase 4.

### Phase 3: Evaluation Harness

**Rationale:** Before training or deploying any model, operators need a safety gate. The evaluation harness depends only on the inference path (Phase 2) and preprocessing (Phase 1). It also generates the threshold re-calibration data required before the new model goes to production. Build this in parallel with Phase 4 where possible.

**Delivers:**
- `evaluation/harness.py` — port of `eval_folder_with_strong_cnn.py`; iterates test WAV folders, runs classifier + aggregation, records all scores
- `evaluation/metrics.py` — confusion matrix, accuracy/precision/recall/F1 (sklearn), p_agg/p_max/p_mean distribution stats (percentiles), per-file detail
- Baseline evaluation comparing legacy EfficientNet vs new CNN on same test data
- Re-calibrated state machine thresholds (`enter`, `exit`) for the new CNN's confidence distribution

**Avoids:** Deploying under-performing or mis-calibrated models; checklist item on threshold re-calibration

### Phase 4: PyTorch Training Pipeline

**Rationale:** Independent of the live inference path. Depends on Phase 1 (preprocessing) and Phase 2 (model architecture). Can be built in parallel with Phase 3. Must be resource-isolated from the audio pipeline.

**Delivers:**
- `training/dataset.py` — PyTorch `Dataset` loading WAV files lazily, extracting random 0.5s segments, applying research preprocessing
- `training/trainer.py` — training loop (DataLoader, Adam lr=1e-3, BCE loss, early stopping patience=8, ReduceLR patience=3, CSV logging); runs in background thread with `os.nice(10)` and `torch.set_num_threads(2)`
- `training/export.py` — `torch.onnx.export()` wrapper with post-export validation
- `training/config.py` — `TrainingConfig` Pydantic model
- REST endpoints: `POST /api/training/start`, `GET /api/training/status`

**Avoids:** Pitfall 20 (resource starvation), Trap 3 (torch.compile warmup), checklist items (seed isolation, augmentation gating)

### Phase 5: Data Collection + REST API Completion

**Rationale:** Completes the operational loop: collect data, train, evaluate, deploy. The collection UI extends existing recording infrastructure and is largely independent of ML pipeline work. REST API completion wraps up all new endpoints.

**Delivers:**
- `collection/recorder.py` — record labeled clips from live UMA-16 via web UI; auto-save to `data/recordings/{label}/{bin}/`
- `collection/metadata.py` — Pydantic schema for recording metadata (label, distance, altitude, session)
- Evaluation REST endpoints: `POST /api/eval/run`, `GET /api/eval/results/{run_id}`
- Training progress WebSocket: `/ws/training/progress`
- Dataset stats endpoint: `GET /api/collection/stats`

**Avoids:** v1 Pitfall 9 (metadata loss), checklist item (recording I/O thread isolation)

### Phase 6: Late Fusion Ensemble (Post-MVP, Conditional)

**Rationale:** Build only after Phase 3 measures single-model accuracy and it proves insufficient. The research ensemble used `exec()` on 10 separate scripts — that implementation is untranslatable. This phase requires a clean redesign from scratch. If single-model F1 meets the accuracy requirement, skip this phase entirely.

**Delivers:**
- `classification/ensemble.py` — `EnsembleClassifier` holding N models via `Classifier` protocol; configurable soft voting with normalized accuracy-based weights
- Training orchestration for N models (loop, not exec)
- Ensemble evaluation vs single-model baseline on the evaluation harness

**Constraint:** Maximum 3 models for real-time deployment (3 x ~15ms = 45ms, within 150ms cycle). 10-model ensemble is offline-only.

**Avoids:** Debt 3 (exec-based ensemble), Trap 2 (latency multiplication), checklist (weight normalization)

### Phase Ordering Rationale

- Phase 1 must come first because normalization and segment duration decisions are prerequisites for all model and training work. A wrong normalization discovered in Phase 4 forces retraining from scratch.
- Phase 2 before Phase 4 because the model architecture and worker integration must be validated before training produces models to load into them.
- Phase 3 parallel with Phase 4 because the evaluation harness is independent of the training loop and needed to validate each trained model.
- Phase 5 after Phases 3-4 because evaluation and training REST endpoints depend on those packages existing.
- Phase 6 gated by Phase 3 results — build only if single-model accuracy is insufficient.
- The existing service (beamforming, ZeroMQ, state machine, WebSocket, React UI) is untouched in all phases. The Classifier protocol boundary ensures no regression to these components.

### Research Flags

Phases needing validation or deeper attention during planning:

- **Phase 1:** torchaudio vs librosa mel-spectrogram numerical equivalence requires empirical confirmation with `norm="slaney"` and `mel_scale="slaney"`. STACK.md has exact parameters but the actual `atol` needs to be measured.
- **Phase 2 (worker redesign):** The bounded ring buffer window size K needs concrete sizing (K=10 suggested; trades latency for accuracy). The eviction policy and its effect on p_agg stability needs validation.
- **Phase 4 (resource isolation):** Exact CPU affinity and thread limit settings for the training process on the target Docker container require profiling. The 150ms beamforming deadline is the hard constraint.
- **Phase 6:** Ensemble inference latency on target hardware is unknown. Batch inference of N models using a single `(N, 1, 128, 64)` batched forward pass may reduce latency significantly.

Phases with standard patterns (skip research-phase):

- **Phase 1 (preprocessing):** STACK.md has exact torchaudio parameters and validation approach. Research is complete.
- **Phase 2 (model architecture):** Architecture fully specified in STACK.md (3-layer CNN, exact layer sizes). No unknowns.
- **Phase 3 (evaluation):** Port of existing research eval script with sklearn metrics. Standard patterns, fully documented.
- **Phase 5 (collection UI):** Extends existing recording infrastructure. FastAPI + WebSocket patterns already established.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Exact versions confirmed from PyPI and PyTorch release notes. torchaudio version-lock requirement and torchaudio/librosa parameter differences verified against multiple sources. |
| Features | HIGH | Feature landscape derived from direct source code comparison between research pipeline and current service. MVP scope clearly bounded by the dependency graph. |
| Architecture | HIGH | Integration points identified from reading every relevant source file in `src/acoustic/classification/` and `src/acoustic/pipeline.py`. Protocol pattern is standard and battle-tested. |
| Pitfalls | HIGH | All v2 pitfalls derived from direct code analysis, not speculation. Normalization mismatch, segment duration mismatch, and CNNWorker queue design are confirmed bugs in the research codebase, not hypothetical risks. |

**Overall confidence:** HIGH

### Gaps to Address

- **Accuracy baseline unknown:** No labeled test dataset exists in the repo. Phase 3 cannot produce meaningful evaluation until training data is available. The Phase 5 collection UI or an external dataset import must be planned as a prerequisite for accuracy measurement.

- **State machine threshold re-calibration values unknown:** The new CNN's confidence distribution will differ from EfficientNet-B0. Correct `enter`/`exit` thresholds for `DetectionStateMachine` can only be determined after running Phase 3 with a trained model on labeled test data. Plan to leave these as configurable env vars and document that they require field calibration before production deployment.

- **Training data volume requirement unknown:** The research pipeline provides no guidance on how many labeled clips are needed. Standard audio classification practice suggests 500-2000 examples per class for a 3-layer CNN; actual requirement needs empirical validation.

- **Docker image size under constraint:** PyTorch CPU-only wheel adds ~280MB to the image. Final image size needs verification against any edge deployment constraints. If size is a hard blocker, evaluate ONNX-only inference (no PyTorch at runtime) with a separate training container.

## Sources

### Primary (HIGH confidence)

- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` — CNN architecture, training hyperparameters, normalization scheme, segment duration
- `Acoustic-UAV-Identification-main-main/eval_folder_with_strong_cnn.py` — segment aggregation (p_max, p_mean, p_agg), evaluation metrics
- `Acoustic-UAV-Identification-main-main/mic_realtime_inference.py` — real-time inference pattern, weighted aggregation, z-score normalization (the canonical mismatch)
- `src/acoustic/classification/preprocessing.py` — current service preprocessing, z-score normalization, 2.0s segments
- `src/acoustic/classification/worker.py` — CNNWorker design, maxsize=1 queue, drop semantics
- `src/acoustic/classification/inference.py` — OnnxDroneClassifier, EfficientNet input shape `(1, 3, 224, 224)`
- [torch 2.11.0 PyPI](https://pypi.org/project/torch/) — Python 3.14 wheel availability confirmed
- [torchaudio 2.11.0 PyPI](https://pypi.org/project/torchaudio/) — version pinning requirement
- [torchaudio MelSpectrogram docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) — parameter reference
- [torchaudio vs librosa mel comparison](https://gist.github.com/aFewThings/f4dde48993709ab67e7223e75c749d9d) — slaney norm requirement confirmed

### Secondary (MEDIUM confidence)

- `Acoustic-UAV-Identification-main-main/4 - Late Fusion Networks/` — ensemble architecture; medium confidence because exec-based implementation is a code smell and ensemble ROI on this hardware is unproven
- `audiomentations` GitHub — waveform augmentation library maturity; medium because augmentation impact on this specific dataset is unmeasured
- [AUDRON Framework paper](https://arxiv.org/pdf/2512.20407) — deep learning for drone audio classification; research paper, not production guidance
- [Robust Low-Cost Drone Detection CNN paper](https://arxiv.org/html/2406.18624v2) — confirms mel-spectrogram + CNN approach

### Tertiary (LOW confidence)

- Training data volume estimates (500-2000 examples per class) — general audio ML practice, not specific to drone detection on UMA-16v2
- Docker image size constraint assumptions — derived from known wheel sizes, not measured on target deployment hardware

---
*Research completed: 2026-04-01*
*Ready for roadmap: yes*
