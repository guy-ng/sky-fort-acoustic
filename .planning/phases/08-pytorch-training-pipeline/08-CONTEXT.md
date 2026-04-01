# Phase 8: PyTorch Training Pipeline - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a PyTorch training pipeline that loads labeled WAV files lazily, extracts random 0.5s segments, trains ResearchCNN with Adam/BCE/early stopping, applies data augmentation (SpecAugment + waveform), runs as a background thread with resource isolation, and produces a .pt model checkpoint on completion. This phase does NOT include REST API endpoints or WebSocket progress streaming (Phase 9), field data collection UI (Phase 10), or ensemble support (Phase 11).

</domain>

<decisions>
## Implementation Decisions

### Dataset & Data Loading
- **D-01:** Lazy loading with random 0.5s segment extraction per WAV file per epoch. Each epoch picks a new random start position, providing implicit augmentation through varying segment positions. No precomputation of spectrograms.
- **D-02:** Class balancing via PyTorch `WeightedRandomSampler` — over-samples minority class each epoch, uses all available data without discarding files.
- **D-03:** 80/20 train/validation split at file level (not segment level) to prevent data leakage between sets.
- **D-04:** Configurable root data directory with label subdirectories (e.g., `{root}/{drone,background,other}/`). Default to `audio-data/data/` for backward compatibility. Phase 10 recordings will auto-organize into compatible layout.

### Training Loop & Early Stopping
- **D-05:** Early stopping with patience-based monitoring on validation loss. Default patience=5 epochs. Saves best model checkpoint based on lowest validation loss.
- **D-06:** Research-aligned default hyperparameters: Adam optimizer, lr=1e-3, batch_size=32, max_epochs=50, patience=5. All configurable via training config.
- **D-07:** BCE loss (binary cross-entropy) with sigmoid output — matches ResearchCNN architecture (Sigmoid final layer).
- **D-08:** Export format is PyTorch `.pt` (state_dict) only. No ONNX export. Consistent with Phase 7 D-05 — classifier factory already loads .pt files.

### Data Augmentation
- **D-09:** SpecAugment applied on mel-spectrogram after preprocessing: random time masks (up to 20 frames) + frequency masks (up to 8 mel bins).
- **D-10:** Waveform augmentation applied before mel-spectrogram extraction: Gaussian noise injection (SNR 10-40dB range) + random gain scaling (±6dB).
- **D-11:** Single config toggle to enable/disable all augmentation. When off, no augmentation applied (useful for quick testing). Requirements TRN-04 satisfied.

### Resource Isolation & Background Execution
- **D-12:** Training runs as a daemon background thread with `os.nice(10)` to lower CPU priority and `torch.set_num_threads(2)` to cap compute threads. Must not degrade live detection below 150ms beamforming deadline.
- **D-13:** Thread-safe in-memory progress state object (epoch, loss, val_loss, val_acc, status, best_val_loss). Updated by training thread, polled by Phase 9 API/WebSocket endpoints. No file-based persistence — progress is ephemeral.
- **D-14:** Training is cancellable mid-run via a threading stop event. Training thread checks the event between epochs. Best checkpoint saved so far is preserved on cancellation.
- **D-15:** Single concurrent training run. Starting a new run while one is active returns an error. Simple and predictable resource usage.

### Claude's Discretion
- Where to place training module code (e.g., `src/acoustic/training/` or `src/acoustic/classification/training.py`)
- PyTorch Dataset class implementation details (caching strategy, error handling for corrupt files)
- Learning rate scheduler choice (CosineAnnealing, ReduceLROnPlateau, or none)
- Exact SpecAugment parameters (num_masks, mask_length ranges)
- Thread-safe state object implementation (dataclass with lock, or atomic fields)
- Whether to reuse `ResearchPreprocessor` directly or create a training-specific preprocessor variant

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Research Reference Implementation
- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` — Canonical research training pipeline: data loading, segment extraction, mel-spectrogram preprocessing, CNN training with Adam/BCE/early stopping. Ground truth for training approach.

### Existing Service Code (Phase 8 touches these)
- `src/acoustic/classification/research_cnn.py` — `ResearchCNN` model class (3-layer Conv2D 32/64/128). This is the model to train. `ResearchClassifier` wrapper for inference.
- `src/acoustic/classification/config.py` — `MelConfig` dataclass with research preprocessing parameters.
- `src/acoustic/classification/preprocessing.py` — `ResearchPreprocessor` with torchaudio MelSpectrogram. May be reused or adapted for training pipeline.
- `src/acoustic/classification/protocols.py` — Classifier and Preprocessor protocols. Training output must produce models compatible with these.
- `src/acoustic/config.py` — `AcousticSettings` with `cnn_model_path` pointing to where trained model is saved.
- `scripts/train_cnn.py` — Old EfficientNet training script. Reference for data loading patterns (collect_wav_files, directory structure) but architecture/preprocessing is obsolete.

### Existing Training Data
- `audio-data/data/` — Labeled directories: `drone/`, `background/`, `other/` with WAV files. Default training data source.

### Requirements
- `.planning/REQUIREMENTS.md` — TRN-01, TRN-02, TRN-03, TRN-04 define acceptance criteria for this phase.

### Prior Phase Context
- `.planning/phases/06-preprocessing-parity-foundation/06-CONTEXT.md` — MelConfig, torchaudio decision, protocol design.
- `.planning/phases/07-research-cnn-and-inference-integration/07-CONTEXT.md` — ResearchCNN architecture, .pt-only format, classifier factory.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ResearchCNN` in `research_cnn.py` — Model class ready for training. Sigmoid output, (N, 1, 128, 64) input shape.
- `ResearchPreprocessor` in `preprocessing.py` — torchaudio mel-spectrogram pipeline. May be reusable in training Dataset.
- `MelConfig` in `config.py` — Frozen dataclass with all preprocessing constants.
- `AcousticSettings` in `config.py` — Pydantic BaseSettings pattern for config via env vars.
- `collect_wav_files()` pattern in `scripts/train_cnn.py` — Directory scanning logic (adaptable for configurable root).

### Established Patterns
- Config via Pydantic `BaseSettings` with `ACOUSTIC_` env prefix
- Daemon threads for background processing (CNNWorker pattern)
- Protocol-based dependency injection (Classifier, Preprocessor, Aggregator)
- FastAPI lifespan for startup/shutdown coordination

### Integration Points
- Training produces a `.pt` file at `cnn_model_path` location — same path the classifier factory reads at startup
- Training thread must coexist with CNNWorker inference thread without resource contention
- Progress state object will be consumed by Phase 9 REST/WebSocket endpoints
- `main.py` lifespan will need to manage training thread lifecycle (startup optional, shutdown cleanup)

</code_context>

<specifics>
## Specific Ideas

- Research `train_strong_cnn.py` uses random segment extraction — adopted as D-01
- ResearchCNN already has Sigmoid output, so BCE loss applies directly (no BCEWithLogitsLoss needed)
- Old script used EfficientNet with 224x224 resize — completely replaced. New pipeline uses native (1, 1, 128, 64) input
- Training data currently lives in `audio-data/data/` but is git-excluded (large files). Pipeline must handle missing/empty data gracefully

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-pytorch-training-pipeline*
*Context gathered: 2026-04-01*
