# Phase 8: PyTorch Training Pipeline - Research

**Researched:** 2026-04-01
**Domain:** PyTorch training pipeline, data augmentation, background execution with resource isolation
**Confidence:** HIGH

## Summary

Phase 8 builds a PyTorch training pipeline for ResearchCNN that loads WAV files lazily, trains with Adam/BCE/early stopping, applies SpecAugment + waveform augmentation, and runs as a background thread without degrading live detection. The existing codebase already has ResearchCNN (Phase 7), ResearchPreprocessor with torchaudio MelSpectrogram (Phase 6), and a CNNWorker daemon thread pattern to reference.

The research reference implementation (`train_strong_cnn.py`) uses TensorFlow `tf.data` for lazy loading with `numpy_function` wrappers. The PyTorch equivalent uses a custom `Dataset` subclass with `DataLoader`. torchaudio provides `TimeMasking` and `FrequencyMasking` transforms for SpecAugment directly on tensors -- no custom implementation needed.

**Primary recommendation:** Build a `DroneAudioDataset(Dataset)` with lazy WAV loading and random segment extraction in `__getitem__`, reuse `ResearchPreprocessor` for mel-spectrogram conversion, apply augmentation as composable transforms, and use `threading.Thread` with `os.nice(10)` for background execution. Do NOT call `torch.set_num_threads()` from the training thread -- it is process-global and will throttle inference. Instead, set thread limits before any PyTorch operations at process startup.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Lazy loading with random 0.5s segment extraction per WAV file per epoch. Each epoch picks a new random start position, providing implicit augmentation through varying segment positions. No precomputation of spectrograms.
- **D-02:** Class balancing via PyTorch `WeightedRandomSampler` -- over-samples minority class each epoch, uses all available data without discarding files.
- **D-03:** 80/20 train/validation split at file level (not segment level) to prevent data leakage between sets.
- **D-04:** Configurable root data directory with label subdirectories (e.g., `{root}/{drone,background,other}/`). Default to `audio-data/data/` for backward compatibility.
- **D-05:** Early stopping with patience-based monitoring on validation loss. Default patience=5 epochs. Saves best model checkpoint based on lowest validation loss.
- **D-06:** Research-aligned default hyperparameters: Adam optimizer, lr=1e-3, batch_size=32, max_epochs=50, patience=5. All configurable via training config.
- **D-07:** BCE loss (binary cross-entropy) with sigmoid output -- matches ResearchCNN architecture (Sigmoid final layer).
- **D-08:** Export format is PyTorch `.pt` (state_dict) only. No ONNX export. Consistent with Phase 7 D-05.
- **D-09:** SpecAugment applied on mel-spectrogram after preprocessing: random time masks (up to 20 frames) + frequency masks (up to 8 mel bins).
- **D-10:** Waveform augmentation applied before mel-spectrogram extraction: Gaussian noise injection (SNR 10-40dB range) + random gain scaling (+/-6dB).
- **D-11:** Single config toggle to enable/disable all augmentation.
- **D-12:** Training runs as a daemon background thread with `os.nice(10)` to lower CPU priority and `torch.set_num_threads(2)` to cap compute threads.
- **D-13:** Thread-safe in-memory progress state object (epoch, loss, val_loss, val_acc, status, best_val_loss). Updated by training thread, polled by Phase 9 API.
- **D-14:** Training is cancellable mid-run via a threading stop event. Training thread checks the event between epochs.
- **D-15:** Single concurrent training run. Starting a new run while one is active returns an error.

### Claude's Discretion
- Where to place training module code
- PyTorch Dataset class implementation details (caching strategy, error handling for corrupt files)
- Learning rate scheduler choice (CosineAnnealing, ReduceLROnPlateau, or none)
- Exact SpecAugment parameters (num_masks, mask_length ranges)
- Thread-safe state object implementation (dataclass with lock, or atomic fields)
- Whether to reuse `ResearchPreprocessor` directly or create a training-specific preprocessor variant

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRN-01 | PyTorch training pipeline loads WAV files lazily, extracts random 0.5s segments, trains with Adam/BCE/early stopping | Dataset class with lazy loading, random segment in `__getitem__`, standard PyTorch training loop, EarlyStopping class |
| TRN-02 | Training runs as background thread with resource isolation without degrading live detection | `threading.Thread(daemon=True)` + `os.nice(10)`, process-level thread cap at startup, stop event for cancellation |
| TRN-03 | Training produces a model checkpoint and exports to deployable format on completion | `torch.save(model.state_dict(), path)` producing `.pt` file compatible with classifier factory |
| TRN-04 | Training data augmentation applies SpecAugment and waveform augmentation during training | torchaudio `TimeMasking`/`FrequencyMasking` for SpecAugment, custom waveform noise/gain augmentation |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0 | Training loop, optimizer, loss | Already installed, ResearchCNN uses it |
| torchaudio | 2.11.0 | SpecAugment transforms (TimeMasking, FrequencyMasking) | Already installed, used by ResearchPreprocessor |
| soundfile | 0.13.1 | WAV file reading in Dataset | Already installed, lightweight I/O |
| numpy | 2.4.4 | Audio array manipulation | Already installed |

### No New Dependencies Required
This phase requires zero new pip packages. All functionality is covered by the existing stack.

## Architecture Patterns

### Recommended Module Structure
```
src/acoustic/training/
    __init__.py
    config.py          # TrainingConfig (Pydantic BaseSettings)
    dataset.py          # DroneAudioDataset(Dataset)
    augmentation.py     # WaveformAugmentation + SpecAugmentTransform
    trainer.py          # TrainingRunner (training loop + early stopping)
    manager.py          # TrainingManager (thread lifecycle, progress state, concurrency guard)
```

**Rationale:** Separate `src/acoustic/training/` package rather than nesting inside `classification/`. Training is a distinct concern from inference -- different lifecycle, different dependencies on config, and different callers (Phase 9 API). The classification package stays focused on runtime inference.

### Pattern 1: Lazy-Loading Dataset with Random Segment Extraction
**What:** Custom `Dataset` that reads WAV files on-the-fly in `__getitem__`, picks a random 0.5s segment each call, and produces a mel-spectrogram tensor.
**When to use:** When dataset is too large to precompute (42k+ WAV files) and random segment extraction provides implicit augmentation.
**Example:**
```python
# Source: PyTorch Dataset API + research train_strong_cnn.py pattern
class DroneAudioDataset(Dataset):
    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        mel_config: MelConfig,
        augmentation: AugmentationPipeline | None = None,
    ):
        self._paths = file_paths
        self._labels = labels
        self._mel_config = mel_config
        self._augmentation = augmentation
        self._preprocessor = ResearchPreprocessor(mel_config)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sr = sf.read(self._paths[idx], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono

        # Random 0.5s segment extraction (D-01)
        n = self._mel_config.segment_samples
        if len(audio) >= n:
            start = random.randint(0, len(audio) - n)
            segment = audio[start:start + n]
        else:
            segment = np.pad(audio, (0, n - len(audio)))

        # Waveform augmentation before mel-spec (D-10)
        if self._augmentation:
            segment = self._augmentation.waveform(segment)

        # Resample if needed, then mel-spectrogram
        # NOTE: Cannot reuse ResearchPreprocessor.process() directly because
        # it takes the last segment, not a random one. Extract mel-spec logic.
        features = self._extract_mel(segment, sr)

        # SpecAugment after mel-spec (D-09)
        if self._augmentation:
            features = self._augmentation.specaugment(features)

        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return features.squeeze(0), label  # (1, 128, 64), scalar
```

### Pattern 2: WeightedRandomSampler for Class Balancing
**What:** Assigns per-sample weights inversely proportional to class frequency, so minority class is oversampled.
**When to use:** Binary classification with imbalanced drone vs background counts (D-02).
**Example:**
```python
# Source: PyTorch docs - torch.utils.data.WeightedRandomSampler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Calculate per-sample weights
class_counts = [num_background, num_drone]
class_weights = [1.0 / c for c in class_counts]
sample_weights = [class_weights[label] for label in all_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,  # Required for oversampling
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,  # Cannot use shuffle=True with sampler
    num_workers=0,     # Keep 0 -- training runs in background thread
)
```

### Pattern 3: Early Stopping with Best Checkpoint
**What:** Monitor validation loss, save best model, stop if no improvement for `patience` epochs.
**When to use:** Every training run (D-05).
**Example:**
```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # improved -- save checkpoint
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False      # no improvement
```

### Pattern 4: Thread-Safe Progress State
**What:** Dataclass guarded by a threading.Lock for atomic reads/writes from training thread and API polling.
**When to use:** D-13 progress reporting consumed by Phase 9 endpoints.
**Example:**
```python
import threading
from dataclasses import dataclass, field
from enum import Enum

class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class TrainingProgress:
    status: TrainingStatus = TrainingStatus.IDLE
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    best_val_loss: float = float("inf")
    error: str | None = None

class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._progress = TrainingProgress()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def get_progress(self) -> TrainingProgress:
        with self._lock:
            # Return a copy to avoid race conditions
            return TrainingProgress(**vars(self._progress))

    def _update_progress(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._progress, k, v)
```

### Anti-Patterns to Avoid
- **Calling `torch.set_num_threads()` from training thread:** This is PROCESS-GLOBAL and will throttle the CNNWorker inference thread. See Pitfall 1.
- **Using `num_workers > 0` in DataLoader:** Training runs in a daemon background thread. Spawning subprocesses from daemon threads is unreliable and risks zombie processes on cancellation.
- **Precomputing all spectrograms into memory:** 42k WAV files at 0.5s each would be manageable, but defeats the random segment extraction that provides per-epoch augmentation variety.
- **Using `BCEWithLogitsLoss`:** ResearchCNN already has Sigmoid in its final layer. Using BCEWithLogitsLoss would apply sigmoid twice. Use `nn.BCELoss` instead.
- **Splitting by segment instead of file:** Per D-03, the train/val split must be at file level. Never put different segments from the same file in both train and val sets.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SpecAugment | Custom masking logic | `torchaudio.transforms.TimeMasking` + `FrequencyMasking` | Battle-tested, GPU-compatible, correct masking distribution |
| Learning rate scheduling | Manual LR decay | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Handles plateau detection, warmup, min_lr. Research reference uses it |
| Weighted sampling | Manual epoch rebalancing | `torch.utils.data.WeightedRandomSampler` | Correct probabilistic oversampling, integrates with DataLoader |
| Audio I/O | Manual WAV parsing | `soundfile.read()` | Handles formats, dtypes, multi-channel correctly |

**Key insight:** PyTorch's DataLoader ecosystem handles all the lazy loading, batching, and sampling mechanics. The only custom code needed is the Dataset `__getitem__` (segment extraction + preprocessing) and the training loop itself.

## Common Pitfalls

### Pitfall 1: torch.set_num_threads Is Process-Global (CRITICAL)
**What goes wrong:** D-12 specifies `torch.set_num_threads(2)` in the training thread. This call is PROCESS-GLOBAL -- it affects ALL PyTorch operations in the entire process, including the CNNWorker inference thread. Setting it to 2 from the training thread will throttle inference.
**Why it happens:** PyTorch's thread count is stored as a global variable (OMP/MKL), not per-thread.
**How to avoid:** Two options:
1. **Recommended:** Set `torch.set_num_threads()` ONCE at process startup to a reasonable total (e.g., 4), and accept that training and inference share the thread pool. `os.nice(10)` already deprioritizes training at the OS level.
2. **Alternative:** Do not call `torch.set_num_threads()` at all. Rely solely on `os.nice(10)` for priority isolation. The OS scheduler handles CPU allocation.
**Warning signs:** Live detection latency spikes when training starts.

### Pitfall 2: BCELoss vs BCEWithLogitsLoss
**What goes wrong:** Using `nn.BCEWithLogitsLoss` with ResearchCNN which already has `nn.Sigmoid()` in its forward pass. This applies sigmoid twice, producing wrong gradients.
**Why it happens:** Many tutorials use BCEWithLogitsLoss (numerically more stable). But ResearchCNN was designed to match the research Keras model which has `activation="sigmoid"` in the output layer.
**How to avoid:** Use `nn.BCELoss()`. The model output is already in [0, 1].
**Warning signs:** Training loss doesn't decrease, model outputs cluster near 0.5.

### Pitfall 3: Data Leakage via Segment-Level Split
**What goes wrong:** Using PyTorch's `random_split` on a Dataset where multiple segments come from the same WAV file puts correlated data in both train and val sets.
**Why it happens:** The research reference (`train_strong_cnn.py`) splits at file level using `train_test_split` before creating datasets. Our lazy Dataset extracts one random segment per file per epoch, so each Dataset item IS a file. But if a future change extracts multiple segments per file, splitting the Dataset directly would leak.
**How to avoid:** Split the file list FIRST (D-03: 80/20 at file level), then create separate Dataset instances for train and val. Never use `random_split` on the Dataset itself.
**Warning signs:** Suspiciously high validation accuracy that doesn't generalize.

### Pitfall 4: Daemon Thread Cleanup on Cancellation
**What goes wrong:** Training thread holds GPU/memory resources after cancellation. If not properly cleaned up, subsequent training runs fail or leak memory.
**Why it happens:** `stop_event.set()` signals between epochs, but mid-batch computation continues until the current epoch finishes.
**How to avoid:** Check stop event at epoch boundaries (between train and val). On cancellation, explicitly `del` the DataLoader, clear CUDA cache if applicable, and save the best checkpoint found so far.
**Warning signs:** Memory usage doesn't return to baseline after cancellation.

### Pitfall 5: ResearchPreprocessor Cannot Be Reused Directly
**What goes wrong:** `ResearchPreprocessor.process()` takes `audio + sr`, extracts the LAST `segment_samples` from the waveform, and returns `(1, 1, 128, 64)`. Training needs a RANDOM segment, not the last one.
**Why it happens:** The preprocessor was designed for inference (process the most recent audio chunk), not training (random augmentation).
**How to avoid:** Extract the mel-spectrogram conversion logic (MelSpectrogram transform + power_to_db + normalization + padding) into a reusable function or use the preprocessor's `_mel_spec` transform directly. The Dataset handles segment extraction separately.
**Warning signs:** All training samples from the same file are identical (always the last 0.5s).

## Code Examples

### SpecAugment with torchaudio
```python
# Source: torchaudio.transforms docs
import torchaudio.transforms as T

class SpecAugment:
    """SpecAugment: time + frequency masking on mel-spectrogram."""

    def __init__(
        self,
        time_mask_param: int = 20,   # D-09: up to 20 frames
        freq_mask_param: int = 8,    # D-09: up to 8 mel bins
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self._time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param)
            for _ in range(num_time_masks)
        ])
        self._freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(num_freq_masks)
        ])

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # spectrogram shape: (1, 128, 64) = (channel, time, freq)
        # TimeMasking expects (..., freq, time), so transpose
        x = spectrogram.transpose(-1, -2)  # (1, 64, 128)
        for mask in self._time_masks:
            x = mask(x)
        for mask in self._freq_masks:
            x = mask(x)
        return x.transpose(-1, -2)  # back to (1, 128, 64)
```

**Important note on tensor shape:** torchaudio's `TimeMasking` and `FrequencyMasking` expect input shape `(..., freq, time)`. ResearchCNN uses `(batch, 1, time_frames, n_mels)` = `(N, 1, 128, 64)`. The spectrogram from preprocessing is `(1, 128, 64)` which is `(channel, time, freq)`. Must transpose to `(channel, freq, time)` before applying torchaudio masking transforms, then transpose back.

### Waveform Augmentation
```python
# Source: D-10 requirements + standard practice
import numpy as np

class WaveformAugmentation:
    """Gaussian noise injection + random gain scaling."""

    def __init__(self, snr_range: tuple[float, float] = (10.0, 40.0), gain_db: float = 6.0):
        self.snr_range = snr_range
        self.gain_db = gain_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # Gaussian noise injection (D-10)
        snr_db = np.random.uniform(*self.snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), audio.shape)
        audio = audio + noise.astype(np.float32)

        # Random gain scaling +/-6dB (D-10)
        gain_db = np.random.uniform(-self.gain_db, self.gain_db)
        gain_linear = 10 ** (gain_db / 20)
        audio = audio * gain_linear

        return audio.astype(np.float32)
```

### Training Loop with Cancellation and Progress
```python
# Source: Standard PyTorch training pattern + D-12/D-13/D-14
def _train_loop(self, config: TrainingConfig, stop_event: threading.Event):
    os.nice(10)  # D-12: lower CPU priority

    model = ResearchCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()  # NOT BCEWithLogitsLoss -- model has Sigmoid
    early_stopping = EarlyStopping(patience=config.patience)

    for epoch in range(config.max_epochs):
        if stop_event.is_set():  # D-14: cancellation check
            break

        # --- Train epoch ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        # --- Validate epoch ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x).squeeze(-1)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = (output >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        # --- Early stopping + checkpoint ---
        improved = early_stopping.step(val_loss / val_total)
        if improved:
            torch.save(model.state_dict(), config.checkpoint_path)

        # --- Update progress (D-13) ---
        self._update_progress(
            epoch=epoch + 1,
            train_loss=train_loss / len(train_loader.dataset),
            val_loss=val_loss / val_total,
            val_acc=val_correct / val_total,
            best_val_loss=early_stopping.best_loss,
        )

        if early_stopping.should_stop:
            break
```

### Model Checkpoint Save/Load
```python
# Source: PyTorch docs + D-08
# Save (training side)
torch.save(model.state_dict(), "models/research_cnn_trained.pt")

# Load (inference side -- already implemented in classifier factory)
model = ResearchCNN()
model.load_state_dict(torch.load("models/research_cnn_trained.pt", weights_only=True))
model.eval()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TF `tf.data.Dataset` with `numpy_function` | PyTorch `Dataset` + `DataLoader` | Project decision (Phase 6+) | Entire ML stack is now PyTorch |
| EfficientNet-B0 (224x224 resize) | ResearchCNN (128x64 native) | Phase 7 | No resize needed, faster training |
| librosa mel-spectrogram | torchaudio MelSpectrogram | Phase 6 | GPU-compatible, no librosa dependency |
| ONNX export | `.pt` state_dict only | Phase 7 D-05 | Simpler, classifier factory handles loading |
| Precomputed spectrograms in memory | Lazy loading per-epoch | D-01 | Scales to large datasets, random segment augmentation |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 + pytest-asyncio 1.3.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `.venv/bin/pytest tests/unit/ -x -q` |
| Full suite command | `.venv/bin/pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRN-01 | Dataset loads WAV lazily, random 0.5s segment, trains with Adam/BCE/early stopping | unit | `.venv/bin/pytest tests/unit/test_training_dataset.py -x` | Wave 0 |
| TRN-01 | Training loop converges on tiny dataset (smoke test) | integration | `.venv/bin/pytest tests/integration/test_training_smoke.py -x` | Wave 0 |
| TRN-02 | Training runs as background thread, cancellable, single-run guard | unit | `.venv/bin/pytest tests/unit/test_training_manager.py -x` | Wave 0 |
| TRN-03 | Training saves .pt checkpoint loadable by classifier factory | unit | `.venv/bin/pytest tests/unit/test_training_checkpoint.py -x` | Wave 0 |
| TRN-04 | SpecAugment + waveform augmentation applied during training | unit | `.venv/bin/pytest tests/unit/test_augmentation.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/pytest tests/unit/ -x -q`
- **Per wave merge:** `.venv/bin/pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_training_dataset.py` -- covers TRN-01 (dataset, segment extraction, class balancing)
- [ ] `tests/unit/test_augmentation.py` -- covers TRN-04 (SpecAugment, waveform augmentation)
- [ ] `tests/unit/test_training_manager.py` -- covers TRN-02 (thread lifecycle, cancellation, progress)
- [ ] `tests/unit/test_training_checkpoint.py` -- covers TRN-03 (save/load .pt checkpoint)
- [ ] `tests/integration/test_training_smoke.py` -- covers TRN-01 (end-to-end training on synthetic data)

## Open Questions

1. **Learning rate scheduler choice**
   - What we know: Research reference uses `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)`. The old EfficientNet script uses `CosineAnnealingLR`.
   - What's unclear: Which is better for ResearchCNN specifically.
   - Recommendation: Use `ReduceLROnPlateau` -- matches research reference, adapts to actual loss dynamics. CosineAnnealing assumes a fixed schedule. This is Claude's discretion per CONTEXT.md.

2. **ResearchPreprocessor reuse strategy**
   - What we know: Cannot call `process()` directly (it takes the last segment, not random). But the internal `_mel_spec` transform and `_power_to_db` function contain the exact preprocessing math.
   - What's unclear: Whether to extract a shared function, subclass, or duplicate the logic.
   - Recommendation: Create a `mel_spectrogram_from_segment(segment, mel_config) -> Tensor` utility function in `preprocessing.py` that both ResearchPreprocessor and the training Dataset can call. This avoids duplication while keeping the segment selection logic separate.

3. **torch.set_num_threads vs os.nice for resource isolation**
   - What we know: `torch.set_num_threads()` is process-global (verified). D-12 specifies both `os.nice(10)` and `torch.set_num_threads(2)`.
   - What's unclear: Whether the user insists on thread limiting or if `os.nice` alone satisfies the intent.
   - Recommendation: Apply `os.nice(10)` in the training thread (safe, per-thread). For thread limiting, set `torch.set_num_threads()` ONCE at application startup to a moderate value (e.g., 4) that works for both inference and training. Document this tradeoff in the plan.

## Sources

### Primary (HIGH confidence)
- `src/acoustic/classification/research_cnn.py` -- ResearchCNN architecture (existing code)
- `src/acoustic/classification/preprocessing.py` -- ResearchPreprocessor mel-spec pipeline (existing code)
- `src/acoustic/classification/worker.py` -- CNNWorker daemon thread pattern (existing code)
- `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` -- Research training reference
- `scripts/train_cnn.py` -- Existing training script (EfficientNet, patterns for data loading)
- [PyTorch DataLoader docs](https://docs.pytorch.org/docs/stable/data.html) -- Dataset, DataLoader, WeightedRandomSampler API
- [torchaudio TimeMasking](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html) -- SpecAugment time masking parameters
- [torchaudio FrequencyMasking](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html) -- SpecAugment frequency masking parameters

### Secondary (MEDIUM confidence)
- [torch.set_num_threads docs](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html) -- Confirmed process-global behavior
- [CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) -- Thread pool architecture
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) -- Resource isolation guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages already installed and version-verified
- Architecture: HIGH -- patterns directly derived from existing codebase + PyTorch standard practices
- Pitfalls: HIGH -- torch.set_num_threads global scope verified with multiple sources, BCELoss/BCEWithLogitsLoss distinction verified against ResearchCNN code
- Augmentation: HIGH -- torchaudio SpecAugment API verified in official docs

**Research date:** 2026-04-01
**Valid until:** 2026-05-01 (stable -- PyTorch 2.11 and torchaudio 2.11 are installed and pinned)
