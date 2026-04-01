# Phase 7: Research CNN and Inference Integration - Research

**Researched:** 2026-04-01
**Domain:** PyTorch CNN model porting, segment aggregation, protocol-based DI
**Confidence:** HIGH

## Summary

Phase 7 integrates the research CNN architecture into the live detection pipeline. The work is well-bounded: port the TF `build_model()` to a PyTorch `nn.Module`, add an `Aggregator` protocol alongside existing `Classifier`/`Preprocessor`, buffer overlapping 0.5s segments in `CNNWorker`, wire the classifier factory in `main.py` lifespan, and add aggregation weight config to `AcousticSettings`. The state machine requires zero code changes -- thresholds are already env-configurable.

The existing codebase is clean and ready for this phase. `ResearchPreprocessor` produces `(1, 1, 128, 64)` tensors, `CNNWorker` has the dormant classifier slot (`classifier=None`), and `main.py` has the comment "No classifier until Phase 7." All protocols are `@runtime_checkable`. PyTorch 2.11.0 and torchaudio 2.11.0 are installed.

**Primary recommendation:** Port the TF model layer-by-layer to `nn.Module`, add segment buffering inside `CNNWorker._loop()`, create `Aggregator` protocol + `WeightedAggregator` default, and wire the factory in `main.py` lifespan -- all using the established protocol injection pattern.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Aggregation window is 2 seconds (4 overlapping 0.5s segments). Matches the old 2.0s CNN window size, balancing latency vs. stability.
- **D-02:** Segments overlap by 50% (0.25s hop between segments). Standard in audio ML to capture events at segment boundaries.
- **D-03:** Final probability uses weighted combination: `p_agg = w_max * p_max + w_mean * p_mean` with configurable weights (default 0.5/0.5).
- **D-04:** When the configured model file doesn't exist at startup, the service boots normally with CNNWorker in dormant mode (classifier=None). Logs a warning.
- **D-05:** Only PyTorch `.pt` format supported. No legacy .h5 or ONNX format support.
- **D-06:** Classifier factory in main.py reads `cnn_model_path` from AcousticSettings, instantiates ResearchCNN, loads state_dict, and injects into CNNWorker. Falls back to None if model file missing.
- **D-07:** Keep current thresholds (enter=0.80, exit=0.40, confirm_hits=2) as defaults. Already configurable via env vars. No code changes to state machine.
- **D-08:** Aggregated p_agg feeds directly into the existing state machine as the single probability value.
- **D-09:** New `Aggregator` protocol: `aggregate(probabilities: list[float]) -> float`. Injected into CNNWorker as third protocol dependency.
- **D-10:** Aggregation weights configurable via `ACOUSTIC_CNN_AGG_W_MAX` and `ACOUSTIC_CNN_AGG_W_MEAN` env vars. Defaults 0.5/0.5.

### Claude's Discretion
- PyTorch model class placement (e.g., `classification/model.py` or `classification/research_cnn.py`)
- How CNNWorker manages the segment buffer internally (ring buffer, deque, list)
- Whether to add a model validation step on load (forward pass with dummy tensor)
- How overlapping segments are generated from the audio buffer in pipeline.py
- Default Aggregator implementation class name and location

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MDL-01 | Research CNN architecture (3-layer Conv2D 32/64/128 + BN + MaxPool, GlobalAvgPool, Dense 128, Dropout 0.3, Sigmoid) implemented in PyTorch | TF->PyTorch layer mapping documented; exact architecture from `train_strong_cnn.py` line 208 |
| MDL-02 | Segment aggregation splits audio into overlapping 0.5s segments and computes p_max, p_mean, p_agg with configurable weights | Aggregator protocol design, segment buffer strategy, weighted combination formula |
| MDL-03 | CNNWorker accepts injected Classifier/Preprocessor/Aggregator via protocols; classifier factory selects implementation at startup | Existing protocol pattern extended; factory wiring in main.py lifespan |
| MDL-04 | State machine thresholds are re-calibratable via config for the new CNN's confidence distribution | Already configurable via env vars; no code changes needed, just verification |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.11.0 (installed) | CNN model definition + inference | Already installed, research model ports directly |
| torchaudio | 2.11.0 (installed) | Mel-spectrogram preprocessing | Already used by ResearchPreprocessor |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| NumPy | (installed) | Audio buffer management | Segment slicing, mono conversion |
| pydantic-settings | (installed) | Config with env var override | New aggregation weight settings |

No new dependencies required. All libraries are already installed and in use.

## Architecture Patterns

### Recommended File Placement

**Discretion decision: Use `classification/research_cnn.py` for the model class.**

Rationale: Separating the model from a generic `model.py` makes it clear this is specifically the research CNN. Future models (ensemble members, different architectures) get their own files. The `classification/` package grows organically:

```
src/acoustic/classification/
    __init__.py
    config.py              # MelConfig (unchanged)
    preprocessing.py       # ResearchPreprocessor (unchanged)
    protocols.py           # Classifier, Preprocessor, + NEW Aggregator
    research_cnn.py        # NEW: ResearchCNN(nn.Module) + ResearchClassifier
    aggregation.py         # NEW: WeightedAggregator (default Aggregator impl)
    state_machine.py       # DetectionStateMachine (unchanged)
    worker.py              # CNNWorker (modified: add aggregator + segment buffer)
```

### Pattern 1: TF-to-PyTorch Model Port

**What:** Exact layer-by-layer translation of `build_model()` from TF/Keras to PyTorch `nn.Module`.

**Key mappings from the research code (`train_strong_cnn.py` line 208-235):**

| TF/Keras Layer | PyTorch Equivalent | Notes |
|----------------|-------------------|-------|
| `Conv2D(32, (3,3), padding="same", activation="relu")` | `nn.Conv2d(1, 32, 3, padding=1)` + `nn.ReLU()` | in_channels=1 for single mel channel |
| `BatchNormalization()` | `nn.BatchNorm2d(32)` | Channels match preceding conv |
| `MaxPooling2D(pool_size=(2,2))` | `nn.MaxPool2d(2)` | |
| `Conv2D(64, (3,3), padding="same", activation="relu")` | `nn.Conv2d(32, 64, 3, padding=1)` + `nn.ReLU()` | |
| `Conv2D(128, (3,3), padding="same", activation="relu")` | `nn.Conv2d(64, 128, 3, padding=1)` + `nn.ReLU()` | |
| `GlobalAveragePooling2D()` | `nn.AdaptiveAvgPool2d(1)` + `flatten` | Output: (B, 128) |
| `Dense(128, activation="relu")` | `nn.Linear(128, 128)` + `nn.ReLU()` | |
| `Dropout(0.3)` | `nn.Dropout(0.3)` | |
| `Dense(1, activation="sigmoid")` | `nn.Linear(128, 1)` + `nn.Sigmoid()` | |

**Input tensor format difference:**
- TF: `(N, 128, 64, 1)` -- channels-last (NHWC)
- PyTorch: `(N, 1, 128, 64)` -- channels-first (NCHW)
- `ResearchPreprocessor` already outputs `(1, 1, 128, 64)` -- correct for PyTorch.

**Example:**
```python
import torch
import torch.nn as nn


class ResearchCNN(nn.Module):
    """Research CNN matching train_strong_cnn.py build_model() exactly.

    Architecture: 3x (Conv2d->BN->ReLU->MaxPool) -> GlobalAvgPool -> Dense 128 -> Dropout 0.3 -> Sigmoid
    Input: (N, 1, 128, 64) -- (batch, channels, max_frames, n_mels)
    Output: (N, 1) -- drone probability per sample
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Pattern 2: Classifier Wrapper (Satisfies Classifier Protocol)

**What:** Wrap `ResearchCNN` in a class that satisfies the `Classifier` protocol with `predict(features) -> float`.

```python
class ResearchClassifier:
    """Wraps ResearchCNN to satisfy the Classifier protocol."""

    def __init__(self, model: ResearchCNN) -> None:
        self._model = model
        self._model.eval()

    def predict(self, features: torch.Tensor) -> float:
        with torch.no_grad():
            prob = self._model(features)
        return prob.item()
```

### Pattern 3: Aggregator Protocol + Default Implementation

**What:** New protocol for segment probability aggregation, with a configurable weighted implementation.

```python
# In protocols.py
@runtime_checkable
class Aggregator(Protocol):
    def aggregate(self, probabilities: list[float]) -> float:
        """Aggregate per-segment probabilities into a single score."""
        ...

# In aggregation.py
class WeightedAggregator:
    """Default aggregator: p_agg = w_max * max(probs) + w_mean * mean(probs)."""

    def __init__(self, w_max: float = 0.5, w_mean: float = 0.5) -> None:
        self._w_max = w_max
        self._w_mean = w_mean

    def aggregate(self, probabilities: list[float]) -> float:
        if not probabilities:
            return 0.0
        p_max = max(probabilities)
        p_mean = sum(probabilities) / len(probabilities)
        return self._w_max * p_max + self._w_mean * p_mean
```

### Pattern 4: Segment Buffer in CNNWorker

**Discretion decision: Use `collections.deque` with maxlen for the segment buffer.**

Rationale: `deque(maxlen=N)` automatically drops oldest items, which is exactly what a rolling window needs. Simpler than a ring buffer, more efficient than list slicing.

**What:** CNNWorker accumulates individual segment probabilities in a deque of size 4 (per D-01: 2s window / 0.5s segments). The aggregator combines them before feeding the state machine.

**Key design:** The segment buffer lives inside `CNNWorker._loop()`. Each inference pass:
1. Preprocess the audio segment (0.5s)
2. Run classifier.predict() to get a single probability
3. Append probability to deque(maxlen=4)
4. Run aggregator.aggregate(list(deque)) to get p_agg
5. Store p_agg as the ClassificationResult.drone_probability

This means the state machine sees the aggregated value -- no state machine code changes needed (D-08).

### Pattern 5: Overlapping Segment Generation from Pipeline

**Discretion decision: Adjust pipeline._process_cnn() to push audio to CNNWorker every 0.25s (hop) instead of every 0.5s.**

The current pipeline accumulates mono audio in `_mono_buffer` and pushes 0.5s segments. For 50% overlap (D-02), the pipeline needs to push every 0.25s instead. The CNNWorker receives 0.5s segments at a 0.25s rate, processes each independently, and the deque accumulates up to 4 probabilities covering a 2s window.

**Implementation approach:** Change `_cnn_interval` from `0.5` to `0.25` and keep `_cnn_segment_samples` at `int(sr * 0.5)`. Each push still sends the last 0.5s of audio, but pushes happen twice as often, creating the overlap.

### Pattern 6: Model Factory in main.py Lifespan

**What:** Load model at startup, inject into CNNWorker.

```python
# In main.py lifespan, after creating preprocessor:
classifier = None
if os.path.isfile(settings.cnn_model_path):
    try:
        model = ResearchCNN()
        model.load_state_dict(torch.load(settings.cnn_model_path, weights_only=True))
        model.eval()
        classifier = ResearchClassifier(model)
        logger.info("Loaded CNN model from %s", settings.cnn_model_path)
    except Exception:
        logger.exception("Failed to load CNN model — running without classifier")
else:
    logger.warning("CNN model not found at %s — running in dormant mode", settings.cnn_model_path)
```

**Discretion decision: Add a model validation step on load.** Run a dummy forward pass `model(torch.zeros(1, 1, 128, 64))` after loading to catch shape mismatches early. Cost is negligible (~1ms).

### Anti-Patterns to Avoid
- **Modifying state_machine.py:** Thresholds are already configurable. Do not add aggregation logic there.
- **Moving aggregation to pipeline.py:** Aggregation is a classification concern, not a beamforming concern. Keep it in CNNWorker.
- **Using BCEWithLogitsLoss at inference:** The model has `nn.Sigmoid()` as the last layer. Output is already a probability. Do not apply sigmoid again.
- **Forgetting model.eval():** BatchNorm and Dropout behave differently in train vs eval mode. The classifier wrapper must set eval mode.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rolling window buffer | Custom ring buffer class | `collections.deque(maxlen=4)` | Auto-evicts old items, thread-safe for single-writer/single-reader |
| Model serialization | Custom save/load | `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path, weights_only=True))` | Standard PyTorch pattern, `weights_only=True` is safer |
| Global average pooling | Manual mean over spatial dims | `nn.AdaptiveAvgPool2d(1)` | Handles any input spatial size, standard layer |

## Common Pitfalls

### Pitfall 1: TF vs PyTorch Convolution Dimension Order
**What goes wrong:** TF Conv2D expects NHWC `(batch, height, width, channels)`, PyTorch expects NCHW `(batch, channels, height, width)`. Getting this wrong produces shape errors or silent wrong results.
**Why it happens:** The research code uses TF format `(MAX_FRAMES, N_MELS, 1)`.
**How to avoid:** ResearchPreprocessor already outputs `(1, 1, 128, 64)` in PyTorch NCHW format. Verify the model's first Conv2d has `in_channels=1`. Test with a dummy tensor.
**Warning signs:** Shape mismatch errors at the first conv layer, or output probabilities always near 0.5.

### Pitfall 2: BatchNorm in Eval Mode
**What goes wrong:** BatchNorm uses running statistics in eval mode but batch statistics in train mode. If model.eval() is not called, inference results will vary with input statistics.
**Why it happens:** PyTorch models default to train mode after construction.
**How to avoid:** The `ResearchClassifier` wrapper calls `self._model.eval()` in `__init__`. Never call `model.train()` in the inference path.
**Warning signs:** Different probabilities for the same input across calls.

### Pitfall 3: Aggregation with Empty Buffer
**What goes wrong:** On the first few inference cycles, the deque has fewer than 4 probabilities. Aggregation must handle 1-3 probabilities gracefully.
**Why it happens:** Startup transient -- takes 4 pushes (1 second with 0.25s hop) to fill the buffer.
**How to avoid:** `WeightedAggregator.aggregate()` handles any non-empty list. The deque starts empty and grows naturally. First aggregation happens after the first inference.
**Warning signs:** Division by zero or index errors during the first second of operation.

### Pitfall 4: Thread Safety of Segment Deque
**What goes wrong:** If the deque is accessed from multiple threads without protection, race conditions occur.
**Why it happens:** `CNNWorker._loop()` runs in a background thread. `get_latest()` is called from the pipeline thread.
**How to avoid:** The deque is only written inside `_loop()` and the aggregated result is stored in `_latest` under the existing `_lock`. The deque itself is private to the worker thread -- no cross-thread access needed.
**Warning signs:** Intermittent incorrect probability values.

### Pitfall 5: `torch.load` Security Warning
**What goes wrong:** `torch.load` without `weights_only=True` can execute arbitrary Python code from the checkpoint file.
**Why it happens:** Default behavior uses pickle, which is unsafe for untrusted files.
**How to avoid:** Always use `torch.load(path, weights_only=True)` for state_dict loading.
**Warning signs:** PyTorch prints a FutureWarning about weights_only.

## Code Examples

### Config Extension (AcousticSettings)
```python
# Add to AcousticSettings class in config.py
cnn_agg_w_max: float = 0.5
cnn_agg_w_mean: float = 0.5
```
These map to `ACOUSTIC_CNN_AGG_W_MAX` and `ACOUSTIC_CNN_AGG_W_MEAN` env vars per the existing `env_prefix="ACOUSTIC_"` pattern.

### CNNWorker Segment Buffer Integration
```python
from collections import deque

class CNNWorker:
    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        classifier: Classifier | None = None,
        aggregator: Aggregator | None = None,  # NEW
        *,
        fs_in: int = 48000,
        silence_threshold: float = 0.001,
        segment_buffer_size: int = 4,  # D-01: 2s / 0.5s = 4 segments
    ) -> None:
        # ... existing init ...
        self._aggregator = aggregator
        self._segment_probs: deque[float] = deque(maxlen=segment_buffer_size)
```

In `_loop()`, after getting raw probability from classifier:
```python
raw_prob = self._classifier.predict(features)
self._segment_probs.append(raw_prob)

if self._aggregator is not None and self._segment_probs:
    prob = self._aggregator.aggregate(list(self._segment_probs))
else:
    prob = raw_prob
```

### Model Validation on Load
```python
# After loading state_dict
dummy = torch.zeros(1, 1, 128, 64)
with torch.no_grad():
    out = model(dummy)
assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
logger.info("Model validation passed (dummy forward pass OK)")
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x with pytest-asyncio |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/unit/ -x -q` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MDL-01 | ResearchCNN accepts (N,1,128,64), produces (N,1) | unit | `python -m pytest tests/unit/test_research_cnn.py -x` | Wave 0 |
| MDL-01 | ResearchCNN layer count and sizes match spec | unit | `python -m pytest tests/unit/test_research_cnn.py::test_architecture_matches_spec -x` | Wave 0 |
| MDL-02 | WeightedAggregator computes correct p_agg | unit | `python -m pytest tests/unit/test_aggregation.py -x` | Wave 0 |
| MDL-02 | Segment buffer fills and rolls over correctly | unit | `python -m pytest tests/unit/test_worker.py::TestSegmentBuffer -x` | Wave 0 |
| MDL-03 | Aggregator protocol runtime check | unit | `python -m pytest tests/unit/test_protocols.py::TestAggregatorProtocol -x` | Wave 0 |
| MDL-03 | CNNWorker accepts all three protocols | unit | `python -m pytest tests/unit/test_worker.py::TestCNNWorkerConstructor -x` | Wave 0 |
| MDL-03 | Factory loads model and injects into worker | integration | `python -m pytest tests/integration/test_cnn_pipeline.py -x` | Wave 0 |
| MDL-04 | State machine thresholds configurable via env vars | unit | `python -m pytest tests/unit/test_config.py -x` | existing (extend) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_research_cnn.py` -- covers MDL-01 (model shape, architecture)
- [ ] `tests/unit/test_aggregation.py` -- covers MDL-02 (WeightedAggregator, edge cases)
- [ ] Extend `tests/unit/test_protocols.py` with `TestAggregatorProtocol` -- covers MDL-03
- [ ] Extend `tests/unit/test_worker.py` with `TestSegmentBuffer` -- covers MDL-02/MDL-03
- [ ] Extend `tests/unit/test_config.py` with aggregation weight fields -- covers MDL-04

## Open Questions

1. **Overlap generation responsibility: pipeline.py vs CNNWorker?**
   - What we know: Currently pipeline.py pushes 0.5s segments to CNNWorker. For 50% overlap, pushes need to happen every 0.25s.
   - What's unclear: Whether to change the push interval in pipeline.py (simpler) or have CNNWorker internally subdivide incoming audio (more encapsulated).
   - Recommendation: Change `_cnn_interval` in pipeline.py from 0.5 to 0.25. This is the simpler approach -- pipeline already manages push timing. CNNWorker processes whatever it receives.

2. **Model file path for dormant mode testing**
   - What we know: No trained model exists yet (Phase 8). Service must boot and run without crashing.
   - What's unclear: Whether tests should create a dummy model checkpoint or only test the dormant path.
   - Recommendation: Tests should cover both paths -- (a) dormant with classifier=None, (b) active with a freshly initialized (untrained) ResearchCNN. Create a test fixture that saves an untrained model's state_dict to a temp file.

## Sources

### Primary (HIGH confidence)
- `train_strong_cnn.py` lines 208-235 -- canonical model architecture (TF/Keras)
- `run_strong_inference.py` lines 29-47, 117-125 -- aggregation reference (2s segments, 0.7/0.3 weighting)
- `src/acoustic/classification/protocols.py` -- existing Classifier/Preprocessor protocols
- `src/acoustic/classification/worker.py` -- current CNNWorker implementation
- `src/acoustic/classification/preprocessing.py` -- ResearchPreprocessor (output shape verified)
- `src/acoustic/config.py` -- AcousticSettings with env_prefix pattern
- `src/acoustic/main.py` -- lifespan factory with dormant classifier slot
- `src/acoustic/pipeline.py` -- _process_cnn() segment push logic
- PyTorch 2.11.0 installed and verified locally

### Secondary (MEDIUM confidence)
- PyTorch nn.Module documentation for Conv2d, BatchNorm2d, AdaptiveAvgPool2d layer equivalences

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and in use, no new dependencies
- Architecture: HIGH -- extending well-established protocol pattern, TF->PyTorch port is mechanical
- Pitfalls: HIGH -- common PyTorch gotchas (eval mode, dimension order) are well-known

**Research date:** 2026-04-01
**Valid until:** 2026-05-01 (stable -- no fast-moving dependencies)
