# Phase 14: EfficientAT Model Architecture with AudioSet Transfer Learning - Research

**Researched:** 2026-04-04
**Domain:** Audio classification / Transfer learning / EfficientAT MobileNetV3
**Confidence:** HIGH

## Summary

Phase 14 replaces the custom 3-layer ResearchCNN (~46K params) with EfficientAT's MobileNetV3 mn10 (~4.88M params) pretrained on AudioSet-527. The EfficientAT repository (fschmid56/EfficientAT) provides pretrained weights as GitHub release assets, and the model architecture source code can be vendored directly into the project. The key challenge is that EfficientAT uses different preprocessing parameters (32kHz sample rate, 128 mel bands, window_size=800, hop_size=320) compared to the project's current MelConfig (16kHz, 64 mel bands, n_fft=1024, hop_length=256). The model needs its own preprocessor or the existing one must be adapted.

The existing codebase already has the right abstractions: the `Classifier` protocol, the model registry in `ensemble.py`, and the classifier factory in `main.py`. The EfficientAT model needs a wrapper class implementing `predict(features) -> float` and a registered loader function. The three-stage unfreezing transfer learning recipe is not built into EfficientAT -- it must be implemented in the training runner.

**Primary recommendation:** Vendor the EfficientAT MN model source code (model.py + inverted_residual.py + preprocess.py) into `src/acoustic/classification/efficientat/`, adapt the classifier head from 527 to 1 class (binary sigmoid), implement three-stage unfreezing in a new `EfficientATTrainingRunner`, and register `"efficientat_mn10"` in the model registry.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MDL-10 | EfficientAT mn10 model loads with AudioSet-pretrained weights and implements the Classifier protocol | Vendored model code + `EfficientATClassifier` wrapper satisfying `Classifier.predict()`, pretrained weights from GitHub releases (mn10_as_mAP_471.pt, 18MB) |
| MDL-11 | Three-stage transfer learning: Stage 1 (head only, lr=1e-3), Stage 2 (last 2-3 blocks, lr=1e-4), Stage 3 (all layers, lr=1e-5) with cosine annealing | New `EfficientATTrainingRunner` with parameter group freezing, `CosineAnnealingLR` scheduler, stage transitions based on epoch ranges |
| MDL-12 | Model swappable at startup via config (classifier factory) | Extend existing `register_model()` pattern in `ensemble.py` with `"efficientat_mn10"` type, add `ACOUSTIC_CNN_MODEL_TYPE` config field |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| EfficientAT (vendored) | v0.0.1 | MobileNetV3 mn10 model architecture | Official AudioSet-pretrained audio CNN from fschmid56/EfficientAT. Vendored (not pip-installed) because it is not a PyPI package |
| PyTorch | >=2.11,<2.12 | Training + inference | Already in project requirements.txt |
| torchaudio | >=2.11,<2.12 | Mel spectrogram transforms | Already in project requirements.txt |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.optim.lr_scheduler.CosineAnnealingLR | (stdlib) | Cosine annealing schedule | Stage 2 and Stage 3 of transfer learning |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Vendoring EfficientAT | pip install from git | Not a proper package, no setup.py/pyproject.toml -- would need fork maintenance |
| mn10 (4.88M params) | mn04 (1.85M) | Smaller but significantly lower mAP (43.9 vs 47.1); mn10 is the sweet spot |
| mn10 (4.88M params) | dymn10 (10.57M) | Better mAP (47.7) but 2x params, dynamic computation adds complexity |
| Vendoring full repo | Cherry-pick model files only | Full repo has dataset loaders and training scripts we don't need; just vendor model + preprocess |

**No new pip packages needed.** EfficientAT model code is vendored as Python source files.

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/classification/
  efficientat/
    __init__.py              # Exports get_model, AugmentMelSTFT
    model.py                 # Vendored MN class from EfficientAT (adapted)
    inverted_residual.py     # Vendored InvertedResidual blocks
    preprocess.py            # Vendored AugmentMelSTFT
    classifier.py            # EfficientATClassifier (Classifier protocol wrapper)
    config.py                # EfficientATMelConfig (128 mels, 32kHz params)
src/acoustic/training/
    efficientat_trainer.py   # Three-stage unfreezing TrainingRunner
```

### Pattern 1: EfficientAT Model Loading with Head Replacement
**What:** Load mn10 pretrained on AudioSet-527, replace 527-class head with binary (1 output + sigmoid)
**When to use:** Initial model setup and checkpoint loading
**Example:**
```python
# Source: EfficientAT model.py get_model()
from acoustic.classification.efficientat.model import get_model

# Load pretrained mn10 with 527 AudioSet classes
model = get_model(
    num_classes=527,
    pretrained_name="mn10_as",
    width_mult=1.0,
    head_type="mlp",
    input_dim_f=128,   # 128 mel bands
    input_dim_t=1000,  # ~10s at hop=320/sr=32000
)

# Replace classifier head for binary drone detection
# The MLP head structure: Linear(960, 1280) -> Hardswish -> Dropout -> Linear(1280, num_classes)
# Replace only the final linear layer
model.classifier[-1] = nn.Linear(1280, 1)
# Add sigmoid for binary output
```

### Pattern 2: Three-Stage Unfreezing Transfer Learning
**What:** Progressive unfreezing to preserve pretrained features while adapting to drone detection
**When to use:** Fine-tuning on DADS dataset
**Example:**
```python
# Stage 1: Freeze everything except classifier head
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
optimizer = Adam(model.classifier.parameters(), lr=1e-3)

# Stage 2: Unfreeze last 2-3 inverted residual blocks
# model.features is nn.Sequential of InvertedResidual blocks
for block in model.features[-3:]:
    for param in block.parameters():
        param.requires_grad = True
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)

# Stage 3: Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
optimizer = Adam(model.parameters(), lr=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=stage3_epochs)
```

### Pattern 3: Classifier Protocol Wrapper
**What:** Wrap EfficientAT model to satisfy the existing `Classifier` protocol
**When to use:** Integration with CNNWorker and ensemble system
**Example:**
```python
class EfficientATClassifier:
    """Wraps EfficientAT MN model to satisfy Classifier protocol."""

    def __init__(self, model: nn.Module, mel_config: EfficientATMelConfig) -> None:
        self._model = model
        self._model.eval()
        self._mel = AugmentMelSTFT(
            n_mels=mel_config.n_mels,
            sr=mel_config.sample_rate,
            win_length=mel_config.win_length,
            hopsize=mel_config.hop_size,
        )
        self._mel.eval()  # No augmentation at inference

    def predict(self, features: torch.Tensor) -> float:
        """Run inference. Input is raw audio or pre-computed mel features."""
        with torch.no_grad():
            logits, _ = self._model(features)
            prob = torch.sigmoid(logits).item()
        return prob
```

### Pattern 4: Model Registry Integration
**What:** Register `"efficientat_mn10"` type in the existing model registry
**When to use:** Ensemble config and classifier factory
**Example:**
```python
# In src/acoustic/classification/efficientat/__init__.py
from acoustic.classification.ensemble import register_model

def _load_efficientat_mn10(path: str) -> Classifier:
    from acoustic.classification.efficientat.model import get_model
    model = get_model(num_classes=1, width_mult=1.0, head_type="mlp")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return EfficientATClassifier(model)

register_model("efficientat_mn10", _load_efficientat_mn10)
```

### Anti-Patterns to Avoid
- **Using EfficientAT's 32kHz preprocessing for training but project's 16kHz for inference:** The model MUST use the same preprocessing at train and inference time. Either resample DADS audio to 32kHz or train with a custom mel config. Since DADS is 16kHz, resampling to 32kHz is recommended to match EfficientAT's pretrained expectations.
- **Unfreezing all layers from the start:** Destroys pretrained AudioSet features. The three-stage recipe exists specifically to prevent this.
- **Keeping the 527-class sigmoid output:** EfficientAT uses multi-label sigmoid for AudioSet. For binary drone detection, replace the head with a single output + sigmoid.
- **Vendoring the entire EfficientAT repo:** Only vendor the model files (model.py, inverted_residual.py, preprocess.py). The training scripts, dataset loaders, and other utilities are project-specific to EfficientAT and not needed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MobileNetV3 architecture | Custom MobileNetV3 | Vendor EfficientAT's MN class | Inverted residuals, squeeze-excitation, head types already implemented and tested |
| Mel spectrogram with EfficientAT params | Custom torchaudio transform matching EfficientAT | Vendor AugmentMelSTFT | Includes preemphasis, (melspec+4.5)/5 normalization matching pretrained weights |
| AudioSet pretrained weights | Train from scratch | Download mn10_as_mAP_471.pt (18MB) from GitHub releases | AudioSet-527 pretraining provides rich audio features; training from scratch on DADS alone would be insufficient |
| Cosine annealing schedule | Manual LR decay | torch.optim.lr_scheduler.CosineAnnealingLR | Built-in, well-tested, standard for transfer learning |

**Key insight:** The EfficientAT preprocessing (preemphasis + mel + log + (x+4.5)/5 normalization) is tightly coupled to the pretrained weights. Using different preprocessing will degrade transfer learning performance. Vendor the entire preprocessing pipeline.

## Common Pitfalls

### Pitfall 1: Preprocessing Mismatch
**What goes wrong:** Model trained at 32kHz/128 mels receives 16kHz/64 mel input at inference time, producing random predictions
**Why it happens:** Project currently uses MelConfig(16kHz, 64 mels) for ResearchCNN. EfficientAT expects different params.
**How to avoid:** Create a separate `EfficientATPreprocessor` that uses `AugmentMelSTFT` with the correct params (32kHz, 128 mels, hop=320, win=800). The `Preprocessor` protocol already supports this -- each model type can have its own preprocessor.
**Warning signs:** High training accuracy but poor inference accuracy; prediction probabilities clustered around 0.5

### Pitfall 2: Forgetting to Resample Audio
**What goes wrong:** DADS audio is 16kHz mono. EfficientAT expects 32kHz. Feeding 16kHz audio through 32kHz mel transform produces spectrograms at wrong time/frequency resolution.
**Why it happens:** EfficientAT was trained on AudioSet resampled to 32kHz.
**How to avoid:** Use `torchaudio.functional.resample(waveform, 16000, 32000)` before mel computation. This should be part of the preprocessor, not a manual step.
**Warning signs:** Mel spectrograms look compressed/stretched compared to reference

### Pitfall 3: float16 vs float32 Weight Loading
**What goes wrong:** EfficientAT pretrained weights may be saved in float16. Loading on CPU requires float32 conversion.
**Why it happens:** EfficientAT trains with half-precision for speed.
**How to avoid:** After loading state_dict, call `model.float()` to ensure float32. The README notes minor performance degradation with float32 on CPU -- this is acceptable for our use case.
**Warning signs:** RuntimeError about dtype mismatch, or NaN outputs

### Pitfall 4: Wrong Output Interpretation
**What goes wrong:** EfficientAT outputs raw logits (no activation). Applying sigmoid twice (if the head already has sigmoid) or not at all gives wrong probabilities.
**Why it happens:** AudioSet models use `torch.sigmoid(logits)` post-hoc for multi-label classification. The original architecture has no sigmoid in the forward pass.
**How to avoid:** Ensure the binary head outputs a single logit, apply sigmoid externally in the `predict()` method. Do NOT add sigmoid inside the model if using BCEWithLogitsLoss for training.
**Warning signs:** Probabilities always > 0.5 or always < 0.5; loss not converging

### Pitfall 5: Stage Transition Timing
**What goes wrong:** Training too many epochs in Stage 1 (head only) overfits the head to the frozen features, making Stage 2 unstable.
**Why it happens:** The classifier head converges fast (~5-10 epochs) since it only has ~1280 params to tune.
**How to avoid:** Use early stopping per stage, or fixed epoch counts: Stage 1 = 5-10 epochs, Stage 2 = 10-15 epochs, Stage 3 = 10-20 epochs. Monitor validation loss at each transition.
**Warning signs:** Validation loss spike when transitioning between stages

### Pitfall 6: Input Dimension Mismatch
**What goes wrong:** The mn10 model expects `input_dim_f=128` (mel bands) and `input_dim_t=1000` (time frames for 10s audio). DADS clips are 0.5-1s, producing far fewer time frames.
**Why it happens:** AudioSet clips are 10 seconds. DADS segments are shorter.
**How to avoid:** EfficientAT uses AdaptiveAvgPool2d, so variable-length input works. However, set `input_dim_t` to match actual DADS segment length during model construction for optimal performance. For 1s audio at 32kHz with hop=320: `1 * 32000 / 320 = 100` time frames.
**Warning signs:** Model works but accuracy is lower than expected; internal feature map sizes are suboptimal

## Code Examples

### Loading Pretrained mn10 and Adapting for Binary Classification
```python
# Source: Adapted from EfficientAT inference.py and model.py
import torch
import torch.nn as nn
from acoustic.classification.efficientat.model import get_model

# Step 1: Load AudioSet-pretrained mn10 (527 classes)
model = get_model(
    num_classes=527,
    pretrained_name="mn10_as",
    width_mult=1.0,
    head_type="mlp",
    input_dim_f=128,
    input_dim_t=100,  # Adjusted for ~1s DADS segments
)

# Step 2: Replace classification head for binary output
# MLP head: features(960) -> Linear(960,1280) -> Hardswish -> Dropout -> Linear(1280,527)
# Keep the feature expansion, replace only final layer
in_features = model.classifier[-1].in_features  # 1280
model.classifier[-1] = nn.Linear(in_features, 1)

# Step 3: Ensure float32 for CPU compatibility
model.float()
```

### EfficientAT Mel Preprocessing
```python
# Source: EfficientAT models/preprocess.py
from acoustic.classification.efficientat.preprocess import AugmentMelSTFT

mel = AugmentMelSTFT(
    n_mels=128,
    sr=32000,
    win_length=800,
    hopsize=320,
    n_fft=1024,
    freqm=0,     # No augmentation at inference
    timem=0,
)
mel.eval()

# Input: waveform tensor (batch, samples) at 32kHz
# Output: mel spectrogram (batch, 1, n_mels, time_frames)
# Normalization: (log_mel + 4.5) / 5.0
```

### Three-Stage Training Loop Structure
```python
# Source: Project-specific implementation based on transfer learning best practices
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_efficientat(model, train_loader, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()  # Raw logits, no sigmoid in model

    # --- Stage 1: Head only ---
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
    for epoch in range(10):
        train_epoch(model, train_loader, criterion, opt1, device)

    # --- Stage 2: Head + last 3 blocks ---
    for block in model.features[-3:]:
        for p in block.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt2 = optim.Adam(trainable, lr=1e-4)
    sched2 = CosineAnnealingLR(opt2, T_max=15)
    for epoch in range(15):
        train_epoch(model, train_loader, criterion, opt2, device)
        sched2.step()

    # --- Stage 3: All layers ---
    for p in model.parameters():
        p.requires_grad = True
    opt3 = optim.Adam(model.parameters(), lr=1e-5)
    sched3 = CosineAnnealingLR(opt3, T_max=20)
    for epoch in range(20):
        train_epoch(model, train_loader, criterion, opt3, device)
        sched3.step()
```

### Model Registry Registration
```python
# Source: Existing pattern from src/acoustic/classification/ensemble.py
from acoustic.classification.ensemble import register_model
from acoustic.classification.protocols import Classifier

def _load_efficientat_mn10(path: str) -> Classifier:
    """Load fine-tuned EfficientAT mn10 checkpoint."""
    from acoustic.classification.efficientat.classifier import EfficientATClassifier
    from acoustic.classification.efficientat.model import get_model

    model = get_model(num_classes=1, width_mult=1.0, head_type="mlp",
                      input_dim_f=128, input_dim_t=100)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return EfficientATClassifier(model)

register_model("efficientat_mn10", _load_efficientat_mn10)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom 3-layer CNN (46K params) | EfficientAT MobileNetV3 mn10 (4.88M params) | This phase | 100x more parameters but pretrained on AudioSet-527 classes; expected >95% accuracy vs ~90% baseline |
| Train from scratch on field data | Transfer learning from AudioSet | This phase | AudioSet pretraining provides general audio features; much less labeled drone data needed |
| Single learning rate training | Three-stage unfreezing with cosine annealing | This phase | Preserves pretrained features while adapting to domain-specific task |
| ResearchPreprocessor (16kHz, 64 mels) | AugmentMelSTFT (32kHz, 128 mels) | This phase | Matches EfficientAT's pretrained preprocessing expectations |

## Open Questions

1. **DADS segment duration for EfficientAT**
   - What we know: DADS has variable-length clips. EfficientAT was trained on 10s AudioSet clips. The model uses AdaptiveAvgPool2d so variable lengths work.
   - What's unclear: Optimal segment duration for drone detection with mn10. Shorter segments (0.5-1s) give faster real-time response but fewer time frames.
   - Recommendation: Start with 1s segments (100 time frames at 32kHz/hop=320). This matches the existing project's segment_seconds=0.5 scaled for 32kHz. Validate accuracy against 2s segments if results are below target.

2. **Pretrained weight download mechanism**
   - What we know: Weights are at `https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt` (18MB)
   - What's unclear: Should weights be downloaded at build time (Docker), first run, or committed to repo?
   - Recommendation: Download script in `scripts/download_pretrained.py` that fetches to `models/pretrained/mn10_as.pt`. Add to Docker build. Do NOT commit 18MB binary to git.

3. **Preprocessor integration with CNNWorker**
   - What we know: CNNWorker accepts a `Preprocessor` protocol object. Current preprocessor produces (1,1,128,64) tensors.
   - What's unclear: EfficientAT expects (batch, samples) raw waveforms through AugmentMelSTFT, not pre-computed mel spectrograms. The `predict()` method needs to handle this differently.
   - Recommendation: The `EfficientATClassifier.predict()` should accept the same `features: torch.Tensor` input but include its own preprocessing internally. Alternatively, create an `EfficientATPreprocessor` implementing the `Preprocessor` protocol that outputs the EfficientAT-compatible mel format.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 with pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/unit/test_efficientat.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MDL-10 | mn10 model loads pretrained weights, output shape is (N,1) | unit | `python -m pytest tests/unit/test_efficientat.py::test_model_loads_pretrained -x` | Wave 0 |
| MDL-10 | EfficientATClassifier satisfies Classifier protocol | unit | `python -m pytest tests/unit/test_efficientat.py::test_classifier_protocol -x` | Wave 0 |
| MDL-10 | Model param count ~4.88M | unit | `python -m pytest tests/unit/test_efficientat.py::test_param_count -x` | Wave 0 |
| MDL-11 | Stage 1 freezes all except head | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage1_freeze -x` | Wave 0 |
| MDL-11 | Stage 2 unfreezes last N blocks | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage2_unfreeze -x` | Wave 0 |
| MDL-11 | Stage 3 unfreezes all layers | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_stage3_unfreeze -x` | Wave 0 |
| MDL-11 | Cosine annealing LR schedule applied | unit | `python -m pytest tests/unit/test_efficientat_training.py::test_cosine_schedule -x` | Wave 0 |
| MDL-12 | Registry loads "efficientat_mn10" type | unit | `python -m pytest tests/unit/test_efficientat.py::test_registry_load -x` | Wave 0 |
| MDL-12 | Config selects model type at startup | unit | `python -m pytest tests/unit/test_config.py::test_model_type_config -x` | Extend existing |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/test_efficientat.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_efficientat.py` -- covers MDL-10, MDL-12 (model loading, protocol, registry)
- [ ] `tests/unit/test_efficientat_training.py` -- covers MDL-11 (three-stage freezing, cosine schedule)
- [ ] Extend `tests/unit/test_config.py` -- covers MDL-12 (model type configuration)

## Sources

### Primary (HIGH confidence)
- [fschmid56/EfficientAT GitHub](https://github.com/fschmid56/EfficientAT) - Model architecture, pretrained weights, fine-tuning examples
- [EfficientAT Releases v0.0.1](https://github.com/fschmid56/EfficientAT/releases) - mn10_as_mAP_471.pt (18MB, 4.88M params, 47.1 mAP on AudioSet)
- Codebase analysis: `src/acoustic/classification/protocols.py`, `ensemble.py`, `research_cnn.py`, `preprocessing.py`, `config.py`

### Secondary (MEDIUM confidence)
- [EfficientAT model.py](https://github.com/fschmid56/EfficientAT/blob/main/models/mn/model.py) - MN class, get_model(), head types, pretrained loading
- [EfficientAT inference.py](https://github.com/fschmid56/EfficientAT/blob/main/inference.py) - AugmentMelSTFT usage, preprocessing params (32kHz, 128 mels, hop=320, win=800)
- [EfficientAT preprocess.py](https://github.com/fschmid56/EfficientAT/blob/main/models/preprocess.py) - AugmentMelSTFT class: preemphasis + mel + log + (x+4.5)/5 normalization

### Tertiary (LOW confidence)
- Three-stage unfreezing recipe: Based on general transfer learning best practices, not an EfficientAT-specific published recipe. The EfficientAT fine-tuning examples (ex_esc50.py, ex_dcase20.py) train all parameters with a single learning rate. The three-stage approach is the roadmap's design choice, validated by transfer learning literature but not by EfficientAT specifically.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - EfficientAT is well-documented, weights verified at 18MB, architecture inspected from source
- Architecture: HIGH - Existing codebase abstractions (Classifier protocol, model registry) are well-suited for integration
- Pitfalls: HIGH - Preprocessing mismatch is the #1 risk, identified from source code comparison
- Transfer learning recipe: MEDIUM - Three-stage unfreezing is standard practice but EfficientAT's own examples use simpler fine-tuning. The roadmap specifies this recipe, so we implement it.

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (EfficientAT repo is stable, last release v0.0.1)
