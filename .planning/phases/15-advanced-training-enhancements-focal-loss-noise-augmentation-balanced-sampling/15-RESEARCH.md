# Phase 15: Advanced Training Enhancements - Focal Loss, Noise Augmentation, Balanced Sampling - Research

**Researched:** 2026-04-04
**Domain:** Audio ML training enhancements / Loss functions / Data augmentation / Sampling strategies
**Confidence:** HIGH

## Summary

Phase 15 adds four training enhancements to the existing training pipeline: (1) focal loss replacing BCE for better handling of hard examples, (2) background noise augmentation by mixing drone audio with ESC-50/UrbanSound8K environmental sounds at variable SNR, (3) class-balanced sampling targeting 50/50 drone/no-drone ratio per batch, and (4) waveform augmentations (pitch shift, time stretch, gain) via the audiomentations library. The goal is to achieve <5% false positive rate with >95% recall on the DADS test set.

The existing codebase already has a `build_weighted_sampler()` function in `dataset.py` that does inverse-frequency weighted sampling -- this is already used in `trainer.py`. The existing `WaveformAugmentation` class handles Gaussian noise + gain. Phase 15 replaces/extends these with more sophisticated versions. The critical new dependency is `audiomentations>=0.43.1` for pitch shift and time stretch (these require librosa-quality implementations that should not be hand-rolled). For focal loss, `torchvision.ops.sigmoid_focal_loss` is already available (torchvision 0.26.0 is installed) -- no new dependency needed. ESC-50 and UrbanSound8K are external datasets that need download scripts and configurable paths.

**Primary recommendation:** Implement focal loss using `torchvision.ops.sigmoid_focal_loss` (already installed), add `audiomentations>=0.43.1` for pitch/time-stretch/gain augmentation, create a `BackgroundNoiseMixer` that loads ESC-50/UrbanSound8K WAV files and mixes them at random SNR, and extend `TrainingConfig` with all new hyperparameters. The existing `build_weighted_sampler()` already handles balanced sampling -- verify it works correctly with DADS class distribution.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRN-10 | Focal Loss (gamma=2.0, alpha=0.25) replaces BCE, with weighted BCE fallback | `torchvision.ops.sigmoid_focal_loss` available in installed torchvision 0.26.0; wrap in `FocalLoss` nn.Module for drop-in replacement of `nn.BCELoss` |
| TRN-11 | Background noise augmentation with ESC-50/UrbanSound8K at SNR -10 to +20 dB | `BackgroundNoiseMixer` class loads noise WAV files from configurable directory, mixes at random SNR; audiomentations `AddBackgroundNoise` is an alternative but custom is simpler for our pipeline |
| TRN-12 | Waveform augmentations (pitch shift +/-3 semitones, time stretch 0.85-1.15x, gain -6 to +6 dB) via audiomentations | `audiomentations.Compose([PitchShift, TimeStretch, Gain])` with configurable probabilities; replaces existing `WaveformAugmentation` class |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchvision | 0.26.0 (installed) | `sigmoid_focal_loss` | Official PyTorch ecosystem; already installed; no new dependency |
| audiomentations | 0.43.1 | Pitch shift, time stretch, gain augmentation | De facto standard for audio augmentation in Python; Kaggle competition winner; handles edge cases in pitch/time manipulation |
| PyTorch | >=2.11,<2.12 | Training framework | Already in requirements.txt |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| soundfile | >=0.13 (installed) | Load ESC-50/UrbanSound8K WAV files | Noise augmentation dataset loading |
| numpy | >=1.26 (installed) | SNR mixing arithmetic | Background noise mixing at specified dB levels |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchvision sigmoid_focal_loss | Custom FocalLoss implementation | torchvision is official, tested, maintained; custom risks numerical bugs |
| audiomentations | torch-audiomentations (GPU) | torch-audiomentations runs on GPU but adds complexity; CPU augmentation in DataLoader workers is fast enough for this dataset size |
| audiomentations | Custom numpy pitch/time-stretch | Pitch shift and time stretch are numerically complex (phase vocoder); hand-rolling risks artifacts |
| Custom BackgroundNoiseMixer | audiomentations AddBackgroundNoise | AddBackgroundNoise requires a directory of noise files and handles it internally; custom mixer gives more control over SNR range and is simpler to integrate with existing pipeline |

**Installation:**
```bash
pip install audiomentations>=0.43.1
```

Add to `requirements.txt`:
```
audiomentations>=0.43,<1.0
```

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/training/
  augmentation.py          # Extended: BackgroundNoiseMixer, AudiomentationsAugmentation (replaces WaveformAugmentation)
  losses.py                # NEW: FocalLoss wrapper around torchvision.ops.sigmoid_focal_loss
  config.py                # Extended: focal loss params, noise augmentation paths, audiomentations params
  trainer.py               # Modified: use FocalLoss, pass noise mixer to dataset
  dataset.py               # Modified: accept BackgroundNoiseMixer in augmentation pipeline
  parquet_dataset.py       # Modified: accept BackgroundNoiseMixer in augmentation pipeline
```

### Pattern 1: FocalLoss as nn.Module Wrapper
**What:** Wrap `torchvision.ops.sigmoid_focal_loss` in an `nn.Module` so it's a drop-in replacement for `nn.BCELoss` in the training loop
**When to use:** Default loss function for all training runs
**Example:**
```python
# Source: torchvision.ops.sigmoid_focal_loss docs
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Binary focal loss wrapping torchvision.ops.sigmoid_focal_loss.

    Expects raw logits (no sigmoid applied) as inputs.
    For binary classification with a single output neuron.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs, targets,
            alpha=self.alpha, gamma=self.gamma,
            reduction="mean",
        )
```

**IMPORTANT:** `sigmoid_focal_loss` expects **raw logits** (before sigmoid), not probabilities. This means the model's final layer must NOT have sigmoid when using focal loss. The current `ResearchCNN` has sigmoid in its forward pass -- this must be made configurable or a separate head used. Phase 14's EfficientAT already outputs raw logits, so focal loss works directly there.

### Pattern 2: Background Noise Mixer
**What:** Load environmental noise files from ESC-50/UrbanSound8K, mix with drone audio at random SNR
**When to use:** Waveform-level augmentation during training
**Example:**
```python
import numpy as np
import soundfile as sf
from pathlib import Path


class BackgroundNoiseMixer:
    """Mix background noise from ESC-50/UrbanSound8K at random SNR."""

    def __init__(
        self,
        noise_dirs: list[Path],
        snr_range: tuple[float, float] = (-10.0, 20.0),
        sample_rate: int = 16000,
        p: float = 0.5,
    ) -> None:
        self._noise_files: list[Path] = []
        for d in noise_dirs:
            if d.is_dir():
                self._noise_files.extend(sorted(d.rglob("*.wav")))
        self._snr_range = snr_range
        self._sr = sample_rate
        self._p = p
        self._rng = np.random.default_rng()
        # Pre-load noise into memory for speed
        self._noise_cache: list[np.ndarray] = []

    def warm_cache(self) -> None:
        """Pre-load all noise files into memory."""
        for f in self._noise_files:
            audio, sr = sf.read(f, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample if needed (simple case: ESC-50 is 44.1kHz)
            if sr != self._sr:
                import torchaudio.functional as F
                import torch
                t = torch.from_numpy(audio).unsqueeze(0)
                t = F.resample(t, sr, self._sr)
                audio = t.squeeze(0).numpy()
            self._noise_cache.append(audio)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self._rng.random() > self._p or not self._noise_cache:
            return audio

        # Pick random noise clip
        noise = self._noise_cache[self._rng.integers(len(self._noise_cache))]

        # Extract segment matching audio length
        n = len(audio)
        if len(noise) >= n:
            start = self._rng.integers(0, len(noise) - n + 1)
            noise_seg = noise[start:start + n]
        else:
            noise_seg = np.zeros(n, dtype=np.float32)
            noise_seg[:len(noise)] = noise

        # Mix at random SNR
        snr_db = self._rng.uniform(*self._snr_range)
        sig_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_seg ** 2)
        if noise_power > 1e-10 and sig_power > 1e-10:
            scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
            return (audio + scale * noise_seg).astype(np.float32)
        return audio
```

### Pattern 3: Audiomentations Compose for Waveform Augmentation
**What:** Replace custom `WaveformAugmentation` with `audiomentations.Compose`
**When to use:** Pitch shift, time stretch, gain during training
**Example:**
```python
from audiomentations import Compose, PitchShift, TimeStretch, Gain


def build_waveform_augmentation(
    pitch_semitones: float = 3.0,
    time_stretch_range: tuple[float, float] = (0.85, 1.15),
    gain_db: float = 6.0,
    p: float = 0.5,
    sample_rate: int = 16000,
) -> Compose:
    return Compose([
        PitchShift(
            min_semitones=-pitch_semitones,
            max_semitones=pitch_semitones,
            p=p,
        ),
        TimeStretch(
            min_rate=time_stretch_range[0],
            max_rate=time_stretch_range[1],
            p=p,
        ),
        Gain(
            min_gain_db=-gain_db,
            max_gain_db=gain_db,
            p=p,
        ),
    ])


# Usage in dataset __getitem__:
# augmented = augment(samples=segment, sample_rate=16000)
```

### Pattern 4: Loss Function Selection in TrainingRunner
**What:** Make loss function configurable with focal loss as default, BCE as fallback
**When to use:** TrainingRunner.run() setup
**Example:**
```python
from acoustic.training.losses import FocalLoss

# In TrainingRunner.run():
if cfg.loss_function == "focal":
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    # Model must output logits (no sigmoid)
elif cfg.loss_function == "bce_weighted":
    # Weighted BCE for class imbalance
    pos_weight = torch.tensor([cfg.bce_pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
else:
    criterion = nn.BCEWithLogitsLoss()
```

### Anti-Patterns to Avoid
- **Applying sigmoid before focal loss:** `sigmoid_focal_loss` applies sigmoid internally. Double-sigmoid produces wrong gradients. The model must output raw logits.
- **Mixing augmented noise at extreme SNR without clipping:** SNR of -10 dB means noise is 10x louder than signal. Ensure output stays in [-1, 1] range or normalize after mixing.
- **Loading noise files on every __getitem__ call:** ESC-50 has 2000 files (5s each at 44.1kHz). Pre-load and cache them in memory (~800 MB for ESC-50 + UrbanSound8K).
- **Time-stretching after mel spectrogram:** Time stretch must happen on raw waveform, not on spectrogram. The audiomentations library handles this correctly.
- **Using both WeightedRandomSampler and focal loss alpha without coordination:** Both address class imbalance. Using aggressive alpha (0.25 for positive class) with balanced sampling can over-correct. Start with alpha=0.25 + balanced sampling, adjust if recall drops.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Focal loss | Custom focal loss from scratch | `torchvision.ops.sigmoid_focal_loss` | Numerical stability, gradient correctness, official PyTorch |
| Pitch shifting | Custom phase vocoder | `audiomentations.PitchShift` | Phase vocoder has subtle artifacts; audiomentations handles edge cases |
| Time stretching | Custom WSOLA/phase vocoder | `audiomentations.TimeStretch` | Same as pitch shift; well-tested implementation |
| SNR mixing formula | Ad-hoc noise addition | Standard SNR formula: `scale = sqrt(P_signal / (P_noise * 10^(SNR/10)))` | Getting SNR right matters for reproducibility |

**Key insight:** The existing `WaveformAugmentation` class only does Gaussian noise + gain. Phase 15 needs pitch shift and time stretch, which are substantially more complex algorithms. Using audiomentations is the right call -- it is the standard library for this exact purpose.

## Common Pitfalls

### Pitfall 1: Sigmoid/Logits Mismatch with Focal Loss
**What goes wrong:** Model applies sigmoid in forward(), then focal loss applies sigmoid again, producing near-zero gradients and the model never trains.
**Why it happens:** The current `ResearchCNN.forward()` ends with `torch.sigmoid(x)`. `sigmoid_focal_loss` expects raw logits.
**How to avoid:** When using focal loss, the model must output logits. Add a `return_logits: bool` parameter to the model or use `nn.BCEWithLogitsLoss` (which also expects logits). For Phase 15 building on Phase 14's EfficientAT, the EfficientAT model already outputs logits -- this is correct.
**Warning signs:** Loss is very small from epoch 1 but accuracy is ~50% (random); gradients near zero.

### Pitfall 2: ESC-50 Sample Rate Mismatch
**What goes wrong:** ESC-50 files are 44.1kHz but the training pipeline expects 16kHz (or 32kHz for EfficientAT). Mixing without resampling produces spectral artifacts.
**Why it happens:** ESC-50 was recorded at various rates and standardized to 44.1kHz. UrbanSound8K files have variable sample rates.
**How to avoid:** Resample all noise files to the target sample rate when loading into cache. Use `torchaudio.functional.resample()` which is already a project dependency.
**Warning signs:** Augmented spectrograms have unexpected high-frequency content or aliasing artifacts.

### Pitfall 3: Memory Pressure from Noise Cache
**What goes wrong:** ESC-50 (2000 x 5s x 44.1kHz = ~880 MB raw) + UrbanSound8K (8732 x 4s = ~2.8 GB raw) overwhelms memory.
**Why it happens:** Both datasets loaded into RAM simultaneously.
**How to avoid:** Option A: Use only ESC-50 (~880 MB after resampling to 16kHz, ~200 MB). Option B: Subsample UrbanSound8K to ~1000 representative files. Option C: Load lazily (random file per call, not pre-cached). Recommendation: ESC-50 only for initial implementation (~200 MB at 16kHz), add UrbanSound8K support as configurable option.
**Warning signs:** OOM errors during dataset initialization; Docker container killed.

### Pitfall 4: Over-Augmentation Degrading Signal
**What goes wrong:** Combining noise augmentation (SNR -10 dB) + pitch shift + time stretch + gain + SpecAugment produces unrecognizable audio that the model can't learn from.
**Why it happens:** Each augmentation has probability p=0.5, but combined probability of at least one augmentation is ~97%. At extreme settings, the signal is destroyed.
**How to avoid:** Use moderate probabilities (p=0.3-0.5 per transform). The SNR range of -10 to +20 dB is aggressive at the low end -- consider starting at -5 to +20 dB and only going to -10 dB if the model needs more robustness.
**Warning signs:** Training loss plateaus at a high value; model can't distinguish augmented drone from pure noise.

### Pitfall 5: Balanced Sampler Already Exists but May Not Handle DADS
**What goes wrong:** `build_weighted_sampler()` uses inverse frequency weighting. If DADS is already balanced (50/50), the sampler does nothing. If DADS is imbalanced, verify the sampler produces correct batch ratios.
**Why it happens:** The existing sampler was built for small field datasets; DADS has 180K files and may have different class distribution.
**How to avoid:** Log class distribution before and after sampling. Verify batch-level balance during first epoch.
**Warning signs:** Validation metrics skewed toward majority class despite sampler.

## Code Examples

### Complete FocalLoss Module
```python
# Source: torchvision.ops.sigmoid_focal_loss official docs
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Binary focal loss for drone detection training.

    Wraps torchvision.ops.sigmoid_focal_loss for drop-in use in training loop.
    Expects raw logits (pre-sigmoid) as input.

    Args:
        alpha: Weighting factor for positive class. Default 0.25.
        gamma: Focusing parameter for hard examples. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits (N,) -- NO sigmoid applied.
            targets: Binary labels (N,) with values 0 or 1.

        Returns:
            Scalar loss value.
        """
        return sigmoid_focal_loss(
            inputs, targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean",
        )
```

### TrainingConfig Extensions
```python
# Add to acoustic/training/config.py TrainingConfig class:

# Loss function (TRN-10)
loss_function: str = "focal"  # "focal", "bce", "bce_weighted"
focal_alpha: float = 0.25
focal_gamma: float = 2.0
bce_pos_weight: float = 1.0  # For weighted BCE fallback

# Background noise augmentation (TRN-11)
noise_augmentation_enabled: bool = True
noise_dirs: list[str] = []  # Paths to ESC-50/UrbanSound8K directories
noise_snr_range_low: float = -10.0
noise_snr_range_high: float = 20.0
noise_probability: float = 0.5

# Waveform augmentation via audiomentations (TRN-12)
pitch_shift_semitones: float = 3.0
time_stretch_min: float = 0.85
time_stretch_max: float = 1.15
waveform_gain_db: float = 6.0
augmentation_probability: float = 0.5
```

### Audiomentations Integration
```python
# Source: audiomentations docs + README examples
from audiomentations import Compose, PitchShift, TimeStretch, Gain
import numpy as np


class AudiomentationsAugmentation:
    """Waveform augmentation using audiomentations library.

    Replaces the custom WaveformAugmentation with standardized transforms.
    """

    def __init__(
        self,
        pitch_semitones: float = 3.0,
        time_stretch_range: tuple[float, float] = (0.85, 1.15),
        gain_db: float = 6.0,
        p: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self._sample_rate = sample_rate
        self._augment = Compose([
            PitchShift(
                min_semitones=-pitch_semitones,
                max_semitones=pitch_semitones,
                p=p,
            ),
            TimeStretch(
                min_rate=time_stretch_range[0],
                max_rate=time_stretch_range[1],
                p=p,
            ),
            Gain(
                min_gain_db=-gain_db,
                max_gain_db=gain_db,
                p=p,
            ),
        ])

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation to 1-D float32 mono audio.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Augmented audio as float32 array.
        """
        return self._augment(samples=audio, sample_rate=self._sample_rate)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nn.BCELoss | Focal Loss (torchvision) | This phase | Better handling of easy/hard examples; standard for detection tasks since RetinaNet (2017) |
| Gaussian noise only | ESC-50/UrbanSound8K background mixing | This phase | Real-world noise conditions vs synthetic Gaussian; most impactful augmentation per drone detection research |
| Custom gain-only augmentation | audiomentations PitchShift + TimeStretch + Gain | This phase | Pitch and speed variation simulate Doppler effects and different drone types |
| Inverse-frequency sampler | Same (already implemented) | Phase 8 | `build_weighted_sampler()` already handles class imbalance; verify with DADS distribution |

## Open Questions

1. **DADS class distribution**
   - What we know: DADS has 180,320 files labeled drone/no-drone. The exact class ratio is not documented in the phase 13 research.
   - What's unclear: If DADS is already balanced (50/50), the weighted sampler is a no-op. If imbalanced, sampler handles it.
   - Recommendation: Log class distribution at training start. The `build_weighted_sampler()` already handles any ratio -- just verify the output batch distribution.

2. **ESC-50 vs UrbanSound8K: which to use**
   - What we know: ESC-50 is 2000 files / ~600 MB download / 50 classes. UrbanSound8K is 8732 files / ~6 GB download / 10 urban sound classes.
   - What's unclear: Which provides better noise augmentation for outdoor drone detection. UrbanSound8K has more urban sounds (sirens, drilling, etc.) but ESC-50 has nature sounds (rain, wind, birds) that are more relevant to outdoor deployment.
   - Recommendation: Start with ESC-50 (smaller, contains relevant outdoor sounds). Add UrbanSound8K as optional second noise source. Make noise_dirs configurable as a list.

3. **ResearchCNN sigmoid vs logits for focal loss**
   - What we know: Current ResearchCNN applies sigmoid in forward(). Focal loss needs raw logits.
   - What's unclear: Whether Phase 15 builds on ResearchCNN (Phase 8) or EfficientAT (Phase 14).
   - Recommendation: Phase 15 depends on Phase 14 (EfficientAT). EfficientAT outputs logits natively. If ResearchCNN is also supported, add a `logits_mode` flag or separate the sigmoid from the forward pass.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torchvision | Focal loss | Yes | 0.26.0 | Custom focal loss implementation |
| audiomentations | Waveform augmentation | No (not installed) | -- | pip install audiomentations>=0.43.1 |
| ESC-50 dataset | Noise augmentation | No (must download) | -- | Skip noise augmentation; use Gaussian noise fallback |
| UrbanSound8K dataset | Noise augmentation (optional) | No (must download) | -- | Use ESC-50 only |
| PyTorch | Training | Yes | 2.11 | -- |
| torchaudio | Resampling noise files | Yes | 2.11 | -- |
| soundfile | Loading WAV files | Yes | installed | -- |

**Missing dependencies with no fallback:**
- None (all blocking deps have fallbacks)

**Missing dependencies with fallback:**
- `audiomentations` -- must be installed via pip (new dependency)
- ESC-50/UrbanSound8K datasets -- must be downloaded; noise augmentation is disabled without them

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 with pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/unit/test_focal_loss.py tests/unit/test_noise_augmentation.py tests/unit/test_audiomentations_aug.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRN-10 | FocalLoss produces correct gradient direction | unit | `python -m pytest tests/unit/test_focal_loss.py::test_focal_loss_gradient -x` | Wave 0 |
| TRN-10 | FocalLoss with gamma=0 matches BCE | unit | `python -m pytest tests/unit/test_focal_loss.py::test_focal_matches_bce_at_gamma_zero -x` | Wave 0 |
| TRN-10 | Loss function selection in config | unit | `python -m pytest tests/unit/test_focal_loss.py::test_loss_config_selection -x` | Wave 0 |
| TRN-10 | Weighted BCE fallback works | unit | `python -m pytest tests/unit/test_focal_loss.py::test_weighted_bce_fallback -x` | Wave 0 |
| TRN-11 | BackgroundNoiseMixer produces output at correct SNR | unit | `python -m pytest tests/unit/test_noise_augmentation.py::test_snr_range -x` | Wave 0 |
| TRN-11 | BackgroundNoiseMixer handles empty noise dir gracefully | unit | `python -m pytest tests/unit/test_noise_augmentation.py::test_empty_noise_dir -x` | Wave 0 |
| TRN-11 | Noise files resampled to target sample rate | unit | `python -m pytest tests/unit/test_noise_augmentation.py::test_resample -x` | Wave 0 |
| TRN-12 | Audiomentations augmentation produces correct output shape | unit | `python -m pytest tests/unit/test_audiomentations_aug.py::test_output_shape -x` | Wave 0 |
| TRN-12 | Pitch shift stays within semitone range | unit | `python -m pytest tests/unit/test_audiomentations_aug.py::test_pitch_shift -x` | Wave 0 |
| TRN-12 | Time stretch changes duration within range | unit | `python -m pytest tests/unit/test_audiomentations_aug.py::test_time_stretch -x` | Wave 0 |
| TRN-12 | Augmentation probability p=0 returns unchanged audio | unit | `python -m pytest tests/unit/test_audiomentations_aug.py::test_no_augmentation -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/test_focal_loss.py tests/unit/test_noise_augmentation.py tests/unit/test_audiomentations_aug.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_focal_loss.py` -- covers TRN-10 (focal loss, config, fallback)
- [ ] `tests/unit/test_noise_augmentation.py` -- covers TRN-11 (background noise mixing, SNR, resampling)
- [ ] `tests/unit/test_audiomentations_aug.py` -- covers TRN-12 (pitch, time stretch, gain, compose)

## Project Constraints (from CLAUDE.md)

- **Runtime:** Python backend, single Docker container
- **ML Framework:** PyTorch >=2.11,<2.12 (already pinned)
- **Testing:** pytest with pytest-asyncio; tests in `tests/unit/` and `tests/integration/`
- **Linting:** Ruff for formatting and linting
- **Config:** pydantic-settings with `ACOUSTIC_TRAINING_*` env prefix
- **Training isolation:** Background thread with `os.nice(10)`, `torch.set_num_threads(2)`

## Sources

### Primary (HIGH confidence)
- [torchvision sigmoid_focal_loss](https://docs.pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html) - Official API, installed version 0.26.0
- [audiomentations PyPI](https://pypi.org/project/audiomentations/) - Version 0.43.1, transform API
- [audiomentations GitHub](https://github.com/iver56/audiomentations) - PitchShift, TimeStretch, Gain, AddBackgroundNoise docs
- Codebase analysis: `src/acoustic/training/trainer.py`, `dataset.py`, `augmentation.py`, `config.py`, `parquet_dataset.py`

### Secondary (MEDIUM confidence)
- [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50) - Dataset structure, 2000 files, 50 classes, CC-BY-NC license
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) - 8732 files, 10 folds, variable sample rates
- Phase 14 RESEARCH.md - EfficientAT model outputs logits (no sigmoid), confirming focal loss compatibility

### Tertiary (LOW confidence)
- ESC-50/UrbanSound8K memory usage estimates based on file counts and durations -- actual memory depends on resampling and dtype

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - torchvision already installed; audiomentations is well-established (0.43.1, active development)
- Architecture: HIGH - Extends existing augmentation pipeline pattern; drop-in loss replacement
- Pitfalls: HIGH - Sigmoid/logits mismatch is the #1 risk, well-documented; memory concerns quantifiable
- Noise datasets: MEDIUM - ESC-50 well-documented; UrbanSound8K requires registration; memory estimates approximate

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (all libraries stable; audiomentations actively maintained)
