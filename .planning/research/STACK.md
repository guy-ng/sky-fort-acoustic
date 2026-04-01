# Stack Research: Classification Pipeline Migration

**Domain:** Acoustic UAV classification pipeline migration (TF/Keras to PyTorch)
**Researched:** 2026-04-01
**Confidence:** HIGH
**Scope:** NEW dependencies only -- existing stack (FastAPI, sounddevice, NumPy, SciPy, pyzmq) is validated and unchanged.

## What Is Being Replaced

The research codebase (`Acoustic-UAV-Identification-main-main/`) uses:
- `tensorflow` / `tf.keras` for model definition, training loops, callbacks
- `librosa` for mel-spectrogram computation (both training and inference)
- `sklearn.model_selection.train_test_split` for dataset splitting
- `soundfile` for WAV I/O (already in our stack)
- Manual NumPy metrics (confusion matrix, accuracy, precision, recall, F1)

The current service (`src/acoustic/classification/`) uses:
- `onnxruntime` for EfficientNet-B0 inference (ONNX model)
- `librosa` for mel-spectrogram computation
- `scipy.ndimage.zoom` for resizing spectrograms to 224x224

Both ONNX Runtime and the EfficientNet-B0 architecture are being replaced entirely.

## Recommended Stack Additions

### Core ML Framework

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| torch | >=2.11.0,<2.12 | CNN model definition, training, inference | Already decided in project stack. PyTorch 2.11 supports Python 3.14 (matching current venv). `torch.compile()` available for inference optimization. Eager mode makes debugging the 3-layer CNN trivial. | HIGH |
| torchaudio | >=2.11.0,<2.12 | Mel-spectrogram transforms | Replaces librosa for spectrogram computation. `torchaudio.transforms.MelSpectrogram` produces GPU-acceleratable, differentiable spectrograms. Must version-match torch exactly (shared C++ extensions). | HIGH |

### Training Pipeline

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| scikit-learn | >=1.6 | Train/val/test split, evaluation metrics | The research code already uses `sklearn.model_selection.train_test_split`. Also provides `classification_report`, `confusion_matrix`, `accuracy_score`, `precision_recall_fscore_support` -- all used in the eval script. Lightweight, no GPU needed for metrics. Prefer over torchmetrics because: (1) evaluation runs post-training on small arrays, no batched metric accumulation needed, (2) sklearn is the universal standard for reporting, (3) avoids adding yet another PyTorch-ecosystem dependency for a simple binary classification task. | HIGH |

### Data Augmentation

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| torchaudio (built-in) | (same as above) | Time/frequency masking | `torchaudio.transforms.TimeMasking` and `FrequencyMasking` implement SpecAugment -- the standard augmentation for audio classification. No extra dependency needed. Operates on mel-spectrogram tensors directly. | HIGH |
| audiomentations | >=0.39 | Waveform-level augmentation | Time stretching, pitch shifting, adding background noise, gain variation -- applied BEFORE mel-spectrogram computation. More comprehensive than torch-audiomentations (which is still "early development stage" with multiprocessing memory leak issues). audiomentations is mature, NumPy-based, works on CPU in the data loading pipeline. | MEDIUM |

### What Stays (Already in requirements.txt)

| Technology | Role in Classification | Notes |
|------------|----------------------|-------|
| numpy | Array manipulation, aggregation math (p_max, p_mean, p_agg) | No version change needed |
| scipy | `resample_poly` for 48kHz to 16kHz downsampling | No version change needed |
| soundfile | WAV I/O for training data loading | No version change needed |

## What to REMOVE

| Remove | Why | Replace With |
|--------|-----|-------------|
| onnxruntime | EfficientNet-B0 ONNX model is being replaced by PyTorch CNN | `torch` native inference |
| onnx | Only needed for ONNX export/validation | Not needed -- inference stays in PyTorch |
| librosa (from classification) | Replaced by torchaudio for mel-spectrograms | `torchaudio.transforms.MelSpectrogram` |

**Note on librosa removal:** librosa is currently used in `src/acoustic/classification/preprocessing.py` for `melspectrogram()` and `power_to_db()`. The PyTorch port must use `torchaudio.transforms.MelSpectrogram` instead, configured to match the research parameters. librosa can be kept as a dev dependency for validation/comparison during porting, but must not be in the production inference path. Reasons:
1. torchaudio spectrograms are differentiable (enables future end-to-end training)
2. torchaudio avoids the librosa->NumPy->torch conversion overhead
3. One fewer C dependency (libsndfile is still needed for soundfile, but librosa also pulls in numba, llvmlite, etc.)

## Critical: Matching Research Preprocessing Parameters

The research code uses these exact parameters that must be reproduced in torchaudio:

```python
# Research parameters (from train_strong_cnn.py)
SR = 16000          # Sample rate after resampling
N_FFT = 1024        # FFT window size
HOP_LENGTH = 256    # Hop between frames
N_MELS = 64         # Mel filter banks
MAX_FRAMES = 128    # Time dimension after pad/trim

# Normalization: (S_db + 80.0) / 80.0, clipped to [0, 1]
# This is a fixed-range normalization, NOT zero-mean/unit-variance
```

The torchaudio equivalent:

```python
import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    power=2.0,           # Power spectrogram (matches librosa power=2.0)
    norm="slaney",       # Must match librosa default
    mel_scale="slaney",  # Must match librosa default
)

amplitude_to_db = T.AmplitudeToDB(
    stype="power",
    top_db=80.0,         # Matches librosa power_to_db(ref=np.max) behavior
)
```

**WARNING:** torchaudio and librosa mel-spectrograms differ by default. You MUST set `norm="slaney"` and `mel_scale="slaney"` in torchaudio to match librosa's defaults. Validate numerically during porting -- max absolute difference should be < 1e-4.

## Architecture of the Research CNN (Port Target)

The research CNN is a straightforward 3-layer architecture from `train_strong_cnn.py`:

```
Conv2D(32, 3x3) -> BN -> MaxPool(2x2)
Conv2D(64, 3x3) -> BN -> MaxPool(2x2)  
Conv2D(128, 3x3) -> BN -> MaxPool(2x2)
GlobalAvgPool2D
Dense(128) -> Dropout(0.3)
Dense(1, sigmoid)  -- binary classification
```

Input shape: `(batch, 1, 128, 64)` -- single channel mel-spectrogram (PyTorch NCHW format).

This is ~95K parameters. No need for torchvision, no pretrained weights, no complex architectures. Define it directly with `torch.nn`.

## Late Fusion Ensemble

The research inference (`run_strong_inference.py`) uses segment aggregation:
- `p_max`: max probability across segments
- `p_mean`: mean probability across segments  
- `p_agg`: 1 - prod(1 - p_i) -- probability that at least one segment contains drone

The milestone also calls for multi-model ensemble (soft/hard voting). This is pure NumPy math on model output arrays -- no additional library needed.

## Installation

```bash
# NEW dependencies for classification migration
pip install \
  torch>=2.11.0,<2.12 \
  torchaudio>=2.11.0,<2.12 \
  scikit-learn>=1.6

# Optional: waveform augmentation for training
pip install audiomentations>=0.39

# REMOVE from requirements.txt:
# onnxruntime
# onnx
# librosa (move to dev-only if needed for validation)
```

**Docker image size impact:** PyTorch CPU-only wheel is ~280MB. This is significant but unavoidable for a training pipeline. Use `--index-url https://download.pytorch.org/whl/cpu` in Docker to avoid shipping CUDA libraries (~2GB savings).

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|------------------------|
| torchaudio `MelSpectrogram` | librosa `melspectrogram` | Only during porting validation -- compare outputs numerically, then remove librosa from production path |
| scikit-learn metrics | torchmetrics | If you later need batched metric accumulation during training with DDP. Overkill for binary classification on a single GPU/CPU |
| scikit-learn metrics | Manual NumPy (as in research code) | Never -- sklearn is more tested, handles edge cases, produces standard reports |
| audiomentations | torch-audiomentations | When you need GPU-accelerated augmentation in the training loop. Currently torch-audiomentations has memory leak issues in multiprocessing and is "early development stage" |
| audiomentations | torchaudio built-in only | If you only need SpecAugment (time/freq masking). Add audiomentations when you need waveform-level augmentation (noise injection, time stretch) |
| torch native inference | ONNX Runtime | If inference latency becomes critical on resource-constrained devices. Can always `torch.onnx.export()` later. For now, keep the stack simple -- one framework for train and inference |
| Custom training loop | PyTorch Lightning | If training becomes complex (multi-GPU, mixed precision, complex logging). The research CNN is 95K params with binary crossentropy -- a 50-line training loop is sufficient. Lightning adds abstraction without benefit here |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| TensorFlow / Keras | Project decision: PyTorch only. Two ML frameworks = maintenance nightmare | torch + torchaudio |
| torchvision | No pretrained models needed. The CNN is a custom 3-layer architecture, not EfficientNet/ResNet | Define model directly with torch.nn |
| PyTorch Lightning | Overkill for a simple binary classifier with ~95K params | Custom training loop (~50 lines) |
| torchmetrics | Extra dependency for metrics that sklearn already handles. No distributed training to sync across | scikit-learn |
| torch-audiomentations | "Early development stage", known memory leaks with multiprocessing | audiomentations (CPU, NumPy-based, mature) |
| librosa (in production) | Adds numba + llvmlite (~150MB), slower than torchaudio, not differentiable | torchaudio.transforms |
| TensorBoard (via torch) | Heavy dependency for logging | CSV logging + custom FastAPI endpoint for training progress. TensorBoard can be added later as optional dev tool |
| Weights & Biases / MLflow | Adds external service dependency for a single-model training pipeline | File-based logging, expose via REST API |

## Version Compatibility Matrix

| Package | Compatible With | Notes |
|---------|----------------|-------|
| torch 2.11.x | Python 3.10-3.14 | Python 3.14 confirmed supported. 3.14t (freethreaded) experimental |
| torchaudio 2.11.x | torch 2.11.x only | Must version-match exactly. Different minor versions will fail to load |
| scikit-learn 1.6.x | Python >=3.10, NumPy >=1.21 | Compatible with our NumPy >=1.26 |
| audiomentations 0.39.x | NumPy >=1.21, SciPy >=1.3 | Pure Python + NumPy, no torch dependency |
| torch 2.11.x | NumPy >=1.26,<3 | Our existing NumPy constraint is compatible |

## Integration Points with Existing Service

### Preprocessing Pipeline Change

**Current** (`src/acoustic/classification/preprocessing.py`):
```
mono_audio -> resample(scipy) -> librosa.melspectrogram -> scipy.ndimage.zoom(224x224) -> ONNX
```

**New** (after migration):
```
mono_audio -> resample(scipy) -> torchaudio.MelSpectrogram(128x64) -> normalize -> torch CNN
```

Key differences:
1. No more 224x224 resize (EfficientNet-B0 input size). Research CNN takes native 128x64.
2. Normalization changes from zero-mean/unit-variance to `(S_db + 80) / 80` clipped to [0,1].
3. Output is `torch.Tensor` not `np.ndarray`.

### Inference Integration

The existing `OnnxDroneClassifier` class in `src/acoustic/classification/inference.py` will be replaced by a `PyTorchDroneClassifier` that:
1. Loads a `.pt` or `.pth` state dict (not `.h5` Keras or `.onnx`)
2. Runs `model.eval()` + `torch.no_grad()` for inference
3. Returns segment-level probabilities
4. Performs aggregation (p_max, p_mean, p_agg) to produce file-level prediction

### Training Integration

Training runs via a FastAPI endpoint or CLI command. The training loop:
1. Uses `torch.utils.data.Dataset` + `DataLoader` (replaces `tf.data.Dataset`)
2. Reads WAV files with `soundfile` (already in stack)
3. Computes mel-spectrograms with `torchaudio.transforms.MelSpectrogram`
4. Standard PyTorch training: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
5. Early stopping and LR scheduling via manual callbacks (replaces Keras callbacks)
6. Saves model with `torch.save(model.state_dict(), path)`

## Sources

- [PyTorch 2.11.0 Release](https://github.com/pytorch/pytorch/releases) - Python 3.14 support confirmed
- [torchaudio MelSpectrogram docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) - Parameter reference
- [torchaudio PyPI](https://pypi.org/project/torchaudio/) - Version 2.11.0 confirmed
- [torch PyPI](https://pypi.org/project/torch/) - Version 2.11.0, Python 3.14 wheels available
- [torchmetrics 1.9.0 docs](https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html) - Evaluated and rejected for this use case
- [torch-audiomentations GitHub](https://github.com/asteroid-team/torch-audiomentations) - "Early development stage" warning noted
- [audiomentations GitHub](https://github.com/iver56/audiomentations) - Mature waveform augmentation library
- [torchaudio vs librosa mel comparison](https://gist.github.com/aFewThings/f4dde48993709ab67e7223e75c749d9d) - Demonstrates parameter matching requirements
- [librosa/torchaudio mel matching on Kaggle](https://www.kaggle.com/code/nomorevotch/create-the-same-mel-from-librosa-and-torchaudio) - norm="slaney", mel_scale="slaney" required
- Research source: `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` (CNN architecture, preprocessing params)
- Research source: `Acoustic-UAV-Identification-main-main/run_strong_inference.py` (segment aggregation logic)
- Research source: `Acoustic-UAV-Identification-main-main/eval_folder_with_strong_cnn.py` (evaluation metrics, distribution analysis)
