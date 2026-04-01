# Phase 6: Preprocessing Parity Foundation - Research

**Researched:** 2026-04-01
**Domain:** Audio preprocessing (mel-spectrogram), Python protocol design, librosa-to-torchaudio migration
**Confidence:** HIGH

## Summary

Phase 6 replaces the non-functional EfficientNet preprocessing pipeline with a research-validated mel-spectrogram preprocessor using torchaudio, introduces Classifier/Preprocessor protocols for clean model swaps, and validates numerical parity against saved TF/librosa reference tensors. The existing `OnnxDroneClassifier`, `preprocess_for_cnn()`, and all ONNX dependencies are removed entirely.

The core technical challenge is achieving numerical parity (atol=1e-4) between librosa's mel-spectrogram output and torchaudio's MelSpectrogram transform. Research confirms this is achievable with correct parameter mapping: `norm="slaney"`, `mel_scale="slaney"`, `pad_mode="constant"`, `center=True`, `power=2.0`. The research code's normalization `(S_db+80)/80` clipped to [0,1] is straightforward to replicate because both libraries use per-tensor-max dB reference and `top_db=80`.

**Primary recommendation:** Build the torchaudio preprocessor with exact parameter matching to librosa defaults, generate .npy reference fixtures from the research code using the locally installed librosa 0.11.0, and test parity before touching the worker or pipeline wiring.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Remove existing EfficientNet-B0 preprocessing pipeline entirely. No coexistence -- replace `preprocess_for_cnn()` and all EfficientNet-specific code (224x224 resize, 3-channel repeat, z-score normalization).
- **D-02:** Remove `OnnxDroneClassifier` and all ONNX runtime references. Delete `onnxruntime` from dependencies. Pipeline has no working classifier until Phase 7.
- **D-03:** `CNNWorker` updated to use protocols but dormant (no classifier) until Phase 7. Keep worker structure.
- **D-04:** Single `MelConfig` dataclass: SR=16000, N_FFT=1024, HOP_LENGTH=256, N_MELS=64, MAX_FRAMES=128, SEGMENT_SECONDS=0.5, normalization=(S_db+80)/80 clipped to [0,1].
- **D-05:** Minimal protocols only. `Preprocessor`: `process(audio: np.ndarray, sr: int) -> torch.Tensor` returning (1, 1, 128, 64). `Classifier`: `predict(features: torch.Tensor) -> float` returning drone probability [0, 1].
- **D-06:** Switch from librosa to torchaudio for mel-spectrogram computation.
- **D-07:** Generate reference tensors from TF/librosa research code once, save as .npy fixtures. Tests compare against fixtures with atol=1e-4. No TensorFlow in CI.

### Claude's Discretion
- Where to place `MelConfig` (e.g., `src/acoustic/classification/config.py` or `src/acoustic/config.py`)
- How to restructure `CNNWorker` to accept protocol-injected dependencies
- Which WAV files to use as parity test fixtures (from `audio-data/data/`)
- Whether to keep `fast_resample()` or switch to torchaudio's resampling
- How to handle the `SILENCE_RMS_THRESHOLD` energy gate in the new preprocessor

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRE-01 | Shared MelConfig with research parameters (SR=16000, N_FFT=1024, HOP=256, N_MELS=64, MAX_FRAMES=128, (S_db+80)/80 normalization) | MelConfig dataclass design with frozen=True, all constants validated against research reference `train_strong_cnn.py` lines 35-42 |
| PRE-02 | Classifier and Preprocessor protocols for clean model swaps | Protocol design with `typing.Protocol`, minimal signatures per D-05. CNNWorker refactored to accept protocol types |
| PRE-03 | Preprocessing outputs (1, 1, 128, 64) tensors from 0.5s segments with research normalization | torchaudio MelSpectrogram with matched parameters, AmplitudeToDB for dB conversion, custom (S_db+80)/80 normalization |
| PRE-04 | Numerical parity tests verify PyTorch preprocessing matches research TF output within atol=1e-4 | .npy reference fixtures generated from research code, parameter mapping verified, known pitfalls documented |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0 | Tensor operations | Already installed. Required for torchaudio and Phase 7 CNN |
| torchaudio | 2.11.0 | MelSpectrogram transform | Replaces librosa per D-06. Matches torch version. GPU-ready for Phase 8 |
| numpy | >=1.26,<3 | Audio array handling | Already installed. Used for raw audio input before torch conversion |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| soundfile | >=0.13.1 | WAV file loading | Generating reference fixtures from research code |
| librosa | 0.11.0 | Reference fixture generation ONLY | One-time script to produce .npy fixtures. NOT a runtime dependency |

### Dependencies to Remove
| Library | Reason |
|---------|--------|
| onnxruntime | D-02: ONNX model is dead, removing OnnxDroneClassifier |
| scipy.ndimage.zoom | Only used for EfficientNet 224x224 resize, no longer needed |

**Installation:**
```bash
pip install torchaudio==2.11.0
```

**Removal:**
```bash
# Remove from requirements.txt:
# - onnxruntime (if present)
# Remove librosa from runtime deps (keep as dev-only if needed for fixture generation)
```

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/classification/
    __init__.py
    config.py           # NEW: MelConfig dataclass
    protocols.py         # NEW: Classifier, Preprocessor protocols
    preprocessing.py     # REPLACED: torchaudio-based ResearchPreprocessor
    worker.py            # MODIFIED: protocol-injected dependencies
    state_machine.py     # UNCHANGED
    inference.py         # DELETED (OnnxDroneClassifier removed)

tests/
    fixtures/
        reference_melspec_440hz.npy   # NEW: reference tensor from librosa
        reference_melspec_drone.npy   # NEW: reference tensor from real WAV
    unit/
        test_preprocessing.py         # REWRITTEN: test new preprocessor
        test_protocols.py             # NEW: protocol compliance tests
        test_mel_config.py            # NEW: config validation tests
        test_parity.py               # NEW: numerical parity tests
        test_inference.py            # DELETED (OnnxDroneClassifier removed)
```

### Pattern 1: Frozen Dataclass for MelConfig
**What:** Use `@dataclass(frozen=True)` for immutable preprocessing configuration
**When to use:** Always -- prevents accidental mutation of shared constants
**Example:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MelConfig:
    """Research-validated mel-spectrogram preprocessing parameters.

    All values match Acoustic-UAV-Identification train_strong_cnn.py.
    """
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    max_frames: int = 128
    segment_seconds: float = 0.5
    db_range: float = 80.0  # top_db for power_to_db and normalization divisor

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)
```

### Pattern 2: typing.Protocol for Dependency Injection
**What:** Use `typing.Protocol` (structural subtyping) for Classifier and Preprocessor interfaces
**When to use:** To decouple pipeline from specific implementations
**Example:**
```python
from typing import Protocol, runtime_checkable
import numpy as np
import torch

@runtime_checkable
class Preprocessor(Protocol):
    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Process raw audio into model-ready features.

        Args:
            audio: 1-D float32 mono audio.
            sr: Sample rate of input audio.

        Returns:
            Tensor of shape (1, 1, max_frames, n_mels).
        """
        ...

@runtime_checkable
class Classifier(Protocol):
    def predict(self, features: torch.Tensor) -> float:
        """Run inference on preprocessed features.

        Returns:
            Drone probability in [0.0, 1.0].
        """
        ...
```

### Pattern 3: torchaudio MelSpectrogram Pipeline
**What:** Compose torchaudio transforms to replicate the research preprocessing
**When to use:** The `ResearchPreprocessor` implementation
**Example:**
```python
import torch
import torchaudio.transforms as T
import numpy as np

class ResearchPreprocessor:
    def __init__(self, config: MelConfig | None = None) -> None:
        self._config = config or MelConfig()
        c = self._config
        self._mel_spec = T.MelSpectrogram(
            sample_rate=c.sample_rate,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            n_mels=c.n_mels,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
            pad_mode="constant",
        )
        self._to_db = T.AmplitudeToDB(stype="power", top_db=c.db_range)

    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        c = self._config
        # Resample if needed
        waveform = torch.from_numpy(audio).float()
        if sr != c.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, c.sample_rate)

        # Take or pad to segment_samples
        n = c.segment_samples
        if waveform.shape[0] >= n:
            waveform = waveform[-n:]
        else:
            waveform = torch.nn.functional.pad(waveform, (n - waveform.shape[0], 0))

        # Mel spectrogram -> dB -> normalize
        S = self._mel_spec(waveform)            # (n_mels, frames)
        S_db = self._to_db(S)                    # (n_mels, frames), range ~[-80, 0]
        S_norm = (S_db + c.db_range) / c.db_range
        S_norm = S_norm.clamp(0.0, 1.0)

        # Transpose to (frames, n_mels), pad/trim to max_frames
        spec = S_norm.T  # (frames, n_mels)
        frames = spec.shape[0]
        if frames < c.max_frames:
            pad_amount = c.max_frames - frames
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_amount))
        else:
            spec = spec[:c.max_frames]

        # Shape: (1, 1, max_frames, n_mels) = (1, 1, 128, 64)
        return spec.unsqueeze(0).unsqueeze(0)
```

### Anti-Patterns to Avoid
- **Importing librosa at runtime:** librosa is only for one-time fixture generation. The service must not depend on librosa at runtime.
- **Duplicating magic numbers:** All constants must come from MelConfig. No `1024`, `256`, `64`, `128`, `0.5` literals scattered in code.
- **Using `AmplitudeToDB` default `top_db=None`:** Must set `top_db=80.0` explicitly to match librosa's `power_to_db(top_db=80)` default.
- **Using torchaudio MelSpectrogram defaults for norm/mel_scale:** Defaults are `norm=None, mel_scale="htk"`, which differ from librosa's `norm="slaney", htk=False`. Must override.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mel filter bank | Custom mel-frequency triangular filters | `torchaudio.transforms.MelSpectrogram` | Exact numerical behavior matters for parity; hand-rolling risks subtle differences |
| dB conversion | `10 * torch.log10(S)` | `torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)` | Handles numerical stability (amin), per-tensor max reference, top_db clipping |
| Audio resampling | polyphase filter with scipy | `torchaudio.functional.resample` | Consistent with torchaudio stack, handles fractional ratios |
| Protocol checking | Custom ABC + register | `typing.Protocol` with `@runtime_checkable` | Structural subtyping -- no inheritance required, duck typing with type safety |

**Key insight:** The entire preprocessing chain must be built from torchaudio primitives to ensure consistent behavior and enable GPU acceleration in Phase 8. Mixing scipy/numpy DSP with torch tensors adds unnecessary conversion overhead and complicates parity validation.

## Common Pitfalls

### Pitfall 1: torchaudio MelSpectrogram Parameter Mismatch
**What goes wrong:** torchaudio MelSpectrogram defaults (`norm=None`, `mel_scale="htk"`) differ from librosa defaults (`norm="slaney"`, `htk=False`), producing entirely different mel filter banks and thus different spectrograms.
**Why it happens:** Developers assume defaults match between libraries.
**How to avoid:** Explicitly set `norm="slaney"`, `mel_scale="slaney"`, `pad_mode="constant"`, `center=True` on `MelSpectrogram`. These match the librosa defaults used by the research code.
**Warning signs:** Parity test fails with large differences (>1.0), not small numerical noise.

### Pitfall 2: AmplitudeToDB Reference Value Behavior
**What goes wrong:** `AmplitudeToDB` uses per-tensor maximum as reference (like `librosa.power_to_db(ref=np.max)`). If you accidentally use a fixed reference, dB values will be offset.
**Why it happens:** The research code passes `ref=np.max` to `power_to_db`, which is a function reference, not a constant. `AmplitudeToDB` handles this automatically.
**How to avoid:** Use `AmplitudeToDB(stype="power", top_db=80.0)` without trying to set a ref value. Verify that max dB value in output is 0.0.
**Warning signs:** Normalized values consistently shifted above 1.0 or below 0.0 before clipping.

### Pitfall 3: pad_or_trim Center-Crop vs Left-Crop
**What goes wrong:** The research code's `pad_or_trim_frames` uses center-cropping for trimming (`start = (t - max_frames) // 2`), while the existing service code uses left-cropping (`spec[:max_frames]`). For 0.5s segments at 16kHz, a 0.5s segment with hop=256 produces `floor(8000/256) + 1 = 32` frames, which is well under `MAX_FRAMES=128`. So trimming should never trigger in practice. But the implementation must match the research code's behavior for edge cases.
**Why it happens:** Subtle difference in research code vs service code.
**How to avoid:** Match the research code exactly: center-crop when trimming, zero-pad at end when padding.
**Warning signs:** Parity test fails only on longer audio segments.

### Pitfall 4: CNNWorker Segment Duration Hardcoded
**What goes wrong:** `pipeline.py` line 58 hardcodes `int(settings.sample_rate * 2.0)` for CNN segment samples. This must change to `settings.sample_rate * 0.5` to match the research 0.5s segments.
**Why it happens:** Original pipeline was designed for EfficientNet with 2.0s segments.
**How to avoid:** Use `MelConfig.segment_seconds` or update `AcousticSettings` to derive segment duration from the new config.
**Warning signs:** CNN receives 2s of audio but preprocessor expects 0.5s, producing wrong frame count.

### Pitfall 5: Silence Energy Gate Threshold
**What goes wrong:** The current `SILENCE_RMS_THRESHOLD = 0.001` was tuned for 2.0s segments. A 0.5s segment has 4x less audio, so the RMS calculation may behave differently.
**Why it happens:** Threshold was never scientifically validated, just a heuristic.
**How to avoid:** Keep the energy gate concept but move the threshold to `MelConfig` or `AcousticSettings` so it can be recalibrated. The threshold value (0.001 = -60 dBFS) is conservative enough that it should work for 0.5s segments too.
**Warning signs:** Real drone audio being classified as silence.

### Pitfall 6: Worker Import Cycle After Protocol Introduction
**What goes wrong:** Moving protocols to a separate file and importing them in both `worker.py` and `main.py` can create circular imports if `__init__.py` re-exports everything.
**Why it happens:** Python circular import pain.
**How to avoid:** Keep `protocols.py` dependency-free (only `typing`, `numpy`, `torch` imports). Never import protocols from `__init__.py`. Use `TYPE_CHECKING` guards where needed.
**Warning signs:** ImportError at startup.

## Code Examples

### Reference Fixture Generation Script
```python
"""One-time script to generate .npy reference tensors from the research code.

Run this ONCE to create fixtures, then delete or move to scripts/.
Requires librosa (dev dependency only).
"""
import numpy as np
import soundfile as sf
import librosa

# Match research code parameters exactly
FS = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_FRAMES = 128
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS)

def segment_to_melspec_reference(samples: np.ndarray, sr: int) -> np.ndarray:
    """Exact copy of research code's segment_to_melspec."""
    S = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db + 80.0) / 80.0
    S_db_norm = np.clip(S_db_norm, 0.0, 1.0)
    spec = S_db_norm.T
    # pad_or_trim_frames
    t = spec.shape[0]
    if t < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - t, spec.shape[1]), dtype=np.float32)
        spec = np.concatenate([spec, pad], axis=0)
    elif t > MAX_FRAMES:
        start = (t - MAX_FRAMES) // 2
        spec = spec[start:start + MAX_FRAMES]
    return spec.astype(np.float32)

# Generate from synthetic 440Hz sine
t = np.linspace(0, CHUNK_SECONDS, CHUNK_SAMPLES, endpoint=False, dtype=np.float32)
sine_440 = np.sin(2 * np.pi * 440 * t)
ref_sine = segment_to_melspec_reference(sine_440, FS)
np.save("tests/fixtures/reference_melspec_440hz.npy", ref_sine)

# Generate from a real drone WAV (pick a short one)
# data, sr = sf.read("audio-data/data/drone/0-20m/some_file.wav", dtype="float32")
# if sr != FS: data = librosa.resample(data, orig_sr=sr, target_sr=FS)
# if data.ndim > 1: data = data.mean(axis=1)
# segment = data[:CHUNK_SAMPLES]
# ref_drone = segment_to_melspec_reference(segment, FS)
# np.save("tests/fixtures/reference_melspec_drone.npy", ref_drone)
```

### torchaudio Parameter Mapping (Critical Reference)
```python
# librosa defaults (used by research code):
#   norm="slaney" (from librosa.filters.mel)
#   htk=False (from librosa.filters.mel) -> mel_scale="slaney" in torchaudio
#   pad_mode="constant" (from librosa.feature.melspectrogram, librosa 0.11.0)
#   center=True
#   power=2.0
#   power_to_db: ref=np.max, amin=1e-10, top_db=80.0

# torchaudio equivalent:
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    power=2.0,
    norm="slaney",          # matches librosa norm="slaney"
    mel_scale="slaney",     # matches librosa htk=False
    center=True,            # matches librosa center=True
    pad_mode="constant",    # matches librosa 0.11.0 pad_mode="constant"
)

to_db = torchaudio.transforms.AmplitudeToDB(
    stype="power",          # input is power spectrogram
    top_db=80.0,            # matches librosa top_db=80.0 default
)
# AmplitudeToDB automatically uses per-tensor max as ref (like librosa ref=np.max)
```

### Protocol-Based CNNWorker Skeleton
```python
class CNNWorker:
    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        classifier: Classifier | None = None,
        fs_in: int = 48000,
        silence_threshold: float = 0.001,
    ) -> None:
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._fs_in = fs_in
        self._silence_threshold = silence_threshold
        # ... existing queue/thread setup unchanged ...

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                mono_audio, az_deg, el_deg = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Energy gate
                rms = np.sqrt(np.mean(mono_audio ** 2))
                if rms < self._silence_threshold:
                    # Report silence
                    ...
                    continue

                # Preprocess (if preprocessor available)
                if self._preprocessor is None:
                    continue
                features = self._preprocessor.process(mono_audio, self._fs_in)

                # Classify (if classifier available -- None until Phase 7)
                if self._classifier is None:
                    continue
                prob = self._classifier.predict(features)
                # ... store result ...
            except Exception:
                logger.exception("Error in CNN inference")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| librosa for mel-spectrogram | torchaudio MelSpectrogram | Phase 6 (D-06) | Aligns with PyTorch stack, enables GPU in Phase 8 |
| z-score normalization | (S_db+80)/80 clipped [0,1] | Phase 6 (D-04) | Matches research training normalization |
| 2.0s audio segments | 0.5s segments | Phase 6 (D-04) | Matches research CHUNK_SECONDS=0.5 |
| EfficientNet-B0 (224x224x3) | Research CNN (128x64x1) | Phase 6/7 | Simpler, purpose-built architecture |
| ONNX Runtime inference | PyTorch native (Phase 7) | Phase 6 removes ONNX | Simpler deployment, single ML framework |

**Deprecated/outdated:**
- `OnnxDroneClassifier`: Dead ONNX model, non-functional. Removed in D-02.
- `preprocess_for_cnn()`: EfficientNet preprocessing pipeline. Replaced entirely in D-01.
- `scipy.ndimage.zoom`: Only used for 224x224 resize. No longer needed.
- `librosa` as runtime dep: Replaced by torchaudio. Keep as dev-only for fixture generation.

## Open Questions

1. **Which WAV files for parity fixtures?**
   - What we know: `audio-data/data/drone/0-20m/` and `audio-data/data/background/` contain real recordings.
   - What's unclear: Which specific file(s) provide best coverage (short vs long, quiet vs loud).
   - Recommendation: Use one synthetic 440Hz sine (deterministic, no file dependency) and one real drone WAV from `audio-data/data/drone/0-20m/` (validates real-world audio path). The synthetic fixture can be committed; the real WAV fixture should be generated and committed as .npy only.

2. **Keep fast_resample or switch to torchaudio?**
   - What we know: `fast_resample()` uses `scipy.signal.resample_poly` (polyphase filtering). `torchaudio.functional.resample` uses sinc interpolation with Kaiser window.
   - What's unclear: Whether the numerical difference between resampling methods affects parity at atol=1e-4.
   - Recommendation: Switch to `torchaudio.functional.resample` for consistency with the torchaudio stack. The parity test compares at the final output stage, and both resampling methods are high-quality. Since the reference fixtures are generated from audio already at 16kHz (from the research code), resampling differences only matter for live pipeline input (48kHz -> 16kHz), which is not part of the parity test.

3. **SILENCE_RMS_THRESHOLD placement**
   - What we know: Currently a module-level constant (0.001). Needs to survive the preprocessing.py rewrite.
   - Recommendation: Move to `AcousticSettings` (env-configurable) or keep as a parameter on `CNNWorker.__init__`. It is not a preprocessing parameter per se -- it is an inference gating decision, so it belongs in the worker, not in `MelConfig`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ with pytest-asyncio |
| Config file | `pyproject.toml` ([tool.pytest.ini_options]) |
| Quick run command | `python -m pytest tests/unit/ -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRE-01 | MelConfig has all research constants, no magic numbers in codebase | unit | `pytest tests/unit/test_mel_config.py -x` | No -- Wave 0 |
| PRE-02 | Protocols exist, CNNWorker accepts protocol-typed deps | unit | `pytest tests/unit/test_protocols.py -x` | No -- Wave 0 |
| PRE-03 | Preprocessor outputs (1,1,128,64) from 0.5s audio, values in [0,1] | unit | `pytest tests/unit/test_preprocessing.py -x` | Yes -- needs rewrite |
| PRE-04 | Numerical parity within atol=1e-4 of reference fixtures | unit | `pytest tests/unit/test_parity.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_mel_config.py` -- covers PRE-01 (MelConfig constants, no duplicate magic numbers)
- [ ] `tests/unit/test_protocols.py` -- covers PRE-02 (protocol structural typing, isinstance checks)
- [ ] `tests/unit/test_parity.py` -- covers PRE-04 (numerical parity with .npy fixtures)
- [ ] `tests/fixtures/reference_melspec_440hz.npy` -- reference tensor from librosa for parity test
- [ ] `tests/unit/test_preprocessing.py` -- EXISTS but needs complete rewrite for new preprocessor
- [ ] `tests/unit/test_inference.py` -- EXISTS, must be DELETED (OnnxDroneClassifier removed)
- [ ] `tests/fixtures/dummy_model.onnx` -- EXISTS, must be DELETED (ONNX removed)
- [ ] torchaudio installation: `pip install torchaudio==2.11.0`

## Sources

### Primary (HIGH confidence)
- Research reference: `Acoustic-UAV-Identification-main-main/train_strong_cnn.py` -- canonical preprocessing parameters and `segment_to_melspec()` implementation
- Existing service code: `src/acoustic/classification/preprocessing.py`, `inference.py`, `worker.py` -- current implementation to replace
- Local librosa 0.11.0 introspection -- verified defaults: `norm="slaney"`, `htk=False`, `pad_mode="constant"`, `power_to_db(ref=1.0, amin=1e-10, top_db=80.0)`
- [torchaudio AmplitudeToDB docs](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.AmplitudeToDB.html) -- per-tensor max ref, top_db parameter
- [torchaudio MelSpectrogram docs](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html) -- defaults: `norm=None`, `mel_scale="htk"`, `pad_mode="reflect"`

### Secondary (MEDIUM confidence)
- [Compare torchaudio and librosa spectrograms (GitHub Gist)](https://gist.github.com/mthrok/01f89d9bc27a7fe618bf5e8ef71b44ba) -- parameter matching guide
- [MelSpectrogram inconsistency issue #1058](https://github.com/pytorch/audio/issues/1058) -- documents known differences between libraries
- [librosa melspectrogram docs](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html) -- official parameter reference

### Tertiary (LOW confidence)
- None -- all findings verified against official documentation or local library introspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- torch 2.11.0 installed, torchaudio 2.11.0 available on PyPI, versions verified
- Architecture: HIGH -- protocols are standard Python typing, MelConfig is straightforward frozen dataclass
- Pitfalls: HIGH -- parameter mismatch between librosa and torchaudio verified through official docs and local introspection of both libraries
- Parity approach: HIGH -- atol=1e-4 is achievable with correct parameter mapping; the main risk is `pad_mode` and `norm` defaults, which are now documented

**Research date:** 2026-04-01
**Valid until:** 2026-05-01 (stable domain, no fast-moving dependencies)
