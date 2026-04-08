# Phase 22: EfficientAT v8 Retrain — Research

**Researched:** 2026-04-08
**Domain:** PyTorch training pipeline + train/serve contract enforcement + Vertex AI remote training
**Confidence:** HIGH (everything below is verified via file read or filesystem probe; no training-data assumptions)

## Summary

v7 regressed because a single literal (`window_samples = int(0.5 * _SOURCE_SR)` at `src/acoustic/training/efficientat_trainer.py:456`) produced 0.5 s training windows while the inference pipeline feeds 1.0 s windows. The model is shape-agnostic so nothing crashed — it silently shipped with a broken decision surface. Phase 22 must fix this by making the training window length impossible to drift from `EfficientATMelConfig().segment_samples = 32000` (1.0 s @ 32 kHz), add a fail-loud dataset-level assertion, add a runtime WARN in the inference classifier, and fold the new 2026-04-08 UMA-16 field recordings (15.9 min drone + 2.3 min background) into training with a real-device hold-out split used only for the promotion gate.

**Primary recommendation:** Create `src/acoustic/classification/efficientat/window_contract.py` exporting `EFFICIENTAT_WINDOW_SECONDS = 1.0`, `EFFICIENTAT_TARGET_SR = 32000`, `EFFICIENTAT_SEGMENT_SAMPLES = EfficientATMelConfig().segment_samples`, and derive a source-rate helper `efficientat_source_window_samples(source_sr: int) -> int` returning `int(EFFICIENTAT_WINDOW_SECONDS * source_sr)`. Both `src/acoustic/pipeline.py` and `src/acoustic/training/efficientat_trainer.py` import from it. Then fine-tune v8 from v6 (not from AudioSet — lower-risk, faster on L4) with carryovers from Phase 20, executing Plan 20-06 on a hold-out that contains `20260408_091054_136dc5.wav` + the two longest background recordings.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

| Constraint | Value | Why |
|---|---|---|
| Training window length | **1.0 second** @ 32 kHz = 32000 samples | Must equal `EfficientATMelConfig().segment_samples` (the inference contract). Anything else reproduces the v7 bug. |
| Sliding-window overlap | **50%** (`window_overlap_ratio=0.5`) | User-specified. Note: DADS clips are uniformly 1.0 s, so 50% overlap on 1.0 s windows of 1.0 s clips yields 1 window per clip. Meaningful overlap only on multi-second field recordings. |
| Cloud region | **`us-east1`** | User-specified. |
| Cloud accelerator | **NVIDIA L4** | User-specified. Phase 20 used L4 in us-central1 — must verify us-east1 L4 quota before submission. |
| New training data | All `data/field/drone/20260408_*.wav` + `data/field/background/20260408_*.wav` | User-specified. |
| `20260408_091054_136dc5.wav` ("10inch payload 4kg") | Already trimmed to 61.4 s (was 71.4 s). Backup at `.bak`. | Use as-is. Do not re-trim. |
| Data integrity preflight | **Required** — assert every new recording is transferred, decoded, sample-rate-correct, and reaches the training loop with the right label | "make sure all the data is transfer correctly" |

### Claude's Discretion

1. **Fine-tune from v6 vs train from AudioSet checkpoint** — recommend below in Research Focus 6.
2. **Sliding-window sampler retirement for 1 s clips** — recommend below in Research Focus 7.
3. **Class balance with new field data** — WeightedRandomSampler re-weighting proposal in Research Focus 8.
4. **Vertex base image rebuild** — reuse-or-rebuild decision in Research Focus 9.

### Deferred Ideas (OUT OF SCOPE)

- None documented; all CONTEXT.md items are either locked decisions or open questions.

### Carry Over From Phase 20 (DO NOT revert)

- Wide-gain augmentation
- Room IR augmentation
- BG noise negatives (ESC50 / UrbanSound8K / FSD50K subset at `data/noise/`)
- Focal loss (focal_alpha=0.25, focal_gamma=2.0)
- Save gate D-32 (logits-mode parity check + degenerate-output refusal)
- Narrow Stage 1 schedule (unfreeze only the final binary head)
- RmsNormalize as LAST augmentation in the chain

### Promotion Gate (hard stop)

1. Plan 20-06 executed (eval harness + D-27 real-device TPR/FPR gate)
2. `real_TPR >= 0.80` AND `real_FPR <= 0.05` on UMA-16 hold-out
3. Hold-out files MUST NOT have been used in training
</user_constraints>

<phase_requirements>
## Phase Requirements

Phase 22 is a remediation phase driven by post-mortem, not by REQUIREMENTS.md. No existing REQ-IDs map to it cleanly. The planner should treat the following as phase-scoped requirements and surface them in PLAN.md must-haves (not added to REQUIREMENTS.md because they are mechanical engineering invariants, not product requirements):

| Phase-local ID | Description | Research Support |
|---|---|---|
| P22-W1 | Training window length equals inference window length for EfficientAT family (single source of truth) | Research Focus 1 — concrete refactor plan |
| P22-W2 | `WindowedHFDroneDataset.__getitem__` asserts `returned_tensor.shape[-1] == EfficientATMelConfig().segment_samples` | Research Focus 2 |
| P22-W3 | `EfficientATClassifier.predict` logs WARN when `features.shape[-1] != segment_samples` | Research Focus 2 |
| P22-W4 | Training RmsNormalize runs in the SAME sample-rate domain as inference (32 kHz, post-resample) | Research Focus 3 |
| P22-D1 | 2026-04-08 field recordings ingested into training with correct labels | Research Focus 4 |
| P22-D2 | Deterministic train/eval split for 2026-04-08 recordings, hold-out never seen by training | Research Focus 5 |
| P22-D3 | Data-integrity preflight fails fast on missing / wrong-SR / NaN audio before any training step | Research Focus 12 |
| P22-G1 | Plan 20-06 executed against v8 with real_TPR >= 0.80 AND real_FPR <= 0.05 | Research Focus 11 |
| P22-G2 | v8 replaces v6 as `models/efficientat_mn10.pt` only after P22-G1 | Research Focus 11 |

Phase 22 does not touch REQUIREMENTS.md IDs directly, but the fix restores CLS-01 / CLS-03 behavior to pre-v7 levels.
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Python `>=3.11`, PyTorch `>=2.11,<2.12`, torchaudio matched to PyTorch minor
- `numpy>=1.26,<3` (keep conservative for acoular compat)
- No hand-rolling of standard audio I/O, DSP, or augmentation primitives — use existing `torchaudio`, `sounddevice`, `soundfile`, `audiomentations`, `pyroomacoustics`, and the vendored `EfficientAT` package under `src/acoustic/classification/efficientat/`
- GSD workflow enforcement — all file changes must come through a GSD phase/plan
- The POC's 180-line custom SRP-PHAT stays; do not pull in acoular

---

## Standard Stack

This phase does not introduce new libraries. Everything already in the tree is sufficient.

### Core (already present, verified)

| Library | Version constraint (from CLAUDE.md) | Purpose in Phase 22 | Source |
|---|---|---|---|
| PyTorch | `>=2.11,<2.12` | Model + training loop | `efficientat_trainer.py:19-22` [VERIFIED: file read] |
| torchaudio | `>=2.11.0` | `F_audio.resample` 16k→32k, used in `WindowedHFDroneDataset.__getitem__:316` and `_LazyEfficientATDataset:144` [VERIFIED] |
| soundfile | `>=0.13.1` | Preflight file enumeration + read for data-integrity check [VERIFIED: used in `scripts/vertex_train.py` and existing WAV loaders] |
| audiomentations | existing | Pitch / stretch / gain augmentation chain (`AudiomentationsAugmentation`) [VERIFIED: `augmentation.py`] |
| pyroomacoustics | existing | `RoomIRAugmentation` — baked into `Dockerfile.vertex-base` [VERIFIED] |
| sklearn | existing | `roc_curve` for Plan 20-06 evaluator extension [VERIFIED: `20-06-eval-harness-and-promotion-gate-PLAN.md:156`] |
| google-cloud-aiplatform | existing | Vertex job submission (`CustomContainerTrainingJob`) [VERIFIED: `scripts/vertex_submit.py:198-218`] |

### New supporting module (to be created)

| File | Purpose |
|---|---|
| `src/acoustic/classification/efficientat/window_contract.py` | Single source of truth for window-length constants — see Research Focus 1 |
| `scripts/preflight_v8_data.py` | Data integrity preflight — see Research Focus 12 |

**No `pip install` needed.** No version verification performed because no new dependencies are introduced.

---

## Architecture Patterns

### Pattern 1: Contract-module single-source-of-truth (CANONICAL for Phase 22)

Place invariants in a tiny module that both consumers import. Prevents drift because there is literally nowhere else to hard-code the value.

```python
# src/acoustic/classification/efficientat/window_contract.py
"""EfficientAT train/serve window contract (single source of truth).

Phase 22 remediation for the v7 regression. Every consumer — trainer,
inference pipeline, dataset, classifier runtime check — imports from here.
If this file is wrong, EVERYTHING is wrong in the same way, which is
detectable by the integration tests. If call sites hard-code their own
value, the failure mode is silent (see v7 post-mortem).
"""
from __future__ import annotations
from .config import EfficientATMelConfig

EFFICIENTAT_WINDOW_SECONDS: float = 1.0
EFFICIENTAT_TARGET_SR: int = 32000
EFFICIENTAT_SEGMENT_SAMPLES: int = EfficientATMelConfig().segment_samples  # 32000

def source_window_samples(source_sr: int) -> int:
    """Window length in SOURCE-rate samples (pre-resample)."""
    return int(EFFICIENTAT_WINDOW_SECONDS * source_sr)

# Contract self-check — runs at import time, crashes on mismatch
assert EFFICIENTAT_SEGMENT_SAMPLES == int(EFFICIENTAT_WINDOW_SECONDS * EFFICIENTAT_TARGET_SR), (
    f"EfficientAT window contract broken: segment_samples={EFFICIENTAT_SEGMENT_SAMPLES} "
    f"!= window_seconds * target_sr = {int(EFFICIENTAT_WINDOW_SECONDS * EFFICIENTAT_TARGET_SR)}"
)
```

### Pattern 2: Dataset-level fail-loud assertion

`WindowedHFDroneDataset.__getitem__` must assert output shape before returning. A training collapse or silent shape-drift is caught at the first bad sample.

### Pattern 3: Inference-side runtime invariant (WARN, not raise)

`EfficientATClassifier.predict` logs a WARN when input length disagrees with contract. Rationale: raising would DoS a live operational pipeline for a soft-drift; a WARN gives operators a visible signal without killing detection. This mirrors the pattern CONTEXT.md explicitly asks for.

### Anti-Patterns to Avoid

- **Hard-coded numeric literals for window length anywhere outside `window_contract.py`.** Any `int(0.5 * ...)`, `int(1.0 * ...)`, `8000`, `16000`, or `32000` in a training/inference path is suspect.
- **Deriving the trainer window from `TrainingConfig.window_overlap_ratio` alone.** Overlap and window length are orthogonal.
- **Re-trimming `20260408_091054_136dc5.wav`.** CONTEXT.md locks it at 61.4 s; the `.bak` exists [VERIFIED: filesystem probe — 4569680 bytes `.bak` vs 3929680 bytes trimmed].

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| WAV decoding / sample-rate check for preflight | Custom RIFF header parser | `soundfile.info(path)` | Returns `samplerate`, `frames`, `channels`, `duration` without decoding. Microseconds per file. [CITED: python-soundfile docs] |
| File-level deterministic hashing for split | `random.seed` + shuffle | `hashlib.md5(basename.encode()).hexdigest()[:8]` then int | Reproducible across runs without carrying a seed, filename-stable |
| ROC curve for Plan 20-06 | Custom threshold sweep | `sklearn.metrics.roc_curve` | Already planned in `20-06-eval-harness-and-promotion-gate-PLAN.md:156` [VERIFIED] |
| L4 quota check | parse `gcloud` output by hand | Reuse `check_l4_quota()` from `scripts/vertex_submit.py:57-83` but pass `region="us-east1"` | Already exists; just call it with the new region [VERIFIED: file read] |
| Window-length derivation | 3 separate literals | Import `window_contract` everywhere | Whole point of the phase |

---

## Runtime State Inventory

Phase 22 is a retrain phase, not a rename. Runtime state is minimal but worth documenting.

| Category | Items Found | Action Required |
|---|---|---|
| Stored data | None — training produces a new checkpoint file `models/efficientat_mn10_v8.pt`, does not mutate existing stores | None |
| Live service config | `DetectionSession.window_seconds` is derived at session start from `_training_window_seconds(model_type)` [VERIFIED: `src/acoustic/pipeline.py:298-314`]. After v8 promotion this still returns 1.0 for `"efficientat"`. | None — contract unchanged |
| OS-registered state | None | None |
| Secrets/env vars | Vertex submission reads `ACOUSTIC_TRAINING_*` env vars from `build_env_vars_v7` in `scripts/vertex_submit.py:86-144`. Phase 22 will add/rename to `build_env_vars_v8`. No secret keys renamed. | Create new env builder, do not touch v7 code path |
| Build artifacts | `models/efficientat_mn10.pt` currently points at v6 [VERIFIED: `ls -la models/` — `efficientat_mn10.pt` dated Apr 5, same size as other v6 checkpoints]. After successful v8 promotion, `models/efficientat_mn10.pt` is overwritten from `efficientat_mn10_v8.pt`. `models/efficientat_mn10_v6.pt` stays as rollback. `models/efficientat_mn10_v6_fp32.onnx` and `models/efficientat_mn10_v6_int8.onnx` exist [VERIFIED] — these belong to Phase 21 (Pi edge app) and MUST NOT be touched by v8 promotion. | Phase 22 promotion must copy v8 → `models/efficientat_mn10.pt` only; leave `*_v6_*.onnx` alone |

---

## Research Focus 1 — Window-length single-source-of-truth refactor

### Where the constant should live

**Recommendation:** Create `src/acoustic/classification/efficientat/window_contract.py` (new file). Rationale:

- It lives next to `EfficientATMelConfig` (`config.py:12-29`), which is where `segment_samples` is already defined. Keeping the contract geographically colocated with the mel config makes it obvious that they co-vary.
- Importing from `pipeline.py` would create a training→runtime dependency (trainer imports pipeline). That's backward — training is more foundational.
- Classification package has no import cycle risk — it does not import from `training/` or `pipeline.py`.

Then:
- `src/acoustic/pipeline.py:72-86` `_training_window_seconds()` — import `EFFICIENTAT_WINDOW_SECONDS` and return it when model_type matches the efficientat family. Keep the `research_cnn` branch as-is (different training window, intentional).
- `src/acoustic/training/efficientat_trainer.py:456` — replace the literal with `source_window_samples(_SOURCE_SR)` (returns 16000 for 16 kHz source).

### Exact call sites that hard-code a window assumption

Verified by grep over `src/acoustic/` [VERIFIED: Grep results above]:

| File:Line | Current | Fix |
|---|---|---|
| `src/acoustic/training/efficientat_trainer.py:456` | `window_samples = int(0.5 * _SOURCE_SR)  # 8000 samples = 0.5 s @ 16 kHz` | `window_samples = source_window_samples(_SOURCE_SR)  # 16000 = 1.0s @ 16kHz` |
| `src/acoustic/training/efficientat_trainer.py:461` | `test_hop = window_samples  # D-16: non-overlapping test split` | Keep — derived |
| `src/acoustic/training/efficientat_trainer.py:459` | `int(window_samples * (1.0 - cfg.window_overlap_ratio))` | Keep — uses derived value |
| `src/acoustic/training/hf_dataset.py:219` | `window_samples: int = 8000,` (default arg to `WindowedHFDroneDataset.__init__`) | Change default to `16000` OR make `Optional[int]` and default to `source_window_samples(sample_rate)` inside `__init__`. Trainer always passes explicitly, so changing default is low-risk. |
| `src/acoustic/training/hf_dataset.py:220` | `hop_samples: int = 3200,` | Change default to `8000` (50% overlap on 16000) — matches new contract |
| `src/acoustic/training/hf_dataset.py:203-212` | Docstring math uses `window=8000` / 3-windows-per-clip example | Update to `window=16000`, 1 window per 1 s clip, and note that the new overlap semantics only apply to multi-second files |
| `src/acoustic/training/hf_dataset.py:222` | `assumed_clip_samples: int = 16000,` | **Unchanged** — this is the DADS source-clip length, not a window length. It equals the new window length by coincidence (both are 1 s @ 16 kHz). The class should assert `window_samples <= assumed_clip_samples` to preserve the original intent (pad otherwise). For the new multi-second field recordings, this field is the ENTIRE FILE LENGTH and must be computed per file — see Focus 7 for recommendation. |
| `src/acoustic/pipeline.py:83-84` | `if "efficientat" in mt ... return 1.0` | Replace `1.0` with `EFFICIENTAT_WINDOW_SECONDS` import |
| `src/acoustic/pipeline.py:159` | `self._cnn_segment_samples = int(settings.sample_rate * _training_window_seconds("research_cnn"))` | Unchanged — uses helper |

### Second-order references (docstrings / comments — fix for consistency)

| File:Line | Current |
|---|---|
| `src/acoustic/training/hf_dataset.py:39` | docstring "extracts a random 0.5s segment" (`HFDroneDataset`) — this class is for the legacy non-EfficientAT path, leave alone |
| `src/acoustic/training/hf_dataset.py:79` | "Random 0.5s segment extraction" — same, legacy path |
| `src/acoustic/training/hf_dataset.py:212` | "yielding length `window_samples * 2` (e.g. 16000 samples = 0.5 s @ 32 kHz)" — UPDATE to `32000 samples = 1.0 s @ 32 kHz` |
| `src/acoustic/pipeline.py:67` | `DetectionSession.window_seconds: float = 0.5` — default for research_cnn, leave alone (overridden per-session from `_training_window_seconds`) |
| `src/acoustic/pipeline.py:79` | `research_cnn: MelConfig.segment_seconds = 0.5` — correct, leave alone |

---

## Research Focus 2 — Length assertions and runtime WARN

### `WindowedHFDroneDataset.__getitem__` assertion

Current signature [VERIFIED: `hf_dataset.py:264-319`]:

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    ...  # decode, slice window, optional aug, resample 16k->32k
    segment_t = F_audio.resample(segment_t, _SOURCE_SR, _TARGET_SR)  # line 316
    label_tensor = torch.tensor(self._labels_cache[idx], dtype=torch.float32)
    return segment_t, label_tensor
```

**Insert the assertion between lines 316 and 318**, right after resample and before returning. Must use the contract module (not a literal):

```python
from acoustic.classification.efficientat.window_contract import EFFICIENTAT_SEGMENT_SAMPLES
...
assert segment_t.shape[-1] == EFFICIENTAT_SEGMENT_SAMPLES, (
    f"WindowedHFDroneDataset contract violation at idx={idx} file_idx={file_idx}: "
    f"expected {EFFICIENTAT_SEGMENT_SAMPLES} samples, got {segment_t.shape[-1]}. "
    f"This is the v7 train/serve mismatch — do not silence."
)
```

Reason to place AFTER resample: the contract is expressed in TARGET-rate samples (32000), which is what the model sees. Asserting in source-rate samples would be the same math (16000) but less directly connected to `EfficientATMelConfig.segment_samples`.

### `EfficientATClassifier.predict` runtime WARN

Current shape [VERIFIED: `classifier.py:41-58`]:

```python
def predict(self, features: torch.Tensor) -> float:
    with torch.no_grad():
        x = features
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mel = self._mel(x)
        mel = mel.unsqueeze(1)
        logits, _ = self._model(mel)
        return torch.sigmoid(logits).item()
```

**Insert the WARN immediately after `x = x.unsqueeze(0)`** (line 54), before `mel = self._mel(x)`:

```python
import logging
_logger = logging.getLogger(__name__)
...
expected = EfficientATMelConfig().segment_samples  # 32000
actual = int(x.shape[-1])
if actual != expected:
    _logger.warning(
        "EfficientAT input length mismatch: got %d samples, expected %d "
        "(%.3fs vs %.3fs @ 32kHz). Model will run but is out-of-domain — "
        "this is exactly the v7 regression signature. Check "
        "DetectionSession.window_seconds and pipeline._cnn_segment_samples.",
        actual, expected, actual / 32000, expected / 32000,
    )
```

Log at WARN (not ERROR, not raise). One-shot suppression is OPTIONAL — the pipeline pushes roughly every `interval_seconds` (default ~0.2 s), so a persistent mismatch would flood logs. Recommend a module-level `_warned_mismatch = False` flag, reset on classifier construction, to log once per `EfficientATClassifier` instance.

---

## Research Focus 3 — RmsNormalize domain parity

### Current state [VERIFIED by file read]

**Training path** (`efficientat_trainer.py:246-252`):
```
WideGain -> RoomIR -> Audiomentations -> BackgroundNoiseMixer -> RmsNormalize
```
The whole chain is a `ComposedAugmentation` stored as `waveform_aug` on `WindowedHFDroneDataset`. It runs INSIDE `WindowedHFDroneDataset.__getitem__` at line 309 (`segment = self._waveform_aug(segment)`), **BEFORE** the 16k→32k resample at line 316. So `RmsNormalize` sees 16 kHz audio.

**Inference path** (`src/acoustic/classification/preprocessing.py:195-209`):
```python
def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
    # ... gain, resample sr->32kHz ...
    if self._rms_target is not None:
        waveform = _rms_normalize(waveform, target=self._rms_target)
```
`_rms_normalize` runs AFTER resample → 32 kHz. So inference sees 32 kHz audio when normalizing.

### The fix

Move `RmsNormalize` out of the 16-kHz chain and apply it in the 32-kHz domain inside `WindowedHFDroneDataset.__getitem__`, after the resample on line 316.

**Concrete plan:**
1. In `EfficientATTrainingRunner._build_train_augmentation` (`efficientat_trainer.py:171-252`), STOP appending `RmsNormalize` to `augs` (delete lines 246-250). Return the chain without it.
2. Similarly in `_build_eval_augmentation` (lines 254-277), return a chain without `RmsNormalize` (but keep the noise mixer).
3. In `WindowedHFDroneDataset.__init__`, accept a new parameter `post_resample_norm: Callable | None = None`. Default `None` to preserve existing behavior for non-EfficientAT callers.
4. In `WindowedHFDroneDataset.__getitem__`, after line 316 (`segment_t = F_audio.resample(...)`), call:
   ```python
   if self._post_resample_norm is not None:
       # RmsNormalize expects numpy float32; convert, apply, convert back
       arr = segment_t.numpy()
       arr = self._post_resample_norm(arr)
       segment_t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
   ```
5. In the trainer, pass `post_resample_norm=RmsNormalize(target=cfg.rms_normalize_target)` when constructing both `train_ds` and `val_ds` (around `efficientat_trainer.py:463-476`).

### Will this break existing augmentation order assumptions?

[VERIFIED by reading `efficientat_trainer.py:171-252` and `augmentation.py:511-531`]:

- `WideGainAugmentation` operates at `sample_rate=_SOURCE_SR` (16000) — SR-agnostic math (just a multiplicative gain). **Moving RMS out does not affect it.**
- `RoomIRAugmentation` — constructed with `sample_rate=_SOURCE_SR=16000` [VERIFIED: `efficientat_trainer.py:195`]. Uses pyroomacoustics IRs synthesized at that SR. Moving RMS out does not affect it.
- `AudiomentationsAugmentation` (pitch / stretch / gain) — constructed with `sample_rate=_SOURCE_SR=16000`. Moving RMS out does not affect it.
- `BackgroundNoiseMixer` — `sample_rate=_SOURCE_SR=16000`. Mixes noise at a target SNR. **Previously** `RmsNormalize` came after the mixer to land the final signal at a fixed RMS regardless of noise level. With this fix that still happens, just in the 32 kHz domain. Polyphase resample preserves RMS to within ~2%, so the downstream distribution is essentially identical [CITED: v7 debug report line 103 — "torchaudio resampling preserves RMS up to polyphase filter bleed (~2%)"].
- `SpecAugment` — not in this chain; runs per-batch on device in `mel_train`. Unaffected.
- **D-34 contract from Phase 20** says "RmsNormalize LAST in the chain, after the noise mixer." After the fix, RMS is STILL last — just outside the `ComposedAugmentation` object, in a separate post-resample step. The semantic intent (every sample lands at a fixed target RMS regardless of prior augmentation) is preserved.

**Verdict: no augmentation order breakage.** The RMS move is a pure domain-parity fix.

---

## Research Focus 4 — 2026-04-08 data inventory

[VERIFIED via `soundfile.info` + filesystem probe]:

### Drone recordings — 13 files, 956.8 s (15.9 min)

| File | SR | Ch | Dur (s) | Sub-label |
|---|---|---|---|---|
| `20260408_084222_44dc5c.wav` | 16000 | 1 | 21.0 | 5inch |
| `20260408_084504_bb1e7f.wav` | 16000 | 1 | 45.5 | 5inch |
| `20260408_084750_2502ab.wav` | 16000 | 1 | 66.0 | 5inch |
| `20260408_085238_691dcc.wav` | 16000 | 1 | 62.9 | 10inch |
| `20260408_085814_243000.wav` | 16000 | 1 | 193.1 | 10inch payload 1.5kg |
| `20260408_090355_61c59e.wav` | 16000 | 1 | 31.2 | 10inch 1.5 kg payload |
| `20260408_090535_89a52a.wav` | 16000 | 1 | 39.8 | 10inch payload 1.5kg |
| `20260408_091054_136dc5.wav` | 16000 | 1 | **61.4** | 10inch 4kg (TRIMMED — `.bak` = 4569680 bytes, current = 3929680 bytes) |
| `20260408_091724_bb0ed8.wav` | 16000 | 1 | 52.5 | phantom 4 |
| `20260408_091900_e079a6.wav` | 16000 | 1 | 35.1 | phantom 4 |
| `20260408_092022_130096.wav` | 16000 | 1 | 102.3 | phantom 4 |
| `20260408_092615_1a055f.wav` | 16000 | 1 | 218.4 | 10inch heavy |
| `20260408_093648_cf1b45.wav` | 16000 | 1 | 27.8 | 5inch nopayload |

### Background recordings — 4 files, 139.2 s (2.3 min)

| File | SR | Ch | Dur (s) |
|---|---|---|---|
| `20260408_073941_743505.wav` | 16000 | 1 | 5.0 |
| `20260408_082944_26056d.wav` | 16000 | 1 | 24.9 |
| `20260408_085109_668cc3.wav` | 16000 | 1 | 5.0 |
| `20260408_090757_1c50e9.wav` | 16000 | 1 | 104.4 |

### Anomalies / notes

- **All 17 files are 16 kHz mono** — identical domain to DADS, so they feed through `F_audio.resample(16k→32k)` with the exact same code path as DADS. No special casing needed.
- `20260408_091054_136dc5.wav` is confirmed trimmed to 61.4 s; `.bak` exists at 4569680 bytes (the original pre-trim). Do not re-trim or re-read `.bak`.
- All files have sidecar `.json` metadata with `sample_rate=16000` [VERIFIED] — matches WAV header. No drift.
- **Class imbalance is severe**: 956.8 s drone vs 139.2 s background = **6.87:1 drone-heavy**. Phase 20's DADS training is roughly class-balanced; adding this batch raw would skew further toward drone. See Focus 8.
- Drone class diversity: 5inch, 10inch (various payload), 10inch heavy, phantom 4 — 4 sub-types with 1-3 files each. Small sample size makes hash-based splitting fragile (see Focus 5).
- Total new audio: 18.3 min. Compared to DADS (hundreds of hours at 1 s per clip), this is <1% by duration, but it is the ONLY real-device data — disproportionately important for the D-27 real-device gate.

---

## Research Focus 5 — Hold-out split strategy

### Constraints

1. Must be deterministic and reproducible.
2. Hold-out must NOT leak into training (session-level isolation).
3. Hold-out must represent "operational use" — mix of drone types, including payload-loaded drones, and non-trivial background.
4. Total data is tiny (17 files). Hash-based splits on 17 items are high-variance.
5. Drone types must be represented in BOTH splits or the hold-out metric is dominated by one sub-label.

### Options considered

(a) **Hash-based 70/30 by filename** — simple, deterministic, but with 17 files it risks landing e.g. all three "phantom 4" files on one side. Unacceptable for such a small and diverse set.

(b) **Stratified split by sub-label** — keeps drone type distribution balanced but fragile for sub-labels with only 1-2 files (e.g., "10inch heavy" is one 218 s file; it can only be in one split).

(c) **Explicit hand-picked files by operational-value criteria** — transparent and defensible but not purely deterministic.

(d) **Duration-balanced** — keep total hold-out duration near target (e.g., ~30%) and pick files to achieve that.

### Recommendation: (c) explicit hand-picked hold-out + deterministic selection

Hold out these **5 files (4 drone + 1 background)**:

| File | Dur | Label | Rationale |
|---|---|---|---|
| `20260408_091054_136dc5.wav` | 61.4 | drone (10inch 4kg) | The heaviest payload condition; directly cited in CONTEXT.md open questions; represents the hardest detection case. User explicitly flagged this file. |
| `20260408_092615_1a055f.wav` | 218.4 | drone (10inch heavy) | Longest drone clip; only "10inch heavy" sample — must be in ONE split, and the hold-out needs the hardest condition. |
| `20260408_091724_bb0ed8.wav` | 52.5 | drone (phantom 4) | Drone-type diversity — one of the three phantom 4 files; holds out a non-DJI-FPV condition. |
| `20260408_084222_44dc5c.wav` | 21.0 | drone (5inch) | Drone-type diversity — smallest prop class. |
| `20260408_090757_1c50e9.wav` | 104.4 | background | Only sizeable background clip besides `082944` (which is shorter). Training still sees three other backgrounds. |

**Hold-out duration**: 353.3 s drone + 104.4 s bg = 457.7 s (7.6 min). Remaining in train: 603.4 s drone + 34.8 s bg = 638.2 s (10.6 min). Rough 41/59 hold-out/train by duration, but CONTEXT.md's operational priority (hold-out includes 4kg payload AND 10inch heavy AND phantom 4 AND 5inch — all four drone categories) justifies the larger-than-usual hold-out.

**Document in PLAN.md as a hard-coded list of filenames**, committed to git, with an `ASSERT` in the training data loader that NONE of these files are present in the training file list. That is the session-level isolation invariant.

**Alternative for the planner:** If the user prefers a purely deterministic rule, fall back to option (a) with seeds 42 and filename sort-stable order — but this is NOT recommended for 17 files.

---

## Research Focus 6 — Fine-tune from v6 vs train from EfficientAT AudioSet checkpoint

### v6 checkpoint

[VERIFIED by `ls -la models/`]:
- `models/efficientat_mn10_v6.pt` — 17019638 bytes, dated Apr 6 10:48
- `models/efficientat_mn10.pt` — 17020041 bytes, dated Apr 5 — this is NOT v6 but an older checkpoint (different size, pre-v6). The "current default" is actually older than v6. **Planner should verify** which checkpoint the live service loads by default before making promotion decisions.
- **No v6 sidecar JSON exists** [VERIFIED: `ls /models/*.json` returned empty]. Training config for v6 is NOT recorded as a sidecar. The v6 training config must be reconstructed from git history (commit before the `f007e91` quick task — the `_LazyEfficientATDataset` legacy path, `window_overlap_ratio=0`, `rir_enabled=False`).

### Fine-tune from v6 (RECOMMENDED)

**Pros:**
- v6 is the proven baseline — it survived real-device use and beats v7 on the 2026-04-08 recordings per the post-mortem.
- Faster convergence on L4 — v6 already has DADS-adapted weights. Expected stage 1 + stage 2 runtime ~30-50% of Phase 20's v7 run.
- Lower risk of catastrophic forgetting because the only delta is (a) fixed window length and (b) new field data in the mix. Weights are already in the right neighborhood.
- Phase 20 augmentation stack applies directly — wide gain / room IR / BG noise at 1 s windows is *more* consistent with v6 than with 0.5 s v7.

**Cons:**
- Cannot rule out v6 carrying a subtle bias from its earlier training distribution.
- If v6 is structurally limited on the 2026-04-08 data, fine-tuning won't fix it.

**Catastrophic forgetting risk** [ASSUMED — based on general transfer-learning literature, not measured]: low because (1) only one epoch of stage 1 unfreezes just the binary head, so the backbone is protected; (2) stage 2 unfreezes last 3 blocks at 1e-4 — this is exactly the Phase 20 recipe that trained v7 successfully (the issue was the window length, not the schedule).

**Expected delta on hold-out [ASSUMED]:** v6 on hold-out drone TPR is the operational baseline (unknown until Plan 20-06 is run for the first time). v8 fine-tuned from v6 with +18 min of real-device field data should improve TPR on the held-out drone types it has now seen similar examples of (5inch, phantom 4, 10inch with payload). Improvement on the held-out 4kg+heavy conditions is harder to predict because of the operational mismatch (training still trained mostly on DADS, with field data as ~1% of duration).

### Train from AudioSet checkpoint

**Pros:**
- Clean baseline, no legacy bias.
- Matches Phase 20 protocol exactly (that's what Phase 20 did for v7).

**Cons:**
- Costs 1.5-2x the L4 time on Vertex — three full stages from AudioSet weights.
- Larger surface area for things to go wrong (see Phase 20's 4 fix iterations).
- Does not leverage the fact that v6 already works.

### Recommendation

**Fine-tune from v6.** Set `cfg.pretrained_weights = "models/efficientat_mn10_v6.pt"` and adjust the loader to handle the fact that v6 has a binary head (`Linear(1280, 1)`) while the AudioSet pretrained weights have `Linear(1280, 527)`. The existing loader [VERIFIED: `efficientat_trainer.py:531-541`] only handles the shape-mismatch fallback ("training from scratch") — it does NOT currently support loading a binary checkpoint. The planner must either:

1. Add explicit branching: if `pretrained_weights` is a binary checkpoint (detect by loading + checking classifier[-1] output size), load it AFTER the binary head replacement on line 546.
2. Or load v6 AFTER `model.classifier[-1] = nn.Linear(in_features, 1)` so the shapes match directly.

Option 2 is simpler — move the `torch.load` + `load_state_dict` call to after line 547 when the config indicates "fine-tune from trained checkpoint". Gate on a new config flag `TrainingConfig.finetune_from_trained: bool = False` so the existing AudioSet-init path remains untouched.

---

## Research Focus 7 — Sliding-window sampler decision

### What `WindowedHFDroneDataset` actually does on 1 s clips with 1 s windows

[VERIFIED by code read + math]:

With `window_samples=16000` (new), `hop_samples=8000` (50% overlap), `assumed_clip_samples=16000`:

```
num_w = max(1, 1 + max(0, (16000 - 16000)) // 8000) = max(1, 1) = 1
```

**One window per DADS clip.** The sliding window becomes a no-op. Every `(file_idx, offset)` pair in `self._items` has `offset=0` and spans the entire clip.

This is exactly what `_LazyEfficientATDataset` does via random-crop fallback on clips of length equal to `segment_samples`. They are semantically equivalent — except `WindowedHFDroneDataset` emits deterministic items (one per file) while `_LazyEfficientATDataset` does a random crop that, when clip length == segment length, is also deterministic (crop from offset 0).

### Multi-second field recordings behavior

For the 2026-04-08 files at 16 kHz, window=16000, hop=8000:

| File | Length (s) | Source samples | Windows @ 50% overlap | Notes |
|---|---|---|---|---|
| 084222 (21.0 s) | 21.0 | 336000 | `1 + (336000-16000)//8000 = 1 + 40 = 41` | |
| 085814 (193.1 s) | 193.1 | ~3089600 | ~386 | |
| 092615 (218.4 s) | 218.4 | ~3494400 | ~436 | |
| 090757 bg (104.4 s) | 104.4 | ~1670400 | ~208 | |
| Train-side totals (12 kept drone + 3 kept bg files after hold-out removal) | — | — | **~1900-2200 windows total** | Sampler covers all of them per epoch |

### Problem with current `WindowedHFDroneDataset`

The class was written with `_assumed_clip_samples` as a uniform per-dataset parameter [VERIFIED: `hf_dataset.py:235,251-253`]. For DADS this is fine (all clips uniform). For the 2026-04-08 field recordings, the files have *different* lengths — `assumed_clip_samples` cannot be a single value.

The current tolerance code at `hf_dataset.py:280-299` pads or truncates mismatched clips back to `_assumed_clip_samples`. Running the field recordings through this class with `_assumed_clip_samples=16000` would **truncate every recording to 1 s** — throwing away ~99% of the field data. **Unacceptable.**

### Recommendation

**Split the data into two datasets** and wrap them in `torch.utils.data.ConcatDataset`:

1. **DADS dataset** → use `_LazyEfficientATDataset` (the legacy path, `efficientat_trainer.py:111-161`). Segment length = 32000 (post-resample). One random-crop segment per file. This is what v6 used. **Delete the Phase 20 sliding-window branch for DADS** — it adds nothing when clip length == window length.
2. **Field dataset (2026-04-08)** → write a new `FieldRecordingDataset` or generalize `WindowedHFDroneDataset` to accept `per_file_lengths: list[int]` instead of a single `assumed_clip_samples`. The sliding window with 50% overlap then produces many windows per multi-second file.

**Alternative (simpler): one-shot generalization of `WindowedHFDroneDataset`**

Compute `num_windows` PER FILE in `__init__` by actually decoding each file's length. This doubles the init time but is a one-time cost. Remove the `_assumed_clip_samples` uniform assumption.

Either way, the cleanest outcome is:
- `_LazyEfficientATDataset` for DADS (uniform 1 s clips, random crop is a no-op → deterministic)
- Generalized `WindowedHFDroneDataset` or a new `FieldRecordingDataset` for 2026-04-08 files
- `ConcatDataset([dads_ds, field_ds])` wired into the trainer
- `_build_weighted_sampler` reads `dataset.labels` which on `ConcatDataset` delegates to each child — verify this works or provide a flat labels list explicitly

**Recommendation to planner: go with alternative — generalize `WindowedHFDroneDataset`.** It keeps the session-level-isolation guarantees that Phase 20 locked in, and the per-file length computation is trivial (we already know the lengths from the preflight step — Focus 12).

---

## Research Focus 8 — Class balance with new field data

### Raw counts

- New field data (2026-04-08): drone 956.8 s / background 139.2 s ≈ 6.87:1 drone-heavy
- After hold-out removal (Focus 5): drone 603.4 s / background 34.8 s ≈ 17.3:1 drone-heavy in the training subset of field data (EVEN WORSE — hold-out took most of the background)

### Per-window counts (at window=1 s, 50% overlap)

Approximate windows from remaining training field data:
- Drone: ~1200 windows (603 s × 2 overlap-corrected factor)
- Background: ~70 windows (34.8 s × 2)

Compare to DADS (existing training mix) [INFERRED from Phase 20 context]: DADS has tens of thousands of 1 s clips, roughly class-balanced. The field data is ~1-2% of the total window count regardless of split.

### Will WeightedRandomSampler need re-weighting?

[VERIFIED by reading `_build_weighted_sampler` at `efficientat_trainer.py:47-67`]: the sampler is **inverse-frequency** based on `Counter(dataset.labels)` — it automatically re-weights whenever you change the dataset composition. No manual re-weighting code needed.

**However:** the sampler guarantees per-class balance in expectation, meaning a drone window is drawn proportionally to inverse drone count. With field data added, drone windows become MORE numerous → each individual drone window is drawn LESS often → every epoch still shows roughly equal numbers of drone and background samples. **This is what we want.** The 18 min of field data will be oversampled relative to its raw fraction because background windows are rare, and the model will see a disproportionate share of background field samples — exactly the direction we want for improving real-device FPR.

**Concrete action:** Nothing to change in the sampler code. Just make sure `train_ds.labels` for the concatenated dataset returns the flat per-window label list. Verify via a Wave 0 unit test.

### Optional: per-source oversampling

If the planner wants to push field data harder, wrap the field dataset in a `RepeatDataset(field_ds, repeat=5)` or similar before concatenating. Not recommended without empirical evidence — the sampler already handles the imbalance.

---

## Research Focus 9 — Vertex base image rebuild check

### Current state [VERIFIED by file read]

`Dockerfile.vertex-base` [VERIFIED: `cat Dockerfile.vertex-base`]:
- Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- Installs `libsndfile1`
- Installs `requirements-vertex.txt`
- Bakes `data/noise/` (ESC50 + UrbanSound8K + FSD50K subset) and `data/field/uma16_ambient/` into the image

Tag: `us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v1`

### What Phase 22 needs that isn't in the existing image

1. **2026-04-08 field recordings** (`data/field/drone/20260408_*.wav` + `data/field/background/20260408_*.wav`) — NOT baked in the v1 tag because they postdate Phase 20.
2. Trainer code change (window contract + RmsNormalize move) — this is code, not data, so it lives in the code layer rebuild on top of the base image, not in the base image itself.

### Rebuild decision

**Rebuild the base image.** Bake the 2026-04-08 field recordings in via a new `COPY data/field/drone /app/data/field/drone` + `COPY data/field/background /app/data/field/background` step. Push as a new tag:

```
us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v2
```

(Or `-v8` to match model version — planner's call.)

**Rationale:** Phase 20 deliberately bakes noise data into the base image to keep per-code-change pushes under ~50 MB. The field data is similar — small, static, referenced by absolute path in env vars (`/app/data/field/...`). Baking it follows the same pattern.

**Alternative:** Upload field data to GCS and have the trainer pull it at startup. Slower cold-start, more moving parts, more failure modes. Not recommended.

**No new Python deps** are needed — `requirements-vertex.txt` already has torch/torchaudio/pyroomacoustics/audiomentations. The window contract refactor introduces no new imports.

**Region note:** The base image is hosted in `us-central1-docker.pkg.dev`. Vertex jobs in `us-east1` CAN pull from `us-central1` Artifact Registry [ASSUMED: Artifact Registry is global per project — verify with gcloud]. If cross-region pull is a problem, replicate the image to a `us-east1-docker.pkg.dev` repo, but this is usually unnecessary.

---

## Research Focus 10 — Vertex L4 us-east1 quota check

### Existing helper [VERIFIED: `scripts/vertex_submit.py:57-83`]

```python
def check_l4_quota(project: str, region: str = "us-central1") -> bool:
    """Pre-flight L4 GPU quota check (Phase 20 D-22, Research Pitfall 5).

    Returns True if NVIDIA_L4 quota in the region appears > 0, else False.
    """
    ...
    result = subprocess.run(
        ["gcloud", "compute", "regions", "describe", region,
         ...
         "--format=value(quotas)"],
        ...
    )
    if "NVIDIA_L4" not in stdout:
        return False
    return "limit: 0" not in stdout and "limit: 0.0" not in stdout
```

### How to verify us-east1 quota before submitting

Command (run manually or inside a preflight step):

```bash
gcloud compute regions describe us-east1 \
    --project=interception-dashboard \
    --format="value(quotas)" | grep -A 1 "NVIDIA_L4"
```

Or call the existing helper:

```python
from scripts.vertex_submit import check_l4_quota
assert check_l4_quota("interception-dashboard", region="us-east1"), (
    "L4 quota unavailable in us-east1 — request quota increase or fall back to us-central1"
)
```

### Required changes in `vertex_submit.py` for Phase 22

[VERIFIED by reading `vertex_submit.py:39,57,161`]:

1. `GCP_REGION = "us-central1"` at line 39 is a module-level constant.
2. `check_l4_quota` default is `"us-central1"` at line 57.
3. `submit_v7_job` calls `check_l4_quota(GCP_PROJECT, GCP_REGION)` at line 161 — hard-coded to the module constant.

Phase 22 should:
- NOT mutate `GCP_REGION` (breaks v7 rollback scripts).
- Add a new `GCP_REGION_V8 = "us-east1"` constant.
- Add a new `build_env_vars_v8(...)` alongside `build_env_vars_v7`.
- Add a new `submit_v8_job(image_uri)` that uses `GCP_REGION_V8` and calls `check_l4_quota(GCP_PROJECT, GCP_REGION_V8)`.
- The docker image URI may also need to change if the Artifact Registry lives in us-central1 — see Focus 9.

**DO NOT submit a job during research or planning.** The submission is a Task inside Phase 22's plan file, not something research does.

---

## Research Focus 11 — Plan 20-06 readiness

### Is it executable as-is?

[VERIFIED: full read of `.planning/phases/20-.../20-06-eval-harness-and-promotion-gate-PLAN.md`]

**Conditionally yes, with v8-specific updates.** The plan is well-structured and produces the artifacts Phase 22 needs (`uma16_eval.py`, `promotion.py`, `evaluator.py` ROC extension, `promote_v7.py` CLI). But it is v7-specific in several places:

### v7-specific strings that Phase 22 must generalize or fork

| File / Function | v7 hard-coding | v8 fix |
|---|---|---|
| `src/acoustic/evaluation/promotion.py :: promote_v7_if_gates_pass` | function name, default `source_path="models/efficientat_mn10_v7.pt"` | Rename to `promote_if_gates_pass(source_path, target_path, ...)` — generic. Keep a thin `promote_v7_if_gates_pass` wrapper for backward compat. |
| `scripts/promote_v7.py` | CLI name, hard-coded v7 path, `--expected-sha256 <hex from 20-05-SUMMARY>` | Create `scripts/promote_v8.py` OR generalize to `scripts/promote_efficientat.py --checkpoint <path> --expected-sha256 <hex>` |
| `DADS_ACC_MIN = 0.95` | This came from Phase 20 D-26 for v7. Phase 22 CONTEXT.md does NOT lock a DADS threshold — it locks `real_TPR >= 0.80` and `real_FPR <= 0.05`. | Planner decision: keep 0.95 DADS threshold for consistency OR drop it and gate only on real-device metrics. **Recommendation: keep it.** Gives a fast first-line sanity check. |
| `models/efficientat_mn10_v7.pt` → `models/efficientat_mn10.pt` | Promotion target hard-coded | Promotion should take `source_path` = v8, `target_path` = `models/efficientat_mn10.pt` |
| `.planning/phases/20-.../20-05-SUMMARY.md` sha256 reference | Plan 20-06 expects this | Phase 22 produces its own summary file with v8 sha256 |
| `data/eval/uma16_real/labels.json` | ALREADY EXISTS [VERIFIED: `ls data/eval/uma16_real/` → `audio labels.json labels.json.example`] with Phase 20 / Feb-2026 `take0740` drone recordings. | **This is a pre-existing eval set, NOT the 2026-04-08 hold-out.** Phase 22 should EXTEND `labels.json` to include the 5 hold-out files from Focus 5, OR create a separate `data/eval/uma16_real_v8/labels.json` containing ONLY the 2026-04-08 hold-out files. **Recommendation: separate file** — the 2026-04-08 data is the only source of truth for the new regression gate. |

### Concrete blockers for Phase 22 executing Plan 20-06

1. **Plan 20-06 has never been run.** Per STATE.md and Phase 20-05 SUMMARY, Plan 20-06 was deferred. **This means `src/acoustic/evaluation/uma16_eval.py`, `src/acoustic/evaluation/promotion.py`, and `scripts/promote_v7.py` DO NOT EXIST YET.** [VERIFIED — the plan is pending, not executed]. Phase 22 must either:
   - Execute Plan 20-06 verbatim first (creates v7-specific artifacts), then add a thin v8 wrapper on top, OR
   - Skip Plan 20-06 and create the eval harness + promotion gate directly as v8-generic in Phase 22.
   **Recommendation: option 2.** Plan 20-06 was written for v7 and will produce v7-specific names that then need renaming. Write it once correctly in Phase 22 as `promote_if_gates_pass` and `scripts/promote_efficientat.py` with a `--version` flag.
2. **Existing `data/eval/uma16_real/labels.json`** is an older eval set. Phase 22 needs a NEW hold-out file specifically for the 2026-04-08 data. See Focus 5 for the 5-file list.
3. **Evaluator interface** — Plan 20-06 Task 2 notes "If the existing Evaluator interface differs from the assumed `evaluate_classifier(path)`, adapt the calls in this script to match". The planner must verify the current `Evaluator` class API before writing the CLI wrapper.

### Path forward for Phase 22

The planner should treat Plan 20-06 as a **design document for the eval harness architecture** and rewrite the plan in Phase 22's own wave 4 or 5 (after training succeeds) with v8-generic names. Copy-paste the code blocks, rename v7→efficientat, replace `data/eval/uma16_real` with the new 2026-04-08 hold-out dir, and execute against v8. This is strictly simpler than running Plan 20-06 then patching v8 on top.

---

## Research Focus 12 — Data integrity preflight

### What "assert every recording is correctly transferred, decoded, and reaches the training loop" means in code

Four distinct checks, each with a specific failure mode:

1. **File existence** — every file in the expected list exists on disk.
2. **Decode success** — `soundfile.read` returns a non-empty float32 array.
3. **Sample rate match** — header SR matches the expected 16000 for DADS / field data.
4. **No NaN / Inf** — numeric integrity.

Plus:
5. **Label balance** — drone vs background counts match expectations (catches directory mix-ups).
6. **Cardinality match** — the number of items the DataLoader emits equals the expected (files × windows) count.

### Where in the training pipeline

**BEFORE any DataLoader is constructed** and BEFORE any augmentation. The preflight must abort the Vertex job (exit non-zero) before the base image has burned any L4 time on a broken dataset.

Concretely: add a preflight function in a new `scripts/preflight_v8_data.py` and call it at the top of `EfficientATTrainingRunner.run()` or inside `scripts/vertex_train.py` before the runner is invoked.

### Proposed preflight (concrete)

```python
# scripts/preflight_v8_data.py
"""Phase 22 data integrity preflight.

Fails fast if any training file is missing, wrong SR, un-decodable, or NaN.
Runs inside the Vertex job before any DataLoader is constructed.
"""
from __future__ import annotations
import json, sys, logging
from pathlib import Path
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

EXPECTED_SR = 16000

def preflight_field_recordings(
    drone_dir: Path, bg_dir: Path, holdout_files: set[str],
) -> dict[str, list[tuple[Path, int, float]]]:
    """Enumerate training files, validate, return per-class manifest.

    Returns {"drone": [(path, samples, duration_s), ...], "background": [...]}
    excluding holdout_files. Raises AssertionError on any integrity failure.
    """
    manifest = {"drone": [], "background": []}
    errors: list[str] = []

    for label, subdir in [("drone", drone_dir), ("background", bg_dir)]:
        if not subdir.is_dir():
            errors.append(f"missing dir: {subdir}")
            continue
        for wav in sorted(subdir.glob("20260408_*.wav")):
            if wav.name in holdout_files:
                log.info("HOLDOUT (excluded from training): %s", wav.name)
                continue
            if not wav.exists():
                errors.append(f"missing file: {wav}")
                continue
            try:
                info = sf.info(str(wav))
            except Exception as exc:
                errors.append(f"sf.info failed for {wav}: {exc}")
                continue
            if info.samplerate != EXPECTED_SR:
                errors.append(f"{wav}: sr={info.samplerate} expected {EXPECTED_SR}")
                continue
            if info.channels != 1:
                errors.append(f"{wav}: channels={info.channels} expected 1")
                continue
            try:
                audio, sr = sf.read(str(wav), dtype="float32")
            except Exception as exc:
                errors.append(f"sf.read failed for {wav}: {exc}")
                continue
            if audio.size == 0:
                errors.append(f"{wav}: empty audio")
                continue
            if not np.isfinite(audio).all():
                errors.append(f"{wav}: NaN or Inf in audio")
                continue
            manifest[label].append((wav, int(audio.shape[0]), audio.shape[0] / sr))

    # Cardinality expectations (from Focus 4 inventory)
    expected_training_drone = 13 - 4  # 9 after holdout
    expected_training_bg = 4 - 1      # 3 after holdout
    if len(manifest["drone"]) != expected_training_drone:
        errors.append(f"drone count: {len(manifest['drone'])} != {expected_training_drone}")
    if len(manifest["background"]) != expected_training_bg:
        errors.append(f"bg count: {len(manifest['background'])} != {expected_training_bg}")

    if errors:
        for e in errors:
            log.error("PREFLIGHT: %s", e)
        raise AssertionError(f"Phase 22 preflight failed: {len(errors)} errors")

    log.info("PREFLIGHT OK: %d drone files (%.1fs), %d bg files (%.1fs)",
             len(manifest["drone"]), sum(d for _,_,d in manifest["drone"]),
             len(manifest["background"]), sum(d for _,_,d in manifest["background"]))
    return manifest
```

**DADS preflight** is a separate concern — Phase 20.1 already added a host-side check for the noise corpora (`scripts/acquire_noise_corpora.py` + preflight). The field-data preflight is additive.

**Windowed cardinality check** — after constructing the datasets but before training, assert:

```python
assert len(train_ds) == expected_total_windows, (
    f"training dataset cardinality {len(train_ds)} != expected {expected_total_windows} — "
    f"window math or file-length assumption is off"
)
```

---

## Common Pitfalls

### Pitfall 1: Silent train/serve drift (THE v7 bug)
**What goes wrong:** A single literal embeds an assumption that contradicts another single literal in a different file. Model runs without crashing, metrics look good, production is broken.
**Why it happens:** Two-sided invariants ("the trainer and the pipeline agree on X") are the most fragile things in ML systems. Nothing ties them together in the compiler or the type system.
**How to avoid:** Shared constants in a contract module + a dataset-level assertion + a runtime invariant WARN. Redundant by design — ANY of the three would have caught v7.
**Warning signs:** Val metrics great, real-device metrics bad. Val set drawn from same pipeline as train.

### Pitfall 2: Val set doesn't measure the deployed regime
**What goes wrong:** `WindowedHFDroneDataset` is used for both train and val splits → val uses the SAME broken window length as train → val says "HEALTHY".
**How to avoid:** Hold-out eval MUST go through the *inference* code path (`EfficientATClassifier.predict` on raw 32 kHz waveforms of length `segment_samples`), not through the training dataset class. This is what Plan 20-06 does via `evaluate_classifier`. Make sure the Phase 22 plan enforces "hold-out eval never touches `WindowedHFDroneDataset`."
**Warning signs:** Training dataset class appears in eval code path.

### Pitfall 3: Double-sigmoid on checkpoint load
**What goes wrong:** v7 was saved as state_dict (logits), but if someone exports to TorchScript baking in sigmoid and then the classifier wraps it in `torch.sigmoid` again, outputs are squashed.
**How to avoid:** The existing loader uses `torch.load(..., weights_only=True)` + `load_state_dict` [VERIFIED: `efficientat/__init__.py` and `classifier.py:57`]. No change needed — just don't export to TorchScript without re-verifying.

### Pitfall 4: DataLoader num_workers=0 on Vertex
**What goes wrong:** `_build_weighted_sampler` calls `Counter(dataset.labels)` which materializes the label list. For 1.9k-2.2k windows this is fast, but with `num_workers > 0` the augmentation chain must be picklable. `ComposedAugmentation` IS picklable [VERIFIED: `augmentation.py:533-546`], and `RmsNormalize` has explicit `__setstate__` support [VERIFIED: line 507-508].
**How to avoid:** Keep existing `num_workers` logic in `efficientat_trainer.py:499-503`. Don't introduce lambdas or closures in the augmentation chain.

### Pitfall 5: Field-data augmentation domain mismatch
**What goes wrong:** Field data is at 16 kHz — if the planner accidentally wires it into a path expecting 32 kHz, everything becomes half-speed.
**How to avoid:** The preflight check (Focus 12) asserts SR=16000 for every file. The generalized `WindowedHFDroneDataset` resamples 16k→32k identically to the DADS path. Use the same dataset class / sample rate contract.

### Pitfall 6: Cross-region Artifact Registry pull
**What goes wrong:** Vertex job in us-east1 fails to pull image from us-central1 Artifact Registry due to IAM or quota.
**How to avoid:** [ASSUMED: Artifact Registry is project-global and pulls cross-region — planner should verify with a dry-run gcloud command before submitting the real job.] If cross-region pull fails, replicate the image to a `us-east1` repo. Small blast radius, easy to detect.

### Pitfall 7: Holding out files that are class-unique
**What goes wrong:** Holding out the ONLY "10inch heavy" file means training never sees that condition, then the hold-out metric is also the model's only exposure to it — tautologically bad.
**How to avoid:** The recommended split (Focus 5) holds out one file per drone sub-class. Training still sees other examples of each class from DADS + the other field files.

---

## Code Examples

### Contract module (new file)
See Research Focus 1 for the full `window_contract.py`.

### Generalized WindowedHFDroneDataset constructor (adapted from `hf_dataset.py:215-260`)

```python
def __init__(
    self,
    hf_dataset,
    file_indices: list[int],
    *,
    window_samples: int | None = None,  # None = derive from contract
    hop_samples: int | None = None,      # None = 50% overlap
    waveform_aug: Callable | None = None,
    post_resample_norm: Callable | None = None,
    per_file_lengths: list[int] | None = None,  # NEW: per-file length override
    sample_rate: int = 16000,
) -> None:
    from acoustic.classification.efficientat.window_contract import source_window_samples
    if window_samples is None:
        window_samples = source_window_samples(sample_rate)  # 16000
    if hop_samples is None:
        hop_samples = window_samples // 2
    # ... existing assert + label cache ...

    if per_file_lengths is None:
        # Legacy uniform-length DADS path
        per_file_lengths = [self._assumed_clip_samples] * len(file_indices)

    self._items: list[tuple[int, int]] = []
    self._labels_cache: list[int] = []
    for file_idx, clip_len in zip(file_indices, per_file_lengths, strict=True):
        num_w = max(1, 1 + max(0, (clip_len - window_samples)) // hop_samples)
        label_int = int(all_labels[file_idx])
        for w in range(num_w):
            self._items.append((int(file_idx), w * hop_samples))
            self._labels_cache.append(label_int)
```

### Runtime WARN in EfficientATClassifier (adapted from `classifier.py:41-58`)

```python
import logging
_logger = logging.getLogger(__name__)

class EfficientATClassifier:
    def __init__(self, model, mel_config=None):
        # ... existing init ...
        self._warned_mismatch = False
        self._expected_samples = self._cfg.segment_samples  # 32000

    def predict(self, features: torch.Tensor) -> float:
        with torch.no_grad():
            x = features
            if x.dim() == 1:
                x = x.unsqueeze(0)
            actual = int(x.shape[-1])
            if actual != self._expected_samples and not self._warned_mismatch:
                _logger.warning(
                    "EfficientAT input length %d != expected %d (%.3fs vs %.3fs) — "
                    "v7 regression signature. Check DetectionSession.window_seconds.",
                    actual, self._expected_samples,
                    actual / 32000, self._expected_samples / 32000,
                )
                self._warned_mismatch = True
            mel = self._mel(x)
            mel = mel.unsqueeze(1)
            logits, _ = self._model(mel)
            return torch.sigmoid(logits).item()
```

---

## State of the Art

| Old Approach (v7) | Current Approach (v8) | When Changed | Impact |
|---|---|---|---|
| Hard-coded `int(0.5 * _SOURCE_SR)` in trainer | `source_window_samples(_SOURCE_SR)` from contract module | Phase 22 | Can no longer drift from inference |
| Training RmsNormalize at 16 kHz | Training RmsNormalize at 32 kHz (post-resample) | Phase 22 | ~2% amplitude parity gap closed |
| No dataset-level shape assertion | Assert in `WindowedHFDroneDataset.__getitem__` after resample | Phase 22 | Fail-loud on first bad item |
| No runtime WARN in classifier | WARN when `features.shape[-1] != segment_samples` | Phase 22 | Operator-visible signal for any future drift |
| Train-from-AudioSet recipe (Phase 20) | Fine-tune from v6 (Phase 22) | Phase 22 | Faster convergence, lower risk of catastrophic regression |
| Vertex us-central1 L4 | Vertex us-east1 L4 | Phase 22 | Region change requires quota preflight |
| `WindowedHFDroneDataset` used for DADS | `_LazyEfficientATDataset` for DADS, generalized `WindowedHFDroneDataset` for field | Phase 22 | Sliding window only applies where it's meaningful |

### Deprecated in Phase 22

- The 0.5 s training window path — no retrievable value, was only an artifact of an author-time guess.
- `build_env_vars_v7` — stays for v7 rollback reproducibility, but Phase 22 writes `build_env_vars_v8`. Do not mutate v7.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|---|---|---|
| A1 | Catastrophic-forgetting risk is low when fine-tuning v6 with narrow Stage 1 | Focus 6 | If wrong, v8 could be worse than v6 on DADS — caught by the DADS accuracy threshold in Plan 20-06 |
| A2 | Artifact Registry is project-global and pulls cross-region from us-central1 → us-east1 workloads | Focus 9, Focus 10 | If wrong, Vertex job fails immediately on image pull — easy to detect and fix by replicating image to us-east1 repo |
| A3 | Fine-tuning from v6 improves TPR on held-out drone conditions the model has already seen similar examples of | Focus 6 | If wrong, Plan 20-06 gate blocks promotion; v6 stays as operational default |
| A4 | `ConcatDataset([dads, field]).labels` delegates correctly to `_build_weighted_sampler` | Focus 7, Focus 8 | If wrong, sampler crashes at construction — immediate failure, pre-training |
| A5 | Polyphase torchaudio resample preserves RMS to ~2% — small enough to ignore in training loss | Focus 3 | If wrong, post-resample RMS normalization still fixes the final amplitude; the fix is correct either way |
| A6 | v6 checkpoint is loadable as a state_dict with the existing `torch.load(..., weights_only=True)` path | Focus 6 | If wrong, the fine-tune-from-v6 code path needs a different loader. File is 17019638 bytes (same as v7) so likely the same format. |

---

## Open Questions

1. **Is `models/efficientat_mn10.pt` (dated Apr 5, size 17020041) actually v5 or something older — NOT v6?**
   - What we know: `models/efficientat_mn10_v6.pt` is a distinct file (17019638 bytes, Apr 6). `models/efficientat_mn10.pt` is a different size and predates v6.
   - What's unclear: which checkpoint the live service currently loads by default.
   - Recommendation: Before Phase 22 promotion, read the service's actual model-load code path and confirm which file it reads. If it's not v6, Phase 22 promotion is replacing the WRONG baseline.

2. **Does Artifact Registry support cross-region pull from us-central1 → us-east1 Vertex jobs without extra setup?**
   - Recommendation: `gcloud` dry-run on a small test image before Phase 22 real submission.

3. **Should we overfit-check v8 on the 2026-04-08 training fraction as a Wave 0 sanity test?**
   - Recommendation: yes — a quick 1-epoch smoke run on ONLY the 2026-04-08 training files (no DADS) should reach high accuracy on training. If it doesn't, something is wrong with the preprocessing pipeline before the real Vertex job burns hours.

4. **Is there an existing `.sha256` file or sidecar metadata for `efficientat_mn10_v6.pt`?**
   - VERIFIED: no sidecar JSON in `models/`. Only `efficientat_mn10_v6_onnx.sha256` exists (for the Phase 21 ONNX export, not the PT checkpoint).
   - Recommendation: compute and record the v6 PT sha256 in Phase 22's summary so fine-tune-from-v6 is reproducible.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|---|---|---|---|---|
| `soundfile` | Preflight (Focus 12) | ✓ | ≥0.13 per CLAUDE.md | — |
| `torchaudio` | Resample, augmentation | ✓ | ≥2.11 | — |
| `pyroomacoustics` | Room IR aug (carry-over) | ✓ | in `requirements-vertex.txt` | — |
| `audiomentations` | Pitch/stretch aug (carry-over) | ✓ | in `requirements-vertex.txt` | — |
| `sklearn` | ROC curve for Plan 20-06 artifacts | ✓ | existing | — |
| `google-cloud-aiplatform` | Vertex submission | ✓ | existing, used by `scripts/vertex_submit.py` | — |
| `gcloud` CLI | L4 quota preflight | [ASSUMED available on dev host, not verified in research — planner to confirm] | — | Run quota check manually in GCP console |
| `docker` | Base image rebuild | [ASSUMED] | — | Use Cloud Build instead |
| `EfficientAT` vendored pkg | Model architecture | ✓ | vendored at `src/acoustic/classification/efficientat/` | — |
| `data/field/drone/20260408_*.wav` + `data/field/background/20260408_*.wav` | Training + hold-out | ✓ | 17 files, 18.3 min total | — |
| `data/field/drone/20260408_091054_136dc5.wav.bak` | Not needed — trimmed file is authoritative | ✓ (present but unused) | 4569680 bytes | — |
| `data/eval/uma16_real/labels.json` | Pre-existing eval set (Phase 20) | ✓ | Feb-2026 take0740 recordings | Create a new `data/eval/uma16_real_v8/labels.json` for Phase 22 hold-out instead |

**Missing dependencies with no fallback:** none identified.

**Missing dependencies with fallback:** `gcloud` and `docker` availability not probed — planner to verify on the dev host before Wave 5 (Vertex submission).

---

## Validation Architecture

Nyquist validation is enabled (`.planning/config.json :: workflow.nyquist_validation = true`) — this section is required.

### Test Framework

| Property | Value |
|---|---|
| Framework | `pytest` (>=8.0) + `pytest-asyncio` (>=0.24) per CLAUDE.md stack |
| Config file | `pytest.ini` or `pyproject.toml` [VERIFIED: tests run via `pytest tests/...` in Phase 20 plans] |
| Quick run command | `pytest tests/unit/test_efficientat.py tests/unit/test_hf_dataset.py tests/unit/test_windowed_dataset_non_uniform.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |
| New test files created by Phase 22 | See "Wave 0 Gaps" below |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|---|---|---|---|---|
| P22-W1 | `source_window_samples(16000) == 16000` and `EFFICIENTAT_SEGMENT_SAMPLES == 32000` | unit | `pytest tests/unit/test_window_contract.py -x` | ❌ Wave 0 |
| P22-W1 | Import from contract module works from trainer AND pipeline without circular import | unit | `pytest tests/unit/test_window_contract.py::test_no_import_cycle -x` | ❌ Wave 0 |
| P22-W1 | `_training_window_seconds("efficientat") == EFFICIENTAT_WINDOW_SECONDS` (wiring check) | unit | `pytest tests/unit/test_pipeline_window.py::test_efficientat_uses_contract -x` | ❌ Wave 0 |
| P22-W2 | `WindowedHFDroneDataset.__getitem__` returns tensor of length `EFFICIENTAT_SEGMENT_SAMPLES` on synthetic 1s input | unit | `pytest tests/unit/test_hf_dataset.py::test_window_contract_post_resample -x` | ❌ Wave 0 (extend existing file) |
| P22-W2 | `WindowedHFDroneDataset.__getitem__` raises `AssertionError` when contract is violated (inject a broken window_samples) | unit | `pytest tests/unit/test_hf_dataset.py::test_length_assertion_catches_drift -x` | ❌ Wave 0 |
| P22-W3 | `EfficientATClassifier.predict` logs WARN when input length mismatches (caplog) | unit | `pytest tests/unit/test_efficientat.py::test_predict_warns_on_length_mismatch -x` | ❌ Wave 0 (extend existing file) |
| P22-W3 | `EfficientATClassifier.predict` does NOT warn when input matches contract | unit | `pytest tests/unit/test_efficientat.py::test_predict_silent_on_match -x` | ❌ Wave 0 |
| P22-W4 | Training sample RMS after full augmentation chain equals inference sample RMS within tolerance (domain parity) | unit | `pytest tests/unit/test_rms_parity.py::test_train_serve_rms_same_domain -x` | ❌ Wave 0 |
| P22-D1 | Field data manifest has 13 drone + 4 bg files at 16 kHz mono | integration | `pytest tests/integration/test_field_data_preflight.py -x` | ❌ Wave 0 |
| P22-D2 | Training dataset constructed with hold-out exclusion does NOT contain any of the 5 hold-out filenames | unit | `pytest tests/unit/test_holdout_split.py -x` | ❌ Wave 0 |
| P22-D3 | Preflight fails loudly on a fixture with wrong SR / NaN / missing file | unit | `pytest tests/unit/test_preflight.py -x` | ❌ Wave 0 |
| P22-G1 | Evaluator on synthetic 2-class eval set returns ROC curve + TPR/FPR | unit | `pytest tests/unit/test_evaluator.py -x` (already exists per Plan 20-06) | ❌ Wave 0 (create, since Plan 20-06 was deferred) |
| P22-G1 | `promote_if_gates_pass` returns False on TPR=0.75, True on TPR=0.85 with FPR=0.04 | unit | `pytest tests/unit/test_promotion_gate.py -x` | ❌ Wave 0 |
| P22-G1 | End-to-end: eval harness on fixture hold-out produces JSON report with 4 fields | integration | `pytest tests/integration/test_eval_harness_e2e.py -x` | ❌ Wave 0 |
| P22-G2 | v8 promotion replaces `models/efficientat_mn10.pt` only when gate passes | integration (uses tmp_path) | `pytest tests/integration/test_v8_promotion.py -x` | ❌ Wave 0 |
| Full training smoke | Synthetic 1-epoch training on random data + 17 synthetic field files runs end-to-end without crashing | integration | `pytest tests/integration/test_efficientat_train_smoke.py -x` | ❌ Wave 0 (optional — heavy) |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/test_window_contract.py tests/unit/test_hf_dataset.py tests/unit/test_efficientat.py -x -q` (< 10 s)
- **Per wave merge:** `pytest tests/unit/ tests/integration/ -x -q` (< 2 min)
- **Phase gate:** Full suite green + Plan 20-06 CLI exits 0 against real v8 checkpoint on real hold-out

### Wave 0 Gaps

- [ ] `tests/unit/test_window_contract.py` — new, covers P22-W1
- [ ] `tests/unit/test_pipeline_window.py` — new, wiring check for pipeline using contract
- [ ] `tests/unit/test_hf_dataset.py` — EXTEND existing file with P22-W2 tests (length assertion + post-resample shape)
- [ ] `tests/unit/test_efficientat.py` — EXTEND existing file with P22-W3 tests (WARN on mismatch, silent on match)
- [ ] `tests/unit/test_rms_parity.py` — new, P22-W4 domain parity
- [ ] `tests/unit/test_holdout_split.py` — new, P22-D2
- [ ] `tests/unit/test_preflight.py` — new, P22-D3
- [ ] `tests/integration/test_field_data_preflight.py` — new, P22-D1 (reads real files)
- [ ] `tests/unit/test_evaluator.py` — new (or verify doesn't exist), Plan 20-06 was deferred so this file likely does not exist
- [ ] `tests/unit/test_promotion_gate.py` — new, same reason
- [ ] `tests/integration/test_eval_harness_e2e.py` — new
- [ ] `tests/integration/test_v8_promotion.py` — new
- [ ] (optional) `tests/integration/test_efficientat_train_smoke.py` — slow, opt-in

No test framework install needed — pytest is already present [INFERRED from Phase 20 plans running `pytest tests/...`].

---

## Security Domain

`security_enforcement` is absent from `.planning/config.json` — treat as enabled. This is a training and model-promotion phase, so the security surface is narrow.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---|---|---|
| V2 Authentication | no | N/A — no new auth surface |
| V3 Session Management | no | N/A |
| V4 Access Control | partial | GCS bucket `gs://<GCP_BUCKET>/models/vertex/efficientat_mn10_v8/` must inherit Phase 20 IAM — no new roles, existing project-level access |
| V5 Input Validation | yes | Data integrity preflight (Focus 12) — validates file existence, SR, decode success, NaN |
| V6 Cryptography | yes | sha256 verification on v8 checkpoint before promotion (carry forward from Plan 20-06 pattern) |
| V9 Communications | partial | Vertex uses TLS by default; no custom transport |
| V10 Malicious Code | partial | `torch.load(..., weights_only=True)` already used [VERIFIED: `efficientat_trainer.py:533`] — prevents arbitrary code execution on checkpoint load |
| V14 Configuration | yes | Vertex env vars carry training config — treat as configuration data, no secrets embedded |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---|---|---|
| Malicious pickle in checkpoint | Tampering / RCE | `torch.load(weights_only=True)` — already in place at `efficientat_trainer.py:533` |
| Checkpoint swap between training and promotion (wrong file promoted) | Tampering | sha256 verification in `promote_if_gates_pass` — carry from Plan 20-06 |
| Data poisoning via contaminated field recording | Integrity | Hold-out isolation (Focus 5) prevents training contamination from leaking into the gate metric |
| Training config drift (wrong env var → wrong model) | Tampering / Repudiation | Log ALL `ACOUSTIC_TRAINING_*` env vars on job start (already in `scripts/vertex_train.py:157-159`) and include in Phase 22 summary file |
| Promotion bypass (manual cp without gate) | Elevation of privilege | Gate is advisory (`cp` is always available to the operator) but auditable — document in plan that manual cp is forbidden without a written exception |
| Cross-region Artifact Registry IAM misconfiguration | Denial of Service | Dry-run pull before submission (Focus 10) |

No cryptographic primitives hand-rolled. No new auth surfaces. No new secrets.

---

## Sources

### Primary (HIGH confidence — all verified by file read this session)

- `.planning/phases/22-.../CONTEXT.md` — locked constraints
- `.planning/REQUIREMENTS.md` — project REQ-IDs
- `.planning/STATE.md` — recent decisions D-31 through D-34, Phase 20/21 status, sliding-window sampler fixes
- `.planning/debug/efficientat-v7-regression-vs-v6.md` — root cause analysis with specific line numbers and evidence
- `.planning/phases/20-.../20-06-eval-harness-and-promotion-gate-PLAN.md` — eval harness architecture (deferred, must be re-executed generically in Phase 22)
- `src/acoustic/training/efficientat_trainer.py` — full read (733 lines); confirmed `:456` window literal and `:443-496` Phase 20 branch
- `src/acoustic/training/hf_dataset.py` — full read; `WindowedHFDroneDataset :185-329`, `_assumed_clip_samples` logic at `:251-299`, resample at `:316`
- `src/acoustic/pipeline.py:60-160` and `:285-344` — session and `_training_window_seconds`
- `src/acoustic/classification/efficientat/config.py` — full read; `segment_samples = input_dim_t * hop_size = 32000`
- `src/acoustic/classification/efficientat/classifier.py` — full read; `predict` signature at `:41-58`
- `src/acoustic/classification/preprocessing.py:195-214` — inference RmsNormalize at 32 kHz
- `src/acoustic/training/augmentation.py:511-546` — RmsNormalize + ComposedAugmentation pickling
- `src/acoustic/training/config.py` (excerpts) — `window_overlap_ratio`, `rir_enabled` defaults
- `scripts/vertex_submit.py` — `check_l4_quota`, `build_env_vars_v7`, `submit_v7_job`
- `Dockerfile.vertex-base` — full read; bakes `data/noise/` + `data/field/uma16_ambient/`
- Filesystem probe of `data/field/drone/20260408_*.wav` + `background/20260408_*.wav` via `soundfile.info`
- Filesystem probe of `models/*.pt` — v6/v7 checkpoint sizes and dates
- Filesystem probe of `data/eval/uma16_real/labels.json` — existing eval set from Phase 20
- `.planning/config.json` — workflow.nyquist_validation=true, commit_docs=true

### Secondary (MEDIUM confidence)

- python-soundfile docs — `sf.info` returns `samplerate`, `frames`, `channels`, `duration` [CITED: python-soundfile docs, used in practice in existing code]
- sklearn.metrics.roc_curve — canonical ROC implementation, already referenced by Plan 20-06

### Tertiary (LOW confidence — assumptions flagged in Assumptions Log)

- Cross-region Artifact Registry pull behavior (A2)
- Catastrophic forgetting risk in narrow-Stage-1 fine-tuning (A1)
- Expected TPR improvement magnitude (A3)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps, all existing code verified by file read
- Architecture (contract module): HIGH — pattern is idiomatic Python, verified no import cycle possible from `efficientat/` package → `pipeline.py`
- Window-length refactor call sites: HIGH — exhaustive grep over `src/acoustic/` + manual verification of each hit
- Hold-out split: MEDIUM — specific file selection is judgment-based on 17 files; defensible but not empirically validated
- Fine-tune-from-v6 vs from-AudioSet: MEDIUM — recommendation based on transfer-learning heuristics, not measured
- Plan 20-06 readiness: HIGH — full file read confirmed it was deferred and is v7-specific
- Data integrity preflight: HIGH — concrete code using existing libraries
- Vertex base image rebuild: HIGH — Dockerfile content verified; us-east1 L4 quota is an unknown (planner to check)
- Validation architecture: HIGH — framework exists, most tests are small targeted units

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (30 days — the codebase is in active Phase 20/21/22 work so re-verify before any future re-use)
