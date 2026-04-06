# Phase 20: Retrain v7 with Wide Gain + Room-IR Augmentation, Vertex Remote - Research

**Researched:** 2026-04-06
**Domain:** Audio ML data augmentation, sliding-window dataset enumeration with session-level splits, procedural RIR generation, Vertex AI custom training, deployment-matched eval
**Confidence:** HIGH (most decisions are LOCKED in CONTEXT.md; research focused on implementation specifics)

## Summary

Phase 20 retrains EfficientAT mn10 to close the deployment gap documented in `.planning/debug/uma16-no-detections.md`: real UMA-16v2 ambient mono RMS is ~8e-5 (-82 dBFS), while DADS training data sits ~50-60 dB above that. Training v6 never saw input that quiet, so the silence-gated CNN never produces probabilities on live captures. Phase 20 fixes the data distribution by adding (a) ±40 dB random gain, (b) procedural pyroomacoustics RIRs, (c) UMA-16 ambient noise mixing, (d) sliding-window enumeration with session-level split preservation, and (e) an expanded BG-noise negative pool. Training runs remotely on Vertex AI L4 via the existing `vertex_submit.py` infrastructure.

The CONTEXT.md file LOCKS almost every architectural decision (D-01 through D-29). This research focuses on implementation specifics within Claude's discretion areas: pyroomacoustics ShoeBox API, FFT convolution path, FSD50K subset acquisition, sliding-window indexing pitfalls, Vertex L4 / Docker COPY layer mechanics, and the validation harness extension.

**Primary recommendation:** Implement a `RoomIRAugmentation` class that pre-generates a pool of 500 RIRs at startup (CPU-cheap, ~2-3s), uses `scipy.signal.fftconvolve` for the per-sample convolution (matches the existing numpy DataLoader path), sits between `WideGainAugmentation` and `BackgroundNoiseMixer` in `ComposedAugmentation`. Use file-index-based session splits before expanding to window indices. Bundle FSD50K and UMA-16 ambient into the Docker image (small enough — see Docker Layer Strategy) so Vertex jobs need no additional GCS data plumbing.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Gain Augmentation:**
- D-01: Replace `waveform_gain_db=6.0` with wide-range gain ±40 dB, uniform per-sample. New config key `wide_gain_db: float = 40.0`.
- D-02: Apply wide gain BEFORE BG noise mixing AND BEFORE RIR. Clip to [-1, 1] post-mix.
- D-03: Justified by debug evidence (UMA-16 ~-82 dBFS vs DADS ~-20 to -30 dBFS, ~50-60 dB gap).
- D-04: Keep existing audiomentations Gain ±6 dB as small-jitter; wide gain is a separate pre-stage.

**Room IR (RIR):**
- D-05: pyroomacoustics procedural RIR (NOT real-recorded IR datasets). Add `pyroomacoustics>=0.7`.
- D-06: Per-sample random rooms: 3×3×2.5 m to 12×12×4 m, absorption 0.2-0.7, source-mic distance 1-8 m.
- D-07: New class `RoomIRAugmentation` in `augmentation.py`. Order: AFTER wide gain, BEFORE BG noise. `rir_probability=0.7`.
- D-08: RIR DISABLED on val/test splits.

**UMA-16 Ambient:**
- D-09: Collect ≥30 min real UMA-16 ambient (indoor quiet/HVAC, outdoor quiet/wind). Mono 16 kHz WAV in `data/field/uma16_ambient/`.
- D-10: Wire via existing `BackgroundNoiseMixer` — add to `noise_dirs`.
- D-11: Dedicated SNR -5 to +15 dB for UMA-16 ambient.
- D-12: ~10% of negative mini-batch sourced from UMA-16 ambient as pure label=0 samples.

**60% Overlap Sliding Windows:**
- D-13: Replace random 0.5s segment with sliding windows, hop=40% of window length (0.2s for 0.5s window).
- D-14: Re-index dataset: `__len__` = total window count; `__getitem__` maps idx → (file_idx, window_offset).
- D-15: **Session-level split MANDATORY** — overlapping windows from same source file MUST go in same split. Update `parquet_dataset.split_indices` to operate on file indices.
- D-16: Overlap on train+val ONLY. Test = non-overlapping (hop = window length).

**Expanded BG Noise:**
- D-17: Keep ESC-50 + UrbanSound8K.
- D-18: Add FSD50K subset: Wind, Rain, Traffic_noise_and_roadway_noise, Mechanical_fan, Engine, Bird → `data/noise/fsd50k_subset/`.
- D-19: Add DroneAudioSet non-drone clips IF freely available (see D-19 discussion below).
- D-20: `noise_dirs` auto-populates with `[esc50, urbansound8k, fsd50k_subset, uma16_ambient]`.

**Vertex Remote Training:**
- D-21: Use existing `vertex_submit.py` + `vertex_train.py` + `Dockerfile.vertex`. NOT local.
- D-22: Machine `g2-standard-8` + NVIDIA L4 (1 GPU). Fallback `NVIDIA_TESLA_T4`.
- D-23: Three-stage recipe: stage1=10 @ 1e-3, stage2=15 @ 1e-4, stage3=20 @ 1e-5; batch=64; focal(α=0.25, γ=2.0); patience=7. Total ≤45 epochs.
- D-24: HF source unchanged (`geronimobasso/drone-audio-detection-samples`). Noise/RIR/UMA-16 ambient bundled into Docker image at `/app/data/noise/` and `/app/data/field/uma16_ambient/`.
- D-25: Output `gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/best_model.pt`. Download to `models/efficientat_mn10_v7.pt`.

**Eval / Promotion:**
- D-26: v7 must keep ≥95% accuracy on DADS test split.
- D-27: TPR ≥ 0.80 AND FPR ≤ 0.05 on real-capture eval set ≥20 min UMA-16 (≥5 min drone, ≥15 min ambient) at `data/eval/uma16_real/labels.json`.
- D-28: Extend Phase 9 evaluator to accept the UMA-16 eval set; output confusion matrix + ROC curve alongside checkpoint.
- D-29: Promote v7 → `models/efficientat_mn10.pt` ONLY if both D-26 AND D-27 pass.

### Claude's Discretion

- pyroomacoustics room shape sampler (uniform vs log-uniform).
- RIR caching (precompute pool of 500 vs per-sample generation).
- `torchaudio.functional.fftconvolve` vs `scipy.signal.fftconvolve` for RIR convolution.
- FSD50K class-slug → filename glob mapping.
- DataLoader workers on Vertex L4 (4 or 8).
- Docker image layer caching for new noise/ambient directories.

### Deferred Ideas (OUT OF SCOPE)

- Real recorded RIR datasets (MIT, BUT Reverb, OpenAIR).
- EfficientAT architecture changes / ResNet-Mamba.
- ONNX/TFLite export of v7 (Phase 16).
- Hyperparameter sweeps.
- Online / continuous learning.
- Doppler / range targets.

## Project Constraints (from CLAUDE.md)

- Tech stack matches the project's STACK.md (PyTorch ≥2.11, torchaudio, audiomentations, soundfile, NumPy <3, scipy ≥1.14, pytest+pytest-asyncio, Ruff, mypy).
- Workflow enforcement: All edits must come through a GSD command (`/gsd:execute-phase`), not free-hand.
- Tests/lint must remain green; existing test pattern uses pytest classes per module (see `tests/unit/test_augmentation.py`).
- `nyquist_validation: true` in `.planning/config.json` → Validation Architecture section is REQUIRED.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| (none locked in roadmap) | Phase 20 was added late and has no formal REQ-IDs. Treat D-01..D-29 as the requirement set. Decision IDs from CONTEXT.md serve as traceable identifiers. | All sections below |

**Action for planner:** Before plan-write, the planner SHOULD propose adding REQ-IDs to REQUIREMENTS.md (e.g., `TRN-20` Wide-gain augmentation, `TRN-21` Procedural RIR, `TRN-22` Sliding-window enumeration, `TRN-23` UMA-16 ambient negatives, `TRN-24` Real-capture eval set, `TRN-25` v7 promotion gate) so traceability is preserved.

## Standard Stack

### New Dependencies
| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| pyroomacoustics | >=0.8,<0.11 | Procedural ShoeBox RIR generation | [VERIFIED: pyroomacoustics docs] — current docs reference v0.10.0; v0.7+ has stable ShoeBox+Material API |
| soundata | >=1.0,<2.0 | (Optional) FSD50K downloader with partial-download class filtering | [VERIFIED: soundata 1.0.1 docs — fsd50k loader supports `partial_download`] |

### Already Present (no changes)
| Library | Current | Purpose |
|---------|---------|---------|
| audiomentations | >=0.43,<1.0 | Existing PitchShift/TimeStretch/Gain Compose chain |
| torch / torchaudio | 2.5.1 (Vertex base image) | Trainer + mel preprocessing |
| scipy | >=1.14 | `scipy.signal.fftconvolve` for RIR convolution path |
| soundfile | >=0.13 | UMA-16 ambient WAV loading |
| datasets | >=3.0,<4.0 | HF DADS source unchanged |

**Installation (additions to `requirements-vertex.txt`):**
```
pyroomacoustics>=0.8,<0.11
# soundata is OPTIONAL — only needed for one-shot FSD50K download in a build script,
# not at training time. Prefer running soundata from a local helper script and bundling
# the resulting WAVs into the Docker image to avoid Vertex container bloat.
```

**Version verification:**
- `pip index versions pyroomacoustics` SHOULD be run before locking. As of search date the docs reference v0.10.0 (released Dec 2025) [CITED: pyroomacoustics.readthedocs.io/en/pypi-release/changelog.html]. Pin `>=0.8,<0.11` to allow patch updates without surprise major changes.
- `pip index versions soundata` — soundata 1.0.1 is current per [CITED: soundata.readthedocs.io 1.0.1 docs].

## Architecture Patterns

### Augmentation Chain Order (LOCKED by D-02, D-07)

```
raw_audio (16 kHz mono float32, ~16k samples for 1s)
  │
  ├─→ WideGainAugmentation         # ±40 dB uniform, NEW (D-01)
  │
  ├─→ RoomIRAugmentation           # pyroomacoustics ShoeBox, p=0.7, NEW (D-05..D-08)
  │
  ├─→ AudiomentationsAugmentation  # PitchShift + TimeStretch + Gain ±6 (existing, kept per D-04)
  │
  └─→ BackgroundNoiseMixer         # ESC-50 + UrbanSound8K + FSD50K + UMA-16 ambient (existing, expanded D-10/D-18)
       └─→ clip to [-1, 1]
```

This order is justified because (a) wide gain must affect the *clean* signal so the RIR and noise paths see realistic levels, (b) RIR must precede environmental noise so reverb is applied to the source not the noise, (c) audiomentations pitch/stretch modifies the dry source, (d) noise mixing is the final stage to model the deployment microphone's additive noise floor. This matches the canonical order in compass doc §4 ("Augmentation pipeline ordered by impact"). [CITED: docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md §4]

Wire via existing `ComposedAugmentation` (`augmentation.py:251`) — already pickle-safe for `num_workers > 0`.

### Sliding-Window Dataset (D-13..D-16)

**Window math (16 kHz, 0.5s window, 60% overlap = 40% hop):**
- `window_samples = 8000`
- `hop_samples = 3200` (0.2s)
- For an N-sample file: `num_windows = max(1, 1 + (N - window_samples) // hop_samples)` if `N >= window_samples`, else 1 (pad).

**Indexing structure (recommended):**
```python
class WindowedHFDroneDataset(Dataset):
    def __init__(self, hf_dataset, file_indices, window_samples, hop_samples,
                 mel_config, augmentation, ...):
        self._hf_ds = hf_dataset
        self._mel_config = mel_config
        self._aug = augmentation
        self._window_samples = window_samples
        self._hop_samples = hop_samples

        # Build flat (file_idx, window_offset) index list
        self._items: list[tuple[int, int]] = []
        self._labels_cache: list[int] = []
        all_labels = list(hf_dataset["label"])

        # We need source file lengths to enumerate windows.
        # OPTION A (preferred): use HF Arrow column 'num_samples' if present, else
        # OPTION B: probe row-by-row at init (one-time scan, ~minutes for 180k files,
        #          but cacheable to disk via .planning/cache/window_index.json)
        # OPTION C: assume fixed length per source clip (DADS clips are uniform 1s @ 16k = 16000 samples).
        #          → DADS-specific shortcut: 1 file = (16000-8000)//3200 + 1 = 3 windows.
        for file_idx in file_indices:
            n_samples = self._infer_samples(file_idx)  # see options above
            num_w = self._num_windows(n_samples)
            for w in range(num_w):
                self._items.append((file_idx, w * hop_samples))
                self._labels_cache.append(all_labels[file_idx])

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        file_idx, offset = self._items[idx]
        audio = decode_wav_bytes(self._hf_ds[file_idx]["audio"]["bytes"])
        segment = audio[offset : offset + self._window_samples]
        if len(segment) < self._window_samples:
            segment = pad_or_loop(segment, self._window_samples)
        if self._aug:
            segment = self._aug(segment)
        ...
```

**Session-level split (D-15) — CORRECT pattern:**

```python
def split_file_indices(num_files: int, seed=42, train=0.70, val=0.15):
    """Operate on FILE indices, not window indices, to prevent leakage."""
    files = list(range(num_files))
    random.Random(seed).shuffle(files)
    n_tr = int(num_files * train)
    n_va = int(num_files * val)
    return files[:n_tr], files[n_tr:n_tr+n_va], files[n_tr+n_va:]

# Usage:
train_files, val_files, test_files = split_file_indices(builder.total_rows)

train_ds = WindowedHFDroneDataset(hf, train_files, hop_samples=3200, ...)  # 60% overlap
val_ds   = WindowedHFDroneDataset(hf, val_files,   hop_samples=3200, ...)  # 60% overlap (D-16)
test_ds  = WindowedHFDroneDataset(hf, test_files,  hop_samples=8000, ...)  # NO overlap (D-16)
```

**Anti-pattern (forbidden by D-15):** splitting flat window indices after enumeration. This causes adjacent overlapping windows from the same source file to land in different splits → ~10-20% inflated val/test metrics. [CITED: compass doc §4 "Data splitting: session-level grouping is non-negotiable"; Plötz 2021; Kapoor & Narayanan 2023]

### Procedural RIR Generation Pattern

```python
import pyroomacoustics as pra
import numpy as np
from scipy.signal import fftconvolve

class RoomIRAugmentation:
    """Procedural ShoeBox RIR convolution (D-05..D-08).

    Pre-generates a pool of N RIRs at construction. Each __call__ samples one
    from the pool and convolves with the input audio. Faster than per-call
    generation (~5-15 ms per ShoeBox simulation) and removes pyroomacoustics
    from the per-batch hot path.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pool_size: int = 500,
        room_dim_min: tuple[float, float, float] = (3.0, 3.0, 2.5),
        room_dim_max: tuple[float, float, float] = (12.0, 12.0, 4.0),
        absorption_range: tuple[float, float] = (0.2, 0.7),
        source_distance_range: tuple[float, float] = (1.0, 8.0),
        max_order: int = 10,
        p: float = 0.7,
        seed: int = 42,
    ) -> None:
        self._sr = sample_rate
        self._p = p
        self._rng = np.random.default_rng(seed)
        self._pool: list[np.ndarray] = []
        for _ in range(pool_size):
            self._pool.append(self._generate_one(room_dim_min, room_dim_max,
                                                  absorption_range, source_distance_range,
                                                  max_order))
        # Worker-RNG for __call__ (re-seeded per worker via worker_init_fn)
        self._call_rng = np.random.default_rng()

    def _generate_one(self, dmin, dmax, absorption_range, dist_range, max_order):
        # Sample room dims uniform; sample absorption uniform; place mic and source
        # such that source-mic distance falls in dist_range AND both lie inside the room.
        room_dim = self._rng.uniform(dmin, dmax)               # (3,)
        absorption = self._rng.uniform(*absorption_range)
        room = pra.ShoeBox(
            room_dim.tolist(),
            fs=self._sr,
            materials=pra.Material(absorption),
            max_order=max_order,
        )
        # Mic placed near room center
        mic_pos = room_dim / 2.0
        # Source placed at random direction at sampled distance
        for _attempt in range(8):
            dist = self._rng.uniform(*dist_range)
            theta = self._rng.uniform(0, 2 * np.pi)
            phi = self._rng.uniform(np.pi / 4, 3 * np.pi / 4)  # mostly horizontal
            offset = np.array([
                dist * np.sin(phi) * np.cos(theta),
                dist * np.sin(phi) * np.sin(theta),
                dist * np.cos(phi),
            ])
            src_pos = mic_pos + offset
            margin = 0.3
            if np.all(src_pos > margin) and np.all(src_pos < room_dim - margin):
                break
        else:
            src_pos = mic_pos + np.array([1.0, 0.0, 0.0])  # fallback
        room.add_source(src_pos.tolist())
        room.add_microphone(mic_pos.tolist())
        room.compute_rir()
        rir = room.rir[0][0].astype(np.float32)
        # Truncate long tails to keep convolution cost bounded
        max_len = self._sr  # 1 second of RIR is plenty for these absorption ranges
        if len(rir) > max_len:
            rir = rir[:max_len]
        return rir

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self._call_rng.random() >= self._p or not self._pool:
            return audio
        rir = self._pool[self._call_rng.integers(len(self._pool))]
        # full convolution then trim to original length
        out = fftconvolve(audio, rir, mode="full")[: len(audio)]
        # Re-normalize to preserve perceived level
        if np.abs(out).max() > 1e-8:
            out = out * (np.abs(audio).max() / np.abs(out).max())
        return out.astype(np.float32)
```

[VERIFIED: pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html — confirms `ShoeBox(p, fs, max_order, materials=pra.Material(absorption))`, `add_source(position)`, `add_microphone(position)`, `compute_rir()`, `room.rir[mic_idx][src_idx]` numpy array access]

### WideGainAugmentation Pattern

```python
class WideGainAugmentation:
    """Wide ±wide_gain_db uniform gain (D-01..D-03).

    Replaces the WaveformAugmentation gain stage. Runs as a separate
    pre-stage in the ComposedAugmentation chain.
    """
    def __init__(self, wide_gain_db: float = 40.0, p: float = 1.0) -> None:
        self._wide_gain_db = wide_gain_db
        self._p = p
        self._rng = np.random.default_rng()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self._p:
            return audio
        gain_db = self._rng.uniform(-self._wide_gain_db, self._wide_gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        out = (audio * gain_linear).astype(np.float32)
        # Soft-clip catastrophic gain — float32 won't overflow but downstream
        # clips to [-1, 1] anyway. Hard-clip here so RIR convolution sees a
        # bounded signal.
        return np.clip(out, -1.0, 1.0)
```

### TrainingConfig Additions

```python
# In src/acoustic/training/config.py — append to TrainingConfig

# Wide-gain augmentation (D-01)
wide_gain_db: float = 40.0
wide_gain_probability: float = 1.0  # always-on per D-02

# Procedural RIR (D-05..D-08)
rir_enabled: bool = False           # default off; enable for v7 run
rir_probability: float = 0.7
rir_pool_size: int = 500
rir_room_dim_min: list[float] = [3.0, 3.0, 2.5]
rir_room_dim_max: list[float] = [12.0, 12.0, 4.0]
rir_absorption_min: float = 0.2
rir_absorption_max: float = 0.7
rir_source_distance_min: float = 1.0
rir_source_distance_max: float = 8.0
rir_max_order: int = 10

# Sliding window enumeration (D-13..D-16)
window_overlap_ratio: float = 0.6   # 60% overlap → hop = 40% of window
window_overlap_test: float = 0.0    # test set non-overlapping (D-16)

# UMA-16 ambient SNR (D-11)
uma16_ambient_snr_low: float = -5.0
uma16_ambient_snr_high: float = 15.0
uma16_ambient_pure_negative_ratio: float = 0.10  # D-12
```

### Anti-Patterns to Avoid

- **Per-sample pyroomacoustics generation in `__getitem__`.** ShoeBox + image source method is ~5-15 ms per call; with `num_workers=4` and batch=64 you'd add ~200-300 ms per batch. Use the precomputed pool pattern.
- **`torchaudio.functional.fftconvolve` inside numpy DataLoader workers.** It works, but forces an extra `numpy → tensor → numpy` round trip per sample. The DataLoader path is numpy-native; stay in numpy with `scipy.signal.fftconvolve`. Reserve `torchaudio.functional.fftconvolve` for if/when you move RIR to a GPU collate hook (out of scope for Phase 20).
- **Splitting after window enumeration.** See D-15 — file-index split first, then expand.
- **Saving RIR pool to disk and reloading.** RIR generation for 500 rooms takes ~3-7 seconds at startup — cheaper than I/O serialization and avoids cache invalidation bugs.
- **Forgetting `worker_init_fn` to reseed RNGs per DataLoader worker.** Without it, all workers generate identical augmentation sequences. Pattern: `def _seed_worker(worker_id): np.random.seed(torch.initial_seed() % 2**32)` → pass to `DataLoader(worker_init_fn=_seed_worker)`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Room impulse response generation | Custom image source method | `pyroomacoustics.ShoeBox` + `compute_rir()` | Image source method is well-tested, handles 3D geometry, materials, scattering — implementing it correctly is a multi-week project |
| FFT convolution | `numpy.convolve` (O(N²)) | `scipy.signal.fftconvolve` (O(N log N)) | For 8000-sample audio × 16000-sample RIR, fftconvolve is 50-100× faster |
| Class-balanced sampling | Custom batch logic | Existing `WeightedRandomSampler` (already in `efficientat_trainer.py:298`) | Trainer already does this |
| FSD50K download with class filter | Wget loops + manual class mapping | `soundata` partial_download | Handles 200-class vocabulary, splits, integrity checks |
| Audio resampling | Scipy signal resample | `torchaudio.functional.resample` (already used) | GPU-aware, faster, deterministic |
| ROC curve / confusion matrix | Custom metric code | `sklearn.metrics.roc_curve` / existing `Evaluator._compute_confusion` | Existing harness already has confusion + precision/recall — only need ROC addition |

**Key insight:** Phase 20 is ~95% configuration and orchestration on top of existing primitives (BackgroundNoiseMixer, ComposedAugmentation, EfficientATTrainingRunner, vertex_submit.py, Evaluator). The two genuinely new pieces are `WideGainAugmentation` (~30 LOC) and `RoomIRAugmentation` (~100 LOC). Everything else is config plumbing, data acquisition, and a windowed dataset variant.

## Runtime State Inventory

> Phase 20 is a training/data-pipeline phase. There are no renames, refactors, OS-registered tasks, or stored databases to migrate. Filling this in for completeness:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no database keys, collection names, or cached IDs reference v6/v7 by name. The active model is selected by file path in `AcousticSettings.cnn_model_path`, not a stored ID. | None — code edit only (D-29 promotion is `cp` of `efficientat_mn10_v7.pt` → `efficientat_mn10.pt`) |
| Live service config | None — no n8n / Datadog / Tailscale resources tied to model versioning | None |
| OS-registered state | None — no Task Scheduler / launchd / systemd units reference model versions | None |
| Secrets/env vars | `ACOUSTIC_TRAINING_*` env vars are read by `TrainingConfig` pydantic. New keys (D-01..D-20 implementations) must propagate through `vertex_submit.py` `env_vars` dict to `vertex_train.py`. | Code edit: extend `env_vars` in `vertex_submit.py:submit_job` to include `ACOUSTIC_TRAINING_RIR_ENABLED`, `_WIDE_GAIN_DB`, `_NOISE_DIRS` (comma-joined → parse in pydantic), `_WINDOW_OVERLAP_RATIO`, etc. |
| Build artifacts / installed packages | Docker image `gcr.io/interception-dashboard/acoustic-trainer:latest` is the prior v6 build. New training run requires a fresh build that COPIes `data/noise/` and `data/field/uma16_ambient/`. | New `docker build` per Phase 20. Tag `:phase20` for traceability per D-25. |

## Common Pitfalls

### Pitfall 1: Window-Index Split Leakage (HIGH severity)

**What goes wrong:** Splits operate on flat window indices instead of file indices. Adjacent overlapping windows from the same source recording land in different splits → val/test metrics inflate by 10-20%.

**Why it happens:** It's the simpler code path and "looks reproducible" with a seeded shuffle.

**How to avoid:** D-15 mandates file-index split first, window expansion second. Validation: write a test that asserts no `(file_idx, window_offset_a)` appears in train AND `(file_idx, window_offset_b)` appears in val for any shared `file_idx`.

**Warning signs:** v7 val accuracy > 99% on DADS but <60% on the real UMA-16 eval set is the smoking gun.

[CITED: compass doc §4 "session-level grouping is non-negotiable"; Plötz 2021]

### Pitfall 2: Wide-Gain Driving the Signal Outside [-1, 1] Before RIR

**What goes wrong:** WideGainAug emits signal at +40 dB (×100 amplitude). RIR convolution then sees a clipped signal with massive harmonic distortion that doesn't represent real-world data.

**Why it happens:** The clip-to-[-1,1] step is forgotten between stages, or done only at the very end.

**How to avoid:** WideGainAugmentation MUST clip to [-1, 1] before returning. RoomIRAugmentation MUST renormalize after convolution (the example above does this via `out * (peak_in / peak_out)`).

**Warning signs:** Audible distortion in a unit-test playback of an augmented sample; spectral histograms show peaks pinned at ±1.

### Pitfall 3: pyroomacoustics ShoeBox with `max_order` Too High

**What goes wrong:** `max_order=30` makes a single `compute_rir()` take 200 ms+ and produces a 5-second RIR tail that costs more in convolution than it benefits the model.

**Why it happens:** Default examples use `max_order=15-17`. For drone classification we don't need that fidelity.

**How to avoid:** `max_order=10` is sufficient for the absorption range 0.2-0.7. Truncate the RIR to ≤1 second post-generation.

**Warning signs:** RIR pool generation takes >30 seconds at training start; convolution time per sample exceeds 5 ms.

### Pitfall 4: UMA-16 Ambient Without Per-Channel Sanity Check

**What goes wrong:** The collected `data/field/uma16_ambient/` files are 16-channel (UMA-16 native) instead of mono 16 kHz, breaking `BackgroundNoiseMixer.warm_cache()` which uses `audio.mean(axis=1)` (handles stereo) but the SNR math will be off vs DADS-scale signals.

**Why it happens:** Field recording UI defaults to 16-channel raw capture.

**How to avoid:** D-09 says "store as mono 16 kHz WAV". Preprocessing step in the collection task: `mono = capture[:, 0]` (or `mean(axis=1)` for diversity), `resample 48000 → 16000`, `sf.write(..., samplerate=16000)`. The existing `BackgroundNoiseMixer` already resamples and converts to mono, so a recipe error here is recoverable but slows down warm_cache.

### Pitfall 5: Vertex L4 Quota Denied → Job Stuck Pending

**What goes wrong:** L4 quota is region-specific and often denied for new projects. `aiplatform.CustomContainerTrainingJob.run(sync=False)` does NOT fail loudly — the job sits in PENDING state for hours.

**Why it happens:** L4 GPUs are popular and quota is per-region. `interception-dashboard` may not have it in `us-central1`.

**How to avoid:** D-22 explicitly mandates the fallback `NVIDIA_TESLA_T4`. Add a pre-flight quota check in `vertex_submit.py`:
```bash
gcloud compute regions describe us-central1 --project interception-dashboard \
  --format="value(quotas[].metric,quotas[].limit)" | grep -i nvidia_l4
```
Or simply submit with T4 as the default for v7 and only escalate to L4 if the user explicitly requests it. T4 trains EfficientAT mn10 in roughly 1.4-1.7× the time of L4 — acceptable for ≤45 epochs.

### Pitfall 6: Docker COPY Layer Bloat

**What goes wrong:** `COPY data/noise/ data/noise/` includes ESC-50 + UrbanSound8K + FSD50K subset → easily 5-10 GB → push to GCR takes 20+ minutes per build, kills iteration speed.

**Why it happens:** Naive COPY layers.

**How to avoid (Docker layer strategy):**
1. Build a separate **base image** `acoustic-trainer-base:v1` that COPIes the noise data once. Push once.
2. The phase-20 image FROM that base, only COPies source code on top. Each iteration push is <50 MB.
3. Alternatively: keep noise data in a GCS bucket, download in `vertex_train.py` startup (2-3 min one-time per job, but no image bloat). This trades image size for per-job startup cost — for a phase that may need 5-10 retries during tuning, the base-image strategy is better.

**Recommended:**
```dockerfile
# Dockerfile.vertex-base (built once, pushed once)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-vertex.txt .
RUN pip install --no-cache-dir -r requirements-vertex.txt
COPY data/noise /app/data/noise
COPY data/field/uma16_ambient /app/data/field/uma16_ambient

# Dockerfile.vertex (rebuilt per phase / per code change)
FROM us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v1
WORKDIR /app
COPY src/ src/
COPY scripts/vertex_train.py vertex_train.py
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "vertex_train.py"]
```

## Code Examples

### Wiring the Composed Augmentation in EfficientATTrainingRunner

```python
# In efficientat_trainer.py — replace the current data loading path

from acoustic.training.augmentation import (
    WideGainAugmentation, RoomIRAugmentation,
    AudiomentationsAugmentation, BackgroundNoiseMixer, ComposedAugmentation,
)

def _build_train_augmentation(self) -> ComposedAugmentation:
    cfg = self._config
    augs = []

    if cfg.wide_gain_db > 0:
        augs.append(WideGainAugmentation(
            wide_gain_db=cfg.wide_gain_db,
            p=cfg.wide_gain_probability,
        ))

    if cfg.rir_enabled:
        augs.append(RoomIRAugmentation(
            sample_rate=16000,  # source SR; resample happens later
            pool_size=cfg.rir_pool_size,
            room_dim_min=tuple(cfg.rir_room_dim_min),
            room_dim_max=tuple(cfg.rir_room_dim_max),
            absorption_range=(cfg.rir_absorption_min, cfg.rir_absorption_max),
            source_distance_range=(cfg.rir_source_distance_min, cfg.rir_source_distance_max),
            max_order=cfg.rir_max_order,
            p=cfg.rir_probability,
        ))

    if cfg.use_audiomentations:
        augs.append(AudiomentationsAugmentation(
            pitch_semitones=cfg.pitch_shift_semitones,
            time_stretch_range=(cfg.time_stretch_min, cfg.time_stretch_max),
            gain_db=cfg.waveform_gain_db,
            p=cfg.augmentation_probability,
        ))

    if cfg.noise_augmentation_enabled and cfg.noise_dirs:
        bg_mixer = BackgroundNoiseMixer(
            noise_dirs=[Path(d) for d in cfg.noise_dirs],
            snr_range=(cfg.noise_snr_range_low, cfg.noise_snr_range_high),
            sample_rate=16000,
            p=cfg.noise_probability,
        )
        bg_mixer.warm_cache()
        augs.append(bg_mixer)

    return ComposedAugmentation(augs)

# Note: this augmentation produces 16 kHz audio; the existing _LazyEfficientATDataset
# resamples to 32 kHz AFTER segment extraction. For Phase 20 we need to apply
# augmentation at 16 kHz BEFORE resampling, so the dataset class needs a small
# refactor: insert `if augmentation: segment = augmentation(segment)` between
# decode and resample.
```

### Vertex Submission for v7 Run

```python
# Augment vertex_submit.py — new env vars for Phase 20

env_vars = {
    "ACOUSTIC_TRAINING_DADS_HF_REPO": hf_repo,
    "ACOUSTIC_TRAINING_MODEL_TYPE": "efficientat_mn10",
    "ACOUSTIC_TRAINING_BATCH_SIZE": "64",
    "ACOUSTIC_TRAINING_LOSS_FUNCTION": "focal",
    "ACOUSTIC_TRAINING_FOCAL_ALPHA": "0.25",
    "ACOUSTIC_TRAINING_FOCAL_GAMMA": "2.0",
    "ACOUSTIC_TRAINING_PATIENCE": "7",
    "ACOUSTIC_TRAINING_STAGE1_EPOCHS": "10",
    "ACOUSTIC_TRAINING_STAGE2_EPOCHS": "15",
    "ACOUSTIC_TRAINING_STAGE3_EPOCHS": "20",
    "ACOUSTIC_TRAINING_STAGE1_LR": "1e-3",
    "ACOUSTIC_TRAINING_STAGE2_LR": "1e-4",
    "ACOUSTIC_TRAINING_STAGE3_LR": "1e-5",
    # Phase 20 additions:
    "ACOUSTIC_TRAINING_WIDE_GAIN_DB": "40.0",
    "ACOUSTIC_TRAINING_RIR_ENABLED": "true",
    "ACOUSTIC_TRAINING_RIR_PROBABILITY": "0.7",
    "ACOUSTIC_TRAINING_NOISE_AUGMENTATION_ENABLED": "true",
    "ACOUSTIC_TRAINING_NOISE_DIRS": "/app/data/noise/esc50,/app/data/noise/urbansound8k,/app/data/noise/fsd50k_subset,/app/data/field/uma16_ambient",
    "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO": "0.6",
    "ACOUSTIC_TRAINING_CHECKPOINT_PATH": "/tmp/models/efficientat_mn10_v7.pt",
    "ACOUSTIC_TRAINING_PRETRAINED_GCS": GCS_PRETRAINED,
}

# Note: TrainingConfig.noise_dirs is a `list[str]` in pydantic. To accept a
# comma-joined env var, either (a) override with a custom validator that splits
# on ',', or (b) use ACOUSTIC_TRAINING_NOISE_DIRS as a JSON string '["a","b"]'.
# pydantic-settings v2 supports JSON-encoded list env vars natively.
```

### Real-Capture Eval Set Loader (D-27, D-28)

```python
# New file: src/acoustic/evaluation/uma16_eval.py

import json
from pathlib import Path
import soundfile as sf
import numpy as np

def load_uma16_eval_set(eval_dir: Path) -> list[tuple[Path, int]]:
    """Load UMA-16 real-capture eval set from labels.json.

    Expected layout:
        data/eval/uma16_real/
            labels.json   # [{"file": "clip_001.wav", "label": "drone", "start_s": 0, "end_s": 5}, ...]
            clip_001.wav
            clip_002.wav

    Returns:
        List of (wav_path, label_int) where label is 1=drone, 0=no_drone.
    """
    labels_file = eval_dir / "labels.json"
    entries = json.loads(labels_file.read_text())
    out = []
    for entry in entries:
        path = eval_dir / entry["file"]
        label = 1 if entry["label"] == "drone" else 0
        out.append((path, label))
    return out
```

Then extend `Evaluator.evaluate_classifier` (or add `evaluate_uma16`) to consume this list and emit the same `EvaluationResult` shape, with one addition: an `roc_curve: list[tuple[float, float, float]]` field (threshold, fpr, tpr) computed via `sklearn.metrics.roc_curve` from per-segment scores.

## State of the Art

| Old Approach (v6) | Current Approach (v7) | Why Changed | Impact |
|-------------------|----------------------|-------------|--------|
| ±6 dB gain | ±40 dB wide gain | UMA-16 deployment evidence (debug doc) | Closes 50-60 dB level gap |
| Random 0.5s segment per `__getitem__` | Sliding 0.5s @ 60% overlap, deterministic | More samples per file, reproducible val | ~2.5× effective dataset size |
| ESC-50 + UrbanSound8K only | + FSD50K subset (Wind/Rain/Traffic/Fan/Engine/Bird) + UMA-16 ambient | Compass doc §1 BG noise list; deployment matching | Better real-world FPR |
| No reverb | Procedural pyroomacoustics RIR (p=0.7) | Compass doc §4 ranks RIR as top-3 augmentation | Generalizes to outdoor/indoor environments |
| Local training | Vertex AI L4 / T4 | Phase 14 already established Vertex pipeline | Frees local machine; standard cloud workflow |
| Random file-level split | File-index split → window expansion | Compass doc §4 + Plötz 2021 | Eliminates leakage |
| DADS-only eval | DADS + real UMA-16 eval set | Debug doc mandates real-capture validation | Catches deployment-distribution drift |

**Deprecated/replaced:**
- `WaveformAugmentation` (the gain ±6 path) — superseded by separate `WideGainAugmentation` + audiomentations `Gain`. Keep the class for backward compatibility but don't use in v7 chain.
- Random-segment `__getitem__` in `HFDroneDataset` — replaced by `WindowedHFDroneDataset` for v7. Keep old class for v5/v6 compatibility.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | DADS clips are uniform 1s @ 16 kHz (16000 samples) | Sliding-window indexing | If clips vary in length, the per-file `num_windows` calculation must probe each row at init time (~5-15 min one-time scan for 180k files). Build a `.planning/cache/dads_window_index.json` to avoid re-scanning. [ASSUMED — verify by reading 5-10 random clips from `geronimobasso/drone-audio-detection-samples`] |
| A2 | DroneAudioSet (ahlab-drone-project/DroneAudioSet on HF) provides downloadable non-drone clips | D-19 implementation | If unavailable or license-restricted, drop D-19 silently and rely on FSD50K + ESC-50 + UrbanSound8K + UMA-16 ambient. The compass doc references the GitHub repo; the actual HF dataset content was not directly verified in this research session. [ASSUMED] |
| A3 | Vertex `interception-dashboard` project has L4 quota in `us-central1` | D-22 / Pitfall 5 | If denied, fallback to T4 (already specified in D-22). T4 increases training wall time ~1.4-1.7×. [ASSUMED] |
| A4 | UMA-16 ambient collection at "indoor quiet, indoor HVAC, outdoor quiet, outdoor wind" produces a representative noise distribution for the deployment environment | D-09 | If the deployment site has dominant noise sources not in this set (e.g., highway, machinery, crowd), real-capture FPR will exceed 5%. Mitigation: D-27 eval set must come from the actual deployment site. [ASSUMED] |
| A5 | `pyroomacoustics>=0.8` ShoeBox + Material API is stable across 0.8/0.9/0.10 | Standard Stack | API has been stable since 0.5; risk is low. [ASSUMED based on docs spanning multiple versions] |
| A6 | The 0.5s window length is correct for v7 (matches v6 + EfficientAT mel preprocessing) | Sliding-window | The trainer resamples to 32 kHz before mel computation; segment_samples comes from `EfficientATMelConfig`. Verify this matches 0.5s @ 16 kHz source = 1s @ 32 kHz target. If EfficientAT expects 1s segments, the 60% overlap math still applies but window_samples differs. [ASSUMED — verify by reading `EfficientATMelConfig.segment_samples`] |
| A7 | RIR convolution in scipy.signal.fftconvolve at ~1-3 ms per call is fast enough for `num_workers=4-8` to keep up with L4 GPU | Convolution choice | If profiling shows DataLoader becomes the bottleneck, fall back to `torchaudio.functional.fftconvolve` on the GPU via a collate hook. [ASSUMED; profile in a Wave 0 smoke test] |
| A8 | "Measurable improvement" on D-27 means TPR≥0.80, FPR≤0.05 are achievable with these augmentations alone (no architecture change) | Promotion gate | If v7 fails D-27, the next move is more UMA-16 ambient data and/or a real-recorded RIR dataset (deferred). Phase 20 explicitly excludes architecture changes. [ASSUMED based on compass doc evidence that augmentation > architecture] |

**Mitigation:** All assumptions are tagged. Planner SHOULD include a Wave 0 smoke-test plan that verifies A1, A6, A7 in the first 30 minutes of work, and surfaces A2/A3 to the user for confirmation before the first Vertex submission.

## Open Questions

1. **Is DroneAudioSet's non-drone subset actually downloadable and license-compatible?**
   - What we know: Compass doc §1 references `augmented-human-lab/DroneAudioSet-code` on GitHub; HF dataset is at `ahlab-drone-project/DroneAudioSet`. Direct verification of non-drone clip availability was not possible in this session.
   - What's unclear: Whether the HF dataset exposes non-drone clips as a separate split or whether they're only embedded in mixed recordings.
   - Recommendation: Make D-19 OPTIONAL in the plan. If a 30-minute exploration spike doesn't surface a clean non-drone subset, drop it and rely on FSD50K + UMA-16 ambient. Don't block the phase on this.

2. **What window length should Phase 20 use — 0.5s @ 16 kHz source, or 1s @ 32 kHz post-resample (matching `EfficientATMelConfig`)?**
   - What we know: v6 used random 0.5s segments at 16 kHz source, then resampled to 32 kHz. EfficientAT itself takes ~1s @ 32 kHz.
   - What's unclear: Whether to enumerate windows in the source-rate (16 kHz) or target-rate (32 kHz) space. If at source rate and overlap is 60%, the post-resample windows still align.
   - Recommendation: Enumerate at SOURCE rate (16 kHz), apply augmentation in source rate, then resample inside the dataset class. This keeps `BackgroundNoiseMixer` (16 kHz cached noise) on the fast path.

3. **Should the new TrainingConfig fields (RIR, wide gain, window overlap) default to enabled or disabled?**
   - Recommendation: Default DISABLED (`rir_enabled: bool = False`, `window_overlap_ratio: float = 0.0`). This makes Phase 20 a config-driven opt-in and preserves v6 reproducibility. The Phase 20 Vertex submission explicitly sets the flags via env vars.

4. **Does the existing `BackgroundNoiseMixer.warm_cache()` scale to ~10k+ FSD50K clips (potentially >2 GB in RAM)?**
   - What we know: ESC-50 + UrbanSound8K is ~10k clips ~1.5 GB. Adding the FSD50K subset of 6 classes is ~3-5k more clips → another ~600 MB-1 GB.
   - What's unclear: Vertex L4 instance RAM is 32 GB (g2-standard-8). 2-3 GB cached noise + 4-5 GB DADS HF Arrow + model + activations should fit, but the margin shrinks.
   - Recommendation: Plan a Wave 0 step that profiles `warm_cache()` RAM consumption and short-circuits to lazy loading if it exceeds 4 GB. Lazy loading is a 30-line change to `BackgroundNoiseMixer`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pyroomacoustics | RoomIRAugmentation (D-05) | ✗ (not yet in requirements-vertex.txt) | — | None — must add |
| soundata | FSD50K download (D-18) | ✗ | — | Manual zip download from Zenodo + custom class filter script |
| scipy.signal.fftconvolve | RIR convolution | ✓ (scipy >=1.14 already in requirements-vertex.txt) | 1.14+ | torchaudio.functional.fftconvolve |
| audiomentations | Existing chain | ✓ | >=0.43,<1.0 | None needed |
| google-cloud-aiplatform | Vertex submission | ✓ | >=1.72 | None |
| Docker | Vertex image build | ✓ (assumed — Phase 14 already used it) | — | None |
| gcloud CLI authenticated | `docker push` to GCR | ✓ (assumed — Phase 14 worked) | — | None |
| L4 GPU quota in us-central1 | D-22 preferred path | ✗ (UNVERIFIED) | — | NVIDIA_TESLA_T4 (D-22 explicit fallback) |
| FSD50K dataset download (~24 GB full) | D-18 | ✗ | — | Manual via Zenodo: https://zenodo.org/records/4060432 |
| UMA-16v2 hardware for ambient capture | D-09 | ✓ (project has it; existing field-recording UI from Phase 10) | — | None — required to start phase |
| Drone audio source for real-capture eval (D-27) | Eval set | ✓ (UMA-16 + drone is the project's mission) | — | Recordings already collected during prior phases may be reusable |

**Missing dependencies with no fallback:**
- pyroomacoustics — must be added to `requirements-vertex.txt` and pip-installed in image rebuild. ~5 MB wheel + 30 MB scipy already present.
- UMA-16v2 ambient recordings — collection task is itself a phase deliverable (Wave 0 likely).

**Missing dependencies with fallback:**
- L4 quota → T4 (D-22).
- soundata → Manual Zenodo download script.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio (existing) |
| Config file | `pyproject.toml` (or `pytest.ini` if present) |
| Quick run command | `pytest tests/unit/test_augmentation.py tests/unit/test_efficientat_training.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map

| Decision | Behavior | Test Type | Automated Command | File Exists? |
|----------|----------|-----------|-------------------|-------------|
| D-01 | WideGainAug emits ±40 dB output, clipped to [-1,1] | unit | `pytest tests/unit/test_augmentation.py::TestWideGainAugmentation -x` | ❌ Wave 0 |
| D-02 | Augmentation chain order: WideGain → RIR → Audio → BG | unit | `pytest tests/unit/test_augmentation.py::TestComposedOrder -x` | ❌ Wave 0 |
| D-05 | RoomIRAug builds a pool, samples from it, convolves | unit | `pytest tests/unit/test_augmentation.py::TestRoomIRAugmentation -x` | ❌ Wave 0 |
| D-07 | RIR is applied with probability p, length preserved | unit | `pytest tests/unit/test_augmentation.py::TestRoomIRAugmentation::test_probability_and_length -x` | ❌ Wave 0 |
| D-08 | RIR DISABLED on val/test datasets | unit | `pytest tests/unit/test_efficientat_training.py::test_val_no_rir -x` | ❌ Wave 0 |
| D-10 | BackgroundNoiseMixer accepts uma16_ambient dir | unit | `pytest tests/unit/test_augmentation.py::TestBackgroundNoiseMixer::test_uma16_ambient_dir -x` | ❌ Wave 0 (extend existing test) |
| D-13 | Sliding window enumeration produces correct count for fixed-length file | unit | `pytest tests/unit/test_hf_dataset.py::test_window_count -x` | ❌ Wave 0 |
| D-14 | `__getitem__(idx)` maps to consistent (file_idx, offset) | unit | `pytest tests/unit/test_hf_dataset.py::test_idx_mapping -x` | ❌ Wave 0 |
| D-15 | **Session-level split: no file appears in two splits** | unit (CRITICAL) | `pytest tests/unit/test_hf_dataset.py::test_no_file_leakage -x` | ❌ Wave 0 |
| D-16 | Test split uses non-overlapping windows | unit | `pytest tests/unit/test_hf_dataset.py::test_test_split_no_overlap -x` | ❌ Wave 0 |
| D-21 | Vertex submission script accepts new env vars without crashing | unit | `pytest tests/unit/test_vertex_submit.py::test_env_var_propagation -x` | ❌ Wave 0 |
| D-23 | TrainingConfig loads new fields from env vars | unit | `pytest tests/unit/test_augmentation.py::TestTrainingConfig::test_phase20_fields -x` | ❌ Wave 0 (extend existing) |
| D-24 | Dockerfile.vertex COPies the new directories | smoke | `docker build -f Dockerfile.vertex -t test:phase20 . && docker run --rm test:phase20 ls /app/data/noise/fsd50k_subset` | ❌ Wave 0 (manual) |
| D-26 | DADS test split accuracy ≥0.95 | manual-only (full training run) | Cannot automate — gated by Vertex job completion | ❌ Phase Gate |
| D-27 | UMA-16 eval TPR≥0.80, FPR≤0.05 | manual-only (post-training) | `python -m acoustic.evaluation.uma16_eval --model models/efficientat_mn10_v7.pt --eval-dir data/eval/uma16_real/` | ❌ Phase Gate |
| D-28 | Eval harness emits ROC curve | unit | `pytest tests/unit/test_evaluator.py::test_uma16_eval_roc -x` | ❌ Wave 0 |
| D-29 | Promotion script copies v7 → default ONLY if both gates pass | unit | `pytest tests/unit/test_promotion.py::test_promotion_gate -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_augmentation.py tests/unit/test_hf_dataset.py -x` (~10 s)
- **Per wave merge:** `pytest tests/unit/ -x` (~30-60 s)
- **Phase gate:** Full suite `pytest tests/ -x` green + successful Vertex training run + D-26/D-27 eval pass before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_augmentation.py` — extend with `TestWideGainAugmentation`, `TestRoomIRAugmentation`, `TestComposedOrder`, `TestBackgroundNoiseMixer::test_uma16_ambient_dir`, `TestTrainingConfig::test_phase20_fields`
- [ ] `tests/unit/test_hf_dataset.py` — NEW file: `test_window_count`, `test_idx_mapping`, `test_no_file_leakage`, `test_test_split_no_overlap`
- [ ] `tests/unit/test_evaluator.py` — extend with `test_uma16_eval_roc`
- [ ] `tests/unit/test_vertex_submit.py` — NEW file: `test_env_var_propagation`
- [ ] `tests/unit/test_promotion.py` — NEW file: `test_promotion_gate`
- [ ] `tests/unit/test_efficientat_training.py` — extend with `test_val_no_rir`
- [ ] Framework install: none — pyroomacoustics added to `requirements-vertex.txt` AND root `requirements.txt` (or wherever local dev installs from)
- [ ] Data acquisition tasks (these are NOT tests, but Wave 0 prerequisites):
  - [ ] Collect ≥30 min UMA-16 ambient → `data/field/uma16_ambient/` (D-09)
  - [ ] Download FSD50K subset → `data/noise/fsd50k_subset/` (D-18)
  - [ ] Collect ≥20 min UMA-16 real-capture eval → `data/eval/uma16_real/` + `labels.json` (D-27)

## Sources

### Primary (HIGH confidence)
- [pyroomacoustics 0.10.0 Room module docs](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html) — ShoeBox API, Material, compute_rir, room.rir layout
- [pyroomacoustics 0.10.0 changelog](https://pyroomacoustics.readthedocs.io/en/pypi-release/changelog.html) — current version verification
- [pyroomacoustics on PyPI](https://pypi.org/project/pyroomacoustics/)
- [soundata 1.0.1 fsd50k loader](https://soundata.readthedocs.io/en/stable/_modules/soundata/datasets/fsd50k.html) — partial_download mechanism
- [soundata getting started](https://soundata.readthedocs.io/en/latest/source/tutorial.html)
- [FSD50K Zenodo release](https://zenodo.org/records/4060432) — fallback manual download
- `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md` §1, §4 — augmentation ordering, session-level splits, BG noise list
- `.planning/debug/uma16-no-detections.md` — RMS evidence for wide gain
- `src/acoustic/training/augmentation.py`, `hf_dataset.py`, `parquet_dataset.py`, `efficientat_trainer.py`, `config.py` — existing infrastructure (read in this research session)
- `scripts/vertex_submit.py`, `scripts/vertex_train.py`, `Dockerfile.vertex` — existing Vertex pipeline
- `src/acoustic/evaluation/evaluator.py` — Phase 9 eval harness to extend per D-28

### Secondary (MEDIUM confidence)
- [Fhrozen/FSD50k on HuggingFace](https://huggingface.co/datasets/Fhrozen/FSD50k) — alternative FSD50K mirror
- [LCAV/pyroomacoustics releases](https://github.com/LCAV/pyroomacoustics/releases)
- WebSearch result: pyroomacoustics latest release Dec 2025 (not directly verified to specific version)

### Tertiary (LOW confidence)
- DroneAudioSet non-drone subset availability — referenced in compass doc §1 but not directly verified to be downloadable. Treat D-19 as opportunistic.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified against current docs
- Architecture (sliding window, augmentation order): HIGH — locked by CONTEXT.md, pattern matches existing pickle-safe code
- pyroomacoustics API specifics: HIGH — verified against current docs
- DroneAudioSet non-drone clip availability (D-19): LOW — flagged as optional
- UMA-16 ambient SNR range tuning (D-11): MEDIUM — chosen by user, no literature reference
- Vertex L4 quota: LOW — unverified, fallback specified

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (30 days — pyroomacoustics and soundata are slow-moving; Vertex AI surfaces may shift faster but the SDK we use is stable)
