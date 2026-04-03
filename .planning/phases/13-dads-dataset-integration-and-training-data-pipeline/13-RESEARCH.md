# Phase 13: DADS Dataset Integration and Training Data Pipeline - Research

**Researched:** 2026-04-03
**Domain:** PyTorch Dataset from Parquet, audio decoding, data splitting, pyarrow integration
**Confidence:** HIGH

## Summary

Phase 13 integrates the DADS dataset (180,320 WAV files stored as Parquet shards) into the existing PyTorch training pipeline. The dataset is already present locally as 39 Parquet files in `data/`. Each row contains a `audio` struct column (with `bytes` and `path` fields) and an integer `label` column (0=no-drone, 1=drone). Audio bytes are standard WAV files (16kHz, mono, 16-bit PCM) with consistent 44-byte headers.

The primary work is: (1) a new `ParquetDataset` class that reads audio bytes from Parquet shards, decodes WAV to numpy, extracts random 0.5s segments, and produces mel-spectrogram tensors identical to the existing `DroneAudioDataset`; (2) a deterministic 70/15/15 file-level split with seed; (3) integration with `TrainingRunner` and `TrainingConfig`; (4) updating the field recording pipeline to write Parquet; and (5) adding `pyarrow` as a project dependency.

**Primary recommendation:** Build `ParquetDataset` as a sibling to `DroneAudioDataset` in `src/acoustic/training/dataset.py` (or a new `parquet_dataset.py`). Build a global index at construction time by scanning all shard metadata. Decode WAV bytes on each `__getitem__` call using raw numpy parsing (skip 44-byte header). Reuse `mel_spectrogram_from_segment()`, `build_weighted_sampler()`, and the existing augmentation pipeline unchanged.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Stream audio directly from Parquet via pyarrow -- no WAV extraction to disk, no caching layer. Decode audio bytes from Parquet on every `__getitem__` call.
- **D-02:** New `ParquetDataset` class (separate from existing `DroneAudioDataset`). Implements the same interface (returns mel-spec tensor + label). Clean separation.
- **D-03:** No caching layer -- decode fresh from Parquet every epoch.
- **D-04:** File-level random 70/15/15 train/val/test split. DADS clips are pre-trimmed independent recordings.
- **D-05:** Fixed random seed (deterministic) for reproducible splits.
- **D-06:** DADS binary labels used directly: label 1 -> 1.0, label 0 -> 0.0.
- **D-07:** Existing `WeightedRandomSampler` handles the 10:1 class imbalance (163K drone vs 16K no-drone).
- **D-08:** Phase 10's field recording pipeline updated to save Parquet rows.
- **D-09:** DADS data at `data/`, configurable via `ACOUSTIC_TRAINING_DADS_PATH` env var, default `data/`.
- **D-10:** No download automation -- dataset is a manual prerequisite.

### Claude's Discretion
- Pyarrow audio decoding implementation details (byte parsing, sample rate validation)
- ParquetDataset internal structure (index building, shard management)
- How to integrate ParquetDataset with existing TrainingRunner/TrainingManager
- Recording pipeline Parquet schema design (columns, metadata fields)
- Validation logic for DADS data integrity (sample rate, bit depth, mono checks)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DAT-01 | DADS dataset downloaded and validated (PCM 16-bit, mono, 16 kHz) | Parquet schema verified: `audio` struct with `bytes` (WAV) + `path`, `label` int64. All samples confirmed 16kHz mono 16-bit via wave module inspection across shards. |
| DAT-02 | Dataset loader handles DADS structure and integrates with training pipeline | ParquetDataset class design with shard scanning, global index, WAV byte decoding, mel-spec extraction via existing `mel_spectrogram_from_segment()` |
| DAT-03 | Session-level data splitting prevents leakage (70/15/15 train/val/test) | File-level IS session-level for DADS (pre-trimmed clips). Deterministic seed-based split with index shuffling. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyarrow | >=23.0 | Parquet I/O | Apache Arrow Parquet reader. Fastest columnar reads in Python. Already installed (23.0.1). |
| torch (Dataset) | >=2.11 | Dataset/DataLoader | Already in use. ParquetDataset extends `torch.utils.data.Dataset`. |
| numpy | >=1.26 | Audio array handling | Already in use. WAV bytes decoded to float32 numpy arrays. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| soundfile | >=0.13.1 | Field recording Parquet write | Already installed. Used to read audio in recording pipeline before writing to Parquet. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyarrow direct | HuggingFace datasets | Adds heavy dependency (datasets, fsspec, dill). pyarrow is lighter and sufficient. |
| Raw byte skip-44 decode | wave module decode | wave module is 2.6x slower (9.4ms vs 3.6ms per 1000 decodes). Raw is safe since all headers are 44 bytes. |
| Single Parquet file | Keep 39 shards | Shards enable parallel reads and reduce memory. Keep as-is. |

**Installation:**
```bash
pip install pyarrow>=23.0
```

**Version verification:** pyarrow 23.0.1 is the latest (confirmed via pip index). Published April 2026.

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/training/
    config.py           # Add dads_path field
    dataset.py          # Keep DroneAudioDataset (legacy WAV)
    parquet_dataset.py  # NEW: ParquetDataset + helpers
    trainer.py          # Update to accept ParquetDataset
    manager.py          # Minor config routing
    augmentation.py     # Unchanged (reused)
src/acoustic/api/
    pipeline_routes.py  # Update field recording to Parquet
    training_routes.py  # May need data_source param
```

### Pattern 1: Global Index from Parquet Shards
**What:** At ParquetDataset construction, scan all Parquet shard files and build a flat index mapping global row index to (shard_index, local_row_index). Store shard file paths and per-shard row counts. Read labels eagerly (label column only -- fast: ~40ms per shard, ~1.6s for all 39).
**When to use:** Always -- this is the initialization pattern.
**Example:**
```python
import pyarrow.parquet as pq
from pathlib import Path

class ParquetDataset(Dataset):
    def __init__(self, data_dir: str | Path, split_indices: list[int], ...):
        # Build shard metadata
        self._shards: list[Path] = sorted(Path(data_dir).glob("train-*.parquet"))
        self._shard_offsets: list[int] = []  # cumulative row offsets
        self._all_labels: list[int] = []
        
        offset = 0
        for shard_path in self._shards:
            table = pq.read_table(shard_path, columns=["label"])
            labels = table.column("label").to_pylist()
            self._all_labels.extend(labels)
            self._shard_offsets.append(offset)
            offset += len(labels)
        
        # Apply split indices
        self._indices = split_indices  # pre-computed by splitter
        self._labels = [self._all_labels[i] for i in split_indices]
    
    def _locate(self, global_idx: int) -> tuple[int, int]:
        """Map global index to (shard_index, local_row_index)."""
        import bisect
        shard_idx = bisect.bisect_right(self._shard_offsets, global_idx) - 1
        local_idx = global_idx - self._shard_offsets[shard_idx]
        return shard_idx, local_idx
```

### Pattern 2: WAV Byte Decoding (Fast Path)
**What:** Decode WAV audio bytes from Parquet by skipping the 44-byte header and interpreting remaining bytes as int16 PCM.
**When to use:** Every `__getitem__` call.
**Why safe:** Verified that 100% of sampled rows have standard 44-byte WAV headers.
**Example:**
```python
def _decode_audio(self, wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 numpy array. Skip 44-byte header."""
    audio_int16 = np.frombuffer(wav_bytes[44:], dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0
```
**Fallback for safety:** Use `wave` module if header validation is desired:
```python
import io, wave
def _decode_audio_safe(self, wav_bytes: bytes) -> np.ndarray:
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, 'rb') as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
```

### Pattern 3: Deterministic 70/15/15 Split
**What:** Shuffle all global indices with a fixed seed, then split into train/val/test.
**When to use:** Before constructing ParquetDataset instances.
**Example:**
```python
import random

def split_indices(total: int, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)
    
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    return train, val, test
```

### Pattern 4: Lazy Shard Loading (Per-Access)
**What:** Load only the needed shard's audio column on each `__getitem__`. Use `pq.read_table(path, columns=["audio"]).slice(row, 1)` for single-row access. Do NOT load all 39 shards into memory.
**When to use:** Every `__getitem__` call. Parquet columnar reads are efficient for single-column access.
**Performance note:** Single-row read from Parquet is ~0.06ms. With WAV decode ~0.004ms and mel-spec ~0.3ms, total per-sample is ~0.4ms. At batch_size=32 with num_workers=0, this is ~13ms per batch -- well within training throughput needs.
**Example:**
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_idx = self._indices[idx]
    shard_idx, local_idx = self._locate(global_idx)
    
    # Read single row's audio from the right shard
    table = pq.read_table(self._shards[shard_idx], columns=["audio"])
    wav_bytes = table.column("audio")[local_idx].as_py()["bytes"]
    
    audio = self._decode_audio(wav_bytes)
    # ... segment extraction, mel-spec, augmentation (same as DroneAudioDataset)
```

### Pattern 5: Field Recording Parquet Schema
**What:** Define a Parquet schema for field recordings that matches DADS structure.
**When to use:** When updating pipeline_routes.py to save recordings as Parquet.
**Example:**
```python
import pyarrow as pa
import pyarrow.parquet as pq

RECORDING_SCHEMA = pa.schema([
    ("audio", pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ])),
    ("label", pa.int64()),
])

def save_recording_as_parquet(
    audio_bytes: bytes,
    filename: str,
    label: int,
    output_path: Path,
) -> None:
    table = pa.table({
        "audio": [{"bytes": audio_bytes, "path": filename}],
        "label": [label],
    }, schema=RECORDING_SCHEMA)
    pq.write_table(table, str(output_path))
```

### Anti-Patterns to Avoid
- **Loading all shards into memory:** 39 shards x ~100MB each = ~4GB. Read per-access instead.
- **Extracting WAV files to disk:** Decision D-01 explicitly forbids this. Read from Parquet directly.
- **Using HuggingFace `datasets` library:** Adds unnecessary dependency. pyarrow alone is sufficient.
- **Building custom caching:** Decision D-03 says no caching. Fresh decode every epoch.
- **Modifying DroneAudioDataset:** Decision D-02 says keep it separate. New class only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet reading | Custom binary parser | pyarrow.parquet | Standard, battle-tested, handles compression/encoding |
| Mel spectrogram | New transform pipeline | Existing `mel_spectrogram_from_segment()` | Already validated, matches ResearchPreprocessor exactly |
| Class balancing | Custom over/under-sampling | Existing `build_weighted_sampler()` | Already handles arbitrary class ratios |
| WAV writing (recordings) | Manual header construction | `soundfile.write()` -> `io.BytesIO` | Handles format details correctly |
| Data augmentation | New augmentation code | Existing `WaveformAugmentation` + `SpecAugment` | Already tested and integrated |

**Key insight:** Almost all signal processing and ML infrastructure already exists. This phase is primarily a data loading adapter (Parquet -> same tensor interface) plus plumbing.

## Common Pitfalls

### Pitfall 1: Shard Boundary Errors in Global Index
**What goes wrong:** Off-by-one errors when mapping global row index to (shard, local_row). If shard offsets are [0, 4624, 9248, ...] and you access global index 4624, it must map to shard 1, row 0 -- not shard 0, row 4624.
**Why it happens:** `bisect.bisect_right` vs `bisect_left` confusion.
**How to avoid:** Use `bisect.bisect_right(offsets, idx) - 1`. Write explicit boundary tests: first row of each shard, last row of each shard.
**Warning signs:** IndexError when accessing rows near shard boundaries.

### Pitfall 2: Short Audio Clips (< 0.5s)
**What goes wrong:** Some no-drone clips are as short as 0.12s (1,920 samples). The segment extraction expects 8,000 samples (0.5s). Without padding, numpy slicing fails silently or produces wrong-sized arrays.
**Why it happens:** DADS no-drone clips have variable duration (0.12s to 16s).
**How to avoid:** Same zero-padding logic as `DroneAudioDataset.__getitem__`: if `len(audio) < segment_samples`, zero-pad. Already implemented in the existing dataset class -- just replicate.
**Warning signs:** Tensor shape mismatches during DataLoader collation.

### Pitfall 3: Memory Pressure from Label Scanning
**What goes wrong:** Reading all 39 shards at construction time to build the label index could be slow if reading full tables.
**Why it happens:** Parquet columnar format means reading just the `label` column is fast (~40ms/shard), but reading `audio` column is ~270ms/shard.
**How to avoid:** Always specify `columns=["label"]` when scanning for index building. Total scan time: ~1.6s for all 39 shards (acceptable for one-time init).
**Warning signs:** Multi-minute dataset construction time.

### Pitfall 4: Inconsistent Split Across Train/Val/Test Datasets
**What goes wrong:** Each ParquetDataset instance rebuilds the shard index independently, but if shard order or contents change, splits become inconsistent.
**Why it happens:** File system glob ordering or dataset updates.
**How to avoid:** Use `sorted()` for shard listing. Compute splits once in a factory function that returns train/val/test datasets. Fixed seed ensures reproducibility.
**Warning signs:** Val/test metrics suspiciously good (data leakage from split inconsistency).

### Pitfall 5: Drone vs No-Drone Duration Asymmetry
**What goes wrong:** Drone clips average 0.6s (close to the 0.5s window), while no-drone clips average 7.3s. This means drone clips contribute ~1 segment per file, while no-drone clips could contribute ~14 segments. With file-level sampling, each file yields one random segment per epoch -- this is correct behavior per the existing design, but may surprise if someone expects segment-level coverage.
**Why it happens:** DADS design -- drone clips are pre-trimmed, no-drone clips are longer ambient recordings.
**How to avoid:** This is actually fine with `WeightedRandomSampler` which oversamples the minority class. Just be aware that long no-drone clips have diverse segment content (good for generalization).
**Warning signs:** None -- this is expected behavior.

### Pitfall 6: pyarrow Not in pyproject.toml
**What goes wrong:** Works locally but fails in Docker or CI because pyarrow is not declared as a dependency.
**Why it happens:** pyarrow was installed manually for testing but not added to project deps.
**How to avoid:** Add `pyarrow>=23.0` to project dependencies in the first task.
**Warning signs:** ModuleNotFoundError in fresh environments.

## Code Examples

### Complete ParquetDataset.__getitem__ Pattern
```python
# Source: Derived from existing DroneAudioDataset pattern + pyarrow API
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_idx = self._indices[idx]
    shard_idx, local_idx = self._locate(global_idx)
    
    # Read audio bytes from Parquet shard
    table = pq.read_table(self._shards[shard_idx], columns=["audio"])
    wav_bytes = table.column("audio")[local_idx].as_py()["bytes"]
    
    # Decode WAV: skip 44-byte header, int16 -> float32
    audio = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
    
    n = self._mel_config.segment_samples  # 8000 (0.5s at 16kHz)
    
    # Random segment extraction (same as DroneAudioDataset)
    if len(audio) >= n:
        start = random.randint(0, len(audio) - n)
        segment = audio[start:start + n]
    else:
        segment = np.zeros(n, dtype=np.float32)
        segment[:len(audio)] = audio
    
    # Waveform augmentation
    if self._waveform_aug is not None:
        segment = self._waveform_aug(segment)
    
    # Mel spectrogram: (1, 1, 128, 64) -> squeeze -> (1, 128, 64)
    features = mel_spectrogram_from_segment(segment, self._mel_config).squeeze(0)
    
    # Spec augmentation
    if self._spec_aug is not None:
        features = self._spec_aug(features)
    
    label_tensor = torch.tensor(self._labels[idx], dtype=torch.float32)
    return features, label_tensor
```

### TrainingRunner Integration Pattern
```python
# In trainer.py run() method -- swap data source based on config
if cfg.dads_path and Path(cfg.dads_path).is_dir():
    from acoustic.training.parquet_dataset import ParquetDataset, split_indices
    
    # Build global index and split
    dataset_builder = ParquetDataset.scan_shards(cfg.dads_path)
    train_idx, val_idx, test_idx = split_indices(
        dataset_builder.total_rows, seed=42,
        train_frac=0.70, val_frac=0.15,
    )
    
    train_ds = ParquetDataset(cfg.dads_path, train_idx, mel_config,
                              waveform_aug=wave_aug, spec_aug=spec_aug)
    val_ds = ParquetDataset(cfg.dads_path, val_idx, mel_config)
    
    train_labels = train_ds.labels
    # ... rest of training loop unchanged
else:
    # Legacy WAV path (existing code)
    all_paths, all_labels = collect_wav_files(cfg.data_root, cfg.label_map)
    # ...
```

### Field Recording to Parquet
```python
# In pipeline_routes.py or recording manager
import io
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

def save_recording_parquet(
    audio: np.ndarray,
    sr: int,
    label: int,
    filename: str,
    output_dir: Path,
) -> Path:
    """Save a field recording as a single-row Parquet file."""
    # Encode audio to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    
    table = pa.table({
        "audio": [{"bytes": wav_bytes, "path": filename}],
        "label": [label],
    })
    
    output_path = output_dir / f"{Path(filename).stem}.parquet"
    pq.write_table(table, str(output_path))
    return output_path
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| WAV files in directories | Parquet shards (HuggingFace format) | 2024-2025 | Standard for large audio datasets. Faster I/O, smaller footprint. |
| librosa mel-spec | torchaudio mel-spec | Phase 11 (2026-04-02) | Already migrated. ParquetDataset reuses torchaudio path. |
| Train/val split only | Train/val/test 70/15/15 | This phase | Adds held-out test set for proper evaluation. |

**Deprecated/outdated:**
- `DroneAudioDataset` + `collect_wav_files()`: Still in codebase but DADS replaces `audio-data/data/` as primary dataset. Keep for backward compatibility but DADS is the default going forward.
- `data_root` config field for WAV directories: Superseded by `dads_path` for Parquet data.

## Open Questions

1. **Shard-level caching of pyarrow tables**
   - What we know: Reading a full shard each `__getitem__` call reads ~100MB just to get one row's audio. pyarrow memory-maps by default which helps, but repeated reads of the same shard may benefit from keeping the table reference.
   - What's unclear: Whether pyarrow's memory mapping is sufficient or whether an LRU cache of recently-read shard tables improves throughput.
   - Recommendation: Start without caching per D-03. If profiling shows I/O bottleneck, consider keeping the last N shard tables in memory (N=2-4, since DataLoader access patterns are sequential within a shard after index sorting).

2. **num_workers > 0 for DataLoader**
   - What we know: Current pipeline uses `num_workers=0`. With 180K samples and Parquet I/O, multi-worker loading could improve throughput.
   - What's unclear: Whether pyarrow handles multi-process reads safely (it should -- Parquet files are read-only).
   - Recommendation: Start with `num_workers=0` to match existing pattern. Profile and upgrade later if needed.

3. **Test split persistence**
   - What we know: The deterministic seed ensures the same split every time. But there is no explicit test split file saved.
   - What's unclear: Whether downstream evaluation (Phase 9 harness) needs an explicit test set manifest.
   - Recommendation: The seed-based approach is sufficient since it's deterministic. If evaluation needs explicit indices, add a utility to export split indices to JSON.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pyarrow | Parquet I/O | Yes (installed) | 23.0.1 | -- |
| numpy | Audio decoding | Yes | >=1.26 | -- |
| torch | Dataset/DataLoader | Yes | >=2.11 | -- |
| torchaudio | Mel spectrogram | Yes | >=2.11 | -- |
| soundfile | Recording Parquet write | Yes | 0.13.1 | -- |
| pytest | Testing | Yes | >=8.0 | -- |

**Missing dependencies with no fallback:** None -- pyarrow is the only new dependency and is already installed.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/test_parquet_dataset.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DAT-01 | DADS validation (format, SR, channels) | unit | `pytest tests/unit/test_parquet_dataset.py::TestDADSValidation -x` | Wave 0 |
| DAT-02 | ParquetDataset loads and returns correct tensors | unit | `pytest tests/unit/test_parquet_dataset.py::TestParquetDataset -x` | Wave 0 |
| DAT-02 | Integration with TrainingRunner end-to-end | integration | `pytest tests/integration/test_parquet_training.py -x` | Wave 0 |
| DAT-03 | 70/15/15 split is deterministic and non-overlapping | unit | `pytest tests/unit/test_parquet_dataset.py::TestSplitIndices -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_parquet_dataset.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_parquet_dataset.py` -- covers DAT-01, DAT-02, DAT-03
- [ ] `tests/integration/test_parquet_training.py` -- covers end-to-end training with Parquet data
- [ ] Framework install: None needed -- pytest already configured

## Project Constraints (from CLAUDE.md)

- **Stack:** Python backend, React+Vite+TypeScript+Tailwind frontend
- **Testing:** pytest with pytest-asyncio for async tests
- **Linting:** Ruff (replaces flake8+black+isort)
- **Config:** Pydantic BaseSettings with `ACOUSTIC_TRAINING_` env prefix
- **ML:** PyTorch for training, torchaudio for audio transforms
- **GSD Workflow:** All changes through GSD commands
- **Audio:** 16kHz sample rate, MelConfig as shared preprocessing constants

## Sources

### Primary (HIGH confidence)
- Direct Parquet schema inspection via pyarrow on local `data/train-*.parquet` files
- Existing codebase: `src/acoustic/training/dataset.py`, `trainer.py`, `config.py`, `manager.py`
- `data/README.md` -- DADS dataset description and statistics
- 13-CONTEXT.md -- All locked decisions (D-01 through D-10)

### Secondary (MEDIUM confidence)
- pyarrow 23.0.1 API (installed locally, verified via Python import)
- Performance benchmarks (local timing: WAV decode, Parquet reads)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pyarrow is the only new dependency, already installed and verified
- Architecture: HIGH - ParquetDataset follows exact same pattern as existing DroneAudioDataset, Parquet schema verified empirically
- Pitfalls: HIGH - All identified from direct data inspection (duration asymmetry, header consistency, shard boundaries)

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable -- data format and pyarrow API are mature)
