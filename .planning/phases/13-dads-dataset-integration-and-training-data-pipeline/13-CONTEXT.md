# Phase 13: DADS Dataset Integration and Training Data Pipeline - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Download, validate, and integrate the DADS dataset (180,320 files from HuggingFace Parquet shards) into the training pipeline with session-level data splitting. Also update the field recording pipeline (Phase 10) to save new recordings as Parquet, making Parquet the unified training data format. This phase does NOT include model architecture changes (Phase 14), advanced training enhancements (Phase 15), or edge export (Phase 16).

</domain>

<decisions>
## Implementation Decisions

### Data Format & Loading
- **D-01:** Stream audio directly from Parquet via pyarrow — no WAV extraction to disk, no caching layer. Decode audio bytes from Parquet on every `__getitem__` call. Parquet columnar reads are fast and random segment extraction means each access differs anyway.
- **D-02:** New `ParquetDataset` class (separate from existing `DroneAudioDataset`). Implements the same interface (returns mel-spec tensor + label). Clean separation — `DroneAudioDataset` stays for any legacy WAV needs, `ParquetDataset` handles DADS and future Parquet data. Training code selects the right Dataset based on data source config.
- **D-03:** No caching layer — decode fresh from Parquet every epoch. Simpler code, no memory pressure from caching 180K files.

### Data Splitting
- **D-04:** File-level random 70/15/15 train/val/test split. Since DADS clips are pre-trimmed independent recordings (not segments from longer sessions), file-level IS session-level. Consistent with Phase 8 D-03.
- **D-05:** Fixed random seed (deterministic) for reproducible splits. Same seed produces the same split every time — consistent test set across model iterations.

### Label Mapping & Class Imbalance
- **D-06:** DADS binary labels used directly: label 1 (drone) → 1.0, label 0 (no-drone) → 0.0. ParquetDataset handles this internally. Existing WAV-based training data (`audio-data/data/`) is deprecated — DADS is the primary dataset going forward.
- **D-07:** Existing `WeightedRandomSampler` (Phase 8 D-02) handles the 10:1 class imbalance (163K drone vs 16K no-drone). Oversamples minority class each epoch, no data discarded.

### Recording Pipeline Update
- **D-08:** Phase 10's field recording pipeline is updated in this phase to save new recordings as Parquet rows (audio bytes + label + metadata). Training data format becomes Parquet-only across the board.

### Data Path & Configuration
- **D-09:** DADS data lives at `data/` (39 Parquet shards, ~4.2GB). Path is configurable via `ACOUSTIC_TRAINING_DADS_PATH` env var, defaulting to `data/`. Follows existing Pydantic BaseSettings pattern with `ACOUSTIC_TRAINING_` prefix.
- **D-10:** No download automation — dataset is a manual prerequisite (already present locally). Documentation explains how to acquire the dataset.

### Claude's Discretion
- Pyarrow audio decoding implementation details (byte parsing, sample rate validation)
- ParquetDataset internal structure (index building, shard management)
- How to integrate ParquetDataset with existing TrainingRunner/TrainingManager
- Recording pipeline Parquet schema design (columns, metadata fields)
- Validation logic for DADS data integrity (sample rate, bit depth, mono checks)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### DADS Dataset
- `data/README.md` — Dataset description, statistics (180,320 files, 60.9 hours), label schema (0=no-drone, 1=drone), source attribution
- `data/train-*.parquet` — 39 Parquet shards containing audio bytes + labels. HuggingFace Datasets format with `audio` (dtype: audio) and `label` (dtype: class_label) columns

### Existing Training Pipeline (Phase 13 extends/replaces)
- `src/acoustic/training/dataset.py` — `DroneAudioDataset`, `collect_wav_files()`, `build_weighted_sampler()`. ParquetDataset is a sibling class. WeightedRandomSampler reused.
- `src/acoustic/training/config.py` — `TrainingConfig` with `data_root`, `label_map`, `val_split`. Extend with `dads_path` field.
- `src/acoustic/training/trainer.py` — `TrainingRunner` that creates DataLoaders. Must accept ParquetDataset.
- `src/acoustic/training/manager.py` — `TrainingManager` orchestrates training runs. May need updates for Parquet data source selection.
- `src/acoustic/training/augmentation.py` — `SpecAugment`, `WaveformAugmentation`. Reused by ParquetDataset.

### Preprocessing
- `src/acoustic/classification/preprocessing.py` — `ResearchPreprocessor`, `mel_spectrogram_from_segment()`. ParquetDataset reuses mel-spec extraction.
- `src/acoustic/classification/config.py` — `MelConfig` (SR=16000, segment_samples for 0.5s). DADS audio is already 16kHz mono — matches.

### Field Recording Pipeline (Phase 13 modifies)
- `src/acoustic/api/pipeline_routes.py` — Recording API endpoints (Phase 10). Update to save Parquet.

### Prior Phase Context
- `.planning/phases/08-pytorch-training-pipeline/08-CONTEXT.md` — Training pipeline decisions (D-01 lazy loading, D-02 WeightedRandomSampler, D-03 file-level splitting)
- `.planning/phases/09-evaluation-harness-and-api/09-CONTEXT.md` — Evaluation harness, test data paths
- `.planning/phases/10-field-data-collection/10-CONTEXT.md` — Field recording pipeline structure

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_weighted_sampler()` in `dataset.py` — Class-balanced sampling, directly reusable with ParquetDataset labels
- `mel_spectrogram_from_segment()` in `preprocessing.py` — Converts numpy audio segment to mel-spec tensor. ParquetDataset calls this after decoding Parquet audio
- `SpecAugment` and `WaveformAugmentation` in `augmentation.py` — Augmentation pipeline, reusable as-is
- `MelConfig` in `config.py` — Shared preprocessing constants (SR=16000, 0.5s segments)
- `TrainingRunner` in `trainer.py` — Training loop with early stopping. Accepts DataLoader — ParquetDataset just needs to produce compatible DataLoader

### Established Patterns
- Config via Pydantic `BaseSettings` with `ACOUSTIC_TRAINING_` env prefix
- PyTorch `Dataset` → `DataLoader` with `WeightedRandomSampler`
- Background daemon thread for training (TrainingManager pattern)
- Protocol-based injection (Classifier, Preprocessor protocols)

### Integration Points
- `TrainingRunner.train()` creates DataLoaders from Dataset — swap in ParquetDataset
- `TrainingConfig` needs new `dads_path` field for Parquet data location
- Recording pipeline endpoints need Parquet write capability
- Evaluation harness may need to read from Parquet for test data (or keep using the split from ParquetDataset)

</code_context>

<specifics>
## Specific Ideas

- DADS audio is already 16kHz mono PCM 16-bit — matches MelConfig.sample_rate exactly, no resampling needed
- Parquet shards are 39 files (train-00000-of-00039.parquet through train-00038-of-00039.parquet), each containing a subset of the 180,320 rows
- DADS drone clips average 0.6s (close to the 0.5s segment extraction window), while no-drone clips average 7.3s — segment extraction behavior differs significantly between classes
- The `drone-audio-detection-samples/` git clone also exists at project root but `data/` is the primary path to use
- Existing `DroneAudioDataset` and WAV-based pipeline are being deprecated in favor of Parquet-only

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-dads-dataset-integration-and-training-data-pipeline*
*Context gathered: 2026-04-03*
