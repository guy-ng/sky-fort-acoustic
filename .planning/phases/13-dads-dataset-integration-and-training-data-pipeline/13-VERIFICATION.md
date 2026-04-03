---
phase: 13-dads-dataset-integration-and-training-data-pipeline
verified: 2026-04-03T00:00:00Z
status: human_needed
score: 11/11 must-haves verified
human_verification:
  - test: "Train on the real 39-shard DADS dataset (180,320 rows) end-to-end"
    expected: "Training completes at least one epoch, produces a .pt checkpoint, dads_path='data/' auto-detects the shards without any config change"
    why_human: "Real DADS shards are not checked into the repo. Synthetic tests use 20-row fixtures and cannot verify 180,320-row scale, shard count, or real WAV byte structure."
  - test: "Verify baseline model accuracy on the DADS test split"
    expected: ">90% binary detection accuracy as stated in Phase 13 ROADMAP success criterion 4"
    why_human: "Model accuracy requires actually running training to convergence on real data — cannot be verified statically."
  - test: "Confirm DADS dataset has exactly 39 shards and 180,320 rows after download"
    expected: "ParquetDatasetBuilder.total_rows == 180320, len(shards) == 39"
    why_human: "Dataset is on HuggingFace, not in the repo. The plan states 39 shards / 180,320 rows as a truth but no automated check validates this against the real data."
---

# Phase 13: DADS Dataset Integration and Training Data Pipeline — Verification Report

**Phase Goal:** Download, validate, and integrate the DADS dataset (geronimobasso/drone-audio-detection-samples from HuggingFace) into the training pipeline with proper session-level data splitting
**Verified:** 2026-04-03
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

All 11 must-have truths from the two PLANs have been verified against the actual codebase.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ParquetDataset scans Parquet shards and builds a global index via bisect | VERIFIED | `parquet_dataset.py` lines 178-194: `sorted(data_path.glob("train-*.parquet"))`, cumulative `_shard_offsets`, `bisect.bisect_right` in `_locate()` |
| 2 | Each `__getitem__` decodes WAV bytes, extracts a random 0.5s segment, returns (1, 128, 64) mel-spectrogram tensor + float32 label | VERIFIED | Lines 114-156: `pq.read_table`, `decode_wav_bytes`, random segment or zero-pad, `mel_spectrogram_from_segment(...).squeeze(0)`, `torch.tensor(..., dtype=torch.float32)` |
| 3 | `split_indices()` produces deterministic 70/15/15 train/val/test splits with no overlap | VERIFIED | Lines 44-53: `random.Random(seed).shuffle`, integer slicing; test suite `TestSplitIndices` (6 tests) passes |
| 4 | Short audio clips (<0.5s) are zero-padded without errors | VERIFIED | Lines 137-139: `np.zeros(n, dtype=np.float32); segment[:len(audio)] = audio`; `test_short_audio_zero_padded` passes |
| 5 | Shard boundary indices map correctly (no off-by-one) | VERIFIED | `_locate()` uses `bisect_right(...) - 1`; `test_locate_shard_boundary` and `test_locate_last_shard_boundary` both pass |
| 6 | `TrainingRunner.run()` uses ParquetDataset when `dads_path` is configured and directory has shards | VERIFIED | `trainer.py` lines 111-127: `use_parquet` gate checks `dads_dir.is_dir()` AND `list(dads_dir.glob("train-*.parquet"))`; integration test `test_trainer_uses_parquet_when_shards_exist` passes, checkpoint produced |
| 7 | `TrainingRunner.run()` falls back to DroneAudioDataset when `dads_path` is not set or directory missing | VERIFIED | Lines 128-152: legacy WAV path unchanged; `test_trainer_falls_back_to_wav` raises `ValueError("Data root does not exist")` confirming fallback executes |
| 8 | Data split is 70/15/15 train/val/test with seed=42 for reproducibility | VERIFIED | `trainer.py` line 119: `split_indices(builder.total_rows, seed=42)` |
| 9 | Field recordings save as Parquet files matching DADS schema | VERIFIED | `recorder.py` lines 105-127: `to_parquet()` writes `{audio: {bytes, path}, label}`; `manager.py` lines 168-176: `label_recording()` does the same after WAV move; 4 tests cover both paths |
| 10 | `TrainingConfig.dads_path` defaults to `"data/"` and is configurable via `ACOUSTIC_TRAINING_DADS_PATH` env var | VERIFIED | `config.py` line 27: `dads_path: str = "data/"`; `SettingsConfigDict(env_prefix="ACOUSTIC_TRAINING_")` auto-wires env var; test `test_training_config_dads_path_from_env` passes |
| 11 | Training produces a checkpoint file when run with DADS Parquet data | VERIFIED | `trainer.py` lines 238-239: `torch.save(model.state_dict(), str(ckpt_path))` on improvement; integration test asserts `result is not None` and `result.exists()` after 1 epoch on synthetic shards |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/acoustic/training/parquet_dataset.py` | ParquetDataset class, split_indices, decode_wav_bytes | VERIFIED | 223 lines. Exports `ParquetDataset`, `ParquetDatasetBuilder`, `split_indices`, `decode_wav_bytes`. All wired via imports from `acoustic.classification.config` and `acoustic.classification.preprocessing`. |
| `tests/unit/test_parquet_dataset.py` | Unit tests for DAT-01/02/03 | VERIFIED | 216 lines. Contains `TestSplitIndices` (6 tests), `TestDecodeWavBytes` (3 tests), `TestParquetDataset` (8 tests). All 17 pass. |
| `src/acoustic/training/config.py` | Extended TrainingConfig with `dads_path` | VERIFIED | Line 27: `dads_path: str = "data/"` present. |
| `src/acoustic/training/trainer.py` | Updated TrainingRunner with ParquetDataset branch | VERIFIED | Lines 110-153: `use_parquet` branch with `ParquetDatasetBuilder` and `split_indices`. Legacy WAV path fully preserved. |
| `src/acoustic/recording/recorder.py` | RecordingSession with `to_parquet()` | VERIFIED | Lines 105-127: `to_parquet(label: int) -> Path` implemented. `path` property present at line 93. `import pyarrow as pa` and `import pyarrow.parquet as pq` at top. |
| `tests/unit/test_parquet_training_integration.py` | Integration tests | VERIFIED | 221 lines. `TestTrainingConfigDadsPath` (2), `TestTrainerParquetBranch` (2), `TestRecordingSessionToParquet` (2), `TestLabelRecordingCreatesParquet` (2). All 8 pass. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `parquet_dataset.py` | `acoustic.classification.preprocessing` | `from acoustic.classification.preprocessing import mel_spectrogram_from_segment` | WIRED | Line 20: import present; line 146: called in `__getitem__` |
| `parquet_dataset.py` | `acoustic.classification.config` | `from acoustic.classification.config import MelConfig` | WIRED | Line 19: import present; used as default `MelConfig()` in constructor |
| `parquet_dataset.py` | Parquet shards | `pq.read_table` | WIRED | Lines 124, 191: `pq.read_table` called in both `__getitem__` and `ParquetDatasetBuilder.__init__` |
| `trainer.py` | `parquet_dataset.py` | `from acoustic.training.parquet_dataset import ParquetDatasetBuilder, split_indices` | WIRED | Line 115: lazy import inside `use_parquet` branch; `ParquetDatasetBuilder` instantiated line 118, `split_indices` called line 119 |
| `trainer.py` | `config.py` | `cfg.dads_path` | WIRED | Lines 111-112: `cfg.dads_path` read and used to construct `dads_dir` |
| `recorder.py` | `pyarrow.parquet` | `pq.write_table` for Parquet output | WIRED | Line 126: `pq.write_table(table, str(parquet_path))` in `to_parquet()` |
| `manager.py` | `pyarrow.parquet` | `pq.write_table` in `label_recording()` | WIRED | Line 176: `pq.write_table(parquet_table, str(parquet_path))` |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `ParquetDataset.__getitem__` | `wav_bytes` from `audio_struct["bytes"]` | `pq.read_table(shard, columns=["audio"])` — real Parquet I/O | Yes — reads actual bytes from disk | FLOWING |
| `TrainingRunner.run()` (Parquet branch) | `train_ds`, `val_ds` | `ParquetDatasetBuilder(dads_dir).build(split_indices)` | Yes — builder reads label columns from all shards, build() passes real shard paths | FLOWING |
| `RecordingSession.to_parquet()` | `wav_bytes` | `wav_path.read_bytes()` — reads the completed WAV file | Yes — reads real bytes written by `stop()` | FLOWING |
| `RecordingManager.label_recording()` | `wav_bytes` | `target_wav.read_bytes()` after shutil.move | Yes — reads the moved real WAV file | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 17 parquet_dataset unit tests pass | `PYTHONPATH=src python -m pytest tests/unit/test_parquet_dataset.py -x` | 17 passed in 0.79s | PASS |
| All 8 integration tests pass | `PYTHONPATH=src python -m pytest tests/unit/test_parquet_training_integration.py -x` | 8 passed | PASS |
| Combined suite (25 tests) | `PYTHONPATH=src python -m pytest tests/unit/test_parquet_dataset.py tests/unit/test_parquet_training_integration.py -x` | 25 passed in 4.58s | PASS |
| TrainingConfig default | `python -c "from acoustic.training.config import TrainingConfig; c = TrainingConfig(); print(c.dads_path)"` | `data/` | PASS |
| Key imports resolve | `python -c "from acoustic.training.trainer import TrainingRunner; from acoustic.training.parquet_dataset import ParquetDataset, split_indices"` | imports OK | PASS |
| RecordingSession.to_parquet exists | `python -c "from acoustic.recording.recorder import RecordingSession; print(hasattr(RecordingSession, 'to_parquet'))"` | True | PASS |

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DAT-01 | 13-01, 13-02 | DADS dataset loading from Parquet shards without disk extraction | SATISFIED | `decode_wav_bytes` skips 44-byte WAV header and decodes in-memory; no temp file writes anywhere in `__getitem__` |
| DAT-02 | 13-01, 13-02 | Dataset integrates with existing DroneAudioDataset-based training pipeline | SATISFIED | `ParquetDataset` returns same `(1, 128, 64) float32 / float32` tuple shape as `DroneAudioDataset`; `labels` property supports `build_weighted_sampler`; `trainer.py` uses both behind same DataLoader construction |
| DAT-03 | 13-01, 13-02 | Session-level (file-level) data splitting prevents data leakage — 70/15/15 | SATISFIED | `split_indices()` shuffles with fixed seed before slicing; indices are file-level row indices; `test_no_overlap` confirms no leakage |

**ORPHANED REQUIREMENTS:** DAT-01, DAT-02, DAT-03 are referenced in both PLANs but do not appear in `.planning/REQUIREMENTS.md`. The traceability table in REQUIREMENTS.md has no entry for Phase 13 or any DAT-* IDs. These requirements were used as the contractual basis for both plans but were never registered in the requirements document. This is a documentation gap, not a code gap — the implementation satisfies the requirements as specified in the PLANs. REQUIREMENTS.md should be updated to add DAT-01/02/03 and their Phase 13 traceability row.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `trainer.py` | 112 | `list(dads_dir.glob(...))` eagerly evaluated as truthiness check | Info | Correct behavior — converts to list to check non-empty. Slightly wasteful (allocates list for glob result) but not a bug. |

No TODOs, FIXMEs, placeholders, empty returns, or hardcoded empty data found in any of the 6 modified/created files. No stubs detected.

---

### Human Verification Required

#### 1. Real DADS Dataset End-to-End Training

**Test:** Download geronimobasso/drone-audio-detection-samples from HuggingFace to `data/`. Start the service and trigger training via the web UI or `ACOUSTIC_TRAINING_DADS_PATH=data/ python -m acoustic.training.trainer` equivalent. Observe log output showing "Using DADS Parquet data from data/" and watch training progress for at least one epoch.

**Expected:** Training runs against 39 Parquet shards (180,320 rows), the 70/15/15 split produces ~126K/27K/27K indices, a checkpoint is saved to `models/research_cnn_trained.pt` after the first epoch that improves validation loss.

**Why human:** The real DADS dataset is not in the repo. The 39-shard / 180,320-row claim cannot be verified without downloading the dataset. Synthetic tests use 20-row fixtures only.

#### 2. Baseline Accuracy on DADS Test Split

**Test:** After training converges (early stopping triggers), load the checkpoint and evaluate on the held-out 15% test indices using the evaluation harness from Phase 9.

**Expected:** Binary detection accuracy >90% on the DADS test split (per ROADMAP Phase 13 Success Criterion 4).

**Why human:** Accuracy requires training to convergence on real data. This cannot be statically verified. The architecture (ResearchCNN 3-layer CNN) is fixed from Phase 7; accuracy depends on the quality of the DADS data and training hyperparameters.

#### 3. DADS Dataset Schema Validation

**Test:** After downloading, run: `PYTHONPATH=src python -c "from acoustic.training.parquet_dataset import ParquetDatasetBuilder; b = ParquetDatasetBuilder('data/'); print(f'shards={len(b._shards)}, rows={b.total_rows}')"`

**Expected:** `shards=39, rows=180320` — confirming the real dataset matches the plan's expected dimensions.

**Why human:** Cannot verify the HuggingFace dataset structure without the actual files.

---

### Gaps Summary

No code gaps. All automated checks pass. The three human verification items are about real dataset scale and model accuracy — they cannot be validated without downloading the 60.9-hour DADS dataset.

The single documentation gap (DAT-01/02/03 absent from REQUIREMENTS.md) does not block the phase goal; it is a housekeeping item.

---

_Verified: 2026-04-03_
_Verifier: Claude (gsd-verifier)_
