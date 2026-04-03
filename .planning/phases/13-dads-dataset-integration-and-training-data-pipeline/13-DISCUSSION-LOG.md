# Phase 13: DADS Dataset Integration and Training Data Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 13-dads-dataset-integration-and-training-data-pipeline
**Areas discussed:** Parquet extraction, Data splitting, Label mapping, Download method

---

## Parquet Extraction

### Q1: How should we handle the DADS Parquet format?

| Option | Description | Selected |
|--------|-------------|----------|
| Extract WAVs to disk | One-time extraction of all 180K WAVs into directories. ~36GB disk but zero pipeline changes. | |
| HuggingFace datasets lib | Add `datasets` dependency. Load Parquet natively, decode on-the-fly. Adds heavy dependency. | |
| Stream from Parquet directly | Use pyarrow to read Parquet shards and decode audio bytes on-the-fly. Custom reader needed. | ✓ |

**User's choice:** Stream from Parquet directly
**Notes:** User chose the most lightweight approach despite higher implementation complexity.

### Q2: Should there be a caching layer?

| Option | Description | Selected |
|--------|-------------|----------|
| No caching | Decode from Parquet on every __getitem__ call. Simpler code, no memory pressure. | ✓ |
| LRU memory cache | Cache decoded numpy arrays in memory. Faster repeat access but high RAM usage. | |
| Disk cache (WAV) | First access extracts to disk, subsequent reads from WAV. Hybrid approach. | |

**User's choice:** No caching (Recommended)
**Notes:** None

### Q3: New class or extend DroneAudioDataset?

| Option | Description | Selected |
|--------|-------------|----------|
| New ParquetDataset class | Separate class, same interface. Clean separation. | ✓ |
| Extend DroneAudioDataset | Add Parquet support inside existing class. Risk of bloat. | |
| You decide | Claude's discretion. | |

**User's choice:** New ParquetDataset class (Recommended)
**Notes:** None

---

## Data Splitting

### Q4: How should we split DADS data?

| Option | Description | Selected |
|--------|-------------|----------|
| File-level random split | 70/15/15 random split at file level. Simple, reproducible with fixed seed. | ✓ |
| Source-dataset grouping | Split by source dataset origin. Prevents cross-source leakage. | |
| Stratified split | Random 70/15/15 stratified by class to maintain ratio in each split. | |

**User's choice:** File-level random split (Recommended)
**Notes:** None

### Q5: Should the split be persisted or re-randomized?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed seed, deterministic | Same seed produces same split every time. Reproducible experiments. | ✓ |
| Persist split to manifest | Save split assignment to JSON/CSV. Most reproducible. | |
| Re-randomize each run | New random split every run. More diversity but harder to compare. | |

**User's choice:** Fixed seed, deterministic (Recommended)
**Notes:** None

---

## Label Mapping

### Q6: How should DADS labels integrate with existing system?

| Option | Description | Selected |
|--------|-------------|----------|
| Direct binary mapping | DADS label 1→1.0, label 0→0.0. ParquetDataset handles internally. | |
| Unified label registry | Shared registry for both WAV and Parquet datasets. | |
| You decide | Claude's discretion. | |

**User's choice:** (Free text) "ignore existing data, we will use DADS. new recording need to save in new parquet file"
**Notes:** Major decision — existing WAV-based data is deprecated. DADS becomes the primary dataset. New recordings must save as Parquet.

### Q7: Should recording pipeline be updated in this phase?

| Option | Description | Selected |
|--------|-------------|----------|
| Update in this phase | Modify recording pipeline to save Parquet. Bigger scope but cleaner. | ✓ |
| Separate follow-up | Phase 13 focuses only on DADS. Recording changes later. | |
| You decide | Claude's discretion. | |

**User's choice:** Update in this phase
**Notes:** User explicitly wants Parquet-only training data format.

### Q8: How to handle 10:1 class imbalance?

| Option | Description | Selected |
|--------|-------------|----------|
| WeightedRandomSampler | Reuse existing approach. Oversamples minority class each epoch. | ✓ |
| Subsample drone class | Drop drone files to match no-drone count. Discards 90% of data. | |
| You decide | Claude's discretion. | |

**User's choice:** WeightedRandomSampler (Recommended)
**Notes:** None

---

## Download Method

### Q9: How should DADS be acquired?

| Option | Description | Selected |
|--------|-------------|----------|
| Manual prerequisite | Document that DADS must be downloaded beforehand. Config points to path. | |
| CLI download script | Python script to clone/download. User runs once. | |
| API-triggered download | REST endpoint to trigger download. | |

**User's choice:** (Free text) "the data is loaded locally at: /Users/guyelisha/Projects/sky-fort-acoustic/data"
**Notes:** Data already present at `data/` with 39 Parquet shards. Manual prerequisite confirmed.

### Q10: Should dataset path be configurable?

| Option | Description | Selected |
|--------|-------------|----------|
| Env var | ACOUSTIC_TRAINING_DADS_PATH, defaulting to 'data/'. Consistent with BaseSettings pattern. | ✓ |
| Hardcoded default | Just use 'data/'. Less config surface. | |

**User's choice:** Env var (Recommended)
**Notes:** None

---

## Claude's Discretion

- Pyarrow audio decoding implementation details
- ParquetDataset internal structure (index building, shard management)
- Recording pipeline Parquet schema design
- Validation logic for DADS data integrity

## Deferred Ideas

None — discussion stayed within phase scope
