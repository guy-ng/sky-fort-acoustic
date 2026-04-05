---
status: awaiting_human_verify
trigger: "Training fails because macOS kills the process due to excessive memory consumption (over 90GB RAM)"
created: 2026-04-04T00:00:00Z
updated: 2026-04-04T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - in-memory audio cache loads entire 180K-sample dataset (~58GB decoded) into RAM
test: Fix applied and self-verified
expecting: Human verification that training completes without OOM
next_action: Await user confirmation

## Symptoms

expected: Training should complete without consuming excessive memory (should stay under ~16GB)
actual: Process consumes over 90GB of system RAM, macOS kills it
errors: macOS OOM kill - no Python exception, OS terminates the process
reproduction: Start a training run via the web UI or API
started: Currently happening, unclear if it ever worked at lower memory

## Eliminated

## Evidence

- timestamp: 2026-04-04T00:01:00Z
  checked: Dataset size on disk
  found: 39 parquet shards, 6.3GB on disk, 180320 total rows
  implication: Large dataset, compressed on disk

- timestamp: 2026-04-04T00:02:00Z
  checked: Per-sample decoded audio size (first 5 shards, 23120 samples)
  found: Avg 337.8 KB/sample, max 16s audio (256000 samples), high variance (0.3s to 16s)
  implication: Full dataset decoded = ~58.1 GB of float32 arrays

- timestamp: 2026-04-04T00:03:00Z
  checked: Cache mechanism in ParquetDataset and DroneAudioDataset
  found: Both use unbounded dict _audio_cache that stores every sample after first access. warm_cache(limit=0) loads ALL samples. Background thread in trainer.py calls warm_cache() with no limit.
  implication: Entire 58GB dataset gets loaded into Python dict in RAM

- timestamp: 2026-04-04T00:04:00Z
  checked: DataLoader worker config in trainer.py
  found: num_workers=min(8, cpu_count), persistent_workers=True, prefetch_factor=4. Workers fork the process, getting CoW copies of the dataset that get dirtied as cache grows.
  implication: Memory multiplied by worker processes on top of 58GB base cache

- timestamp: 2026-04-04T00:05:00Z
  checked: _load_audio cache miss path in ParquetDataset
  found: On cache miss, reads ENTIRE shard (up to 1.3GB parquet) via pq.read_table just to get one row, then caches the decoded audio
  implication: Even without warm_cache, individual access is wasteful and still builds unbounded cache

- timestamp: 2026-04-04T00:06:00Z
  checked: All 60 dataset/augmentation/parquet tests after fix
  found: All pass (60 passed, 0 failed)
  implication: Fix does not break existing functionality

## Resolution

root_cause: Both ParquetDataset and DroneAudioDataset have unbounded in-memory audio caches (_audio_cache dict) that accumulate every sample after first access. The trainer.py runs warm_cache() in a background thread with no limit, pre-loading all 180K samples (~58GB decoded float32) into RAM. Combined with DataLoader workers (up to 8 forked processes with persistent_workers=True), memory easily exceeds 90GB.
fix: (1) Removed in-memory _audio_cache dict from both dataset classes - audio loaded from disk/parquet on every __getitem__. (2) Removed warm_cache() method from both datasets. (3) Removed _warm_caches() static method and background cache thread from TrainingRunner. (4) Reduced max DataLoader workers from 8 to 4. Cache progress fields in UI gracefully degrade (stay at 0, progress bar hidden).
verification: All 60 dataset/augmentation/parquet tests pass. Imports verified. No _audio_cache, warm_cache, or _warm_caches remain in training code.
files_changed:
  - src/acoustic/training/dataset.py
  - src/acoustic/training/parquet_dataset.py
  - src/acoustic/training/trainer.py
