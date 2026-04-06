---
phase: 20
plan: 03
type: execute
wave: 1
depends_on:
  - "20-00"
files_modified:
  - src/acoustic/training/hf_dataset.py
  - src/acoustic/training/parquet_dataset.py
autonomous: true
requirements:
  - D-13
  - D-14
  - D-15
  - D-16
must_haves:
  truths:
    - "WindowedHFDroneDataset enumerates sliding windows with hop = (1 - overlap) * window_samples"
    - "Test split uses non-overlapping windows (hop == window_samples)"
    - "Train and val splits use 60% overlap (hop = 40% of window length) when window_overlap_ratio=0.6"
    - "split_file_indices operates on FILE indices, not window indices, preserving session-level isolation"
    - "test_no_file_leakage_across_splits passes — zero file_idx overlap between any two splits"
  artifacts:
    - path: src/acoustic/training/hf_dataset.py
      provides: "class WindowedHFDroneDataset with __len__ + __getitem__(idx) -> (file_idx, offset)-based window extraction"
      contains: "class WindowedHFDroneDataset"
    - path: src/acoustic/training/parquet_dataset.py
      provides: "split_file_indices(num_files, seed, train, val) helper"
      contains: "def split_file_indices"
  key_links:
    - from: src/acoustic/training/hf_dataset.py
      to: src/acoustic/training/parquet_dataset.py
      via: "WindowedHFDroneDataset constructed from file_indices produced by split_file_indices"
      pattern: "split_file_indices"
---

<objective>
Implement the sliding-window dataset (D-13, D-14) and the file-index session-level split helper
(D-15) with non-overlapping test split (D-16). The legacy HFDroneDataset uses random 0.5s segment
extraction; v7 needs deterministic enumeration so that 60% overlap inflates effective sample count
~2.5× without leaking adjacent windows from the same source recording across splits.

Purpose: This is the highest-risk plan in Phase 20 (Pitfall 1 in research is "Window-Index Split
Leakage"). Getting D-15 wrong causes ~10-20% inflated val/test metrics that hide deployment failure.
The Wave 0 test_no_file_leakage_across_splits is the regression-prevention oracle.

Output: New WindowedHFDroneDataset class, new split_file_indices helper, all sliding-window unit
tests pass.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@src/acoustic/training/hf_dataset.py
@src/acoustic/training/parquet_dataset.py
@tests/unit/test_sliding_window_dataset.py
@tests/unit/test_hf_dataset.py
@tests/unit/test_parquet_dataset.py

<interfaces>
Existing HFDroneDataset (src/acoustic/training/hf_dataset.py:25):
- __init__(self, hf_dataset, mel_config, augmentation, ...)
- __len__ returns total file count
- __getitem__(idx) decodes WAV bytes from hf_dataset[idx]["audio"]["bytes"], extracts random 0.5s segment

DADS clip assumption (Research A1): clips are uniform 1s @ 16 kHz = 16000 samples. If wrong,
fallback is per-row probe at init time. For Phase 20 we ASSUME uniform and add a sanity assertion
in __init__ that aborts loudly if a probed clip differs.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add split_file_indices helper to parquet_dataset.py</name>
  <files>
    src/acoustic/training/parquet_dataset.py
  </files>
  <read_first>
    src/acoustic/training/parquet_dataset.py,
    tests/unit/test_sliding_window_dataset.py,
    tests/unit/test_parquet_dataset.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    After this task:
    - `from src.acoustic.training.parquet_dataset import split_file_indices` works
    - `split_file_indices(num_files=100, seed=42, train=0.7, val=0.15)` returns three disjoint lists
      with sizes 70, 15, 15 whose union covers all 100 file indices exactly once
    - Same seed → deterministic output across runs
    - All Wave 0 tests that import this helper turn GREEN
  </behavior>
  <action>
    Append (do not modify existing functions) to src/acoustic/training/parquet_dataset.py:

    ```python
    import random as _random  # avoid clash if numpy imported elsewhere

    def split_file_indices(
        num_files: int,
        seed: int = 42,
        train: float = 0.70,
        val: float = 0.15,
    ) -> tuple[list[int], list[int], list[int]]:
        """Session-level (file-level) split (Phase 20 D-15).

        Operates on FILE indices, not window indices. This is the ONLY correct way
        to split a sliding-window dataset because adjacent overlapping windows from
        the same source file would otherwise leak across splits and inflate val/test
        metrics by 10-20% (compass doc §4 "Data splitting: session-level grouping is
        non-negotiable"; Plötz 2021).

        Returns three DISJOINT lists of file indices:
            (train_files, val_files, test_files)
        whose union is exactly range(num_files).
        """
        if not (0.0 < train < 1.0 and 0.0 < val < 1.0 and train + val < 1.0):
            raise ValueError(f"invalid train/val ratios: train={train}, val={val}")
        files = list(range(num_files))
        rng = _random.Random(seed)
        rng.shuffle(files)
        n_tr = int(num_files * train)
        n_va = int(num_files * val)
        train_files = files[:n_tr]
        val_files = files[n_tr : n_tr + n_va]
        test_files = files[n_tr + n_va :]
        return train_files, val_files, test_files
    ```

    Do NOT touch the existing `split_indices` function — leave for backward compatibility.
  </action>
  <verify>
    <automated>pytest tests/unit/test_parquet_dataset.py -x -q && python -c "from src.acoustic.training.parquet_dataset import split_file_indices; tr, va, te = split_file_indices(100, seed=42); assert len(tr)==70 and len(va)==15 and len(te)==15 and set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "def split_file_indices" src/acoustic/training/parquet_dataset.py` returns one match
    - `grep -n "session-level grouping" src/acoustic/training/parquet_dataset.py` confirms documentation comment
    - The python -c assertion above exits 0
    - Existing test_parquet_dataset.py still passes
  </acceptance_criteria>
  <done>
    split_file_indices function exists, deterministic + disjoint, no regressions in v6 split logic.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement WindowedHFDroneDataset with leakage-safe enumeration</name>
  <files>
    src/acoustic/training/hf_dataset.py
  </files>
  <read_first>
    src/acoustic/training/hf_dataset.py,
    src/acoustic/training/parquet_dataset.py,
    tests/unit/test_sliding_window_dataset.py,
    tests/unit/test_hf_dataset.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    After this task:
    - `from src.acoustic.training.hf_dataset import WindowedHFDroneDataset` works
    - `WindowedHFDroneDataset(hf_ds, file_indices=[0,1,2], window_samples=8000, hop_samples=3200, ...)`
      builds an internal `_items: list[tuple[int, int]]` mapping flat idx → (file_idx, window_offset)
    - `__len__()` returns len(_items)
    - For a uniform 16000-sample-per-file dataset with window=8000, hop=3200: 3 windows per file
    - For hop_samples == window_samples (test split): 1 window per 16000-sample file (or 2 if file has 16000 exactly)
    - `__getitem__(idx)` returns the same (mel/tensor, label) tuple shape as the legacy HFDroneDataset
    - Augmentation is applied AT 16 kHz BEFORE mel computation (matches existing pattern)
    - All Wave 0 sliding-window tests turn GREEN, including test_no_file_leakage_across_splits
  </behavior>
  <action>
    Append a new class `WindowedHFDroneDataset` to src/acoustic/training/hf_dataset.py — do NOT
    modify the existing HFDroneDataset (keep for v5/v6 reproducibility per Research §State of the Art).

    Use this implementation skeleton (window math + indexing pattern from Research §Sliding-Window
    Dataset) — fill in the same audio decode + mel + return-shape logic as the legacy class:

    ```python
    class WindowedHFDroneDataset(Dataset):
        """Sliding-window dataset for Phase 20 v7 training (D-13..D-16).

        Replaces HFDroneDataset's random 0.5s segment extraction with deterministic
        sliding-window enumeration. CRITICAL: must be constructed with FILE INDICES
        (from parquet_dataset.split_file_indices), not window indices, to preserve
        session-level split isolation (D-15). Adjacent overlapping windows from the
        same source file would otherwise leak across splits.

        Window math:
            num_windows_per_file = max(1, 1 + (n_samples - window_samples) // hop_samples)
        For DADS uniform 1s @ 16k clips, window=8000, hop=3200:
            num_windows = 1 + (16000-8000)//3200 = 3 windows per file (60% overlap, D-13)
        For test split with hop=8000:
            num_windows = 1 + (16000-8000)//8000 = 2 windows per file (no overlap, D-16)
        """

        def __init__(
            self,
            hf_dataset,
            file_indices: list[int],
            window_samples: int = 8000,
            hop_samples: int = 3200,
            mel_config=None,
            augmentation=None,
            assumed_clip_samples: int = 16000,
            sample_rate: int = 16000,
        ) -> None:
            self._hf = hf_dataset
            self._window_samples = int(window_samples)
            self._hop_samples = int(hop_samples)
            self._mel_config = mel_config
            self._aug = augmentation
            self._sr = int(sample_rate)
            self._assumed_clip_samples = int(assumed_clip_samples)

            # Build flat index list of (file_idx, window_offset)
            self._items: list[tuple[int, int]] = []
            self._labels_cache: list[int] = []
            all_labels = list(hf_dataset["label"])

            n = self._assumed_clip_samples
            num_w = max(1, 1 + max(0, (n - self._window_samples)) // self._hop_samples)
            for file_idx in file_indices:
                label_int = int(all_labels[file_idx])
                for w in range(num_w):
                    self._items.append((int(file_idx), w * self._hop_samples))
                    self._labels_cache.append(label_int)

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            file_idx, offset = self._items[idx]
            audio = self._decode_audio(self._hf[file_idx])
            segment = audio[offset : offset + self._window_samples]
            if len(segment) < self._window_samples:
                # zero-pad
                pad = np.zeros(self._window_samples - len(segment), dtype=np.float32)
                segment = np.concatenate([segment, pad])
            if self._aug is not None:
                segment = self._aug(segment)
            mel = self._to_mel(segment)
            label = self._labels_cache[idx]
            return mel, label

        def _decode_audio(self, row) -> np.ndarray:
            # Reuse the same WAV-bytes decode pattern as the legacy HFDroneDataset.
            # Match the existing module-level helper (search for "decode" in the file)
            # to keep behavior identical.
            ...

        def _to_mel(self, segment: np.ndarray):
            # Reuse the existing mel computation hook used by HFDroneDataset.
            ...
    ```

    For `_decode_audio` and `_to_mel`, copy the implementation lines from the existing
    HFDroneDataset class — do not invent a new path. The intent is "same downstream tensor
    shape, different upstream enumeration".

    Add the file-leakage assertion as a class-level comment AND in __init__ assert:
    `assert len(set(file_indices)) == len(file_indices), "duplicate file indices not allowed"`
  </action>
  <verify>
    <automated>pytest tests/unit/test_sliding_window_dataset.py tests/unit/test_hf_dataset.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "class WindowedHFDroneDataset" src/acoustic/training/hf_dataset.py` returns one match
    - `grep -n "session-level split" src/acoustic/training/hf_dataset.py` confirms doc comment
    - `grep -n "self._items" src/acoustic/training/hf_dataset.py` shows the (file_idx, offset) list
    - `pytest tests/unit/test_sliding_window_dataset.py::test_no_file_leakage_across_splits -x -q` exits 0 (CRITICAL — D-15 oracle)
    - `pytest tests/unit/test_sliding_window_dataset.py -x -q` exits 0 (all five sliding-window tests GREEN)
    - `pytest tests/unit/test_hf_dataset.py -x -q` exits 0 (no v6 regression)
    - WindowedHFDroneDataset DOES NOT modify or shadow the existing HFDroneDataset class (verify with grep that both `class HFDroneDataset` AND `class WindowedHFDroneDataset` exist)
  </acceptance_criteria>
  <done>
    WindowedHFDroneDataset implemented; sliding-window enumeration produces correct counts; file-level
    leakage test passes; legacy HFDroneDataset untouched.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| HF dataset → in-memory Arrow | DADS dataset is HuggingFace-hosted; standard ML data trust boundary. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-03-01 | Tampering | DADS HF dataset bytes | accept | Public dataset, downloaded via HF datasets library which validates archive integrity. Phase 13 already accepted this. |
| T-20-03-02 | Information Disclosure | None | accept | Dataset is fully public; no PII handling. |
| T-20-03-03 | Repudiation (validation gap) | session-level split correctness | mitigate | test_no_file_leakage_across_splits is the explicit oracle; tested in CI on every commit. |
</threat_model>

<verification>
- All five sliding-window tests pass (especially the file-leakage one)
- split_file_indices is deterministic and disjoint
- Legacy HFDroneDataset untouched; v6 tests still GREEN
</verification>

<success_criteria>
- `pytest tests/unit/test_sliding_window_dataset.py tests/unit/test_hf_dataset.py tests/unit/test_parquet_dataset.py -x -q` exits 0
- `grep -c "class HFDroneDataset\|class WindowedHFDroneDataset" src/acoustic/training/hf_dataset.py` returns 2
- `grep -c "def split_file_indices\|def split_indices" src/acoustic/training/parquet_dataset.py` returns ≥2 (both helpers present)
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-03-SUMMARY.md`
</output>
