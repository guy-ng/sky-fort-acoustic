"""RED stubs for WindowedHFDroneDataset (D-13..D-16).

These tests MUST currently fail with ImportError because
``WindowedHFDroneDataset`` and ``split_file_indices`` do not yet exist.
Plan 20-03 implements them.

The most important test is ``test_no_file_leakage_across_splits`` which
encodes the Pitfall 1 contract from 20-RESEARCH.md: file-level splits must
NEVER leak across train/val/test boundaries when expanded to overlapping
windows.
"""

from __future__ import annotations

import pytest


def _import_dataset():
    """Late import so collection succeeds while RED."""
    from acoustic.training.hf_dataset import WindowedHFDroneDataset

    return WindowedHFDroneDataset


def _import_split_file_indices():
    from acoustic.training.parquet_dataset import split_file_indices

    return split_file_indices


def test_window_count_for_uniform_clip() -> None:
    """A 16000-sample file with window=8000 hop=3200 yields 3 windows.

    1 + (16000 - 8000) // 3200 = 1 + 2 = 3
    """
    WindowedHFDroneDataset = _import_dataset()
    ds = WindowedHFDroneDataset(
        file_lengths=[16000],
        labels=[1],
        window_samples=8000,
        hop_samples=3200,
    )
    assert len(ds) == 3


def test_idx_mapping_consistent() -> None:
    """__getitem__ must round-trip via _items[idx] == (file_idx, offset)."""
    WindowedHFDroneDataset = _import_dataset()
    ds = WindowedHFDroneDataset(
        file_lengths=[16000, 32000],
        labels=[0, 1],
        window_samples=8000,
        hop_samples=8000,
    )
    items = ds._items
    assert items[0][0] == 0  # first window in first file
    assert items[0][1] == 0
    assert items[-1][0] == 1  # last window in second file


def test_no_file_leakage_across_splits() -> None:
    """CRITICAL (D-15, Research Pitfall 1): split-by-file boundary holds.

    Build a 100-file dataset, split file indices 70/15/15, expand to windows,
    then assert that the file-index sets for train/val/test are pairwise
    disjoint AND that every window's file_idx belongs to its split's file set.
    """
    WindowedHFDroneDataset = _import_dataset()
    split_file_indices = _import_split_file_indices()

    n_files = 100
    file_lengths = [16000] * n_files
    labels = [i % 2 for i in range(n_files)]

    train_files, val_files, test_files = split_file_indices(
        n_files=n_files, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Pairwise disjoint
    assert set(train_files).isdisjoint(set(val_files))
    assert set(train_files).isdisjoint(set(test_files))
    assert set(val_files).isdisjoint(set(test_files))

    train_ds = WindowedHFDroneDataset(
        file_lengths=[file_lengths[i] for i in train_files],
        labels=[labels[i] for i in train_files],
        window_samples=8000,
        hop_samples=3200,
        file_id_map=train_files,
    )
    # Every window's file_idx must belong to train_files
    for file_idx, _offset in train_ds._items:
        assert file_idx in set(train_files)


def test_test_split_no_overlap() -> None:
    """Test split must have hop_samples == window_samples (D-16)."""
    WindowedHFDroneDataset = _import_dataset()
    ds = WindowedHFDroneDataset(
        file_lengths=[16000],
        labels=[1],
        window_samples=8000,
        hop_samples=8000,
    )
    # 1 + (16000 - 8000) // 8000 = 1 + 1 = 2
    assert len(ds) == 2


def test_train_val_overlap() -> None:
    """Train/val splits use hop < window (D-13/D-14)."""
    WindowedHFDroneDataset = _import_dataset()
    ds = WindowedHFDroneDataset(
        file_lengths=[16000],
        labels=[1],
        window_samples=8000,
        hop_samples=3200,
    )
    assert len(ds) > 2  # overlap means more windows than non-overlap
