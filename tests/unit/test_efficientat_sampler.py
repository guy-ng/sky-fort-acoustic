"""Quick task 260407-nir: sampler must cover every __getitem__ index.

Regression guard for Phase 20 v7: the previous sampler used a file-level label
list and ``num_samples=len(train_lbl)``, which capped each epoch at file
count and threw away 2/3 of ``WindowedHFDroneDataset``'s windows every epoch.
The refactored ``_build_weighted_sampler`` reads labels from the dataset
directly so coverage matches ``len(dataset)`` for every path.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from acoustic.training import hf_dataset as hf_dataset_module
from acoustic.training.efficientat_trainer import (
    _LazyEfficientATDataset,
    _build_weighted_sampler,
)
from acoustic.training.hf_dataset import WindowedHFDroneDataset


class _FakeHFDataset:
    """Minimal HF dataset stub (mirrors the helper in
    test_windowed_dataset_non_uniform.py but kept local so the tests are
    independent and can be read on their own).
    """

    def __init__(self, labels: list[int]) -> None:
        self._labels = labels

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return list(self._labels)
            raise KeyError(key)
        return {"audio": {"bytes": f"row-{key}".encode()}, "label": self._labels[key]}

    def __len__(self) -> int:
        return len(self._labels)


def test_sampler_covers_all_windows_for_windowed_dataset(monkeypatch):
    """Windowed path: sampler must draw `len(train_ds)` per epoch.

    10 uniform files × 3 windows/file = 30 items. The old sampler drew only
    10 items per epoch (file count). The refactored sampler draws 30.
    """
    n_files = 10
    uniform_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(
        hf_dataset_module,
        "decode_wav_bytes",
        lambda _wav_bytes: uniform_audio,
    )

    labels = [i % 2 for i in range(n_files)]  # alternating 0/1
    fake_ds = _FakeHFDataset(labels=labels)

    train_ds = WindowedHFDroneDataset(
        hf_dataset=fake_ds,
        file_indices=list(range(n_files)),
        window_samples=8000,
        hop_samples=3200,
        assumed_clip_samples=16000,
    )

    # Sanity: 1 + (16000-8000)//3200 = 3 windows per file
    assert len(train_ds) == n_files * 3 == 30

    sampler = _build_weighted_sampler(train_ds)

    # The critical assertion that would have caught the bug:
    assert sampler.num_samples == 30, (
        f"sampler.num_samples={sampler.num_samples} but dataset has "
        f"{len(train_ds)} items — the v7 regression is back"
    )
    assert len(sampler.weights) == 30

    # Weights must be inverse-frequency over WINDOW-level labels
    # (not file-level). Each label appears 5 files × 3 windows = 15 times.
    window_labels = train_ds.labels
    window_counts = Counter(window_labels)
    assert window_counts[0] == 15
    assert window_counts[1] == 15
    for i, label in enumerate(window_labels):
        expected_weight = 1.0 / window_counts[label]
        assert abs(float(sampler.weights[i]) - expected_weight) < 1e-9


def test_sampler_is_noop_for_legacy_file_dataset(monkeypatch):
    """Legacy per-file path: refactor must produce identical weights + num_samples.

    Guards against any drift where the refactor accidentally changes behavior
    for the v6-style `_LazyEfficientATDataset` path.
    """
    # Stub decode_wav_bytes so _LazyEfficientATDataset can be constructed; it
    # only calls decode during __getitem__ (which we don't exercise here).
    monkeypatch.setattr(
        hf_dataset_module,
        "decode_wav_bytes",
        lambda _wav_bytes: np.zeros(16000, dtype=np.float32),
    )

    labels = [0, 1, 0, 1, 0]  # 3 × 0, 2 × 1
    fake_ds = _FakeHFDataset(labels=labels)

    train_ds = _LazyEfficientATDataset(
        hf_dataset=fake_ds,
        split_indices=list(range(5)),
        labels=labels,
        segment_samples=16000,
    )

    assert len(train_ds) == 5

    sampler = _build_weighted_sampler(train_ds)
    assert sampler.num_samples == 5
    assert len(sampler.weights) == 5

    # Manually reproduce the old expression to prove equivalence
    expected_weights = [1.0 / max(1, labels.count(l)) for l in labels]
    for i, expected in enumerate(expected_weights):
        assert abs(float(sampler.weights[i]) - expected) < 1e-9, (
            f"legacy path drifted: index {i} expected {expected} got {sampler.weights[i]}"
        )


def test_sampler_handles_single_class_gracefully(monkeypatch):
    """Edge case: all samples in one class.

    `max(1, count)` protects against division by zero if some label has zero
    occurrences in the `Counter`, but the main concern is the all-same-class
    case producing finite uniform weights (each weight = 1/N).
    """
    monkeypatch.setattr(
        hf_dataset_module,
        "decode_wav_bytes",
        lambda _wav_bytes: np.zeros(16000, dtype=np.float32),
    )

    labels = [1] * 7
    fake_ds = _FakeHFDataset(labels=labels)
    train_ds = _LazyEfficientATDataset(
        hf_dataset=fake_ds,
        split_indices=list(range(7)),
        labels=labels,
        segment_samples=16000,
    )

    sampler = _build_weighted_sampler(train_ds)
    assert sampler.num_samples == 7
    for w in sampler.weights:
        assert torch.isfinite(torch.as_tensor(float(w)))
        assert abs(float(w) - (1.0 / 7.0)) < 1e-9
