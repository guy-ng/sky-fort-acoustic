"""Quick task 260407-jx8: WindowedHFDroneDataset must tolerate non-uniform clips.

Phase 20 v7 Vertex training crashed when DADS file_idx=174151 decoded to 8000
samples instead of the assumed 16000. The dataset now logs once and falls back
to pad_or_loop / truncate so training can finish.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from acoustic.training import hf_dataset as hf_dataset_module
from acoustic.training.hf_dataset import WindowedHFDroneDataset


class _FakeHFDataset:
    """Minimal stand-in for a HuggingFace Dataset object.

    Supports the two access patterns WindowedHFDroneDataset uses:
      - column access: ``hf_ds["label"]`` returns a list-like of all labels
      - row access:    ``hf_ds[i]`` returns a row dict shaped like a DADS row
                       with an ``audio.bytes`` field

    The audio bytes are an opaque token here — `decode_wav_bytes` is
    monkeypatched in the tests so the actual bytes never need to round-trip
    through a real WAV decoder.
    """

    def __init__(self, labels: list[int]) -> None:
        self._labels = labels

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return list(self._labels)
            raise KeyError(key)
        # Row access — file_idx -> dict
        return {"audio": {"bytes": f"row-{key}".encode()}, "label": self._labels[key]}

    def __len__(self) -> int:
        return len(self._labels)


@pytest.fixture
def fake_decoder(monkeypatch):
    """Patch decode_wav_bytes to return per-file canned waveforms.

    file_idx=0 -> uniform 16000 samples
    file_idx=1 -> short    8000 samples (the v7 outlier shape)
    file_idx=2 -> long    24000 samples (extra coverage for the truncate path)
    """

    canned: dict[int, np.ndarray] = {
        0: np.linspace(-0.5, 0.5, 16000, dtype=np.float32),
        1: np.linspace(-0.25, 0.25, 8000, dtype=np.float32),
        2: np.linspace(-0.1, 0.1, 24000, dtype=np.float32),
    }

    def fake_decode(wav_bytes: bytes) -> np.ndarray:
        # The fake row encodes file_idx in the bytes payload as "row-<idx>".
        idx = int(wav_bytes.decode().split("-", 1)[1])
        return canned[idx]

    monkeypatch.setattr(hf_dataset_module, "decode_wav_bytes", fake_decode)
    return canned


def test_non_uniform_clips_do_not_crash(fake_decoder, caplog):
    """Iterating the dataset over short/long clips must not raise.

    Window math (assumed 16000, window 8000, hop 3200) yields 3 windows per
    file. With 3 files that's 9 __getitem__ calls — every one must return a
    (mel, label) pair without raising AssertionError.
    """
    fake_ds = _FakeHFDataset(labels=[0, 1, 1])

    ds = WindowedHFDroneDataset(
        hf_dataset=fake_ds,
        file_indices=[0, 1, 2],
        window_samples=8000,
        hop_samples=3200,
        assumed_clip_samples=16000,
    )

    # 3 windows/file × 3 files
    assert len(ds) == 9

    with caplog.at_level(logging.WARNING, logger=hf_dataset_module.__name__):
        for k in range(len(ds)):
            features, label = ds[k]
            assert isinstance(features, torch.Tensor)
            assert features.shape == (1, 128, 64)
            assert isinstance(label, torch.Tensor)

    # First non-uniform clip must produce exactly one warning, then suppress.
    non_uniform_warnings = [
        rec for rec in caplog.records if "non-uniform clip detected" in rec.message
    ]
    assert len(non_uniform_warnings) == 1, (
        f"expected exactly one non-uniform warning, got {len(non_uniform_warnings)}"
    )


def test_uniform_clips_emit_no_warning(fake_decoder, caplog):
    """Uniform-only datasets must stay silent (no spurious warnings)."""
    fake_ds = _FakeHFDataset(labels=[0])

    ds = WindowedHFDroneDataset(
        hf_dataset=fake_ds,
        file_indices=[0],  # only the uniform clip
        window_samples=8000,
        hop_samples=3200,
        assumed_clip_samples=16000,
    )

    with caplog.at_level(logging.WARNING, logger=hf_dataset_module.__name__):
        for k in range(len(ds)):
            ds[k]

    assert not any(
        "non-uniform clip detected" in rec.message for rec in caplog.records
    )
