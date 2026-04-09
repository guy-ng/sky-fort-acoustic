"""Phase 22 Wave 2 (Plan 03): WindowedHFDroneDataset length contract + per_file_lengths.

Guardrail tests driving the Plan 03 hardening of ``WindowedHFDroneDataset``:
- ``__getitem__`` MUST return a tensor of exactly ``EFFICIENTAT_SEGMENT_SAMPLES``
  (= 32000) samples, or raise ``AssertionError`` with the v7 regression signature.
- The class MUST accept ``per_file_lengths`` so multi-second field recordings
  (2026-04-08 corpus) produce multiple sliding windows per file instead of
  silently truncating to the uniform DADS length assumption.
- The class MUST accept ``post_resample_norm`` so the trainer can push
  ``RmsNormalize`` into the 32 kHz domain (REQ-22-W4 train/serve parity).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch


def _encode_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode a float32 mono waveform to in-memory WAV bytes.

    Uses ``soundfile`` at PCM_16 because ``parquet_dataset.decode_wav_bytes``
    skips the 44-byte header and reads raw int16 PCM. Matches the HF drone
    dataset wire format exactly.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _make_synthetic_hf_dataset(
    num_files: int = 3, clip_samples: int = 16000,
) -> list[dict]:
    """Tiny stand-in for an HF dataset -- list of dicts indexable by int.

    Each row matches the WindowedHFDroneDataset wire contract: ``row["audio"]``
    is a dict with ``bytes`` (raw WAV) and ``row["label"]`` is an int. Also
    exposes ``["label"]`` column access so ``list(hf_dataset["label"])``
    works in ``WindowedHFDroneDataset.__init__``.
    """
    rng = np.random.default_rng(42)
    rows = [
        {
            "audio": {
                "bytes": _encode_wav_bytes(
                    rng.standard_normal(clip_samples).astype(np.float32),
                    sr=16000,
                ),
                "sampling_rate": 16000,
            },
            "label": i % 2,
        }
        for i in range(num_files)
    ]

    class _HFShim(list):
        """list subclass that supports HF-style column access."""

        def __getitem__(self, key):  # type: ignore[override]
            if isinstance(key, str):
                return [r[key] for r in list.__iter__(self)]
            return list.__getitem__(self, key)

    return _HFShim(rows)


class TestLengthContract:
    """Plan 03 Task 1 — fail-loud length assertion + post-resample normalize."""

    def test_getitem_returns_32000_samples_for_1s_clip(self) -> None:
        from acoustic.classification.efficientat.window_contract import (
            EFFICIENTAT_SEGMENT_SAMPLES,
        )
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=3, clip_samples=16000)
        ds = WindowedHFDroneDataset(hf, file_indices=[0, 1, 2])
        audio, _label = ds[0]
        assert audio.shape[-1] == EFFICIENTAT_SEGMENT_SAMPLES, (
            f"expected {EFFICIENTAT_SEGMENT_SAMPLES} samples, got {audio.shape[-1]}"
        )
        assert audio.dtype == torch.float32

    def test_assertion_message_mentions_v7_regression(self) -> None:
        """Fail-loud assertion should mention the v7 signature for operators."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=1, clip_samples=16000)
        # Force a contract violation by constructing with an absurd window size
        # that bypasses the resample×2 invariant (window_samples=7000 @ 16k ->
        # 14000 samples @ 32k, NOT 32000). This simulates a trainer bug.
        ds = WindowedHFDroneDataset(
            hf,
            file_indices=[0],
            window_samples=7000,
            hop_samples=7000,
            assumed_clip_samples=16000,
        )
        with pytest.raises(AssertionError, match="v7 train/serve mismatch"):
            ds[0]


class TestPostResampleNorm:
    """Plan 03 Task 1 — post_resample_norm hook runs in 32 kHz domain."""

    def test_post_resample_norm_runs_on_32khz_tensor(self) -> None:
        """Custom post_resample_norm sees the resampled tensor, returns 32000 samples."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        captured: dict = {}

        def capture_norm(arr: np.ndarray) -> np.ndarray:
            captured["length"] = len(arr)
            captured["dtype"] = arr.dtype
            # Scale by 2 so we can verify it actually ran
            return (arr * 2.0).astype(np.float32)

        hf = _make_synthetic_hf_dataset(num_files=1, clip_samples=16000)
        ds = WindowedHFDroneDataset(
            hf, file_indices=[0], post_resample_norm=capture_norm,
        )
        audio, _ = ds[0]
        assert captured["length"] == 32000, (
            f"post_resample_norm should see 32 kHz tensor (32000 samples), "
            f"got {captured.get('length')}"
        )
        assert audio.shape[-1] == 32000, "output still 32000 after norm"

    def test_none_post_resample_norm_preserves_behavior(self) -> None:
        """Default (None) post_resample_norm preserves legacy path."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=1, clip_samples=16000)
        ds_default = WindowedHFDroneDataset(hf, file_indices=[0])
        ds_explicit = WindowedHFDroneDataset(
            hf, file_indices=[0], post_resample_norm=None,
        )
        a1, _ = ds_default[0]
        a2, _ = ds_explicit[0]
        assert torch.allclose(a1, a2), (
            "post_resample_norm=None must not change output vs default"
        )


class TestPerFileLengths:
    """Plan 03 Task 1 — per_file_lengths generalizes multi-second clips."""

    def test_per_file_lengths_produces_sliding_windows(self) -> None:
        """A 2s (32k-sample) file at window=16k, hop=8k yields 3 windows."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        # File 0: 2s @ 16 kHz = 32000 samples → 3 windows at hop=8000
        # File 1: 1s @ 16 kHz = 16000 samples → 1 window
        hf = _make_synthetic_hf_dataset(num_files=2, clip_samples=32000)
        # Shim produces all 32000-sample rows; override per file via lengths
        # but we also need to rebuild row 1 at 16000 samples.
        rng = np.random.default_rng(7)
        hf[1] = {
            "audio": {
                "bytes": _encode_wav_bytes(
                    rng.standard_normal(16000).astype(np.float32),
                    sr=16000,
                ),
                "sampling_rate": 16000,
            },
            "label": 0,
        }

        ds = WindowedHFDroneDataset(
            hf,
            file_indices=[0, 1],
            window_samples=16000,
            hop_samples=8000,
            per_file_lengths=[32000, 16000],
        )
        # File 0: max(1, 1 + (32000-16000)//8000) = 1 + 2 = 3 windows
        # File 1: max(1, 1 + (16000-16000)//8000) = 1 window
        assert len(ds) == 4, f"expected 4 windows total, got {len(ds)}"

        # Every window must return the 32000-sample contract length
        for i in range(len(ds)):
            audio, _ = ds[i]
            assert audio.shape[-1] == 32000

    def test_per_file_lengths_length_mismatch_rejected(self) -> None:
        """Mismatched per_file_lengths vs file_indices must raise immediately."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=2, clip_samples=16000)
        with pytest.raises(AssertionError, match="per_file_lengths"):
            WindowedHFDroneDataset(
                hf,
                file_indices=[0, 1],
                per_file_lengths=[16000],  # length 1, should be 2
            )

    def test_legacy_uniform_path_still_works(self) -> None:
        """Without per_file_lengths, legacy DADS uniform path is preserved."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=4, clip_samples=16000)
        ds = WindowedHFDroneDataset(hf, file_indices=[0, 1, 2, 3])
        # DADS uniform 1s clips at window=16000 hop=8000 → 1 window per file
        assert len(ds) == 4
        for i in range(len(ds)):
            audio, _ = ds[i]
            assert audio.shape[-1] == 32000

    def test_per_file_lengths_labels_flat_list(self) -> None:
        """dataset.labels should expose one entry per window, not per file."""
        from acoustic.training.hf_dataset import WindowedHFDroneDataset

        hf = _make_synthetic_hf_dataset(num_files=2, clip_samples=32000)
        rng = np.random.default_rng(11)
        hf[1] = {
            "audio": {
                "bytes": _encode_wav_bytes(
                    rng.standard_normal(16000).astype(np.float32),
                    sr=16000,
                ),
                "sampling_rate": 16000,
            },
            "label": 1,  # file 0 label=0, file 1 label=1
        }

        ds = WindowedHFDroneDataset(
            hf,
            file_indices=[0, 1],
            window_samples=16000,
            hop_samples=8000,
            per_file_lengths=[32000, 16000],
        )
        labels = ds.labels
        assert len(labels) == len(ds) == 4
        # File 0 contributes 3 windows of label 0, file 1 contributes 1 window of label 1
        assert labels == [0, 0, 0, 1]
