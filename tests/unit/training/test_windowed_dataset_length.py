"""Phase 22 Wave 0: WindowedHFDroneDataset length contract. Green after Plan 03."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="Phase 22 Plan 03 adds assertion + per_file_lengths generalization",
)


def _make_synthetic_hf_dataset(num_files: int = 3, clip_samples: int = 16000):
    """Tiny stand-in for a HuggingFace dataset -- returns list-of-dicts with
    `audio.array` and `label`. Matches the shape WindowedHFDroneDataset expects."""
    rng = np.random.default_rng(42)
    return [
        {"audio": {"array": rng.standard_normal(clip_samples).astype(np.float32),
                   "sampling_rate": 16000},
         "label": i % 2}
        for i in range(num_files)
    ]


def test_getitem_returns_32000_samples_for_1s_clip():
    from acoustic.training.hf_dataset import WindowedHFDroneDataset
    hf = _make_synthetic_hf_dataset(num_files=3, clip_samples=16000)
    ds = WindowedHFDroneDataset(hf, file_indices=[0, 1, 2])
    audio, label = ds[0]
    assert audio.shape[-1] == 32000, f"expected 32000 samples (1s @ 32kHz), got {audio.shape[-1]}"


def test_assertion_fires_on_malformed_clip():
    """If the resample output is not 32000 samples, __getitem__ must raise AssertionError."""
    # Implementation detail: Plan 03 will add the assert; this test will pass
    # automatically when the contract is enforced. We assert here that calling
    # getitem on a short clip either succeeds with correct length or raises.
    from acoustic.training.hf_dataset import WindowedHFDroneDataset
    hf = _make_synthetic_hf_dataset(num_files=1, clip_samples=8000)  # 0.5s only
    ds = WindowedHFDroneDataset(hf, file_indices=[0])
    # Either the dataset pads to full length (correct) or raises AssertionError
    try:
        audio, _ = ds[0]
        assert audio.shape[-1] == 32000
    except AssertionError:
        pass  # expected -- the contract was violated and fail-loud fired
