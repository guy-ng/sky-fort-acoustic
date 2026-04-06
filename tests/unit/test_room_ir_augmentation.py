"""RED stubs for RoomIRAugmentation (D-05..D-08).

These tests MUST currently fail with ImportError because
``RoomIRAugmentation`` does not yet exist in
``acoustic.training.augmentation``. Plan 20-01 implements the class.
"""

from __future__ import annotations

import pickle
import time

import numpy as np

def _import_cls():
    """Late import so pytest can still COLLECT this module while RED."""
    from acoustic.training.augmentation import RoomIRAugmentation

    return RoomIRAugmentation


def test_pool_built_at_init(synthetic_waveform: np.ndarray) -> None:
    """RoomIRAugmentation pre-builds a fixed pool of IRs at __init__ (D-06)."""
    RoomIRAugmentation = _import_cls()
    aug = RoomIRAugmentation(pool_size=8, p=1.0, max_order=10)
    # The pool is exposed via a public-ish attribute or property; tolerate either.
    pool = getattr(aug, "pool", None) or getattr(aug, "_pool", None)
    assert pool is not None, "RoomIRAugmentation must expose its IR pool"
    assert len(pool) == 8


def test_output_length_preserved(synthetic_waveform: np.ndarray) -> None:
    """Convolution output must be cropped to the input length (D-08)."""
    RoomIRAugmentation = _import_cls()
    aug = RoomIRAugmentation(pool_size=4, p=1.0, max_order=10)
    out = aug(synthetic_waveform)
    assert out.shape == synthetic_waveform.shape


def test_probability_zero_pass_through(synthetic_waveform: np.ndarray) -> None:
    """p=0.0 must return the input unchanged (D-07)."""
    RoomIRAugmentation = _import_cls()
    aug = RoomIRAugmentation(pool_size=4, p=0.0, max_order=10)
    out = aug(synthetic_waveform)
    assert np.array_equal(out, synthetic_waveform)


def test_dtype_preserved(synthetic_waveform: np.ndarray) -> None:
    """Float32 in → float32 out (D-08)."""
    RoomIRAugmentation = _import_cls()
    aug = RoomIRAugmentation(pool_size=4, p=1.0, max_order=10)
    out = aug(synthetic_waveform.astype(np.float32))
    assert out.dtype == np.float32


def test_pickle_safe() -> None:
    """Object must be picklable for DataLoader num_workers > 0 safety."""
    RoomIRAugmentation = _import_cls()
    aug = RoomIRAugmentation(pool_size=4, p=0.7, max_order=10)
    restored = pickle.loads(pickle.dumps(aug))
    assert restored is not None


def test_max_order_bounded() -> None:
    """Pool of 8 IRs at max_order=10 must build in <30s (Research Pitfall 3)."""
    RoomIRAugmentation = _import_cls()
    start = time.monotonic()
    RoomIRAugmentation(pool_size=8, p=1.0, max_order=10)
    elapsed = time.monotonic() - start
    assert elapsed < 30.0, f"pool build too slow: {elapsed:.1f}s"
