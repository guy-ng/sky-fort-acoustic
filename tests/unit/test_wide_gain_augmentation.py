"""RED stubs for WideGainAugmentation (D-01..D-04).

These tests MUST currently fail with ImportError because
``WideGainAugmentation`` does not yet exist in
``acoustic.training.augmentation``. Plan 20-01 implements the class.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

def _import_cls():
    """Late import so pytest can still COLLECT this module while RED.

    Returns the WideGainAugmentation class. Will raise ImportError until
    Plan 20-01 ships the implementation.
    """
    from acoustic.training.augmentation import WideGainAugmentation

    return WideGainAugmentation


def test_emits_within_clipping_bounds(synthetic_waveform: np.ndarray) -> None:
    """Output must respect [-1, 1] clipping bound even at extreme gain (D-04)."""
    WideGainAugmentation = _import_cls()
    aug = WideGainAugmentation(wide_gain_db=40.0, p=1.0)
    out = aug(synthetic_waveform)
    assert float(np.max(np.abs(out))) <= 1.0


def test_gain_range_uniform(synthetic_waveform: np.ndarray) -> None:
    """Sampled gain must span at least [-30, +30] dB across 1000 calls (D-01).

    Subset of the configured ±40 dB range — this guards against degenerate
    sampling distributions.
    """
    WideGainAugmentation = _import_cls()
    aug = WideGainAugmentation(wide_gain_db=40.0, p=1.0)
    base_rms = float(np.sqrt(np.mean(synthetic_waveform**2)))
    observed_db = []
    for _ in range(1000):
        out = aug(synthetic_waveform)
        out_rms = float(np.sqrt(np.mean(out**2)))
        if out_rms > 1e-12 and base_rms > 1e-12:
            observed_db.append(20.0 * np.log10(out_rms / base_rms))
    assert min(observed_db) <= -30.0
    assert max(observed_db) >= 30.0


def test_probability_zero_pass_through(synthetic_waveform: np.ndarray) -> None:
    """p=0.0 must return the input unchanged (D-02)."""
    WideGainAugmentation = _import_cls()
    aug = WideGainAugmentation(wide_gain_db=40.0, p=0.0)
    out = aug(synthetic_waveform)
    assert np.array_equal(out, synthetic_waveform)


def test_dtype_preserved(synthetic_waveform: np.ndarray) -> None:
    """Float32 input must produce float32 output (D-03)."""
    WideGainAugmentation = _import_cls()
    aug = WideGainAugmentation(wide_gain_db=40.0, p=1.0)
    out = aug(synthetic_waveform.astype(np.float32))
    assert out.dtype == np.float32


def test_pickle_safe() -> None:
    """Object must be picklable for DataLoader num_workers > 0 safety."""
    WideGainAugmentation = _import_cls()
    aug = WideGainAugmentation(wide_gain_db=40.0, p=0.5)
    restored = pickle.loads(pickle.dumps(aug))
    assert restored is not None
