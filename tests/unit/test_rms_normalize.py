"""Unit tests for the shared ``_rms_normalize`` helper (Phase 20 D-34).

The helper is the single source of truth for per-sample RMS normalization
used on BOTH the training dataset path and ``RawAudioPreprocessor.process()``.
"""

from __future__ import annotations

import numpy as np
import torch

from acoustic.classification.preprocessing import _rms_normalize


def _rms_np(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _rms_torch(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x * x)).item())


def test_numpy_input_rms_matches_target() -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000).astype(np.float32) * 5.0
    out = _rms_normalize(audio, target=0.1)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert abs(_rms_np(out) - 0.1) < 1e-5


def test_torch_input_rms_matches_target_and_type() -> None:
    rng = np.random.default_rng(1)
    audio = torch.from_numpy(
        (rng.standard_normal(16000).astype(np.float32) * 0.002)
    )
    out = _rms_normalize(audio, target=0.1)
    assert isinstance(out, torch.Tensor)
    assert abs(_rms_torch(out) - 0.1) < 1e-5


def test_silence_unchanged_numpy() -> None:
    audio = np.zeros(8000, dtype=np.float32)
    out = _rms_normalize(audio, target=0.1)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, audio)


def test_silence_unchanged_torch() -> None:
    audio = torch.zeros(8000, dtype=torch.float32)
    out = _rms_normalize(audio, target=0.1)
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, audio)


def test_sub_eps_signal_unchanged() -> None:
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(4000).astype(np.float32) * 1e-9
    # RMS << eps=1e-6
    out = _rms_normalize(audio, target=0.1, eps=1e-6)
    assert np.array_equal(out, audio)


def test_idempotent() -> None:
    rng = np.random.default_rng(3)
    audio = rng.standard_normal(16000).astype(np.float32) * 2.5
    once = _rms_normalize(audio, target=0.1)
    twice = _rms_normalize(once, target=0.1)
    assert np.allclose(once, twice, atol=1e-6)
    assert abs(_rms_np(twice) - 0.1) < 1e-5


def test_target_override() -> None:
    rng = np.random.default_rng(4)
    audio = rng.standard_normal(16000).astype(np.float32) * 0.03
    out = _rms_normalize(audio, target=0.5)
    assert abs(_rms_np(out) - 0.5) < 1e-5
