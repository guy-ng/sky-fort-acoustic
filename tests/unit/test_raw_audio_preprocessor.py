"""Tests for RawAudioPreprocessor — RMS normalization contract (D-34)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acoustic.classification.preprocessing import RawAudioPreprocessor


def _mk_audio(rms: float, sr: int = 48000, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    audio = rng.standard_normal(n).astype(np.float32)
    current_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return (audio * (rms / current_rms)).astype(np.float32)


@pytest.mark.parametrize("input_rms", [0.001, 0.01, 0.1, 1.0, 10.0])
def test_process_normalizes_to_target_rms(input_rms: float) -> None:
    """D-34: regardless of input level, process() output RMS equals target (0.1)."""
    sr = 48000
    audio = _mk_audio(rms=input_rms, sr=sr, seed=42)
    pre = RawAudioPreprocessor(
        target_sr=32000, input_gain=1.0, rms_normalize_target=0.1
    )
    out = pre.process(audio, sr).numpy()
    out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert abs(out_rms - 0.1) < 1e-3, (
        f"input_rms={input_rms} → out_rms={out_rms}"
    )


def test_process_disabled_normalization_leaves_gain_unchanged() -> None:
    """Escape hatch: rms_normalize_target=None skips normalization entirely."""
    sr = 48000
    audio = _mk_audio(rms=0.05, sr=sr, seed=7)
    pre = RawAudioPreprocessor(
        target_sr=32000, input_gain=1.0, rms_normalize_target=None
    )
    out = pre.process(audio, sr).numpy()
    out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    # No normalization → RMS does NOT snap to the 0.1 target. Resampling
    # bleeds energy by ~20-25% but stays well below the normalize threshold.
    assert out_rms < 0.08, f"expected no normalization; got RMS={out_rms}"
    assert out_rms > 0.02, f"expected signal preserved; got RMS={out_rms}"


def test_process_silence_stays_silent() -> None:
    """Silence stays silence — no noise-floor amplification."""
    sr = 48000
    audio = np.zeros(sr, dtype=np.float32)
    pre = RawAudioPreprocessor(
        target_sr=32000, input_gain=1.0, rms_normalize_target=0.1
    )
    out = pre.process(audio, sr)
    assert torch.allclose(out, torch.zeros_like(out))


def test_process_returns_tensor() -> None:
    sr = 48000
    audio = _mk_audio(rms=0.3, sr=sr)
    pre = RawAudioPreprocessor(
        target_sr=32000, input_gain=1.0, rms_normalize_target=0.1
    )
    out = pre.process(audio, sr)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32


def test_default_cnn_rms_normalize_target_is_wired_through_settings() -> None:
    """Constructing via the default AcousticSettings lands at target=0.1."""
    from acoustic.config import AcousticSettings

    settings = AcousticSettings()
    assert settings.cnn_rms_normalize_target == 0.1
    assert settings.cnn_input_gain == 1.0  # D-34: legacy default retired
    pre = RawAudioPreprocessor(
        input_gain=settings.cnn_input_gain,
        rms_normalize_target=settings.cnn_rms_normalize_target,
    )
    sr = 48000
    audio = _mk_audio(rms=2.0, sr=sr, seed=11)
    out = pre.process(audio, sr).numpy()
    out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert abs(out_rms - 0.1) < 1e-3
