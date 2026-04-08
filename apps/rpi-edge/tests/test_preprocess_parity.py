"""D-04: numerical parity between vendored numpy preprocess and training-side torch reference.

Loads the golden 48 kHz WAV fixtures from 21-01, resamples to 32 kHz, runs both the
numpy port (`skyfort_edge.preprocess.NumpyMelSTFT`) and the training torch
`AugmentMelSTFT(freqm=0, timem=0).eval()`, and asserts np.allclose to atol=1e-5.

Torch is pytest.importorskip-guarded so the test still collects cleanly on a torch-less
Pi host; the main-repo CI copy in tests/integration/test_edge_preprocess_drift.py
enforces the check on every PR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from skyfort_edge.preprocess import NumpyMelSTFT

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "fixture_name", ["golden_drone_1s_48k.wav", "golden_silence_1s_48k.wav"]
)
def test_vendored_preprocess_matches_training_reference_within_atol_1e_5(fixture_name):
    audio_48k, sr = sf.read(FIXTURES / fixture_name, dtype="float32")
    assert sr == 48000, f"expected 48k fixture, got sr={sr}"
    audio_32k = resample_poly(audio_48k, up=2, down=3).astype(np.float32)

    # Numpy vendored port
    numpy_mel = NumpyMelSTFT().forward(audio_32k)

    # Torch training reference -- host-only import
    torch = pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.acoustic.classification.efficientat.preprocess import AugmentMelSTFT

    torch_preprocess = AugmentMelSTFT(freqm=0, timem=0)
    torch_preprocess.eval()
    with torch.no_grad():
        wave_t = torch.from_numpy(audio_32k).unsqueeze(0)  # (1, samples)
        torch_mel = torch_preprocess(wave_t).squeeze(0).numpy()  # (128, T)

    assert numpy_mel.shape == torch_mel.shape, (
        f"shape mismatch: numpy {numpy_mel.shape} vs torch {torch_mel.shape}"
    )
    np.testing.assert_allclose(numpy_mel, torch_mel, atol=1e-5, rtol=1e-4)
