"""D-04 drift guard: ensure apps/rpi-edge/ stays parity-locked to training-side preprocess.

This test lives in the MAIN repo CI path (tests/integration/) so any PR that modifies
src/acoustic/classification/efficientat/preprocess.py or mel_banks_128_1024_32k.pt will
re-run the numerical parity check against the vendored numpy port in apps/rpi-edge/
and fail loudly if they diverge, forcing the author to re-sync the port (and its
golden-parity test).

Unlike a byte-identity test (which would fail by construction since the vendored copy is
a numpy rewrite, not a clone), this asserts numerical equivalence on the golden fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_PREPROCESS = REPO_ROOT / "src/acoustic/classification/efficientat/preprocess.py"
TRAINING_MEL_BANKS = REPO_ROOT / "src/acoustic/classification/efficientat/mel_banks_128_1024_32k.pt"
EDGE_PREPROCESS = REPO_ROOT / "apps/rpi-edge/skyfort_edge/preprocess.py"
EDGE_MEL_BANKS = REPO_ROOT / "apps/rpi-edge/skyfort_edge/mel_banks_128_1024_32k.npy"
FIXTURE = REPO_ROOT / "apps/rpi-edge/tests/fixtures/golden_drone_1s_48k.wav"


def test_training_preprocess_and_mel_banks_exist():
    assert TRAINING_PREPROCESS.exists(), f"missing {TRAINING_PREPROCESS}"
    assert TRAINING_MEL_BANKS.exists(), (
        f"missing {TRAINING_MEL_BANKS} -- the training-side mel filterbank must be present "
        "for the drift guard to run. It is gitignored (*.pt) so ensure it has been regenerated."
    )
    assert EDGE_PREPROCESS.exists(), f"missing {EDGE_PREPROCESS}"
    assert EDGE_MEL_BANKS.exists(), f"missing {EDGE_MEL_BANKS}"


def test_edge_mel_banks_numpy_matches_training_pt():
    """Confirm the vendored .npy filterbank matches the training .pt tensor bit-for-bit."""
    if not TRAINING_MEL_BANKS.exists():
        pytest.skip(f"training mel_banks .pt not present on disk: {TRAINING_MEL_BANKS}")
    torch = pytest.importorskip("torch")
    training = (
        torch.load(str(TRAINING_MEL_BANKS), map_location="cpu", weights_only=True)
        .numpy()
        .astype(np.float32)
    )
    edge = np.load(str(EDGE_MEL_BANKS))
    assert training.shape == edge.shape, (
        f"shape drift: training {training.shape} vs edge {edge.shape} -- "
        "re-run the .pt->.npy conversion in apps/rpi-edge/skyfort_edge/"
    )
    np.testing.assert_array_equal(
        training,
        edge,
        err_msg=(
            "DRIFT DETECTED in mel_banks. Re-sync apps/rpi-edge/skyfort_edge/"
            "mel_banks_128_1024_32k.npy from src/acoustic/classification/efficientat/"
            "mel_banks_128_1024_32k.pt via torch.load(...).numpy().astype(np.float32)."
        ),
    )


def test_edge_numpy_preprocess_parity_against_training_torch_reference():
    """Re-run the numerical parity check at the main-repo CI level (drift guard)."""
    if not TRAINING_MEL_BANKS.exists():
        pytest.skip(f"training mel_banks .pt not present on disk: {TRAINING_MEL_BANKS}")
    torch = pytest.importorskip("torch")

    edge_root = REPO_ROOT / "apps/rpi-edge"
    if str(edge_root) not in sys.path:
        sys.path.insert(0, str(edge_root))

    from skyfort_edge.preprocess import NumpyMelSTFT  # noqa: E402
    from scipy.signal import resample_poly  # noqa: E402

    from src.acoustic.classification.efficientat.preprocess import (  # noqa: E402
        AugmentMelSTFT,
    )

    audio_48k, sr = sf.read(FIXTURE, dtype="float32")
    assert sr == 48000, f"expected 48k fixture, got sr={sr}"
    audio_32k = resample_poly(audio_48k, up=2, down=3).astype(np.float32)

    numpy_mel = NumpyMelSTFT().forward(audio_32k)

    torch_pp = AugmentMelSTFT(freqm=0, timem=0)
    torch_pp.eval()
    with torch.no_grad():
        torch_mel = (
            torch_pp(torch.from_numpy(audio_32k).unsqueeze(0)).squeeze(0).numpy()
        )

    np.testing.assert_allclose(
        numpy_mel,
        torch_mel,
        atol=1e-5,
        rtol=1e-4,
        err_msg=(
            "DRIFT DETECTED: apps/rpi-edge numpy preprocess no longer matches "
            "training-side torch preprocess. Re-sync "
            "apps/rpi-edge/skyfort_edge/preprocess.py and/or "
            "apps/rpi-edge/skyfort_edge/mel_banks_128_1024_32k.npy against "
            "src/acoustic/classification/efficientat/preprocess.py."
        ),
    )
