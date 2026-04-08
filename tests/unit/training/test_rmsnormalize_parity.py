"""Phase 22 Wave 0: training vs inference RMS normalization parity. Green after Plan 03."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="Phase 22 Plan 03 moves RmsNormalize post-resample",
)


def test_train_serve_rms_parity_within_1e4():
    """Load a fixture WAV, run both preprocessing paths, assert amplitude parity.

    Fixture path: tests/fixtures/efficientat_v8/parity_sample.wav (16kHz mono 1s).
    Plan 03 will create the fixture OR this test uses a synthetic tone.
    """
    from acoustic.training.hf_dataset import WindowedHFDroneDataset  # noqa
    from acoustic.classification.preprocessing import AudioPreprocessor  # noqa
    rng = np.random.default_rng(0)
    audio_16k = rng.standard_normal(16000).astype(np.float32) * 0.1
    # Training path (post-Plan-03): resample -> RmsNormalize -> waveform
    # Inference path: resample -> RmsNormalize -> waveform
    # Assert max|train - serve| < 1e-4
    pytest.skip("needs Plan 03 post_resample_norm hook + preprocessor access")
