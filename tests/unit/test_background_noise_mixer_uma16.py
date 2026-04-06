"""RED stubs for BackgroundNoiseMixer UMA-16 ambient extensions (D-10..D-12).

Plan 20-02 will extend ``BackgroundNoiseMixer`` (or add a sibling class) to
support per-source SNR ranges and a pure-negative pass-through branch for
UMA-16 ambient noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_uma16_ambient_dir_accepted(temp_noise_dir: Path) -> None:
    """BackgroundNoiseMixer must accept the uma16_ambient sub-dir as a noise source."""
    from acoustic.training.augmentation import BackgroundNoiseMixer

    # Phase 20: BackgroundNoiseMixer must accept a per-dir SNR map (new kwarg).
    mixer = BackgroundNoiseMixer(
        noise_dirs=[temp_noise_dir],
        uma16_ambient_dir=temp_noise_dir,  # NEW kwarg in Plan 20-02
        uma16_snr_range=(-5.0, 15.0),
        pure_negative_ratio=0.10,
        sample_rate=16000,
        p=1.0,
    )
    mixer.warm_cache()
    assert hasattr(mixer, "_uma16_files")


def test_uma16_specific_snr_range(temp_noise_dir: Path) -> None:
    """When source is the uma16_ambient subdir the SNR must come from
    (-5, 15) not the default (-10, 20) (D-11)."""
    from acoustic.training.augmentation import BackgroundNoiseMixer

    mixer = BackgroundNoiseMixer(
        noise_dirs=[temp_noise_dir],
        uma16_ambient_dir=temp_noise_dir,
        uma16_snr_range=(-5.0, 15.0),
        pure_negative_ratio=0.0,
        sample_rate=16000,
        p=1.0,
    )
    assert mixer._uma16_snr_range == (-5.0, 15.0)


def test_pure_negative_branch(temp_noise_dir: Path) -> None:
    """pure_negative_ratio=0.10 → ~10% of label-0 samples returned as raw
    UMA-16 ambient with no drone mix (D-12)."""
    from acoustic.training.augmentation import BackgroundNoiseMixer

    mixer = BackgroundNoiseMixer(
        noise_dirs=[temp_noise_dir],
        uma16_ambient_dir=temp_noise_dir,
        uma16_snr_range=(-5.0, 15.0),
        pure_negative_ratio=0.10,
        sample_rate=16000,
        p=1.0,
    )
    mixer.warm_cache()
    # The mixer must expose a method or branch that returns a pure negative
    # sample. Plan 20-02 names the method ``sample_pure_negative``.
    assert hasattr(mixer, "sample_pure_negative")
    sample = mixer.sample_pure_negative(n_samples=8000)
    assert isinstance(sample, np.ndarray)
    assert sample.dtype == np.float32
    assert sample.shape == (8000,)
