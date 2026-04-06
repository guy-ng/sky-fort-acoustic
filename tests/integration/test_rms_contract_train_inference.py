"""End-to-end RMS parity contract between train and inference paths (D-34).

This is the regression gate for Plan 20-08. If it fails, train/inference
amplitude has drifted again — either because somebody removed RmsNormalize
from a chain, changed the target, or reintroduced the old cnn_input_gain=500
hack. Re-read `.planning/debug/training-collapse-constant-output.md` before
"fixing" the test.

The three reference amplitudes mirror the empirical findings from
`scripts/verify_rms_domain_mismatch.py`:

  - 0.001  → near-silence floor (stays unchanged, below eps)
  - 0.18   → DADS-typical raw waveform RMS
  - 9.3    → live UMA-16 waveform AFTER the legacy 500x gain

All three MUST land at RMS=0.1 ± 1e-3 on every path that reaches the model —
except the sub-eps silence case, which must stay untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acoustic.classification.preprocessing import RawAudioPreprocessor
from acoustic.training.augmentation import ComposedAugmentation
from acoustic.training.config import TrainingConfig
from acoustic.training.efficientat_trainer import EfficientATTrainingRunner


_TARGET_RMS = 0.1
_TOL = 1e-3


def _mk(rms: float, n: int = 16000, seed: int = 0) -> np.ndarray:
    if rms == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(n).astype(np.float32)
    current = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return (audio * (rms / current)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _make_bare_config() -> TrainingConfig:
    """Build a TrainingConfig with every non-RMS augmentation disabled so the
    RMS contract is the only thing being measured."""
    return TrainingConfig(
        wide_gain_db=0.0,
        rir_enabled=False,
        use_audiomentations=False,
        noise_augmentation_enabled=False,
        noise_dirs=[],
    )


@pytest.mark.parametrize(
    "input_rms,expect_target",
    [
        (0.0, False),     # exact silence — stays unchanged (below eps=1e-6)
        (1e-8, False),    # sub-eps floor — stays unchanged
        (0.001, True),    # DADS no-drone-class typical (low but above eps)
        (0.18, True),     # DADS drone-class typical
        (9.3, True),      # live UMA-16 post-legacy-500x
    ],
)
def test_inference_path_lands_at_target(input_rms: float, expect_target: bool) -> None:
    """RawAudioPreprocessor.process() must land all non-silence inputs at the
    RMS target, regardless of legacy input_gain."""
    sr_in = 48000  # Live pipeline rate
    audio = _mk(rms=input_rms, n=sr_in, seed=123)
    pre = RawAudioPreprocessor(
        target_sr=32000, input_gain=1.0, rms_normalize_target=_TARGET_RMS
    )
    out = pre.process(audio, sr_in).numpy()
    out_rms = _rms(out)
    if expect_target:
        assert abs(out_rms - _TARGET_RMS) < _TOL, (
            f"inference path: input_rms={input_rms} → out_rms={out_rms}, "
            f"expected {_TARGET_RMS}±{_TOL}"
        )
    else:
        # Sub-eps short-circuit: silence stays silent — _rms_normalize
        # returns the input unchanged. We only assert it did NOT get
        # amplified up to the target.
        assert out_rms < 1e-5, (
            f"silence should stay silent; got {out_rms}"
        )


@pytest.mark.parametrize(
    "input_rms,expect_target",
    [
        (0.0, False),
        (1e-8, False),
        (0.001, True),
        (0.18, True),
        (9.3, True),
    ],
)
def test_train_chain_lands_at_target(input_rms: float, expect_target: bool) -> None:
    """The trainer's composed train augmentation chain must end at RMS=0.1
    for non-silence inputs."""
    cfg = _make_bare_config()
    runner = EfficientATTrainingRunner(config=cfg)
    chain: ComposedAugmentation = runner._build_train_augmentation()
    audio = _mk(rms=input_rms, n=16000, seed=456)
    out = chain(audio)
    out_rms = _rms(out)
    if expect_target:
        assert abs(out_rms - _TARGET_RMS) < _TOL, (
            f"train chain: input_rms={input_rms} → out_rms={out_rms}, "
            f"expected {_TARGET_RMS}±{_TOL}"
        )
    else:
        assert out_rms < _TARGET_RMS / 10.0, (
            f"silence should stay silent; got {out_rms}"
        )


@pytest.mark.parametrize(
    "input_rms,expect_target",
    [
        (0.0, False),
        (1e-8, False),
        (0.001, True),
        (0.18, True),
        (9.3, True),
    ],
)
def test_eval_chain_lands_at_target(input_rms: float, expect_target: bool) -> None:
    """The trainer's composed eval augmentation chain must end at RMS=0.1
    for non-silence inputs — eval is NOT exempt from the contract."""
    cfg = _make_bare_config()
    runner = EfficientATTrainingRunner(config=cfg)
    chain = runner._build_eval_augmentation()
    assert chain is not None, "eval chain must not be None (D-34)"
    audio = _mk(rms=input_rms, n=16000, seed=789)
    out = chain(audio)
    out_rms = _rms(out)
    if expect_target:
        assert abs(out_rms - _TARGET_RMS) < _TOL, (
            f"eval chain: input_rms={input_rms} → out_rms={out_rms}, "
            f"expected {_TARGET_RMS}±{_TOL}"
        )
    else:
        assert out_rms < _TARGET_RMS / 10.0, (
            f"silence should stay silent; got {out_rms}"
        )


def test_train_and_inference_use_identical_math() -> None:
    """Both paths must delegate to the same ``_rms_normalize`` helper so there
    is exactly one source of truth for the normalization math."""
    # Identity test: feed the same (post-resample, post-gain) waveform to
    # _rms_normalize directly and through the train chain — they must match.
    from acoustic.classification.preprocessing import _rms_normalize

    audio = _mk(rms=0.7, n=16000, seed=31415)
    direct = _rms_normalize(audio.copy(), target=_TARGET_RMS)
    cfg = _make_bare_config()
    runner = EfficientATTrainingRunner(config=cfg)
    chain = runner._build_train_augmentation()
    via_chain = chain(audio.copy())
    # Both should produce the same numeric result since no other stage is
    # active in the bare config.
    assert np.allclose(direct, via_chain, atol=1e-6), (
        "train chain RMS output drifted from direct _rms_normalize — math divergence"
    )
    # And it should match the inference path too (modulo resample bleed,
    # which is why we skip resample here by using sr_in == target_sr).
    pre = RawAudioPreprocessor(
        target_sr=16000, input_gain=1.0, rms_normalize_target=_TARGET_RMS
    )
    inf_out = pre.process(audio.copy(), 16000).numpy()
    assert abs(_rms(inf_out) - _rms(direct)) < 1e-6, (
        "inference path RMS diverged from direct _rms_normalize"
    )
