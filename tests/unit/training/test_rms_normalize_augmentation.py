"""Tests for the RmsNormalize augmentation and its wiring into the trainer.

Phase 20 D-34: RMS normalization must run so the model sees identical amplitude
distributions at train and inference time.

Phase 22 REQ-22-W4 (Plan 03): RmsNormalize was moved OUT of the 16 kHz
ComposedAugmentation chain and INTO ``WindowedHFDroneDataset``'s
``post_resample_norm`` hook so it runs in the 32 kHz domain — parity with
``RawAudioPreprocessor.process``. The semantic D-34 contract ("RMS is the last
waveform-domain op") is preserved; it now just lives on the dataset, not the
augmentation chain. Tests below reflect the new wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acoustic.training.augmentation import (
    ComposedAugmentation,
    RmsNormalize,
)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Direct class-level behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [0.003, 0.05, 0.5, 3.0])
def test_rms_normalize_snaps_to_target(amplitude: float) -> None:
    rng = np.random.default_rng(int(amplitude * 1000))
    n = 16000
    audio = rng.standard_normal(n).astype(np.float32)
    audio *= amplitude / _rms(audio)
    aug = RmsNormalize(target=0.1)
    out = aug(audio, sample_rate=16000)
    assert abs(_rms(out) - 0.1) < 1e-5


def test_rms_normalize_target_override() -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(8000).astype(np.float32) * 0.02
    aug = RmsNormalize(target=0.25)
    out = aug(audio, sample_rate=16000)
    assert abs(_rms(out) - 0.25) < 1e-5


def test_rms_normalize_silence_unchanged() -> None:
    audio = np.zeros(4000, dtype=np.float32)
    aug = RmsNormalize(target=0.1)
    out = aug(audio, sample_rate=16000)
    assert np.array_equal(out, audio)


def test_rms_normalize_inside_composed_augmentation() -> None:
    """RmsNormalize works when driven through ComposedAugmentation.__call__."""
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(8000).astype(np.float32) * 5.0
    composed = ComposedAugmentation([RmsNormalize(target=0.1)])
    out = composed(audio)
    assert abs(_rms(out) - 0.1) < 1e-5


def test_rms_normalize_is_picklable() -> None:
    """Must be pickle-safe for DataLoader num_workers>0 (Phase 20 D-34)."""
    import pickle

    aug = RmsNormalize(target=0.1)
    restored = pickle.loads(pickle.dumps(aug))
    assert isinstance(restored, RmsNormalize)
    assert restored.target == 0.1


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


def _build_runner(noise_dir: Path):
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig(
        wide_gain_db=40.0,
        rir_enabled=True,
        rir_probability=0.7,
        rir_pool_size=4,
        noise_augmentation_enabled=True,
        noise_dirs=[str(noise_dir)],
        use_audiomentations=True,
    )
    return EfficientATTrainingRunner(config=cfg)


def _build_runner_no_noise():
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig(
        wide_gain_db=0.0,
        rir_enabled=False,
        noise_augmentation_enabled=False,
        noise_dirs=[],
        use_audiomentations=False,
    )
    return EfficientATTrainingRunner(config=cfg)


def test_train_chain_does_not_contain_rms_normalize(
    temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 22 Plan 03: RmsNormalize is no longer appended to the 16 kHz
    ComposedAugmentation chain. It moved to ``post_resample_norm`` on the
    dataset (32 kHz domain) for train/serve parity with ``RawAudioPreprocessor``.
    """
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer, "warm_cache", lambda self: None, raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    train_aug = runner._build_train_augmentation()
    members = train_aug._augmentations
    assert not any(isinstance(a, RmsNormalize) for a in members), (
        "Phase 22 Plan 03 moved RmsNormalize out of the 16 kHz train chain "
        "into WindowedHFDroneDataset.post_resample_norm (32 kHz domain)"
    )


def test_eval_chain_does_not_contain_rms_normalize(
    temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eval chain mirrors the train chain change (Phase 22 Plan 03)."""
    from acoustic.training import augmentation as aug_mod

    monkeypatch.setattr(
        aug_mod.BackgroundNoiseMixer, "warm_cache", lambda self: None, raising=True,
    )
    runner = _build_runner(temp_noise_dir)
    eval_aug = runner._build_eval_augmentation()
    # May return None when noise is disabled; with noise enabled returns a
    # ComposedAugmentation containing ONLY BackgroundNoiseMixer — no RMS.
    assert eval_aug is not None
    members = eval_aug._augmentations
    assert not any(isinstance(a, RmsNormalize) for a in members)


def test_eval_chain_is_none_without_noise() -> None:
    """When noise mixing is disabled the eval aug is ``None`` post-Plan-03
    (previously it held a single RmsNormalize). RMS still runs on the dataset
    via ``post_resample_norm`` so val metrics match live inference amplitude.
    """
    runner = _build_runner_no_noise()
    eval_aug = runner._build_eval_augmentation()
    assert eval_aug is None


def test_trainer_wires_rms_normalize_as_post_resample_norm(
    temp_noise_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the trainer constructs WindowedHFDroneDataset with a RmsNormalize
    instance on ``post_resample_norm``. The instance is created fresh inside
    ``run()`` so we check the instance type lives on the dataset after
    constructing it the same way the trainer does.
    """
    from acoustic.training.hf_dataset import WindowedHFDroneDataset

    # Build a tiny dataset via the same construction pattern the trainer uses.
    # No need to actually run training — we just verify the wiring contract.
    import io as _io

    import soundfile as sf

    rng = np.random.default_rng(3)
    clip = rng.standard_normal(16000).astype(np.float32)
    buf = _io.BytesIO()
    sf.write(buf, clip, 16000, subtype="PCM_16", format="WAV")
    wav_bytes = buf.getvalue()

    class _HFShim(list):
        def __getitem__(self, key):
            if isinstance(key, str):
                return [r[key] for r in list.__iter__(self)]
            return list.__getitem__(self, key)

    hf = _HFShim(
        [{"audio": {"bytes": wav_bytes, "sampling_rate": 16000}, "label": 0}],
    )
    post_norm = RmsNormalize(target=0.1)
    ds = WindowedHFDroneDataset(
        hf, file_indices=[0], post_resample_norm=post_norm,
    )
    assert ds._post_resample_norm is post_norm
    audio, _ = ds[0]
    assert audio.shape[-1] == 32000
    # Output should actually be RMS-normalized to ~0.1
    audio_np = audio.detach().cpu().numpy()
    assert abs(_rms(audio_np) - 0.1) < 1e-3
