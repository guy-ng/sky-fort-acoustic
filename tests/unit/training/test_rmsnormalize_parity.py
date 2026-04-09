"""Phase 22 Wave 2 (Plan 03): training vs inference RMS normalization parity.

Guardrail tests for REQ-22-W4. The v7 post-mortem found a ~2 % amplitude
skew between training (RmsNormalize at 16 kHz, pre-resample) and inference
(RmsNormalize at 32 kHz, post-resample). Plan 03 moves the trainer into the
32 kHz domain so both paths normalize in the same sample-rate regime.

Tests:
- Direct parity: training's post-resample RmsNormalize output equals inference
  ``RawAudioPreprocessor.process`` output within 1e-4 on the same input.
- Regression guard: ``_build_train_augmentation`` no longer appends
  ``RmsNormalize`` inside the 16 kHz waveform chain.
- ``WindowedHFDroneDataset`` wired with ``post_resample_norm=RmsNormalize`` RMS
  normalizes the 32 kHz tensor it returns.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch


def _encode_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _make_synthetic_hf_dataset(
    num_files: int = 1, clip_samples: int = 16000,
) -> list[dict]:
    rng = np.random.default_rng(123)
    rows = [
        {
            "audio": {
                "bytes": _encode_wav_bytes(
                    rng.standard_normal(clip_samples).astype(np.float32),
                    sr=16000,
                ),
                "sampling_rate": 16000,
            },
            "label": i % 2,
        }
        for i in range(num_files)
    ]

    class _HFShim(list):
        def __getitem__(self, key):  # type: ignore[override]
            if isinstance(key, str):
                return [r[key] for r in list.__iter__(self)]
            return list.__getitem__(self, key)

    return _HFShim(rows)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def test_train_serve_rms_parity_within_1e4() -> None:
    """Training RmsNormalize (post-resample) matches inference preprocessor output.

    Both paths should land at the same target RMS because they run the SAME
    ``_rms_normalize`` helper on audio resampled to 32 kHz — the Plan 03 fix.
    """
    from acoustic.classification.preprocessing import (
        RawAudioPreprocessor,
        _rms_normalize,
    )

    rng = np.random.default_rng(0)
    # Synthetic 1 s @ 16 kHz at low amplitude so we can tell if normalization ran
    audio_16k = (rng.standard_normal(16000) * 0.02).astype(np.float32)

    target_rms = 0.1

    # --- Inference path: resample 16k→32k → _rms_normalize → tensor ---
    infer_pre = RawAudioPreprocessor(
        target_sr=32000, rms_normalize_target=target_rms,
    )
    serve_tensor = infer_pre.process(audio_16k, sr=16000)
    serve_out = serve_tensor.detach().cpu().numpy().astype(np.float32)

    # --- Training path (post-Plan-03): the dataset returns a resampled tensor,
    # then the post_resample_norm callable runs in the 32 kHz domain. We emulate
    # the dataset's in-loop behavior: resample first, then run the SAME
    # _rms_normalize helper that RmsNormalize delegates to.
    import torchaudio.functional as F_audio

    train_32k = F_audio.resample(
        torch.from_numpy(audio_16k), 16000, 32000,
    ).numpy()
    train_out = _rms_normalize(train_32k.copy(), target=target_rms).astype(
        np.float32,
    )

    # Parity within 1e-4 — both paths normalize the SAME 32 kHz resampled signal
    max_diff = float(np.max(np.abs(train_out - serve_out)))
    assert max_diff < 1e-4, (
        f"train/serve RMS parity broken: max diff = {max_diff}"
    )

    # Sanity: both paths land near the target RMS (not the 0.02 source level)
    assert abs(_rms(train_out) - target_rms) < 1e-3
    assert abs(_rms(serve_out) - target_rms) < 1e-3


def test_build_train_augmentation_no_rmsnormalize_in_chain() -> None:
    """Regression guard: trainer's train aug chain must not contain RmsNormalize.

    Phase 22 Plan 03 moved RmsNormalize out of the 16 kHz ComposedAugmentation
    and into the 32 kHz post_resample_norm hook on WindowedHFDroneDataset.
    """
    from acoustic.training.augmentation import RmsNormalize
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig()
    runner = EfficientATTrainingRunner(cfg)
    chain = runner._build_train_augmentation()
    members = getattr(chain, "_augmentations", [])
    assert not any(isinstance(a, RmsNormalize) for a in members), (
        "RmsNormalize must NOT live inside the 16 kHz train augmentation chain; "
        "it is now a post_resample_norm on the dataset (32 kHz domain)"
    )


def test_build_eval_augmentation_no_rmsnormalize_in_chain() -> None:
    """Regression guard: eval aug chain also purged of RmsNormalize."""
    from acoustic.training.augmentation import RmsNormalize
    from acoustic.training.config import TrainingConfig
    from acoustic.training.efficientat_trainer import EfficientATTrainingRunner

    cfg = TrainingConfig()
    runner = EfficientATTrainingRunner(cfg)
    chain = runner._build_eval_augmentation()
    # ``_build_eval_augmentation`` may return None when noise is disabled —
    # that also satisfies the "no RmsNormalize in 16 kHz chain" guarantee.
    if chain is None:
        return
    members = getattr(chain, "_augmentations", [])
    assert not any(isinstance(a, RmsNormalize) for a in members), (
        "eval chain must not contain RmsNormalize post-Plan-03"
    )


def test_dataset_post_resample_norm_runs_rms_normalize() -> None:
    """WindowedHFDroneDataset wired with RmsNormalize returns RMS-normalized audio.

    This is the concrete integration: trainer constructs the dataset with
    ``post_resample_norm=RmsNormalize(target=0.1)``, dataset calls it on the
    32 kHz tensor, output audio has RMS ≈ 0.1.
    """
    from acoustic.training.augmentation import RmsNormalize
    from acoustic.training.hf_dataset import WindowedHFDroneDataset

    hf = _make_synthetic_hf_dataset(num_files=2, clip_samples=16000)
    norm = RmsNormalize(target=0.1)
    ds = WindowedHFDroneDataset(
        hf, file_indices=[0, 1], post_resample_norm=norm,
    )
    audio, _ = ds[0]
    assert audio.shape[-1] == 32000
    audio_np = audio.detach().cpu().numpy()
    assert abs(_rms(audio_np) - 0.1) < 1e-3, (
        f"post-norm dataset output RMS should be ~0.1, got {_rms(audio_np):.4f}"
    )
