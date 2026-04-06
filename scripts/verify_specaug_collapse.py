"""Verification script for PRIMARY-A hypothesis.

Measures how often SpecAugment collapses the training mel spectrogram to
near-zero on 1-second (input_dim_t=100) clips, comparing the current
legacy trainer params (freqm=48, timem=192) to proposed Phase 20 defaults
(freqm=8, timem=10).

Usage: .venv/bin/python scripts/verify_specaug_collapse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from acoustic.classification.efficientat.config import EfficientATMelConfig
from acoustic.classification.efficientat.preprocess import AugmentMelSTFT


def measure(freqm: int, timem: int, seed: int = 0) -> dict:
    mel_cfg = EfficientATMelConfig()
    torch.manual_seed(seed)
    mel = AugmentMelSTFT(
        n_mels=mel_cfg.n_mels,
        sr=mel_cfg.sample_rate,
        win_length=mel_cfg.win_length,
        hopsize=mel_cfg.hop_size,
        n_fft=mel_cfg.n_fft,
        freqm=freqm,
        timem=timem,
    )
    mel.train()  # ensure freqm/timem active

    # Simulate training batches: random [-1,1] waveforms at segment_samples length.
    # RMS ~= 0.577 which is higher than real training data (~0.05) but mask
    # behavior is independent of input amplitude — we're measuring which
    # time-frames got zeroed (well, log-zero) by TimeMasking.
    batches = 16
    batch = 32
    n_samples = mel_cfg.segment_samples

    # Stats accumulators
    total = 0
    total_time_frames = 0
    per_sample_time_collapse_frac: list[float] = []
    abs_values: list[float] = []

    for b in range(batches):
        wav = (torch.rand(batch, n_samples) * 2 - 1) * 0.05  # ~training RMS
        with torch.no_grad():
            m = mel(wav)  # (batch, n_mels, T)
        abs_values.append(m.abs().mean().item())

        # Normalized frame value at log(eps)=log(1e-5)=-11.5 then +4.5 /5 = -1.4
        # TimeMasking replaces with zero *before* the final normalization? No:
        # torchaudio TimeMasking uses mask_value=mean of input or 0. In EfficientAT
        # AugmentMelSTFT, masking happens AFTER log and BEFORE (+4.5)/5, and the
        # torchaudio default mask_value=0 (for TimeMasking). So masked frames
        # become 0 in log-space, then (0 + 4.5)/5 = 0.9 after normalization.
        # We therefore detect masked frames by looking at values close to 0.9
        # (the normalized log(1.0)=0 masked region) — distinct from unmasked
        # normalized log-mel of random noise (~-1.0 or so).
        #
        # More robust: detect collapse by looking at the post-norm stddev per
        # time-frame. A fully-masked frame has stddev ~0 (all bins equal 0.9).
        per_frame_std = m.std(dim=1)  # (batch, T)
        frame_masked = (per_frame_std < 1e-4).float()  # (batch, T)
        frac_per_sample = frame_masked.mean(dim=1)  # (batch,)
        per_sample_time_collapse_frac.extend(frac_per_sample.tolist())

        total += batch
        total_time_frames = m.shape[-1]

    fracs = torch.tensor(per_sample_time_collapse_frac)
    return {
        "freqm": freqm,
        "timem": timem,
        "n_samples": total,
        "time_frames": total_time_frames,
        "mean_abs": sum(abs_values) / len(abs_values),
        "frac_gt50_collapsed": float((fracs > 0.50).float().mean().item()),
        "frac_gt90_collapsed": float((fracs > 0.90).float().mean().item()),
        "mean_frac_frames_masked": float(fracs.mean().item()),
        "max_frac_frames_masked": float(fracs.max().item()),
    }


def main() -> None:
    legacy = measure(freqm=48, timem=192, seed=0)
    proposed = measure(freqm=8, timem=10, seed=0)

    print("=" * 72)
    print("PRIMARY-A VERIFICATION: SpecAugment collapse on 100-frame inputs")
    print("=" * 72)
    for label, r in [("LEGACY (freqm=48, timem=192)", legacy),
                     ("PROPOSED (freqm=8, timem=10)", proposed)]:
        print(f"\n{label}")
        print(f"  samples={r['n_samples']}  time_frames/sample={r['time_frames']}")
        print(f"  mean |mel|                      = {r['mean_abs']:.4f}")
        print(f"  mean fraction frames masked     = {r['mean_frac_frames_masked']:.3f}")
        print(f"  max  fraction frames masked     = {r['max_frac_frames_masked']:.3f}")
        print(f"  P(>50% time-frames masked)      = {r['frac_gt50_collapsed']:.3f}")
        print(f"  P(>90% time-frames masked)      = {r['frac_gt90_collapsed']:.3f}")

    print()
    print("=" * 72)
    legacy_bad = legacy["frac_gt50_collapsed"]
    proposed_bad = proposed["frac_gt50_collapsed"]
    print(f"Legacy >50%-collapse rate:   {legacy_bad*100:.1f}%")
    print(f"Proposed >50%-collapse rate: {proposed_bad*100:.1f}%")
    if legacy_bad > 0.10 and proposed_bad < 0.01:
        print("VERDICT: PRIMARY-A CONFIRMED")
    elif legacy_bad > 0.10:
        print("VERDICT: PRIMARY-A CONFIRMED (legacy bad) — proposed still non-trivial")
    else:
        print("VERDICT: PRIMARY-A REFUTED (legacy masking not dominant)")
    print("=" * 72)


if __name__ == "__main__":
    main()
