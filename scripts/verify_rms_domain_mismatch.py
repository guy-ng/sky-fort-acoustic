"""Verification script for PRIMARY-B hypothesis.

Compares the RMS / log-mel amplitude distribution of:
  A) HF DADS training samples (decoded via decode_wav_bytes)
  B) Live UMA-16v2 recordings passed through RawAudioPreprocessor with the
     production cnn_input_gain value.

If the ratio of post-processed RMS (live / dads) is >10x, the model sees
inputs from very different amplitude regimes at train vs inference.

Usage: .venv/bin/python scripts/verify_rms_domain_mismatch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
import torch

from acoustic.classification.efficientat.config import EfficientATMelConfig
from acoustic.classification.efficientat.preprocess import AugmentMelSTFT
from acoustic.classification.preprocessing import RawAudioPreprocessor
from acoustic.config import AcousticSettings

# ---------------------------------------------------------------------------
# 1. HF DADS side
# ---------------------------------------------------------------------------

def load_hf_dads_samples(n: int = 10) -> list[np.ndarray]:
    """Load n raw waveforms exactly the way the trainer does."""
    from acoustic.training.hf_dataset import HFDatasetBuilder

    print(f"Loading HF DADS (geronimobasso/drone-audio-detection-samples)...")
    builder = HFDatasetBuilder("geronimobasso/drone-audio-detection-samples")
    total = builder.total_rows
    print(f"  total rows: {total}")
    # Deterministic spread across the dataset
    rng = np.random.default_rng(0)
    idxs = rng.choice(total, size=n, replace=False).tolist()
    waveforms, labels = builder.load_raw_waveforms(idxs)
    return waveforms, labels


# ---------------------------------------------------------------------------
# 2. Live UMA-16v2 side
# ---------------------------------------------------------------------------

# Raw UMA-16v2 captures preserved from field testing. test_samples/*.wav are
# 16 kHz single-channel clips saved at the native mic level (pre-gain), which
# matches what RawAudioPreprocessor receives in the live pipeline.
LIVE_CANDIDATES = [
    "data/test_samples/background_000.wav",
    "data/test_samples/background_003.wav",
    "data/test_samples/background_007.wav",
    "data/test_samples/background_010.wav",
    "data/test_samples/drone_005.wav",
    "data/test_samples/drone_007.wav",
    "data/test_samples/drone_008.wav",
    "data/test_samples/drone_011.wav",
    "data/test_samples/drone_018.wav",
    "data/test_samples/drone_019.wav",
]


def load_live_samples() -> list[tuple[str, np.ndarray, int]]:
    out = []
    for rel in LIVE_CANDIDATES:
        p = ROOT / rel
        if not p.exists():
            continue
        data, sr = sf.read(str(p))
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        out.append((p.name, data, sr))
    return out


# ---------------------------------------------------------------------------
# 3. Mel stats helper (for bonus log-mel mean measurement)
# ---------------------------------------------------------------------------

def mel_mean(waveform: torch.Tensor) -> float:
    """Run waveform through mel_eval (no SpecAugment) and return mean norm value."""
    mel_cfg = EfficientATMelConfig()
    mel = AugmentMelSTFT(
        n_mels=mel_cfg.n_mels, sr=mel_cfg.sample_rate,
        win_length=mel_cfg.win_length, hopsize=mel_cfg.hop_size,
        n_fft=mel_cfg.n_fft, freqm=0, timem=0,
    )
    mel.eval()
    with torch.no_grad():
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        m = mel(waveform)
    return float(m.mean().item())


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = AcousticSettings()
    cnn_input_gain = settings.cnn_input_gain
    target_sr = 32000
    print("=" * 72)
    print("PRIMARY-B VERIFICATION: train/inference RMS domain mismatch")
    print("=" * 72)
    print(f"AcousticSettings.cnn_input_gain = {cnn_input_gain}")
    print(f"target_sr (EfficientAT)        = {target_sr}")
    print()

    # ---- HF DADS ----
    try:
        hf_waves, hf_labels = load_hf_dads_samples(n=10)
    except Exception as e:
        print(f"WARNING: failed to load HF DADS: {e}")
        hf_waves = []
        hf_labels = []

    print("\n-- HF DADS training samples (decode_wav_bytes → float32 [-1,1]) --")
    hf_rms = []
    hf_mel_means = []
    for i, w in enumerate(hf_waves):
        r = rms(w)
        hf_rms.append(r)
        # Mel on 32k domain — DADS WAVs are 16 kHz int16, so resample the same
        # way the trainer would (it doesn't resample — it just feeds them at
        # whatever rate AugmentMelSTFT expects). Checking the trainer path:
        # it passes waveforms directly to mel_train which assumes sr=32000.
        # That's ANOTHER latent bug, but not what we're measuring here.
        # We use the waveform as-is to mirror trainer behavior.
        wav_tensor = torch.from_numpy(w.astype(np.float32))
        try:
            mm = mel_mean(wav_tensor)
        except Exception:
            mm = float("nan")
        hf_mel_means.append(mm)
        print(f"  [{i}] label={hf_labels[i] if i < len(hf_labels) else '?'} n={len(w):>6} rms={r:.4g} mel_mean={mm:.4g}")

    if hf_rms:
        print(f"\n  DADS RMS: min={min(hf_rms):.4g} mean={np.mean(hf_rms):.4g} max={max(hf_rms):.4g}")
        print(f"  DADS mel_mean: mean={np.nanmean(hf_mel_means):.4g}")

    # ---- Live UMA-16 ----
    live = load_live_samples()
    print(f"\n-- Live UMA-16 recordings ({len(live)} found) --")
    live_pre_rms = []
    live_post_rms = []
    live_mel_means = []
    if not live:
        print("  NO UMA-16 recordings found in data/test_samples/")
    else:
        preproc = RawAudioPreprocessor(target_sr=target_sr, input_gain=cnn_input_gain)
        for name, data, sr in live:
            pre = rms(data)
            processed = preproc.process(data, sr)  # returns torch tensor
            proc_np = processed.detach().cpu().numpy()
            post = rms(proc_np)
            mm = mel_mean(processed.float())
            live_pre_rms.append(pre)
            live_post_rms.append(post)
            live_mel_means.append(mm)
            print(f"  {name:32s} sr={sr} n={len(data):>6} raw_rms={pre:.4g} post_gain_rms={post:.4g} mel_mean={mm:.4g}")

        print(f"\n  LIVE raw RMS:  min={min(live_pre_rms):.4g}  mean={np.mean(live_pre_rms):.4g}  max={max(live_pre_rms):.4g}")
        print(f"  LIVE post-gain RMS (× {cnn_input_gain}): min={min(live_post_rms):.4g}  mean={np.mean(live_post_rms):.4g}  max={max(live_post_rms):.4g}")
        print(f"  LIVE mel_mean: mean={np.nanmean(live_mel_means):.4g}")

    # ---- Verdict ----
    print("\n" + "=" * 72)
    if hf_rms and live_post_rms:
        hf_mean = float(np.mean(hf_rms))
        live_mean = float(np.mean(live_post_rms))
        ratio = live_mean / max(hf_mean, 1e-12)
        print(f"Ratio = live_post_gain_rms / dads_rms = {live_mean:.4g} / {hf_mean:.4g} = {ratio:.2f}x")
        if ratio > 10.0:
            print("VERDICT: PRIMARY-B CONFIRMED (>10x amplitude domain gap)")
        elif ratio < 2.0:
            print("VERDICT: PRIMARY-B REFUTED (<2x gap)")
        else:
            print("VERDICT: PRIMARY-B PARTIAL (2–10x gap)")

        # Bonus: log-mel region
        if hf_mel_means and live_mel_means:
            dads_mm = float(np.nanmean(hf_mel_means))
            live_mm = float(np.nanmean(live_mel_means))
            print(f"Normalized log-mel mean: DADS={dads_mm:.3f}  LIVE={live_mm:.3f}  Δ={live_mm - dads_mm:+.3f}")
    elif not live_post_rms:
        print("VERDICT: PRIMARY-B PARTIAL (no live UMA-16 recordings available — "
              "only the DADS side measured)")
    else:
        print("VERDICT: PRIMARY-B INCONCLUSIVE (DADS loader failed)")
    print("=" * 72)


if __name__ == "__main__":
    main()
