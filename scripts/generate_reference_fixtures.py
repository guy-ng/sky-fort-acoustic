"""One-time script to generate .npy reference tensors from the research code.

Run: python scripts/generate_reference_fixtures.py
Requires: librosa (dev dependency only, NOT a runtime dep).

Produces:
  tests/fixtures/reference_melspec_440hz.npy  -- shape (128, 64) float32
"""

import os

import librosa
import numpy as np

# Match research code constants exactly (train_strong_cnn.py lines 35-42)
FS = 16000
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_FRAMES = 128


def pad_or_trim_frames(spec, max_frames):
    t = spec.shape[0]
    if t < max_frames:
        pad = np.zeros((max_frames - t, spec.shape[1]), dtype=np.float32)
        return np.concatenate([spec, pad], axis=0)
    elif t > max_frames:
        start = (t - max_frames) // 2
        return spec[start : start + max_frames]
    return spec


def segment_to_melspec_reference(samples, sr):
    S = librosa.feature.melspectrogram(
        y=samples,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db + 80.0) / 80.0
    S_db_norm = np.clip(S_db_norm, 0.0, 1.0)
    spec = S_db_norm.T
    spec = pad_or_trim_frames(spec, MAX_FRAMES)
    return spec.astype(np.float32)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    # Deterministic 440Hz sine wave at 16kHz for 0.5s
    t = np.linspace(0, CHUNK_SECONDS, CHUNK_SAMPLES, endpoint=False, dtype=np.float32)
    sine_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ref = segment_to_melspec_reference(sine_440, FS)
    out_path = os.path.join(out_dir, "reference_melspec_440hz.npy")
    np.save(out_path, ref)
    print(f"Saved: {out_path}  shape={ref.shape}  dtype={ref.dtype}")
    print(f"  min={ref.min():.4f}  max={ref.max():.4f}  mean={ref.mean():.4f}")
